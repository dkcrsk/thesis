"""
model.py — Combibrug multi-period workforce MILP

OBJECTIVE
    Minimise total cost (internal staff + last-resort freelancer fill) plus
    soft penalties for switching employees between sites and for spreading
    one employee across many sites.

FREELANCERS
    Freelancers are NOT rows in the employee template. They are handled as
    a continuous gap-fill variable u[j,t] at a flat hourly rate (set in the
    sidebar). The model uses internal staff first, freelancers only when
    demand cannot otherwise be met. Uncovered freelancer hours are reported
    per site so the planner knows where they are needed.

CONSTRAINTS (all use _safe() for site names to avoid LP-file mangling)
    C1   Contract hours per employee per week
    C2   Demand coverage at each active site-week (equality, u absorbs gap)
    C3   Company boundary (CC vs Combibrug, enforced via eligibility set)
    C4   Role eligibility (enforced via eligibility set if req_<role> > 0)
    C5   Linking: k[i,j,t] <= max_blocks * y[i,j,t]
    C6   Per-role headcount per site per week (optional)
    C7   Continuity switch detection (soft penalty)
    C8   Four-Eyes minimum headcount (Combiworld/MDT projects)
    C9   Fixed project leader: HARD — leader assigned every active week
    C10  Stay until project end (monotonicity, with employee-exit exception)
    C11  Location-spread linking (soft penalty)
    C12  Utilisation MINIMUM (HARD): permanent >= 50%, fixed_term >= 30% of
         contract hours over horizon. Managers EXCLUDED — they're assigned
         only via C9.
    C13a Daily time non-overlap (same employee, same day, overlapping windows)
    C13b 8-hour daily cap per employee
    C13c One site per day per employee (hard)
    C14  Weekly-to-daily linking: sum of chosen window lengths == k[i,j,t]
"""

import pandas as pd
from pulp import (
    LpProblem, LpVariable, LpMinimize, LpBinary, LpInteger,
    lpSum, PULP_CBC_CMD, LpStatus, value,
)

from data_loader import ALL_ROLES


DEFAULT_FREELANCER_COST = 50.0
DEFAULT_BLOCK_SIZE = 1
DEFAULT_HORIZON_WEEKS = 13
DEFAULT_CONTINUITY_PENALTY = 25.0
DEFAULT_LOCATION_PENALTY = 15.0


def _safe(name):
    """Sanitise a string for use in a PuLP variable / constraint name.

    PuLP writes the LP file with these names; LP format does NOT allow
    spaces, hyphens, or punctuation in identifiers. Without sanitising,
    constraints involving sites like 'BSC Oost' get silently mangled by
    the solver and effectively dropped. EVERY name in this module passes
    through _safe() for any string component.
    """
    return "".join(c if c.isalnum() else "_" for c in str(name))


def _hours_to_hhmm(h):
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh += 1
        mm = 0
    return f"{hh:02d}:{mm:02d}"


def solve_allocation(employees, locations,
                     freelancer_cost=DEFAULT_FREELANCER_COST,
                     enforce_headcount=True,
                     block_size=DEFAULT_BLOCK_SIZE,
                     horizon_weeks=DEFAULT_HORIZON_WEEKS,
                     continuity_penalty=DEFAULT_CONTINUITY_PENALTY,
                     location_penalty=DEFAULT_LOCATION_PENALTY,
                     utilisation_penalty=None,  # accepted but unused (C12 is hard now)
                     stay_until_end=True,
                     enable_daily_scheduling=False,
                     max_hours_per_day=8,
                     time_limit=180,
                     verbose=False):
    """Build and solve the multi-period MILP."""

    emp = employees.to_dict("records")
    loc = locations.to_dict("records")
    emp_by_id = {e["id"]: e for e in emp}
    weeks = list(range(1, horizon_weeks + 1))

    def active_weeks(l):
        sw = int(l.get("start_week", 1))
        dur = int(l.get("duration_weeks", horizon_weeks))
        return [t for t in weeks if sw <= t < sw + dur]

    loc_active = {l["site"]: active_weeks(l) for l in loc}

    eligible_pairs = []
    for e in emp:
        for l in loc:
            if e["company"] != l["company"]:
                continue
            total_req = sum(l.get(f"req_{r}", 0) for r in ALL_ROLES)
            if total_req > 0 and l.get(f"req_{e['role']}", 0) == 0:
                continue
            eligible_pairs.append((e["id"], l["site"]))

    eligible_pairs_t = [
        (i, j, t) for (i, j) in eligible_pairs for t in loc_active.get(j, [])
    ]

    prob = LpProblem("Combibrug_Workforce_MultiPeriod", LpMinimize)

    max_blocks = {
        i: int(emp_by_id[i]["contract_hours"] // block_size)
        for i in {p[0] for p in eligible_pairs}
    }

    # ---- variables (ALL names sanitised) -----------------------------------
    k = {
        (i, j, t): LpVariable(
            f"k_{i}_{_safe(j)}_{t}", lowBound=0,
            upBound=max_blocks.get(i, 0), cat=LpInteger
        )
        for (i, j, t) in eligible_pairs_t
    }
    y = {
        (i, j, t): LpVariable(f"y_{i}_{_safe(j)}_{t}", cat=LpBinary)
        for (i, j, t) in eligible_pairs_t
    }
    u = {
        (l["site"], t): LpVariable(f"u_{_safe(l['site'])}_{t}", lowBound=0)
        for l in loc for t in loc_active.get(l["site"], [])
    }

    s = {}
    if continuity_penalty > 0:
        for (i, j) in eligible_pairs:
            active = loc_active.get(j, [])
            for t in active:
                if t > 1 and (t - 1) in active:
                    s[(i, j, t)] = LpVariable(
                        f"s_{i}_{_safe(j)}_{t}", cat=LpBinary
                    )

    loc_used = {}
    if location_penalty > 0:
        for (i, j) in eligible_pairs:
            loc_used[(i, j)] = LpVariable(
                f"locused_{i}_{_safe(j)}", cat=LpBinary
            )

    # ---- C12 utilisation targets (managers excluded) ----------------------
    UTIL_TARGET_FRACTION = {"permanent": 0.50, "fixed_term": 0.30}
    util_target_hours = {}
    for e in emp:
        if e.get("role") == "Manager":
            continue
        fraction = UTIL_TARGET_FRACTION.get(e.get("contract_type", ""), 0.0)
        if fraction <= 0:
            continue
        eid = e["id"]
        end_w = e.get("end_week")
        try:
            end_w = int(end_w) if end_w is not None and not pd.isna(end_w) else None
        except (TypeError, ValueError):
            end_w = None
        active_weeks_for_emp = [t for t in weeks if end_w is None or t <= end_w]
        pairs_emp = [(i, j, tt) for (i, j, tt) in eligible_pairs_t if i == eid]
        if not pairs_emp or not active_weeks_for_emp:
            continue
        target = e["contract_hours"] * fraction * len(active_weeks_for_emp)
        util_target_hours[eid] = target

    # ---- objective ---------------------------------------------------------
    internal_cost = lpSum(
        emp_by_id[i]["hourly_cost"] * block_size * k[(i, j, t)]
        for (i, j, t) in eligible_pairs_t
    )
    external_cost = lpSum(
        freelancer_cost * u[(l["site"], t)]
        for l in loc for t in loc_active.get(l["site"], [])
    )
    switch_penalty = lpSum(continuity_penalty * s[(i, j, t)] for (i, j, t) in s)
    location_spread_penalty = lpSum(
        location_penalty * loc_used[(i, j)] for (i, j) in loc_used
    )
    prob += internal_cost + external_cost + switch_penalty + location_spread_penalty

    # Constraint counts — printed at the end so you can verify in Streamlit logs
    counts = {
        "C1_contract": 0, "C2_demand": 0, "C5_link": 0, "C6_role_hc": 0,
        "C7_continuity": 0, "C8_min_hc": 0, "C8_block": 0,
        "C9_leader_y": 0, "C9_leader_k": 0, "C10_stay": 0,
        "C11_locused": 0, "C12_min_util": 0,
        "C13a_overlap": 0, "C13b_day8h": 0, "C13c_one_site": 0,
        "C14_daily_link": 0, "z_implies_y": 0,
    }

    # C1. Contract hours per employee per week
    for e in emp:
        eid = e["id"]
        for t in weeks:
            pairs_t = [p for p in eligible_pairs_t if p[0] == eid and p[2] == t]
            if pairs_t:
                prob += (
                    block_size * lpSum(k[p] for p in pairs_t) <= e["contract_hours"],
                    f"C1_contract_{eid}_w{t}",
                )
                counts["C1_contract"] += 1

    # C2. Demand coverage = internal + u (last-resort freelancer fill)
    for l in loc:
        for t in loc_active.get(l["site"], []):
            pairs_t = [p for p in eligible_pairs_t
                       if p[1] == l["site"] and p[2] == t]
            prob += (
                block_size * lpSum(k[p] for p in pairs_t) + u[(l["site"], t)]
                == l["weekly_hours"],
                f"C2_demand_{_safe(l['site'])}_w{t}",
            )
            counts["C2_demand"] += 1

    # C5. Linking k <= M * y
    for (i, j, t) in eligible_pairs_t:
        prob += (
            k[(i, j, t)] <= max_blocks.get(i, 0) * y[(i, j, t)],
            f"C5_link_{i}_{_safe(j)}_w{t}",
        )
        counts["C5_link"] += 1

    # C6. Per-role headcount
    if enforce_headcount:
        for l in loc:
            for role in ALL_ROLES:
                req = l.get(f"req_{role}", 0)
                if req > 0:
                    for t in loc_active.get(l["site"], []):
                        elig_emps = [
                            i for i in {p[0] for p in eligible_pairs}
                            if emp_by_id[i]["role"] == role
                            and (i, l["site"], t) in y
                        ]
                        if elig_emps:
                            prob += (
                                lpSum(y[(i, l["site"], t)] for i in elig_emps) >= req,
                                f"C6_role_hc_{_safe(l['site'])}_{_safe(role)}_w{t}",
                            )
                            counts["C6_role_hc"] += 1

    # C7. Continuity switch indicator
    for (i, j, t) in s:
        prob += s[(i, j, t)] >= y[(i, j, t)] - y[(i, j, t - 1)]
        prob += s[(i, j, t)] >= y[(i, j, t - 1)] - y[(i, j, t)]
        counts["C7_continuity"] += 2

    # C8. Four-Eyes minimum headcount + min-block per assigned employee
    for l in loc:
        min_hc = int(l.get("min_headcount", 0) or 0)
        if min_hc > 0:
            for t in loc_active.get(l["site"], []):
                elig = [i for i in {p[0] for p in eligible_pairs}
                        if (i, l["site"], t) in y]
                if elig:
                    prob += (
                        lpSum(y[(i, l["site"], t)] for i in elig) >= min_hc,
                        f"C8_min_hc_{_safe(l['site'])}_w{t}",
                    )
                    counts["C8_min_hc"] += 1
                    for i in elig:
                        prob += (
                            k[(i, l["site"], t)] >= y[(i, l["site"], t)],
                            f"C8_block_{i}_{_safe(l['site'])}_w{t}",
                        )
                        counts["C8_block"] += 1

    # C9. Fixed project leader — HARD. Defensive typing for pd.NA / np.int64.
    for l in loc:
        leader_raw = l.get("project_leader_id")
        if leader_raw is None or pd.isna(leader_raw):
            continue
        try:
            leader_id = int(leader_raw)
        except (TypeError, ValueError):
            continue
        if leader_id not in emp_by_id:
            continue
        for t in loc_active.get(l["site"], []):
            if (leader_id, l["site"], t) not in y:
                continue
            prob += (
                y[(leader_id, l["site"], t)] == 1,
                f"C9_leader_y_{leader_id}_{_safe(l['site'])}_w{t}",
            )
            prob += (
                k[(leader_id, l["site"], t)] >= 1,
                f"C9_leader_k_{leader_id}_{_safe(l['site'])}_w{t}",
            )
            counts["C9_leader_y"] += 1
            counts["C9_leader_k"] += 1

    # C10. Stay until project end
    if stay_until_end:
        proj_sites = {}
        for l in loc:
            proj = l.get("project") or l["site"]
            proj_sites.setdefault(proj, []).append(l["site"])

        proj_active = {}
        for proj, sites in proj_sites.items():
            wk = set()
            for sname in sites:
                wk.update(loc_active.get(sname, []))
            proj_active[proj] = sorted(wk)

        y_proj = {}
        emp_ids_in_pairs = {p[0] for p in eligible_pairs}
        for i in emp_ids_in_pairs:
            for proj, sites in proj_sites.items():
                elig_sites = [s for s in sites if (i, s) in eligible_pairs]
                if not elig_sites:
                    continue
                for t in proj_active[proj]:
                    site_vars = [(i, s, t) for s in elig_sites
                                 if (i, s, t) in y]
                    if not site_vars:
                        continue
                    yp = LpVariable(
                        f"yproj_{i}_{_safe(proj)}_{t}", cat=LpBinary
                    )
                    y_proj[(i, proj, t)] = yp
                    for sv in site_vars:
                        prob += yp >= y[sv]
                    prob += yp <= lpSum(y[sv] for sv in site_vars)

        for (i, proj, t) in list(y_proj.keys()):
            end_w = emp_by_id[i].get("end_week")
            try:
                end_w = int(end_w) if end_w is not None and not pd.isna(end_w) else None
            except (TypeError, ValueError):
                end_w = None
            active = proj_active[proj]
            later = [w for w in active if w > t]
            if not later:
                continue
            t_next = later[0]
            if (i, proj, t_next) not in y_proj:
                continue
            if end_w is not None and t_next > end_w:
                continue
            prob += (
                y_proj[(i, proj, t)] <= y_proj[(i, proj, t_next)],
                f"C10_stay_{i}_{_safe(proj)}_w{t}",
            )
            counts["C10_stay"] += 1

    # C11. Location-spread linking
    for (i, j) in loc_used:
        for t in loc_active.get(j, []):
            if (i, j, t) in y:
                prob += loc_used[(i, j)] >= y[(i, j, t)]
                counts["C11_locused"] += 1

    # C12. Utilisation MINIMUM — hard, over the whole horizon
    for eid, target in util_target_hours.items():
        pairs_emp = [p for p in eligible_pairs_t if p[0] == eid]
        prob += (
            block_size * lpSum(k[p] for p in pairs_emp) >= target,
            f"C12_min_util_{eid}",
        )
        counts["C12_min_util"] += 1

    # ---- Daily scheduling layer (C13a / C13b / C13c / C14 + z links) -------
    z = {}
    if enable_daily_scheduling:
        site_windows = {}
        for l in loc:
            wins = l.get("windows") or []
            if wins:
                site_windows[l["site"]] = list(wins)

        for (i, j, t) in eligible_pairs_t:
            wins = site_windows.get(j)
            if not wins:
                continue
            for w_idx, _ in enumerate(wins):
                z[(i, j, t, w_idx)] = LpVariable(
                    f"z_{i}_{_safe(j)}_{t}_{w_idx}", cat=LpBinary
                )

        site_day_windows = {}
        for j, wins in site_windows.items():
            for w_idx, (day, s_h, e_h) in enumerate(wins):
                site_day_windows.setdefault((j, day), []).append(
                    (w_idx, s_h, e_h, e_h - s_h)
                )

        emp_ids_with_pairs = sorted({p[0] for p in eligible_pairs})

        # C13a. Time non-overlap on same day across sites
        for i in emp_ids_with_pairs:
            for t in weeks:
                for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                    slots = []
                    for (jj, d), entries in site_day_windows.items():
                        if d != day or (i, jj, t) not in y:
                            continue
                        for (w_idx, s_h, e_h, _) in entries:
                            if (i, jj, t, w_idx) in z:
                                slots.append((jj, w_idx, s_h, e_h))
                    for a in range(len(slots)):
                        for b in range(a + 1, len(slots)):
                            ja, wa, sa, ea = slots[a]
                            jb, wb, sb, eb = slots[b]
                            if sa < eb and sb < ea:
                                prob += (
                                    z[(i, ja, t, wa)] + z[(i, jb, t, wb)] <= 1,
                                    f"C13a_ovl_{i}_{_safe(ja)}_{wa}_"
                                    f"{_safe(jb)}_{wb}_{day}_w{t}",
                                )
                                counts["C13a_overlap"] += 1

        # C13b. 8h daily cap across all sites
        for i in emp_ids_with_pairs:
            for t in weeks:
                for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                    terms = []
                    for (jj, d), entries in site_day_windows.items():
                        if d != day or (i, jj, t) not in y:
                            continue
                        for (w_idx, s_h, e_h, length) in entries:
                            if (i, jj, t, w_idx) in z:
                                terms.append(length * z[(i, jj, t, w_idx)])
                    if terms:
                        prob += (
                            lpSum(terms) <= max_hours_per_day,
                            f"C13b_day8h_{i}_{day}_w{t}",
                        )
                        counts["C13b_day8h"] += 1

        # C13c. One site per day (hard)
        for i in emp_ids_with_pairs:
            for t in weeks:
                for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                    sites_today = sorted({
                        jj for (jj, d) in site_day_windows
                        if d == day and (i, jj, t) in y
                    })
                    if len(sites_today) <= 1:
                        continue
                    indicators = []
                    for jj in sites_today:
                        w_idxs = [w_idx for (w_idx, sh, eh, _)
                                  in site_day_windows[(jj, day)]
                                  if (i, jj, t, w_idx) in z]
                        if not w_idxs:
                            continue
                        sd = LpVariable(
                            f"sd_{i}_{_safe(jj)}_{day}_w{t}", cat=LpBinary
                        )
                        for w_idx in w_idxs:
                            prob += sd >= z[(i, jj, t, w_idx)]
                        prob += sd <= lpSum(z[(i, jj, t, w_idx)] for w_idx in w_idxs)
                        indicators.append(sd)
                    if len(indicators) > 1:
                        prob += (
                            lpSum(indicators) <= 1,
                            f"C13c_one_site_{i}_{day}_w{t}",
                        )
                        counts["C13c_one_site"] += 1

        # C14. sum(length * z) == k
        for (i, j, t) in eligible_pairs_t:
            wins = site_windows.get(j)
            if not wins:
                continue
            terms = [(e_h - s_h) * z[(i, j, t, w_idx)]
                     for w_idx, (_, s_h, e_h) in enumerate(wins)]
            prob += (
                lpSum(terms) == block_size * k[(i, j, t)],
                f"C14_daily_link_{i}_{_safe(j)}_w{t}",
            )
            counts["C14_daily_link"] += 1

        # z implies y
        for (i, j, t, w_idx), zv in z.items():
            if (i, j, t) in y:
                prob += zv <= y[(i, j, t)]
                counts["z_implies_y"] += 1

    # ---- log constraint counts so you can verify in Streamlit logs ---------
    print("\n" + "=" * 60)
    print("  COMBIBRUG MILP — constraint counts")
    print("=" * 60)
    for k_name, n_count in counts.items():
        print(f"    {k_name:25s} {n_count:>8d}")
    n_vars = len(k) + len(y) + len(u) + len(s) + len(loc_used) + len(z)
    print(f"    {'TOTAL variables':25s} {n_vars:>8d}")
    print(f"    {'TOTAL constraints':25s} {sum(counts.values()):>8d}")
    print("=" * 60 + "\n", flush=True)

    # ---- solve ------------------------------------------------------------
    solver = PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    prob.solve(solver)
    status = LpStatus[prob.status]
    print(f"  Solver status: {status}\n", flush=True)

    site_to_project = {l["site"]: l.get("project") or l["site"] for l in loc}

    # ---- extract assignments ONLY if Optimal -------------------------------
    rows = []
    if status == "Optimal":
        for (i, j, t), var in k.items():
            blocks = int(round(var.value() or 0))
            hours = blocks * block_size
            if hours > 0:
                rows.append({
                    "employee_id": i,
                    "employee_role": emp_by_id[i]["role"],
                    "contract_type": emp_by_id[i]["contract_type"],
                    "company": emp_by_id[i]["company"],
                    "project": site_to_project.get(j, j),
                    "site": j,
                    "week": t,
                    "blocks": blocks,
                    "hours": hours,
                    "cost": round(hours * emp_by_id[i]["hourly_cost"], 2),
                })
    assignments = pd.DataFrame(rows)

    daily_rows = []
    if status == "Optimal" and enable_daily_scheduling and z:
        site_windows_lookup = {
            l["site"]: list(l.get("windows") or [])
            for l in loc if l.get("windows")
        }
        for (i, j, t, w_idx), zv in z.items():
            if int(round(zv.value() or 0)) == 1:
                day, s_h, e_h = site_windows_lookup[j][w_idx]
                daily_rows.append({
                    "employee_id": i,
                    "employee_role": emp_by_id[i]["role"],
                    "company": emp_by_id[i]["company"],
                    "project": site_to_project.get(j, j),
                    "site": j,
                    "week": t,
                    "day": day,
                    "start": _hours_to_hhmm(s_h),
                    "end": _hours_to_hhmm(e_h),
                    "hours": round(e_h - s_h, 2),
                })
    daily_slots = pd.DataFrame(daily_rows)

    # ---- summary ----------------------------------------------------------
    freelancer_hours = {
        (l["site"], t): (u[(l["site"], t)].value() or 0)
        for l in loc for t in loc_active.get(l["site"], [])
    } if status == "Optimal" else {}

    total_internal = float(sum(r["cost"] for r in rows))
    total_external = float(sum(h * freelancer_cost for h in freelancer_hours.values()))
    n_switches = (
        int(round(sum((v.value() or 0) for v in s.values())))
        if status == "Optimal" else 0
    )

    freelancer_hours_by_site = {}
    for (j, t), h in freelancer_hours.items():
        if h > 0.01:
            freelancer_hours_by_site[j] = freelancer_hours_by_site.get(j, 0.0) + h

    n_distinct_locations = (
        int(round(sum((v.value() or 0) for v in loc_used.values())))
        if (status == "Optimal" and loc_used) else None
    )

    summary = {
        "status": status,
        "objective": value(prob.objective) if (status == "Optimal" and prob.objective is not None) else None,
        "internal_cost": total_internal,
        "external_cost": total_external,
        "switch_penalty_cost": float(n_switches * continuity_penalty),
        "n_switches": n_switches,
        "n_distinct_location_assignments": n_distinct_locations,
        "horizon_weeks": horizon_weeks,
        "freelancer_hours_by_period": freelancer_hours,
        "freelancer_hours_by_site": freelancer_hours_by_site,
        "total_freelancer_hours": float(sum(freelancer_hours.values())),
        "constraint_counts": counts,
        "daily_scheduling_enabled": enable_daily_scheduling,
        "daily_slots": daily_slots,
    }
    return status, assignments, summary


def validate_assignments(employees, locations, assignments, summary):
    """Return a list of validation messages for the Diagnostics tab."""
    checks = []
    if assignments.empty:
        checks.append("FAIL: no assignments produced.")
        return checks

    hours = assignments.groupby(["employee_id", "week"])["hours"].sum()
    over = []
    for (eid, week), h in hours.items():
        contract = float(employees.loc[employees["id"] == eid, "contract_hours"].iloc[0])
        if h > contract + 0.01:
            over.append((eid, week, h, contract))
    if over:
        for eid, week, h, c in over[:5]:
            checks.append(f"FAIL: employee {eid} in week {week}: {h:.1f}h > contract {c:.1f}h")
    else:
        checks.append("OK: no contract-hour violations")

    merged = assignments.merge(
        locations[["site", "company"]].rename(columns={"company": "loc_co"}),
        on="site", how="left",
    )
    cross = merged[merged["company"] != merged["loc_co"]]
    if len(cross) > 0:
        checks.append(f"FAIL: {len(cross)} cross-company assignments")
    else:
        checks.append("OK: no cross-company assignments")

    return checks


def explain_assignment(employee_id, employees, locations, assignments):
    """Return a DataFrame showing which sites an employee was eligible for
    and what they were assigned. Used by the Diagnostics tab."""
    if employee_id not in employees["id"].values:
        return pd.DataFrame()
    emp = employees[employees["id"] == employee_id].iloc[0]
    chosen = {}
    if not assignments.empty:
        sub = assignments[assignments["employee_id"] == employee_id]
        for site_name, g in sub.groupby("site"):
            chosen[site_name] = {
                "weeks_worked": len(g),
                "total_hours": float(g["hours"].sum()),
                "weeks": sorted(int(w) for w in g["week"].tolist()),
            }
    rows = []
    for _, l in locations.iterrows():
        if l["company"] != emp["company"]:
            continue
        total_req = sum(l.get(f"req_{r}", 0) for r in ALL_ROLES)
        if total_req > 0 and l.get(f"req_{emp['role']}", 0) == 0:
            continue
        c = chosen.get(l["site"])
        rows.append({
            "site": l["site"],
            "project": l.get("project") or l["site"],
            "weekly_hours_demand": l["weekly_hours"],
            "duration_weeks": l.get("duration_weeks", "—"),
            "start_week": l.get("start_week", 1),
            "hourly_cost": emp["hourly_cost"],
            "chosen_by_model": c is not None,
            "weeks_worked": c["weeks_worked"] if c else 0,
            "total_hours_assigned": c["total_hours"] if c else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        ["chosen_by_model", "total_hours_assigned", "weekly_hours_demand"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
