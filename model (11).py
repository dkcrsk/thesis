"""
model.py — Combibrug multi-period workforce MILP

Objective: minimise total cost (internal staff + freelancer) plus soft penalties
           for switching employees between sites and for spreading one employee
           across many sites.

Constraints
    C1   Contract hours: each employee's weekly hours <= contract_hours
    C2   Demand coverage: internal + freelancer hours == site weekly demand
    C3   Company boundary: CC staff on CC sites only, Combibrug on Combibrug only
    C4   Role eligibility: employee role must match site's required roles (if set)
    C5   Assignment linking: k[i,j,t] <= max_blocks * y[i,j,t]
    C6   Per-role headcount per site per week (optional)
    C7   Continuity switch detection (soft penalty)
    C8   Minimum total headcount — Four-Eyes rule for Combiworld/MDT sites
    C9   Fixed project leader: named employee must be assigned every active week
    C10  Stay until project end: once assigned to a project, employee stays
         through all active weeks unless their end_week is reached first
    C11  Location-spread linking (soft penalty)
    C12  Minimum utilisation: permanent >= 40%, fixed_term >= 20% of contract hours
    C13a Non-overlap: same employee cannot take two time-overlapping windows on same day
    C13b 8-hour daily cap across all sites per employee per day
    C14  Daily link: sum of chosen window lengths == block_size * k[i,j,t]
"""

import pandas as pd
from pulp import (
    LpProblem, LpVariable, LpMinimize, LpBinary, LpInteger,
    lpSum, PULP_CBC_CMD, LpStatus, value,
)

from data_loader import ALL_ROLES


DEFAULT_FREELANCER_COST = 50.0
DEFAULT_BLOCK_SIZE = 1
DEFAULT_FALLBACK_RATE = 22.0
DEFAULT_STAGE_RATE = 8.0
DEFAULT_HORIZON_WEEKS = 13
DEFAULT_CONTINUITY_PENALTY = 25.0
DEFAULT_LOCATION_PENALTY = 15.0


def apply_hourly_costs(employees_df, rates_df=None,
                       rate_column="rate_incl",
                       fallback_rate=DEFAULT_FALLBACK_RATE,
                       stage_rate=DEFAULT_STAGE_RATE):
    """Attach an hourly_cost column to employees_df from an optional rates file."""
    out = employees_df.copy()
    missing_ids = []

    if rates_df is None or rates_df.empty:
        out["hourly_cost"] = fallback_rate
        out["cost_source"] = "fallback"
        missing_ids = list(out["id"])
        return out, missing_ids

    rates_lookup = rates_df.set_index("id")
    costs, sources = [], []
    for _, row in out.iterrows():
        eid = row["id"]
        if eid in rates_lookup.index:
            r = rates_lookup.loc[eid]
            if bool(r.get("is_stage", False)):
                costs.append(stage_rate)
                sources.append("stage")
            else:
                v = r.get(rate_column)
                if pd.isna(v) or v is None:
                    costs.append(fallback_rate)
                    sources.append("fallback")
                else:
                    costs.append(float(v))
                    sources.append(rate_column)
        else:
            costs.append(fallback_rate)
            sources.append("missing")
            missing_ids.append(int(eid))

    out["hourly_cost"] = costs
    out["cost_source"] = sources
    return out, missing_ids


def _safe(name):
    """Sanitise a string for use in a PuLP variable name."""
    return "".join(c if c.isalnum() else "_" for c in str(name))


def _hours_to_hhmm(h):
    """Convert decimal hours (8.5) to 'HH:MM' string ('08:30')."""
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
                     stay_until_end=True,
                     enable_daily_scheduling=False,
                     max_hours_per_day=8,
                     time_limit=120,
                     verbose=False):
    """Build and solve the multi-period MILP.

    Returns
    -------
    status : str            e.g. 'Optimal', 'Infeasible'
    assignments : DataFrame one row per non-zero (employee, site, week)
    summary : dict          cost breakdown, freelancer hours, daily slots
    """
    emp = employees.to_dict("records")
    loc = locations.to_dict("records")
    emp_by_id = {e["id"]: e for e in emp}

    weeks = list(range(1, horizon_weeks + 1))

    def active_weeks(l):
        sw = int(l.get("start_week", 1))
        dur = int(l.get("duration_weeks", horizon_weeks))
        return [t for t in weeks if sw <= t < sw + dur]

    loc_active = {l["site"]: active_weeks(l) for l in loc}

    # Pre-filter eligible (employee, site) pairs: same company + role match
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
    k = {
        (i, j, t): LpVariable(f"k_{i}_{j}_{t}", lowBound=0,
                               upBound=max_blocks.get(i, 0), cat=LpInteger)
        for (i, j, t) in eligible_pairs_t
    }
    y = {
        (i, j, t): LpVariable(f"y_{i}_{j}_{t}", cat=LpBinary)
        for (i, j, t) in eligible_pairs_t
    }
    u = {
        (l["site"], t): LpVariable(f"u_{l['site']}_{t}", lowBound=0)
        for l in loc for t in loc_active.get(l["site"], [])
    }

    # Switch variables for C7 continuity penalty
    s = {}
    if continuity_penalty > 0:
        for (i, j) in eligible_pairs:
            active = loc_active.get(j, [])
            for t in active:
                if t > 1 and (t - 1) in active:
                    s[(i, j, t)] = LpVariable(f"s_{i}_{j}_{t}", cat=LpBinary)

    # Location-spread indicators for C11 soft penalty
    loc_used = {}
    if location_penalty > 0:
        for (i, j) in eligible_pairs:
            loc_used[(i, j)] = LpVariable(f"locused_{i}_{_safe(j)}", cat=LpBinary)

    # Objective
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

    # C1. Contract hours per employee per week
    for e in emp:
        eid = e["id"]
        for t in weeks:
            pairs_t = [(i, j, tt) for (i, j, tt) in eligible_pairs_t
                       if i == eid and tt == t]
            if pairs_t:
                prob += (
                    block_size * lpSum(k[p] for p in pairs_t) <= e["contract_hours"],
                    f"contract_{eid}_w{t}",
                )

    # C2. Demand coverage: internal + freelancer == weekly_hours (equality)
    for l in loc:
        for t in loc_active.get(l["site"], []):
            pairs_t = [(i, j, tt) for (i, j, tt) in eligible_pairs_t
                       if j == l["site"] and tt == t]
            prob += (
                block_size * lpSum(k[p] for p in pairs_t) + u[(l["site"], t)]
                == l["weekly_hours"],
                f"demand_{l['site']}_w{t}",
            )
            prob += (
                block_size * lpSum(k[p] for p in pairs_t) <= l["weekly_hours"],
                f"no_overalloc_{l['site']}_w{t}",
            )

    # C5. Link k and y
    for (i, j, t) in eligible_pairs_t:
        prob += k[(i, j, t)] <= max_blocks.get(i, 0) * y[(i, j, t)]

    # C6. Per-role headcount per site per week (optional)
    if enforce_headcount:
        for l in loc:
            for role in ALL_ROLES:
                req = l.get(f"req_{role}", 0)
                if req > 0:
                    for t in loc_active.get(l["site"], []):
                        eligible_emps = [
                            i for i in {p[0] for p in eligible_pairs}
                            if emp_by_id[i]["role"] == role
                            and (i, l["site"], t) in y
                        ]
                        if eligible_emps:
                            prob += (
                                lpSum(y[(i, l["site"], t)] for i in eligible_emps) >= req,
                                f"headcount_{l['site']}_{role}_w{t}",
                            )

    # C7. Continuity: s[i,j,t] >= |y[i,j,t] - y[i,j,t-1]|
    for (i, j, t) in s:
        prob += s[(i, j, t)] >= y[(i, j, t)] - y[(i, j, t - 1)]
        prob += s[(i, j, t)] >= y[(i, j, t - 1)] - y[(i, j, t)]

    # C8. Four-Eyes rule: minimum headcount per site + k >= y for each person
    for l in loc:
        min_hc = int(l.get("min_headcount", 0) or 0)
        if min_hc > 0:
            for t in loc_active.get(l["site"], []):
                eligible_emps = [
                    i for i in {p[0] for p in eligible_pairs}
                    if (i, l["site"], t) in y
                ]
                if eligible_emps:
                    prob += (
                        lpSum(y[(i, l["site"], t)] for i in eligible_emps) >= min_hc,
                        f"min_headcount_{l['site']}_w{t}",
                    )
                    for i in eligible_emps:
                        prob += (
                            k[(i, l["site"], t)] >= y[(i, l["site"], t)],
                            f"min_block_{i}_{l['site']}_w{t}",
                        )

    # C9. Fixed project leader must be assigned every active week
    for l in loc:
        leader_id = l.get("project_leader_id")
        if leader_id is None or (isinstance(leader_id, float) and pd.isna(leader_id)):
            continue
        try:
            leader_id = int(leader_id)
        except (TypeError, ValueError):
            continue
        if leader_id not in emp_by_id:
            continue
        for t in loc_active.get(l["site"], []):
            if (leader_id, l["site"], t) not in y:
                continue
            prob += (
                y[(leader_id, l["site"], t)] == 1,
                f"project_leader_{leader_id}_{l['site']}_w{t}",
            )
            prob += (
                k[(leader_id, l["site"], t)] >= 1,
                f"leader_block_{leader_id}_{l['site']}_w{t}",
            )

    # C10. Stay until project end (monotonicity on y_proj)
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
                elig_sites = [sname for sname in sites if (i, sname) in eligible_pairs]
                if not elig_sites:
                    continue
                for t in proj_active[proj]:
                    site_vars = [(i, sname, t) for sname in elig_sites
                                 if (i, sname, t) in y]
                    if not site_vars:
                        continue
                    yp = LpVariable(f"yproj_{i}_{_safe(proj)}_{t}", cat=LpBinary)
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
                f"stay_{i}_{_safe(proj)}_{t}",
            )

    # C11. loc_used[(i,j)] = 1 if employee i works site j in any week
    for (i, j) in loc_used:
        week_vars = [(i, j, t) for t in loc_active.get(j, []) if (i, j, t) in y]
        for wv in week_vars:
            prob += loc_used[(i, j)] >= y[wv]

    # C12. Minimum utilisation: permanent >= 40%, fixed_term >= 20% of contract hours
    MIN_UTIL = {"permanent": 0.40, "fixed_term": 0.20}
    for e in emp:
        fraction = MIN_UTIL.get(e.get("contract_type", ""), 0.0)
        if fraction <= 0:
            continue
        min_hours = e["contract_hours"] * fraction
        if min_hours <= 0:
            continue
        eid = e["id"]
        for t in weeks:
            pairs_t = [(i, j, tt) for (i, j, tt) in eligible_pairs_t
                       if i == eid and tt == t]
            if not pairs_t:
                continue
            end_w = e.get("end_week")
            try:
                end_w = int(end_w) if end_w is not None and not pd.isna(end_w) else None
            except (TypeError, ValueError):
                end_w = None
            if end_w is not None and t > end_w:
                continue
            prob += (
                block_size * lpSum(k[p] for p in pairs_t) >= min_hours,
                f"min_util_{eid}_w{t}",
            )

    # Daily scheduling layer: C13a non-overlap, C13b 8h cap, C14 weekly link
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

        # C13a. Non-overlap: two overlapping windows on the same day -> at most one
        for i in emp_ids_with_pairs:
            for t in weeks:
                for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                    day_slots = []
                    for (j, d), entries in site_day_windows.items():
                        if d != day:
                            continue
                        if (i, j, t) not in y:
                            continue
                        for (w_idx, s_h, e_h, _) in entries:
                            if (i, j, t, w_idx) in z:
                                day_slots.append((j, w_idx, s_h, e_h))

                    for a in range(len(day_slots)):
                        for b in range(a + 1, len(day_slots)):
                            ja, wa, sa, ea = day_slots[a]
                            jb, wb, sb, eb = day_slots[b]
                            if sa < eb and sb < ea:
                                prob += (
                                    z[(i, ja, t, wa)] + z[(i, jb, t, wb)] <= 1,
                                    f"nooverlap_{i}_{_safe(ja)}_{wa}_"
                                    f"{_safe(jb)}_{wb}_{day}_w{t}",
                                )

        # C13b. 8-hour daily cap across all sites
        for i in emp_ids_with_pairs:
            for t in weeks:
                for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                    day_terms = []
                    for (j, d), entries in site_day_windows.items():
                        if d != day:
                            continue
                        if (i, j, t) not in y:
                            continue
                        for (w_idx, s_h, e_h, length) in entries:
                            if (i, j, t, w_idx) in z:
                                day_terms.append(length * z[(i, j, t, w_idx)])
                    if day_terms:
                        prob += (
                            lpSum(day_terms) <= max_hours_per_day,
                            f"day8h_{i}_{day}_w{t}",
                        )

        # C14. Weekly k must equal sum of chosen window lengths
        for (i, j, t) in eligible_pairs_t:
            wins = site_windows.get(j)
            if not wins:
                continue
            window_terms = [
                (e_h - s_h) * z[(i, j, t, w_idx)]
                for w_idx, (_, s_h, e_h) in enumerate(wins)
            ]
            prob += (
                lpSum(window_terms) == block_size * k[(i, j, t)],
                f"daily_link_{i}_{_safe(j)}_w{t}",
            )

        # z implies y
        for (i, j, t, w_idx), zv in z.items():
            if (i, j, t) in y:
                prob += zv <= y[(i, j, t)]

    # Solve
    solver = PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    prob.solve(solver)
    status = LpStatus[prob.status]

    site_to_project = {l["site"]: l.get("project") or l["site"] for l in loc}

    rows = []
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
    if enable_daily_scheduling and z:
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

    freelancer_hours = {
        (l["site"], t): (u[(l["site"], t)].value() or 0)
        for l in loc for t in loc_active.get(l["site"], [])
    }
    total_internal = float(sum(r["cost"] for r in rows))
    total_external = float(sum(h * freelancer_cost for h in freelancer_hours.values()))
    n_switches = int(round(sum((v.value() or 0) for v in s.values())))

    freelancer_hours_by_site = {}
    for (j, t), h in freelancer_hours.items():
        freelancer_hours_by_site[j] = freelancer_hours_by_site.get(j, 0.0) + h

    n_distinct_locations = (
        int(round(sum((v.value() or 0) for v in loc_used.values())))
        if loc_used else None
    )

    summary = {
        "status": status,
        "objective": value(prob.objective) if prob.objective is not None else None,
        "internal_cost": total_internal,
        "external_cost": total_external,
        "switch_penalty_cost": float(n_switches * continuity_penalty),
        "n_switches": n_switches,
        "n_distinct_location_assignments": n_distinct_locations,
        "horizon_weeks": horizon_weeks,
        "freelancer_hours_by_period": freelancer_hours,
        "freelancer_hours_by_site": freelancer_hours_by_site,
        "total_freelancer_hours": float(sum(freelancer_hours.values())),
        "daily_scheduling_enabled": enable_daily_scheduling,
        "daily_slots": daily_slots,
    }
    return status, assignments, summary


def validate_assignments(employees, locations, assignments, summary):
    """Return a list of validation messages for the Diagnostics tab."""
    checks = []

    if assignments.empty:
        checks.append("FAIL: no assignments were produced (likely an infeasible plan).")
        return checks

    hours_per_emp_week = assignments.groupby(["employee_id", "week"])["hours"].sum()
    over = []
    for (eid, week), h in hours_per_emp_week.items():
        contract = float(employees.loc[employees["id"] == eid, "contract_hours"].iloc[0])
        if h > contract + 0.01:
            over.append((eid, week, h, contract))
    if over:
        for eid, week, h, c in over[:5]:
            checks.append(f"FAIL: employee {eid} in week {week} assigned {h:.1f}h > contract {c:.1f}h")
        if len(over) > 5:
            checks.append(f"  ... and {len(over) - 5} more contract-hour violations")
    else:
        checks.append("OK: no employee exceeds contract hours in any week")

    merged = assignments.merge(
        locations[["site", "company"]].rename(columns={"company": "loc_company"}),
        on="site", how="left",
    )
    cross = merged[merged["company"] != merged["loc_company"]]
    if len(cross) > 0:
        checks.append(f"FAIL: {len(cross)} cross-company assignments")
    else:
        checks.append("OK: no cross-company assignments")

    horizon = int(summary.get("horizon_weeks", 13))
    weeks = list(range(1, horizon + 1))
    def _active(l):
        sw = int(l.get("start_week", 1))
        dur = int(l.get("duration_weeks", horizon))
        return [t for t in weeks if sw <= t < sw + dur]

    fl_period = summary.get("freelancer_hours_by_period", {})
    int_per_site_week = (assignments.groupby(["site", "week"])["hours"].sum()
                        if not assignments.empty else pd.Series(dtype=float))
    under = []
    for _, l in locations.iterrows():
        for t in _active(l):
            internal = int_per_site_week.get((l["site"], t), 0)
            external = fl_period.get((l["site"], t), 0)
            if internal + external + 0.01 < l["weekly_hours"]:
                under.append((l["site"], t, internal + external, l["weekly_hours"]))
    if under:
        for site, t, got, need in under[:5]:
            checks.append(f"FAIL: {site} in week {t} under-covered ({got:.1f} < {need:.1f})")
        if len(under) > 5:
            checks.append(f"  ... and {len(under) - 5} more under-covered (site, week) cells")
    else:
        checks.append("OK: all sites covered in all active weeks")

    n_switches = int(summary.get("n_switches", 0))
    if n_switches == 0:
        checks.append("OK: zero switches — perfect continuity across weeks")
    else:
        checks.append(f"INFO: {n_switches} (employee, location) switches across the horizon")

    return checks


def explain_assignment(employee_id, employees, locations, assignments):
    """Return a DataFrame showing what sites an employee was eligible for
    and how many weeks they were actually assigned to each one.
    Used by the Diagnostics tab.
    """
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
            "hourly_cost_if_chosen": emp["hourly_cost"],
            "chosen_by_model": c is not None,
            "weeks_worked": c["weeks_worked"] if c else 0,
            "total_hours_assigned": c["total_hours"] if c else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        ["chosen_by_model", "total_hours_assigned", "weekly_hours_demand"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
