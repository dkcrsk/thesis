"""
model.py — Combibrug workforce planner (simple, robust weekly model)

Design goals (kept deliberately small so it always solves):
  - Weekly allocation only. No daily time-window layer.
  - Assign as many permanent / fixed-term staff as possible.
  - Put each project leader at their project as much as possible (SOFT, via a
    reward in the objective — never causes infeasibility).
  - Cover demand; if internal staff fall short, a freelancer gap variable fills
    the rest and is reported per site.

Variables
  x[i,j,t]  >= 0 integer : hours employee i works at site j in week t
  y[i,j,t]  in {0,1}     : 1 if employee i works at site j in week t
  u[j,t]    >= 0         : freelancer gap hours at site j in week t

Objective (minimise)
  internal cost + freelancer cost
  - leader_reward  * (leader assigned to own site)
  - permanent_reward * (permanent/fixed-term hours used)

The two rewards are NEGATIVE cost terms: they pull the optimiser toward using
permanent staff and placing leaders, without any hard constraint that could make
the model infeasible.

Constraints (all minimal, all names sanitised)
  H1  hours <= contract hours per employee per week
  H2  internal hours + freelancer gap == weekly demand (per active site-week)
  H3  link x and y
  H4  company + role eligibility (handled by building only eligible vars)
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
DEFAULT_CONTINUITY_PENALTY = 0.0   # not used in the simple model
DEFAULT_LOCATION_PENALTY = 0.0     # not used in the simple model

# Rewards (negative cost) — tuned so permanent staff and leaders are strongly
# preferred but the model never becomes infeasible.
LEADER_REWARD = 1000.0       # per week a leader is at their own site
PERMANENT_REWARD = 8.0       # per hour a permanent employee is used
FIXED_TERM_REWARD = 5.0      # per hour a fixed-term employee is used


def _safe(name):
    """Make a string safe for a PuLP variable / constraint name."""
    return "".join(c if c.isalnum() else "_" for c in str(name))


def solve_allocation(employees, locations,
                     freelancer_cost=DEFAULT_FREELANCER_COST,
                     enforce_headcount=True,
                     block_size=DEFAULT_BLOCK_SIZE,
                     horizon_weeks=DEFAULT_HORIZON_WEEKS,
                     continuity_penalty=DEFAULT_CONTINUITY_PENALTY,
                     location_penalty=DEFAULT_LOCATION_PENALTY,
                     utilisation_penalty=None,
                     stay_until_end=False,
                     enable_daily_scheduling=False,
                     max_hours_per_day=8,
                     time_limit=120,
                     verbose=False,
                     **_ignored_kwargs):
    """Build and solve a simple, robust weekly allocation MILP.

    Any extra keyword arguments passed by an older app.py are silently
    accepted and ignored, so a stale app.py won't crash the model.
    """

    emp = employees.to_dict("records")
    loc = locations.to_dict("records")
    emp_by_id = {e["id"]: e for e in emp}
    weeks = list(range(1, horizon_weeks + 1))

    def active_weeks(l):
        sw = int(l.get("start_week", 1) or 1)
        dur = int(l.get("duration_weeks", horizon_weeks) or horizon_weeks)
        return [t for t in weeks if sw <= t < sw + dur]

    loc_active = {l["site"]: active_weeks(l) for l in loc}

    # Eligible (employee, site) pairs: same company + role match (if site sets roles)
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

    prob = LpProblem("Combibrug_Simple", LpMinimize)

    max_blocks = {
        i: max(1, int(emp_by_id[i]["contract_hours"] // block_size))
        for i in {p[0] for p in eligible_pairs}
    }

    # ---- variables ---------------------------------------------------------
    x = {
        (i, j, t): LpVariable(
            f"x_{i}_{_safe(j)}_{t}", lowBound=0,
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

    # ---- objective: cost  -  permanent reward  -  leader reward ------------
    internal_cost = lpSum(
        emp_by_id[i]["hourly_cost"] * block_size * x[(i, j, t)]
        for (i, j, t) in eligible_pairs_t
    )
    freelancer_gap_cost = lpSum(
        freelancer_cost * u[(l["site"], t)]
        for l in loc for t in loc_active.get(l["site"], [])
    )

    # reward permanent / fixed-term usage (negative cost per hour)
    permanent_reward = lpSum(
        PERMANENT_REWARD * block_size * x[(i, j, t)]
        for (i, j, t) in eligible_pairs_t
        if emp_by_id[i].get("contract_type") == "permanent"
        and emp_by_id[i].get("role") != "Manager"
    )
    fixed_reward = lpSum(
        FIXED_TERM_REWARD * block_size * x[(i, j, t)]
        for (i, j, t) in eligible_pairs_t
        if emp_by_id[i].get("contract_type") == "fixed_term"
        and emp_by_id[i].get("role") != "Manager"
    )

    # reward leaders being present at their own site (SOFT)
    leader_terms = []
    leader_map = {}   # site -> leader_id (cleaned)
    for l in loc:
        raw = l.get("project_leader_id")
        if raw is None or pd.isna(raw):
            continue
        try:
            lid = int(raw)
        except (TypeError, ValueError):
            continue
        if lid not in emp_by_id:
            continue
        leader_map[l["site"]] = lid
        for t in loc_active.get(l["site"], []):
            if (lid, l["site"], t) in y:
                leader_terms.append(y[(lid, l["site"], t)])
    leader_reward = lpSum(LEADER_REWARD * yv for yv in leader_terms)

    prob += internal_cost + freelancer_gap_cost - permanent_reward - fixed_reward - leader_reward

    # ---- H1. contract hours per employee per week --------------------------
    for e in emp:
        eid = e["id"]
        for t in weeks:
            pairs_t = [p for p in eligible_pairs_t if p[0] == eid and p[2] == t]
            if pairs_t:
                prob += (
                    block_size * lpSum(x[p] for p in pairs_t) <= e["contract_hours"],
                    f"H1_hours_{eid}_w{t}",
                )

    # ---- H2. demand coverage = internal + freelancer gap -------------------
    for l in loc:
        for t in loc_active.get(l["site"], []):
            pairs_t = [p for p in eligible_pairs_t
                       if p[1] == l["site"] and p[2] == t]
            prob += (
                block_size * lpSum(x[p] for p in pairs_t) + u[(l["site"], t)]
                == l["weekly_hours"],
                f"H2_demand_{_safe(l['site'])}_w{t}",
            )

    # ---- H3. link x and y --------------------------------------------------
    for (i, j, t) in eligible_pairs_t:
        prob += (
            x[(i, j, t)] <= max_blocks.get(i, 0) * y[(i, j, t)],
            f"H3_link_{i}_{_safe(j)}_w{t}",
        )

    # ---- solve -------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  COMBIBRUG SIMPLE MODEL")
    print("=" * 55)
    print(f"  employees:        {len(emp)}")
    print(f"  sites:            {len(loc)}")
    print(f"  eligible (i,j,t): {len(eligible_pairs_t)}")
    print(f"  leaders mapped:   {len(leader_map)}  -> {leader_map}")
    print(f"  variables:        {len(x)+len(y)+len(u)}")
    print("=" * 55, flush=True)

    solver = PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    prob.solve(solver)
    status = LpStatus[prob.status]
    print(f"  Solver status: {status}\n", flush=True)

    site_to_project = {l["site"]: l.get("project") or l["site"] for l in loc}

    # ---- extract assignments ----------------------------------------------
    rows = []
    if status in ("Optimal", "Not Solved", "Undefined"):
        # Even if CBC reports a non-Optimal-but-feasible status, read whatever
        # values it produced (better than showing nothing). Empty if truly none.
        for (i, j, t), var in x.items():
            val = var.value()
            blocks = int(round(val)) if val is not None else 0
            hours = blocks * block_size
            if hours > 0:
                rows.append({
                    "employee_id": i,
                    "employee_role": emp_by_id[i]["role"],
                    "contract_type": emp_by_id[i].get("contract_type", ""),
                    "company": emp_by_id[i]["company"],
                    "project": site_to_project.get(j, j),
                    "site": j,
                    "week": t,
                    "blocks": blocks,
                    "hours": hours,
                    "cost": round(hours * emp_by_id[i]["hourly_cost"], 2),
                })
    assignments = pd.DataFrame(rows)

    # ---- freelancer gap reporting -----------------------------------------
    freelancer_hours = {}
    for l in loc:
        for t in loc_active.get(l["site"], []):
            v = u[(l["site"], t)].value()
            freelancer_hours[(l["site"], t)] = (v or 0) if v is not None else 0

    freelancer_hours_by_site = {}
    for (j, t), h in freelancer_hours.items():
        if h > 0.01:
            freelancer_hours_by_site[j] = freelancer_hours_by_site.get(j, 0.0) + h

    total_internal = float(sum(r["cost"] for r in rows))
    total_external = float(sum(h * freelancer_cost for h in freelancer_hours.values()))

    # which leaders actually got placed (for diagnostics)
    leaders_placed = {}
    for site, lid in leader_map.items():
        weeks_present = 0
        for t in loc_active.get(site, []):
            if (lid, site, t) in y:
                v = y[(lid, site, t)].value()
                if v is not None and round(v) == 1:
                    weeks_present += 1
        leaders_placed[site] = {"leader_id": lid,
                                "weeks_present": weeks_present,
                                "active_weeks": len(loc_active.get(site, []))}

    summary = {
        "status": status,
        "objective": value(prob.objective) if prob.objective is not None else None,
        "internal_cost": total_internal,
        "external_cost": total_external,
        "freelancer_cost": total_external,
        "n_switches": 0,
        "horizon_weeks": horizon_weeks,
        "freelancer_hours_by_period": freelancer_hours,
        "freelancer_hours_by_site": freelancer_hours_by_site,
        "total_freelancer_hours": float(sum(freelancer_hours.values())),
        "leaders_placed": leaders_placed,
        "daily_scheduling_enabled": False,
        "daily_slots": pd.DataFrame(),
    }
    return status, assignments, summary


def validate_assignments(employees, locations, assignments, summary):
    """Minimal validation messages for the Diagnostics tab."""
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
            checks.append(f"FAIL: employee {eid} week {week}: {h:.1f}h > contract {c:.1f}h")
    else:
        checks.append("OK: no contract-hour violations")

    merged = assignments.merge(
        locations[["site", "company"]].rename(columns={"company": "loc_co"}),
        on="site", how="left",
    )
    cross = merged[merged["company"] != merged["loc_co"]]
    checks.append(f"{'FAIL' if len(cross) else 'OK'}: "
                  f"{len(cross)} cross-company assignments")

    # leaders placed?
    lp = summary.get("leaders_placed", {})
    missing = [s for s, d in lp.items() if d["weeks_present"] == 0]
    if missing:
        checks.append(f"INFO: leaders not placed at: {missing}")
    else:
        checks.append(f"OK: all {len(lp)} project leaders placed at their sites")

    return checks


def explain_assignment(employee_id, employees, locations, assignments):
    """Show which sites an employee was eligible for and what they got."""
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
            "hourly_cost": emp["hourly_cost"],
            "chosen_by_model": c is not None,
            "weeks_worked": c["weeks_worked"] if c else 0,
            "total_hours_assigned": c["total_hours"] if c else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        ["chosen_by_model", "total_hours_assigned"], ascending=[False, False]
    ).reset_index(drop=True)
