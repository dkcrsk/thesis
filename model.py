"""
model.py
--------
The MILP for Combibrug workforce allocation.

Decision variables
    x[i, j] >= 0   weekly hours employee i is assigned to site j
    y[i, j] in {0, 1}   1 if employee i is assigned to site j at all
    u[j]    >= 0   weekly hours filled by a freelancer at site j

Note: 'site' is the physical place where work happens (e.g. Olympiaplein,
Buurtcentrum Watergraafsmeer). One project can span multiple sites — for
example, Combiworld-Oost may run at Park Frankendael, Sportpark Drieburg,
and Bouwkeet Indische Buurt simultaneously. The model decides per-site;
the project name is metadata for grouping in the output.

Objective
    minimise sum_{i,j} cost_i * x[i,j] + freelancer_cost * sum_j u[j]

Constraints
    C1  Each employee's total assigned hours <= contract_hours
    C2  Each site's demand is covered exactly (internal + freelancer hours)
    C3  Company boundary: CC employees only on CC sites, etc.
    C4  Role eligibility: employee can only go to sites requesting their role
    C5  Linking: x[i,j] <= contract_hours_i * y[i,j]
    C6  Optional per-role headcount per site
    C7  Soft continuity penalty (preferred: same person at same site over weeks)
    C8  Minimum total headcount per site (Four-Eyes rule for Combiworld/MDT)
    C9  Fixed project leader per site (specified employee must be assigned)
"""

import pandas as pd
from pulp import (
    LpProblem, LpVariable, LpMinimize, LpBinary, LpInteger,
    lpSum, PULP_CBC_CMD, LpStatus, value,
)

from data_loader import ALL_ROLES


def _safe(name):
    """Sanitise a string for use inside a PuLP variable/constraint name.
    PuLP rejects names with spaces or some punctuation, so replace anything
    that isn't alphanumeric with an underscore.
    """
    return "".join(c if c.isalnum() else "_" for c in str(name))


DEFAULT_FREELANCER_COST = 50.0  # Per Rosa: Combibrug charges €50/hour for ZZP'ers

# Default block size for discrete hour assignment.
# When block_size = 2, an employee is assigned to a location in multiples of
# 2 hours (0, 2, 4, 6, ...) — never 1.5 or 3.7 hours, which would be
# impractical in practice. Configurable in the dashboard sidebar.
DEFAULT_BLOCK_SIZE = 1

# Fallback rates used only when an employee is missing from the rates file
# or marked as 'stage'. These are exposed as sidebar inputs in app.py.
DEFAULT_FALLBACK_RATE = 22.0   # for employees missing from the rates file
DEFAULT_STAGE_RATE = 8.0       # for stagiaires (literal "stage" in rates file)


# ---------------------------------------------------------------------------
# Cost-table helper
# ---------------------------------------------------------------------------

def apply_hourly_costs(employees_df, rates_df=None,
                       rate_column="rate_incl",
                       fallback_rate=DEFAULT_FALLBACK_RATE,
                       stage_rate=DEFAULT_STAGE_RATE):
    """Attach an 'hourly_cost' column to employees_df.

    Parameters
    ----------
    employees_df : DataFrame
        Output of load_employees.
    rates_df : DataFrame or None
        Output of load_rates. If None, every employee gets ``fallback_rate``.
    rate_column : {"rate_base", "rate_incl"}
        Which column from rates_df to use. The planner picks this in the UI.
    fallback_rate : float
        Used for employees in employees_df who are not in rates_df, and
        for any rate_column value that is missing/NaN.
    stage_rate : float
        Used for rows flagged ``is_stage`` in rates_df.

    Returns
    -------
    (out_df, missing_ids) where missing_ids is the list of employee ids
    that were not found in rates_df. The UI can surface this as a warning.
    """
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


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

DEFAULT_HORIZON_WEEKS = 13
DEFAULT_CONTINUITY_PENALTY = 25.0  # cost per (employee, location) switch between periods
DEFAULT_LOCATION_PENALTY = 15.0    # cost per distinct extra site an employee is spread across


def solve_allocation(employees, locations,
                     freelancer_cost=DEFAULT_FREELANCER_COST,
                     enforce_headcount=True,
                     block_size=DEFAULT_BLOCK_SIZE,
                     horizon_weeks=DEFAULT_HORIZON_WEEKS,
                     continuity_penalty=DEFAULT_CONTINUITY_PENALTY,
                     location_penalty=DEFAULT_LOCATION_PENALTY,
                     stay_until_end=True,
                     time_limit=120,
                     verbose=False):
    """Build and solve the multi-period MILP.

    The model assigns employees to project locations across a planning
    horizon of `horizon_weeks` weeks. A project is active in a given
    week only if start_week <= t <= start_week + duration_weeks - 1.

    Parameters
    ----------
    employees : DataFrame
        Must contain columns id, role, company, contract_hours,
        contract_type, hourly_cost.
    locations : DataFrame
        Must contain columns location, company, weekly_hours,
        duration_weeks, start_week, plus req_<role> columns.
    freelancer_cost : float
        Hourly cost when demand is covered by an external freelancer.
    enforce_headcount : bool
        If True, apply per-role headcount constraints (C6).
    block_size : int
        Internal assignments are made in whole multiples of this many
        hours per week (default 2). Freelancer hours remain continuous.
    horizon_weeks : int
        Number of weeks in the planning horizon (default 13 — one quarter).
    continuity_penalty : float
        Cost added to the objective for each "switch" — i.e. each
        (employee, location) pair where the employee is assigned in
        week t but not in week t+1, or vice versa. Higher values make
        the optimiser prefer keeping people in the same place longer.
        Set to 0 to disable.
    time_limit : int
        Solver time limit in seconds.

    Returns
    -------
    status : str        e.g. "Optimal", "Infeasible"
    assignments : DataFrame  one row per non-zero (employee, location, week)
    summary : dict      objective, internal/external split, freelancer hours
    """
    emp = employees.to_dict("records")
    loc = locations.to_dict("records")
    emp_by_id = {e["id"]: e for e in emp}

    weeks = list(range(1, horizon_weeks + 1))

    # For each location, determine which weeks it's active in.
    # A location with start_week=3 and duration_weeks=10 is active in weeks 3..12.
    def active_weeks(l):
        sw = int(l.get("start_week", 1))
        dur = int(l.get("duration_weeks", horizon_weeks))
        return [t for t in weeks if sw <= t < sw + dur]

    loc_active = {l["site"]: active_weeks(l) for l in loc}

    # Pre-filter eligible (employee, location) pairs (the time-independent part).
    # An employee is eligible for a location if (a) same company, and
    # (b) either the location has no role requirement or the employee's role
    # is one of the requested roles.
    eligible_pairs = []
    for e in emp:
        for l in loc:
            if e["company"] != l["company"]:
                continue
            total_req = sum(l.get(f"req_{r}", 0) for r in ALL_ROLES)
            if total_req > 0 and l.get(f"req_{e['role']}", 0) == 0:
                continue
            eligible_pairs.append((e["id"], l["site"]))

    # Time-indexed variables: only create them for weeks the location is active.
    eligible_pairs_t = [
        (i, j, t) for (i, j) in eligible_pairs for t in loc_active.get(j, [])
    ]

    # Build the LP
    prob = LpProblem("Combibrug_Workforce_MultiPeriod", LpMinimize)

    # k[i,j,t] is the integer number of block_size-hour blocks employee i
    # works at location j in week t. Hours assigned = block_size * k.
    max_blocks = {
        i: int(emp_by_id[i]["contract_hours"] // block_size)
        for i in {p[0] for p in eligible_pairs}
    }
    k = {
        (i, j, t): LpVariable(
            f"k_{i}_{j}_{t}",
            lowBound=0,
            upBound=max_blocks.get(i, 0),
            cat=LpInteger,
        )
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

    # Continuity switch variables. s[(i, j, t)] = 1 if y differs between
    # week t-1 and week t for this (i, j) pair. Defined for t >= 2 and
    # only for pairs where both t-1 and t are in the active-weeks set.
    s = {}
    if continuity_penalty > 0:
        for (i, j) in eligible_pairs:
            active = loc_active.get(j, [])
            for t in active:
                if t > 1 and (t - 1) in active:
                    # both adjacent weeks are active — a switch is meaningful
                    s[(i, j, t)] = LpVariable(
                        f"s_{i}_{j}_{t}", cat=LpBinary
                    )

    # Location-spread indicators: loc_used[(i, j)] = 1 if employee i is
    # assigned to site j in ANY week. Used by the soft "minimize distinct
    # locations" penalty (C11) so the optimiser prefers concentrating each
    # employee in fewer sites.
    loc_used = {}
    if location_penalty > 0:
        for (i, j) in eligible_pairs:
            loc_used[(i, j)] = LpVariable(f"locused_{i}_{j}", cat=LpBinary)

    # Objective: internal cost + freelancer cost + continuity penalty
    #            + location-spread penalty.
    internal_cost = lpSum(
        emp_by_id[i]["hourly_cost"] * block_size * k[(i, j, t)]
        for (i, j, t) in eligible_pairs_t
    )
    external_cost = lpSum(freelancer_cost * u[(l["site"], t)]
                          for l in loc for t in loc_active.get(l["site"], []))
    switch_penalty = lpSum(continuity_penalty * s[(i, j, t)] for (i, j, t) in s)
    location_spread_penalty = lpSum(
        location_penalty * loc_used[(i, j)] for (i, j) in loc_used
    )
    prob += internal_cost + external_cost + switch_penalty + location_spread_penalty

    # C1. Contract hours per employee, per week
    #     block_size * sum_j k[i,j,t] <= h_i  for all i, t
    # Each employee's weekly hours cannot exceed their contract in any week.
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

    # C2. Demand coverage per location, per active week — exact equality.
    #     block_size * sum_i k[i,j,t] + u[j,t] == weekly_hours_j
    # Plus an explicit cap that internal alone cannot exceed demand:
    #     block_size * sum_i k[i,j,t] <= weekly_hours_j
    # Together these guarantee: internal supplies as much as it can in
    # block_size chunks, and freelancer hours u[j,t] fill the residual
    # (which can be any fractional amount, since u is continuous). No
    # over-allocation is possible.
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

    # C6. Headcount per role per location, per active week (optional)
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

    # C7. Continuity switch detection (only if continuity_penalty > 0).
    #     s[i,j,t] >= |y[i,j,t] - y[i,j,t-1]|, captured by two linear constraints:
    #         s[i,j,t] >= y[i,j,t] - y[i,j,t-1]
    #         s[i,j,t] >= y[i,j,t-1] - y[i,j,t]
    for (i, j, t) in s:
        prob += s[(i, j, t)] >= y[(i, j, t)] - y[(i, j, t - 1)]
        prob += s[(i, j, t)] >= y[(i, j, t - 1)] - y[(i, j, t)]

    # C8. Minimum total headcount per site, per active week (the Four-Eyes
    #     rule). For sites with min_headcount > 0, force at least that many
    #     distinct employees to be assigned each active week. We also force
    #     each assigned employee to contribute at least one block — without
    #     this, the model could set y=1 with k=0 (a phantom assignment that
    #     doesn't actually work any hours).
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
                    # Force k >= y for these pairs, so a flagged assignment
                    # contributes at least 1 block of real work.
                    for i in eligible_emps:
                        prob += (
                            k[(i, l["site"], t)] >= y[(i, l["site"], t)],
                            f"min_block_{i}_{l['site']}_w{t}",
                        )

    # C9. Fixed project leader per site (optional). For sites where
    #     project_leader_id is set, force that specific employee to be
    #     assigned (y = 1) every active week.
    for l in loc:
        leader_id = l.get("project_leader_id")
        if leader_id is None or (isinstance(leader_id, float) and pd.isna(leader_id)):
            continue
        try:
            leader_id = int(leader_id)
        except (TypeError, ValueError):
            continue
        # Verify the leader exists in the eligible pool for this site
        if leader_id not in emp_by_id:
            continue
        for t in loc_active.get(l["site"], []):
            if (leader_id, l["site"], t) not in y:
                # Leader is not eligible (likely a company mismatch). The
                # model can't satisfy the constraint; skip silently and let
                # the dashboard's diagnostics flag the issue.
                continue
            prob += (
                y[(leader_id, l["site"], t)] == 1,
                f"project_leader_{leader_id}_{l['site']}_w{t}",
            )
            # Like C8, force the leader to contribute at least one block.
            prob += (
                k[(leader_id, l["site"], t)] >= 1,
                f"leader_block_{leader_id}_{l['site']}_w{t}",
            )

    # C10. Stay until project end (hard, optional via stay_until_end).
    #      Once an employee is assigned to a PROJECT, they remain assigned to
    #      that project every subsequent active week, until either the project
    #      ends or the employee leaves the company (end_week), whichever is
    #      first. Implemented at the project level: y_proj[i,p,t] = 1 if the
    #      employee works ANY site of project p in week t.
    #
    #      The monotonicity constraint y_proj[i,p,t] <= y_proj[i,p,t+1] means
    #      "once you start, you can't drop out" — late starts are allowed,
    #      early exits are not. We cap it at the employee's end_week so an
    #      employee who leaves the company is not forced to keep working.
    if stay_until_end:
        # Build project -> list of sites lookup
        proj_sites = {}
        for l in loc:
            proj = l.get("project") or l["site"]
            proj_sites.setdefault(proj, []).append(l["site"])

        # Build project active-weeks = union of its sites' active weeks
        proj_active = {}
        for proj, sites in proj_sites.items():
            wk = set()
            for sname in sites:
                wk.update(loc_active.get(sname, []))
            proj_active[proj] = sorted(wk)

        # Project-level assignment indicator y_proj[(i, proj, t)]
        y_proj = {}
        emp_ids = {p[0] for p in eligible_pairs}
        for i in emp_ids:
            for proj, sites in proj_sites.items():
                # only if employee is eligible for at least one site of proj
                elig_sites = [sname for sname in sites if (i, sname) in eligible_pairs]
                if not elig_sites:
                    continue
                for t in proj_active[proj]:
                    # only weeks where the employee actually has variables
                    site_vars = [(i, sname, t) for sname in elig_sites
                                 if (i, sname, t) in y]
                    if not site_vars:
                        continue
                    yp = LpVariable(f"yproj_{i}_{_safe(proj)}_{t}", cat=LpBinary)
                    y_proj[(i, proj, t)] = yp
                    # Link: y_proj >= each site y; y_proj <= sum of site y's
                    for sv in site_vars:
                        prob += yp >= y[sv]
                    prob += yp <= lpSum(y[sv] for sv in site_vars)

        # Monotonicity: y_proj[i,p,t] <= y_proj[i,p,t_next] for consecutive
        # active weeks, capped at the employee's end_week.
        for (i, proj, t) in list(y_proj.keys()):
            end_w = emp_by_id[i].get("end_week")
            active = proj_active[proj]
            # find the next active week after t
            later = [w for w in active if w > t]
            if not later:
                continue
            t_next = later[0]
            if (i, proj, t_next) not in y_proj:
                continue
            # If the employee leaves before t_next, don't force them to stay.
            if end_w is not None and t_next > end_w:
                continue
            prob += (
                y_proj[(i, proj, t)] <= y_proj[(i, proj, t_next)],
                f"stay_{i}_{_safe(proj)}_{t}",
            )

    # C11. Location-spread linking. loc_used[(i,j)] must be 1 if employee i
    #      is assigned to site j in any week (the penalty in the objective
    #      then discourages spreading one person across many sites).
    for (i, j) in loc_used:
        week_vars = [(i, j, t) for t in loc_active.get(j, []) if (i, j, t) in y]
        for wv in week_vars:
            prob += loc_used[(i, j)] >= y[wv]

    # Solve
    solver = PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    prob.solve(solver)
    status = LpStatus[prob.status]

    # Site → project lookup, used to enrich the assignments DataFrame with
    # the project name. If a site has no project (older data), project = site.
    site_to_project = {}
    for l in loc:
        site_to_project[l["site"]] = l.get("project") or l["site"]

    # Extract assignments (one row per non-zero (i, j, t))
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

    freelancer_hours = {(l["site"], t): (u[(l["site"], t)].value() or 0)
                        for l in loc for t in loc_active.get(l["site"], [])}
    total_internal = float(sum(r["cost"] for r in rows))
    total_external = float(sum(h * freelancer_cost for h in freelancer_hours.values()))
    n_switches = int(round(sum((v.value() or 0) for v in s.values())))

    # Aggregate freelancer hours per site (summed across active weeks). The
    # dashboard uses this for the "freelancer hours per site" summary table.
    freelancer_hours_by_site = {}
    for (j, t), h in freelancer_hours.items():
        freelancer_hours_by_site[j] = freelancer_hours_by_site.get(j, 0.0) + h

    n_distinct_locations = int(round(sum((v.value() or 0) for v in loc_used.values()))) if loc_used else None

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
    }
    return status, assignments, summary


# ---------------------------------------------------------------------------
# Validation (the sanity checks from notebook section 6)
# ---------------------------------------------------------------------------

def validate_assignments(employees, locations, assignments, summary):
    """Return a list of plain-text validation messages for the diagnostics tab."""
    checks = []

    if assignments.empty:
        checks.append("FAIL: no assignments were produced (likely an infeasible plan).")
        return checks

    # 1. Per-week, no employee exceeds contract hours
    # Group by (employee, week) since contract limits apply per week.
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

    # 2. No cross-company assignments
    merged = assignments.merge(
        locations[["site", "company"]].rename(columns={"company": "loc_company"}),
        on="site", how="left",
    )
    cross = merged[merged["company"] != merged["loc_company"]]
    if len(cross) > 0:
        checks.append(f"FAIL: {len(cross)} cross-company assignments")
    else:
        checks.append("OK: no cross-company assignments")

    # 3. Every (site, active-week) is covered (internal + freelancer)
    # Build active-weeks lookup the same way the solver does.
    horizon = int(summary.get("horizon_weeks", 13))
    weeks = list(range(1, horizon + 1))
    def _active(l):
        sw = int(l.get("start_week", 1))
        dur = int(l.get("duration_weeks", horizon))
        return [t for t in weeks if sw <= t < sw + dur]

    fl_period = summary.get("freelancer_hours_by_period", {})
    # Per (site, week) internal coverage
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

    # 4. Switch count (informational, not a pass/fail)
    n_switches = int(summary.get("n_switches", 0))
    if n_switches == 0:
        checks.append("OK: zero switches — perfect continuity across weeks")
    else:
        checks.append(f"INFO: {n_switches} (employee, location) switches across the horizon")

    return checks


# ---------------------------------------------------------------------------
# Interpretability helper for the "why this assignment?" diagnostics view
# ---------------------------------------------------------------------------

def explain_assignment(employee_id, employees, locations, assignments):
    """Return a small DataFrame showing the alternatives the model could have
    chosen for a given employee, and how many weeks they actually worked at
    each one. Used by the Diagnostics tab.
    """
    if employee_id not in employees["id"].values:
        return pd.DataFrame()

    emp = employees[employees["id"] == employee_id].iloc[0]

    # Build a (site → weeks-worked, total-hours) summary for this employee
    chosen = {}
    if not assignments.empty:
        sub = assignments[assignments["employee_id"] == employee_id]
        for site_name, g in sub.groupby("site"):
            chosen[site_name] = {
                "weeks_worked": len(g),
                "total_hours": float(g["hours"].sum()),
                "weeks": sorted(int(w) for w in g["week"].tolist()),
            }

    # Eligible sites = same company, and the site requests the role
    # (or has no specific role requirement at all)
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
