"""
model.py
--------
MILP for Combibrug workforce allocation.

Decision variables
    x[i, j] >= 0          weekly hours of employee i at location j
    y[i, j] in {0, 1}     1 if employee i is assigned to location j
    s[i]    >= 0          underassignment slack for permanent employees only

Objective
    minimise  sum_{i,j} cost_i * x[i,j]  +  underassignment_penalty * sum_perm s[i]

Constraints
    C1a (permanent):  sum_j x[i,j] + s[i] = contract_hours_i
    C1b (freelancer): sum_j x[i,j] <= contract_hours_i
    C2  Each location's demand is fully met by assigned hours
    C3  Company boundary (enforced by the eligibility pre-filter)
    C4  Role eligibility (enforced by the eligibility pre-filter)
    C5  Linking: x[i,j] <= contract_hours_i * y[i,j]
    C6  Optional: per-role headcount per location

Freelancers are real employees in the employee table (is_freelancer = True),
not an abstract per-location fallback variable. The underassignment penalty
applies only to permanent staff, so the solver fills permanent capacity
first and only uses freelancers when there's genuinely nothing left.

Unavailable employees (vacation, sick leave) are filtered out before the
model is built.
"""

import pandas as pd
from pulp import (
    LpProblem, LpVariable, LpMinimize, LpBinary,
    lpSum, PULP_CBC_CMD, LpStatus, value,
)

from data_loader import ALL_ROLES


# Fallback rates used when an employee has no hourly_rate and no rates-file
# match, or is flagged as a stagiair.
DEFAULT_FALLBACK_RATE = 22.0
DEFAULT_STAGE_RATE = 8.0

# Per-hour penalty applied to underassigned permanent staff. Keep this much
# higher than any real wage so the solver always prefers to fully assign
# permanent employees before using freelancers.
DEFAULT_UNDERASSIGNMENT_PENALTY = 1000.0


# ---------------------------------------------------------------------------
# Cost-table helper
# ---------------------------------------------------------------------------

def apply_hourly_costs(employees_df, rates_df=None,
                       rate_column="rate_incl",
                       fallback_rate=DEFAULT_FALLBACK_RATE,
                       stage_rate=DEFAULT_STAGE_RATE):
    """Attach an 'hourly_cost' column to employees_df.

    Resolution order for each employee's hourly cost:
        1. ``hourly_rate`` column on the employee row, if present and not NaN
        2. rates_df by id, using ``rate_column`` (or stage_rate if flagged)
        3. fallback_rate (or stage_rate if contract_type == 'stage')

    Returns
    -------
    (out_df, missing_ids) — missing_ids is the list of employee ids that
    had no direct rate and no match in rates_df.
    """
    out = employees_df.copy()
    missing_ids = []

    has_direct = "hourly_rate" in out.columns
    rates_lookup = None
    if rates_df is not None and not rates_df.empty:
        rates_lookup = rates_df.set_index("id")

    costs, sources = [], []
    for _, row in out.iterrows():
        eid = row["id"]

        # 1. Direct hourly_rate on the employee row
        if has_direct:
            direct = row.get("hourly_rate")
            if direct is not None and not pd.isna(direct):
                costs.append(float(direct))
                sources.append("employee_template")
                continue

        # 2. Rates file
        if rates_lookup is not None and eid in rates_lookup.index:
            r = rates_lookup.loc[eid]
            if bool(r.get("is_stage", False)):
                costs.append(stage_rate)
                sources.append("stage")
                continue
            v = r.get(rate_column)
            if pd.isna(v) or v is None:
                costs.append(fallback_rate)
                sources.append("fallback")
            else:
                costs.append(float(v))
                sources.append(rate_column)
            continue

        # 3. Nothing found — use stage rate if flagged, else fallback
        if str(row.get("contract_type", "")).lower() == "stage":
            costs.append(stage_rate)
            sources.append("stage")
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

def solve_allocation(employees, locations,
                     underassignment_penalty=DEFAULT_UNDERASSIGNMENT_PENALTY,
                     enforce_headcount=True,
                     time_limit=120,
                     verbose=False):
    """Build and solve the MILP.

    Parameters
    ----------
    employees : DataFrame
        Must contain columns id, role, company, contract_hours,
        contract_type, hourly_cost. Optional: is_freelancer, available
        (default False / True respectively if the columns are missing).
    locations : DataFrame
        Must contain columns location, company, weekly_hours, plus
        req_<role> columns for each role in ALL_ROLES.
    underassignment_penalty : float
        Per-hour cost added to the objective when a permanent employee is
        not fully assigned.
    enforce_headcount : bool
        If True, apply per-role headcount constraints (C6).
    time_limit : int
        Solver time limit in seconds.

    Returns
    -------
    status : str            e.g. "Optimal", "Infeasible"
    assignments : DataFrame one row per non-zero (employee, location, hours)
    summary : dict          objective, perm/freelancer cost split,
                            underassignment table, hours totals.
                            summary["error"] is set (non-None) on any
                            early-exit path so the UI can show a clear
                            message instead of crashing on missing keys.
    """
    if employees is None or employees.empty:
        return "Infeasible", pd.DataFrame(), _empty_summary(
            error="No employees were provided."
        )
    if locations is None or locations.empty:
        return "Infeasible", pd.DataFrame(), _empty_summary(
            error="No locations/demand were provided."
        )

    # --- Filter out unavailable employees (vacation / sick leave this week)
    all_emp = employees.to_dict("records")
    emp = [e for e in all_emp if bool(e.get("available", True))]
    unavailable_ids = [int(e["id"]) for e in all_emp if not bool(e.get("available", True))]

    if not emp:
        return "Infeasible", pd.DataFrame(), _empty_summary(
            error="Every employee is marked unavailable this week.",
            unavailable_employee_ids=unavailable_ids,
        )

    # Normalise is_freelancer on every record (handles missing column / NaN)
    for e in emp:
        v = e.get("is_freelancer", False)
        try:
            e["is_freelancer"] = bool(v) if not pd.isna(v) else False
        except TypeError:
            e["is_freelancer"] = bool(v)

    loc = locations.to_dict("records")
    emp_by_id = {e["id"]: e for e in emp}

    perm_ids = [e["id"] for e in emp if not e["is_freelancer"]]
    free_ids = [e["id"] for e in emp if e["is_freelancer"]]

    # --- Pre-filter eligible (employee, location) pairs.
    # Eligible if same company AND (location has no role requirement at all,
    # OR the location specifically requests the employee's role).
    eligible_pairs = []
    for e in emp:
        for l in loc:
            if e["company"] != l["company"]:
                continue
            total_req = sum(l.get(f"req_{r}", 0) for r in ALL_ROLES)
            if total_req > 0 and l.get(f"req_{e['role']}", 0) == 0:
                continue
            eligible_pairs.append((e["id"], l["location"]))

    if not eligible_pairs:
        return "Infeasible", pd.DataFrame(), _empty_summary(
            error=(
                "No eligible (employee, location) pairs were found. Check "
                "that at least one employee's company and role matches at "
                "least one location's company and role requirement."
            ),
            unavailable_employee_ids=unavailable_ids,
        )

    # --- Build the LP
    prob = LpProblem("Combibrug_Workforce", LpMinimize)

    x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0) for (i, j) in eligible_pairs}
    y = {(i, j): LpVariable(f"y_{i}_{j}", cat=LpBinary) for (i, j) in eligible_pairs}
    s = {i: LpVariable(f"s_{i}", lowBound=0) for i in perm_ids}

    pairs_by_emp = {}
    for (i, j) in eligible_pairs:
        pairs_by_emp.setdefault(i, []).append((i, j))

    pairs_by_loc = {}
    for (i, j) in eligible_pairs:
        pairs_by_loc.setdefault(j, []).append((i, j))

    # --- Objective: wage cost + underassignment penalty for permanent staff
    wage_cost = lpSum(emp_by_id[i]["hourly_cost"] * x[(i, j)]
                      for (i, j) in eligible_pairs)
    penalty = lpSum(underassignment_penalty * s[i] for i in perm_ids)
    prob += wage_cost + penalty

    # --- C1a: permanent employees -> assigned hours + slack = contract hours
    for i in perm_ids:
        e = emp_by_id[i]
        pairs = pairs_by_emp.get(i, [])
        prob += (
            lpSum(x[p] for p in pairs) + s[i] == e["contract_hours"],
            f"contract_perm_{i}",
        )

    # --- C1b: freelancers -> assigned hours capped at contract_hours
    for i in free_ids:
        e = emp_by_id[i]
        pairs = pairs_by_emp.get(i, [])
        if pairs:
            prob += (
                lpSum(x[p] for p in pairs) <= e["contract_hours"],
                f"cap_free_{i}",
            )

    # --- C2: demand coverage (hard constraint, every location)
    for l in loc:
        pairs = pairs_by_loc.get(l["location"], [])
        prob += (
            lpSum(x[p] for p in pairs) >= l["weekly_hours"],
            f"demand_{l['location']}",
        )

    # --- C5: linking x and y
    for (i, j) in eligible_pairs:
        prob += x[(i, j)] <= emp_by_id[i]["contract_hours"] * y[(i, j)]

    # --- C6: optional headcount per role per location
    if enforce_headcount:
        for l in loc:
            for role in ALL_ROLES:
                req = l.get(f"req_{role}", 0)
                if req > 0:
                    eligible_emps = [
                        i for i in {p[0] for p in pairs_by_loc.get(l["location"], [])}
                        if emp_by_id[i]["role"] == role
                    ]
                    if eligible_emps:
                        prob += (
                            lpSum(y[(i, l["location"])] for i in eligible_emps) >= req,
                            f"headcount_{l['location']}_{role}",
                        )

    # --- Solve
    solver = PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
    prob.solve(solver)
    status = LpStatus[prob.status]

    # --- Extract assignments
    rows = []
    for (i, j), var in x.items():
        v = var.value() or 0
        if v > 0.01:
            rows.append({
                "employee_id": i,
                "employee_role": emp_by_id[i]["role"],
                "contract_type": emp_by_id[i]["contract_type"],
                "is_freelancer": emp_by_id[i]["is_freelancer"],
                "company": emp_by_id[i]["company"],
                "location": j,
                "hours": round(v, 2),
                "cost": round(v * emp_by_id[i]["hourly_cost"], 2),
            })
    assignments = pd.DataFrame(rows)

    # --- Underassignment per permanent employee (visible in output)
    under_rows = []
    for i in perm_ids:
        slack = s[i].value() or 0
        if slack > 0.01:
            e = emp_by_id[i]
            under_rows.append({
                "employee_id": i,
                "role": e["role"],
                "company": e["company"],
                "contract_hours": e["contract_hours"],
                "assigned_hours": round(e["contract_hours"] - slack, 2),
                "underassigned_hours": round(slack, 2),
            })
    underassignment = pd.DataFrame(under_rows)

    perm_cost = sum(r["cost"] for r in rows if not r["is_freelancer"])
    free_cost = sum(r["cost"] for r in rows if r["is_freelancer"])
    perm_hours = sum(r["hours"] for r in rows if not r["is_freelancer"])
    free_hours = sum(r["hours"] for r in rows if r["is_freelancer"])

    summary = {
        "status": status,
        "objective": value(prob.objective) if prob.objective is not None else None,
        "permanent_cost": float(perm_cost),
        "freelancer_cost": float(free_cost),
        "wage_cost_total": float(perm_cost + free_cost),
        "permanent_hours": float(perm_hours),
        "freelancer_hours": float(free_hours),
        "total_underassigned_hours": float(sum(r["underassigned_hours"] for r in under_rows)),
        "n_underassigned_employees": len(under_rows),
        "underassignment": underassignment,
        "unavailable_employee_ids": unavailable_ids,
        "error": None,
    }
    return status, assignments, summary


def _empty_summary(error=None, unavailable_employee_ids=None):
    """Summary dict with safe defaults for early-exit / error paths, so the
    UI can always read the same keys without raising a KeyError."""
    return {
        "status": "Infeasible",
        "objective": None,
        "permanent_cost": 0.0,
        "freelancer_cost": 0.0,
        "wage_cost_total": 0.0,
        "permanent_hours": 0.0,
        "freelancer_hours": 0.0,
        "total_underassigned_hours": 0.0,
        "n_underassigned_employees": 0,
        "underassignment": pd.DataFrame(),
        "unavailable_employee_ids": unavailable_employee_ids or [],
        "error": error,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_assignments(employees, locations, assignments, summary):
    """Return a list of plain-text validation messages for the Diagnostics tab."""
    checks = []

    if summary.get("error"):
        checks.append(f"FAIL: {summary['error']}")
        return checks

    if assignments.empty and not locations.empty:
        checks.append("FAIL: no assignments were produced — model is likely infeasible.")
        return checks

    if not assignments.empty:
        # 1. No employee exceeds contract hours
        hours_per_emp = assignments.groupby("employee_id")["hours"].sum()
        over = []
        for eid, h in hours_per_emp.items():
            match = employees.loc[employees["id"] == eid, "contract_hours"]
            if match.empty:
                continue
            contract = float(match.iloc[0])
            if h > contract + 0.01:
                over.append((eid, h, contract))
        if over:
            for eid, h, c in over:
                checks.append(f"FAIL: employee {eid} assigned {h:.1f}h > contract {c:.1f}h")
        else:
            checks.append("OK: no employee exceeds contract hours")

        # 2. No cross-company assignments
        merged = assignments.merge(
            locations[["location", "company"]].rename(columns={"company": "loc_company"}),
            on="location", how="left",
        )
        cross = merged[merged["company"] != merged["loc_company"]]
        if len(cross) > 0:
            checks.append(f"FAIL: {len(cross)} cross-company assignments")
        else:
            checks.append("OK: no cross-company assignments")

        # 3. Every location is covered
        covered = assignments.groupby("location")["hours"].sum()
        under = []
        for _, l in locations.iterrows():
            got = covered.get(l["location"], 0)
            if got + 0.01 < l["weekly_hours"]:
                under.append((l["location"], got, l["weekly_hours"]))
        if under:
            for loc, got, need in under[:5]:
                checks.append(f"FAIL: {loc} under-covered ({got:.1f} < {need:.1f})")
            if len(under) > 5:
                checks.append(f"  ... and {len(under) - 5} more under-covered locations")
        else:
            checks.append("OK: all locations fully covered")

    n_under = summary.get("n_underassigned_employees", 0)
    if n_under > 0:
        total = summary.get("total_underassigned_hours", 0)
        checks.append(
            f"INFO: {n_under} permanent employee(s) underassigned for "
            f"{total:.1f}h total — carry forward to next week or review demand."
        )
    else:
        checks.append("OK: every available permanent employee is fully assigned")

    unavail = summary.get("unavailable_employee_ids", [])
    if unavail:
        checks.append(
            f"INFO: {len(unavail)} employee(s) excluded as unavailable this week: "
            f"{sorted(unavail)[:10]}{'...' if len(unavail) > 10 else ''}"
        )

    return checks


# ---------------------------------------------------------------------------
# "Why was this employee assigned where they were?" diagnostic
# ---------------------------------------------------------------------------

def explain_assignment(employee_id, employees, locations, assignments):
    """Return a small DataFrame showing the alternatives the model could have
    chosen for a given employee, sorted by demand. Used by the Diagnostics tab.
    """
    if employees.empty or employee_id not in employees["id"].values:
        return pd.DataFrame()

    emp = employees[employees["id"] == employee_id].iloc[0]
    chosen_locs = set(
        assignments.loc[assignments["employee_id"] == employee_id, "location"]
        if not assignments.empty else []
    )

    rows = []
    for _, l in locations.iterrows():
        if l["company"] != emp["company"]:
            continue
        total_req = sum(l.get(f"req_{r}", 0) for r in ALL_ROLES)
        if total_req > 0 and l.get(f"req_{emp['role']}", 0) == 0:
            continue
        rows.append({
            "location": l["location"],
            "weekly_hours_demand": l["weekly_hours"],
            "duration_months": l.get("duration_months", 12),
            "hourly_cost_if_chosen": emp.get("hourly_cost"),
            "chosen_by_model": l["location"] in chosen_locs,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["chosen_by_model", "weekly_hours_demand"],
        ascending=[False, False],
    ).reset_index(drop=True)
