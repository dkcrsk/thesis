

"""
app.py
------
Streamlit dashboard for the Combibrug workforce planning artifact.

Run from this directory with:

    streamlit run app.py

The app has four tabs:
    1. Upload files — pick the input files from your computer
    2. Inputs       — view loaded data and cost-source breakdown
    3. Run plan     — solve the MILP, see KPIs and the assignment table
    4. Diagnostics  — validation checks and per-employee explanations
"""

from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

from data_loader import (
    load_employee_template, generate_employee_template,
    load_demand_template, generate_demand_template_example,
    DataLoadError,
)
from model import (
    DEFAULT_FREELANCER_COST,
    solve_allocation,
    validate_assignments,
    explain_assignment,
)
from schedule import build_timetable, detect_day_conflicts, DAY_ORDER


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Combibrug Workforce Planner",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — cost settings only (file upload moved to main area)
# ---------------------------------------------------------------------------

st.sidebar.title("Combibrug Workforce Planner")

st.sidebar.markdown("### Cost assumptions")
freelancer_cost = st.sidebar.number_input(
    "Freelancer rate (€/h)",
    value=DEFAULT_FREELANCER_COST, step=1.0,
    help="What Combibrug pays an external ZZP'er per hour. Used as the "
         "alternative cost when internal staff can't cover demand.",
)
fallback_rate = st.sidebar.number_input(
    "Fallback rate — employees with no rate in the template (€/h)",
    value=22.0, step=1.0,
    help="Applied to any employee whose hourly_cost cell is blank in the "
         "employee template.",
)

st.sidebar.markdown("### Solver options")
block_size = st.sidebar.selectbox(
    "Hours block size (per assignment)",
    options=[1, 2, 4],
    index=0,  # default = 1 hour
    help="Internal assignments are made in whole multiples of this many hours. "
         "1 = full flexibility (recommended for daily scheduling).",
)

horizon_weeks = st.sidebar.number_input(
    "Planning horizon (weeks)",
    min_value=1, max_value=52, value=13, step=1,
    help="How many weeks to plan ahead. 13 weeks = one school quarter.",
)

continuity_penalty = st.sidebar.number_input(
    "Continuity penalty (€ per switch)",
    min_value=0.0, max_value=500.0, value=25.0, step=5.0,
    help="Cost per (employee, site) switch between consecutive weeks. "
         "Higher = keep people at the same site longer. 0 disables it.",
)

location_penalty = st.sidebar.number_input(
    "Location-spread penalty (€ per extra site)",
    min_value=0.0, max_value=500.0, value=15.0, step=5.0,
    help="Cost per distinct site each employee is spread across. "
         "Higher = each person works fewer different sites. 0 disables it.",
)

stay_until_end = st.sidebar.checkbox(
    "Stay until project end (hard)", value=True,
    help="Once assigned to a project, an employee stays for its whole "
         "duration (unless they leave the company, per end_week).",
)

enforce_headcount = st.sidebar.checkbox(
    "Enforce per-role headcount", value=True,
    help="When ticked, the model enforces minimum staff per role per site.",
)

time_limit = st.sidebar.number_input(
    "Solver time limit (seconds)",
    min_value=10, max_value=600, value=180, step=10,
)


# ---------------------------------------------------------------------------
# Tabs  (Upload is now the first tab)
# ---------------------------------------------------------------------------

tab_upload, tab_inputs, tab_run, tab_diag = st.tabs(
    ["📁 Upload files", "📋 Inputs", "▶️ Run plan", "🔍 Diagnostics"]
)


# ---------------------------------------------------------------------------
# Tab 0 — Upload files
# ---------------------------------------------------------------------------

with tab_upload:
    st.header("Upload your data files")
    st.write(
        "Upload two Excel files: the **employee template** (staff + hourly rates "
        "in one file) and the **demand template** (the sites to staff this cycle). "
        "Both have a download button below if you don't have them yet."
    )

    from io import BytesIO
    import tempfile, os, io

    col_emp, col_dem = st.columns(2)

    # ---- Employee template ----
    with col_emp:
        st.subheader("1 — Employee template (required)")
        emp_upload = st.file_uploader(
            "Choose employee template",
            type=["xlsx"],
            key="emp_uploader",
            label_visibility="collapsed",
        )
        st.caption(
            "Columns: id, role, company, contract_hours, hourly_cost, "
            "contract_type, end_week (optional), is_dreammaker (optional)."
        )
        with st.expander("Need to create the employee template?"):
            st.write(
                "Upload your raw staff file (medewerkers…xlsx) once and the app "
                "will pre-fill an employee template you can edit (add the real "
                "hourly_cost, mark Dreammakers, set end_week for anyone leaving)."
            )
            seed_upload = st.file_uploader(
                "Raw medewerkers file",
                type=["xlsx"], key="seed_uploader",
            )
            if seed_upload is not None:
                tmp_in = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False); tmp_in.close()
                tmp_out = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False); tmp_out.close()
                try:
                    with open(tmp_in.name, "wb") as f:
                        f.write(seed_upload.read())
                    generate_employee_template(tmp_in.name, tmp_out.name)
                    with open(tmp_out.name, "rb") as f:
                        emp_template_bytes = f.read()
                    st.download_button(
                        "⬇️  Download pre-filled employee template",
                        data=emp_template_bytes,
                        file_name="employee_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"Could not build the template: {e}")
                finally:
                    os.unlink(tmp_in.name); os.unlink(tmp_out.name)

    # ---- Demand template ----
    with col_dem:
        st.subheader("2 — Demand template (required)")
        loc_upload = st.file_uploader(
            "Choose demand template",
            type=["xlsx", "csv"],
            key="loc_template_uploader",
            label_visibility="collapsed",
        )
        st.caption(
            "Columns: site, company, weekly_hours, project, duration_months, "
            "start_week, min_headcount, project_leader_id (all optional except "
            "the first three)."
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False); tmp.close()
        try:
            generate_demand_template_example(tmp.name)
            with open(tmp.name, "rb") as f:
                demand_bytes = f.read()
        finally:
            os.unlink(tmp.name)
        st.download_button(
            "⬇️  Download example demand template",
            data=demand_bytes,
            file_name="demand_template_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()

    if emp_upload is None or loc_upload is None:
        missing = []
        if emp_upload is None:
            missing.append("employee template")
        if loc_upload is None:
            missing.append("demand template")
        st.info(f"Waiting for: {', '.join(missing)}. Upload the files above to continue.")
        st.stop()

    @st.cache_data(show_spinner="Reading and cleaning files…")
    def _load_uploaded(emp_bytes, emp_name, loc_bytes, loc_name):
        emp_buf = io.BytesIO(emp_bytes); emp_buf.name = emp_name
        loc_buf = io.BytesIO(loc_bytes); loc_buf.name = loc_name
        employees = load_employee_template(emp_buf)
        locations = load_demand_template(loc_buf)
        return employees, locations

    try:
        employees_df, locations_df = _load_uploaded(
            emp_upload.read(), emp_upload.name,
            loc_upload.read(), loc_upload.name,
        )
    except DataLoadError as e:
        st.error(f"❌ Could not load the files.\n\n{e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while reading files: {e}")
        st.stop()

    # Apply the sidebar fallback rate to any employee with a blank hourly_cost
    missing_rate_ids = employees_df.loc[
        employees_df["hourly_cost"].isna(), "id"
    ].tolist()
    employees_df = employees_df.copy()
    employees_df["hourly_cost"] = employees_df["hourly_cost"].fillna(fallback_rate)

    st.session_state["employees_df"]     = employees_df
    st.session_state["locations_df"]     = locations_df
    st.session_state["missing_rate_ids"] = missing_rate_ids

    st.success("✅ Files loaded successfully.")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Active employees", len(employees_df))
    s2.metric("Sites to staff", len(locations_df))
    s3.metric("Total weekly demand", f"{locations_df['weekly_hours'].sum():.0f} h")
    s4.metric("Rates from template",
              f"{len(employees_df) - len(missing_rate_ids)} / {len(employees_df)}")
    if missing_rate_ids:
        st.warning(
            f"{len(missing_rate_ids)} employee(s) had no hourly_cost in the "
            f"template and were given the fallback rate (€{fallback_rate:.0f}/h): "
            f"{missing_rate_ids}"
        )
    st.caption("Switch to the Inputs tab to inspect the data, or go to Run plan.")


# ---------------------------------------------------------------------------
# Guard: all other tabs need the uploaded data
# ---------------------------------------------------------------------------

def _require_data():
    """Return (employees_df, locations_df, missing_rate_ids) from session
    state, or stop with a friendly message if files haven't been uploaded."""
    if "employees_df" not in st.session_state:
        st.info("Go to the **Upload files** tab first and upload your data.")
        st.stop()
    return (
        st.session_state["employees_df"],
        st.session_state["locations_df"],
        st.session_state["missing_rate_ids"],
    )




# ---------------------------------------------------------------------------
# Tab 1 — Inputs
# ---------------------------------------------------------------------------

with tab_inputs:
    employees_df, locations_df, missing_rate_ids = _require_data()

    st.header("Loaded data")
    st.write(
        "These are the cleaned employee and location tables that will feed the "
        "optimisation model. Edit the cost assumptions in the sidebar to update "
        "the per-employee hourly cost shown below."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active employees", len(employees_df))
    col2.metric("Project locations", len(locations_df))
    col3.metric("Total weekly demand", f"{locations_df['weekly_hours'].sum():.0f} h")
    col4.metric(
        "Demand split (CC / Combibrug)",
        f"{locations_df[locations_df['company']=='CC']['weekly_hours'].sum():.0f} / "
        f"{locations_df[locations_df['company']=='Combibrug']['weekly_hours'].sum():.0f} h",
    )

    st.subheader("Employees")

    # Transparency: where did each employee's hourly cost come from?
    src_counts = employees_df["cost_source"].value_counts().to_dict()
    n_from_template = src_counts.get("template", 0)
    n_missing = src_counts.get("missing", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rates from template", n_from_template)
    c2.metric("Missing rate (fallback used)", len(missing_rate_ids))
    c3.metric("Dreammakers", int(employees_df.get("is_dreammaker", pd.Series(dtype=bool)).sum()))

    if missing_rate_ids:
        st.warning(
            f"{len(missing_rate_ids)} employee(s) had no hourly_cost in the "
            f"template and are using the fallback rate of €{fallback_rate:.2f}/h: "
            f"{sorted(missing_rate_ids)}"
        )

    st.dataframe(employees_df, use_container_width=True, height=300)

    st.subheader("Sites to staff")
    st.info(
        "Demand loaded from the **demand template**. Each row is a site to "
        "staff this planning cycle."
    )

    # Defensive: only show columns that actually exist.
    wanted_cols = [
        "site", "project", "company", "weekly_hours",
        "duration_months", "duration_weeks", "start_week",
        "min_headcount", "project_leader_id", "duration_source",
    ]
    if "site" not in locations_df.columns and "location" in locations_df.columns:
        st.error(
            "Your `data_loader.py` looks out of date — it produced a `location` "
            "column instead of `site`/`project`. Please re-download all three "
            "Python files (app.py, model.py, data_loader.py) together; updating "
            "only one of them leaves the app in an inconsistent state."
        )
    display_cols = [c for c in wanted_cols if c in locations_df.columns]
    st.dataframe(locations_df[display_cols], use_container_width=True, height=300)

    # --- Validate project_leader_id and min_headcount at upload time so
    # Rosa catches typos before pressing Solve.
    if "project_leader_id" in locations_df.columns:
        valid_emp_ids = set(employees_df["id"].astype(int))
        leader_problems = []
        for _, l in locations_df.iterrows():
            leader = l.get("project_leader_id")
            if leader is None or pd.isna(leader):
                continue
            try:
                leader_int = int(leader)
            except (TypeError, ValueError):
                leader_problems.append(
                    f"Site **{l['site']}**: project_leader_id '{leader}' is not a valid number."
                )
                continue
            if leader_int not in valid_emp_ids:
                leader_problems.append(
                    f"Site **{l['site']}**: project_leader_id {leader_int} "
                    f"is not in the staff file."
                )
                continue
            emp_row = employees_df[employees_df["id"] == leader_int].iloc[0]
            if emp_row["company"] != l["company"]:
                leader_problems.append(
                    f"Site **{l['site']}** (company {l['company']}): "
                    f"project_leader_id {leader_int} is contracted to {emp_row['company']}. "
                    f"The model will be infeasible — CC employees cannot lead Combibrug "
                    f"sites and vice versa."
                )
        if leader_problems:
            st.warning(
                "Project-leader issues detected:\n\n" +
                "\n".join(f"- {msg}" for msg in leader_problems)
            )

    if "min_headcount" in locations_df.columns and "weekly_hours" in locations_df.columns:
        # min_headcount × block_size must be <= weekly_hours, otherwise the
        # model can't simultaneously cover demand exactly AND have enough
        # people. Warn at the smallest block size since block_size is
        # configurable at solve time.
        hc_problems = []
        for _, l in locations_df.iterrows():
            mhc = int(l.get("min_headcount", 0) or 0)
            if mhc > 0 and l["weekly_hours"] < 2 * mhc:
                hc_problems.append(
                    f"Site **{l['site']}**: min_headcount={mhc} but only "
                    f"{l['weekly_hours']:.1f}h/week — at block_size 2, each of {mhc} "
                    f"people needs ≥2h, so demand must be ≥{2 * mhc}h. "
                    f"Increase weekly_hours or reduce min_headcount."
                )
        if hc_problems:
            st.warning(
                "Headcount feasibility warnings:\n\n" +
                "\n".join(f"- {msg}" for msg in hc_problems)
            )


# ---------------------------------------------------------------------------
# Tab 2 — Run plan
# ---------------------------------------------------------------------------

def _run_and_display(employees, locations, freelancer_c, enforce_hc, block_sz,
                     horizon, cont_penalty, loc_penalty, stay_end, t_limit):
    """Solve the multi-period MILP and return its results."""
    with st.spinner(f"Solving {horizon}-week multi-period MILP…"):
        status, assignments, summary = solve_allocation(
            employees,
            locations,
            freelancer_cost=freelancer_c,
            enforce_headcount=enforce_hc,
            block_size=block_sz,
            horizon_weeks=int(horizon),
            continuity_penalty=float(cont_penalty),
            location_penalty=float(loc_penalty),
            stay_until_end=bool(stay_end),
            time_limit=int(t_limit),
        )
    return status, assignments, summary


with tab_run:
    employees_df, locations_df, _ = _require_data()
    st.header("Generate the workforce plan")
    st.caption(
        f"Horizon: **{horizon_weeks} wks** | Continuity: **€{continuity_penalty:.0f}** | "
        f"Location spread: **€{location_penalty:.0f}** | "
        f"Stay-until-end: **{'on' if stay_until_end else 'off'}** | "
        f"Sites: **{len(locations_df)}**"
    )

    if st.button("Solve", type="primary"):
        status, assignments, summary = _run_and_display(
            employees_df,
            locations_df,
            freelancer_cost,
            enforce_headcount,
            block_size,
            horizon_weeks,
            continuity_penalty,
            location_penalty,
            stay_until_end,
            time_limit,
        )

        st.session_state["mp_status"] = status
        st.session_state["mp_assignments"] = assignments
        st.session_state["mp_summary"] = summary
        st.session_state["mp_locations_df"] = locations_df

        # Diagnostics tab uses these generic names.
        st.session_state["last_status"] = status
        st.session_state["last_assignments"] = assignments
        st.session_state["last_summary"] = summary
        st.session_state["last_locations_df"] = locations_df

    # --- Render results
    if "mp_summary" in st.session_state:
        status = st.session_state["mp_status"]
        assignments = st.session_state["mp_assignments"]
        summary = st.session_state["mp_summary"]
        plan_locations = st.session_state.get("mp_locations_df", locations_df)
        horizon = int(summary.get("horizon_weeks", horizon_weeks))
        weeks = list(range(1, horizon + 1))

        if status == "Optimal":
            st.success(f"{horizon}-week plan solved successfully.")
        else:
            st.error(f"Solver status: {status}")

        # Coverage — sum over active (location, week) cells.
        # Note: with the exact-equality demand constraint (C2), coverage is
        # always 100% by construction, and over-allocation is impossible.
        # We still compute the percentage for display.
        fl_period = summary.get("freelancer_hours_by_period", {})
        total_demand = 0.0
        total_covered = 0.0
        if not assignments.empty:
            ihw = assignments.groupby(["site", "week"])["hours"].sum()
        else:
            ihw = pd.Series(dtype=float)
        for _, loc in plan_locations.iterrows():
            sw = int(loc.get("start_week", 1))
            dur = int(loc.get("duration_weeks", horizon))
            active = [t for t in weeks if sw <= t < sw + dur]
            for t in active:
                supplied = ihw.get((loc["site"], t), 0) + fl_period.get((loc["site"], t), 0)
                total_demand += loc["weekly_hours"]
                total_covered += supplied

        # --- KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total cost", f"€{(summary.get('objective') or 0):,.0f}")
        k2.metric("Internal", f"€{(summary.get('internal_cost') or 0):,.0f}")
        k3.metric("Freelancer", f"€{(summary.get('external_cost') or 0):,.0f}")
        k4.metric(
            "Coverage",
            f"{(total_covered / total_demand * 100):.1f}%" if total_demand else "—",
        )
        k5.metric("Switches", f"{summary.get('n_switches', 0)}")

        # ══════════════════════════════════════════════════════════════════
        # MAIN OUTPUT — the daily timetable (Phase 2 schedule)
        # ══════════════════════════════════════════════════════════════════
        st.subheader("📅 Daily timetable")
        if assignments.empty:
            st.info("No assignments to schedule.")
        else:
            week_options = sorted(assignments["week"].unique())
            sel_week = st.selectbox(
                "Show timetable for week",
                options=week_options,
                index=0,
                format_func=lambda w: f"Week {w}",
            )
            timetable = build_timetable(assignments, week=int(sel_week))
            conflicts = detect_day_conflicts(timetable)

            if timetable.empty:
                st.info("No slots placed for this week.")
            else:
                # Pivot into a grid: rows = employee, columns = Mon..Fri,
                # cells = the slots that day ("Site 08:00–13:00").
                emp_lookup = employees_df.set_index("id")[["role", "company"]]

                def _cell(group):
                    parts = []
                    for _, r in group.iterrows():
                        tag = "≈" if r["assumed"] else ""
                        parts.append(f"{r['site']} {r['start']}–{r['end']}{tag}")
                    return "\n".join(parts)

                grid_rows = []
                for emp_id, g in timetable.groupby("employee_id"):
                    row = {"employee_id": emp_id}
                    row["role"] = emp_lookup["role"].get(emp_id, "")
                    for day in DAY_ORDER:
                        dg = g[g["day"] == day]
                        row[day] = _cell(dg) if not dg.empty else ""
                    grid_rows.append(row)

                grid = pd.DataFrame(grid_rows).set_index("employee_id")
                grid = grid[["role"] + DAY_ORDER]

                st.caption(
                    "Each row is an employee, each column a weekday. "
                    "Cells show site and time. A '≈' marks an **assumed** "
                    "time slot (the site had no known shift pattern, so the "
                    "hours were spread across mornings as a placeholder)."
                )
                st.dataframe(grid, use_container_width=True, height=520)

                n_assumed = int(timetable["assumed"].sum())
                n_real = int((~timetable["assumed"]).sum())
                cap1, cap2 = st.columns(2)
                cap1.metric("Slots from real patterns", n_real)
                cap2.metric("Assumed slots (≈)", n_assumed)

                if conflicts:
                    st.warning(
                        "Some employees have overlapping slots on the same day "
                        "(the weekly optimiser is day-blind, so two sites can "
                        "collide). Review these manually:\n\n" +
                        "\n".join(f"- {c}" for c in conflicts[:15])
                    )

                # Download the timetable
                from io import BytesIO
                tbuf = BytesIO()
                with pd.ExcelWriter(tbuf, engine="openpyxl") as writer:
                    timetable.to_excel(writer, sheet_name=f"week_{sel_week}", index=False)
                    grid.to_excel(writer, sheet_name="grid")
                st.download_button(
                    f"⬇️  Download week {sel_week} timetable (Excel)",
                    data=tbuf.getvalue(),
                    file_name=f"combibrug_timetable_week{sel_week}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        st.divider()

        # ──────────────────────────────────────────────────────────────────
        # SUPPORTING VIEW — weekly schedule (which weeks each person works)
        # ──────────────────────────────────────────────────────────────────
        st.subheader(f"Weekly overview ({horizon} weeks)")
        st.caption(
            "Supporting view: which weeks each employee works at each site. "
            "The daily timetable above is generated from this."
        )

        if assignments.empty:
            st.info("No internal assignments produced.")
        else:
            def _label(project, site, hrs):
                if project and project != site:
                    return f"{project} · {site} ({hrs}h)"
                return f"{site} ({hrs}h)"

            # Aggregate: one row per (employee, week) — if assigned to multiple
            # sites in the same week, concatenate them.
            grouped = (
                assignments
                .groupby(["employee_id", "week"], as_index=False)
                .apply(lambda g: pd.Series({
                    "cell": " · ".join(
                        _label(r.get("project"), r["site"], int(r["hours"]))
                        for _, r in g.iterrows()
                    )
                }))
                .reset_index(drop=True)
            )

            schedule = grouped.pivot(index="employee_id", columns="week", values="cell").fillna("")
            schedule = schedule.reindex(columns=weeks, fill_value="")
            schedule.columns = [f"W{t}" for t in weeks]

            # Add a role/company column on the left for readability
            emp_lookup = employees_df.set_index("id")[["role", "company"]]
            schedule.insert(0, "role", emp_lookup["role"].reindex(schedule.index).values)
            schedule.insert(1, "company", emp_lookup["company"].reindex(schedule.index).values)

            # Order: employees with more total hours first
            total_hours = assignments.groupby("employee_id")["hours"].sum()
            schedule["__sort"] = schedule.index.map(total_hours).fillna(0)
            schedule = schedule.sort_values("__sort", ascending=False).drop(columns="__sort")

            st.dataframe(schedule, use_container_width=True, height=400)

        # ──────────────────────────────────────────────────────────────────
        # Per-week cost chart
        # ──────────────────────────────────────────────────────────────────
        st.subheader("Cost per week")
        # Build cost-per-week from the assignments and freelancer hours.
        week_internal = (assignments.groupby("week")["cost"].sum().reindex(weeks, fill_value=0)
                        if not assignments.empty else pd.Series([0.0] * len(weeks), index=weeks))
        week_fl_hours = {t: sum(h for (loc, tt), h in fl_period.items() if tt == t) for t in weeks}
        week_fl_cost = pd.Series({t: week_fl_hours[t] * freelancer_cost for t in weeks})
        week_cost_df = pd.DataFrame({
            "internal": week_internal,
            "freelancer": week_fl_cost,
        })
        week_cost_df.index = [f"W{t}" for t in weeks]
        st.bar_chart(week_cost_df)

        # ──────────────────────────────────────────────────────────────────
        # Detailed assignments table
        # ──────────────────────────────────────────────────────────────────
        st.subheader("Detailed assignments")
        if assignments.empty:
            st.write("No assignments.")
        else:
            st.dataframe(
                assignments.sort_values(["week", "project", "site", "employee_id"]),
                use_container_width=True,
                height=320,
                hide_index=True,
            )

        # ──────────────────────────────────────────────────────────────────
        # Freelancer hours table (per site, summed over horizon)
        # ──────────────────────────────────────────────────────────────────
        # Lookup so the freelancer table can show the project name too
        site_to_project = dict(zip(plan_locations["site"], plan_locations["project"]))
        fl_rows = [
            {"project": site_to_project.get(s, s), "site": s,
             "freelancer_hours": round(h, 2)}
            for s, h in summary["freelancer_hours_by_site"].items()
            if h > 0.01
        ]
        if fl_rows:
            st.subheader("Freelancer hours needed (totals across the horizon)")
            st.dataframe(
                pd.DataFrame(fl_rows).sort_values("freelancer_hours", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        # ──────────────────────────────────────────────────────────────────
        # Download the full plan as Excel
        # ──────────────────────────────────────────────────────────────────
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Summary
            pd.DataFrame([{
                "horizon_weeks": horizon,
                "status": status,
                "total_cost": round(summary.get("objective") or 0, 2),
                "internal_cost": round(summary.get("internal_cost") or 0, 2),
                "freelancer_cost": round(summary.get("external_cost") or 0, 2),
                "freelancer_hours": round(summary.get("total_freelancer_hours") or 0, 1),
                "n_switches": summary.get("n_switches", 0),
                "continuity_penalty_per_switch": continuity_penalty,
            }]).to_excel(writer, sheet_name="summary", index=False)

            if not assignments.empty:
                assignments.to_excel(writer, sheet_name="assignments", index=False)
                # Schedule sheet (the pivoted view)
                if "schedule" in dir():
                    schedule.to_excel(writer, sheet_name="schedule")

            if fl_rows:
                pd.DataFrame(fl_rows).to_excel(writer, sheet_name="freelancers", index=False)

        st.download_button(
            f"Download {horizon}-week plan as Excel",
            data=buf.getvalue(),
            file_name=f"combibrug_{horizon}_week_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No plan has been generated yet. Press **Solve**.")


# ---------------------------------------------------------------------------
# Tab 4 — Diagnostics
# ---------------------------------------------------------------------------

with tab_diag:
    employees_df, locations_df, _ = _require_data()
    st.header("Diagnostics")
    st.write(
        "Validation checks and per-employee explanations for the most recent "
        "plan generated in the **Run plan** tab."
    )

    if "last_assignments" not in st.session_state:
        st.info("Generate a plan first in the **Run plan** tab.")
    else:
        assignments = st.session_state["last_assignments"]
        summary = st.session_state["last_summary"]
        status = st.session_state["last_status"]
        # Validate against the locations that were active in the diagnosed
        # month, not the full template — otherwise the "all covered" check
        # would fail for projects that were not yet active that month.
        diag_locations = st.session_state.get("last_locations_df", locations_df)

        st.subheader("Validation")
        checks = validate_assignments(employees_df, diag_locations, assignments, summary)
        for c in checks:
            if c.startswith("OK"):
                st.success(c)
            else:
                st.error(c)

        st.subheader("Why was this employee assigned where they were?")
        st.caption(
            "Pick an employee to see all locations they were eligible for, "
            "and which ones the model actually chose."
        )
        emp_id = st.selectbox(
            "Employee",
            options=employees_df["id"].tolist(),
            format_func=lambda i: (
                f"{i} — {employees_df.loc[employees_df['id']==i, 'role'].iloc[0]} "
                f"({employees_df.loc[employees_df['id']==i, 'company'].iloc[0]}, "
                f"{employees_df.loc[employees_df['id']==i, 'contract_hours'].iloc[0]:.0f}h)"
            ),
        )
        expl = explain_assignment(emp_id, employees_df, diag_locations, assignments)
        if expl.empty:
            st.info("No eligible locations for this employee.")
        else:
            st.dataframe(expl, use_container_width=True, height=300)

        st.subheader("Status")
        if status == "Infeasible":
            st.error(
                "The model is infeasible — no plan exists under the current "
                "constraints. Common causes: "
                "(1) total demand exceeds total available employee hours within "
                "a company, "
                "(2) a location's headcount requirement cannot be met by any "
                "eligible employee, "
                "(3) the cost table contains invalid values."
            )
        else:
            st.info(f"Last solver status: **{status}**")


