

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


# Configuration

st.set_page_config(
    page_title="Combibrug Workforce Planner",
    page_icon="🧭",
    layout="wide",
)

# Sidebar — cost settings only (file upload moved to main area)

st.sidebar.title("Combibrug Workforce Planner")

st.sidebar.markdown("### Cost assumptions")
freelancer_cost = st.sidebar.number_input(
    "Last-resort freelancer rate (€/h)",
    value=DEFAULT_FREELANCER_COST, step=1.0,
    help="Used only when neither internal staff nor named freelancers in "
         "the employee template can cover a site's demand. Named "
         "freelancers in the template compete on their own hourly_cost, "
         "same as internal staff.",
)

st.sidebar.markdown("### Solver options")
block_size = st.sidebar.selectbox(
    "Hours block size (per assignment)",
    options=[1, 2, 4],
    index=0,  # default = 1 hour
    help="Internal assignments are integer multiples of this many hours. "
         "1 = full flexibility (recommended).",
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

# that applies to every employee. These are not user choices.
enable_daily_scheduling = True
max_hours_per_day = 8

enforce_headcount = st.sidebar.checkbox(
    "Enforce per-role headcount", value=True,
    help="When ticked, the model enforces minimum staff per role per site.",
)

time_limit = st.sidebar.number_input(
    "Solver time limit (seconds)",
    min_value=10, max_value=600, value=180, step=10,
)


# Tabs  (Upload is now the first tab)

tab_upload, tab_inputs, tab_run, tab_diag = st.tabs(
    ["📁 Upload files", "📋 Inputs", "▶️ Run plan", "🔍 Diagnostics"]
)


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
            "contract_type (permanent/fixed_term/stage/oproep/freelancer), "
            "end_week (optional), is_dreammaker (optional)."
        )
        with st.expander("Need a blank employee template to start from?"):
            st.write(
                "Download a blank template with example rows (including one "
                "freelancer row) to see the expected format. Fill it in with "
                "your staff, then upload it above."
            )
            if st.button("Generate blank employee template", key="gen_emp_template"):
                tmp_out = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
                tmp_out.close()
                try:
                    generate_employee_template(tmp_out.name)
                    with open(tmp_out.name, "rb") as f:
                        emp_template_bytes = f.read()
                    st.download_button(
                        "⬇️  Download blank employee template",
                        data=emp_template_bytes,
                        file_name="employee_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"Could not build the template: {e}")
                finally:
                    os.unlink(tmp_out.name)

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

    st.session_state["employees_df"] = employees_df
    st.session_state["locations_df"] = locations_df

    st.success("✅ Files loaded successfully.")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Active employees", len(employees_df))
    s2.metric("Sites to staff", len(locations_df))
    s3.metric("Total weekly demand", f"{locations_df['weekly_hours'].sum():.0f} h")
    freelancers = employees_df["is_freelancer"].sum() if "is_freelancer" in employees_df.columns else 0
    s4.metric("Named freelancers", int(freelancers))
    st.caption("Switch to the Inputs tab to inspect the data, or go to Run plan.")


def _require_data():
    """Return (employees_df, locations_df) from session state,
    or stop with a friendly message if files haven't been uploaded."""
    if "employees_df" not in st.session_state:
        st.info("Go to the **Upload files** tab first and upload your data.")
        st.stop()
    return (
        st.session_state["employees_df"],
        st.session_state["locations_df"],
    )


with tab_inputs:
    employees_df, locations_df = _require_data()

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

    n_freelancers = int(employees_df["is_freelancer"].sum()) if "is_freelancer" in employees_df.columns else 0
    n_dream = int(employees_df["is_dreammaker"].sum()) if "is_dreammaker" in employees_df.columns else 0
    n_perm = int((employees_df["contract_type"] == "permanent").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Permanent staff", n_perm)
    c2.metric("Named freelancers", n_freelancers)
    c3.metric("Dreammakers", n_dream)

    st.dataframe(employees_df, use_container_width=True, height=300)

    st.subheader("Sites to staff")
    st.info(
        "Demand loaded from the **demand template**. Each row is a site to "
        "staff this planning cycle."
    )

    wanted_cols = [
        "site", "project", "company", "weekly_hours",
        "duration_months", "duration_weeks", "start_week",
        "min_headcount", "project_leader_id", "duration_source",
    ]
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


def _run_and_display(employees, locations, freelancer_c, enforce_hc, block_sz,
                     horizon, cont_penalty, loc_penalty, stay_end,
                     enable_daily, max_per_day, t_limit):
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
            enable_daily_scheduling=bool(enable_daily),
            max_hours_per_day=int(max_per_day),
            time_limit=int(t_limit),
        )
    return status, assignments, summary


with tab_run:
    employees_df, locations_df = _require_data()
    st.header("Generate the workforce plan")
    st.caption(
        f"Horizon: **{horizon_weeks} wks** | "
        f"Continuity: **€{continuity_penalty:.0f}** | "
        f"Location spread: **€{location_penalty:.0f}** | "
        f"Stay-until-end: **{'on' if stay_until_end else 'off'}** | "
        f"Max **{max_hours_per_day}h**/day | "
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
            enable_daily_scheduling,
            max_hours_per_day,
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
        k3.metric("Named freelancers", f"€{(summary.get('named_freelancer_cost') or 0):,.0f}")
        k4.metric(
            "Coverage",
            f"{(total_covered / total_demand * 100):.1f}%" if total_demand else "—",
        )
        uncovered_h = summary.get("total_uncovered_hours", 0)
        k5.metric(
            "Uncovered hours",
            f"{uncovered_h:.1f} h",
            help="Hours that could not be assigned to any employee and require "
                 "last-resort freelancer fill. See the Diagnostics tab for details.",
        )

        # and the solver actually placed slots
        daily_slots = summary.get("daily_slots")
        if (summary.get("daily_scheduling_enabled")
                and daily_slots is not None and not daily_slots.empty):
            st.subheader("📅 Schedule")

            DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            DAY_FULL = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
                        "Thu": "Thursday", "Fri": "Friday"}
            emp_lookup = employees_df.set_index("id")[["role", "company"]]

            ctrl1, ctrl2 = st.columns([1, 2])
            with ctrl1:
                week_options = sorted(daily_slots["week"].unique())
                sel_week = st.selectbox(
                    "Week",
                    options=week_options,
                    index=0,
                    format_func=lambda w: f"Week {w}",
                    key="daily_week_sel",
                )
            tt_week = daily_slots[daily_slots["week"] == sel_week]

            with ctrl2:
                view_mode = st.radio(
                    "View",
                    options=["Whole team", "One employee"],
                    horizontal=True,
                    key="cal_view_mode",
                )

            def _hhmm_to_min(hhmm):
                h, m = hhmm.split(":")
                return int(h) * 60 + int(m)

            # A stable colour per site, so the same site reads the same across
            # the whole calendar.
            site_names = sorted(tt_week["site"].unique())
            palette = [
                "#2E75B6", "#548235", "#BF9000", "#9E480E", "#7030A0",
                "#1F7A7A", "#C00000", "#385723", "#806000", "#5B9BD5",
                "#A9569E", "#C55A11", "#2F5496", "#538135", "#833C00",
            ]
            site_colour = {s: palette[i % len(palette)] for i, s in enumerate(site_names)}

            # Calendar bounds — find the earliest start and latest end across
            # the week so the grid is tight around real working hours.
            all_starts = [_hhmm_to_min(s) for s in tt_week["start"]]
            all_ends = [_hhmm_to_min(e) for e in tt_week["end"]]
            day_start_min = min(all_starts) if all_starts else 8 * 60
            day_end_min = max(all_ends) if all_ends else 18 * 60
            # round to the hour for clean gridlines
            grid_start = (day_start_min // 60) * 60
            grid_end = -(-day_end_min // 60) * 60  # ceil to hour
            total_span = max(grid_end - grid_start, 60)
            PX_PER_MIN = 0.9  # vertical scale

            def _render_day_column(day_slots, show_site=True):
                """Render the absolutely-positioned blocks for one day."""
                blocks_html = ""
                for _, r in day_slots.iterrows():
                    s_min = _hhmm_to_min(r["start"])
                    e_min = _hhmm_to_min(r["end"])
                    top = (s_min - grid_start) * PX_PER_MIN
                    height = max((e_min - s_min) * PX_PER_MIN, 18)
                    colour = site_colour.get(r["site"], "#666")
                    label = r["site"] if show_site else f"#{r['employee_id']}"
                    sub = f"{r['start']}–{r['end']}"
                    blocks_html += (
                        f'<div style="position:absolute;top:{top:.0f}px;'
                        f'height:{height:.0f}px;left:3px;right:3px;'
                        f'background:{colour};border-radius:5px;color:#fff;'
                        f'padding:3px 6px;font-size:11px;overflow:hidden;'
                        f'box-shadow:0 1px 2px rgba(0,0,0,0.25);">'
                        f'<div style="font-weight:600;line-height:1.15;">{label}</div>'
                        f'<div style="opacity:.85;font-size:10px;">{sub}</div>'
                        f'</div>'
                    )
                return blocks_html

            def _render_calendar(slots_df, show_site=True):
                """Build a full Mon–Fri calendar as one HTML string."""
                axis_html = ""
                hour = grid_start
                while hour <= grid_end:
                    top = (hour - grid_start) * PX_PER_MIN
                    hh = hour // 60
                    axis_html += (
                        f'<div style="position:absolute;top:{top:.0f}px;right:6px;'
                        f'font-size:10px;color:#888;transform:translateY(-50%);">'
                        f'{hh:02d}:00</div>'
                        f'<div style="position:absolute;top:{top:.0f}px;left:54px;'
                        f'right:0;border-top:1px solid #ECECEC;"></div>'
                    )
                    hour += 60

                col_height = total_span * PX_PER_MIN
                cols_html = ""
                for day in DAYS:
                    dg = slots_df[slots_df["day"] == day]
                    n = len(dg)
                    header = (
                        f'<div style="text-align:center;font-weight:600;'
                        f'font-size:12px;padding:6px 0;color:#333;'
                        f'border-bottom:2px solid #ddd;">{DAY_FULL[day]}'
                        f'<span style="color:#aaa;font-weight:400;"> · {n}</span></div>'
                    )
                    body = (
                        f'<div style="position:relative;height:{col_height:.0f}px;'
                        f'border-left:1px solid #ECECEC;">'
                        f'{_render_day_column(dg, show_site)}</div>'
                    )
                    cols_html += (
                        f'<div style="flex:1;min-width:0;">{header}{body}</div>'
                    )

                axis_col = (
                    f'<div style="width:54px;position:relative;'
                    f'height:{col_height:.0f}px;margin-top:33px;">{axis_html}</div>'
                )

                return (
                    f'<div style="display:flex;background:#fff;border:1px solid #E0E0E0;'
                    f'border-radius:8px;padding:8px;overflow-x:auto;">'
                    f'{axis_col}{cols_html}</div>'
                )

            # ---- legend --------------------------------------------------
            legend = "".join(
                f'<span style="display:inline-flex;align-items:center;'
                f'margin:2px 10px 2px 0;font-size:11px;">'
                f'<span style="width:11px;height:11px;border-radius:3px;'
                f'background:{site_colour[s]};display:inline-block;'
                f'margin-right:5px;"></span>{s}</span>'
                for s in site_names
            )

            if view_mode == "One employee":
                emp_ids = sorted(tt_week["employee_id"].unique())
                sel_emp = st.selectbox(
                    "Employee",
                    options=emp_ids,
                    format_func=lambda e: f"#{e} · {emp_lookup['role'].get(e,'')}"
                                          f" ({emp_lookup['company'].get(e,'')})",
                    key="cal_emp_sel",
                )
                emp_slots = tt_week[tt_week["employee_id"] == sel_emp]
                wk_hours = emp_slots["hours"].sum()
                st.caption(
                    f"Employee #{sel_emp} — {emp_lookup['role'].get(sel_emp,'')}, "
                    f"{wk_hours:.1f}h scheduled in week {sel_week}"
                )
                st.markdown(_render_calendar(emp_slots, show_site=True),
                            unsafe_allow_html=True)
            else:
                st.caption(
                    f"Whole team, week {sel_week}. Each block is one shift; "
                    "blocks are coloured by site. Switch to 'One employee' for "
                    "a personal calendar."
                )
                st.markdown(f'<div style="margin:6px 0 10px;">{legend}</div>',
                            unsafe_allow_html=True)
                st.markdown(_render_calendar(tt_week, show_site=True),
                            unsafe_allow_html=True)

            # ---- summary stats + download -------------------------------
            st.write("")
            total_emp_days = tt_week.groupby(["employee_id", "day"]).ngroups
            people_scheduled = tt_week["employee_id"].nunique()
            cap1, cap2, cap3 = st.columns(3)
            cap1.metric("People scheduled", people_scheduled)
            cap2.metric("Employee-days", total_emp_days)
            cap3.metric("Shifts placed", len(tt_week))

            from io import BytesIO
            tbuf = BytesIO()
            with pd.ExcelWriter(tbuf, engine="openpyxl") as writer:
                tt_week.to_excel(writer, sheet_name=f"week_{sel_week}", index=False)
                # also write the grid form for spreadsheet users
                grid_rows = []
                for emp_id, g in tt_week.groupby("employee_id"):
                    row = {"employee_id": emp_id,
                           "role": emp_lookup["role"].get(emp_id, "")}
                    for day in DAYS:
                        dg = g[g["day"] == day]
                        row[day] = " · ".join(
                            f"{r['site']} {r['start']}-{r['end']}"
                            for _, r in dg.iterrows()
                        )
                    grid_rows.append(row)
                pd.DataFrame(grid_rows).to_excel(
                    writer, sheet_name="grid", index=False)
            st.download_button(
                f"⬇️  Download week {sel_week} schedule (Excel)",
                data=tbuf.getvalue(),
                file_name=f"combibrug_schedule_week{sel_week}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.divider()
        elif summary.get("daily_scheduling_enabled"):
            st.info(
                "No daily slots were placed. This usually means the demand "
                "template has no availability windows filled in — fill in "
                "available_mon..fri for the sites you want scheduled."
            )

        # "Project · Site (hours)". When they're the same (the common
        # default for one-site projects like most CC locations), the cell
        # just shows "Site (hours)" to avoid visual noise.
        st.subheader(f"Weekly schedule ({horizon} weeks)")
        st.caption(
            "Each row is one employee. Each column is one week. Cells show "
            "where the employee works that week. When a project spans multiple "
            "sites, the project name appears as a prefix. Blank means not "
            "assigned that week."
        )

        if assignments.empty:
            st.info("No internal assignments produced.")
        else:
            def _label(project, site, hrs):
                if project and project != site:
                    return f"{project} · {site} ({hrs}h)"
                return f"{site} ({hrs}h)"

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

            emp_lookup = employees_df.set_index("id")[["role", "company"]]
            schedule.insert(0, "role", emp_lookup["role"].reindex(schedule.index).values)
            schedule.insert(1, "company", emp_lookup["company"].reindex(schedule.index).values)

            total_hours = assignments.groupby("employee_id")["hours"].sum()
            schedule["__sort"] = schedule.index.map(total_hours).fillna(0)
            schedule = schedule.sort_values("__sort", ascending=False).drop(columns="__sort")

            st.dataframe(schedule, use_container_width=True, height=500)

        # Per-week cost chart
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

        # Detailed assignments table
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

        # Uncovered hours table: hours the model could not assign to any
        # employee (named or unnamed). These require last-resort freelancer
        # cover and are shown per site so the planner knows exactly where.
        site_to_project = dict(zip(plan_locations["site"], plan_locations["project"]))
        uncovered_by_site = summary.get("uncovered_hours_by_site", {})
        fl_rows = [
            {"project": site_to_project.get(s, s), "site": s,
             "uncovered_hours": round(h, 2)}
            for s, h in uncovered_by_site.items()
            if h > 0.01
        ]
        if fl_rows:
            st.subheader("⚠️ Uncovered hours (last-resort freelancer needed)")
            st.caption(
                "These hours could not be assigned to any employee in the template. "
                "They need to be covered by an external freelancer not yet listed. "
                "Adding more named freelancers (contract_type = freelancer) to the "
                "employee template will eliminate these gaps."
            )
            st.dataframe(
                pd.DataFrame(fl_rows).sort_values("uncovered_hours", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame([{
                "horizon_weeks": horizon,
                "status": status,
                "total_cost": round(summary.get("objective") or 0, 2),
                "internal_cost": round(summary.get("internal_cost") or 0, 2),
                "named_freelancer_cost": round(summary.get("named_freelancer_cost") or 0, 2),
                "named_freelancer_hours": round(summary.get("named_freelancer_hours") or 0, 1),
                "uncovered_hours": round(summary.get("total_uncovered_hours") or 0, 1),
                "n_switches": summary.get("n_switches", 0),
                "continuity_penalty_per_switch": continuity_penalty,
            }]).to_excel(writer, sheet_name="summary", index=False)

            if not assignments.empty:
                assignments.to_excel(writer, sheet_name="assignments", index=False)
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


with tab_diag:
    employees_df, locations_df = _require_data()
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


