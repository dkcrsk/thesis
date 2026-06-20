"""
app.py
------
Streamlit dashboard for the Combibrug workforce planning artifact.

Run from this directory with:

    streamlit run app.py

Only the TEMPLATE file formats are supported for employees and demand
(see data_loader.py). The historical wide-format files are no longer
parsed directly.

The app has four tabs:
    1. Upload files — upload employee + demand templates (+ optional rates)
    2. Inputs       — view loaded data, cost-source breakdown
    3. Run plan     — solve the MILP, see KPIs, assignments, underassignment
    4. Diagnostics  — validation checks and per-employee explanations
"""

import io
import os
import tempfile
from io import BytesIO

import pandas as pd
import streamlit as st

from data_loader import (
    load_employees, load_locations, load_rates,
    generate_employee_template_example,
    generate_demand_template_example,
    DataLoadError,
)
from model import (
    DEFAULT_FALLBACK_RATE,
    DEFAULT_STAGE_RATE,
    DEFAULT_UNDERASSIGNMENT_PENALTY,
    apply_hourly_costs,
    solve_allocation,
    validate_assignments,
    explain_assignment,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Combibrug Workforce Planner",
    page_icon="🧭",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar — cost & model settings
# ---------------------------------------------------------------------------

st.sidebar.title("Combibrug Workforce Planner")
st.sidebar.caption("Decision-support artifact — BSc Business Analytics thesis")

st.sidebar.markdown("### Cost assumptions")
rate_choice = st.sidebar.radio(
    "Hourly rate to minimise against (rates file)",
    options=["rate_incl", "rate_base"],
    format_func=lambda x: {
        "rate_incl": "Including employer costs (recommended)",
        "rate_base": "Base salary only",
    }[x],
    help=(
        "Only used if an employee has no hourly_rate in the employee "
        "template and IS found in the optional rates file."
    ),
)
fallback_rate = st.sidebar.number_input(
    "Fallback rate (EUR/h)",
    value=DEFAULT_FALLBACK_RATE, step=1.0,
    help="Used when an employee has no hourly_rate and is not in the rates file.",
)
stage_rate = st.sidebar.number_input(
    "Stagiair rate (EUR/h)",
    value=DEFAULT_STAGE_RATE, step=1.0,
)

st.sidebar.markdown("### Model options")
underassignment_penalty = st.sidebar.number_input(
    "Underassignment penalty (EUR/h)",
    value=DEFAULT_UNDERASSIGNMENT_PENALTY, step=100.0,
    help=(
        "Penalty for every hour a permanent employee is NOT assigned. "
        "Keep this much higher than any wage so the solver fills "
        "permanent capacity before using freelancers."
    ),
)
enforce_headcount = st.sidebar.checkbox(
    "Enforce per-role headcount", value=True,
    help="When ticked, locations with req_<role> columns get a minimum headcount per role.",
)
time_limit = st.sidebar.slider("Solver time limit (s)", 10, 300, 120, 10)


# ---------------------------------------------------------------------------
# Tabs
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
        "Upload the employee template and the demand template below. "
        "The rates file is optional — without it, any employee missing an "
        "`hourly_rate` value gets the fallback rate set in the sidebar."
    )

    st.subheader("1 — Employees (required)")
    st.caption(
        "Columns: id, role, company, contract_type, contract_hours "
        "(optional: is_freelancer, available, hourly_rate, name, notes)"
    )
    emp_upload = st.file_uploader(
        "Choose employee template file",
        type=["xlsx", "csv"],
        key="emp_uploader",
        label_visibility="collapsed",
    )

    def _example_bytes(generator_fn):
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        try:
            generator_fn(tmp.name)
            with open(tmp.name, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp.name)

    st.download_button(
        "⬇️ Download example employee template",
        data=_example_bytes(generate_employee_template_example),
        file_name="employee_template_example.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Includes permanent staff, stagiaires, freelancers, and a vacation example.",
    )

    st.divider()

    st.subheader("2 — Demand / locations (required)")
    st.caption(
        "Columns: location, company, weekly_hours "
        "(optional: duration_months, division, req_<role>)"
    )
    loc_upload = st.file_uploader(
        "Choose demand template file",
        type=["xlsx", "csv"],
        key="loc_uploader",
        label_visibility="collapsed",
    )

    st.download_button(
        "⬇️ Download example demand template",
        data=_example_bytes(generate_demand_template_example),
        file_name="demand_template_example.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()

    st.subheader("3 — Rates file (optional)")
    st.caption(
        "Columns: id, rate_base, rate_incl (optional: is_stage). "
        "Only used as a fallback for employees without a direct hourly_rate."
    )
    rates_upload = st.file_uploader(
        "Choose rates file",
        type=["xlsx", "csv"],
        key="rates_uploader",
        label_visibility="collapsed",
    )

    st.divider()

    # --- Load the data when both required files are present
    if emp_upload is None or loc_upload is None:
        missing = []
        if emp_upload is None:
            missing.append("employee template")
        if loc_upload is None:
            missing.append("demand template")
        st.info(f"Waiting for: {', '.join(missing)}. Upload the files above to continue.")
        st.stop()

    @st.cache_data(show_spinner="Reading and cleaning files…")
    def _load_uploaded(emp_bytes, emp_name, loc_bytes, loc_name,
                       rates_bytes, rates_name):
        """Cache key is the raw bytes, so re-uploading the same file hits cache."""
        emp_buf = io.BytesIO(emp_bytes); emp_buf.name = emp_name
        loc_buf = io.BytesIO(loc_bytes); loc_buf.name = loc_name
        rates_buf = None
        if rates_bytes:
            rates_buf = io.BytesIO(rates_bytes)
            rates_buf.name = rates_name

        employees = load_employees(emp_buf)
        locations = load_locations(loc_buf)
        rates = load_rates(rates_buf) if rates_buf else None
        return employees, locations, rates

    try:
        employees_raw, locations_df, rates_df = _load_uploaded(
            emp_upload.read(), emp_upload.name,
            loc_upload.read(), loc_upload.name,
            rates_upload.read() if rates_upload else b"",
            rates_upload.name if rates_upload else "",
        )
    except DataLoadError as e:
        st.error(f"❌ Could not load the files.\n\n{e}")
        st.info(
            "Check that the employee template has columns id, role, company, "
            "contract_type, contract_hours, and that the demand template has "
            "location, company, weekly_hours."
        )
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while reading files: {e}")
        st.stop()

    employees_df, missing_rate_ids = apply_hourly_costs(
        employees_raw,
        rates_df=rates_df,
        rate_column=rate_choice,
        fallback_rate=fallback_rate,
        stage_rate=stage_rate,
    )

    # Store in session state so all other tabs can access
    st.session_state["employees_df"] = employees_df
    st.session_state["locations_df"] = locations_df
    st.session_state["missing_rate_ids"] = missing_rate_ids

    st.success("✅ Files loaded successfully.")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Employees", len(employees_df))
    s2.metric("  — of which freelancers", int(employees_df["is_freelancer"].sum()))
    s3.metric("Project locations", len(locations_df))
    s4.metric("Total weekly demand", f"{locations_df['weekly_hours'].sum():.0f} h")
    st.caption("Switch to the Inputs tab to inspect the cleaned data, or go straight to Run plan.")


# ---------------------------------------------------------------------------
# Guard: all other tabs need the uploaded data
# ---------------------------------------------------------------------------

def _require_data():
    """Return (employees_df, locations_df, missing_rate_ids) from session
    state, or stop with a friendly message if files haven't been uploaded."""
    if "employees_df" not in st.session_state:
        st.info("👆 Go to the **Upload files** tab first and upload your data.")
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
        "These are the cleaned employee and location tables that feed the "
        "optimisation model. Edit the cost assumptions in the sidebar to "
        "update the per-employee hourly cost shown below, then re-upload "
        "if you change the underlying files."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Employees", len(employees_df))
    col2.metric("Freelancers", int(employees_df["is_freelancer"].sum()))
    col3.metric("Unavailable this week", int((~employees_df["available"]).sum()))
    col4.metric("Project locations", len(locations_df))

    st.subheader("Employees")

    src_counts = employees_df["cost_source"].value_counts().to_dict()
    n_template = src_counts.get("employee_template", 0)
    n_from_rates = src_counts.get("rate_incl", 0) + src_counts.get("rate_base", 0)
    n_stage = src_counts.get("stage", 0)
    n_missing = src_counts.get("missing", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rate from template", n_template)
    c2.metric("Rate from rates file", n_from_rates)
    c3.metric("Stagiair flat rate", n_stage)
    c4.metric("Missing — fallback used", n_missing)

    if missing_rate_ids:
        st.warning(
            f"⚠️ {len(missing_rate_ids)} employee(s) have no hourly_rate and "
            f"are not in the rates file. Using the fallback rate of "
            f"€{fallback_rate:.2f}/h: {sorted(missing_rate_ids)}"
        )

    st.dataframe(employees_df, use_container_width=True, height=300)

    st.subheader("Project locations")
    display_cols = ["location", "division", "company", "weekly_hours", "duration_months"]
    st.dataframe(locations_df[display_cols], use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Tab 2 — Run plan
# ---------------------------------------------------------------------------

def _run_and_display(employees, locations, penalty, enforce_hc, t_limit):
    """Helper: solve once and return (status, assignments, summary)."""
    with st.spinner("Solving…"):
        status, assignments, summary = solve_allocation(
            employees, locations,
            underassignment_penalty=penalty,
            enforce_headcount=enforce_hc,
            time_limit=t_limit,
        )
    return status, assignments, summary


with tab_run:
    employees_df, locations_df, _ = _require_data()
    st.header("Generate a workforce plan")
    st.write(
        "Click **Solve** to run the MILP with the current cost assumptions "
        "and solver options from the sidebar."
    )

    if st.button("Solve", type="primary"):
        status, assignments, summary = _run_and_display(
            employees_df, locations_df,
            underassignment_penalty, enforce_headcount, time_limit,
        )
        st.session_state["last_status"] = status
        st.session_state["last_assignments"] = assignments
        st.session_state["last_summary"] = summary

    if "last_status" in st.session_state:
        status = st.session_state["last_status"]
        assignments = st.session_state["last_assignments"]
        summary = st.session_state["last_summary"]

        if summary.get("error"):
            st.error(f"⚠️ {summary['error']}")
        elif status != "Optimal":
            st.error(f"Solver status: **{status}** — see the Diagnostics tab for details.")
        else:
            st.success(f"Solver status: **{status}**")

        if summary.get("objective") is not None:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Wage cost total", f"€{summary['wage_cost_total']:,.0f}")
            k2.metric("Permanent", f"€{summary['permanent_cost']:,.0f}")
            k3.metric("Freelancer", f"€{summary['freelancer_cost']:,.0f}")
            k4.metric(
                "Underassigned",
                f"{summary['total_underassigned_hours']:.0f} h",
                f"{summary['n_underassigned_employees']} people",
            )
            k5.metric("Freelancer hours", f"{summary['freelancer_hours']:.0f} h")

            st.subheader("Assignments")
            if not assignments.empty:
                st.dataframe(
                    assignments.sort_values(["location", "employee_id"]),
                    use_container_width=True,
                    height=400,
                )
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    assignments.to_excel(writer, sheet_name="assignments", index=False)
                    summary_export = {
                        k: v for k, v in summary.items()
                        if k not in ("underassignment", "unavailable_employee_ids", "error")
                    }
                    pd.DataFrame(
                        [{"metric": k, "value": v} for k, v in summary_export.items()]
                    ).to_excel(writer, sheet_name="summary", index=False)
                    underassignment = summary.get("underassignment")
                    if underassignment is not None and not underassignment.empty:
                        underassignment.to_excel(writer, sheet_name="underassignment", index=False)
                st.download_button(
                    "Download plan as Excel",
                    data=buf.getvalue(),
                    file_name="combibrug_plan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.info("No assignments were produced.")

            st.subheader("Underassigned permanent staff")
            under_df = summary.get("underassignment")
            if under_df is None or under_df.empty:
                st.success("Every permanent employee is fully assigned this week.")
            else:
                st.info(
                    f"{len(under_df)} permanent employee(s) have unassigned hours. "
                    "These hours should be picked up in subsequent planning cycles."
                )
                st.dataframe(under_df, use_container_width=True, height=250)
    else:
        st.info("No plan has been generated yet. Press **Solve** to run the model.")


# ---------------------------------------------------------------------------
# Tab 3 — Diagnostics
# ---------------------------------------------------------------------------

with tab_diag:
    employees_df, locations_df, _ = _require_data()
    st.header("Diagnostics")
    st.write(
        "Validation checks and per-employee explanations for the most "
        "recent plan generated in the **Run plan** tab."
    )

    if "last_status" not in st.session_state:
        st.info("Generate a plan first in the **Run plan** tab.")
    else:
        assignments = st.session_state["last_assignments"]
        summary = st.session_state["last_summary"]
        status = st.session_state["last_status"]

        st.subheader("Validation")
        checks = validate_assignments(employees_df, locations_df, assignments, summary)
        for c in checks:
            if c.startswith("OK"):
                st.success(c)
            elif c.startswith("INFO"):
                st.info(c)
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
                f"{employees_df.loc[employees_df['id']==i, 'contract_hours'].iloc[0]:.0f}h"
                f"{', freelancer' if bool(employees_df.loc[employees_df['id']==i, 'is_freelancer'].iloc[0]) else ''})"
            ),
        )
        expl = explain_assignment(emp_id, employees_df, locations_df, assignments)
        if expl.empty:
            st.info("No eligible locations for this employee.")
        else:
            st.dataframe(expl, use_container_width=True, height=300)

        st.subheader("Status")
        if summary.get("error"):
            st.error(summary["error"])
        elif status == "Infeasible":
            st.error(
                "The model is infeasible — no plan exists under the current "
                "constraints. Common causes: "
                "(1) total weekly demand exceeds the combined capacity of all "
                "eligible employees and freelancers for a company; "
                "(2) a location's headcount requirement cannot be met by any "
                "eligible employee — try adding a freelancer with that role; "
                "(3) every eligible employee for a location is marked unavailable."
            )
        else:
            st.info(f"Last solver status: **{status}**")
