"""
data_loader.py
--------------
Reads the two Combibrug planning templates and returns two cleaned DataFrames:
    - employees_df: one row per employee (role, company, contract hours, cost)
    - locations_df: one row per project site with weekly demand and availability

The artefact works with templates only. The raw staff register and the
historical locations file are NOT read here: they were used during Data
Understanding to learn the structure of the data and to build the first
version of the model, and that understanding was turned into the two clean
Excel templates this module loads. The MILP and the UI never touch any raw
Excel file directly — they only consume the cleaned DataFrames returned below.
"""

import pandas as pd


# The eight standard role categories used across the project. Kept here because
# the model (model.py) imports it to build the per-role demand columns.
ALL_ROLES = [
    "Combicoach", "Stagiair", "Jongerenbegeleider", "Buurtsportcoach",
    "Manager", "Cultuurcoordinator", "Leefstijlcoach", "Other",
]


class DataLoadError(Exception):
    """Raised when a template cannot be parsed.

    Use a clear, planner-friendly message — these are surfaced directly
    in the Streamlit UI.
    """
    pass


# ------------------------------------------------------------------
# Demand (site) template
# ------------------------------------------------------------------

DEMAND_TEMPLATE_COLUMNS = {
    "site":           "Name of the specific site/location where work happens",
    "company":        "Which company runs it: 'Combibrug' or 'CC'",
    "weekly_hours":   "Expected hours of work needed per week (decimal)",
    "project":            "Project the site belongs to (default = same as site name)",
    "duration_months":    "How many months the project runs (default = full horizon)",
    "start_week":         "Which week of the horizon the project starts in, 1-indexed (default 1)",
    "min_headcount":      "Minimum number of distinct people required per week (default 0, i.e. no requirement). Use 2 for Combiworld/MDT (Four-Eyes rule).",
    "project_leader_id":  "Employee ID of a designated project leader who must work at this site every active week (default: no fixed leader).",
    "available_mon":      "Time windows the site is available on Monday, e.g. '08:00-16:00' or '08:00-12:00; 13:00-17:00'. Blank = closed.",
    "available_tue":      "Time windows the site is available on Tuesday.",
    "available_wed":      "Time windows the site is available on Wednesday.",
    "available_thu":      "Time windows the site is available on Thursday.",
    "available_fri":      "Time windows the site is available on Friday.",
    "notes":              "Free-text notes — not used by the model",
}

DAY_COLS = ["available_mon", "available_tue", "available_wed",
            "available_thu", "available_fri"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _parse_windows(cell):
    """Parse a day-availability cell into a list of (start_h, end_h) tuples
    in decimal hours.

    Examples:
        "08:00-16:00"               -> [(8.0, 16.0)]
        "08:00-12:00; 13:00-17:00"  -> [(8.0, 12.0), (13.0, 17.0)]
        "" or NaN                   -> []

    The parser tolerates spaces, en-dashes vs hyphens, and both ':' and '.'
    as the H/M separator. Invalid pieces are skipped silently.
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return []

    s = s.replace("\u2013", "-").replace("\u2014", "-")
    out = []
    for piece in s.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        if "-" not in piece:
            continue
        try:
            a, b = piece.split("-", 1)

            def _to_h(t):
                t = t.strip().replace(".", ":")
                if ":" not in t:
                    return float(t)
                hh, mm = t.split(":", 1)
                return int(hh) + int(mm) / 60.0

            start_h = _to_h(a)
            end_h = _to_h(b)
            if end_h > start_h and 0 <= start_h < 24 and 0 < end_h <= 24:
                out.append((start_h, end_h))
        except (ValueError, IndexError):
            continue
    return out


def load_demand_template(path_or_buffer):
    """Load a forward-looking demand template file.

    Accepts either a file path (str / Path) or a file-like object
    (e.g. a Streamlit UploadedFile). Supports both .xlsx and .csv.

    The template has one row per project location and only needs a
    handful of columns — see DEMAND_TEMPLATE_COLUMNS for the full list.

    Returns a cleaned locations DataFrame ready for the optimisation model.
    """
    try:
        name = getattr(path_or_buffer, "name", str(path_or_buffer))
        if str(name).lower().endswith(".csv"):
            df = pd.read_csv(path_or_buffer)
        else:
            df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise DataLoadError(f"Could not read the demand template: {e}")

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "site" not in df.columns and "location" in df.columns:
        df = df.rename(columns={"location": "site"})

    required = ["site", "company", "weekly_hours"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise DataLoadError(
            f"The demand template is missing required columns: {missing_cols}. "
            f"Columns found: {list(df.columns)}. "
            f"Required: {required} (the old name 'location' is also accepted for 'site')."
        )

    df = df.dropna(subset=["site", "weekly_hours"])
    df["site"] = df["site"].astype(str).str.strip()
    df["weekly_hours"] = pd.to_numeric(df["weekly_hours"], errors="coerce").fillna(0)
    df = df[df["weekly_hours"] > 0].copy()

    if df.empty:
        raise DataLoadError(
            "The demand template has no valid rows. "
            "Check that 'weekly_hours' contains numbers greater than 0."
        )

    if "duration_months" not in df.columns:
        df["duration_months"] = 12
    else:
        df["duration_months"] = pd.to_numeric(
            df["duration_months"], errors="coerce"
        ).fillna(12).astype(int)

    df["duration_weeks"] = (df["duration_months"] * 52.0 / 12.0).round().astype(int)

    if "start_week" not in df.columns:
        df["start_week"] = 1
    else:
        df["start_week"] = (pd.to_numeric(df["start_week"], errors="coerce")
                            .fillna(1).astype(int).clip(lower=1))

    if "project" not in df.columns:
        df["project"] = df["site"]
    else:
        df["project"] = df["project"].astype(str).str.strip()
        df.loc[df["project"].isin(["", "nan", "None"]), "project"] = (
            df.loc[df["project"].isin(["", "nan", "None"]), "site"]
        )

    if "min_headcount" not in df.columns:
        df["min_headcount"] = 0
    else:
        df["min_headcount"] = (pd.to_numeric(df["min_headcount"], errors="coerce")
                               .fillna(0).astype(int).clip(lower=0))

    if "project_leader_id" not in df.columns:
        df["project_leader_id"] = pd.NA
    else:
        df["project_leader_id"] = pd.to_numeric(
            df["project_leader_id"], errors="coerce"
        ).astype("Int64")

    def _normalise_company(v):
        s = str(v).strip().lower()
        if "cc" in s or "combicoach" in s:
            return "CC"
        return "Combibrug"

    df["company"] = df["company"].apply(_normalise_company)

    for r in ALL_ROLES:
        df[f"req_{r}"] = 0

    df["duration_source"] = "template"

    parsed_windows = []
    totals = []
    for _, row in df.iterrows():
        site_windows = []
        for col, day in zip(DAY_COLS, DAY_NAMES):
            if col in df.columns:
                for (s_h, e_h) in _parse_windows(row.get(col)):
                    site_windows.append((day, s_h, e_h))
        parsed_windows.append(site_windows)
        totals.append(round(sum(e - s for (_, s, e) in site_windows), 2))
    df["windows"] = parsed_windows
    df["total_window_hours"] = totals

    out_cols = [
        "site", "project", "company",
        "weekly_hours", "duration_months", "duration_weeks", "start_week",
        "min_headcount", "project_leader_id",
        "duration_source",
        "windows", "total_window_hours",
    ]
    for c in DAY_COLS:
        if c in df.columns:
            out_cols.append(c)
    out_cols += [f"req_{r}" for r in ALL_ROLES]

    return df[out_cols].reset_index(drop=True)


# ------------------------------------------------------------------
# Employee template
# ------------------------------------------------------------------

EMPLOYEE_TEMPLATE_COLUMNS = {
    "id":             "Employee number (from the staff system)",
    "role":           "Job role (Combicoach, Stagiair, Manager, ...)",
    "company":        "Combibrug or CC",
    "contract_hours": "Contracted hours per week",
    "hourly_cost":    "Cost to Combibrug per hour, \u20ac (including employer overhead)",
    "contract_type":  "permanent / fixed_term / stage / oproep (informational)",
    "end_week":       "Last week the employee is available (blank = whole horizon)",
    "is_dreammaker":  "TRUE if this employee is a Dreammaker (Combiworld/MDT), else blank",
}


def load_employee_template(path_or_buffer):
    """Load the employee template (one combined file with rates).

    Accepts either a file path (str / Path) or a file-like object
    (e.g. a Streamlit UploadedFile). Returns a cleaned employees DataFrame
    ready for the optimisation model.
    """
    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise DataLoadError(f"Could not open the employee template: {e}")

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    df = df[pd.to_numeric(df["id"], errors="coerce").notna()].copy()

    required = ["id", "role", "company", "contract_hours", "hourly_cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"The employee template is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df["contract_hours"] = pd.to_numeric(df["contract_hours"], errors="coerce").fillna(0).astype(float)
    df["hourly_cost"] = pd.to_numeric(df["hourly_cost"], errors="coerce")

    df["company"] = df["company"].apply(
        lambda v: "CC" if ("cc" in str(v).lower() or "combicoach" in str(v).lower()) else "Combibrug"
    )

    if "contract_type" not in df.columns:
        df["contract_type"] = "unknown"

    # end_week: optional last available week
    if "end_week" not in df.columns:
        df["end_week"] = pd.NA
    else:
        df["end_week"] = pd.to_numeric(df["end_week"], errors="coerce").astype("Int64")

    # is_dreammaker: optional boolean
    if "is_dreammaker" not in df.columns:
        df["is_dreammaker"] = False
    else:
        df["is_dreammaker"] = df["is_dreammaker"].apply(
            lambda v: str(v).strip().lower() in ("true", "1", "yes", "ja", "x")
        )

    df = df[df["contract_hours"] > 0].drop_duplicates(subset=["id"]).reset_index(drop=True)

    missing_rate = df["hourly_cost"].isna()
    df["cost_source"] = "template"
    df.loc[missing_rate, "cost_source"] = "missing"

    out_cols = ["id", "role", "company", "contract_hours", "hourly_cost",
                "contract_type", "end_week", "is_dreammaker", "cost_source"]
    return df[out_cols]
