"""
conditions_registry.py
----------------------

A small, CSV-driven registry of constraints for your OR-Tools hospital scheduler.

How it is used from the main script (scheduler.py)
---------------------------------------------------
After you construct the model and decision variables x[(resident, day, shift)],
call:

    from conditions_registry import load_conditions_from_csv, apply_conditions

    extra_conds = load_conditions_from_csv(DATA_DIR)  # reads data/conditions.csv
    apply_conditions(
        model=model,
        x=x,
        residents=residents,
        days=days,
        shifts=shifts,
        date_map=date_map,
        required_staff=required_staff,
        conditions=extra_conds,
        context={
            "unavailable": unavailable,       # dict[str, list[str]] as built in your script
            "night_shifts": night_shifts,     # dict[str, list[str]] as built in your script
        }
    )

CSV format (data/conditions.csv)
--------------------------------
Columns: key, enabled, params
- key: the constraint identifier (see _REGISTRY below)
- enabled: 1/0 or true/false (case-insensitive)
- params: JSON object with parameters for the constraint (may be "{}")

Example rows:
    coverage_exact_required,1,"{}"
    one_shift_per_day,1,"{}"
    no_afternoon_then_morning,1,"{}"
    enforce_unavailability,1,"{}"
    night_shift_exclusion,1,"{}"
    fairness_strict_minmax,1,"{}"
    max_shifts_per_week,1,"{""limit"":5}"
    max_consecutive_work_days,1,"{""limit"":5}"
    min_free_days_per_week,0,"{""min_free"":1}"
    forbid_resident_shift,0,"{""resident"":""Res0"",""shift"":""afternoon""}"
    max_afternoon_per_week,0,"{""limit"":2}"
    max_total_shifts,0,"{""limit"":20}"

"""

import os
import json
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# CSV LOADING
# ---------------------------------------------------------------------------

def load_conditions_from_csv(data_dir: str, filename: str = "conditions.csv"):
    """
    Read enabled constraints from a CSV file.

    If the CSV also contains settings (like settings.csv), we keep only rows where type == "constraint".
    Returns a list like [{"key": "max_shifts_per_week", "params": {...}}, ...]
    for rows with enabled in {1, "1", "true", "yes"} (case-insensitive).
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"No {filename} found — skipping extra conditions.")
        return []

    # More tolerant CSV parsing (handles spaces after commas and quoted JSON)
    df = pd.read_csv(path, engine="python", skipinitialspace=True)


    # Keep only rows where type == "constraint" (case-insensitive).
    # This lets settings.csv hold both settings and constraints in one file.
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "constraint"]

    required_cols = {"key", "enabled", "params"}
    if not required_cols.issubset(df.columns):
        print(f"{filename} must contain columns: key, enabled, params. Skipping.")
        return []

    # Enabled mask: accept 1/true/yes
    enabled_mask = df["enabled"].astype(str).str.lower().isin(["1", "true", "yes"])
    rows = df[enabled_mask]

    out = []
    for _, row in rows.iterrows():
        key = str(row["key"]).strip()
        raw = row.get("params", "{}")
        params = {}
        if isinstance(raw, str) and raw.strip():
            try:
                params = json.loads(raw)
            except Exception as e:
                print(f"Warning: could not parse params for key={key}: {e}. Using empty params.")
                params = {}
        out.append({"key": key, "params": params})
    if out:
        print(f"Loaded {len(out)} extra condition(s) from {filename}.")
    return out


# ---------------------------------------------------------------------------
# REGISTRY DISPATCHER
# ---------------------------------------------------------------------------

def apply_conditions(
    model,
    x,
    residents,
    days,
    shifts,
    date_map,
    required_staff,
    conditions,
    context=None,
):
    """
    Look up each condition key in the registry and apply it.

    Parameters
    ----------
    model : cp_model.CpModel
        The CP-SAT model instance.
    x : dict[(str,str,str) -> IntVar]
        Decision variables x[(resident, day, shift)] in {0,1}.
    residents : list[str]
        Resident identifiers.
    days : list[str]
        Date labels in 'YYYY-MM-DD'.
    shifts : list[str]
        Shift names (e.g. ["morning", "afternoon"]).
    date_map : dict[str -> datetime]
        Map from date label to datetime object.
    required_staff : dict[str -> int]
        Required coverage per shift name.
    conditions : list[dict]
        Output of load_conditions_from_csv().
    context : dict or None
        Optional data used by some constraints, expected keys:
            - "unavailable": dict[str, list[str]]
            - "night_shifts": dict[str, list[str]]

    Notes
    -----
    - Unknown keys are ignored with a warning.
    - Constraints are added directly to 'model' via cp_model.Add().
    """
    context = context or {}
    for cond in conditions:
        key = cond.get("key")
        params = cond.get("params", {})
        fn = _REGISTRY.get(key)
        if fn is None:
            print(f"Unknown condition key: {key} — ignoring.")
            continue
        print(f"Applying condition: {key}  params={params}")
        fn(model, x, residents, days, shifts, date_map, required_staff, params, context)


# ---------------------------------------------------------------------------
# ORIGINAL CONSTRAINTS (from your scheduler.py)
# Each implemented as a function so they can be toggled from CSV.
# ---------------------------------------------------------------------------

def cond_coverage_exact_required(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    For every (day, shift), the number of assigned residents equals required_staff[shift].
    This mirrors your 'Constraint 1' in scheduler.py.
    """
    for d in days:
        for s in shifts:
            model.Add(sum(x[(r, d, s)] for r in residents) == int(required_staff[s]))

def cond_one_shift_per_day(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Each resident can work at most one shift per day.
    This mirrors your 'Constraint 2' in scheduler.py.
    """
    for r in residents:
        for d in days:
            model.Add(sum(x[(r, d, s)] for s in shifts) <= 1)

def cond_no_afternoon_then_morning(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Rest rule: forbid Afternoon on day d followed by Morning on day d+1 for the same resident.
    This mirrors your 'Constraint 3' in scheduler.py.
    """
    if "afternoon" in shifts and "morning" in shifts:
        for r in residents:
            for i in range(len(days) - 1):
                d1, d2 = days[i], days[i + 1]
                model.Add(x[(r, d1, "afternoon")] + x[(r, d2, "morning")] <= 1)

def cond_enforce_unavailability(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Forbid any assignment on a resident's unavailable dates.
    This mirrors your 'Constraint 4' in scheduler.py.
    Requires context['unavailable'] = dict[resident -> list of dates].
    """
    unavailable = context.get("unavailable", {})
    if not unavailable:
        print("cond_enforce_unavailability: no 'unavailable' in context; skipping.")
        return
    for r, date_list in unavailable.items():
        for d in date_list:
            if d in days:
                for s in shifts:
                    model.Add(x[(r, d, s)] == 0)

def cond_night_shift_exclusion(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Forbid any day shift on the night shift day and the following day.
    This mirrors your 'Constraint 5' in scheduler.py.
    Requires context['night_shifts'] = dict[resident -> list of dates for night].
    """
    night_shifts = context.get("night_shifts", {})
    if not night_shifts:
        print("cond_night_shift_exclusion: no 'night_shifts' in context; skipping.")
        return
    for r, date_list in night_shifts.items():
        for nd in date_list:
            if nd not in date_map:
                continue
            # Exclude all shifts on the night day
            for s in shifts:
                model.Add(x[(r, nd, s)] == 0)
            # Exclude all shifts on the next day, if present
            next_day = (date_map[nd] + (date_map[nd] - date_map[nd].replace(hour=0, minute=0, second=0, microsecond=0))).strftime("%Y-%m-%d")
            # The above trick is overkill; simpler is:
            next_day = (date_map[nd] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if next_day in days:
                for s in shifts:
                    model.Add(x[(r, next_day, s)] == 0)

def cond_fairness_strict_minmax(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Enforce strict fairness: total shifts per resident must be equal or differ by at most 1.
    This mirrors your 'Constraint 6' in scheduler.py.

    Implementation notes:
    - We compute total_required_shifts = len(days) * sum(required_staff.values()).
    - Let avg = total_required_shifts // len(residents).
      If there is a remainder, some residents will have (avg + 1).
    - We create per-resident totals and constrain them within [min_shifts, max_shifts].
    - Uses new variables named 'total_fair_<resident>' to avoid name clashes with your main.
    """
    total_required_shifts = len(days) * sum(int(required_staff[s]) for s in shifts)
    if len(residents) == 0:
        return
    avg = total_required_shifts // len(residents)
    remainder = total_required_shifts % len(residents)
    min_shifts = avg
    max_shifts = avg + (1 if remainder > 0 else 0)

    # Build resident totals and bound them
    for r in residents:
        # Variable bounds [min_shifts, max_shifts] match your original behavior
        total_var = model.NewIntVar(min_shifts, max_shifts, f"total_fair_{r}")
        # Link the variable to actual assignments
        model.Add(total_var == sum(x[(r, d, s)] for d in days for s in shifts))
        # Bounds already encoded in total_var so no extra Add() needed


# ---------------------------------------------------------------------------
# NEW OPTIONAL CONSTRAINTS (commented thoroughly)
# ---------------------------------------------------------------------------

def cond_max_total_shifts(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Hard cap on total shifts in the entire planning horizon, per resident.
    params: {"limit": int}
    Example: {"limit": 20}
    """
    limit = int(params.get("limit", 20))
    for r in residents:
        model.Add(sum(x[(r, d, s)] for d in days for s in shifts) <= limit)

def cond_max_shifts_per_week(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    At most 'limit' shifts per ISO week, for each resident.
    params: {"limit": int}
    Example: {"limit": 5}
    """
    limit = int(params.get("limit", 5))
    week_to_days = _group_days_by_iso_week(days, date_map)
    for r in residents:
        for wk, wk_days in week_to_days.items():
            model.Add(sum(x[(r, d, s)] for d in wk_days for s in shifts) <= limit)

def cond_max_afternoon_per_week(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Cap the number of 'afternoon' shifts per ISO week, per resident.
    params: {"limit": int}
    Example: {"limit": 2}
    """
    if "afternoon" not in shifts:
        return
    limit = int(params.get("limit", 2))
    week_to_days = _group_days_by_iso_week(days, date_map)
    for r in residents:
        for wk, wk_days in week_to_days.items():
            model.Add(sum(x[(r, d, "afternoon")] for d in wk_days) <= limit)

def cond_forbid_resident_shift(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Resident R is never assigned to shift S.
    params: {"resident": "Res0", "shift": "afternoon"}
    """
    r = params.get("resident")
    s = params.get("shift")
    if r not in residents or s not in shifts:
        print(f"forbid_resident_shift: invalid resident or shift: {r}, {s}. Ignoring.")
        return
    for d in days:
        model.Add(x[(r, d, s)] == 0)

def cond_forbid_resident_dates(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Resident R cannot work any shift on the provided list of dates.
    params: {"resident": "Res1", "dates": ["2025-07-08", "2025-07-12"]}
    This is similar to 'enforce_unavailability' but scoped to a single resident via params.
    """
    r = params.get("resident")
    date_list = params.get("dates", [])
    if r not in residents:
        print(f"forbid_resident_dates: invalid resident: {r}. Ignoring.")
        return
    for d in date_list:
        if d in days:
            for s in shifts:
                model.Add(x[(r, d, s)] == 0)

def cond_max_consecutive_work_days(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Forbid more than 'limit' consecutive working days for each resident.

    Definition of "working day" here:
    - A day counts as "worked" if the resident has any shift on that day.
      Your main already enforces at most one shift per day per resident.

    Implementation:
    - Use a sliding window of length (limit + 1).
    - Over any (limit + 1)-day window, the resident can work at most 'limit' days.

    params: {"limit": int}
    Example: {"limit": 5}
    """
    limit = int(params.get("limit", 5))
    window = limit + 1
    if window <= 1:
        return
    for r in residents:
        for i in range(len(days) - window + 1):
            window_days = days[i:i + window]
            model.Add(sum(x[(r, d, s)] for d in window_days for s in shifts) <= limit)

def cond_min_free_days_per_week(model, x, residents, days, shifts, date_map, required_staff, params, context):
    """
    Ensure at least 'min_free' days off per ISO week for each resident.

    Implementation:
    - For each week, the number of worked days <= len(week) - min_free.

    params: {"min_free": int}
    Example: {"min_free": 1}
    """
    min_free = int(params.get("min_free", 1))
    week_to_days = _group_days_by_iso_week(days, date_map)
    for r in residents:
        for wk, wk_days in week_to_days.items():
            # Count of worked days in the week <= total days in the week - required free days
            model.Add(sum(x[(r, d, s)] for d in wk_days for s in shifts) <= max(0, len(wk_days) - min_free))


# ---------------------------------------------------------------------------
# HELPERS AND REGISTRY MAP
# ---------------------------------------------------------------------------

def _group_days_by_iso_week(days, date_map):
    """
    Group date labels by ISO week number using date_map for conversion to datetime.
    Returns a dict: week_number -> list[date_label].
    """
    by_week = defaultdict(list)
    for d in days:
        wk = date_map[d].isocalendar().week
        by_week[wk].append(d)
    return by_week


# Registry mapping: CSV 'key' -> implementation function.
# Include both original constraints and new optional ones.
_REGISTRY = {
    # Original constraints (mirror your scheduler.py):
    "coverage_exact_required":      cond_coverage_exact_required,
    "one_shift_per_day":            cond_one_shift_per_day,
    "no_afternoon_then_morning":    cond_no_afternoon_then_morning,
    "enforce_unavailability":       cond_enforce_unavailability,
    "night_shift_exclusion":        cond_night_shift_exclusion,
    "fairness_strict_minmax":       cond_fairness_strict_minmax,

    # New optional constraints:
    "max_total_shifts":             cond_max_total_shifts,
    "max_shifts_per_week":          cond_max_shifts_per_week,
    "max_afternoon_per_week":       cond_max_afternoon_per_week,
    "forbid_resident_shift":        cond_forbid_resident_shift,
    "forbid_resident_dates":        cond_forbid_resident_dates,
    "max_consecutive_work_days":    cond_max_consecutive_work_days,
    "min_free_days_per_week":       cond_min_free_days_per_week,
}
