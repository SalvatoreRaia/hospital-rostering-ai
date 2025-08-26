# scheduler.py
from ortools.sat.python import cp_model
import pandas as pd
import random
import json
from datetime import datetime, timedelta
import os

# =========================
# PATHS AND TOGGLES (agnostic)
# =========================
# Base folder = folder where this file lives, so the project can be moved anywhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Toggle: use CSV overrides for residents / required_staff / calendar / unavailability / night_shifts
USE_CSV = True  # set to True when your CSVs are ready

# Toggle: use constraints from data/conditions.csv via conditions_registry.py
USE_EXTRA_CONDITIONS = True

# Optional: random seed (for reproducible random picks of unavailability/night_shifts when CSVs omit them)
SEED = None  # e.g., 42; set to None for non-deterministic runs
if SEED is not None:
    random.seed(SEED)

# Ensure outputs directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Make a unique folder for this run using timestamp (YYYYMMDD_HHMMSS)
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUTS_DIR, f"run_{RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)

# === CSV LOADERS (settings + residents) ===

def _parse_bool(v):
    """Parse typical truthy strings/numbers to bool: 1/true/yes/on → True; else False."""
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}

def load_settings_csv(data_dir):
    """
    Read SETTINGS + (later) CONSTRAINTS rows from settings.csv.
    Returns a dict: start_date, end_date, weekdays_only, required_staff, seed.
    Missing values fall back to your defaults (defined above).
    """
    path = os.path.join(data_dir, "settings.csv")
    out = {
        "start_date": None,
        "end_date": None,
        "weekdays_only": True,
        "required_staff": {"morning": 4, "afternoon": 2},
        "seed": None,
    }
    if not os.path.exists(path):
        print("settings.csv not found — using defaults for dates/coverage.")
        return out

    # More tolerant CSV parsing (handles spaces after commas and quoted JSON)
    df = pd.read_csv(path, engine="python", skipinitialspace=True)

    # SETTINGS rows (type == "setting"); if no 'type' column, treat all as settings.
    df_set = df[df["type"].astype(str).str.lower() == "setting"] if "type" in df.columns else df
    if not df_set.empty:
        kv = {str(k): v for k, v in zip(df_set["key"], df_set["value"])}
        # Dates
        sd = kv.get("start_date")
        ed = kv.get("end_date")
        if sd:
            out["start_date"] = datetime.strptime(str(sd), "%Y-%m-%d")
        if ed:
            out["end_date"] = datetime.strptime(str(ed), "%Y-%m-%d")
        # Weekdays only
        wdo = kv.get("weekdays_only")
        if wdo is not None:
            out["weekdays_only"] = _parse_bool(wdo)
        # Required staff
        m = kv.get("required_staff_morning")
        a = kv.get("required_staff_afternoon")
        if m is not None and str(m) != "":
            out["required_staff"]["morning"] = int(m)
        if a is not None and str(a) != "":
            out["required_staff"]["afternoon"] = int(a)
        # Seed
        seed = kv.get("seed")
        if seed is not None and str(seed).strip() != "":
            out["seed"] = int(seed)

    return out

def _coerce_json_list(cell):
    """
    Parse a CSV cell into list[str] robustly.
    Accepts:
      - JSON arrays like '["2025-07-03","2025-07-10"]'
      - '[]' or empty -> []
      - Fallback: '2025-07-01; 2025-07-02' or comma-separated -> ["2025-07-01","2025-07-02"]
      - Single value '2025-07-01' -> ["2025-07-01"]
    Trims spaces and drops empties. This keeps CSV editing flexible (JSON or simple strings).
    """
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "" or s == "[]":
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass
    # Fallback if someone used semicolons/commas instead of JSON
    sep = ";" if ";" in s else ("," if "," in s else None)
    if sep:
        return [t.strip() for t in s.split(sep) if t.strip()]
    return [s] if s else []

def load_residents_csv(data_dir):
    """
    Read residents.csv and build:
      - residents: list[str]
      - unavailable: dict[name] -> list[str]
      - night_shifts: dict[name] -> list[str]
    We KEEP data exactly as given (no random fill, no dedup).
    """
    path = os.path.join(data_dir, "residents.csv")
    residents = []
    unavailable = {}
    night_shifts = {}
    if not os.path.exists(path):
        print("residents.csv not found — using default generated residents.")
        return residents, unavailable, night_shifts

    df = pd.read_csv(path)
    required = {"resident", "unavailability", "night_shifts"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"residents.csv missing columns: {sorted(missing)}")

    # Read rows and keep dates exactly as given (no dedup, no horizon pruning)
    for _, row in df.iterrows():
        name = str(row["resident"]).strip()
        if not name:
            continue  # skip blank names
        residents.append(name)

        unav   = _coerce_json_list(row["unavailability"])
        nshift = _coerce_json_list(row["night_shifts"])

        unavailable[name] = unav
        night_shifts[name] = nshift

    return residents, unavailable, night_shifts



# =========================
# 1) GLOBAL DEFAULTS (can be overridden by CSVs)
# =========================
print("Generating weekdays for July 2025...")
residents = [f"Res{i}" for i in range(9)]
required_staff = {"morning": 4, "afternoon": 2}
shifts = ["morning", "afternoon"]

# Default planning horizon (weekdays only)
start_date = datetime(2025, 7, 1)
end_date = datetime(2025, 7, 31)

# ---- CSV OVERRIDES (settings.csv) ----
WEEKDAYS_ONLY = True  # default; may be overridden by settings.csv
if USE_CSV:
    settings = load_settings_csv(DATA_DIR)
    if settings["start_date"] is not None:
        start_date = settings["start_date"]
    if settings["end_date"] is not None:
        end_date = settings["end_date"]
    WEEKDAYS_ONLY = settings["weekdays_only"]
    required_staff = settings["required_staff"]
    SEED = settings["seed"]
    if SEED is not None:
        random.seed(SEED)
else:
    WEEKDAYS_ONLY = True  # keep original behavior

# Build day list and date map (weekdays vs full week)
days = []
date_map = {}
curr = start_date
while curr <= end_date:
    # Include a date if:
    # - WEEKDAYS_ONLY is False  -> include all days (Mon..Sun)
    # - WEEKDAYS_ONLY is True   -> include only Mon..Fri (weekday() < 5)
    # (not WEEKDAYS_ONLY) short-circuits the OR, so False → test weekday; True → include all.
    if (not WEEKDAYS_ONLY) or (curr.weekday() < 5):
        lab = curr.strftime("%Y-%m-%d")
        days.append(lab)
        date_map[lab] = curr
    curr += timedelta(days=1)

print(f"Found {len(days)} day(s) in the selected period. Weekdays only: {WEEKDAYS_ONLY}")
print("Days being scheduled:", days)

# =========================
# 2) RANDOM OR CSV CONSTRAINT INPUTS (residents, unavailability & night shifts)
# =========================

print("\nLoading residents and constraints input...")

unavailable = {}
night_shifts = {}

if USE_CSV:
    # Read everything from residents.csv (no random fallback)
    res_list, unav, nshift = load_residents_csv(DATA_DIR)
    if res_list:
        residents = res_list
        print(f"Loaded {len(residents)} residents from residents.csv.")
    else:
        print("No residents found in residents.csv — keeping defaults.")
    unavailable = unav
    night_shifts = nshift
else:
    # Original random behavior for non-CSV mode (unchanged)
    for r in residents:
        unavailable[r] = random.sample(days, 1) if len(days) > 0 else []
        night_shifts[r] = random.sample(days, 1) if len(days) > 0 else []

print("\nConstraints used:")
print("Unavailability per resident:")
for r in residents:
    print(f"  {r}: {unavailable.get(r, [])}")
print("\nNight shifts :") # (as provided; no auto-fill in CSV mode)
for r in residents:
    print(f"  {r}: {night_shifts.get(r, [])}")

# =========================
# 3) MODEL AND VARIABLES
# =========================
print("\nBuilding constraint model...")
model = cp_model.CpModel()

# Decision variables: x[(resident, day, shift)] in {0,1}
x = {}
def _safe(r):
    return str(r).replace(" ", "_").replace(".", "_").replace("/", "_")
for r in residents:
    for d in days:
        for s in shifts:
            x[(r, d, s)] = model.NewBoolVar(f"x_{_safe(r)}_{d}_{s}")

# =========================
# 4) CONSTRAINTS FROM REGISTRY (CSV-driven)
# =========================
# We removed the inline constraints block. Enable the equivalents in data/conditions.csv.
conditions_applied = []
if USE_EXTRA_CONDITIONS:
    try:
        from conditions_registry import load_conditions_from_csv, apply_conditions
        conditions_applied = load_conditions_from_csv(DATA_DIR, filename="settings.csv")  # list of {key, params}
        apply_conditions(
            model=model,
            x=x,
            residents=residents,
            days=days,
            shifts=shifts,
            date_map=date_map,
            required_staff=required_staff,
            conditions=conditions_applied,
            context={"unavailable": unavailable, "night_shifts": night_shifts},
        )
    except Exception as e:
        print(f"[conditions_registry] Skipped due to error: {e}")
        conditions_applied = []

# =========================
# 5) SOLVE
# =========================
print("\nSolving the scheduling problem...")
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = False
# Make CP-SAT search reproducible when seed is provided
if SEED is not None:
    try:
        solver.parameters.random_seed = int(SEED)
    except Exception:
        pass
status = solver.Solve(model)

status_map = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}
print(f"Solver status: {status_map.get(status, 'UNKNOWN')}")

# =========================
# 6) OUTPUTS (wide CSV + concise terminal print + run_info)
# =========================
# - We write run_info.csv first (always), then stop if infeasible.
# - We build a single wide CSV "schedule_wide.csv":
#     * First column "Date" starts with totals rows, then all dates.
#     * One column per resident. Cell values:
#         M = morning assignment
#         A = afternoon assignment
#         N = night (night day or next day), shown only if not assigned M/A that day
#         F = free (no M/A, not a night)
#     * Final column "Notations" explains totals rows.
# - We print the same wide table to the terminal.
# - Resident-wise summary at the end is computed from the actual decision variables.
# Always write a run_info.csv capturing all tunables and inputs,
# even if infeasible.

def to_json(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

# 6.0 Run info (always write; useful for debugging even if infeasible)
run_info_rows = [
    {"key": "run_timestamp", "value": RUN_TS},
    {"key": "weekdays_only", "value": True},
    {"key": "start_date", "value": start_date.strftime("%Y-%m-%d")},
    {"key": "end_date", "value": end_date.strftime("%Y-%m-%d")},
    {"key": "n_days", "value": len(days)},
    {"key": "n_residents", "value": len(residents)},
    {"key": "residents", "value": to_json(residents)},
    {"key": "required_staff_morning", "value": required_staff.get("morning")},
    {"key": "required_staff_afternoon", "value": required_staff.get("afternoon")},
    {"key": "USE_CSV", "value": USE_CSV},
    {"key": "USE_EXTRA_CONDITIONS", "value": USE_EXTRA_CONDITIONS},
    {"key": "seed", "value": SEED if SEED is not None else ""},
    {"key": "status", "value": status_map.get(status, "UNKNOWN")},
    {"key": "constraints_used_keys", "value": to_json([c["key"] for c in conditions_applied])},
    {"key": "constraints_used_params", "value": to_json({c["key"]: c.get("params", {}) for c in conditions_applied})},
    {"key": "unavailability", "value": to_json(unavailable)},
    {"key": "night_shifts", "value": to_json(night_shifts)},
]
pd.DataFrame(run_info_rows).to_csv(os.path.join(RUN_DIR, "run_info.csv"), index=False)

# If infeasible or worse, stop after writing run_info.csv (no assignments / heatmap / wide table)
if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
    print("No feasible solution found for the given constraints.")
    raise SystemExit(0)

# 6.1 Build a letters grid (date rows × resident columns)
letters = pd.DataFrame(index=days, columns=residents, data="")

# Precompute night days per resident: mark the night date and the next date
night_lookup = {r: set() for r in residents}
for r in residents:
    for nd in night_shifts.get(r, []):
        if nd in date_map:
            night_lookup[r].add(nd)
            nd2 = (date_map[nd] + timedelta(days=1)).strftime("%Y-%m-%d")
            if nd2 in date_map:
                night_lookup[r].add(nd2)

# Fill letters with precedence M > A > N > F (single letter per cell)
for d in days:
    for r in residents:
        m = solver.Value(x[(r, d, "morning")])
        a = solver.Value(x[(r, d, "afternoon")])
        # If both are 1 (shouldn't happen with one_shift_per_day), prefer M to avoid ambiguity.
        if m == 1:
            letters.at[d, r] = "M"
        elif a == 1:
            letters.at[d, r] = "A"
        elif d in night_lookup[r]:
            letters.at[d, r] = "N"
        else:
            letters.at[d, r] = "F"

# 6.2 Totals per resident (compute from decision variables)
# Definitions:
#  - Morning total       = sum_d x[(r,d,"morning")]
#  - Afternoon total     = sum_d x[(r,d,"afternoon")]
#  - Free days           = {d | no M/A AND d not in night_lookup[r]}
#  - Free + Unavailability = | Free ∪ (unavailability[r] ∩ horizon) |
days_set = set(days)
tot_m = {}
tot_a = {}
tot_free = {}
tot_free_plus_unav = {}
for r in residents:
    tm = sum(solver.Value(x[(r, d, "morning")])   for d in days)
    ta = sum(solver.Value(x[(r, d, "afternoon")]) for d in days)
    free_days_set = {
        d for d in days
        if (solver.Value(x[(r, d, "morning")]) + solver.Value(x[(r, d, "afternoon")]) == 0)
        and (d not in night_lookup[r])
    }
    unav_days_set = set(unavailable.get(r, [])) & days_set

    tot_m[r] = int(tm)
    tot_a[r] = int(ta)
    tot_free[r] = len(free_days_set)
    tot_free_plus_unav[r] = len(free_days_set | unav_days_set)

# Assemble totals rows (rows = metrics, columns = residents)
row_names = [
    "Total morning",
    "Total afternoon",
    "Total free days",
    "Total free + unavailability",
]
df_totals = pd.DataFrame(
    [tot_m, tot_a, tot_free, tot_free_plus_unav],
    index=row_names,
    columns=residents,
)

# 6.3 Stitch totals above the per-date rows and add "Date" as first column
df_out = pd.concat([df_totals, letters], axis=0)
df_out.insert(0, "Date", list(df_out.index))

# 6.4 Add "Notations" column (explanations only on totals rows)
notations = []
for idx in df_out.index:
    if idx == "Total morning":
        notations.append("M = morning")
    elif idx == "Total afternoon":
        notations.append("A = afternoon")
    elif idx == "Total free days":
        notations.append("F = free (no M/A, not night)")
    elif idx == "Total free + unavailability":
        notations.append("F + U = free days + unavailable days")
    else:
        notations.append("")  # date rows: blank
df_out["Notations"] = notations

# 6.5 Write wide CSV
wide_path = os.path.join(RUN_DIR, "schedule_wide.csv")
df_out.to_csv(wide_path, index=False)

# 6.6 Concise terminal output: status + the wide table
print("\n=== Solve summary ===")
print(f"Status: {status_map.get(status, 'UNKNOWN')}")
print(f"Days: {len(days)} | Residents: {len(residents)} | Shifts/day: {len(shifts)}")
print(f"Coverage per shift: {required_staff}")
print("\nFinal Schedule (wide):")
print(df_out.to_string(index=False))

# 6.7 Resident-wise summary (compact & aligned with totals above)
print("\nResident-wise summary (compact):")
for r in residents:
    tm = tot_m[r]
    ta = tot_a[r]
    # Recompute sets here to keep the print in sync with the definitions above
    free_days_set = {
        d for d in days
        if (solver.Value(x[(r, d, "morning")]) + solver.Value(x[(r, d, "afternoon")]) == 0)
        and (d not in night_lookup[r])
    }
    unav_days_set = set(unavailable.get(r, [])) & days_set
    both_in_one_day = sum(
        1 for d in days
        if (solver.Value(x[(r, d, "morning")]) == 1 and solver.Value(x[(r, d, "afternoon")]) == 1)
    )
    if both_in_one_day:
        print(f"[WARN] {r} has {both_in_one_day} day(s) with both M and A assigned.")
    print(f"\n {r}:")
    print(f"    Total shifts:                 {tm + ta}")
    print(f"    Morning shifts:               {tm}")
    print(f"    Afternoon shifts:             {ta}")
    print(f"    Free days (excl nights):      {len(free_days_set)}")
    print(f"    Free + Unavailability days:   {len(free_days_set | unav_days_set)}")


# =========================
# 7) HEATMAP (PNG saved per run)
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

print("\nGenerating color-coded heatmap...")

# Build status matrix per (resident, day)
status_matrix = pd.DataFrame(0, index=residents, columns=days)
for r in residents:
    for d in days:
        # Pick one; with one-shift-per-day active this shouldn’t happen.
        # We map it to 'morning' to avoid creating a value '3'.
        m = solver.Value(x[(r, d, "morning")])
        a = solver.Value(x[(r, d, "afternoon")])
        if m and a:
            status_matrix.loc[r, d] = 1
        elif m:
            status_matrix.loc[r, d] = 1
        elif a:
            status_matrix.loc[r, d] = 2
        else:
            status_matrix.loc[r, d] = 0

# Mark unavailability (-2)
for r in residents:
    for d in unavailable.get(r, []):
        if d in status_matrix.columns:
            status_matrix.loc[r, d] = -2

# Mark night shift exclusions (-1) on night day and the next day
for r in residents:
    for nd in night_shifts.get(r, []):
        if nd in date_map and nd in status_matrix.columns and status_matrix.loc[r, nd] == 0:
            status_matrix.loc[r, nd] = -1
        if nd in date_map:
            nd2 = (date_map[nd] + timedelta(days=1)).strftime("%Y-%m-%d")
            if nd2 in status_matrix.columns and status_matrix.loc[r, nd2] == 0:
                status_matrix.loc[r, nd2] = -1

# Colors
cmap_colors = {
    -2: "#ff6666",  # unavailable
    -1: "#ffcc99",  # night shift exclusion
     0: "#ffffff",  # free
     1: "#6699ff",  # morning
     2: "#ffff66",  # afternoon
}
cmap = ListedColormap([cmap_colors[i] for i in sorted(cmap_colors.keys())])
bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(16, 6))
ax = sns.heatmap(
    status_matrix,
    cmap=cmap,
    norm=norm,
    linewidths=0.5,
    linecolor="gray",
    cbar=False,
    xticklabels=True,
    yticklabels=True,
)

plt.title("Resident Daily Status Heatmap", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Resident")
plt.xticks(rotation=90)

# Leave room for legend on the right
fig = plt.gcf()
fig.subplots_adjust(right=0.82)

legend_patches = [
    mpatches.Patch(color=cmap_colors[-2], label="Unavailable"),
    mpatches.Patch(color=cmap_colors[-1], label="Night shift exclusion"),
    mpatches.Patch(color=cmap_colors[0], label="Free"),
    mpatches.Patch(color=cmap_colors[1], label="Morning"),
    mpatches.Patch(color=cmap_colors[2], label="Afternoon"),
]
ax.legend(
    handles=legend_patches,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0,
    title="Legend",
)

# Save PNG to the run folder before showing
heatmap_path = os.path.join(RUN_DIR, "heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.tight_layout()
plt.show()

print(f"\nSaved run outputs to: {RUN_DIR}")
print(" - assignments_long.csv")
print(" - run_info.csv")
print(" - heatmap.png")
