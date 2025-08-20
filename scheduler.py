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
USE_CSV = False  # set to True when your CSVs are ready

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

# ---- CSV OVERRIDES (optional) ----
if USE_CSV:
    # residents.csv — column: resident
    path = os.path.join(DATA_DIR, "residents.csv")
    if os.path.exists(path):
        df_res = pd.read_csv(path)
        if "resident" in df_res.columns:
            names = [str(x) for x in df_res["resident"].dropna().tolist()]
            if names:
                residents = names
                print(f"Loaded {len(residents)} residents from CSV.")

    # required_staff.csv — columns: shift,required
    path = os.path.join(DATA_DIR, "required_staff.csv")
    if os.path.exists(path):
        df_req = pd.read_csv(path)
        if set(["shift", "required"]).issubset(df_req.columns):
            tmp = {str(row["shift"]): int(row["required"]) for _, row in df_req.iterrows()}
            if "morning" in tmp and "afternoon" in tmp:
                required_staff = {"morning": tmp["morning"], "afternoon": tmp["afternoon"]}
                print(f"Loaded required_staff from CSV: {required_staff}")
            else:
                print("required_staff.csv must define 'morning' and 'afternoon'.")

    # calendar.csv — columns: start_date,end_date (YYYY-MM-DD)
    path = os.path.join(DATA_DIR, "calendar.csv")
    if os.path.exists(path):
        df_cal = pd.read_csv(path)
        if set(["start_date", "end_date"]).issubset(df_cal.columns) and len(df_cal) >= 1:
            sd = str(df_cal.loc[0, "start_date"])
            ed = str(df_cal.loc[0, "end_date"])
            try:
                start_date = datetime.strptime(sd, "%Y-%m-%d")
                end_date = datetime.strptime(ed, "%Y-%m-%d")
                print(f"Loaded calendar from CSV: {sd} → {ed} (weekdays only).")
            except Exception as e:
                print(f"calendar.csv date parse error: {e}")

# Build weekday list and date map
days = []
date_map = {}
curr = start_date
while curr <= end_date:
    if curr.weekday() < 5:  # Mon..Fri
        lab = curr.strftime("%Y-%m-%d")
        days.append(lab)
        date_map[lab] = curr
    curr += timedelta(days=1)

print(f"Found {len(days)} weekdays in the selected period.")
print("Days being scheduled:", days)

# =========================
# 2) RANDOM OR CSV CONSTRAINT INPUTS (unavailability & night shifts)
# =========================
print("\nGenerating random constraints...")

unavailable = {}
night_shifts = {}

if USE_CSV:
    # Initialize
    for r in residents:
        unavailable[r] = []
        night_shifts[r] = []

    # unavailability.csv — columns: resident,date
    path = os.path.join(DATA_DIR, "unavailability.csv")
    if os.path.exists(path):
        df_un = pd.read_csv(path)
        if set(["resident", "date"]).issubset(df_un.columns):
            for _, row in df_un.iterrows():
                r = str(row["resident"])
                d = str(row["date"])
                if r in residents and d in date_map:
                    unavailable[r].append(d)

    # night_shifts.csv — columns: resident,date
    path = os.path.join(DATA_DIR, "night_shifts.csv")
    if os.path.exists(path):
        df_n = pd.read_csv(path)
        if set(["resident", "date"]).issubset(df_n.columns):
            for _, row in df_n.iterrows():
                r = str(row["resident"])
                d = str(row["date"])
                if r in residents and d in date_map:
                    night_shifts[r].append(d)

    # Fallback to preserve original behavior: ensure each resident has at least one of each
    for r in residents:
        if len(unavailable[r]) == 0 and len(days) > 0:
            unavailable[r] = random.sample(days, 1)
        if len(night_shifts[r]) == 0 and len(days) > 0:
            night_shifts[r] = random.sample(days, 1)

else:
    # Original random behavior
    for r in residents:
        unavailable[r] = random.sample(days, 1) if len(days) > 0 else []
        night_shifts[r] = random.sample(days, 1) if len(days) > 0 else []

print("\nConstraints used:")
print("Unavailability per resident:")
for r in residents:
    print(f"  {r}: {unavailable.get(r, [])}")
print("\nNight shifts (>= 1 per resident):")
for r in residents:
    if night_shifts.get(r):
        print(f"  {r}: {night_shifts[r][0]}")
    else:
        print(f"  {r}: []")

# =========================
# 3) MODEL AND VARIABLES
# =========================
print("\nBuilding constraint model...")
model = cp_model.CpModel()

# Decision variables: x[(resident, day, shift)] in {0,1}
x = {}
for r in residents:
    for d in days:
        for s in shifts:
            x[(r, d, s)] = model.NewBoolVar(f"x_{r}_{d}_{s}")

# =========================
# 4) CONSTRAINTS FROM REGISTRY (CSV-driven)
# =========================
# We removed the inline constraints block. Enable the equivalents in data/conditions.csv.
conditions_applied = []
if USE_EXTRA_CONDITIONS:
    try:
        from conditions_registry import load_conditions_from_csv, apply_conditions
        conditions_applied = load_conditions_from_csv(DATA_DIR)  # list of {key, params}
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
solver.parameters.log_search_progress = True
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
# 6) OUTPUTS (CSV + heatmap PNG per run)
# =========================
# Always write a run_info.csv capturing all tunables and inputs,
# even if infeasible (useful for debugging the run).
def to_json(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

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

# If infeasible or worse, stop here after writing run_info.csv (no assignments / heatmap).
if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
    print("No feasible solution found for the given constraints.")
    # Also write an empty assignments file with the right columns for consistency
    pd.DataFrame(columns=["resident", "date", "shift"]).to_csv(
        os.path.join(RUN_DIR, "assignments_long.csv"), index=False
    )
    # Nothing to plot
    raise SystemExit(0)

# Build a long format of assignments: one row per (resident, date, shift) where x=1
rows_long = []
for d in days:
    for s in shifts:
        for r in residents:
            if solver.Value(x[(r, d, s)]) == 1:
                rows_long.append({"resident": r, "date": d, "shift": s})
df_long = pd.DataFrame(rows_long, columns=["resident", "date", "shift"])
df_long.to_csv(os.path.join(RUN_DIR, "assignments_long.csv"), index=False)

# (Optional) Console table like your original script (wide format)
schedule = []
for d in days:
    for s in shifts:
        row = {"Date": d, "Shift": s}
        for r in residents:
            row[r] = solver.Value(x[(r, d, s)])
        schedule.append(row)
df_wide = pd.DataFrame(schedule)
print("\nFinal Schedule (wide table):\n")
print(df_wide.to_string(index=False))

# Resident-wise summary (same logic as before)
print("\nResident-wise summary:")
for r in residents:
    total = 0
    morning_count = 0
    afternoon_count = 0
    both_in_one_day = 0
    free_days = 0
    for d in days:
        m = solver.Value(x[(r, d, "morning")])
        a = solver.Value(x[(r, d, "afternoon")])
        total += m + a
        morning_count += m
        afternoon_count += a
        if m + a == 0:
            free_days += 1
        if m == 1 and a == 1:
            both_in_one_day += 1
    print(f"\n {r}:")
    print(f"    Total shifts:            {total}")
    print(f"    Morning shifts:          {morning_count}")
    print(f"    Afternoon shifts:        {afternoon_count}")
    print(f"    Morning+Afternoon days:  {both_in_one_day}")
    print(f"    Completely free days:    {free_days}")

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
        m = solver.Value(x[(r, d, "morning")])
        a = solver.Value(x[(r, d, "afternoon")])
        if m and a:
            status_matrix.loc[r, d] = 3
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
     3: "#cc99ff",  # both (should not occur with one-shift-per-day)
}
cmap = ListedColormap([cmap_colors[i] for i in sorted(cmap_colors.keys())])
bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
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
    mpatches.Patch(color=cmap_colors[3], label="Morning + Afternoon"),
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
