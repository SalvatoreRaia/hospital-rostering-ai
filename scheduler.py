from ortools.sat.python import cp_model
import pandas as pd
import random
from datetime import datetime, timedelta
import os  # [CSV]

# [CSV] === CSV SWITCH & FOLDER ===
# Turn this ON to read overrides from CSV, OFF to use the hardcoded defaults/randoms.
USE_CSV = False  # set to True when your CSVs are ready
DATA_DIR = r"C:\\Users\salvo\Data(D)\\Registrazioni\AI Fundamentals\\Project\data"

# === 1. DEFINE GLOBAL PARAMETERS ===
print("Generating weekdays for July 2025...")
residents = [f'Res{i}' for i in range(9)]
required_staff = {'morning': 4, 'afternoon': 2}
shifts = ['morning', 'afternoon']

# Generate all weekdays in July 2025  (DEFAULTS)
start_date = datetime(2025, 7, 1)
end_date = datetime(2025, 7, 31)

# [CSV] === OVERRIDE GLOBALS FROM CSV (residents, required_staff, start/end) ===
if USE_CSV:
    # residents.csv — one column: resident
    path = os.path.join(DATA_DIR, "residents.csv")
    if os.path.exists(path):
        df_res = pd.read_csv(path)
        if 'resident' in df_res.columns:
            names = [str(x) for x in df_res['resident'].dropna().tolist()]
            if names:
                residents = names
                print(f"👥 Loaded {len(residents)} residents from CSV.")

    # required_staff.csv — columns: shift,required (only 'morning' and 'afternoon' supported in this script)
    path = os.path.join(DATA_DIR, "required_staff.csv")
    if os.path.exists(path):
        df_req = pd.read_csv(path)
        if set(['shift', 'required']).issubset(df_req.columns):
            tmp = {str(row['shift']): int(row['required']) for _, row in df_req.iterrows()}
            if 'morning' in tmp and 'afternoon' in tmp:
                required_staff = {'morning': tmp['morning'], 'afternoon': tmp['afternoon']}
                print(f"Loaded required_staff from CSV: {required_staff}")
            else:
                print("required_staff.csv must define 'morning' and 'afternoon'.")

    # calendar.csv — columns: start_date,end_date (YYYY-MM-DD)
    path = os.path.join(DATA_DIR, "calendar.csv")
    if os.path.exists(path):
        df_cal = pd.read_csv(path)
        if set(['start_date', 'end_date']).issubset(df_cal.columns) and len(df_cal) >= 1:
            sd = str(df_cal.loc[0, 'start_date'])
            ed = str(df_cal.loc[0, 'end_date'])
            try:
                start_date = datetime.strptime(sd, '%Y-%m-%d')
                end_date = datetime.strptime(ed, '%Y-%m-%d')
                print(f"Loaded calendar from CSV: {sd} → {ed} (weekdays only).")
            except Exception as e:
                print(f"calendar.csv date parse error: {e}")

days = []
date_map = {}
curr = start_date
while curr <= end_date:
    if curr.weekday() < 5:  # Monday to Friday (this script schedules ONLY weekdays)
        day_label = curr.strftime('%Y-%m-%d')
        days.append(day_label)
        date_map[day_label] = curr
    curr += timedelta(days=1)

print(f"Found {len(days)} weekdays in the selected period.")
print("Days being scheduled:", days)

# === 2. GENERATE RANDOM UNAVAILABILITY & NIGHT SHIFTS ===
print("\n Generating random constraints...")

unavailable = {}
night_shifts = {}

# [CSV] === OVERRIDE unavailability/night_shifts FROM CSV WHEN AVAILABLE ===
if USE_CSV:
    # Initialize empty lists for all residents
    for r in residents:
        unavailable[r] = []
        night_shifts[r] = []

    # unavailability.csv — columns: resident,date (YYYY-MM-DD)
    path = os.path.join(DATA_DIR, "unavailability.csv")
    if os.path.exists(path):
        df_un = pd.read_csv(path)
        if set(['resident', 'date']).issubset(df_un.columns):
            for _, row in df_un.iterrows():
                r = str(row['resident'])
                d = str(row['date'])
                if r in unavailable and d in date_map:
                    unavailable[r].append(d)
                else:
                    # silently ignore bad rows (unknown resident or date outside calendar)
                    pass

    # night_shifts.csv — columns: resident,date (YYYY-MM-DD) (one per resident recommended)
    path = os.path.join(DATA_DIR, "night_shifts.csv")
    if os.path.exists(path):
        df_n = pd.read_csv(path)
        if set(['resident', 'date']).issubset(df_n.columns):
            for _, row in df_n.iterrows():
                r = str(row['resident'])
                d = str(row['date'])
                if r in night_shifts and d in date_map:
                    night_shifts[r].append(d)

    # Fallback: if any resident missing entries, fill with one random day (to keep your original behaviour)
    for r in residents:
        if len(unavailable[r]) == 0:
            unavailable[r] = random.sample(days, 1)
        if len(night_shifts[r]) == 0:
            night_shifts[r] = random.sample(days, 1)

else:
    # Original random generation (unchanged)
    for r in residents:
        unavailable[r] = random.sample(days, 1)
        night_shifts[r] = random.sample(days, 1)

print("\nConstraints used:")
print("Unavailability per resident:")
for r in unavailable:
    print(f"  {r}: {unavailable[r]}")
print("\nNight shifts (1 per resident):")
for r in night_shifts:
    # NOTE: if your CSV has more than 1 night for a resident, we still print the first for compatibility.
    print(f"  {r}: {night_shifts[r][0]}")

# === 3. INITIALIZE CP MODEL ===
print("\nBuilding constraint model...")
model = cp_model.CpModel()

# Define assignment variables: x[resident, day, shift] ∈ {0, 1}
x = {}
for r in residents:
    for d in days:
        for s in shifts:
            x[r, d, s] = model.NewBoolVar(f'x_{r}_{d}_{s}')

# === 4. ADD CONSTRAINTS ===

# Constraint 1: Required number of staff per shift
print("Adding staffing constraints...")
for d in days:
    for s in shifts:
        model.Add(sum(x[r, d, s] for r in residents) == required_staff[s])

# Constraint 2: No resident works both shifts in a day
print("Adding daily shift limit constraint...")
for r in residents:
    for d in days:
        model.Add(x[r, d, 'morning'] + x[r, d, 'afternoon'] <= 1)

# Constraint 3: No afternoon → morning consecutive days
print("Adding rest period constraint (no afternoon then morning)...")
for r in residents:
    for i in range(len(days) - 1):
        d1 = days[i]
        d2 = days[i + 1]
        model.Add(x[r, d1, 'afternoon'] + x[r, d2, 'morning'] <= 1)

# Constraint 4: Enforce unavailable days
print("Adding unavailability constraints...")
for r in unavailable:
    for d in unavailable[r]:
        for s in shifts:
            model.Add(x[r, d, s] == 0)

# Constraint 5: Night shift exclusions (no shifts on night shift day and the next day)
print("Adding night shift constraints...")
for r in night_shifts:
    nd = night_shifts[r][0]
    n_date = date_map[nd]

    # Exclude all shifts on the day of the night shift
    model.Add(x[r, nd, 'morning'] == 0)
    model.Add(x[r, nd, 'afternoon'] == 0)

    # Exclude all shifts on the next day
    next_day = (n_date + timedelta(days=1)).strftime('%Y-%m-%d')
    if next_day in days:
        model.Add(x[r, next_day, 'morning'] == 0)
        model.Add(x[r, next_day, 'afternoon'] == 0)

# Constraint 6: Strict Fairness — total shifts per resident must be equal or differ by at most 1
print("Enforcing strict fairness (min/max balance)...")

total_required_shifts = len(days) * (required_staff['morning'] + required_staff['afternoon'])
min_shifts = total_required_shifts // len(residents)
max_shifts = min_shifts + 1 if total_required_shifts % len(residents) > 0 else min_shifts

print(f" Total required shifts: {total_required_shifts}")
print(f" Each resident must have between {min_shifts} and {max_shifts} shifts.")

total_shifts = {}
for r in residents:
    total_shifts[r] = model.NewIntVar(min_shifts, max_shifts, f'total_{r}')
    model.Add(total_shifts[r] == sum(x[r, d, s] for d in days for s in shifts))

# === 5. SOLVE MODEL ===
print("\nSolving the scheduling problem...")
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True  # Optional: enable solver internal logs
status = solver.Solve(model)

# === 6. OUTPUT RESULTS ===
if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
    print("\nSolution found! Generating schedule table...")
    schedule = []
    for d in days:
        for s in shifts:
            row = {'Date': d, 'Shift': s}
            for r in residents:
                row[r] = solver.Value(x[r, d, s])
            schedule.append(row)
    df = pd.DataFrame(schedule)
    print("\n Final Schedule:\n")
    print(df.to_string(index=False))
else:
    print("\nNo feasible solution found for the given constraints.")

print("\nResident-wise summary:")

for r in residents:
    total = 0
    morning_count = 0
    afternoon_count = 0
    both_in_one_day = 0
    free_days = 0

    for d in days:
        m = solver.Value(x[r, d, 'morning'])
        a = solver.Value(x[r, d, 'afternoon'])
        total += m + a
        morning_count += m
        afternoon_count += a
        if m + a == 0:
            free_days += 1
        if m == 1 and a == 1:
            both_in_one_day += 1  # Should be 0 due to your constraint

    print(f"\n {r}:")
    print(f"    Total shifts:            {total}")
    print(f"    Morning shifts:         {morning_count}")
    print(f"    Afternoon shifts:       {afternoon_count}")
    print(f"    Morning+Afternoon days: {both_in_one_day}")
    print(f"    Completely free days:   {free_days}")

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

print("\n Generating color-coded heatmap (fixed)...")

# Step 1: Construct numeric matrix of status values
status_matrix = pd.DataFrame(0, index=residents, columns=days)

for r in residents:
    for d in days:
        morning = solver.Value(x[r, d, 'morning'])
        afternoon = solver.Value(x[r, d, 'afternoon'])
        if morning and afternoon:
            status_matrix.loc[r, d] = 3
        elif morning:
            status_matrix.loc[r, d] = 1
        elif afternoon:
            status_matrix.loc[r, d] = 2
        else:
            status_matrix.loc[r, d] = 0  # default: free

# Mark unavailability (-2)
for r in unavailable:
    for d in unavailable[r]:
        status_matrix.loc[r, d] = -2

# Mark night shift exclusions (-1)
for r in night_shifts:
    n_date = date_map[night_shifts[r][0]]
    for offset in [-1, 0, 1]:
        check_date = n_date + timedelta(days=offset)
        label = check_date.strftime('%Y-%m-%d')
        if label in days and status_matrix.loc[r, label] == 0:
            status_matrix.loc[r, label] = -1

# Step 2: Define color mapping for each value
cmap_colors = {
    -2: '#ff6666',  # unavailable (red)
    -1: '#ffcc99',  # night shift exclusion (orange)
     0: '#ffffff',  # free
     1: '#6699ff',  # morning
     2: '#ffff66',  # afternoon
     3: '#cc99ff',  # both shifts
}

# Create colormap and normalization
from matplotlib.colors import ListedColormap, BoundaryNorm

cmap = ListedColormap([cmap_colors[i] for i in sorted(cmap_colors.keys())])
bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

# Step 3: Plot heatmap
plt.figure(figsize=(16, 6))
ax = sns.heatmap(
    status_matrix,
    cmap=cmap,
    norm=norm,
    linewidths=0.5,
    linecolor='gray',
    cbar=False,
    xticklabels=True,
    yticklabels=True
)

plt.title(" Resident Daily Status Heatmap – July 2025", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Resident")
plt.xticks(rotation=90)
fig = plt.gcf()
fig.subplots_adjust(right=0.82)  # <-- make room on the right

legend_patches = [
    mpatches.Patch(color=cmap_colors[-2], label='Unavailable'),
    mpatches.Patch(color=cmap_colors[-1], label='Night shift exclusion'),
    mpatches.Patch(color=cmap_colors[0], label='Free'),
    mpatches.Patch(color=cmap_colors[1], label='Morning'),
    mpatches.Patch(color=cmap_colors[2], label='Afternoon'),
    mpatches.Patch(color=cmap_colors[3], label='Morning + Afternoon'),
]

# Attach legend to the axes, outside right, vertically centered
ax.legend(
    handles=legend_patches,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.,
    title='Legend'
)

plt.tight_layout()

plt.show()
