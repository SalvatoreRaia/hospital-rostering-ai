# Hospital Rostering — CSV-Driven CP-SAT Scheduler (Part 1 of 2)

A compact, reproducible day-shift scheduler for hospital residents. It builds a CP-SAT model directly from **CSV files**, enforces operational rules (coverage, rest, calendar exclusions), and guarantees **strict fairness** (total shifts per resident differ by at most 1). It exports a wide **schedule CSV**, a **run log**, and a **heatmap** for quick validation.

> This is **Part 1/2** of the README. It covers overview, quick start, repo layout, data model, configuration, and running. Part 2 will include the formal constraints, registry keys table, outputs explained, night-shift semantics, troubleshooting/FAQ, and license/citation.


---

## Features

- **CSV-first configuration**: toggle constraints and set parameters in `data/settings.csv`; maintain calendars in `data/residents.csv`.
- **Constraint registry**: human-readable keys map to model-building functions in `conditions_registry.py`.
- **Hard guarantees**: exact coverage, one shift/day, no PM→AM, unavailability respected, night-shift day + next day blocked.
- **Strict fairness**: total shifts per resident ∈ {avg, avg+1}.
- **Reproducible**: optional seed; timestamped run folders.
- **Visuals**: color-coded heatmap and a “wide” CSV with daily letters (M/A/N/F) and per-resident totals.

Back-end: Python, Google **OR-Tools CP-SAT**, **pandas**, **matplotlib**/**seaborn**.


---

## Quick start (TL;DR)

1) **Clone and enter the repo**
   
   ~~~bash
   git clone https://github.com/SalvatoreRaia/hospital-rostering-ai.git
   cd hospital-rostering-ai
   ~~~

2) **Create a virtual environment (optional but recommended)**
   
   ~~~bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ~~~

3) **Install dependencies**

   Install the requirements like this:
   
   ~~~bash
   pip install -r requirements.txt
   ~~~

4) **Check/edit the CSVs** in `data/`:
   - `settings.csv` → planning window, weekdays-only, required staff, seed, and the list of enabled constraints (keys + params).
   - `residents.csv` → per-resident unavailability and night-shift dates.

5) **Run the solver**

   ~~~bash
   python scheduler.py
   ~~~

   You’ll get a fresh run folder at `outputs/run_YYYYMMDD_HHMMSS/` containing:
   - `schedule_wide.csv`
   - `run_info.csv`
   - `heatmap.png`


---

## Repository layout

hospital-rostering-ai/
├─ scheduler.py # Main entry point: loads CSVs, builds CP-SAT, solves, exports outputs
├─ conditions_registry.py # CSV-driven constraint registry (keys -> functions)
├─ data/
│ ├─ settings.csv # Settings + enabled constraints (type=setting|constraint)
│ └─ residents.csv # Resident calendars (unavailability, night_shifts)
└─ outputs/
└─ run_YYYYMMDD_HHMMSS/ # Auto-created per run (schedule_wide.csv, run_info.csv, heatmap.png)


**Paths** are computed relative to `scheduler.py`, so the repo can be moved anywhere.


---

## Data model

### `data/settings.csv`

- Mixed rows distinguished by the `type` column:
  - `setting` rows define: `start_date`, `end_date`, `weekdays_only`, `required_staff_morning`, `required_staff_afternoon`, `seed`.
  - `constraint` rows define: enabled **keys** and optional **params** (JSON).

- **Truthy values** accepted for enabling: `1/true/yes/on` (case-insensitive).
- If a date range is omitted, defaults are used (see `scheduler.py`).

**Columns**
- `type` — `"setting"` or `"constraint"`
- `key` — setting/constraint name
- `enabled` — `1` or `0` (or `true`/`false` etc.)
- `value` — value for setting rows (empty for constraints)
- `params` — JSON object (e.g. `{ "limit": 5 }`), empty `{}` if none
- `description` — free text

**Example (shortened)**

~~~csv
"type","key","enabled","value","params","description"
"setting","start_date",1,2025-09-01,"{}","Planning window start date (YYYY-MM-DD)."
"setting","end_date",1,2025-10-01,"{}","Planning window end date (YYYY-MM-DD)."
"setting","weekdays_only",1,"true","{}","If true, schedule only Monday–Friday."
"setting","required_staff_morning",1,4,"{}","Required residents for the morning shift."
"setting","required_staff_afternoon",1,3,"{}","Required residents for the afternoon shift."
"setting","seed",1,42,"{}","Random seed for reproducibility."
"constraint","coverage_exact_required",1,,"{}","Exact seats per day/shift."
"constraint","one_shift_per_day",1,,"{}","A resident works at most one shift/day."
"constraint","no_afternoon_then_morning",1,,"{}","Forbid PM(d) → AM(d+1)."
"constraint","enforce_unavailability",1,,"{}","Block assignments on unavailable dates."
"constraint","night_shift_exclusion",1,,"{}","Block day shifts on night day d and d+1."
"constraint","fairness_strict_minmax",1,,"{}","Totals differ by at most one."
~~~

> **Note:** Constraints are read from **`data/settings.csv`** (rows with `type=constraint`).  
> The code calls `load_conditions_from_csv(..., filename="settings.csv")`.


### `data/residents.csv`

- Per-resident calendars with flexible list parsing:
  - JSON arrays (even with doubled quotes from Excel) ✔
  - Semicolon/comma lists ✔
  - Singletons ✔
- Lists are normalized to Python lists of ISO dates.

**Columns**
- `resident` — string (name/ID)
- `unavailability` — list of dates (resident may not work any day shift)
- `night_shifts` — list of dates (night starting on `d` blocks day shifts on `d` and `d+1`)
- `notes` — free text

**Example (shortened)**

~~~csv
"resident","unavailability","night_shifts","notes"
"Giulia B","[""2025-09-03"",""2025-09-10""]","[""2025-09-17""]","prefers mornings"
"Lisa P.","[]","[""2025-09-08"",""2025-09-21""]","available most days"
"Laura","[""2025-09-07""]","[""2025-09-24""]","can swap weekends"
"Alessia","[""2025-09-02"",""2025-09-18""]","[""2025-09-15""]","new resident"
"Daniela","[""2025-09-04"",""2025-09-25""]","[]","requests fewer afternoons"
"Francesco","[""2025-09-29""]","[""2025-09-11"",""2025-09-28""]","backup only"
"Giulia A","[""2025-09-09""]","[""2025-09-03""]","course on 9th"
"Lisa F","[""2025-09-11"",""2025-09-22""]","[]","vacation mid-month"
"Res8","[]","[]","fully flexible"
~~~


---

## Configuration knobs

Most behavior is controlled from the CSVs. A few toggles exist in `scheduler.py`:

- `USE_CSV = True`  
  Read settings and residents from `data/*.csv` (recommended).

- `USE_EXTRA_CONDITIONS = True`  
  Apply the constraints listed under `type=constraint` in `data/settings.csv` via `conditions_registry.py`.

- `SEED = None`  
  Overridden by `seed` in `settings.csv` when present. Used to make CP-SAT search reproducible (and for the older random demo mode).


---

## Running the solver

1) **Edit** `data/settings.csv` and `data/residents.csv` as needed (see examples above).

2) **Run**:

   ~~~bash
   python scheduler.py
   ~~~

3) **Inspect outputs** in the new timestamped folder under `outputs/`:
   - `run_info.csv` — Horizon, coverage targets, enabled constraints (with params), seed, residents list, solver status, and the original calendar inputs. Always written (even if infeasible).
   - `schedule_wide.csv` — First 4 rows are per-resident totals (Morning, Afternoon, Free, Free+Unavailability). Then one row per date with single letters in each resident column:
     - `M` = morning; `A` = afternoon;  
       `N` = night day / next day (shown only if not assigned M/A that day);  
       `F` = free (no day shift, not a night-blocked day).
   - `heatmap.png` — Color-coded daily status (Unavailable, Night exclusion, Free, Morning, Afternoon).

**Return codes & statuses**
- The console prints CP-SAT status (`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, …).  
- If infeasible, the script writes `run_info.csv` and exits before table/heatmap generation.

**Reproducibility**
- Set `seed` in `data/settings.csv` (or `SEED` in `scheduler.py`) to make the CP-SAT search deterministic across runs with the same inputs.


---

## Notes on weekends & horizon

- `weekdays_only=true` in `settings.csv` restricts the planning horizon to Monday–Friday; otherwise all days are included.
- Dates are strings in the form `YYYY-MM-DD`.
- The solver builds an ordered list of days within `[start_date, end_date]` and maintains a `date_map` to compute “next day” for rules like PM→AM and night exclusions.


---

# Hospital Rostering — CSV-Driven CP-SAT Scheduler (Part 2 of 2)

This is **Part 2/2** of the README. It covers: the formal model, the constraint registry reference, how to extend the registry, outputs in detail (including *N* vs *F* semantics), troubleshooting/FAQ, and license/citation.

---

## Formal model (what the solver enforces)

**Decision variables**

- Binary variables `x[r,d,s] ∈ {0,1}` for each resident `r`, date `d` in the horizon, and shift `s ∈ {morning, afternoon}`.

**Coverage (per day/shift)**

- For each day `d`:
  - `Σ_r x[r,d,morning] = required_staff[morning]`
  - `Σ_r x[r,d,afternoon] = required_staff[afternoon]`

**At most one day-shift per resident/day**

- `Σ_s x[r,d,s] ≤ 1` for each `(r,d)`.

**Rest rule (no PM→AM)**

- For consecutive days `d` and `d+1`: `x[r,d,afternoon] + x[r,d+1,morning] ≤ 1`.

**Unavailability**

- If `d ∈ Unavailable[r]`, then `x[r,d,s] = 0` for `s ∈ {morning,afternoon}`.

**Night-shift exclusions**

- If `d ∈ Night[r]` (night starts on `d`), then day shifts on `d` **and** `d+1` are forbidden:
  - `x[r,d,s] = 0` and `x[r,d+1,s] = 0` for `s ∈ {morning,afternoon}` (when `d+1` is in the horizon).

**Strict fairness (±1)**

- Let `S = |Days| * (required_staff[morning] + required_staff[afternoon])` (total seats),
  `R = |Residents|`, `avg = S // R`, `max = avg + 1 if (S % R) > 0 else avg`.
- Each resident’s total load `T[r] = Σ_{d,s} x[r,d,s]` is bounded as:
  - `avg ≤ T[r] ≤ max`.

> Implementation detail: fairness uses a per-resident bounded `IntVar` linked to the sum of that resident’s assignments.

---

## Constraint registry reference

Constraints are toggled via `data/settings.csv` (rows with `type=constraint`) and implemented in `conditions_registry.py`.  
**Built-in keys** (active out of the box):

| Key                         | Effect (summary)                                                                 |
|----------------------------|-----------------------------------------------------------------------------------|
| `coverage_exact_required`  | Exact seats per day/shift match `required_staff_*`.                               |
| `one_shift_per_day`        | A resident may not work both `morning` and `afternoon` on the same day.          |
| `no_afternoon_then_morning`| Forbid `Afternoon(d)` followed by `Morning(d+1)` for the same resident.          |
| `enforce_unavailability`   | Forbid any day shift on dates listed in `unavailability` for that resident.      |
| `night_shift_exclusion`    | Forbid day shifts on night day `d` and day `d+1` for that resident.              |
| `fairness_strict_minmax`   | Balance totals within ±1 across residents (as defined above).                    |

**Optional ideas (scaffolded in the file as comments):**
- `max_total_shifts`, `max_shifts_per_week`, `max_afternoon_per_week`,
  `forbid_resident_shift`, `forbid_resident_dates`, `max_consecutive_work_days`, `min_free_days_per_week`.

> These optional constraints are **included as commented examples**. To use them, see “Extending the registry” below.

---

## Extending the registry (add/enable a new rule)

You can introduce new constraints or enable one of the scaffolded ones.

**A) Enabling one of the scaffolded (commented) examples**

1. Open `conditions_registry.py` and **uncomment** the function definition you want (inside the “NEW OPTIONAL CONSTRAINTS” block).
2. Add it to the `_REGISTRY` mapping (uncomment the corresponding line under “New optional constraints”).
3. Add a constraint row to `data/settings.csv` with `type=constraint`, `key=<your_key>`, `enabled=1`, and the JSON `params` if any.

**Example (`max_total_shifts`)**

- In `conditions_registry.py`, uncomment:

~~~python
def cond_max_total_shifts(model, x, residents, days, shifts, date_map, required_staff, params, context):
    limit = int(params.get("limit", 20))
    for r in residents:
        model.Add(sum(x[(r, d, s)] for d in days for s in shifts) <= limit)
~~~

- And in `_REGISTRY`, add:

~~~python
"max_total_shifts": cond_max_total_shifts,
~~~

- Then in `data/settings.csv` add a row:

~~~csv
"constraint","max_total_shifts",1,,"{""limit"":18}","Optional: total shifts per resident ≤ 18"
~~~

**B) Creating your own new rule**

- Follow the same pattern: implement `def cond_<name>(model, x, residents, days, shifts, date_map, required_staff, params, context)`, register it in `_REGISTRY`, and enable it via a `type=constraint` row.

---

## Outputs explained (files and semantics)

Every run creates a timestamped folder under `outputs/`, e.g., `outputs/run_20250901_113240/`.

### 1) `run_info.csv`
A compact log for reproducibility/debugging, including:
- Horizon, count of days/residents, coverage targets, flags (`USE_CSV`, `USE_EXTRA_CONDITIONS`), seed.
- CP-SAT `status` (`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, ...).
- `constraints_used_keys` and `constraints_used_params` (JSON).
- The raw `unavailability` and `night_shifts` used by the run.

> This file is **always** written, even if the model is infeasible.

### 2) `schedule_wide.csv`
A single “wide” table:
- **First 4 rows** are per-resident totals:
  - `Total morning`
  - `Total afternoon`
  - `Total free days` *(days with no M/A and not night-blocked)*
  - `Total free + unavailability` *(| Free ∪ (unavailability ∩ horizon) |)*
- **Then one row per date** with a single letter per resident column:
  - `M` = morning
  - `A` = afternoon
  - `N` = night day or next day (only if **no** M/A is assigned that day)
  - `F` = free (no M/A and not night-blocked)

> The script also prints a **resident-wise summary** and warns if it ever finds a day with both `M` and `A` for the same resident (which shouldn’t occur if `one_shift_per_day` is enabled).

### 3) `heatmap.png`
A color-coded matrix (residents × days). Encoding:
- **Unavailable** = red
- **Night exclusion** = orange
- **Free** = white
- **Morning** = blue
- **Afternoon** = yellow

---

## Night-shift semantics (clear and explicit)

- A **night shift listed on day `d`** blocks **both** day shifts (`morning`, `afternoon`) on:
  - the **same day** `d`, and
  - the **following day** `d+1` (if within horizon).
- In the **wide table**, a blocked day shows:
  - `N` **only if not assigned** to `M/A` (due to letter precedence).
- In the **heatmap**, *Night exclusion* is shown **only if not assigned** to `M/A` for that day.

This matches the operational rule “night runs from 20:00 of `d` to 08:00 of `d+1`, so no day shifts on either day.”

---

## Troubleshooting

**CSV quoting / doubled quotes (Excel)**
- The loader is tolerant: it accepts *JSON arrays* (even with doubled quotes), *semicolon/comma lists*, and *singletons*.  
  Example cells that all parse correctly:  
  - `[""2025-09-03"",""2025-09-10""]`  
  - `2025-09-03; 2025-09-10`  
  - `2025-09-03,2025-09-10`  
  - `2025-09-03`

**Model infeasible**
- Check for over-tight combinations (e.g., high coverage + many unavailability + night exclusions).
- Temporarily disable `fairness_strict_minmax` to test raw feasibility.
- Relax coverage (e.g., reduce `required_staff_*`) or reduce constraints (disable one at a time).
- Ensure `start_date`, `end_date`, and `weekdays_only` define the horizon you intend (e.g., weekends excluded may shrink capacity).



**“Why does one resident have fewer free days?”**
- *Fairness* equalizes **total shifts**, not **free days**.  
  If a resident has more unavailability or *N* blocks, they might end up with fewer `F` days while still having the same number of total assignments as peers (because coverage must be met each day).

---

## FAQ

**How do I include weekends?**  
Set `"weekdays_only"` to `false` in `data/settings.csv` (row with `type=setting, key=weekdays_only`).

**How do I make runs reproducible?**  
Set `"seed"` in `data/settings.csv`. The script forwards it to CP-SAT.

**Can I cap the number of afternoon shifts per week?**  
Yes—implement or enable an optional constraint like `max_afternoon_per_week` (see “Extending the registry”).

**How do I add a new resident quickly?**  
Add a row to `data/residents.csv`. If they have no constraints, set `unavailability="[]"` and `night_shifts="[]"`.

**Where do I see which constraints were applied?**  
Check `outputs/run_*/run_info.csv`: `constraints_used_keys` and `constraints_used_params`.

---

## Contributing

- Keep **data parsing** robust and forgiving; prefer small helper functions for coercion.
- Keep **constraints** modular (one function = one rule), and document parameters in the code docstring.
- When adding constraints, update this README with the new key and expected `params`.

---

## License & citation

**License:** MIT (recommended). Add a `LICENSE` file or replace this line with your chosen license.

**Cite / Acknowledge**
- Google OR-Tools (CP-SAT): https://developers.google.com/optimization

---

