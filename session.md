# Session Summary – Garmin Snapshot Dashboard

## Overview
- Built a static site generator (Python) + single‑page dashboard (HTML/JS) to aggregate Garmin watch snapshots and visualize:
  - Average sleep duration
  - Average stress level
  - Average BPM
- Aggregations support Day/Week/Month with Europe/Paris time semantics (Mon–Sun weeks, calendar months).

## Data Discovery and Parsing
- Input: snapshot folders under `smart_watch/` each mirroring the watch filesystem.
- Sources used:
  - HR & Stress: `Monitor/*.FIT` (monitoring/record + stress_level messages)
  - Sleep: `Sleep/*.fit` (sleep_level timeline, plus event start/stop)
  - Metrics: `Metrics/*.fit` (aux, rarely used for sleep)
- FIT parsing backends: `fitparse` and `fitdecode` (fallbacks).
- Key logic:
  - Heart rate: reconstruct timestamps when only `timestamp_16` is present using last full timestamp.
  - Stress: parse `stress_level` messages (`stress_level_time` + `stress_level_value`), ignore invalid values (e.g., `-1`).
  - Sleep:
    - Primary: pair explicit sleep `event` messages (event_type start/stop, event code 74) to form sessions.
    - Fallback: derive sessions from `sleep_level` timeline (string levels → asleep/awake), with 30‑min gap rule.
    - Merge overlapping/adjacent sessions across files with 10‑min tolerance.

## Missing Data Detection
- Computes presence per calendar day (Europe/Paris) for HR, Stress, Sleep.
- Groups consecutive missing days into spans and prints summary to stdout, e.g.:
  - `Missing HR from YYYY‑MM‑DD to YYYY‑MM‑DD`
- Also embeds spans in JSON under `missing_spans` for front‑end annotations.

## Daily Aggregation (Generator)
- Emits per‑day aggregates in `data.json` (Europe/Paris):
  - `daily.hr`: `[ {day: YYYY‑MM‑DD, avg: number} ]`
  - `daily.stress`: `[ {day: YYYY‑MM‑DD, avg: number} ]`
  - `daily.sleep`: `[ {day: YYYY‑MM‑DD, hours: number} ]`
- Front‑end uses these for fast Day/Week/Month bucketing.

## Output
- `dist/data.json`: compact data with raw series (for reference), `missing_spans`, and `daily` aggregates.
- `dist/index.html`: dashboard page (copied from `templates/index.html` if missing).

## Front‑End (Dashboard)
- Libraries: Chart.js, Luxon adapter, annotation and zoom plugins (via CDN).
- Features:
  - Date range pickers + Day/Week/Month toggle.
  - Visible point markers (better tooltip targeting).
  - Loader overlay while re‑aggregating.
  - Drag to select span (x‑axis); syncs inputs and data; Reset button clears zoom and restores full range.
  - Shaded annotations for missing spans per metric.
- Performance:
  - Uses `daily.*` series for bucketing; no per‑sample ISO parsing on refresh.
  - Default date range computed from union of `daily` coverage.

## CLI (Generator)
- `uv run -- python garmin_gen.py --input ./smart_watch --out dist [--limit N] [--tz Europe/Paris]`
- Progress logs:
  - Snapshot discovery
  - Per‑category file counts and ~10 progress ticks
  - Parsed counts summary
  - Missing spans + daily aggregates

## File Layout
- `garmin_gen.py` – generator CLI and FIT parsing/aggregation.
- `pyproject.toml` – uv project config; deps: `fitparse`, `fitdecode`.
- `templates/index.html` – canonical dashboard HTML (copied to `dist/` if missing).
- `dist/index.html` – the served dashboard.

## Known Notes / Assumptions
- Stress during sleep: Garmin often doesn’t compute it; “missing stress” can exceed “missing sleep”.
- Daily aggregation uses Europe/Paris (DST aware) and averages only over days with data in the bucket.
- Raw `hr_samples`/`stress_samples` retained in JSON for future features; can be pruned to shrink payload.

## How to Run
1. Generate data (full):
   - `uv run -- python garmin_gen.py --input ./smart_watch --out dist`
2. Serve dashboard:
   - `python3 -m http.server -d dist 8000` → open http://localhost:8000

## Next Ideas
- Option to switch missing detection scope between union / per‑metric / intersection.
- Optional smoothing/downsampling for visuals.
- Export CSV for daily aggregates.
- Tests for FIT parsing edge cases and DST transitions.

