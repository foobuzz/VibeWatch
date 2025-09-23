In the folder `~/Data/smart_watch` are multiple subfolders identified by dates (or partial dates). Each of those is an entire copy of the filesystem of my Garmin watch -- what I'm calling a "snapshot".

Your task is to build a static site generator taking in input a collection of those "snapshots" (basically, the `~/Data/smart_watch` directory), and creating a single HTML file being a dashboard of the data.

## Dashboard details

The dashboard will consists in charts showing data over time, for the following dimensions:

 - Average sleep duration over time
 - Average stress level over time (Garmin stress metric)
 - Average BPM over time (average of all HR samples)

For each chart, the user is able to select a specific time frame, as well as zoom/unzoom over the time-period of what a "dot" is (aggregation bucket):

 - Month
 - Week
 - Day

Time semantics:
- Day = local midnight to midnight, Europe/Paris timezone
- Week = Monday–Sunday (ISO-like week starting Monday)
- Month = calendar month

Each metric is aggregated per selected bucket ("per-dot"), not necessarily per day.

## Implementation details

There are 2 programs to build:

 1. The static site generator itself, which will consist in producing a compact JSON file containing all the data exploitable by the "front-end". => Use Python for this part.
 2. The "front-end", which is a single HTML page containing JavaScript that loads the compact JSON file, and dynamically uses the data inside to power the charts controlled by the user. => Use native JavaScript in the HTML page; third‑party libraries can be loaded via `<script>` from a CDN.

Output layout:
- Emit `dist/data.json` (compact dataset) and `dist/index.html` (single dashboard page) as the build artefacts.

Data discovery and consolidation:
- The generator must introspect the Garmin watch snapshots to find and parse the relevant data for sleep, stress, and heart rate (file formats may include FIT/JSON/CSV within the snapshots).
- Combine data from all snapshots, de‑duplicate overlapping records, and build continuous time series for the three metrics.

Missing timeframe detection (stdout):
- Detect gaps by inspecting the parsed time series (not folder names). Print to stdout any calendar days within the overall min→max date range that lack data for a metric. Prefer a per‑metric listing (Sleep/Stress/HR) of missing days.

Libraries and tooling:
- Python side may use external libraries installed with the `uv` package manager.
- Front‑end may use CDN libraries via `<script>` tags; keep dependencies minimal where practical.

UI controls:
- Provide date pickers for selecting the time window and buttons/toggles for Day/Week/Month aggregation.
