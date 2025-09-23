#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from fitparse import FitFile
except Exception as e:  # pragma: no cover
    FitFile = None  # type: ignore


PARIS_TZ = ZoneInfo("Europe/Paris")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate Garmin watch snapshots into JSON and report missing days.")
    p.add_argument("--input", "-i", type=Path, default=Path.home() / "Data" / "smart_watch",
                   help="Path to the snapshots root directory (default: ~/Data/smart_watch)")
    p.add_argument("--out", "-o", type=Path, default=Path("dist"),
                   help="Output directory for dist files (default: dist)")
    p.add_argument("--tz", type=str, default="Europe/Paris", help="IANA timezone name for bucketing (default: Europe/Paris)")
    p.add_argument("--limit", type=int, default=0, help="For debugging: limit number of FIT files per category")
    return p.parse_args(argv)


def is_snapshot_dir(path: Path) -> bool:
    return path.is_dir() and any((path / sub).exists() for sub in ("Sleep", "Monitor", "Metrics"))


def list_snapshot_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    snaps = [p for p in sorted(root.iterdir()) if is_snapshot_dir(p)]
    return snaps


def load_fit(path: Path) -> Optional["FitFile"]:
    if FitFile is None:
        return None
    try:
        return FitFile(str(path))
    except Exception:
        return None


def ensure_dt(dt: Any) -> Optional[datetime]:
    if isinstance(dt, datetime):
        # FIT usually stores UTC; ensure timezone-aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def extract_monitoring_metrics(fit: "FitFile") -> Tuple[List[Tuple[datetime, int]], List[Tuple[datetime, int]]]:
    """Return (hr_samples, stress_samples) as lists of (utc_dt, value)."""
    hr: List[Tuple[datetime, int]] = []
    stress: List[Tuple[datetime, int]] = []
    try:
        for msg in fit.get_messages():
            name = getattr(msg, "name", "") or ""
            if name not in ("monitoring", "stress", "monitoring_info", "record"):
                continue
            fields = {d.name: d.value for d in msg}
            ts = ensure_dt(fields.get("timestamp"))
            if ts is None:
                continue
            # Heart rate can be on monitoring or record
            hr_val = fields.get("heart_rate")
            if isinstance(hr_val, int) and hr_val > 0:
                hr.append((ts, hr_val))
            # Stress can be on stress/monitoring messages
            for key in ("stress_level", "stress", "stress_level_value"):
                v = fields.get(key)
                if isinstance(v, int) and v >= 0:
                    stress.append((ts, v))
                    break
    except Exception:
        pass
    return hr, stress


def extract_sleep_sessions(fit: "FitFile") -> List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]]:
    """Return list of sleep sessions as (start_utc, end_utc, duration_sec).

    Tries multiple heuristics since FIT sleep schemas vary by device/firmware.
    """
    sessions: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]] = []
    totals: List[int] = []
    starts: List[datetime] = []
    ends: List[datetime] = []
    try:
        for msg in fit.get_messages():
            name = getattr(msg, "name", "") or ""
            fields = {d.name: d.value for d in msg}
            # Common summary message
            if name in ("sleep_summary", "sleep"):
                dur = fields.get("total_sleep_time") or fields.get("total_sleep_duration") or fields.get("sleep_time")
                if isinstance(dur, int) and dur > 0:
                    totals.append(dur)
                st = ensure_dt(fields.get("start_time")) or ensure_dt(fields.get("sleep_start_time"))
                en = ensure_dt(fields.get("end_time")) or ensure_dt(fields.get("sleep_end_time"))
                if st is not None:
                    starts.append(st)
                if en is not None:
                    ends.append(en)
            # Some devices expose intervals with duration
            for dur_key in ("duration", "total_sleep", "light_sleep_time", "deep_sleep_time", "rem_sleep_time"):
                v = fields.get(dur_key)
                if isinstance(v, int) and v > 0:
                    totals.append(int(v))
            # Events with start/end timestamps
            for st_key in ("start_time", "timestamp_start", "sleep_start_time"):
                st = ensure_dt(fields.get(st_key))
                if st is not None:
                    starts.append(st)
            for en_key in ("end_time", "timestamp_end", "sleep_end_time"):
                en = ensure_dt(fields.get(en_key))
                if en is not None:
                    ends.append(en)
    except Exception:
        pass

    # Heuristics to build sessions
    if starts or ends or totals:
        # Pair earliest start with latest end if any; fall back to None
        st = min(starts) if starts else None
        en = max(ends) if ends else None
        if st and en and en > st:
            sessions.append((st, en, int((en - st).total_seconds())))
        elif totals:
            # Use total duration only
            sessions.append((None, None, max(totals)))
    return sessions


def parse_snapshots(root: Path, limit: int = 0) -> Tuple[List[Tuple[datetime, int]], List[Tuple[datetime, int]], List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]]]:
    hr_all: List[Tuple[datetime, int]] = []
    stress_all: List[Tuple[datetime, int]] = []
    sleep_all: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]] = []

    snaps = list_snapshot_dirs(root)
    for snap in snaps:
        # Monitor → HR & stress
        mon_dir = snap / "Monitor"
        if mon_dir.exists():
            files = sorted([p for p in mon_dir.iterdir() if p.suffix.lower() == ".fit"])
            if limit:
                files = files[:limit]
            for f in files:
                fit = load_fit(f)
                if fit is None:
                    continue
                hr, st = extract_monitoring_metrics(fit)
                hr_all.extend(hr)
                stress_all.extend(st)

        # Sleep → sessions
        sl_dir = snap / "Sleep"
        if sl_dir.exists():
            files = sorted([p for p in sl_dir.iterdir() if p.suffix.lower() == ".fit"])
            if limit:
                files = files[:limit]
            for f in files:
                fit = load_fit(f)
                if fit is None:
                    continue
                sleep_all.extend(extract_sleep_sessions(fit))

        # Metrics → sometimes contains sleep summary and stress
        met_dir = snap / "Metrics"
        if met_dir.exists():
            files = sorted([p for p in met_dir.iterdir() if p.suffix.lower() == ".fit"])
            if limit:
                files = files[:limit]
            for f in files:
                fit = load_fit(f)
                if fit is None:
                    continue
                # Try extracting additional sleep durations
                sleep_all.extend(extract_sleep_sessions(fit))
                hr, st = extract_monitoring_metrics(fit)
                # Some metrics files may have resting HR; we ignore resting HR explicitly
                # and only keep samples where message contained instantaneous HR
                hr_all.extend(hr)
                stress_all.extend(st)

    # Deduplicate by timestamp for hr and stress
    def dedup_time_series(series: List[Tuple[datetime, int]]) -> List[Tuple[datetime, int]]:
        seen: Dict[int, int] = {}
        for ts, val in series:
            key = int(ts.replace(tzinfo=timezone.utc).timestamp())
            seen[key] = val  # last wins
        out = [(datetime.fromtimestamp(k, tz=timezone.utc), v) for k, v in seen.items()]
        out.sort(key=lambda x: x[0])
        return out

    hr_all = dedup_time_series(hr_all)
    stress_all = dedup_time_series(stress_all)

    # Merge sleep sessions that are duplicates by duration and close timestamps
    merged_sleep: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]] = []
    seen_sigs = set()
    for st, en, dur in sleep_all:
        sig = (int(dur or 0), int(st.timestamp()) if st else 0, int(en.timestamp()) if en else 0)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        merged_sleep.append((st, en, dur))
    return hr_all, stress_all, merged_sleep


def to_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def json_payload(hr: List[Tuple[datetime, int]], stress: List[Tuple[datetime, int]],
                 sleep: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]]) -> Dict[str, Any]:
    return {
        "hr_samples": [
            {"ts": to_iso(ts), "hr": val} for ts, val in hr
        ],
        "stress_samples": [
            {"ts": to_iso(ts), "stress": val} for ts, val in stress
        ],
        "sleep_sessions": [
            {
                "start": (to_iso(st) if isinstance(st, datetime) else None),
                "end": (to_iso(en) if isinstance(en, datetime) else None),
                "duration_sec": int(dur) if isinstance(dur, int) else None,
            }
            for st, en, dur in sleep
        ],
        "meta": {
            "tz": "Europe/Paris",
            "generated_at": to_iso(datetime.now(timezone.utc)),
        },
    }


def daterange(start: datetime, end: datetime) -> List[datetime]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def local_day_bounds(dt: datetime, tz: ZoneInfo) -> Tuple[datetime, datetime]:
    local = dt.astimezone(tz)
    start_local = datetime(local.year, local.month, local.day, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def missing_days_report(hr: List[Tuple[datetime, int]], stress: List[Tuple[datetime, int]],
                        sleep: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]], tz_name: str) -> str:
    tz = ZoneInfo(tz_name)

    # Build day sets
    def day_key(dt: datetime) -> str:
        return dt.astimezone(tz).strftime("%Y-%m-%d")

    if hr:
        min_dt = hr[0][0]
        max_dt = hr[-1][0]
    else:
        now = datetime.now(timezone.utc)
        min_dt = now - timedelta(days=1)
        max_dt = now

    # Expand range a bit using stress/sleep
    if stress:
        min_dt = min(min_dt, stress[0][0])
        max_dt = max(max_dt, stress[-1][0])
    if sleep:
        s_times = [st for st, _, _ in sleep if isinstance(st, datetime)] + [en for _, en, _ in sleep if isinstance(en, datetime)]
        if s_times:
            min_dt = min(min_dt, min(s_times))
            max_dt = max(max_dt, max(s_times))

    start_local = datetime.fromtimestamp(0, tz)  # placeholder, replaced below
    start_utc_localized = min_dt.astimezone(tz)
    start_local = datetime(start_utc_localized.year, start_utc_localized.month, start_utc_localized.day, tzinfo=tz)
    end_utc_localized = max_dt.astimezone(tz)
    end_local = datetime(end_utc_localized.year, end_utc_localized.month, end_utc_localized.day, tzinfo=tz)

    # Build presence sets
    hr_days = {day_key(ts) for ts, _ in hr}
    st_days = {day_key(ts) for ts, _ in stress}
    sl_days = set()
    for st, en, dur in sleep:
        if isinstance(st, datetime):
            sl_days.add(day_key(st))
        if isinstance(en, datetime):
            sl_days.add(day_key(en))
        # If only duration given, we can't map to a day

    # Iterate calendar days
    report_lines = []
    d = start_local
    while d <= end_local:
        day_str = d.strftime("%Y-%m-%d")
        if day_str not in hr_days:
            report_lines.append(f"Missing HR for {day_str}")
        if day_str not in st_days:
            report_lines.append(f"Missing Stress for {day_str}")
        if day_str not in sl_days:
            report_lines.append(f"Missing Sleep for {day_str}")
        d = d + timedelta(days=1)

    return "\n".join(report_lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    tz = ZoneInfo(args.tz)

    # Parse
    hr, stress, sleep = parse_snapshots(args.input, limit=args.limit)

    # Prepare output dir
    args.out.mkdir(parents=True, exist_ok=True)

    # Write JSON
    payload = json_payload(hr, stress, sleep)
    out_json = args.out / "data.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    # Write missing days to stdout
    report = missing_days_report(hr, stress, sleep, args.tz)
    if report:
        print(report)
    else:
        print("No missing days detected in parsed series.")

    # Write minimal index.html if not present (optional convenience)
    index_path = args.out / "index.html"
    if not index_path.exists():
        index_path.write_text(_DEFAULT_INDEX_HTML, encoding="utf-8")

    print(f"Wrote {out_json} and {index_path}")
    return 0


_DEFAULT_INDEX_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Garmin Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem; }
    header { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; margin-bottom: 1rem; }
    .row { display: grid; grid-template-columns: 1fr; gap: 1.5rem; }
    @media (min-width: 960px) { .row { grid-template-columns: 1fr 1fr; } }
    .card { padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
    label { margin-right: .25rem; }
    .controls { display: flex; gap: .75rem; align-items: center; }
    .seg { display: inline-flex; border: 1px solid #bbb; border-radius: 6px; overflow: hidden; }
    .seg button { border: 0; background: #f6f6f6; padding: .4rem .6rem; cursor: pointer; }
    .seg button.active { background: #333; color: #fff; }
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/luxon@3\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1\"></script>
</head>
<body>
  <h1>Garmin Dashboard</h1>
  <header>
    <div class=\"controls\">
      <label for=\"from\">From</label>
      <input type=\"date\" id=\"from\" />
      <label for=\"to\">To</label>
      <input type=\"date\" id=\"to\" />
    </div>
    <div class=\"controls\">
      <span>Bucket:</span>
      <div class=\"seg\" id=\"bucket\">
        <button data-b=\"day\" class=\"active\">Day</button>
        <button data-b=\"week\">Week</button>
        <button data-b=\"month\">Month</button>
      </div>
    </div>
  </header>
  <div class=\"row\">
    <div class=\"card\"><h3>Average Sleep (hours)</h3><canvas id=\"sleepChart\"></canvas></div>
    <div class=\"card\"><h3>Average Stress</h3><canvas id=\"stressChart\"></canvas></div>
    <div class=\"card\"><h3>Average BPM</h3><canvas id=\"hrChart\"></canvas></div>
  </div>

  <script>
    const TZ = 'Europe/Paris';
    const DateTime = luxon.DateTime;

    function parseISOZ(s) { return DateTime.fromISO(s, { zone: 'utc' }); }

    function bucketKey(dt, bucket) {
      const l = dt.setZone(TZ);
      if (bucket === 'day') return l.toFormat('yyyy-LL-dd');
      if (bucket === 'week') {
        // ISO week, Monday as first day
        const weekStart = l.startOf('week');
        return weekStart.toFormat('kkkk-\'W\'WW');
      }
      if (bucket === 'month') return l.toFormat('yyyy-LL');
      return l.toFormat('yyyy-LL-dd');
    }

    function toDayKey(dt) { return dt.setZone(TZ).toFormat('yyyy-LL-dd'); }

    function aggregate(data, from, to, bucket) {
      // data: { hr_samples, stress_samples, sleep_sessions }
      const fromDT = from ? DateTime.fromISO(from, { zone: TZ }).startOf('day') : null;
      const toDT = to ? DateTime.fromISO(to, { zone: TZ }).endOf('day') : null;

      // HR and Stress: average of samples in bucket
      const aggSeries = (samples, field) => {
        const map = new Map();
        for (const s of samples) {
          const ts = parseISOZ(s.ts);
          if (fromDT && ts < fromDT) continue;
          if (toDT && ts > toDT) continue;
          const k = bucketKey(ts, bucket);
          const e = map.get(k) || { sum: 0, n: 0, ts: ts };
          e.sum += s[field]; e.n += 1; e.ts = ts;
          map.set(k, e);
        }
        const points = [...map.entries()].map(([k, v]) => ({ x: keyToCenterTS(k, bucket), y: v.n ? v.sum / v.n : null }));
        points.sort((a, b) => a.x - b.x);
        return points;
      };

      // Sleep: convert sessions to per-day duration, then average per bucket (days in bucket)
      const dayDur = new Map(); // dayKey -> minutes
      for (const s of data.sleep_sessions) {
        let start = s.start ? parseISOZ(s.start).setZone(TZ) : null;
        let end = s.end ? parseISOZ(s.end).setZone(TZ) : null;
        let durSec = typeof s.duration_sec === 'number' ? s.duration_sec : null;
        if (!start && !end && durSec == null) continue;
        if (!start && end && durSec != null) start = end.minus({ seconds: durSec });
        if (start && !end && durSec != null) end = start.plus({ seconds: durSec });
        if (!start || !end) continue;
        // Split across days
        let cur = start.startOf('day');
        while (cur < end) {
          const next = cur.plus({ days: 1 });
          const segStart = start > cur ? start : cur;
          const segEnd = end < next ? end : next;
          const minutes = Math.max(0, segEnd.diff(segStart, 'minutes').minutes);
          if (minutes > 0) {
            if ((!fromDT || segEnd >= fromDT) && (!toDT || segStart <= toDT)) {
              const k = cur.toFormat('yyyy-LL-dd');
              dayDur.set(k, (dayDur.get(k) || 0) + minutes);
            }
          }
          cur = next;
        }
      }
      // Average per bucket (by days present in bucket)
      const sleepMap = new Map(); // bucketKey -> { sumMin, days }
      for (const [day, min] of dayDur.entries()) {
        const dt = DateTime.fromFormat(day, 'yyyy-LL-dd', { zone: TZ });
        if (fromDT && dt.endOf('day') < fromDT) continue;
        if (toDT && dt.startOf('day') > toDT) continue;
        const k = bucketKey(dt, bucket);
        const e = sleepMap.get(k) || { sum: 0, days: 0 };
        e.sum += min; e.days += 1; sleepMap.set(k, e);
      }
      const sleepPts = [...sleepMap.entries()].map(([k, v]) => ({ x: keyToCenterTS(k, bucket), y: v.days ? (v.sum / v.days) / 60.0 : null }));
      sleepPts.sort((a, b) => a.x - b.x);

      return {
        hr: aggSeries(data.hr_samples, 'hr'),
        stress: aggSeries(data.stress_samples, 'stress'),
        sleep: sleepPts,
      };
    }

    function keyToCenterTS(k, bucket) {
      if (bucket === 'day') return DateTime.fromFormat(k, 'yyyy-LL-dd', { zone: TZ }).plus({ hours: 12 }).toJSDate();
      if (bucket === 'week') {
        const dt = DateTime.fromFormat(k, "kkkk-'W'WW", { zone: TZ }).startOf('week');
        return dt.plus({ days: 3 }).toJSDate(); // mid-week
      }
      if (bucket === 'month') return DateTime.fromFormat(k, 'yyyy-LL', { zone: TZ }).plus({ days: 14 }).toJSDate();
      return DateTime.now().toJSDate();
    }

    async function loadData() {
      const resp = await fetch('data.json');
      if (!resp.ok) throw new Error('Failed to load data.json');
      return await resp.json();
    }

    function newLineChart(ctx, label, color) {
      return new Chart(ctx, {
        type: 'line',
        data: { datasets: [{ label, data: [], borderColor: color, pointRadius: 0, tension: 0.2 }] },
        options: {
          responsive: true,
          scales: {
            x: { type: 'time', time: { unit: 'day' } },
            y: { beginAtZero: true }
          },
          plugins: { legend: { display: false } }
        }
      });
    }

    (async () => {
      const data = await loadData();
      const sleepChart = newLineChart(document.getElementById('sleepChart'), 'Sleep (hours)', '#3b82f6');
      const stressChart = newLineChart(document.getElementById('stressChart'), 'Stress', '#ef4444');
      const hrChart = newLineChart(document.getElementById('hrChart'), 'HR (bpm)', '#10b981');

      // Default range based on data
      const allTs = [
        ...data.hr_samples.map(s => parseISOZ(s.ts)),
        ...data.stress_samples.map(s => parseISOZ(s.ts)),
        ...data.sleep_sessions.flatMap(s => [s.start && parseISOZ(s.start), s.end && parseISOZ(s.end)].filter(Boolean)),
      ];
      const minTs = allTs.length ? DateTime.min(...allTs) : DateTime.now().minus({ months: 1 });
      const maxTs = allTs.length ? DateTime.max(...allTs) : DateTime.now();
      document.getElementById('from').value = minTs.setZone(TZ).toFormat('yyyy-LL-dd');
      document.getElementById('to').value = maxTs.setZone(TZ).toFormat('yyyy-LL-dd');

      let bucket = 'day';

      function refresh() {
        const from = document.getElementById('from').value || null;
        const to = document.getElementById('to').value || null;
        const agg = aggregate(data, from, to, bucket);
        sleepChart.data.datasets[0].data = agg.sleep;
        stressChart.data.datasets[0].data = agg.stress;
        hrChart.data.datasets[0].data = agg.hr;
        sleepChart.update();
        stressChart.update();
        hrChart.update();
      }

      document.getElementById('from').addEventListener('change', refresh);
      document.getElementById('to').addEventListener('change', refresh);
      document.getElementById('bucket').addEventListener('click', (e) => {
        const b = e.target?.dataset?.b; if (!b) return;
        bucket = b;
        for (const btn of e.currentTarget.querySelectorAll('button')) btn.classList.toggle('active', btn.dataset.b === b);
        refresh();
      });

      refresh();
    })().catch(err => {
      console.error(err);
      alert('Failed to initialize dashboard: ' + err.message);
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

