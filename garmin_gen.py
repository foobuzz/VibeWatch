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
except Exception:
    FitFile = None  # type: ignore

try:
    import fitdecode  # type: ignore
    from fitdecode.records import FitDataMessage  # type: ignore
except Exception:
    fitdecode = None  # type: ignore
    FitDataMessage = None  # type: ignore


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


def iter_fit_messages(path: Path):
    """Yield (name, fields_dict) from a FIT file using available backends.

    Tries fitparse first; if unavailable or fails, tries fitdecode.
    """
    # Backend 1: fitparse
    if FitFile is not None:
        try:
            fit = FitFile(str(path))
            for msg in fit.get_messages():
                name = getattr(msg, "name", "") or ""
                fields = {d.name: d.value for d in msg}
                yield name, fields
            return
        except Exception:
            pass
    # Backend 2: fitdecode
    if fitdecode is not None:
        try:
            with fitdecode.FitReader(str(path)) as fr:
                for frame in fr:
                    if isinstance(frame, FitDataMessage):
                        name = frame.name or ""
                        fields = {}
                        for f in frame.fields:
                            try:
                                fields[f.name] = f.value
                            except Exception:
                                pass
                        yield name, fields
            return
        except Exception:
            pass
    # If neither works, yield nothing
    return


def iter_fit_messages_fitdecode(path: Path):
    if fitdecode is None:
        return
    try:
        with fitdecode.FitReader(str(path)) as fr:
            for frame in fr:
                if isinstance(frame, FitDataMessage):
                    name = frame.name or ""
                    fields = {}
                    for f in frame.fields:
                        try:
                            fields[f.name] = f.value
                        except Exception:
                            pass
                    yield name, fields
    except Exception:
        return


def ensure_dt(dt: Any) -> Optional[datetime]:
    if isinstance(dt, datetime):
        # FIT usually stores UTC; ensure timezone-aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _reconstruct_ts16(last_full: Optional[datetime], ts16_val: Any) -> Optional[datetime]:
    # If ts16 is already datetime, use it
    if isinstance(ts16_val, datetime):
        return ensure_dt(ts16_val)
    if last_full is None:
        return None
    try:
        ts16 = int(ts16_val) & 0xFFFF
    except Exception:
        return None
    base = int(last_full.replace(tzinfo=timezone.utc).timestamp())
    candidate = (base & ~0xFFFF) | ts16
    # Adjust for wrap to be closest to base
    if candidate < base - 0x8000:
        candidate += 0x10000
    elif candidate > base + 0x8000:
        candidate -= 0x10000
    return datetime.fromtimestamp(candidate, tz=timezone.utc)


def extract_monitoring_metrics_from_iter(messages) -> Tuple[List[Tuple[datetime, int]], List[Tuple[datetime, int]]]:
    """Return (hr_samples, stress_samples) as lists of (utc_dt, value)."""
    hr: List[Tuple[datetime, int]] = []
    stress: List[Tuple[datetime, int]] = []
    last_full_ts: Optional[datetime] = None
    try:
        for name, fields in messages:
            # Track base timestamp when present
            base_ts = ensure_dt(fields.get("timestamp"))
            if base_ts is not None:
                last_full_ts = base_ts

            if name in ("monitoring", "record", "monitoring_a"):
                # HR may come with only timestamp_16
                ts = base_ts
                if ts is None:
                    ts16 = fields.get("timestamp_16")
                    if ts16 is not None:
                        ts = _reconstruct_ts16(last_full_ts, ts16)
                hr_val = fields.get("heart_rate")
                if ts is not None and isinstance(hr_val, int) and hr_val > 0:
                    hr.append((ts, hr_val))

            # Stress can be in dedicated message with its own time
            if name == "stress_level":
                s_ts = ensure_dt(fields.get("stress_level_time"))
                s_val = fields.get("stress_level_value")
                if s_ts is not None and isinstance(s_val, int) and s_val >= 0:
                    stress.append((s_ts, s_val))
                continue

            # Or come as field on other messages
            for key in ("stress_level", "stress", "stress_level_value"):
                v = fields.get(key)
                ts = base_ts
                if ts is None:
                    ts16 = fields.get("timestamp_16")
                    if ts16 is not None:
                        ts = _reconstruct_ts16(last_full_ts, ts16)
                if ts is not None and isinstance(v, int) and v >= 0:
                    stress.append((ts, v))
                    break
    except Exception:
        pass
    return hr, stress


def extract_sleep_sessions_from_iter(messages) -> List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]]:
    """Return list of sleep sessions as (start_utc, end_utc, duration_sec).

    Tries multiple heuristics since FIT sleep schemas vary by device/firmware.
    """
    sessions: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]] = []
    totals: List[int] = []
    starts: List[datetime] = []
    ends: List[datetime] = []
    try:
        pending_start: Optional[datetime] = None
        timeline: List[Tuple[datetime, int]] = []
        for name, fields in messages:
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
            # Event-based sessions (explicit start/stop in Sleep files)
            if name == "event":
                ts = ensure_dt(fields.get("timestamp"))
                etype = fields.get("event_type")
                ev = fields.get("event")
                if ts is not None and etype in ("start", "stop"):
                    # Many devices use event code 74 for sleep in Sleep FITs
                    if isinstance(ev, int) and ev not in (74,):
                        pass  # not a sleep event
                    else:
                        if etype == "start":
                            pending_start = ts
                        elif etype == "stop" and pending_start is not None:
                            if ts > pending_start:
                                sessions.append((pending_start, ts, int((ts - pending_start).total_seconds())))
                            pending_start = None

            # Sleep level timeline (derive sessions)
            if name == "sleep_level":
                ts = ensure_dt(fields.get("timestamp"))
                lvl = fields.get("sleep_level")
                if ts is not None and isinstance(lvl, int):
                    timeline.append((ts, lvl))
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

        # Note: actual session derivation is done globally in parse_snapshots
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


def _build_sleep_sessions_from_timeline(timeline: List[Tuple[datetime, int]]) -> List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]]:
    sessions: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]] = []
    if not timeline:
        return sessions
    timeline.sort(key=lambda x: x[0])
    in_sleep = False
    cur_start: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    for ts, lvl in timeline:
        if in_sleep:
            if last_ts and (ts - last_ts).total_seconds() > 60 * 30:  # 30+ min gap ends session
                if cur_start and last_ts and last_ts > cur_start:
                    sessions.append((cur_start, last_ts, int((last_ts - cur_start).total_seconds())))
                in_sleep = False
                cur_start = None
        if isinstance(lvl, int) and lvl >= 1:
            if not in_sleep:
                in_sleep = True
                cur_start = ts
        else:
            if in_sleep:
                if cur_start and last_ts and last_ts > cur_start:
                    sessions.append((cur_start, last_ts, int((last_ts - cur_start).total_seconds())))
                in_sleep = False
                cur_start = None
        last_ts = ts
    if in_sleep and cur_start and last_ts and last_ts > cur_start:
        sessions.append((cur_start, last_ts, int((last_ts - cur_start).total_seconds())))
    return sessions


def _extract_sleep_timeline_from_iter(messages) -> List[Tuple[datetime, int]]:
    timeline: List[Tuple[datetime, int]] = []
    try:
        for name, fields in messages:
            if name == "sleep_level":
                ts = ensure_dt(fields.get("timestamp"))
                raw = fields.get("sleep_level")
                lvl_val: Optional[int] = None
                if isinstance(raw, int):
                    lvl_val = raw
                elif isinstance(raw, str):
                    lvl_val = 0 if raw.lower() == "awake" else 1  # treat non-awake as sleep
                if ts is not None and isinstance(lvl_val, int):
                    timeline.append((ts, lvl_val))
    except Exception:
        pass
    return timeline


def parse_snapshots(root: Path, limit: int = 0) -> Tuple[List[Tuple[datetime, int]], List[Tuple[datetime, int]], List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]]]:
    hr_all: List[Tuple[datetime, int]] = []
    stress_all: List[Tuple[datetime, int]] = []
    sleep_all: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]] = []
    sleep_timeline: List[Tuple[datetime, int]] = []

    snaps = list_snapshot_dirs(root)
    print(f"Found {len(snaps)} snapshot(s) in {root}", flush=True)
    for snap in snaps:
        print(f"Snapshot {snap.name}", flush=True)
        def iter_with_progress(files: List[Path], label: str):
            total = len(files)
            if total == 0:
                return
            print(f"  {label}: {total} file(s){' (limited)' if limit and total>limit else ''}", flush=True)
            step = max(total // 10, 1)
            for i, f in enumerate(files, start=1):
                yield i, total, f
                if i % step == 0 or i == total:
                    print(f"  {label}: {i}/{total}", flush=True)
        # Monitor → HR & stress
        mon_dir = snap / "Monitor"
        if mon_dir.exists():
            files = sorted([p for p in mon_dir.iterdir() if p.suffix.lower() == ".fit"])
            if limit:
                files = files[:limit]
            for i, total, f in iter_with_progress(files, "Monitor"):
                msgs = iter_fit_messages(f)
                hr, st = extract_monitoring_metrics_from_iter(msgs)
                hr_all.extend(hr)
                stress_all.extend(st)

        # Sleep → sessions
        sl_dir = snap / "Sleep"
        if sl_dir.exists():
            files = sorted([p for p in sl_dir.iterdir() if p.suffix.lower() == ".fit"])
            if limit:
                files = files[:limit]
            for i, total, f in iter_with_progress(files, "Sleep"):
                # Prefer fitdecode for Sleep files to access sleep_level timeline
                msgs = iter_fit_messages_fitdecode(f)
                sleep_all.extend(extract_sleep_sessions_from_iter(msgs))
                # Collect fine-grained sleep levels for global session derivation
                msgs2 = iter_fit_messages_fitdecode(f)
                sleep_timeline.extend(_extract_sleep_timeline_from_iter(msgs2))

        # Metrics → sometimes contains sleep summary and stress
        met_dir = snap / "Metrics"
        if met_dir.exists():
            files = sorted([p for p in met_dir.iterdir() if p.suffix.lower() == ".fit"])
            if limit:
                files = files[:limit]
            for i, total, f in iter_with_progress(files, "Metrics"):
                msgs = iter_fit_messages(f)
                # Try extracting additional sleep durations
                sleep_all.extend(extract_sleep_sessions_from_iter(msgs))
                # Re-iterate for hr/stress because generator consumed
                msgs2 = iter_fit_messages(f)
                hr, st = extract_monitoring_metrics_from_iter(msgs2)
                # Some metrics files may have resting HR; we ignore resting HR explicitly
                # and only keep samples where message contained instantaneous HR
                hr_all.extend(hr)
                stress_all.extend(st)
                # Also collect any sleep levels present in metrics (rare)
                msgs3 = iter_fit_messages_fitdecode(f)
                sleep_timeline.extend(_extract_sleep_timeline_from_iter(msgs3))
        print(f"  Done {snap.name}", flush=True)

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

    # Build sessions from the global sleep timeline only if no event-based sessions were found
    if not sleep_all and sleep_timeline:
        sleep_all.extend(_build_sleep_sessions_from_timeline(sleep_timeline))

    # Merge overlapping/adjacent sessions to avoid double-counting across files/snapshots
    def merge_sessions(sessions: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]]) -> List[Tuple[datetime, datetime, int]]:
        items: List[Tuple[datetime, datetime]] = []
        for st, en, _ in sessions:
            if isinstance(st, datetime) and isinstance(en, datetime) and en > st:
                items.append((st.astimezone(timezone.utc), en.astimezone(timezone.utc)))
        if not items:
            return []
        items.sort(key=lambda x: x[0])
        merged: List[Tuple[datetime, datetime]] = []
        cur_s, cur_e = items[0]
        tol = timedelta(minutes=10)
        for s, e in items[1:]:
            if s <= cur_e + tol:  # overlap or close gap
                if e > cur_e:
                    cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return [(s, e, int((e - s).total_seconds())) for s, e in merged]

    # Merge duplicates and overlaps
    merged_sleep = merge_sessions(sleep_all)
    print(f"Parsed: HR samples={len(hr_all)}, Stress samples={len(stress_all)}, Sleep sessions={len(merged_sleep)} (timeline points={len(sleep_timeline)})", flush=True)
    return hr_all, stress_all, merged_sleep


def to_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def json_payload(hr: List[Tuple[datetime, int]], stress: List[Tuple[datetime, int]],
                 sleep: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]],
                 missing_spans: Dict[str, List[Dict[str, str]]],
                 daily: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
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
        "missing_spans": missing_spans,
        "daily": daily,
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

def _group_missing_days(day_set: set, start_local: datetime, end_local: datetime) -> List[Tuple[str, str]]:
    spans: List[Tuple[str, str]] = []
    cur_start: Optional[datetime] = None
    cur_end: Optional[datetime] = None
    d = start_local
    while d <= end_local:
        day_str = d.strftime("%Y-%m-%d")
        missing = day_str not in day_set
        if missing:
            if cur_start is None:
                cur_start = d
                cur_end = d
            else:
                cur_end = d
        else:
            if cur_start is not None and cur_end is not None:
                spans.append((cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
                cur_start = None
                cur_end = None
        d = d + timedelta(days=1)
    if cur_start is not None and cur_end is not None:
        spans.append((cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
    return spans


def compute_missing(hr: List[Tuple[datetime, int]], stress: List[Tuple[datetime, int]],
                    sleep: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]], tz_name: str) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
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

    # Group spans per metric
    hr_spans = _group_missing_days(hr_days, start_local, end_local)
    st_spans = _group_missing_days(st_days, start_local, end_local)
    sl_spans = _group_missing_days(sl_days, start_local, end_local)

    lines = []
    for a, b in hr_spans:
        lines.append(f"Missing HR from {a} to {b}")
    for a, b in st_spans:
        lines.append(f"Missing Stress from {a} to {b}")
    for a, b in sl_spans:
        lines.append(f"Missing Sleep from {a} to {b}")

    spans_json = {
        "hr": [{"from": a, "to": b} for a, b in hr_spans],
        "stress": [{"from": a, "to": b} for a, b in st_spans],
        "sleep": [{"from": a, "to": b} for a, b in sl_spans],
    }
    return "\n".join(lines), spans_json


def compute_daily(hr: List[Tuple[datetime, int]], stress: List[Tuple[datetime, int]],
                  sleep: List[Tuple[Optional[datetime], Optional[datetime], Optional[int]]], tz_name: str) -> Dict[str, List[Dict[str, Any]]]:
    tz = ZoneInfo(tz_name)

    def day_str(dt: datetime) -> str:
        return dt.astimezone(tz).strftime("%Y-%m-%d")

    # HR/stress daily averages
    hr_acc: Dict[str, Tuple[int, int]] = {}  # day -> (sum, count)
    for ts, val in hr:
        d = day_str(ts)
        s, c = hr_acc.get(d, (0, 0))
        hr_acc[d] = (s + val, c + 1)
    stress_acc: Dict[str, Tuple[int, int]] = {}
    for ts, val in stress:
        d = day_str(ts)
        s, c = stress_acc.get(d, (0, 0))
        stress_acc[d] = (s + val, c + 1)

    hr_daily = [{"day": d, "avg": (s / c) if c else None} for d, (s, c) in hr_acc.items()]
    stress_daily = [{"day": d, "avg": (s / c) if c else None} for d, (s, c) in stress_acc.items()]
    hr_daily.sort(key=lambda x: x["day"])
    stress_daily.sort(key=lambda x: x["day"])

    # Sleep: per-day total minutes (split sessions across days in local time)
    sleep_min: Dict[str, float] = {}
    for st, en, dur in sleep:
        if isinstance(st, datetime) and isinstance(en, datetime):
            start = st.astimezone(tz)
            end = en.astimezone(tz)
            cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
            while cur < end:
                next_day = cur + timedelta(days=1)
                seg_start = max(start, cur)
                seg_end = min(end, next_day)
                minutes = max(0.0, (seg_end - seg_start).total_seconds() / 60.0)
                if minutes > 0:
                    key = cur.strftime("%Y-%m-%d")
                    sleep_min[key] = sleep_min.get(key, 0.0) + minutes
                cur = next_day
        elif dur and isinstance(dur, int):
            # Duration without timestamps; cannot attribute to a day, skip
            pass

    sleep_daily = [{"day": d, "hours": mins / 60.0} for d, mins in sleep_min.items()]
    sleep_daily.sort(key=lambda x: x["day"])

    return {
        "hr": hr_daily,
        "stress": stress_daily,
        "sleep": sleep_daily,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    tz = ZoneInfo(args.tz)

    # Parse
    hr, stress, sleep = parse_snapshots(args.input, limit=args.limit)

    # Prepare output dir
    args.out.mkdir(parents=True, exist_ok=True)

    # Write JSON
    print("Analyzing missing days ...", flush=True)
    report, spans_json = compute_missing(hr, stress, sleep, args.tz)
    print("Computing daily aggregates ...", flush=True)
    daily = compute_daily(hr, stress, sleep, args.tz)
    print("Writing dist/data.json ...", flush=True)
    payload = json_payload(hr, stress, sleep, spans_json, daily)
    out_json = args.out / "data.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    # Write missing days to stdout
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
  <script src=\"https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2\"></script>
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
      <button id=\"resetRange\" title=\"Reset date range\">Reset</button>
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
    if (window['chartjs-plugin-zoom']) Chart.register(window['chartjs-plugin-zoom']);

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
      // Use precomputed daily series; bucket to day/week/month with fast grouping
      const fromDT = from ? DateTime.fromISO(from, { zone: TZ }).startOf('day') : null;
      const toDT = to ? DateTime.fromISO(to, { zone: TZ }).endOf('day') : null;
      const daily = data.daily || { hr: [], stress: [], sleep: [] };

      const aggDaily = (arr, valueKey) => {
        const map = new Map();
        for (const row of arr) {
          const dt = DateTime.fromFormat(row.day, 'yyyy-LL-dd', { zone: TZ });
          if (fromDT && dt < fromDT) continue;
          if (toDT && dt > toDT) continue;
          const k = bucketKey(dt, bucket);
          const e = map.get(k) || { sum: 0, n: 0 };
          const v = row[valueKey];
          if (typeof v === 'number') { e.sum += v; e.n += 1; }
          map.set(k, e);
        }
        const pts = [...map.entries()].map(([k, v]) => ({ x: keyToCenterTS(k, bucket), y: v.n ? v.sum / v.n : null }));
        pts.sort((a, b) => a.x - b.x);
        return pts;
      };

      return {
        hr: aggDaily(daily.hr, 'avg'),
        stress: aggDaily(daily.stress, 'avg'),
        sleep: aggDaily(daily.sleep, 'hours'),
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
        data: { datasets: [{
          label,
          data: [],
          borderColor: color,
          backgroundColor: color,
          pointRadius: 2.5,
          pointHoverRadius: 6,
          pointHitRadius: 10,
          pointBorderColor: '#fff',
          pointBorderWidth: 1,
          tension: 0.2,
        }] },
        options: {
          responsive: true,
          interaction: { mode: 'nearest', intersect: true },
          scales: {
            x: { type: 'time', time: { unit: 'day' } },
            y: { beginAtZero: true }
          },
          elements: { point: { radius: 2.5, hoverRadius: 6, hitRadius: 10 } },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }

    (async () => {
      const data = await loadData();
      const sleepChart = newLineChart(document.getElementById('sleepChart'), 'Sleep (hours)', '#3b82f6');
      const stressChart = newLineChart(document.getElementById('stressChart'), 'Stress', '#ef4444');
      const hrChart = newLineChart(document.getElementById('hrChart'), 'HR (bpm)', '#10b981');

      // Default range based on daily coverage (union across metrics)
      const daily = data.daily || { hr: [], stress: [], sleep: [] };
      const dailyDates = [
        ...daily.hr.map(r => DateTime.fromFormat(r.day, 'yyyy-LL-dd', { zone: TZ })),
        ...daily.stress.map(r => DateTime.fromFormat(r.day, 'yyyy-LL-dd', { zone: TZ })),
        ...daily.sleep.map(r => DateTime.fromFormat(r.day, 'yyyy-LL-dd', { zone: TZ })),
      ];
      const allTs = dailyDates.length ? dailyDates : [
        ...data.hr_samples.map(s => parseISOZ(s.ts)),
        ...data.stress_samples.map(s => parseISOZ(s.ts)),
        ...data.sleep_sessions.flatMap(s => [s.start && parseISOZ(s.start), s.end && parseISOZ(s.end)].filter(Boolean)),
      ];
      const minTs = allTs.length ? DateTime.min(...allTs) : DateTime.now().minus({ months: 1 });
      const maxTs = allTs.length ? DateTime.max(...allTs) : DateTime.now();
      const defaultFrom = minTs.setZone(TZ).toFormat('yyyy-LL-dd');
      const defaultTo = maxTs.setZone(TZ).toFormat('yyyy-LL-dd');
      document.getElementById('from').value = defaultFrom;
      document.getElementById('to').value = defaultTo;

      let bucket = 'day';

      function onZoomApplied(chart) {
        const scale = chart.scales.x;
        if (!scale) return;
        const from = DateTime.fromMillis(scale.min, { zone: TZ }).toFormat('yyyy-LL-dd');
        const to = DateTime.fromMillis(scale.max, { zone: TZ }).toFormat('yyyy-LL-dd');
        document.getElementById('from').value = from;
        document.getElementById('to').value = to;
        refresh();
        setTimeout(() => { if (chart.resetZoom) chart.resetZoom(); }, 0);
      }

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
      document.getElementById('resetRange').addEventListener('click', () => {
        document.getElementById('from').value = defaultFrom;
        document.getElementById('to').value = defaultTo;
        refresh();
        if (sleepChart.resetZoom) sleepChart.resetZoom();
        if (stressChart.resetZoom) stressChart.resetZoom();
        if (hrChart.resetZoom) hrChart.resetZoom();
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
