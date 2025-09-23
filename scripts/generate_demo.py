#!/usr/bin/env python3
import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic demo dataset for the Garmin dashboard.")
    p.add_argument("--out", "-o", type=Path, default=Path("dist"), help="Output directory (default: dist)")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD in local TZ (default: today-120d)")
    p.add_argument("--days", type=int, default=120, help="Number of days to generate (default: 120)")
    p.add_argument("--tz", type=str, default="Europe/Paris", help="IANA timezone for generation (default: Europe/Paris)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--copy_index", action="store_true", help="Copy templates/index.html into output folder")
    return p.parse_args()


def to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def date_from_str(s: str, tz: ZoneInfo) -> datetime:
    y, m, d = map(int, s.split("-"))
    return datetime(y, m, d, tzinfo=tz)


def generate_hr_and_stress(day_start: datetime, tz: ZoneInfo, rnd: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # HR every 5 minutes; Stress every 15 minutes
    hr_samples: List[Dict[str, Any]] = []
    stress_samples: List[Dict[str, Any]] = []
    local_day = day_start
    # Define active hours 06:00-23:00 local
    start_active = local_day.replace(hour=6, minute=0, second=0, microsecond=0)
    end_active = local_day.replace(hour=23, minute=0, second=0, microsecond=0)

    # Exercise spike window midday on some days
    exercise = rnd.random() < 0.35
    ex_start = local_day.replace(hour=12, minute=0) if exercise else None
    ex_end = local_day.replace(hour=13, minute=0) if exercise else None

    t = start_active
    while t <= end_active:
        # Baseline HR
        hr_base = 62 + 6 * (1 if 8 <= t.hour <= 20 else -1)  # slightly lower outside working hours
        # Add random daily variance
        hr = int(hr_base + rnd.gauss(0, 6))
        # If within exercise window
        if exercise and ex_start <= t <= ex_end:
            hr += 40 + int(15 * rnd.random())
        # Clamp to plausible
        hr = max(45, min(185, hr))
        hr_samples.append({"ts": to_iso_utc(t.astimezone(timezone.utc)), "hr": hr})
        t += timedelta(minutes=5)

    # Stress generation (0-100) during active hours
    t = start_active
    while t <= end_active:
        stress_base = 20 + 15 * (1 if 9 <= t.hour <= 18 else 0)
        if exercise and ex_start <= t <= ex_end:
            stress_base += 25
        stress = int(max(0, min(100, rnd.gauss(stress_base, 8))))
        stress_samples.append({"ts": to_iso_utc(t.astimezone(timezone.utc)), "stress": stress})
        t += timedelta(minutes=15)

    return hr_samples, stress_samples


@dataclass
class SleepSession:
    start: datetime
    end: datetime


def generate_sleep(day_start: datetime, tz: ZoneInfo, rnd: random.Random) -> Optional[SleepSession]:
    # Night sleep centered around 23:30 -> 07:00 next day; random variation
    # Some nights missing entirely
    if rnd.random() < 0.08:
        return None
    go_bed_hour = 22 + rnd.random() * 2.5  # 22:00 - 00:30
    sleep_hours = 6.3 + rnd.random() * 2.6  # ~6.3 - 8.9
    # occasional short night
    if rnd.random() < 0.08:
        sleep_hours -= 1.0
    start = (day_start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=go_bed_hour))
    end = start + timedelta(hours=sleep_hours)
    # Ensure timestamps carry TZ
    start = start.replace(tzinfo=tz)
    end = end.replace(tzinfo=tz)
    return SleepSession(start=start, end=end)


def split_sleep_daily_minutes(sessions: List[SleepSession], tz: ZoneInfo) -> Dict[str, float]:
    minutes: Dict[str, float] = {}
    for s in sessions:
        start_local = s.start.astimezone(tz)
        end_local = s.end.astimezone(tz)
        cur = datetime(start_local.year, start_local.month, start_local.day, tzinfo=tz)
        while cur < end_local:
            nxt = cur + timedelta(days=1)
            seg_start = max(cur, start_local)
            seg_end = min(nxt, end_local)
            dur_min = max(0.0, (seg_end - seg_start).total_seconds() / 60.0)
            if dur_min > 0:
                key = cur.strftime("%Y-%m-%d")
                minutes[key] = minutes.get(key, 0.0) + dur_min
            cur = nxt
    return minutes


def build_missing_spans(days: List[str], present_days: set, rnd: random.Random) -> List[Dict[str, str]]:
    spans: List[Dict[str, str]] = []
    i = 0
    n = len(days)
    while i < n:
        d = days[i]
        if d not in present_days:
            j = i
            while j + 1 < n and days[j + 1] not in present_days:
                j += 1
            spans.append({"from": days[i], "to": days[j]})
            i = j + 1
        else:
            i += 1
    return spans


def main() -> int:
    args = parse_args()
    rnd = random.Random(args.seed)
    tz = ZoneInfo(args.tz)

    if args.start:
        start_local = date_from_str(args.start, tz)
    else:
        today_local = datetime.now(tz)
        start_local = datetime(today_local.year, today_local.month, today_local.day, tzinfo=tz) - timedelta(days=args.days)

    days_list = [
        (start_local + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days)
    ]

    hr_samples: List[Dict[str, Any]] = []
    stress_samples: List[Dict[str, Any]] = []
    sessions: List[SleepSession] = []

    # Randomly choose missing spans per metric (for demo)
    def random_missing_mask() -> set:
        mask = set()
        for _ in range(rnd.randint(1, 3)):
            start_idx = rnd.randint(0, max(0, args.days - 5))
            length = rnd.randint(1, 4)
            for j in range(start_idx, min(args.days, start_idx + length)):
                mask.add(days_list[j])
        return mask

    hr_missing = random_missing_mask()
    stress_missing = random_missing_mask()
    sleep_missing = random_missing_mask()

    for i in range(args.days):
        day_str = days_list[i]
        day_start = date_from_str(day_str, tz)

        # HR/Stress, skip if day is missing
        if day_str not in hr_missing:
            hr_day, _ = generate_hr_and_stress(day_start, tz, rnd)
            hr_samples.extend(hr_day)
        if day_str not in stress_missing:
            _, st_day = generate_hr_and_stress(day_start, tz, rnd)
            stress_samples.extend(st_day)

        # Sleep session for this night (starting this day)
        if day_str not in sleep_missing:
            s = generate_sleep(day_start, tz, rnd)
            if s is not None:
                sessions.append(s)

    # Build JSON structure
    payload: Dict[str, Any] = {
        "hr_samples": hr_samples,
        "stress_samples": stress_samples,
        "sleep_sessions": [
            {"start": to_iso_utc(s.start), "end": to_iso_utc(s.end), "duration_sec": int((s.end - s.start).total_seconds())}
            for s in sessions
        ],
        "meta": {"tz": args.tz, "generated_at": to_iso_utc(datetime.now(timezone.utc))},
    }

    # Daily aggregates
    # HR daily avg
    hr_daily_acc: Dict[str, Tuple[int, int]] = {}
    for s in hr_samples:
        dt = datetime.fromisoformat(s["ts"].replace("Z", "+00:00")).astimezone(tz)
        day = dt.strftime("%Y-%m-%d")
        a, c = hr_daily_acc.get(day, (0, 0))
        hr_daily_acc[day] = (a + int(s["hr"]), c + 1)
    hr_daily = [{"day": d, "avg": (a / c) if c else None} for d, (a, c) in hr_daily_acc.items()]
    hr_daily.sort(key=lambda x: x["day"]) 

    # Stress daily avg
    st_daily_acc: Dict[str, Tuple[int, int]] = {}
    for s in stress_samples:
        dt = datetime.fromisoformat(s["ts"].replace("Z", "+00:00")).astimezone(tz)
        day = dt.strftime("%Y-%m-%d")
        a, c = st_daily_acc.get(day, (0, 0))
        st_daily_acc[day] = (a + int(s["stress"]), c + 1)
    st_daily = [{"day": d, "avg": (a / c) if c else None} for d, (a, c) in st_daily_acc.items()]
    st_daily.sort(key=lambda x: x["day"]) 

    # Sleep daily hours from sessions
    sleep_minutes = split_sleep_daily_minutes(sessions, tz)
    sl_daily = [{"day": d, "hours": m / 60.0} for d, m in sleep_minutes.items()]
    sl_daily.sort(key=lambda x: x["day"]) 

    payload["daily"] = {"hr": hr_daily, "stress": st_daily, "sleep": sl_daily}

    # Missing spans for annotations
    payload["missing_spans"] = {
        "hr": build_missing_spans(days_list, set(d for d, _ in hr_daily_acc.items()), rnd),
        "stress": build_missing_spans(days_list, set(d for d, _ in st_daily_acc.items()), rnd),
        "sleep": build_missing_spans(days_list, set(sleep_minutes.keys()), rnd),
    }

    # Write out
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "data.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    if args.copy_index:
        # Try templates/index.html
        candidates = [Path(__file__).resolve().parents[1] / "templates" / "index.html",
                      Path(__file__).resolve().parents[1] / "dist" / "index.html"]
        for src in candidates:
            if src.exists():
                (args.out / "index.html").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                break

    print(f"Wrote {out_json} ({len(hr_samples)} HR, {len(stress_samples)} stress samples, {len(sessions)} sleep sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
