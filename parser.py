"""
CED Log Parser — Cylinder Stroke Duration Extractor
=====================================================
Reads all Message.txt files under CED/ and extracts:
  - Per-cycle cylinder stroke durations (ms) derived from timestamp differencing
  - 4 cylinders × 2 directions = 8 time series per cycle

Output CSV columns:
  date, hour, cylinder, direction, stroke_index,
  t_start_ms, duration_ms

Usage:
  Set CED_PATH below, then run the cell.
"""

import re
import os
import csv
from pathlib import Path
from datetime import date

# ── CONFIG ────────────────────────────────────────────────────────────────────
CED_PATH   = "/home/dhruvkumarjiguda/code/pynb/CED/"          # <-- change this to your CED folder path
OUTPUT_CSV = "./cylinder_strokes.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Cylinder definitions: each entry maps a short name to its log patterns
CYLINDERS = {
    "Lifting": {
        "on":   "Lifting cylinderelectromagnetic valve[ON]",
        "off":  "Lifting cylinderelectromagnetic valve[OFF]",
        "work": "Lifting cylinderwork position [WORK Complete]",
        "home": "Lifting cylinderhome position [HOME Complete]",
    },
    "Middle": {
        "on":   "Middle lifting cylinderelectromagnetic valve [ON]",
        "off":  "Middle lifting cylinderelectromagnetic valve[OFF]",
        "work": "Middle lifting cylinderWORK位[WORK complete]",
        "home": "Middle lifting cylinderhome position [HOME Complete]",
    },
    "Front_barrier": {
        "on":   "Front barrier lifting cylinderelectromagnetic valve[ON]",
        "off":  "Front barrier lifting cylinderelectromagnetic valve[OFF]",
        "work": "Front barrier lifting cylinderwork position [WORK Complete]",
        "home": "Front barrier lifting cylinderhome position [HOME Complete]",
    },
    "Rear_barrier": {
        "on":   "Rear barrier lifting cylinderelectromagnetic valve[ON]",
        "off":  "Rear barrier lifting cylinderelectromagnetic valve[OFF]",
        "work": "Rear barrier lifting cylinderwork position [WORK Complete]",
        "home": "Rear barrier lifting cylinderhome position [HOME Complete]",
    },
}

TS_PAT = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d+)")

MAX_STROKE_MS = 2000   # reject anything longer — likely a mis-pair
MIN_STROKE_MS = 1      # reject sub-millisecond noise


def ts_to_ms(h, m, s, frac):
    """Convert parsed timestamp parts to total milliseconds since midnight."""
    ms = int(frac.ljust(3, "0")[:3])   # normalise to 3-digit ms
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + ms


def parse_date_from_folder(folder_name):
    """Convert folder name YYMMDD → datetime.date."""
    s = str(folder_name)
    try:
        return date(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]))
    except (ValueError, IndexError):
        return None


def extract_events(filepath):
    """
    Read one Message.txt and return a list of dicts:
      { ts_ms, cylinder, event }
    where event is one of: on, off, work, home
    """
    events = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = TS_PAT.search(line)
                if not m or "cylinder" not in line.lower():
                    continue
                ts_ms = ts_to_ms(*m.groups())
                for cyl_name, patterns in CYLINDERS.items():
                    for event_type, pattern in patterns.items():
                        if pattern in line:
                            events.append({
                                "ts_ms":    ts_ms,
                                "cylinder": cyl_name,
                                "event":    event_type,
                            })
                            break   # a line can only match one event per cylinder
    except FileNotFoundError:
        print(f"  [WARN] File not found: {filepath}")
    return events


def compute_strokes(events, day, hour_of_day=None):
    """
    Given sorted events for one file, pair ON→WORK (extension) and
    OFF→HOME (retraction) to get stroke durations.

    Returns a list of row dicts ready for CSV output.
    """
    rows = []

    for cyl in CYLINDERS:
        cyl_events = [(e["ts_ms"], e["event"])
                      for e in events if e["cylinder"] == cyl]
        cyl_events.sort(key=lambda x: x[0])

        # ── Extension: ON → WORK Complete ────────────────────────────────────
        on_ts = None
        stroke_idx = 0
        for ts, evt in cyl_events:
            if evt == "on":
                on_ts = ts
            elif evt == "work" and on_ts is not None:
                dur = ts - on_ts
                if MIN_STROKE_MS <= dur <= MAX_STROKE_MS:
                    hour = ts // 3_600_000
                    rows.append({
                        "date":         day.isoformat(),
                        "hour":         hour,
                        "cylinder":     cyl,
                        "direction":    "extend",
                        "stroke_index": stroke_idx,
                        "t_start_ms":   on_ts,
                        "duration_ms":  dur,
                    })
                    stroke_idx += 1
                on_ts = None

        # ── Retraction: OFF → HOME Complete ──────────────────────────────────
        off_ts = None
        stroke_idx = 0
        for ts, evt in cyl_events:
            if evt == "off":
                off_ts = ts
            elif evt == "home" and off_ts is not None:
                dur = ts - off_ts
                if MIN_STROKE_MS <= dur <= MAX_STROKE_MS:
                    hour = ts // 3_600_000
                    rows.append({
                        "date":         day.isoformat(),
                        "hour":         hour,
                        "cylinder":     cyl,
                        "direction":    "retract",
                        "stroke_index": stroke_idx,
                        "t_start_ms":   off_ts,
                        "duration_ms":  dur,
                    })
                    stroke_idx += 1
                off_ts = None

    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ced = Path(CED_PATH)
    if not ced.exists():
        raise FileNotFoundError(f"CED folder not found: {ced.resolve()}")

    # Collect all dated sub-folders in chronological order
    day_folders = sorted(
        [d for d in ced.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: d.name
    )
    print(f"Found {len(day_folders)} day folders under {ced.resolve()}\n")

    all_rows = []
    for folder in day_folders:
        day = parse_date_from_folder(folder.name)
        if day is None:
            print(f"  [SKIP] Cannot parse date from folder: {folder.name}")
            continue

        msg_file = folder / "Message.txt"
        if not msg_file.exists():
            print(f"  [SKIP] No Message.txt in {folder.name}")
            continue

        events = extract_events(msg_file)
        events.sort(key=lambda e: e["ts_ms"])

        rows = compute_strokes(events, day)
        all_rows.extend(rows)
        print(f"  {folder.name} ({day})  →  {len(rows):5,} stroke records")

    # Write CSV
    if not all_rows:
        print("\n[ERROR] No rows extracted — check CED_PATH and log format.")
        return

    fieldnames = ["date", "hour", "cylinder", "direction",
                  "stroke_index", "t_start_ms", "duration_ms"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓  Wrote {len(all_rows):,} rows → {Path(OUTPUT_CSV).resolve()}")

    # Quick summary table
    from collections import defaultdict
    import statistics

    summary = defaultdict(list)
    for r in all_rows:
        summary[(r["cylinder"], r["direction"])].append(r["duration_ms"])

    print("\nSummary (mean ± stdev, ms):")
    print(f"  {'Cylinder':<16} {'Direction':<8}  {'n':>6}  {'mean':>7}  {'stdev':>7}  {'min':>6}  {'max':>6}")
    print("  " + "-" * 66)
    for (cyl, dirn), vals in sorted(summary.items()):
        print(f"  {cyl:<16} {dirn:<8}  {len(vals):>6,}  "
              f"{statistics.mean(vals):>7.1f}  "
              f"{statistics.stdev(vals):>7.1f}  "
              f"{min(vals):>6}  {max(vals):>6}")


if __name__ == "__main__":
    main()