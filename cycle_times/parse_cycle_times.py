"""
CED Log Parser — Pallet Cycle Time Extractor
=============================================
Reads all Message.txt files under CED/ and extracts the machine-reported
cycle time for each dispensing cycle.

How a cycle is parsed
---------------------
Within each cycle the log emits a PDCA data block containing:
    <carrier_sn>@pdata@cycle_time@<value>@@@s

This line appears twice (once per DUT slot on the carrier). We take the
first occurrence only. The timestamp we record is from the immediately
following "Dispensing complete" line — this is the cycle-end timestamp.
Cycles that have no cycle_time line (e.g. the very first warm-up cycle)
are silently skipped.

Output CSV columns
------------------
    date          — calendar date (YYYY-MM-DD) from the folder name
    datetime      — full datetime of dispensing complete (cycle end)
    cycle_time_s  — machine-reported cycle time in seconds

Usage
-----
    Set CED_PATH and OUTPUT_CSV below, then run:
        python parse_cycle_times.py
"""

import re
import csv
from pathlib import Path
from datetime import date, datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
CED_PATH   = "/home/dhruvkumarjiguda/code/arima/datasets/CED/"   # <-- your CED folder

OUTPUT_CSV = "./cycle_times.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Matches the timestamp on lines like:
#   RIGHT————————>08:02:47.423  Dispensing complete,...
TS_PAT = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d+)")

# Matches the machine cycle_time line (no timestamp prefix):
#   J63HQG006TB0000WDH@pdata@cycle_time@15.343@@@s
CYCLE_PAT = re.compile(r"@pdata@cycle_time@([0-9]+(?:\.[0-9]+)?)@")


def parse_date_from_folder(folder_name: str) -> date | None:
    """Convert folder name YYMMDD → datetime.date."""
    s = str(folder_name)
    try:
        return date(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]))
    except (ValueError, IndexError):
        return None


def ts_to_datetime(day: date, h: str, m: str, s: str, frac: str) -> datetime:
    """Combine a calendar date with parsed HH:MM:SS.frac into a datetime."""
    # Normalise fractional seconds to microseconds
    us = int(frac.ljust(6, "0")[:6])
    return datetime(day.year, day.month, day.day,
                    int(h), int(m), int(s), us)


def parse_file(filepath: Path, day: date) -> list[dict]:
    """
    Scan one Message.txt and return a list of cycle dicts.

    Logic (single linear pass):
      - When we see a cycle_time line and we don't already have a pending
        value, store it (first-occurrence-only deduplication).
      - When we see "Dispensing complete" and we have a pending cycle_time,
        emit a row and clear the pending value.
      - If we see another cycle_time before "Dispensing complete" we ignore
        it (it's the duplicate for the second DUT slot).
      - If "Dispensing complete" arrives with no pending cycle_time, skip it
        (this covers the first warm-up cycle and any malformed blocks).
    """
    rows = []
    pending_cycle_time = None   # float seconds, set on first cycle_time line

    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for line in fh:

                # ── cycle_time line (no timestamp) ───────────────────────
                m = CYCLE_PAT.search(line)
                if m and pending_cycle_time is None:
                    pending_cycle_time = float(m.group(1))
                    continue

                # ── Dispensing complete (has timestamp) ──────────────────
                if "Dispensing complete" in line and pending_cycle_time is not None:
                    ts = TS_PAT.search(line)
                    if ts:
                        dt = ts_to_datetime(day, *ts.groups())
                        rows.append({
                            "date":         day.isoformat(),
                            "datetime":     dt.isoformat(sep=" "),
                            "cycle_time_s": pending_cycle_time,
                        })
                    pending_cycle_time = None   # reset for next cycle

    except FileNotFoundError:
        print(f"  [WARN] File not found: {filepath}")

    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ced = Path(CED_PATH)
    if not ced.exists():
        raise FileNotFoundError(f"CED folder not found: {ced.resolve()}")

    day_folders = sorted(
        [d for d in ced.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: d.name,
    )
    print(f"Found {len(day_folders)} day folder(s) under {ced.resolve()}\n")

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

        rows = parse_file(msg_file, day)
        all_rows.extend(rows)
        print(f"  {folder.name} ({day})  →  {len(rows):5,} cycles")

    if not all_rows:
        print("\n[ERROR] No cycles extracted — check CED_PATH and log format.")
        return

    # ── Write CSV ────────────────────────────────────────────────────────────
    fieldnames = ["date", "datetime", "cycle_time_s"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓  Wrote {len(all_rows):,} rows  →  {Path(OUTPUT_CSV).resolve()}")

    # ── Quick summary ────────────────────────────────────────────────────────
    times = [r["cycle_time_s"] for r in all_rows]
    print(f"\nCycle time summary (seconds):")
    print(f"  n      : {len(times):,}")
    print(f"  mean   : {sum(times)/len(times):.3f}")
    print(f"  min    : {min(times):.3f}")
    print(f"  max    : {max(times):.3f}")
    slow = sum(1 for t in times if t > 20)
    print(f"  > 20s  : {slow:,}  ({slow/len(times)*100:.1f}%)")


if __name__ == "__main__":
    main()
