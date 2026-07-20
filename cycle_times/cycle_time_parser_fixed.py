import re
import csv
from pathlib import Path
from datetime import date

# ── CONFIG ─────────────────────────────────────────────────────────────
CED_PATH = "/home/dhruvkumarjiguda/code/arima/datasets/CED/"
OUTPUT_CSV = "./cycle_times.csv"

# Example:
# J63HQG006TB0000WDH@pdata@cycle_time@15.343@@@s
CYCLE_PAT = re.compile(
    r"([A-Z0-9]+)@pdata@cycle_time@([0-9]+(?:\.[0-9]+)?)@"
)

# ───────────────────────────────────────────────────────────────────────


def parse_date_from_folder(folder_name: str):
    try:
        return date(
            2000 + int(folder_name[0:2]),
            int(folder_name[2:4]),
            int(folder_name[4:6]),
        )
    except Exception:
        return None


def parse_file(filepath, day):

    rows = []

    with open(filepath, encoding="utf-8", errors="replace") as fh:

        for line in fh:

            m = CYCLE_PAT.search(line)

            if m:

                dut_id = m.group(1)
                cycle_time = float(m.group(2))

                rows.append({
                    "date": day.isoformat(),
                    "id": dut_id,
                    "cycle_time_s": cycle_time,
                })

    return rows


def main():

    ced = Path(CED_PATH)

    all_rows = []

    for folder in sorted(ced.iterdir()):

        if not folder.is_dir():
            continue

        day = parse_date_from_folder(folder.name)
        if day is None:
            continue

        msg_file = folder / "Message.txt"
        if not msg_file.exists():
            continue

        rows = parse_file(msg_file, day)

        print(f"{folder.name} → {len(rows)} rows")

        all_rows.extend(rows)

    if not all_rows:
        print("No cycle data found")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["date", "id", "cycle_time_s"]
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()