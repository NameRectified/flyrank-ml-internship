from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "work" / "outputs" / "action_playbook_queue.csv"
UI_DIR = ROOT / "paper" / "ui"
QUEUE_JS_PATH = UI_DIR / "queue.js"

# Per-client cap keeps the demo light; every client still gets its top pages.
TOP_PER_CLIENT = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper/ui/queue.js from the action-playbook queue."
    )
    parser.add_argument("--queue", default=str(QUEUE_PATH))
    parser.add_argument("--out", default=str(QUEUE_JS_PATH))
    parser.add_argument("--top-per-client", type=int, default=TOP_PER_CLIENT)
    return parser.parse_args()


def to_float(v: object) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else round(f, 4)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.queue)
    total_pages = len(df)
    total_clients = df["client_hash_id"].nunique()

    df = df.sort_values("score", ascending=False)
    capped = df.groupby("client_hash_id", group_keys=False).head(args.top_per_client)

    rows = []
    for _, row in capped.iterrows():
        rows.append(
            {
                "r": int(row["rank"]),
                "c": str(row["client_hash_id"]),
                "p": str(row["content_hash_id"]),
                "s": float(round(row["score"], 1)),
                "rc": [x for x in str(row["reason_codes"]).split("|") if x],
                "ar": str(row["archetype"]),
                "ac": str(row["action"]),
                "pt": str(row["position_tier"]),
                "ct": str(row["content_type"]),
                "mi": str(row["main_intent"]),
                "im": int(row["impressions_fw"]),
                "cr": to_float(row["ctr_fw"]),
                "g": to_float(row["tier_ctr_gap"]),
                "ap": to_float(row["avg_pos_fw"]),
                "se": int(row["sessions_fw"]),
                "en": to_float(row["engagement_rate_fw"]),
                "mc": to_float(row["missed_clicks_fw"]),
                "age": int(row["content_age_days"]),
            }
        )

    payload = f"window.QUEUE = {json.dumps(rows, separators=(',', ':'))};\n"
    UI_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_JS_PATH.write_text(payload)

    mb = len(payload.encode()) / 1e6
    print(
        f"queue pages: {total_pages:,} | clients: {total_clients:,} | "
        f"capped to top {args.top_per_client}/client: {len(rows):,} rows"
    )
    print(f"wrote {QUEUE_JS_PATH} ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
