"""
scheduler.py -- decides which channel slots are due and runs them.

For each channel in ENABLED_CHANNELS, every time in `shorts_upload_times_utc`
whose slot started within the last GRACE_HOURS (GitHub cron runs late) and has
not been completed yet is run once. Completed slots are stored in
state/<channel>_schedule.json (persisted by the workflow cache).
"""
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
GRACE_HOURS = 3


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {"done": []}


def due_slots(config: dict, done: set, now: datetime) -> list:
    out = []
    for t in config.get("shorts_upload_times_utc", []):
        h, m = (int(x) for x in str(t).split(":"))
        slot = now.replace(hour=h, minute=m, second=0, microsecond=0)
        key = slot.strftime("%Y-%m-%dT%H:%M")
        if slot <= now < slot + timedelta(hours=GRACE_HOURS) and key not in done:
            out.append(key)
    return out


def main() -> int:
    from src.main import run_channel

    now = datetime.now(timezone.utc)
    dry_run = bool(os.environ.get("DRY_RUN"))
    channels = [c.strip() for c in os.environ.get("ENABLED_CHANNELS", "gta_vibe").split(",") if c.strip()]
    failed = False

    for ch in channels:
        config = yaml.safe_load((ROOT / "channels" / f"{ch}.yaml").read_text())
        sched = STATE_DIR / f"{ch}_schedule.json"
        state = _load(sched)
        # keep only recent entries
        cutoff = (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M")
        state["done"] = [d for d in state.get("done", []) if d >= cutoff]
        slots = due_slots(config, set(state["done"]), now)
        print(f"[{ch}] now={now:%H:%M} UTC, due slots: {slots or 'none'}")

        for slot in slots:
            try:
                run_channel(ch, dry_run=dry_run)
                if not dry_run:
                    state["done"].append(slot)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                failed = True
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            sched.write_text(json.dumps(state))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
