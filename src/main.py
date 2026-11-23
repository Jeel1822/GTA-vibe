"""
main.py -- run one clipping job for a channel.

    python -m src.main gta_vibe              # clip + upload one Short
    python -m src.main gta_vibe --dry-run    # render only, saves output/gta_vibe_clip.mp4
"""
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def load_channel_config(channel_id: str) -> dict:
    with open(ROOT / "channels" / f"{channel_id}.yaml") as f:
        return yaml.safe_load(f)


def run_channel(channel_id: str, dry_run: bool = False, privacy_status: str = "public") -> None:
    config = load_channel_config(channel_id)
    if config.get("pipeline") != "clipping":
        raise SystemExit(f"{channel_id}: only 'pipeline: clipping' channels are supported in this repo.")
    print(f"=== Running {config['display_name']} | SHORT | dry_run={dry_run} ===")

    if not dry_run:
        # Fail fast if the YouTube token is dead.
        from src.upload_youtube import verify_token_valid
        secret = ROOT / "client_secret.json"
        verify_token_valid(str(ROOT / "tokens" / f"{channel_id}_token.pickle"),
                           str(secret) if secret.exists() else None)
        print("Pre-flight auth check passed.")

    from src.clipper import run_clipping
    run_clipping(channel_id, config, privacy_status, dry_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("channel_id")
    p.add_argument("--short", action="store_true", help="accepted for compatibility; always a Short")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()
    run_channel(a.channel_id, a.dry_run, "private" if a.private else "public")
