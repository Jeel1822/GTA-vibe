"""
clipper.py
Automatic highlight clipping for "clipping" channels (e.g. GTA Vibe).

Flow for one Short:
  1. Pick the next source (a URL or local file from channels/<id>_sources.txt,
     or any video in sources/<id>/).
  2. Licence gate: URL sources must be Creative Commons (or the channel must
     declare own_content: true) -- see config keys below.
  3. Find the most exciting moment: loudness (explosions, shouting, crashes)
     analysed from the audio track, skipping intro/outro and segments that
     were already used.
  4. Cut only that window (yt-dlp --download-sections for URLs, so we never
     pull a whole 3-hour stream), reframe to 9:16 with a blurred backdrop,
     add a short hook title and a credit line.
  5. Upload through the existing upload_youtube.upload_video().

Config keys (channels/<id>.yaml):
  pipeline: clipping
  clip_min_seconds / clip_max_seconds     length of each Short (<= 59)
  clip_pre_roll_seconds                   lead-in before the loudest moment
  skip_head_seconds / skip_tail_seconds   ignore intros / outros
  require_cc_license: true                URL sources must be CC-licensed
  own_content: false                      true = you own / have permission
                                          for every source (skips CC gate)
  hook_text: true                         burn a 3-second hook title
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


# --------------------------------------------------------------------------
# State (which source / which time ranges are already used)
# --------------------------------------------------------------------------
def _state_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_clipper_state.json"


def _load_state(channel_id: str) -> dict:
    p = _state_path(channel_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"sources": {}}


def _save_state(channel_id: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(channel_id).write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def list_sources(channel_id: str) -> list[str]:
    """URLs / paths from channels/<id>_sources.txt plus any local videos in
    sources/<id>/. Lines starting with # are comments."""
    out: list[str] = []
    f = ROOT / "channels" / f"{channel_id}_sources.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    d = ROOT / "sources" / channel_id
    if d.exists():
        out += [str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    return out


def _ytdlp_cmd() -> list[str]:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    return ["python3", "-m", "yt_dlp"]


def _ytdlp_extra_args() -> list[str]:
    """Optional cookies file (helps when YouTube blocks datacenter IPs)."""
    cookies = ROOT / "cookies.txt"
    return ["--cookies", str(cookies)] if cookies.exists() else []


def _is_single_video_url(url: str) -> bool:
    """Channel / playlist URLs (youtube.com/@name, /channel/, /playlist) are
    NOT single videos: yt-dlp -J on them tries to enumerate every upload and
    can hang for many minutes. Only watch/shorts/youtu.be links are accepted."""
    return any(k in url for k in ("watch?v=", "youtu.be/", "/shorts/", "/live/"))


def _fetch_meta(url: str) -> dict:
    if not _is_single_video_url(url):
        raise SkipSource(f"{url}: not a single-video URL (channel/playlist links "
                         f"are not supported; paste individual video links)")
    try:
        res = subprocess.run(
            _ytdlp_cmd() + _ytdlp_extra_args() + ["-J", "--no-playlist", url],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"yt-dlp metadata timed out for {url}")
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed for {url}: {res.stderr.strip()[-400:]}")
    return json.loads(res.stdout)


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


# --------------------------------------------------------------------------
# Highlight detection
# --------------------------------------------------------------------------
def _loudness_series(audio_path: str, hop: float = 0.5) -> np.ndarray:
    """RMS loudness (dB) per `hop` seconds, from a mono 8 kHz decode."""
    sr = 8000
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", audio_path, "-vn", "-ac", "1",
         "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    pcm = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(sr * hop)
    frames = len(pcm) // n
    if frames == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt(np.mean(pcm[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-10)
    return 20.0 * np.log10(rms)


def find_best_window(
    loud: np.ndarray, hop: float, clip_len: float, pre_roll: float,
    skip_head: float, skip_tail: float, used_ranges: list,
) -> tuple[float, float, float] | None:
    """Returns (start, end, score) of the most exciting unused window.

    Score = how far the loudest moment sits above the video's normal level
    (z-score) plus a smaller bonus for sustained loudness across the window.
    The window is placed so the peak lands `pre_roll` seconds in.
    """
    if len(loud) < int(clip_len / hop) + 2:
        return None
    mu, sd = float(np.mean(loud)), float(np.std(loud)) + 1e-6
    z = (loud - mu) / sd
    # light smoothing so a single spiky frame doesn't win
    k = max(1, int(1.0 / hop))
    zs = np.convolve(z, np.ones(k) / k, mode="same")

    total = len(loud) * hop
    best = None
    for i in np.argsort(zs)[::-1][:400]:  # strongest moments first
        peak_t = i * hop
        start = max(skip_head, peak_t - pre_roll)
        end = start + clip_len
        if end > total - skip_tail:
            end = total - skip_tail
            start = end - clip_len
        if start < skip_head:
            continue
        if any(not (end <= u0 or start >= u1) for u0, u1 in used_ranges):
            continue
        a, b = int(start / hop), int(end / hop)
        sustained = float(np.mean(zs[a:b]))
        score = float(zs[i]) + 0.5 * sustained
        if best is None or score > best[2]:
            best = (round(start, 2), round(end, 2), round(score, 3))
        if best and len(used_ranges) == 0:
            break  # first valid peak is the best when nothing is excluded
    return best


# --------------------------------------------------------------------------
# Title / description
# --------------------------------------------------------------------------
def _fallback_hook(source_title: str) -> str:
    return "WAIT FOR IT..."


def make_hook_and_title(source_title: str, niche: str) -> tuple[str, str]:
    """(short on-screen hook, YouTube title). Gemini if available, else a
    safe template. Never invents events: it only knows the source title."""
    try:
        from src.agents.orchestrator import generate_text
        prompt = f"""You name Shorts for a YouTube channel about: {niche}.
The clip is a loud/exciting moment cut from a video titled: "{source_title}".
You do NOT know exactly what happens in the clip, so do not claim specifics.
Return JSON only: {{"hook": "...", "title": "..."}}
- hook: 2-4 words, ALL CAPS, curiosity-driven, shown on screen for 3 seconds
- title: under 70 characters, punchy, no hashtags, no emojis, no quotes
"""
        raw = generate_text(prompt, max_output_tokens=200, json_mode=True)
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())
        hook = str(data.get("hook", "")).strip()[:30].upper()
        title = str(data.get("title", "")).strip()[:70]
        if hook and title:
            return hook, title
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: Gemini hook/title failed ({e}); using template.")
    return _fallback_hook(source_title), f"{source_title[:55].strip()} - Best Moment"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _font() -> str | None:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return f
    return None


def render_vertical(src_clip: str, out_path: str, hook: str | None,
                    brand: str | None, meta: dict | None = None) -> None:
    """16:9 gameplay -> 1080x1920: blurred, zoomed copy as backdrop, sharp
    copy centred on top, hook title for 3 s, small brand tag, loudness-
    normalised audio with short fades."""
    dur = _probe_duration(src_clip)
    font = _font()

    def esc(t: str) -> str:
        return (t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
                 .replace("%", "\\%"))

    chain = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        "boxblur=28:6,eq=brightness=-0.08[bgb];"
        "[fg]scale=1080:-2:flags=lanczos[fgs];"
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]"
    )
    last = "v0"
    if font and hook:
        chain += (
            f";[{last}]drawtext=fontfile='{font}':text='{esc(hook)}':fontsize=96:"
            "fontcolor=yellow:borderw=7:bordercolor=black:x=(w-text_w)/2:y=h*0.17:"
            "enable='lt(t,3)'[v1]"
        )
        last = "v1"
    if font and brand:
        chain += (
            f";[{last}]drawtext=fontfile='{font}':text='{esc(brand)}':fontsize=44:"
            "fontcolor=white@0.85:borderw=4:bordercolor=black@0.8:"
            "x=(w-text_w)/2:y=h*0.86"
        )
        if meta:
            who = meta.get("uploader") or meta.get("channel") or "Original Creator"
            credit_str = f"Source: {who[:35]}"
            chain += (
                f",drawtext=fontfile='{font}':text='{esc(credit_str)}':fontsize=26:"
                f"fontcolor=white@0.75:x=(w-text_w)/2:y=h*0.91"
            )
        chain += "[v2]"
        last = "v2"

    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11,"
          f"afade=t=in:d=0.3,afade=t=out:st={max(dur - 0.5, 0):.2f}:d=0.5")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src_clip,
         "-filter_complex", chain, "-map", f"[{last}]", "-map", "0:a:0?",
         "-af", af, "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", "-t", "59", out_path],
        check=True,
    )


# --------------------------------------------------------------------------
# Source preparation
# --------------------------------------------------------------------------
def _cut_local(path: str, start: float, end: float, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start}", "-i", path,
         "-t", f"{end - start}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-c:a", "aac", out],
        check=True,
    )


def _license_ok(meta: dict, config: dict) -> tuple[bool, str]:
    if config.get("own_content", False):
        return True, ""
    if not config.get("require_cc_license", True):
        return True, ""
    lic = (meta.get("license") or "").lower()
    if "creative commons" in lic:
        return True, ""
    return False, f"licence is {meta.get('license') or 'standard YouTube licence'}"


def _credit_line(meta: dict) -> str:
    who = meta.get("uploader") or meta.get("channel") or "the original creator"
    url = meta.get("webpage_url") or ""
    lic = meta.get("license") or ""
    bits = [f"Source: {meta.get('title', 'original video')} by {who}"]
    if url:
        bits.append(url)
    if lic:
        bits.append(f"({lic})")
    return " ".join(bits)


def prepare_clip(source: str, config: dict, state: dict, tmp: Path
                 ) -> tuple[str, dict, tuple[float, float], str]:
    """Returns (raw_clip_path, meta, (start, end), source_id)."""
    clip_len = float(min(config.get("clip_max_seconds", 45), 58))
    clip_len = max(clip_len, float(config.get("clip_min_seconds", 20)))
    pre = float(config.get("clip_pre_roll_seconds", 12))
    head = float(config.get("skip_head_seconds", 30))
    tail = float(config.get("skip_tail_seconds", 30))
    hop = 0.5

    if _is_url(source):
        meta = _fetch_meta(source)
        ok, why = _license_ok(meta, config)
        if not ok:
            raise SkipSource(f"{source}: {why}")
        sid = str(meta.get("id") or source)
        used = state["sources"].get(sid, {}).get("used", [])
        audio_tpl = str(tmp / "audio.%(ext)s")
        subprocess.run(
            _ytdlp_cmd() + _ytdlp_extra_args() +
            ["-f", "bestaudio", "--no-playlist", "-o", audio_tpl, source],
            check=True, timeout=900,
        )
        audio = next(tmp.glob("audio.*"))
        loud = _loudness_series(str(audio), hop)
        win = find_best_window(loud, hop, clip_len, pre, head, tail, used)
        if not win:
            raise SkipSource(f"{source}: no unused highlight windows left")
        start, end, score = win
        print(f"Best window {start:.1f}-{end:.1f}s (score {score})")
        out_tpl = str(tmp / "raw.%(ext)s")
        subprocess.run(
            _ytdlp_cmd() + _ytdlp_extra_args() +
            ["-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
             "--merge-output-format", "mp4", "--no-playlist",
             "--download-sections", f"*{start}-{end}", "--force-keyframes-at-cuts",
             "-o", out_tpl, source],
            check=True, timeout=900,
        )
        raw = next(p for p in tmp.glob("raw.*") if p.suffix.lower() in VIDEO_EXTS)
        return str(raw), meta, (start, end), sid

    # local file: you own it -> no licence gate
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    sid = f"local:{path.name}"
    meta = {"title": path.stem.replace("_", " ").replace("-", " "),
            "uploader": config.get("display_name", ""), "webpage_url": "", "license": ""}
    used = state["sources"].get(sid, {}).get("used", [])
    loud = _loudness_series(str(path), hop)
    win = find_best_window(loud, hop, clip_len, pre, head, tail, used)
    if not win:
        raise SkipSource(f"{path.name}: no unused highlight windows left")
    start, end, score = win
    print(f"Best window {start:.1f}-{end:.1f}s (score {score})")
    raw = tmp / "raw.mp4"
    _cut_local(str(path), start, end, str(raw))
    return str(raw), meta, (start, end), sid


class SkipSource(Exception):
    pass


# --------------------------------------------------------------------------
# Entry point called from main.py
# --------------------------------------------------------------------------
def run_clipping(channel_id: str, config: dict, privacy_status: str = "public",
                 dry_run: bool = False) -> None:
    sources = list_sources(channel_id)
    if not sources:
        raise RuntimeError(
            f"No sources for {channel_id}. Add URLs to channels/{channel_id}_sources.txt "
            f"or drop video files into sources/{channel_id}/ (see SETUP_GTA_VIBE.md)."
        )
    state = _load_state(channel_id)
    # least-used sources first, so every source gets rotated through
    sources.sort(key=lambda s: len(state["sources"].get(s, {}).get("used", [])))

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        chosen = None
        for src in sources:
            work = tmp / re.sub(r"\W+", "_", src)[-40:]
            work.mkdir(parents=True, exist_ok=True)
            try:
                chosen = (src, *prepare_clip(src, config, state, work))
                break
            except SkipSource as e:
                print(f"Skipping source -> {e}")
            except Exception as e:  # noqa: BLE001  (download blocked, bad file...)
                print(f"WARNING: source failed ({e}); trying next.")
        if not chosen:
            raise RuntimeError("No usable source this run (see messages above).")

        src, raw, meta, (start, end), sid = chosen
        hook, title = make_hook_and_title(meta.get("title", "GTA moment"),
                                          config.get("niche", "GTA gameplay"))
        final = tmp / "final.mp4"
        render_vertical(raw, str(final),
                        hook if config.get("hook_text", True) else None,
                        config.get("brand_tag", f"@{config['display_name'].replace(' ', '')}"),
                        meta=meta)
        print(f"Rendered {final} ({final.stat().st_size / 1e6:.1f} MB)")

        if dry_run:
            dest = ROOT / "output" / f"{channel_id}_clip.mp4"
            dest.parent.mkdir(exist_ok=True)
            shutil.copy(final, dest)
            print(f"[DRY RUN] Saved to {dest} instead of uploading")
            return

        from src.upload_youtube import upload_video

        tags = list(config.get("tags", []))
        hashtags = " ".join("#" + h for h in config.get("hashtags", ["GTA", "GTA6", "Gaming"]))
        description = (
            f"{title}\n\n{_credit_line(meta)}\n\n"
            f"Subscribe to {config['display_name']} for daily GTA moments!\n\n{hashtags} #Shorts"
        )
        video_id = upload_video(
            str(final), f"{title} #Shorts"[:100], description, tags,
            str(config.get("category_id", "20")),
            str(ROOT / "tokens" / f"{channel_id}_token.pickle"),
            privacy_status=privacy_status, is_short=True,
            made_for_kids=False,
        )
        print(f"Uploaded: https://youtube.com/watch?v={video_id}")

        # only mark the window used AFTER a successful upload
        state["sources"].setdefault(sid, {"used": []})["used"].append([start, end])
        state["sources"].setdefault(src, {"used": []})
        if src != sid:
            state["sources"][src]["used"].append([start, end])
        _save_state(channel_id, state)

        try:
            from src.agents.video_registry import record_video
            record_video(channel_id, {
                "video_id": video_id, "topic": meta.get("title", ""),
                "title": title, "is_short": True, "language": "en",
            })
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: video_registry logging failed ({e}).")


if __name__ == "__main__":
    import argparse
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("channel_id")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cfg = yaml.safe_load((ROOT / "channels" / f"{a.channel_id}.yaml").read_text())
    run_clipping(a.channel_id, cfg, dry_run=a.dry_run)
