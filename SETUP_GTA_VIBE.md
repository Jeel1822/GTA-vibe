# GTA Vibe — setup (≈15 minutes)

GTA Vibe is a **clipping** channel: no script / voiceover. Each run finds the
loudest, most exciting moment in a source video, cuts it, reframes to 9:16 with a
blurred backdrop, adds a 3-second hook title + @GTAVibe tag + source credit, and
uploads it as a Short. Schedule: 09:30 / 14:30 / 19:30 UTC (edit in `channels/gta_vibe.yaml`).

## 1. Create the channel (you, in a browser)
1. Sign in to YouTube with **gtavibe4518@gmail.com**.
2. Profile icon → **Create a channel** → name it **GTA Vibe**.
3. Verify the channel by phone at https://www.youtube.com/verify (needed for custom thumbnails and >15 min videos).

## 2. Add that account as a test user
Google Cloud Console → your project → **APIs & Services → OAuth consent screen →
Test users → Add** `gtavibe4518@gmail.com`.

## 3. Authorize the channel (once, on your laptop)
```bash
pip install -r requirements.txt
python -m src.generate_token --client_secret client_secret.json --token tokens/gta_vibe_token.pickle
```
Log in as gtavibe4518@gmail.com, **select the GTA Vibe channel** if asked, approve.

## 4. GitHub secrets
```bash
base64 -i tokens/gta_vibe_token.pickle | pbcopy     # macOS
```
Add repo secret **YOUTUBE_TOKEN_GTA_VIBE** = that value.
(Optional) **YT_COOKIES_B64** — base64 of a cookies.txt exported from a browser, only
needed if YouTube blocks downloads from GitHub's servers.

## 5. Add sources — `channels/gta_vibe_sources.txt`
* YouTube URLs that are **Creative Commons** licensed (the clipper skips others), or
* Videos you own / have written permission for → set `own_content: true` in `gta_vibe.yaml`.
* Local files: put them in `sources/gta_vibe/` (they must be committed for GitHub Actions to see them).

## 6. Test, then go live
```bash
python -m src.clipper gta_vibe --dry-run      # writes output/gta_vibe_clip.mp4, no upload
python -m src.main gta_vibe --short           # real upload
```
Then push to GitHub. The hourly workflow already includes `gta_vibe`.
