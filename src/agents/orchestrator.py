"""Minimal Gemini text helper used by clipper.make_hook_and_title().
If GEMINI_API_KEY is missing or the call fails, the clipper falls back to a
template title, so this is optional."""
import os


def generate_text(prompt: str, max_output_tokens: int = 200, json_mode: bool = False) -> str:
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    config = {"max_output_tokens": max_output_tokens}
    if json_mode:
        config["response_mime_type"] = "application/json"
    resp = genai.Client(api_key=key).models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=config,
    )
    return resp.text or ""
