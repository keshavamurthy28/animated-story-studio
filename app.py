import asyncio
import base64
import json
import math
import os
import tempfile
import textwrap
import time
import urllib.parse
from io import BytesIO

import edge_tts
import numpy as np
import requests
import streamlit as st
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    from moviepy.editor import (
        AudioFileClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    MOVIEPY_V2 = True

st.set_page_config(
    page_title="AI Cinema & Reels Studio",
    page_icon="🎬",
    layout="wide",
)

# High-readability, comfortable dark studio UI
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 15px;
    }
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    .studio-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        background: linear-gradient(90deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 16px;
    }
    .studio-pill {
        background: #ff2e63;
        color: #ffffff;
        font-weight: 700;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 999px;
        letter-spacing: 0.5px;
    }
    .studio-sub {
        color: #9da7b3;
        font-size: 14px;
    }
    .readable-card {
        background: #161b22;
        border: 1px solid #2b313a;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .stButton > button {
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_CHAR_PROMPT = (
    "Expressive main character with clear facial features, natural posture, and consistent outfit "
    "matching the user's story command and environment"
)

DEFAULT_SCENES = [
    {
        "narration": "The journey begins as the setting comes alive with every detail you commanded.",
        "kannada_narration": "ನೀವು ಬಯಸಿದ ಪ್ರತಿಯೊಂದು ವಿವರದೊಂದಿಗೆ ದೃಶ್ಯ ಜೀವಂತವಾಗುತ್ತಾ ಪ್ರಯಾಣ ಆರಂಭವಾಗುತ್ತದೆ.",
        "emotion": "Focused & Present",
        "motion": "Zoom In",
        "suggested_fx": "Cinematic Vignette",
        "suggested_sfx": "🌲 Forest Birds",
        "visual": (
            "Medium cinematic shot matching the exact environment and action requested; "
            "main character reacting naturally with clear expression and lighting"
        ),
    },
    {
        "narration": "A sudden turn shifts the mood and tests how the character responds in the moment.",
        "kannada_narration": "ಹಠಾತ್ ತಿರುವು ವಾತಾವರಣವನ್ನು ಬದಲಾಯಿಸುತ್ತದೆ ಮತ್ತು ಆ ಕ್ಷಣದ ಪ್ರತಿಕ್ರಿಯೆಯನ್ನು ಪರೀಕ್ಷಿಸುತ್ತದೆ.",
        "emotion": "Alert & Determined",
        "motion": "Pan Right",
        "suggested_fx": "Zoom Punch",
        "suggested_sfx": "💥 Vine Boom (Meme)",
        "visual": (
            "Dynamic medium-wide shot in the requested environment; main character showing "
            "expressive posture, movement, and atmospheric lighting"
        ),
    },
    {
        "narration": "Through changing surroundings and emotion, the story builds toward a clear turning point.",
        "kannada_narration": "ಬದಲಾಗುವ ಪರಿಸರ ಮತ್ತು ಭಾವನೆಗಳ ಮೂಲಕ ಕಥೆಯು ಸ್ಪಷ್ಟ ತಿರುವಿನತ್ತ ಸಾಗುತ್ತದೆ.",
        "emotion": "Reflective & Resilient",
        "motion": "Zoom Out",
        "suggested_fx": "None",
        "suggested_sfx": "🌊 Ocean Waves",
        "visual": (
            "Expressive framing showing the main character and surrounding environment "
            "with rich depth, natural motion, and mood lighting"
        ),
    },
    {
        "narration": "With steady resolve, the scene settles into a memorable closing moment.",
        "kannada_narration": "ಸ್ಥಿರ ನಿರ್ಧಾರದೊಂದಿಗೆ ದೃಶ್ಯವು ಒಂದು ಸ್ಮರಣೀಯ ಅಂತ್ಯದತ್ತ ಸಾಗುತ್ತದೆ.",
        "emotion": "Composed & Hopeful",
        "motion": "Pan Left",
        "suggested_fx": "Cinematic Vignette",
        "suggested_sfx": "🔥 Campfire Crackle",
        "visual": (
            "Balanced cinematic shot of the main character in the commanded setting "
            "with clear composition and clean lighting"
        ),
    },
]

PRESET_AUDIO_EFFECTS = {
    "None": "",
    "🌿 Rain & Thunder": "https://assets.mixkit.co/active_storage/sfx/1250/1250-preview.mp3",
    "🌊 Ocean Waves": "https://assets.mixkit.co/active_storage/sfx/1188/1188-preview.mp3",
    "🔥 Campfire Crackle": "https://assets.mixkit.co/active_storage/sfx/2435/2435-preview.mp3",
    "🌲 Forest Birds": "https://assets.mixkit.co/active_storage/sfx/2436/2436-preview.mp3",
    "⚡ Thunder Strike": "https://assets.mixkit.co/active_storage/sfx/1271/1271-preview.mp3",
    "💨 Wind Gust": "https://assets.mixkit.co/active_storage/sfx/2673/2673-preview.mp3",
    "💥 Vine Boom (Meme)": "https://assets.mixkit.co/active_storage/sfx/2578/2578-preview.mp3",
    "🎺 Airhorn (Reels)": "https://assets.mixkit.co/active_storage/sfx/2805/2805-preview.mp3",
    "📻 Record Scratch": "https://assets.mixkit.co/active_storage/sfx/209/209-preview.mp3",
    "🍌 Cartoon Slip": "https://assets.mixkit.co/active_storage/sfx/2021/2021-preview.mp3",
    "🥁 Rimshot Badum Tss": "https://assets.mixkit.co/active_storage/sfx/2007/2007-preview.mp3",
    "👏 Audience Cheer": "https://assets.mixkit.co/active_storage/sfx/442/442-preview.mp3",
    "😂 Funny Laugh": "https://assets.mixkit.co/active_storage/sfx/475/475-preview.mp3",
}

MOOD_FILTERS = {
    "Original": {"b": 1.0, "c": 1.0, "s": 1.0, "fx": "None"},
    "Vivid Reels": {"b": 1.06, "c": 1.18, "s": 1.42, "fx": "None"},
    "Warm Golden": {"b": 1.06, "c": 1.08, "s": 1.22, "fx": "None"},
    "Cinema Moody": {"b": 0.96, "c": 1.18, "s": 0.85, "fx": "Cinematic Vignette"},
    "Vintage Film": {"b": 1.02, "c": 1.10, "s": 0.72, "fx": "Vintage Sepia"},
    "Comedy Punch": {"b": 1.08, "c": 1.22, "s": 1.32, "fx": "Zoom Punch"},
}

FORMAT_PRESETS = {
    "9:16 Reels / Shorts": (720, 1280, "9:16"),
    "16:9 Movie / YouTube": (1280, 720, "16:9"),
    "1:1 Square Post": (1080, 1080, "1:1"),
    "4:5 Portrait Feed": (1080, 1350, "4:5"),
    "3:4 Classic Portrait": (900, 1200, "3:4"),
}

for key, default in [
    ("scenes", [dict(s) for s in DEFAULT_SCENES]),
    ("images", []),
    ("scene_videos", []),
    ("uploaded_video_clips", []),
    ("uploaded_audio_clips", []),
    ("audio_files", []),
    ("bgm_audio_path", ""),
    ("bgm_volume", 0.22),
    ("voice_volume", 1.0),
    ("sfx_volume", 0.75),
    ("own_voice_table", {}),
    ("locked_character_prompt", DEFAULT_CHAR_PROMPT),
    ("exact_character_count", 1),
    ("lock_character_enabled", False),
    ("populated_script_lines", []),
    ("populated_script_lang", "English"),
    ("gen_batch_id", 0),
    ("selected_clip_idx", 0),
    ("format_preset", "9:16 Reels / Shorts"),
    ("bg_fit_mode", "Blur Canvas"),
    ("playback_speed", 1.0),
    ("custom_title_text", ""),
    ("custom_title_lang", "English"),
    ("editor_brightness", 1.0),
    ("editor_contrast", 1.0),
    ("editor_saturation", 1.0),
    ("editor_blur", 0.0),
    ("editor_transition", "Crossfade"),
    ("editor_transition_sec", 0.5),
    ("editor_video_fx", "None"),
    ("scene_audio_fx", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

for idx, sc in enumerate(st.session_state.scenes):
    st.session_state.setdefault(f"nar_{idx}", sc.get("narration", ""))
    st.session_state.setdefault(f"kn_script_{idx}", sc.get("kannada_narration", ""))
    st.session_state.setdefault(f"vis_{idx}", sc.get("visual", ""))
    st.session_state.setdefault(f"emo_{idx}", sc.get("emotion", "Expressive"))
    st.session_state.setdefault(f"mot_{idx}", sc.get("motion", "Auto Alternate"))


def run_async(coro):
    """Safely run async coroutines inside Streamlit's event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def apply_duration(clip, duration):
    return clip.with_duration(duration) if MOVIEPY_V2 else clip.set_duration(duration)


def apply_start(clip, start_t):
    return clip.with_start(start_t) if MOVIEPY_V2 else clip.set_start(start_t)


def apply_audio(clip, audio):
    return clip.with_audio(audio) if MOVIEPY_V2 else clip.set_audio(audio)


def apply_position(clip, pos_fn):
    return clip.with_position(pos_fn) if MOVIEPY_V2 else clip.set_position(pos_fn)


def scale_audio_volume(clip, factor):
    if clip is None or abs(factor - 1.0) < 1e-3:
        return clip
    if MOVIEPY_V2 and hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(factor)
    if hasattr(clip, "volumex"):
        return clip.volumex(factor)
    return clip


def ensure_exact_size(pil_img, target_w, target_h, bg_mode="Blur Canvas"):
    """Place any image onto an exact (target_w, target_h) canvas using Blur Canvas or Crop Fill."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    rgb = pil_img.convert("RGB")
    if rgb.size == (target_w, target_h):
        return rgb

    if bg_mode == "Blur Canvas":
        bg_scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
        bg_w = max(target_w, int(round(rgb.width * bg_scale)))
        bg_h = max(target_h, int(round(rgb.height * bg_scale)))
        bg = rgb.resize((bg_w, bg_h), resampling)
        bx = max(0, (bg_w - target_w) // 2)
        by = max(0, (bg_h - target_h) // 2)
        bg = bg.crop((bx, by, bx + target_w, by + target_h)).filter(ImageFilter.GaussianBlur(radius=18))
        bg = ImageEnhance.Brightness(bg).enhance(0.62)

        fg_scale = min(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
        fg_w = max(1, int(round(rgb.width * fg_scale)))
        fg_h = max(1, int(round(rgb.height * fg_scale)))
        fg = rgb.resize((fg_w, fg_h), resampling)
        fx = (target_w - fg_w) // 2
        fy = (target_h - fg_h) // 2
        bg.paste(fg, (fx, fy))
        return bg

    scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
    nw = max(target_w, int(round(rgb.width * scale)))
    nh = max(target_h, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), resampling)
    left = max(0, (nw - target_w) // 2)
    top = max(0, (nh - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h)).resize(
        (target_w, target_h), resampling
    )


def apply_pil_effects(
    pil_img,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
    blur_radius=0.0,
    video_fx="None",
):
    """Apply live studio color, filter, and video preset FX to a PIL frame."""
    out = pil_img.convert("RGB")
    if abs(brightness - 1.0) > 1e-3:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if abs(contrast - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if abs(saturation - 1.0) > 1e-3:
        out = ImageEnhance.Color(out).enhance(saturation)
    if blur_radius and blur_radius > 0.05:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    if video_fx == "Vintage Sepia":
        sepia_matrix = (
            0.393, 0.769, 0.189, 0,
            0.349, 0.686, 0.168, 0,
            0.272, 0.534, 0.131, 0,
        )
        out = out.convert("RGB", sepia_matrix)
    elif video_fx == "Cinematic Vignette":
        w, h = out.size
        vig = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(vig)
        draw.ellipse([-w * 0.2, -h * 0.2, w * 1.2, h * 1.2], fill=0)
        vig = vig.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.25))
        dark = Image.new("RGB", (w, h), (0, 0, 0))
        out = Image.composite(out, dark, vig)
    elif video_fx == "Flash White":
        out = ImageEnhance.Brightness(out).enhance(1.42)
    elif video_fx == "Glitch RGB":
        r, g, b = out.split()
        r = ImageEnhance.Brightness(r).enhance(1.2)
        b = ImageEnhance.Contrast(b).enhance(1.3)
        out = Image.merge("RGB", (r, g, b))

    return out


def download_fx_audio(fx_name, fx_url):
    """Download and cache preset audio effects locally."""
    if not fx_url:
        return None
    os.makedirs("fx_audio_cache", exist_ok=True)
    safe_name = "".join(c for c in fx_name if c.isalnum() or c in ("_", "-")).strip() or "sfx"
    out_p = os.path.join("fx_audio_cache", f"{safe_name}.mp3")
    if os.path.exists(out_p) and os.path.getsize(out_p) > 2000:
        return out_p
    try:
        r = requests.get(fx_url, timeout=20)
        if r.ok and len(r.content) > 1000:
            with open(out_p, "wb") as f:
                f.write(r.content)
            return out_p
    except Exception:
        pass
    return None


def _is_valid_font_file(path):
    return os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 40000


def _find_font(font_size, lang="English"):
    """Load a suitable TrueType font for English or Kannada, fetching Noto Sans Kannada if needed."""
    if lang == "Kannada":
        kannada_candidates = [
            "fonts/NotoSansKannada-Regular.ttf",
            "fonts/NotoSansKannada-Bold.ttf",
            "NotoSansKannada-Regular.ttf",
            "NotoSansKannada-Bold.ttf",
            "/tmp/NotoSansKannada-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansKannada-Bold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansKannada-Regular.otf",
            "/usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf",
        ]
        for path in kannada_candidates:
            if _is_valid_font_file(path):
                try:
                    return ImageFont.truetype(path, font_size), path
                except Exception:
                    pass

        tmp_font = "/tmp/NotoSansKannada-Regular.ttf"
        font_urls = [
            "https://raw.githubusercontent.com/notofonts/kannada/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf",
            "https://github.com/google/fonts/raw/main/ofl/notosanskannada/static/NotoSansKannada-Regular.ttf",
        ]
        for url in font_urls:
            try:
                r = requests.get(url, timeout=25)
                if r.ok and len(r.content) > 40000 and b"<!DOCTYPE" not in r.content[:200]:
                    with open(tmp_font, "wb") as f:
                        f.write(r.content)
                    if _is_valid_font_file(tmp_font):
                        return ImageFont.truetype(tmp_font, font_size), tmp_font
            except Exception:
                continue

    fallback_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for path in fallback_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size), path
            except Exception:
                pass
    return ImageFont.load_default(), "default_bitmap"


def translate_lines_to_kannada(gemini_key, lines):
    """Translate a list of narration lines into Kannada using Gemini when available."""
    if not gemini_key or not gemini_key.strip() or not lines:
        return lines

    client = genai.Client(api_key=gemini_key.strip())
    prompt = (
        "Translate each string in this JSON array into natural, simple, clear Kannada. "
        "Return ONLY a valid JSON array of strings in the exact same order, with no markdown.\n"
        + json.dumps(lines, ensure_ascii=False)
    )
    for model_name in [
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            raw = (resp.text or "").strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines() if not line.strip().startswith("```")
                ).strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(lines):
                return [str(item).strip() for item in parsed]
        except Exception:
            continue
    return lines


def infer_ml_scene_direction(text, idx):
    """Smart AI/ML-style heuristic director for motion, VFX, and SFX when offline or no LLM key."""
    low = (text or "").lower()
    if any(w in low for w in ["funny", "joke", "oops", "slip", "laugh", "comedy", "silly", "meme"]):
        return "Zoom Punch", "Zoom Punch", "😂 Funny Laugh"
    if any(w in low for w in ["rain", "storm", "thunder", "lightning"]):
        return "Pan Left", "Cinematic Vignette", "🌿 Rain & Thunder"
    if any(w in low for w in ["ocean", "sea", "wave", "beach", "water", "river"]):
        return "Pan Right", "None", "🌊 Ocean Waves"
    if any(w in low for w in ["fire", "flame", "camp", "warm", "torch"]):
        return "Zoom In", "Cinematic Vignette", "🔥 Campfire Crackle"
    if any(w in low for w in ["forest", "jungle", "tree", "bird", "nature", "morning"]):
        return "Pan Right", "None", "🌲 Forest Birds"
    if any(w in low for w in ["shock", "twist", "sudden", "boom", "dramatic", "wow"]):
        return "Zoom Punch", "Shake", "💥 Vine Boom (Meme)"
    motions = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    return motions[idx % len(motions)], "None", "None"


def build_fallback_scenes_from_rough_story(rough_text, count, char_prompt):
    """Build up to 10 scenes directly from the user's rough story, character, and setting commands."""
    target_count = max(1, min(10, int(count)))
    clean_story = " ".join((rough_text or "").strip().split())
    clean_char = " ".join((char_prompt or "").strip().split())
    sentences = [
        s.strip()
        for s in clean_story.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    beat_moods = [
        ("Opening & Atmosphere", "establishing the requested environment and mood"),
        ("Action & Engagement", "character actively interacting with the surroundings"),
        ("Turning Point & Tension", "shift in mood, lighting, or situation"),
        ("Movement & Exploration", "dynamic motion through the scene setting"),
        ("Emotional Reaction", "expressive response to unfolding events"),
        ("Closer Look & Detail", "focused moment highlighting key action and environment"),
        ("Rising Momentum", "growing energy and purposeful movement"),
        ("Reflection & Depth", "quieter, atmospheric moment in the scene"),
        ("Breakthrough", "clear shift toward resolution or discovery"),
        ("Closing Frame", "memorable concluding shot matching the story direction"),
    ]
    scenes = []
    subject_prefix = f"{clean_char}; " if clean_char else ""
    for idx in range(target_count):
        base_line = sentences[idx % len(sentences)] if sentences else (clean_story or "Custom scene")
        emo_label, beat_desc = beat_moods[idx % len(beat_moods)]
        motion_sug, fx_sug, sfx_sug = infer_ml_scene_direction(f"{base_line} {clean_char}", idx)
        scenes.append({
            "narration": f"{base_line}.",
            "kannada_narration": "",
            "emotion": emo_label,
            "motion": motion_sug,
            "suggested_fx": fx_sug,
            "suggested_sfx": sfx_sug,
            "visual": (
                f"Cinematic shot ({beat_desc}) for: {base_line}. "
                f"{subject_prefix}Follow all user commands for character, background, nature, or location accurately."
            ).strip(),
        })
    return scenes


def sync_scenes_to_count(existing_scenes, desired_count, rough_text, char_prompt):
    """Ensure st.session_state.scenes has the exact desired_count (up to 10 scenes)."""
    target_n = max(1, min(10, int(desired_count)))
    base_story = (
        rough_text.strip()
        if rough_text and rough_text.strip()
        else "Custom story following user instructions for characters and environment"
    )
    extra_pool = build_fallback_scenes_from_rough_story(base_story, target_n, char_prompt)

    current = [dict(s) for s in (existing_scenes or [])[:target_n]]
    while len(current) < target_n:
        idx = len(current)
        current.append(dict(extra_pool[idx]))
    return current


def get_script_lines_for_language(scenes, script_lang="English", gemini_key=""):
    """Extract English or Kannada script lines per scene, translating if Kannada isn't written yet."""
    if not scenes:
        return []

    en_lines = [(s.get("narration") or f"Scene {i + 1}.").strip() for i, s in enumerate(scenes)]
    if script_lang != "Kannada":
        return en_lines

    kn_lines = [(s.get("kannada_narration") or "").strip() for s in scenes]
    missing_indices = [i for i, line in enumerate(kn_lines) if not line]

    if missing_indices and gemini_key and gemini_key.strip():
        to_translate = [en_lines[i] for i in missing_indices]
        translated = translate_lines_to_kannada(gemini_key, to_translate)
        for idx_pos, scene_idx in enumerate(missing_indices):
            if idx_pos < len(translated) and translated[idx_pos]:
                kn_lines[scene_idx] = translated[idx_pos]
                scenes[scene_idx]["kannada_narration"] = translated[idx_pos]

    return [
        kn_lines[i] if kn_lines[i] else en_lines[i]
        for i in range(len(scenes))
    ]


def draw_subtitle_on_frame(frame_pil, subtitle_text, target_w, target_h, lang="English"):
    """Draw a readable subtitle/caption banner near the bottom of a scene frame."""
    if not subtitle_text or not subtitle_text.strip() or lang == "None":
        return frame_pil

    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    font_size = max(26, int(round(target_w * 0.038)))
    font, _ = _find_font(font_size, lang=lang)
    lang_code = "kn" if lang == "Kannada" else "en"

    chars_per_line = max(14, int(target_w / max(11, font_size * 0.64)))
    wrapped_lines = textwrap.wrap(subtitle_text.strip(), width=chars_per_line)[:3]
    if not wrapped_lines:
        return frame_pil

    line_h = font_size + 14
    box_h = len(wrapped_lines) * line_h + 24
    pad_x = max(20, int(target_w * 0.04))
    box_top = max(16, target_h - box_h - max(42, int(target_h * 0.07)))
    box_bot = min(target_h - 16, box_top + box_h)

    draw.rounded_rectangle(
        [pad_x, box_top, target_w - pad_x, box_bot],
        radius=14,
        fill=(10, 14, 22, 215),
    )

    cur_y = box_top + 12
    for line in wrapped_lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font, language=lang_code)
        except TypeError:
            bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = max(pad_x + 12, (target_w - text_w) // 2)

        for ox, oy, col in [
            (text_x + 2, cur_y + 2, (0, 0, 0, 240)),
            (text_x, cur_y, (255, 250, 220, 255)),
        ]:
            try:
                draw.text((ox, oy), line, font=font, fill=col, language=lang_code)
            except TypeError:
                draw.text((ox, oy), line, font=font, fill=col)
        cur_y += line_h

    return img_rgba.convert("RGB")


def build_motion_clip(
    img_path,
    target_w,
    target_h,
    duration,
    motion_mode="Auto Alternate",
    scene_idx=0,
    subtitle_text="",
    subtitle_lang="None",
    enable_char_motion=True,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Create animated scene frames with exact target_w x target_h dimensions and studio FX."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    overscan = 1.16
    canvas_w = int(round(target_w * overscan))
    canvas_h = int(round(target_h * overscan))

    with Image.open(img_path) as im:
        exact_base = ensure_exact_size(im, canvas_w, canvas_h, bg_mode=bg_mode)
        canvas_np = np.asarray(exact_base, dtype=np.uint8)

    safe_dur = max(float(duration), 0.5)
    num_steps = max(6, min(20, int(round(safe_dur * 4))))
    step_dur = safe_dur / num_steps
    subclips = []

    os.makedirs("motion_frames", exist_ok=True)
    for s_idx in range(num_steps):
        progress = s_idx / max(1, num_steps - 1)
        ease = progress * progress * (3.0 - 2.0 * progress)

        char_sway_x = int(round(math.sin(s_idx * 1.1 + scene_idx) * 6)) if enable_char_motion else 0
        char_bob_y = int(round(abs(math.cos(s_idx * 1.4 + scene_idx)) * 5)) if enable_char_motion else 0

        if video_fx == "Zoom Punch":
            scale = 1.0 + 0.22 * abs(math.sin(progress * math.pi))
        elif active_mode == "Zoom In":
            scale = 1.0 + 0.12 * ease
        elif active_mode == "Zoom Out":
            scale = 1.12 - 0.12 * ease
        else:
            scale = 1.05

        win_w = max(2, int(round(target_w / scale)))
        win_h = max(2, int(round(target_h / scale)))

        if active_mode == "Pan Right":
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * ease)) + char_sway_x
            y0 = max(0, (canvas_h - win_h) // 2) + char_bob_y
        elif active_mode == "Pan Left":
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * (1.0 - ease))) + char_sway_x
            y0 = max(0, (canvas_h - win_h) // 2) + char_bob_y
        else:
            x0 = (canvas_w - win_w) // 2 + char_sway_x
            y0 = (canvas_h - win_h) // 2 + char_bob_y

        if video_fx == "Shake":
            x0 += int(round(math.sin(s_idx * 3.5) * 8))
            y0 += int(round(math.cos(s_idx * 4.2) * 6))

        x0 = min(max(0, x0), max(0, canvas_w - win_w))
        y0 = min(max(0, y0), max(0, canvas_h - win_h))
        cropped = canvas_np[y0 : y0 + win_h, x0 : x0 + win_w]

        frame_pil = Image.fromarray(cropped)
        if frame_pil.size != (target_w, target_h):
            frame_pil = frame_pil.resize((target_w, target_h), resampling)

        frame_pil = apply_pil_effects(
            frame_pil,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            blur_radius=blur_radius,
            video_fx=video_fx,
        )

        if subtitle_lang != "None" and subtitle_text:
            frame_pil = draw_subtitle_on_frame(
                frame_pil,
                subtitle_text=subtitle_text,
                target_w=target_w,
                target_h=target_h,
                lang=subtitle_lang,
            )

        frame_pil = ensure_exact_size(frame_pil, target_w, target_h, bg_mode="Crop Fill")
        frame_path = os.path.join("motion_frames", f"s{scene_idx}_f{s_idx}.jpg")
        frame_pil.save(frame_path, format="JPEG", quality=92)
        subclips.append(apply_duration(ImageClip(frame_path), step_dur))

    return concatenate_videoclips(subclips, method="chain")


def render_single_scene_video(
    img_path,
    out_path,
    target_w,
    target_h,
    duration=4.0,
    motion_mode="Auto Alternate",
    scene_idx=0,
    subtitle_text="",
    subtitle_lang="None",
    enable_char_motion=True,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Render and save a standalone MP4 video clip for a single scene."""
    clip = build_motion_clip(
        img_path=img_path,
        target_w=target_w,
        target_h=target_h,
        duration=duration,
        motion_mode=motion_mode,
        scene_idx=scene_idx,
        subtitle_text=subtitle_text,
        subtitle_lang=subtitle_lang,
        enable_char_motion=enable_char_motion,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        blur_radius=blur_radius,
        video_fx=video_fx,
        bg_mode=bg_mode,
    )
    try:
        clip.write_videofile(
            out_path,
            fps=24,
            codec="libx264",
            audio=False,
            logger=None,
        )
        return out_path
    finally:
        try:
            clip.close()
        except Exception:
            pass


def build_local_bypass_scene(prompt_text, target_w, target_h, scene_idx, emotion_label=""):
    """Clean, atmospheric storyboard frame when external AI image APIs fail/quota-out."""
    p_low = (prompt_text or "").lower()
    if any(k in p_low for k in ["forest", "jungle", "tree", "woods", "nature", "mountain", "valley", "river"]):
        palettes = [
            ((18, 48, 38), (44, 96, 68), (12, 28, 22)),
            ((24, 58, 52), (58, 114, 92), (14, 32, 28)),
            ((34, 48, 38), (108, 92, 58), (18, 26, 18)),
            ((16, 42, 56), (46, 98, 94), (10, 24, 32)),
        ]
    elif any(k in p_low for k in ["sunset", "sunrise", "golden", "dusk", "dawn", "beach", "ocean", "sea"]):
        palettes = [
            ((42, 26, 58), (156, 78, 58), (20, 16, 30)),
            ((32, 50, 84), (198, 122, 68), (18, 26, 42)),
            ((26, 38, 66), (136, 74, 84), (14, 20, 32)),
            ((52, 36, 64), (178, 106, 64), (20, 16, 28)),
        ]
    elif any(k in p_low for k in ["night", "space", "stars", "moon", "cyber", "neon"]):
        palettes = [
            ((10, 16, 32), (28, 42, 78), (6, 10, 20)),
            ((18, 12, 36), (58, 32, 88), (10, 8, 20)),
            ((8, 24, 34), (24, 72, 88), (6, 14, 20)),
            ((24, 14, 32), (78, 30, 68), (12, 8, 18)),
        ]
    else:
        palettes = [
            ((22, 34, 56), (44, 64, 94), (14, 20, 32)),
            ((36, 28, 54), (68, 54, 92), (18, 14, 28)),
            ((20, 34, 48), (96, 72, 56), (14, 20, 28)),
            ((26, 44, 42), (54, 84, 76), (14, 24, 22)),
        ]

    top_col, mid_col, bot_col = palettes[scene_idx % len(palettes)]

    y_grid = np.linspace(0.0, 1.0, target_h, dtype=np.float32)[:, None, None]
    x_grid = np.linspace(0.0, 1.0, target_w, dtype=np.float32)[None, :, None]
    c_top = np.array(top_col, dtype=np.float32)[None, None, :]
    c_mid = np.array(mid_col, dtype=np.float32)[None, None, :]
    c_bot = np.array(bot_col, dtype=np.float32)[None, None, :]

    blend_upper = np.clip(y_grid / 0.58, 0.0, 1.0)
    blend_lower = np.clip((y_grid - 0.58) / 0.42, 0.0, 1.0)
    sky_wall = c_top * (1.0 - blend_upper) + c_mid * blend_upper
    full_bg = sky_wall * (1.0 - blend_lower) + c_bot * blend_lower

    sun_cx = target_w * (0.28 + 0.14 * (scene_idx % 4))
    sun_cy = target_h * (0.24 + 0.04 * (scene_idx % 3))
    dist_sq = (x_grid[:, :, 0] - sun_cx) ** 2 + (y_grid[:, :, 0] - sun_cy) ** 2
    glow_r = min(target_w, target_h) * 0.24
    glow = np.exp(-dist_sq / (2.0 * glow_r ** 2))[:, :, None]
    full_bg = np.clip(full_bg + glow * np.array([64.0, 52.0, 38.0], dtype=np.float32), 0, 255)

    img = Image.fromarray(full_bg.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    horizon_y = int(target_h * 0.64)
    ridge_pts = [(0, target_h)]
    for x_step in range(0, target_w + 24, 24):
        wave = (
            math.sin(x_step * 0.008 + scene_idx * 1.2) * (target_h * 0.035)
            + math.cos(x_step * 0.02 - scene_idx) * (target_h * 0.015)
        )
        ridge_pts.append((x_step, int(horizon_y + wave)))
    ridge_pts.append((target_w, target_h))
    draw.polygon(ridge_pts, fill=(14, 20, 30, 175))

    cx = int(target_w * (0.50 + 0.06 * math.sin(scene_idx * 1.5)))
    head_r = max(26, int(min(target_w, target_h) * 0.05))
    head_cy = int(target_h * 0.46) + int(round(math.sin(scene_idx * 1.2) * 8))

    draw.rounded_rectangle(
        [cx - int(head_r * 1.9), head_cy + head_r + 6, cx + int(head_r * 1.9), head_cy + head_r + int(head_r * 3.2)],
        radius=28,
        fill=(28, 36, 52, 230),
        outline=(235, 195, 115, 120),
        width=2,
    )
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=(214, 170, 140, 250))
    draw.pieslice([cx - head_r, head_cy - head_r - 2, cx + head_r, head_cy + head_r - 6], 185, 355, fill=(36, 28, 24, 255))

    eye_y = head_cy - 2
    eye_dx = max(9, int(head_r * 0.35))
    draw.ellipse([cx - eye_dx - 4, eye_y - 4, cx - eye_dx + 4, eye_y + 4], fill=(22, 24, 30, 255))
    draw.ellipse([cx + eye_dx - 4, eye_y - 4, cx + eye_dx + 4, eye_y + 4], fill=(22, 24, 30, 255))
    draw.arc([cx - 10, head_cy + 8, cx + 10, head_cy + 22], 15, 165, fill=(115, 58, 52, 255), width=2)

    card_w = int(target_w * 0.88)
    card_h = int(target_h * 0.18)
    cx0 = (target_w - card_w) // 2
    cy0 = int(target_h * 0.73)
    draw.rounded_rectangle(
        [cx0, cy0, cx0 + card_w, cy0 + card_h],
        radius=16,
        fill=(10, 14, 22, 210),
        outline=(255, 46, 99, 175),
        width=2,
    )

    title_font, _ = _find_font(max(18, int(target_w * 0.025)), lang="English")
    body_font, _ = _find_font(max(16, int(target_w * 0.021)), lang="English")
    emo_badge = f" • {emotion_label.upper()}" if emotion_label else ""
    draw.text(
        (cx0 + 18, cy0 + 12),
        f"SCENE {scene_idx + 1}{emo_badge}",
        font=title_font,
        fill=(255, 118, 154, 255),
    )

    clean_action = " ".join((prompt_text or "").split())
    wrapped = textwrap.wrap(clean_action, width=max(26, int(card_w / max(10, int(target_w * 0.021) * 0.56))))[:3]
    cur_y = cy0 + 44
    for line in wrapped:
        draw.text((cx0 + 18, cur_y), line, font=body_font, fill=(242, 246, 252, 245))
        cur_y += int(target_w * 0.025)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    return ensure_exact_size(combined, target_w, target_h, bg_mode="Crop Fill")


def modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, art_style):
    """Adapt an uploaded reference photo/image into an exact target_w x target_h scene frame."""
    base = ensure_exact_size(ref_pil, target_w, target_h, bg_mode="Crop Fill")
    style_lower = (art_style or "").lower()
    if "ghibli" in style_lower or "watercolor" in style_lower:
        framed = base.filter(ImageFilter.SMOOTH_MORE)
        framed = ImageEnhance.Color(framed).enhance(1.18)
    else:
        framed = ImageEnhance.Contrast(base).enhance(1.12)
        framed = ImageEnhance.Color(framed).enhance(1.14)

    mood_factors = [1.0, 1.05, 0.96, 1.03, 0.98, 1.05, 1.02, 0.95, 1.06, 1.01]
    framed = ImageEnhance.Brightness(framed).enhance(mood_factors[scene_idx % len(mood_factors)])
    return ensure_exact_size(framed, target_w, target_h, bg_mode="Crop Fill")


def generate_cloudflare_flux_image(cf_account_id, cf_api_token, prompt_text, target_w, target_h):
    """Call Cloudflare Workers AI FLUX.1 [schnell] REST API."""
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id.strip()}"
        "/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    headers = {
        "Authorization": f"Bearer {cf_api_token.strip()}",
        "Content-Type": "application/json",
    }
    aspect_hint = (
        "vertical 9:16 portrait composition"
        if target_h > target_w
        else "horizontal 16:9 landscape composition"
    )
    bounded_prompt = f"{prompt_text.strip()}, {aspect_hint}"[:2048]
    payload = {"prompt": bounded_prompt}
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    content_type = resp.headers.get("Content-Type", "")
    if "image/" in content_type:
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return ensure_exact_size(img, target_w, target_h, bg_mode="Crop Fill")

    data = resp.json()
    result = data.get("result", {}) if isinstance(data, dict) else {}
    b64_img = result.get("image")
    if b64_img:
        raw_bytes = base64.b64decode(b64_img)
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        return ensure_exact_size(img, target_w, target_h, bg_mode="Crop Fill")

    raise ValueError(f"Unexpected Cloudflare response: {data}")


def generate_pollinations_image(prompt_text, target_w, target_h, seed=42):
    """Fallback AI image generator via Pollinations.ai when primary quotas are reached."""
    w_p = max(256, min(1024, (target_w // 8) * 8))
    h_p = max(256, min(1024, (target_h // 8) * 8))
    short_prompt = prompt_text[:320]
    encoded_prompt = urllib.parse.quote(short_prompt, safe="")
    endpoints = [
        f"https://gen.pollinations.ai/image/{encoded_prompt}?model=flux&width={w_p}&height={h_p}&seed={seed}",
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={w_p}&height={h_p}&seed={seed}&nologo=true",
    ]
    last_err = None
    for url in endpoints:
        try:
            resp = requests.get(url, timeout=60)
            if resp.ok and "image/" in resp.headers.get("Content-Type", ""):
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return ensure_exact_size(img, target_w, target_h, bg_mode="Crop Fill")
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Pollinations fallback failed: {last_err}")


def generate_real_ai_scene(
    engine_choice,
    cf_account_id,
    cf_api_token,
    hf_token,
    gemini_key,
    prompt_text,
    ratio_str,
    target_w,
    target_h,
    scene_idx,
    ref_pil=None,
    allow_local_bypass=True,
    emotion_label="",
    seed_offset=0,
):
    """Generate AI scene images with automatic failover across engines and local bypass fallback."""
    errors = []
    scene_seed = 100 + scene_idx * 37 + seed_offset

    def try_cloudflare():
        if cf_account_id.strip() and cf_api_token.strip():
            img = generate_cloudflare_flux_image(
                cf_account_id, cf_api_token, prompt_text, target_w, target_h
            )
            return img, "Cloudflare FLUX.1-schnell"
        raise RuntimeError("Missing Cloudflare Account ID or API Token.")

    def try_huggingface():
        if not hf_token.strip():
            raise RuntimeError("Missing Hugging Face Token.")
        client = InferenceClient(token=hf_token.strip())
        w_hf = max(512, min(1024, (target_w // 8) * 8))
        h_hf = max(512, min(1024, (target_h // 8) * 8))
        last_hf_err = None
        for model_id in [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ]:
            try:
                img = client.text_to_image(
                    prompt=prompt_text,
                    model=model_id,
                    width=w_hf,
                    height=h_hf,
                )
                return ensure_exact_size(img.convert("RGB"), target_w, target_h, bg_mode="Crop Fill"), f"HF {model_id}"
            except Exception as exc:
                last_hf_err = exc
        raise RuntimeError(str(last_hf_err))

    def try_gemini():
        if not gemini_key.strip():
            raise RuntimeError("Missing Gemini API Key.")
        client = genai.Client(api_key=gemini_key.strip())
        contents = [ref_pil.convert("RGB"), prompt_text] if ref_pil is not None else prompt_text
        last_gem_err = None
        for model_id in ["gemini-3.1-flash-lite-image", "gemini-3.1-flash-image"]:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=ratio_str),
                    ),
                )
                for cand in getattr(response, "candidates", None) or []:
                    content = getattr(cand, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            img = Image.open(BytesIO(inline.data)).convert("RGB")
                            return ensure_exact_size(img, target_w, target_h, bg_mode="Crop Fill"), f"Gemini {model_id}"
                last_gem_err = RuntimeError(f"{model_id}: no image returned")
            except Exception as exc:
                last_gem_err = exc
        raise RuntimeError(str(last_gem_err))

    def try_pollinations():
        img = generate_pollinations_image(prompt_text, target_w, target_h, seed=200 + scene_seed)
        return img, "Pollinations AI Fallback"

    if engine_choice == "Cloudflare Workers AI (Free Daily FLUX)":
        ordered_runners = [
            ("Cloudflare", try_cloudflare),
            ("Hugging Face", try_huggingface),
            ("Gemini", try_gemini),
            ("Pollinations", try_pollinations),
        ]
    elif engine_choice == "Hugging Face (FLUX / SDXL)":
        ordered_runners = [
            ("Hugging Face", try_huggingface),
            ("Cloudflare", try_cloudflare),
            ("Gemini", try_gemini),
            ("Pollinations", try_pollinations),
        ]
    else:
        ordered_runners = [
            ("Gemini", try_gemini),
            ("Cloudflare", try_cloudflare),
            ("Hugging Face", try_huggingface),
            ("Pollinations", try_pollinations),
        ]

    for label, runner in ordered_runners:
        try:
            img, src_name = runner()
            return img, src_name, errors
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    if ref_pil is not None:
        return (
            modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, prompt_text),
            "Uploaded Reference Modification",
            errors,
        )

    if allow_local_bypass:
        return (
            build_local_bypass_scene(prompt_text, target_w, target_h, scene_idx, emotion_label=emotion_label),
            "Local Storyboard Bypass",
            errors,
        )

    return None, None, errors


def apply_clip_transition(clips, transition_name="Crossfade", trans_sec=0.5):
    """Apply smooth transitions across a list of MoviePy video clips."""
    if not clips:
        return None
    if len(clips) == 1 or transition_name == "None" or trans_sec <= 0.05:
        return concatenate_videoclips(clips, method="compose")

    positioned = []
    cur_start = 0.0
    for idx, clp in enumerate(clips):
        dur = max(float(clp.duration or 1.0), 0.2)
        t_dur = min(float(trans_sec), dur * 0.4)
        start_t = max(0.0, cur_start - (t_dur if idx > 0 else 0.0))

        c_mod = clp
        if transition_name in ("Fade", "Crossfade"):
            if hasattr(c_mod, "crossfadein") and idx > 0:
                c_mod = c_mod.crossfadein(t_dur)
            elif hasattr(c_mod, "fadein") and idx > 0:
                c_mod = c_mod.fadein(t_dur)
        elif transition_name == "Slide Left" and idx > 0:
            w_c = getattr(c_mod, "w", 720)
            c_mod = apply_position(
                c_mod,
                lambda t, wc=w_c, td=t_dur: (int(round(wc * max(0.0, 1.0 - t / max(0.05, td)))), 0) if t < td else (0, 0),
            )
        elif transition_name == "Slide Right" and idx > 0:
            w_c = getattr(c_mod, "w", 720)
            c_mod = apply_position(
                c_mod,
                lambda t, wc=w_c, td=t_dur: (-int(round(wc * max(0.0, 1.0 - t / max(0.05, td)))), 0) if t < td else (0, 0),
            )

        positioned.append(apply_start(c_mod, start_t))
        cur_start = start_t + dur

    return CompositeVideoClip(positioned)


# --- TOP STUDIO BANNER ---
st.markdown(
    """
    <div class="studio-header">
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="studio-pill">AI STUDIO PRO</span>
            <span style="font-weight:700;font-size:18px;">Movie, Shorts & Reels Creation Suite</span>
        </div>
        <div class="studio-sub">AI Scene Direction • Live Canvas Preview • Voice, BGM & Sound FX • 9:16 & 16:9 Export</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR: FORMAT, AI KEYS & CHARACTER COMMANDS ---
with st.sidebar:
    st.header("🎬 Format & AI Setup")
    st.session_state.format_preset = st.selectbox(
        "Video Format Preset",
        list(FORMAT_PRESETS.keys()),
        index=list(FORMAT_PRESETS.keys()).index(st.session_state.format_preset),
    )
    width, height, ratio_str = FORMAT_PRESETS[st.session_state.format_preset]
    st.caption(f"Resolution: **{width} × {height}** (`{ratio_str}`)")

    with st.expander("🔑 API Keys & Image Engine", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password")
        image_engine = st.selectbox(
            "AI Scene Image Engine",
            [
                "Cloudflare Workers AI (Free Daily FLUX)",
                "Hugging Face (FLUX / SDXL)",
                "Gemini Image",
            ],
        )
        cf_account_id = st.text_input("Cloudflare Account ID", type="password")
        cf_api_token = st.text_input("Cloudflare API Token", type="password")
        hf_token = st.text_input("Hugging Face Token (optional)", type="password")

    st.divider()
    st.subheader("🎭 Character & Setting Command")
    st.session_state.lock_character_enabled = st.checkbox(
        "Keep character & environment lock active",
        value=st.session_state.lock_character_enabled,
    )
    st.session_state.exact_character_count = st.number_input(
        "Main Characters",
        min_value=1,
        max_value=8,
        value=int(st.session_state.exact_character_count),
        step=1,
    )
    st.session_state.locked_character_prompt = st.text_area(
        "Character / Background / Nature Command",
        value=st.session_state.locked_character_prompt,
        height=110,
    )

# --- TOP VISUAL STAGE & FRIENDLY EDITOR ---
with st.expander("🖥️ Live Preview Stage & Quick Studio Controls", expanded=True):
    stage_left, stage_right = st.columns([1.2, 1.8])

    with stage_left:
        avail_imgs = [p for p in st.session_state.images if os.path.exists(p)]
        avail_vids = [p for p in st.session_state.scene_videos if os.path.exists(p)] + [
            p for p in st.session_state.uploaded_video_clips if os.path.exists(p)
        ]

        monitor_mode = st.radio(
            "Preview Mode",
            ["Live Frame + Effects", "Rendered Clip"],
            horizontal=True,
            key="top_monitor_mode",
        )

        if monitor_mode == "Live Frame + Effects" and avail_imgs:
            c_idx = min(int(st.session_state.selected_clip_idx), len(avail_imgs) - 1)
            with Image.open(avail_imgs[c_idx]) as im_raw:
                prev_img = ensure_exact_size(im_raw, width, height, bg_mode=st.session_state.bg_fit_mode)
            prev_img = apply_pil_effects(
                prev_img,
                brightness=st.session_state.editor_brightness,
                contrast=st.session_state.editor_contrast,
                saturation=st.session_state.editor_saturation,
                blur_radius=st.session_state.editor_blur,
                video_fx=st.session_state.editor_video_fx,
            )
            if st.session_state.custom_title_text.strip():
                prev_img = draw_subtitle_on_frame(
                    prev_img,
                    subtitle_text=st.session_state.custom_title_text,
                    target_w=width,
                    target_h=height,
                    lang=st.session_state.custom_title_lang,
                )
            st.image(
                prev_img,
                caption=f"Scene {c_idx + 1} • {width}x{height} • {st.session_state.editor_video_fx}",
                use_container_width=True,
            )
        elif monitor_mode == "Rendered Clip" and avail_vids:
            v_idx = min(int(st.session_state.selected_clip_idx), len(avail_vids) - 1)
            st.video(avail_vids[v_idx])
        else:
            fallback_img = build_local_bypass_scene(
                st.session_state.locked_character_prompt,
                width,
                height,
                0,
                emotion_label="Studio Preview",
            )
            fallback_img = apply_pil_effects(
                fallback_img,
                brightness=st.session_state.editor_brightness,
                contrast=st.session_state.editor_contrast,
                saturation=st.session_state.editor_saturation,
                blur_radius=st.session_state.editor_blur,
                video_fx=st.session_state.editor_video_fx,
            )
            if st.session_state.custom_title_text.strip():
                fallback_img = draw_subtitle_on_frame(
                    fallback_img,
                    subtitle_text=st.session_state.custom_title_text,
                    target_w=width,
                    target_h=height,
                    lang=st.session_state.custom_title_lang,
                )
            st.image(
                fallback_img,
                caption=f"Preview Canvas ({width}x{height}) — Generate scenes or upload clips to begin",
                use_container_width=True,
            )

    with stage_right:
        ctrl_tabs = st.tabs([
            "🎨 Looks & FX",
            "🔊 Nature & Reels SFX",
            "✍️ Text & Fit",
            "📁 Upload Media",
        ])

        with ctrl_tabs[0]:
            mood_sel = st.selectbox("One-Tap Color Mood", list(MOOD_FILTERS.keys()), key="mood_sel_box")
            if st.button("Apply Color Mood", key="apply_mood_btn"):
                m = MOOD_FILTERS[mood_sel]
                st.session_state.editor_brightness = m["b"]
                st.session_state.editor_contrast = m["c"]
                st.session_state.editor_saturation = m["s"]
                st.session_state.editor_video_fx = m["fx"]
                st.rerun()

            sl1, sl2 = st.columns(2)
            with sl1:
                st.session_state.editor_brightness = st.slider(
                    "Brightness", 0.5, 1.6, float(st.session_state.editor_brightness), 0.05, key="sl_b"
                )
                st.session_state.editor_contrast = st.slider(
                    "Contrast", 0.5, 1.8, float(st.session_state.editor_contrast), 0.05, key="sl_c"
                )
            with sl2:
                st.session_state.editor_saturation = st.slider(
                    "Saturation", 0.0, 2.0, float(st.session_state.editor_saturation), 0.05, key="sl_s"
                )
                st.session_state.editor_blur = st.slider(
                    "Soft Blur", 0.0, 6.0, float(st.session_state.editor_blur), 0.25, key="sl_blur"
                )

            fx1, fx2, fx3 = st.columns(3)
            with fx1:
                st.session_state.editor_video_fx = st.selectbox(
                    "Video Effect",
                    ["None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"],
                    index=["None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"].index(
                        st.session_state.editor_video_fx
                    ),
                    key="sel_vfx",
                )
            with fx2:
                st.session_state.editor_transition = st.selectbox(
                    "Transition",
                    ["None", "Crossfade", "Fade", "Slide Left", "Slide Right"],
                    index=["None", "Crossfade", "Fade", "Slide Left", "Slide Right"].index(
                        st.session_state.editor_transition
                    ),
                    key="sel_trans",
                )
            with fx3:
                st.session_state.editor_transition_sec = st.slider(
                    "Trans (sec)", 0.1, 1.5, float(st.session_state.editor_transition_sec), 0.1, key="sl_tsec"
                )

        with ctrl_tabs[1]:
            sfx_pick = st.selectbox(
                "Pick Sound Effect (Nature, Water, Fire, Funny Reels)",
                list(PRESET_AUDIO_EFFECTS.keys()),
                key="sfx_pick_box",
            )
            if PRESET_AUDIO_EFFECTS[sfx_pick]:
                st.audio(PRESET_AUDIO_EFFECTS[sfx_pick])

            sc_target = st.number_input(
                "Add Effect to Scene #",
                min_value=1,
                max_value=max(10, len(st.session_state.scenes)),
                value=1,
                step=1,
                key="sfx_sc_num",
            ) - 1
            if st.button("➕ Attach Sound Effect to Scene", key="attach_sfx_btn"):
                if sfx_pick == "None":
                    st.session_state.scene_audio_fx.pop(sc_target, None)
                    st.info(f"Removed sound effect from Scene {sc_target + 1}.")
                else:
                    p_sfx = download_fx_audio(sfx_pick, PRESET_AUDIO_EFFECTS[sfx_pick])
                    if p_sfx:
                        st.session_state.scene_audio_fx[sc_target] = p_sfx
                        st.success(f"Attached {sfx_pick} to Scene {sc_target + 1}.")
                    else:
                        st.error("Could not load that audio sample.")

        with ctrl_tabs[2]:
            st.session_state.bg_fit_mode = st.selectbox(
                "Canvas Framing Mode",
                ["Blur Canvas", "Crop Fill"],
                index=["Blur Canvas", "Crop Fill"].index(st.session_state.bg_fit_mode),
                key="bg_fit_box",
            )
            st.session_state.playback_speed = st.select_slider(
                "Video Speed",
                options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                value=float(st.session_state.playback_speed),
                key="speed_slider",
            )
            st.session_state.custom_title_text = st.text_input(
                "Custom Title / Hook Text on Preview",
                value=st.session_state.custom_title_text,
                placeholder="Example: Watch till the end! / ಅಚ್ಚರಿಯ ತಿರುವು!",
                key="title_overlay_input",
            )
            st.session_state.custom_title_lang = st.selectbox(
                "Text Language",
                ["English", "Kannada"],
                index=["English", "Kannada"].index(st.session_state.custom_title_lang),
                key="title_overlay_lang",
            )

        with ctrl_tabs[3]:
            up_v_col, up_a_col = st.columns(2)
            with up_v_col:
                v_files = st.file_uploader(
                    "Upload Video Clips",
                    type=["mp4", "mov", "webm", "mkv"],
                    accept_multiple_files=True,
                    key="up_vid_files",
                )
                if st.button("Add Videos", key="btn_add_vids") and v_files:
                    os.makedirs("uploaded_studio_media", exist_ok=True)
                    new_v = []
                    for i_u, uf in enumerate(v_files):
                        ext = os.path.splitext(uf.name)[1].lower() or ".mp4"
                        vp = os.path.join("uploaded_studio_media", f"vid_{int(time.time())}_{i_u}{ext}")
                        with open(vp, "wb") as f:
                            f.write(uf.getbuffer())
                        new_v.append(vp)
                    st.session_state.uploaded_video_clips.extend(new_v)
                    st.session_state.scene_videos.extend(new_v)
                    st.success(f"Added {len(new_v)} video clip(s).")
                    st.rerun()

            with up_a_col:
                a_files = st.file_uploader(
                    "Upload Voice / Music / BGM",
                    type=["mp3", "wav", "m4a", "ogg"],
                    accept_multiple_files=True,
                    key="up_aud_files",
                )
                if st.button("Add as BGM Track", key="btn_add_bgm") and a_files:
                    os.makedirs("uploaded_studio_media", exist_ok=True)
                    af = a_files[0]
                    ext = os.path.splitext(af.name)[1].lower() or ".mp3"
                    bp = os.path.join("uploaded_studio_media", f"bgm_{int(time.time())}{ext}")
                    with open(bp, "wb") as f:
                        f.write(af.getbuffer())
                    st.session_state.bgm_audio_path = bp
                    st.success("Background music (BGM) track loaded.")
                    st.rerun()

    # Quick clip selector bar
    clip_pool = st.session_state.scene_videos if st.session_state.scene_videos else st.session_state.images
    if clip_pool:
        st.markdown("**🎞️ Select Clip to Preview**")
        c_btns = st.columns(min(8, len(clip_pool)))
        for idx_c in range(min(8, len(clip_pool))):
            with c_btns[idx_c]:
                active_mark = "▶ " if idx_c == st.session_state.selected_clip_idx else ""
                if st.button(f"{active_mark}Scene {idx_c + 1}", key=f"pick_sc_{idx_c}", use_container_width=True):
                    st.session_state.selected_clip_idx = idx_c
                    st.rerun()

# --- MAIN WORKFLOW TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Story & AI Director",
    "2️⃣ Scene Videos",
    "3️⃣ Voice, Music & SFX",
    "4️⃣ Final Movie / Reel Export",
])

# --- TAB 1: STORY & AI DIRECTOR ---
with tab1:
    st.markdown("### Write Your Command, Story, Nature Setting, or Reel Concept")
    rough_story = st.text_area(
        "Story / Reel / Movie Command:",
        value="Follow the character through any setting, nature, or situation described in my command with clear emotion and action.",
        height=115,
    )
    num_scenes = st.slider("Number of Scenes", 2, 10, max(2, min(10, len(st.session_state.scenes) or 6)))

    if st.button("✨ Generate Smart Scenes (AI / Heuristic)", type="primary"):
        if not rough_story.strip():
            st.error("Please enter a story or command first.")
        elif not api_key or not api_key.strip():
            st.session_state.scenes = build_fallback_scenes_from_rough_story(
                rough_story, num_scenes, st.session_state.locked_character_prompt
            )
            for idx, sc in enumerate(st.session_state.scenes):
                st.session_state[f"nar_{idx}"] = sc["narration"]
                st.session_state[f"kn_script_{idx}"] = sc["kannada_narration"]
                st.session_state[f"vis_{idx}"] = sc["visual"]
                st.session_state[f"emo_{idx}"] = sc["emotion"]
                st.session_state[f"mot_{idx}"] = sc.get("motion", "Auto Alternate")
                if sc.get("suggested_sfx") and sc["suggested_sfx"] != "None":
                    p_auto_sfx = download_fx_audio(sc["suggested_sfx"], PRESET_AUDIO_EFFECTS.get(sc["suggested_sfx"], ""))
                    if p_auto_sfx:
                        st.session_state.scene_audio_fx[idx] = p_auto_sfx
            st.success(f"Created {len(st.session_state.scenes)} smart scenes with suggested camera motion & SFX.")
        else:
            fallback_models = [
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
            ]
            char_context = (
                f"\nCharacter/environment instruction ({st.session_state.exact_character_count} main character(s)): {st.session_state.locked_character_prompt}"
                if st.session_state.lock_character_enabled
                else f"\nCharacter/environment notes: {st.session_state.locked_character_prompt}"
            )
            prompt = f"""
            Create a {num_scenes}-scene story script obeying every user command for characters, nature, backgrounds, locations, and actions.{char_context}
            Return ONLY valid JSON as a list of {num_scenes} objects, no markdown.
            Each object must have:
            - "narration": clear English voiceover (1-2 sentences)
            - "kannada_narration": natural Kannada voiceover/caption (1-2 sentences)
            - "emotion": mood or expression
            - "motion": one of ["Zoom In", "Zoom Out", "Pan Right", "Pan Left", "Auto Alternate"]
            - "visual": scene visual description honoring user commands for character, setting/nature/background, and action
            User command: {rough_story.strip()}
            """
            client = genai.Client(api_key=api_key.strip())
            parsed_scenes = None
            last_exc = None

            with st.spinner(f"Generating {num_scenes} AI-directed scenes..."):
                for model_name in fallback_models:
                    for attempt in range(2):
                        try:
                            resp = client.models.generate_content(model=model_name, contents=prompt)
                            raw_text = (resp.text or "").strip()
                            if raw_text.startswith("```"):
                                raw_text = "\n".join(
                                    ln for ln in raw_text.splitlines() if not ln.strip().startswith("```")
                                ).strip()
                            cand = json.loads(raw_text)
                            if isinstance(cand, list) and len(cand) > 0:
                                parsed_scenes = cand
                                break
                        except Exception as exc:
                            last_exc = exc
                            if "503" in str(exc) or "429" in str(exc):
                                time.sleep(1.2 * (attempt + 1))
                                continue
                            break
                    if parsed_scenes is not None:
                        break

            if parsed_scenes is not None:
                normalized = []
                for idx_p, item in enumerate(parsed_scenes[:num_scenes]):
                    m_sug, fx_sug, sfx_sug = infer_ml_scene_direction(
                        f"{item.get('narration', '')} {item.get('visual', '')}", idx_p
                    )
                    chosen_m = str(item.get("motion", m_sug)).strip()
                    if chosen_m not in ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]:
                        chosen_m = m_sug
                    normalized.append({
                        "narration": str(item.get("narration", "")).strip(),
                        "kannada_narration": str(item.get("kannada_narration", "")).strip(),
                        "emotion": str(item.get("emotion", "Expressive")).strip(),
                        "motion": chosen_m,
                        "suggested_fx": fx_sug,
                        "suggested_sfx": sfx_sug,
                        "visual": str(item.get("visual", "")).strip(),
                    })
                st.session_state.scenes = sync_scenes_to_count(
                    normalized, num_scenes, rough_story, st.session_state.locked_character_prompt
                )
                for idx, sc in enumerate(st.session_state.scenes):
                    st.session_state[f"nar_{idx}"] = sc["narration"]
                    st.session_state[f"kn_script_{idx}"] = sc["kannada_narration"]
                    st.session_state[f"vis_{idx}"] = sc["visual"]
                    st.session_state[f"emo_{idx}"] = sc["emotion"]
                    st.session_state[f"mot_{idx}"] = sc.get("motion", "Auto Alternate")
                st.success(f"Generated {len(st.session_state.scenes)} AI-directed scenes!")
            else:
                st.session_state.scenes = build_fallback_scenes_from_rough_story(
                    rough_story, num_scenes, st.session_state.locked_character_prompt
                )
                for idx, sc in enumerate(st.session_state.scenes):
                    st.session_state[f"nar_{idx}"] = sc["narration"]
                    st.session_state[f"kn_script_{idx}"] = sc["kannada_narration"]
                    st.session_state[f"vis_{idx}"] = sc["visual"]
                    st.session_state[f"emo_{idx}"] = sc["emotion"]
                    st.session_state[f"mot_{idx}"] = sc.get("motion", "Auto Alternate")
                st.warning(f"Used smart local scene builder ({last_exc}).")

    if st.session_state.scenes:
        st.markdown(f"### Scene Script & Direction Cards ({len(st.session_state.scenes)} Scenes)")
        if st.button("🌐 Auto-Translate All Scenes to Kannada"):
            if not api_key or not api_key.strip():
                st.error("Add your Gemini API Key in the sidebar to translate with AI.")
            else:
                en_list = [
                    (st.session_state.get(f"nar_{i}", sc.get("narration", "")) or f"Scene {i + 1}.").strip()
                    for i, sc in enumerate(st.session_state.scenes)
                ]
                with st.spinner("Translating scenes to Kannada..."):
                    kn_list = translate_lines_to_kannada(api_key, en_list)
                for i, kn_val in enumerate(kn_list):
                    st.session_state.scenes[i]["kannada_narration"] = kn_val
                    st.session_state[f"kn_script_{i}"] = kn_val
                st.success("Kannada script updated for all scenes.")
                st.rerun()

        for i, scene in enumerate(st.session_state.scenes):
            with st.expander(f"Scene {i + 1} • {scene.get('emotion', 'Expressive')}", expanded=(i < 2)):
                c_left, c_right = st.columns(2)
                with c_left:
                    nar_v = st.text_area(f"English Narration (Scene {i + 1})", value=scene.get("narration", ""), key=f"nar_{i}")
                    emo_v = st.text_input(f"Mood / Emotion (Scene {i + 1})", value=scene.get("emotion", "Expressive"), key=f"emo_{i}")
                    mot_v = st.selectbox(
                        f"Camera Motion (Scene {i + 1})",
                        ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"],
                        index=["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"].index(
                            st.session_state.get(f"mot_{i}", scene.get("motion", "Auto Alternate"))
                            if st.session_state.get(f"mot_{i}", scene.get("motion", "Auto Alternate")) in ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
                            else "Auto Alternate"
                        ),
                        key=f"mot_{i}",
                    )
                    vis_v = st.text_input(f"Visual / Background / Action Command (Scene {i + 1})", value=scene.get("visual", ""), key=f"vis_{i}")
                    scene["narration"] = nar_v
                    scene["emotion"] = emo_v
                    scene["motion"] = mot_v
                    scene["visual"] = vis_v
                with c_right:
                    kn_v = st.text_area(
                        f"Kannada Script / Subtitle (Scene {i + 1})",
                        value=scene.get("kannada_narration", ""),
                        key=f"kn_script_{i}",
                    )
                    scene["kannada_narration"] = kn_v

# --- TAB 2: SCENE VIDEOS ---
with tab2:
    st.markdown("### Generate Scene Videos")
    art_style = st.selectbox(
        "Visual Style",
        [
            "Photorealistic cinematic film still, natural lighting matching the requested environment, lifelike human expressions, 35mm lens",
            "High-detail 3D CGI animation, expressive lifelike characters, rich environment lighting and depth",
            "Detailed storybook illustration, expressive characters, rich scenery and atmosphere",
        ],
    )

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        scene_count_slider = st.slider("Scene Count", 2, 10, max(2, min(10, len(st.session_state.scenes) or 6)))
    with opt2:
        per_scene_sec = st.slider("Base Clip Duration (sec)", 2.0, 10.0, 4.0, 0.5)
    with opt3:
        tab2_sub_lang = st.selectbox("Subtitles in Scene Clips", ["None", "English", "Kannada"], index=2)

    ref_file = st.file_uploader("Optional Reference Image (Character or Scene)", type=["png", "jpg", "jpeg", "webp"])
    allow_local_bypass = st.checkbox("Use local storyboard frames automatically if external image APIs fail/hit quota", value=True)

    ref_pil = None
    if ref_file is not None:
        ref_pil = Image.open(ref_file).convert("RGB")
        st.image(ref_pil, caption="Reference Image", width=220)

    if st.button("🎬 Generate Scene Videos", type="primary"):
        active_scenes = sync_scenes_to_count(
            st.session_state.scenes,
            scene_count_slider,
            rough_story,
            st.session_state.locked_character_prompt,
        )
        for idx, sc in enumerate(active_scenes):
            sc["narration"] = st.session_state.get(f"nar_{idx}", sc.get("narration", ""))
            sc["kannada_narration"] = st.session_state.get(f"kn_script_{idx}", sc.get("kannada_narration", ""))
            sc["visual"] = st.session_state.get(f"vis_{idx}", sc.get("visual", ""))
            sc["emotion"] = st.session_state.get(f"emo_{idx}", sc.get("emotion", "Expressive"))
            sc["motion"] = st.session_state.get(f"mot_{idx}", sc.get("motion", "Auto Alternate"))

        st.session_state.scenes = active_scenes
        scene_subtitles = (
            get_script_lines_for_language(active_scenes, script_lang=tab2_sub_lang, gemini_key=api_key)
            if tab2_sub_lang != "None"
            else [""] * len(active_scenes)
        )

        st.session_state.gen_batch_id += 1
        batch_id = st.session_state.gen_batch_id
        st.session_state.images = []
        st.session_state.scene_videos = []
        os.makedirs("scene_videos", exist_ok=True)

        cast_guard = (
            f"Follow user command strictly: {st.session_state.locked_character_prompt} "
            f"({int(st.session_state.exact_character_count)} main subject(s) when applicable)."
            if st.session_state.lock_character_enabled
            else st.session_state.locked_character_prompt
        )

        prog = st.progress(0)
        for idx, scene in enumerate(active_scenes):
            full_prompt = (
                f"{art_style}. {cast_guard} "
                f"Scene {idx + 1} mood: {scene.get('emotion', 'Expressive')}. "
                f"Scene {idx + 1} command: {scene.get('visual', '')}. "
                f"Story: {rough_story.strip()}"
            ).strip()

            pil_img, _, _ = generate_real_ai_scene(
                engine_choice=image_engine,
                cf_account_id=cf_account_id,
                cf_api_token=cf_api_token,
                hf_token=hf_token,
                gemini_key=api_key,
                prompt_text=full_prompt,
                ratio_str=ratio_str,
                target_w=width,
                target_h=height,
                scene_idx=idx,
                ref_pil=ref_pil,
                allow_local_bypass=allow_local_bypass,
                emotion_label=scene.get("emotion", "Expressive"),
                seed_offset=batch_id * 100,
            )
            if pil_img is not None:
                img_path = f"scene_{idx}_b{batch_id}.png"
                ensure_exact_size(pil_img, width, height, bg_mode=st.session_state.bg_fit_mode).save(img_path, format="PNG")
                st.session_state.images.append(img_path)

                vid_path = os.path.join("scene_videos", f"scene_{idx}_b{batch_id}.mp4")
                render_single_scene_video(
                    img_path=img_path,
                    out_path=vid_path,
                    target_w=width,
                    target_h=height,
                    duration=per_scene_sec,
                    motion_mode=scene.get("motion", "Auto Alternate"),
                    scene_idx=idx,
                    subtitle_text=scene_subtitles[idx] if idx < len(scene_subtitles) else "",
                    subtitle_lang=tab2_sub_lang,
                    enable_char_motion=True,
                    brightness=st.session_state.editor_brightness,
                    contrast=st.session_state.editor_contrast,
                    saturation=st.session_state.editor_saturation,
                    blur_radius=st.session_state.editor_blur,
                    video_fx=st.session_state.editor_video_fx,
                    bg_mode=st.session_state.bg_fit_mode,
                )
                st.session_state.scene_videos.append(vid_path)
            prog.progress((idx + 1) / len(active_scenes))

        st.success(f"Generated {len(st.session_state.scene_videos)} scene videos at {width}x{height}.")

    if st.session_state.scene_videos:
        grid_cols = st.columns(2)
        for idx, vp in enumerate(st.session_state.scene_videos):
            if os.path.exists(vp):
                with grid_cols[idx % 2]:
                    st.markdown(f"**Scene {idx + 1} Video**")
                    st.video(vp)
                    with open(vp, "rb") as vf:
                        st.download_button(
                            label=f"⬇️ Download Scene {idx + 1}",
                            data=vf.read(),
                            file_name=f"scene_{idx + 1}.mp4",
                            mime="video/mp4",
                            key=f"dl_v_{idx}_b{st.session_state.gen_batch_id}",
                        )

# --- TAB 3: VOICE, MUSIC & SFX ---
with tab3:
    st.markdown("### Voiceover, Background Music & Sound Mixing")
    voice_map = {
        "Warm Male Storyteller (US)": ("en-US-ChristopherNeural", "English"),
        "Expressive Female Storyteller (US)": ("en-US-AriaNeural", "English"),
        "Indian Male Narrator (English)": ("en-IN-PrabhatNeural", "English"),
        "Indian Female Narrator (English)": ("en-IN-NeerjaNeural", "English"),
        "Kannada Male Narrator (ಕನ್ನಡ - Gagan)": ("kn-IN-GaganNeural", "Kannada"),
        "Kannada Female Narrator (ಕನ್ನಡ - Sapna)": ("kn-IN-SapnaNeural", "Kannada"),
    }
    vc1, vc2 = st.columns(2)
    with vc1:
        v_choice = st.selectbox("AI Narrator Voice", list(voice_map.keys()))
        v_id, v_lang = voice_map[v_choice]
    with vc2:
        s_lang = st.selectbox("Script Language", ["English", "Kannada"], index=1 if v_lang == "Kannada" else 0)

    lines_to_speak = get_script_lines_for_language(st.session_state.scenes, script_lang=s_lang, gemini_key=api_key)

    async def synth_tts(txt, v_name, out_file):
        comm = edge_tts.Communicate(txt, v_name)
        await comm.save(out_file)

    if st.button("🔊 Generate Voiceover for All Scenes", type="primary"):
        try:
            st.session_state.audio_files = []
            prog_a = st.progress(0)
            for idx, line in enumerate(lines_to_speak):
                ap = f"voice_{idx}.mp3"
                run_async(synth_tts(line, v_id, ap))
                st.session_state.audio_files.append(ap)
                prog_a.progress((idx + 1) / max(1, len(lines_to_speak)))
            st.success(f"Generated {len(st.session_state.audio_files)} voiceover clip(s).")
        except Exception as exc:
            st.error(f"Voiceover generation error: {exc}")

    st.divider()
    st.markdown("#### Mix Levels (Voice, Background Music & Sound FX)")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.session_state.voice_volume = st.slider("Voice Volume", 0.0, 1.5, float(st.session_state.voice_volume), 0.05)
    with m2:
        st.session_state.bgm_volume = st.slider("BGM Volume", 0.0, 0.8, float(st.session_state.bgm_volume), 0.02)
    with m3:
        st.session_state.sfx_volume = st.slider("Sound FX Volume", 0.0, 1.5, float(st.session_state.sfx_volume), 0.05)

    if st.session_state.bgm_audio_path and os.path.exists(st.session_state.bgm_audio_path):
        st.caption(f"Active BGM: `{os.path.basename(st.session_state.bgm_audio_path)}`")
        st.audio(st.session_state.bgm_audio_path)

# --- TAB 4: FINAL MOVIE / REEL EXPORT ---
with tab4:
    st.markdown("### Export Final Video (Movie, Shorts or Reels)")
    exp1, exp2 = st.columns(2)
    with exp1:
        final_sub_lang = st.selectbox("Final Subtitles", ["None", "English", "Kannada"], index=2, key="final_sub_box")
    with exp2:
        final_motion_mode = st.selectbox(
            "Fallback Motion Mode",
            ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"],
            key="final_mot_box",
        )

    st.caption(
        f"Format: `{st.session_state.format_preset}` (`{width}x{height}`) • "
        f"Speed: `{st.session_state.playback_speed}x` • "
        f"Transition: `{st.session_state.editor_transition}` • "
        f"Effect: `{st.session_state.editor_video_fx}`"
    )

    if st.button("🎬 Render Final MP4 Video", type="primary"):
        has_media = bool(st.session_state.scene_videos or st.session_state.images)
        if not has_media:
            st.error("Generate scene videos/images in Tab 2 or upload clips first.")
        else:
            clips = []
            src_vids = []
            aud_list = []
            bgm_clip = None
            final_vid = None
            out_mp4 = f"final_studio_b{st.session_state.gen_batch_id}.mp4"

            try:
                total_slots = max(len(st.session_state.scene_videos), len(st.session_state.images))
                spd = max(0.5, float(st.session_state.playback_speed))

                if final_sub_lang == "None":
                    sub_lines = [""] * total_slots
                else:
                    sub_lines = get_script_lines_for_language(
                        st.session_state.scenes, script_lang=final_sub_lang, gemini_key=api_key
                    )
                while len(sub_lines) < total_slots:
                    sub_lines.append("")

                with st.spinner("Rendering your final video..."):
                    for i in range(total_slots):
                        v_aud = st.session_state.audio_files[i] if i < len(st.session_state.audio_files) else None
                        s_sfx = st.session_state.scene_audio_fx.get(i)

                        mixed_parts = []
                        dur_candidates = [4.0]
                        if v_aud and os.path.exists(v_aud):
                            va = AudioFileClip(v_aud)
                            aud_list.append(va)
                            dur_candidates.append(float(va.duration or 3.0) + 0.25)
                            mixed_parts.append(scale_audio_volume(va, st.session_state.voice_volume))

                        if s_sfx and os.path.exists(s_sfx):
                            sa = AudioFileClip(s_sfx)
                            aud_list.append(sa)
                            mixed_parts.append(scale_audio_volume(sa, st.session_state.sfx_volume))

                        clip_dur = max(max(dur_candidates) / spd, 0.8)

                        caption_txt = sub_lines[i]
                        if st.session_state.custom_title_text.strip():
                            caption_txt = (
                                f"{caption_txt} • {st.session_state.custom_title_text.strip()}"
                                if caption_txt
                                else st.session_state.custom_title_text.strip()
                            )

                        if i < len(st.session_state.scene_videos) and os.path.exists(st.session_state.scene_videos[i]):
                            sv = VideoFileClip(st.session_state.scene_videos[i])
                            src_vids.append(sv)
                            if (sv.duration or 0) < clip_dur and hasattr(sv, "loop"):
                                c_item = apply_duration(sv.loop(duration=clip_dur), clip_dur)
                            else:
                                c_item = apply_duration(sv, clip_dur)
                        else:
                            sc_motion = (
                                st.session_state.scenes[i].get("motion", final_motion_mode)
                                if i < len(st.session_state.scenes)
                                else final_motion_mode
                            )
                            c_item = build_motion_clip(
                                img_path=st.session_state.images[i],
                                target_w=width,
                                target_h=height,
                                duration=clip_dur,
                                motion_mode=sc_motion,
                                scene_idx=i,
                                subtitle_text=caption_txt,
                                subtitle_lang=final_sub_lang,
                                enable_char_motion=True,
                                brightness=st.session_state.editor_brightness,
                                contrast=st.session_state.editor_contrast,
                                saturation=st.session_state.editor_saturation,
                                blur_radius=st.session_state.editor_blur,
                                video_fx=st.session_state.editor_video_fx,
                                bg_mode=st.session_state.bg_fit_mode,
                            )

                        if mixed_parts:
                            trimmed = [apply_duration(mp, min(clip_dur, float(mp.duration or clip_dur))) for mp in mixed_parts]
                            c_item = apply_audio(c_item, CompositeAudioClip(trimmed))

                        clips.append(c_item)

                    final_vid = apply_clip_transition(
                        clips,
                        transition_name=st.session_state.editor_transition,
                        trans_sec=st.session_state.editor_transition_sec,
                    )

                    if st.session_state.bgm_audio_path and os.path.exists(st.session_state.bgm_audio_path) and st.session_state.bgm_volume > 0:
                        bgm_raw = AudioFileClip(st.session_state.bgm_audio_path)
                        bgm_clip = bgm_raw
                        total_d = max(float(final_vid.duration or 1.0), 0.5)
                        if (bgm_raw.duration or 0) < total_d and hasattr(bgm_raw, "loop"):
                            bgm_scaled = apply_duration(bgm_raw.loop(duration=total_d), total_d)
                        else:
                            bgm_scaled = apply_duration(bgm_raw, total_d)
                        bgm_scaled = scale_audio_volume(bgm_scaled, st.session_state.bgm_volume)
                        if final_vid.audio is not None:
                            final_vid = apply_audio(final_vid, CompositeAudioClip([final_vid.audio, bgm_scaled]))
                        else:
                            final_vid = apply_audio(final_vid, bgm_scaled)

                    final_vid.write_videofile(
                        out_mp4,
                        fps=24,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None,
                    )

                st.video(out_mp4)
                with open(out_mp4, "rb") as f_out:
                    st.download_button(
                        label="⬇️ Download Final MP4",
                        data=f_out.read(),
                        file_name="ai_studio_final_video.mp4",
                        mime="video/mp4",
                    )
            except Exception as exc:
                st.error(f"Export failed: {exc}")
            finally:
                for c in clips:
                    try:
                        c.close()
                    except Exception:
                        pass
                for sv in src_vids:
                    try:
                        sv.close()
                    except Exception:
                        pass
                for a in aud_list:
                    try:
                        a.close()
                    except Exception:
                        pass
                if bgm_clip is not None:
                    try:
                        bgm_clip.close()
                    except Exception:
                        pass
                if final_vid is not None:
                    try:
                        final_vid.close()
                    except Exception:
                        pass
