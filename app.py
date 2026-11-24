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
        concatenate_audioclips,
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
        concatenate_audioclips,
        concatenate_videoclips,
    )
    MOVIEPY_V2 = True

st.set_page_config(
    page_title="InShot-Style AI Video Studio",
    page_icon="🎬",
    layout="wide",
)

# Dark InShot-inspired studio styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f3f4f6;
    }
    .inshot-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #171c26;
        border: 1px solid #283142;
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 12px;
    }
    .inshot-badge {
        background: #ff2e63;
        color: #ffffff;
        font-weight: 700;
        font-size: 12px;
        padding: 3px 9px;
        border-radius: 999px;
        letter-spacing: 0.4px;
    }
    .timeline-track {
        background: #141923;
        border: 1px solid #263042;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .clip-pill {
        background: #232c3d;
        border-left: 4px solid #ff2e63;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 12px;
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
        "visual": (
            "Medium cinematic shot matching the exact environment and action requested; "
            "main character reacting naturally with clear expression and lighting"
        ),
    },
    {
        "narration": "A sudden turn shifts the mood and tests how the character responds in the moment.",
        "kannada_narration": "ಹಠಾತ್ ತಿರುವು ವಾತಾವರಣವನ್ನು ಬದಲಾಯಿಸುತ್ತದೆ ಮತ್ತು ಆ ಕ್ಷಣದ ಪ್ರತಿಕ್ರಿಯೆಯನ್ನು ಪರೀಕ್ಷಿಸುತ್ತದೆ.",
        "emotion": "Alert & Determined",
        "visual": (
            "Dynamic medium-wide shot in the requested environment; main character showing "
            "expressive posture, movement, and atmospheric lighting"
        ),
    },
    {
        "narration": "Through changing surroundings and emotion, the story builds toward a clear turning point.",
        "kannada_narration": "ಬದಲಾಗುವ ಪರಿಸರ ಮತ್ತು ಭಾವನೆಗಳ ಮೂಲಕ ಕಥೆಯು ಸ್ಪಷ್ಟ ತಿರುವಿನತ್ತ ಸಾಗುತ್ತದೆ.",
        "emotion": "Reflective & Resilient",
        "visual": (
            "Expressive framing showing the main character and surrounding environment "
            "with rich depth, natural motion, and mood lighting"
        ),
    },
    {
        "narration": "With steady resolve, the scene settles into a memorable closing moment.",
        "kannada_narration": "ಸ್ಥಿರ ನಿರ್ಧಾರದೊಂದಿಗೆ ದೃಶ್ಯವು ಒಂದು ಸ್ಮರಣೀಯ ಅಂತ್ಯದತ್ತ ಸಾಗುತ್ತದೆ.",
        "emotion": "Composed & Hopeful",
        "visual": (
            "Balanced cinematic shot of the main character in the commanded setting "
            "with clear composition and clean lighting"
        ),
    },
]

PRESET_AUDIO_EFFECTS = {
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
    "Vivid Pop": {"b": 1.05, "c": 1.18, "s": 1.42, "fx": "None"},
    "Warm Golden": {"b": 1.06, "c": 1.08, "s": 1.22, "fx": "None"},
    "Cool Cinema": {"b": 0.98, "c": 1.15, "s": 0.85, "fx": "Cinematic Vignette"},
    "Retro Sepia": {"b": 1.02, "c": 1.10, "s": 0.70, "fx": "Vintage Sepia"},
    "Reels Punch": {"b": 1.08, "c": 1.22, "s": 1.35, "fx": "Zoom Punch"},
}

for key, default in [
    ("scenes", [dict(s) for s in DEFAULT_SCENES]),
    ("images", []),
    ("scene_videos", []),
    ("uploaded_video_clips", []),
    ("uploaded_audio_clips", []),
    ("tab2_baked_subtitles", False),
    ("audio_files", []),
    ("own_voice_table", {}),
    ("locked_character_prompt", DEFAULT_CHAR_PROMPT),
    ("exact_character_count", 1),
    ("lock_character_enabled", False),
    ("kannada_subtitles", {}),
    ("populated_script_lines", []),
    ("populated_script_lang", "English"),
    ("gen_batch_id", 0),
    ("inshot_active_tool", "Filter & Adjust"),
    ("inshot_selected_clip", 0),
    ("inshot_canvas_ratio", "9:16 (Reels / Shorts)"),
    ("inshot_bg_mode", "Blur Canvas"),
    ("inshot_speed", 1.0),
    ("inshot_custom_text", ""),
    ("inshot_custom_text_lang", "English"),
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


def get_canvas_dimensions(ratio_label):
    mapping = {
        "9:16 (Reels / Shorts)": (720, 1280, "9:16"),
        "16:9 (YouTube)": (1280, 720, "16:9"),
        "1:1 (Square)": (1080, 1080, "1:1"),
        "4:5 (Portrait Feed)": (1080, 1350, "4:5"),
        "3:4 (Classic Portrait)": (900, 1200, "3:4"),
    }
    return mapping.get(ratio_label, (720, 1280, "9:16"))


def run_async(coro):
    """Safely run async coroutines inside Streamlit's event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with tempfile.NamedTemporaryFile(suffix=".py") as _:
                pass
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


def ensure_exact_size(pil_img, target_w, target_h, bg_mode="Crop Fill"):
    """Place any image onto an exact (target_w, target_h) canvas using Crop Fill or Blur Canvas."""
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
        bg = ImageEnhance.Brightness(bg).enhance(0.65)

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
        out = ImageEnhance.Brightness(out).enhance(1.45)
    elif video_fx == "Glitch RGB":
        r, g, b = out.split()
        r = ImageEnhance.Brightness(r).enhance(1.2)
        b = ImageEnhance.Contrast(b).enhance(1.3)
        out = Image.merge("RGB", (r, g, b))

    return out


def download_fx_audio(fx_name, fx_url):
    """Download and cache preset audio effects locally."""
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
        scenes.append({
            "narration": f"{base_line}.",
            "kannada_narration": "",
            "emotion": emo_label,
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
    """Draw a clean subtitle banner near the bottom of a scene frame with Kannada font support."""
    if not subtitle_text or not subtitle_text.strip() or lang == "None":
        return frame_pil

    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    font_size = max(24, int(round(target_w * 0.036)))
    font, _ = _find_font(font_size, lang=lang)
    lang_code = "kn" if lang == "Kannada" else "en"

    chars_per_line = max(14, int(target_w / max(11, font_size * 0.64)))
    wrapped_lines = textwrap.wrap(subtitle_text.strip(), width=chars_per_line)[:3]
    if not wrapped_lines:
        return frame_pil

    line_h = font_size + 12
    box_h = len(wrapped_lines) * line_h + 24
    pad_x = max(20, int(target_w * 0.04))
    box_top = max(16, target_h - box_h - max(42, int(target_h * 0.07)))
    box_bot = min(target_h - 16, box_top + box_h)

    draw.rounded_rectangle(
        [pad_x, box_top, target_w - pad_x, box_bot],
        radius=14,
        fill=(10, 14, 22, 210),
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
            (text_x + 2, cur_y + 2, (0, 0, 0, 235)),
            (text_x, cur_y, (255, 248, 214, 255)),
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
    bg_mode="Crop Fill",
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
    bg_mode="Crop Fill",
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

    # Soft horizon ridge instead of blocky rectangles
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

    # Clean character silhouette & expression
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


def generate_cloudflare_flux_image(cf_account_id, cf_api_token, prompt_text, target_w, target_h, seed=42):
    """Call Cloudflare Workers AI FLUX.1 [schnell] REST API using strictly valid schema fields."""
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
    payload = {
        "prompt": bounded_prompt,
    }
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
                cf_account_id, cf_api_token, prompt_text, target_w, target_h, seed=scene_seed
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
        if transition_name == "Fade":
            if hasattr(c_mod, "crossfadein") and idx > 0:
                c_mod = c_mod.crossfadein(t_dur)
            elif hasattr(c_mod, "fadein") and idx > 0:
                c_mod = c_mod.fadein(t_dur)
        elif transition_name == "Crossfade":
            if hasattr(c_mod, "crossfadein") and idx > 0:
                c_mod = c_mod.crossfadein(t_dur)
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


# --- TOP INSHOT HEADER BAR ---
st.markdown(
    """
    <div class="inshot-topbar">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="inshot-badge">INSHOT STUDIO</span>
            <span style="font-weight:700;font-size:16px;">AI Video, Reels & Story Editor</span>
        </div>
        <span style="color:#9ca3af;font-size:13px;">Canvas • Music & SFX • Filters • VFX • Subtitles • Timeline</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key (for Script Polishing & Kannada)", type="password")
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

    st.session_state.inshot_canvas_ratio = st.selectbox(
        "Canvas Aspect Ratio",
        [
            "9:16 (Reels / Shorts)",
            "16:9 (YouTube)",
            "1:1 (Square)",
            "4:5 (Portrait Feed)",
            "3:4 (Classic Portrait)",
        ],
        index=[
            "9:16 (Reels / Shorts)",
            "16:9 (YouTube)",
            "1:1 (Square)",
            "4:5 (Portrait Feed)",
            "3:4 (Classic Portrait)",
        ].index(st.session_state.inshot_canvas_ratio),
    )

    st.divider()
    st.subheader("🎛️ Character & Command Lock")
    st.session_state.lock_character_enabled = st.checkbox(
        "Enforce character & setting instructions across scenes",
        value=st.session_state.lock_character_enabled,
    )
    st.session_state.exact_character_count = st.number_input(
        "Character / Subject Count",
        min_value=1,
        max_value=8,
        value=int(st.session_state.exact_character_count),
        step=1,
    )
    st.session_state.locked_character_prompt = st.text_area(
        "Character / Environment Command Prompt",
        value=st.session_state.locked_character_prompt,
        height=110,
    )

width, height, ratio_str = get_canvas_dimensions(st.session_state.inshot_canvas_ratio)

# --- INSHOT WORKSPACE: CANVAS PREVIEW + TOOL DOCK + MULTI-TRACK TIMELINE ---
with st.expander("🎬 InShot Editing Deck — Live Canvas, Tool Dock & Multi-Track Timeline", expanded=True):
    stage_col, dock_col = st.columns([1.25, 1.75])

    with stage_col:
        available_source_imgs = [p for p in st.session_state.images if os.path.exists(p)]
        available_source_vids = [p for p in st.session_state.scene_videos if os.path.exists(p)] + [
            p for p in st.session_state.uploaded_video_clips if os.path.exists(p)
        ]

        stage_view = st.radio(
            "Stage Monitor",
            ["Live Canvas Preview", "Clip Video Player"],
            horizontal=True,
            key="inshot_stage_monitor_mode",
        )

        if stage_view == "Live Canvas Preview" and available_source_imgs:
            max_i = len(available_source_imgs) - 1
            sel_i = min(int(st.session_state.inshot_selected_clip), max_i)
            with Image.open(available_source_imgs[sel_i]) as raw_im:
                canvas_pil = ensure_exact_size(
                    raw_im, width, height, bg_mode=st.session_state.inshot_bg_mode
                )
            canvas_pil = apply_pil_effects(
                canvas_pil,
                brightness=st.session_state.editor_brightness,
                contrast=st.session_state.editor_contrast,
                saturation=st.session_state.editor_saturation,
                blur_radius=st.session_state.editor_blur,
                video_fx=st.session_state.editor_video_fx,
            )
            if st.session_state.inshot_custom_text.strip():
                canvas_pil = draw_subtitle_on_frame(
                    canvas_pil,
                    subtitle_text=st.session_state.inshot_custom_text,
                    target_w=width,
                    target_h=height,
                    lang=st.session_state.inshot_custom_text_lang,
                )
            st.image(
                canvas_pil,
                caption=f"Clip {sel_i + 1} • {width}x{height} • FX: {st.session_state.editor_video_fx} • {st.session_state.inshot_speed}x",
                use_container_width=True,
            )
        elif stage_view == "Clip Video Player" and available_source_vids:
            max_v = len(available_source_vids) - 1
            sel_v = min(int(st.session_state.inshot_selected_clip), max_v)
            st.video(available_source_vids[sel_v])
        else:
            bypass_img = build_local_bypass_scene(
                st.session_state.locked_character_prompt,
                width,
                height,
                0,
                emotion_label="InShot Canvas",
            )
            bypass_img = apply_pil_effects(
                bypass_img,
                brightness=st.session_state.editor_brightness,
                contrast=st.session_state.editor_contrast,
                saturation=st.session_state.editor_saturation,
                blur_radius=st.session_state.editor_blur,
                video_fx=st.session_state.editor_video_fx,
            )
            if st.session_state.inshot_custom_text.strip():
                bypass_img = draw_subtitle_on_frame(
                    bypass_img,
                    subtitle_text=st.session_state.inshot_custom_text,
                    target_w=width,
                    target_h=height,
                    lang=st.session_state.inshot_custom_text_lang,
                )
            st.image(
                bypass_img,
                caption=f"Canvas {width}x{height} ({ratio_str}) — Ready for editing",
                use_container_width=True,
            )

    with dock_col:
        st.markdown("**🛠️ InShot Tool Dock**")
        tool_tabs = st.tabs([
            "📐 Canvas",
            "🎵 Music & SFX",
            "🎨 Filter & Adjust",
            "⚡ Effect & Trans",
            "🔤 Text Overlay",
            "⏩ Speed & Media",
        ])

        with tool_tabs[0]:
            st.session_state.inshot_bg_mode = st.selectbox(
                "Frame Fit Mode",
                ["Crop Fill", "Blur Canvas"],
                index=["Crop Fill", "Blur Canvas"].index(st.session_state.inshot_bg_mode),
            )
            st.caption(f"Active Output Resolution: **{width} × {height}** ({ratio_str})")

        with tool_tabs[1]:
            sfx_c1, sfx_c2 = st.columns([1.6, 1.4])
            with sfx_c1:
                selected_sfx_name = st.selectbox(
                    "Nature & Reels Sound FX",
                    list(PRESET_AUDIO_EFFECTS.keys()),
                    key="inshot_sfx_select",
                )
                st.audio(PRESET_AUDIO_EFFECTS[selected_sfx_name])
            with sfx_c2:
                sfx_target = st.number_input(
                    "Target Scene #",
                    min_value=1,
                    max_value=max(10, len(st.session_state.scenes)),
                    value=1,
                    step=1,
                    key="inshot_sfx_target_num",
                ) - 1
                if st.button("➕ Attach Sound FX", key="inshot_attach_sfx_btn"):
                    cached_p = download_fx_audio(
                        selected_sfx_name, PRESET_AUDIO_EFFECTS[selected_sfx_name]
                    )
                    if cached_p:
                        st.session_state.scene_audio_fx[sfx_target] = cached_p
                        st.success(f"Added {selected_sfx_name} to Scene {sfx_target + 1}!")
                    else:
                        st.error("Unable to fetch audio effect.")

        with tool_tabs[2]:
            preset_name = st.selectbox("One-Tap Filter Mood", list(MOOD_FILTERS.keys()), key="inshot_mood_preset")
            if st.button("✨ Apply Mood Preset"):
                mp = MOOD_FILTERS[preset_name]
                st.session_state.editor_brightness = mp["b"]
                st.session_state.editor_contrast = mp["c"]
                st.session_state.editor_saturation = mp["s"]
                if mp["fx"] != "None":
                    st.session_state.editor_video_fx = mp["fx"]
                st.rerun()

            adj1, adj2 = st.columns(2)
            with adj1:
                st.session_state.editor_brightness = st.slider(
                    "Brightness", 0.5, 1.6, float(st.session_state.editor_brightness), 0.05, key="inshot_b_slider"
                )
                st.session_state.editor_contrast = st.slider(
                    "Contrast", 0.5, 1.8, float(st.session_state.editor_contrast), 0.05, key="inshot_c_slider"
                )
            with adj2:
                st.session_state.editor_saturation = st.slider(
                    "Saturation", 0.0, 2.0, float(st.session_state.editor_saturation), 0.05, key="inshot_s_slider"
                )
                st.session_state.editor_blur = st.slider(
                    "Blur", 0.0, 6.0, float(st.session_state.editor_blur), 0.25, key="inshot_blur_slider"
                )

        with tool_tabs[3]:
            fx_a, fx_b = st.columns(2)
            with fx_a:
                st.session_state.editor_video_fx = st.selectbox(
                    "Video Effect (VFX)",
                    ["None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"],
                    index=["None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"].index(
                        st.session_state.editor_video_fx
                    ),
                    key="inshot_vfx_select",
                )
            with fx_b:
                st.session_state.editor_transition = st.selectbox(
                    "Transition Between Clips",
                    ["None", "Crossfade", "Fade", "Slide Left", "Slide Right"],
                    index=["None", "Crossfade", "Fade", "Slide Left", "Slide Right"].index(
                        st.session_state.editor_transition
                    ),
                    key="inshot_trans_select",
                )
            st.session_state.editor_transition_sec = st.slider(
                "Transition Duration (sec)", 0.1, 1.5, float(st.session_state.editor_transition_sec), 0.1, key="inshot_trans_sec"
            )

        with tool_tabs[4]:
            st.session_state.inshot_custom_text = st.text_input(
                "Custom Text / Sticker Caption on Canvas",
                value=st.session_state.inshot_custom_text,
                placeholder="Type caption or sticker text to preview on canvas...",
            )
            st.session_state.inshot_custom_text_lang = st.selectbox(
                "Caption Script Font",
                ["English", "Kannada"],
                index=["English", "Kannada"].index(st.session_state.inshot_custom_text_lang),
            )

        with tool_tabs[5]:
            st.session_state.inshot_speed = st.select_slider(
                "Playback Speed",
                options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                value=float(st.session_state.inshot_speed),
            )
            up_v, up_a = st.columns(2)
            with up_v:
                extra_videos = st.file_uploader(
                    "Upload Video Clips",
                    type=["mp4", "mov", "webm", "mkv"],
                    accept_multiple_files=True,
                    key="inshot_upload_vids",
                )
                if st.button("➕ Add Video Clips", key="inshot_add_vids_btn") and extra_videos:
                    os.makedirs("uploaded_studio_media", exist_ok=True)
                    added = []
                    for idx_u, uf in enumerate(extra_videos):
                        ext = os.path.splitext(uf.name)[1].lower() or ".mp4"
                        vp = os.path.join("uploaded_studio_media", f"clip_{int(time.time())}_{idx_u}{ext}")
                        with open(vp, "wb") as f:
                            f.write(uf.getbuffer())
                        added.append(vp)
                    st.session_state.uploaded_video_clips.extend(added)
                    st.session_state.scene_videos.extend(added)
                    st.success(f"Added {len(added)} video clip(s).")
                    st.rerun()

            with up_a:
                extra_audios = st.file_uploader(
                    "Upload Audio / Voice / BGM",
                    type=["mp3", "wav", "m4a", "ogg"],
                    accept_multiple_files=True,
                    key="inshot_upload_auds",
                )
                if st.button("🎵 Add Audio Tracks", key="inshot_add_auds_btn") and extra_audios:
                    os.makedirs("uploaded_studio_media", exist_ok=True)
                    added_a = []
                    for idx_a, af in enumerate(extra_audios):
                        ext = os.path.splitext(af.name)[1].lower() or ".mp3"
                        ap = os.path.join("uploaded_studio_media", f"audio_{int(time.time())}_{idx_a}{ext}")
                        with open(ap, "wb") as f:
                            f.write(af.getbuffer())
                        added_a.append(ap)
                    st.session_state.uploaded_audio_clips.extend(added_a)
                    st.session_state.audio_files.extend(added_a)
                    st.success(f"Added {len(added_a)} audio track(s).")
                    st.rerun()

    # --- MULTI-TRACK TIMELINE STRIP ---
    st.markdown("**🎞️ Multi-Track Timeline**")
    timeline_items = st.session_state.scene_videos if st.session_state.scene_videos else st.session_state.images
    if timeline_items:
        t_cols = st.columns(min(6, len(timeline_items)))
        for idx_t in range(min(6, len(timeline_items))):
            with t_cols[idx_t]:
                is_sel = idx_t == st.session_state.inshot_selected_clip
                lbl = f"▶ Clip {idx_t + 1}" if is_sel else f"Clip {idx_t + 1}"
                if st.button(lbl, key=f"tl_sel_{idx_t}", use_container_width=True):
                    st.session_state.inshot_selected_clip = idx_t
                    st.rerun()
                sfx_tag = "🔊 SFX" if idx_t in st.session_state.scene_audio_fx else "No SFX"
                aud_tag = "🎙️ Voice" if idx_t < len(st.session_state.audio_files) else "Silent"
                st.caption(f"{aud_tag} • {sfx_tag}")
    else:
        st.caption("Timeline empty — create scenes in Tab 1 & 2 or upload clips in Speed & Media.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Rough Story & Editor",
    "2️⃣ Scene Videos",
    "3️⃣ Voice & Sound",
    "4️⃣ Render & Download",
])

# --- TAB 1: SCRIPT WRITER & INTERACTIVE EDITING AREA (ENGLISH + KANNADA) ---
with tab1:
    st.subheader("Write Your Rough Idea & Command Any Character or Environment")
    rough_story = st.text_area(
        "Type your rough story, setting, nature/background, or commands:",
        value="Follow the character through any setting, nature, or situation described in my command with clear emotion and action.",
        height=110,
    )
    num_scenes = st.slider("Number of Scenes", 2, 10, max(2, min(10, len(st.session_state.scenes) or 6)))

    if st.button("✨ Polish Story & Create Scenes", type="primary"):
        if not rough_story.strip():
            st.error("Please write a rough story or command first.")
        elif not api_key or not api_key.strip():
            st.session_state.scenes = build_fallback_scenes_from_rough_story(
                rough_story, num_scenes, st.session_state.locked_character_prompt
            )
            for idx, sc in enumerate(st.session_state.scenes):
                st.session_state[f"nar_{idx}"] = sc["narration"]
                st.session_state[f"kn_script_{idx}"] = sc["kannada_narration"]
                st.session_state[f"vis_{idx}"] = sc["visual"]
                st.session_state[f"emo_{idx}"] = sc.get("emotion", "Expressive")
            st.warning(f"No Gemini API Key provided—created {len(st.session_state.scenes)} scenes directly from your command.")
        else:
            fallback_models = [
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
            ]
            char_context = (
                f"\nUser character/environment instruction ({st.session_state.exact_character_count} main character(s)): {st.session_state.locked_character_prompt}"
                if st.session_state.lock_character_enabled
                else f"\nCharacter/environment notes: {st.session_state.locked_character_prompt}"
            )
            prompt = f"""
            Create a {num_scenes}-scene story script that strictly obeys every user command for characters, nature, backgrounds, locations, and actions.{char_context}
            Return ONLY valid JSON as a list of {num_scenes} objects, with no markdown formatting.
            Each object must have:
            - "narration": clear English voiceover text (1-2 sentences)
            - "kannada_narration": natural, clear Kannada translation/script of that scene's voiceover (1-2 sentences)
            - "emotion": mood or expression for that scene
            - "visual": visual scene description honoring the user's requested character, setting/nature/background, camera framing, and action
            User story & commands: {rough_story.strip()}
            """
            client = genai.Client(api_key=api_key.strip())
            last_exc = None
            parsed_scenes = None

            with st.spinner(f"Creating {num_scenes} English & Kannada scenes from your commands..."):
                for model_name in fallback_models:
                    for attempt in range(2):
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                            )
                            raw_text = (response.text or "").strip()
                            if raw_text.startswith("```"):
                                lines = [
                                    line for line in raw_text.splitlines()
                                    if not line.strip().startswith("```")
                                ]
                                raw_text = "\n".join(lines).strip()

                            candidate = json.loads(raw_text)
                            if isinstance(candidate, list) and len(candidate) > 0:
                                parsed_scenes = candidate
                                break
                        except Exception as exc:
                            last_exc = exc
                            msg = str(exc)
                            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                                time.sleep(1.5 * (attempt + 1))
                                continue
                            break
                    if parsed_scenes is not None:
                        break

            if parsed_scenes is not None:
                normalized = [
                    {
                        "narration": str(item.get("narration", "")).strip(),
                        "kannada_narration": str(item.get("kannada_narration", "")).strip(),
                        "emotion": str(item.get("emotion", "Expressive")).strip(),
                        "visual": str(item.get("visual", "")).strip(),
                    }
                    for item in parsed_scenes[:num_scenes]
                ]
                st.session_state.scenes = sync_scenes_to_count(
                    normalized, num_scenes, rough_story, st.session_state.locked_character_prompt
                )
                for idx, sc in enumerate(st.session_state.scenes):
                    st.session_state[f"nar_{idx}"] = sc["narration"]
                    st.session_state[f"kn_script_{idx}"] = sc["kannada_narration"]
                    st.session_state[f"vis_{idx}"] = sc["visual"]
                    st.session_state[f"emo_{idx}"] = sc["emotion"]
                    st.session_state.kannada_subtitles[idx] = sc["kannada_narration"]
                st.success(f"{len(st.session_state.scenes)} English & Kannada scenes ready! Edit any scene below or move to Tab 2.")
            else:
                st.session_state.scenes = build_fallback_scenes_from_rough_story(
                    rough_story, num_scenes, st.session_state.locked_character_prompt
                )
                for idx, sc in enumerate(st.session_state.scenes):
                    st.session_state[f"nar_{idx}"] = sc["narration"]
                    st.session_state[f"kn_script_{idx}"] = sc["kannada_narration"]
                    st.session_state[f"vis_{idx}"] = sc["visual"]
                    st.session_state[f"emo_{idx}"] = sc.get("emotion", "Expressive")
                st.warning(f"Gemini quota/error hit ({last_exc})—created {len(st.session_state.scenes)} fallback scenes from your command.")

    if st.session_state.scenes:
        st.markdown(f"### ✏️ Interactive Story Editor ({len(st.session_state.scenes)} Scenes Active)")
        top_tr_col1, top_tr_col2 = st.columns([1.4, 2.6])
        with top_tr_col1:
            if st.button("🌐 Auto-Translate All Scenes to Kannada"):
                if not api_key or not api_key.strip():
                    st.error("Enter your Gemini API Key in the sidebar to auto-translate to Kannada.")
                else:
                    en_list = [
                        (st.session_state.get(f"nar_{i}", sc.get("narration", "")) or f"Scene {i + 1}.").strip()
                        for i, sc in enumerate(st.session_state.scenes)
                    ]
                    with st.spinner("Translating all scenes to Kannada..."):
                        kn_list = translate_lines_to_kannada(api_key, en_list)
                    for i, kn_val in enumerate(kn_list):
                        st.session_state.scenes[i]["kannada_narration"] = kn_val
                        st.session_state[f"kn_script_{i}"] = kn_val
                        st.session_state.kannada_subtitles[i] = kn_val
                    st.success("Translated all scenes into Kannada! Review or edit each box below.")
                    st.rerun()

        for i, scene in enumerate(st.session_state.scenes):
            with st.expander(f"Scene {i + 1} Editor ({scene.get('emotion', 'Emotion')})", expanded=(i < 4)):
                col_a, col_b = st.columns(2)
                with col_a:
                    nar_val = st.text_area(
                        f"Scene {i + 1} English Script / Narration",
                        value=scene.get("narration", ""),
                        key=f"nar_{i}",
                    )
                    scene["narration"] = nar_val
                    emo_val = st.text_input(
                        f"Scene {i + 1} Mood / Expression",
                        value=scene.get("emotion", "Expressive"),
                        key=f"emo_{i}",
                    )
                    scene["emotion"] = emo_val
                    vis_val = st.text_input(
                        f"Scene {i + 1} Character, Background & Action Command",
                        value=scene.get("visual", ""),
                        key=f"vis_{i}",
                    )
                    scene["visual"] = vis_val
                with col_b:
                    kn_script_val = st.text_area(
                        f"Scene {i + 1} Kannada Script (ಕನ್ನಡ ಕಥೆ)",
                        value=scene.get("kannada_narration", ""),
                        placeholder=f"Enter or edit Kannada script for Scene {i + 1}",
                        key=f"kn_script_{i}",
                    )
                    scene["kannada_narration"] = kn_script_val
                    st.session_state.kannada_subtitles[i] = kn_script_val

                    if st.button(f"🌐 Auto-Translate Scene {i + 1} to Kannada", key=f"tr_scene_{i}"):
                        if not api_key or not api_key.strip():
                            st.error("Enter your Gemini API Key in the sidebar to translate this scene.")
                        elif not nar_val.strip():
                            st.warning("Write or keep English narration in Scene %d first." % (i + 1))
                        else:
                            with st.spinner(f"Translating Scene {i + 1} to Kannada..."):
                                translated_one = translate_lines_to_kannada(api_key, [nar_val.strip()])
                            kn_one = translated_one[0] if translated_one else ""
                            if kn_one:
                                scene["kannada_narration"] = kn_one
                                st.session_state[f"kn_script_{i}"] = kn_one
                                st.session_state.kannada_subtitles[i] = kn_one
                                st.rerun()

# --- TAB 2: SCENE-WISE VIDEO CREATOR ---
with tab2:
    st.subheader("Create Scene-by-Scene Videos from Your Commands")
    art_style = st.selectbox(
        "Choose Visual Style",
        [
            "Photorealistic cinematic film still, natural lighting matching the requested environment, lifelike human expressions, 35mm lens",
            "High-detail 3D CGI animation, expressive lifelike characters, rich environment lighting and depth",
            "Detailed storybook illustration, expressive characters, rich scenery and atmosphere",
        ],
    )

    ref_col1, ref_col2 = st.columns([1.4, 1])
    with ref_col1:
        active_char_prompt = st.text_area(
            "Character & Environment Command Prompt",
            value=st.session_state.locked_character_prompt,
            height=90,
        )
        st.session_state.locked_character_prompt = active_char_prompt
    with ref_col2:
        ref_file = st.file_uploader(
            "Character / Scene Reference Image (Optional)",
            type=["png", "jpg", "jpeg", "webp"],
        )

    vid_opt1, vid_opt2, vid_opt3 = st.columns([1.1, 1.1, 1.4])
    with vid_opt1:
        scene_count_slider = st.slider(
            "Number of Scenes (2 to 10)",
            2,
            10,
            max(2, min(10, len(st.session_state.scenes) or 6)),
        )
    with vid_opt2:
        per_scene_sec = st.slider("Default Clip Length (sec)", 2.0, 10.0, 4.0, 0.5)
    with vid_opt3:
        scene_motion_mode = st.selectbox(
            "Scene Motion Style",
            ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"],
        )

    sub_col1, sub_col2 = st.columns([1.3, 2.7])
    with sub_col1:
        tab2_sub_lang = st.selectbox(
            "Scene Video Subtitles",
            ["None", "English", "Kannada"],
            index=2,
            key="tab2_subtitle_lang",
        )
    with sub_col2:
        allow_local_bypass = st.checkbox(
            "Enable zero-API local bypass frames if all external image APIs fail/quota-out",
            value=True,
        )

    st.session_state["tab2_baked_subtitles"] = tab2_sub_lang != "None"

    ref_pil = None
    if ref_file is not None:
        ref_pil = Image.open(ref_file).convert("RGB")
        st.image(ref_pil, caption="Uploaded Reference Image", width=260)

    if st.button("🎬 Generate Scene Videos", type="primary"):
        active_scenes = sync_scenes_to_count(
            st.session_state.scenes,
            scene_count_slider,
            rough_story,
            st.session_state.locked_character_prompt,
        )
        for idx, sc in enumerate(active_scenes):
            if f"nar_{idx}" in st.session_state:
                sc["narration"] = st.session_state[f"nar_{idx}"]
            if f"kn_script_{idx}" in st.session_state:
                sc["kannada_narration"] = st.session_state[f"kn_script_{idx}"]
            if f"vis_{idx}" in st.session_state:
                sc["visual"] = st.session_state[f"vis_{idx}"]
            if f"emo_{idx}" in st.session_state:
                sc["emotion"] = st.session_state[f"emo_{idx}"]

        st.session_state.scenes = active_scenes
        for idx, sc in enumerate(st.session_state.scenes):
            st.session_state.setdefault(f"nar_{idx}", sc.get("narration", ""))
            st.session_state.setdefault(f"kn_script_{idx}", sc.get("kannada_narration", ""))
            st.session_state.setdefault(f"vis_{idx}", sc.get("visual", ""))
            st.session_state.setdefault(f"emo_{idx}", sc.get("emotion", "Expressive"))

        scene_subtitles = (
            get_script_lines_for_language(
                st.session_state.scenes,
                script_lang=tab2_sub_lang,
                gemini_key=api_key,
            )
            if tab2_sub_lang != "None"
            else [""] * len(st.session_state.scenes)
        )

        st.session_state.gen_batch_id += 1
        batch_id = st.session_state.gen_batch_id

        char_to_use = st.session_state.locked_character_prompt or active_char_prompt
        cast_guard = (
            f"Follow user command strictly: {char_to_use} ({int(st.session_state.exact_character_count)} main subject(s) when applicable). "
            f"Match whatever setting, background, or nature the user commands."
            if st.session_state.lock_character_enabled
            else char_to_use
        )

        st.session_state.images = []
        st.session_state.scene_videos = []
        os.makedirs("scene_videos", exist_ok=True)
        progress = st.progress(0)
        used_sources = []
        all_errors = []

        for idx, scene in enumerate(active_scenes):
            if idx > 0:
                time.sleep(0.35)
            scene_vis = st.session_state.get(f"vis_{idx}", scene.get("visual", ""))
            scene_emo = st.session_state.get(f"emo_{idx}", scene.get("emotion", "Expressive"))
            full_prompt = (
                f"{art_style}. {cast_guard} "
                f"Scene {idx + 1} mood: {scene_emo}. "
                f"Scene {idx + 1} instruction: {scene_vis}. "
                f"Story context: {rough_story.strip()}"
            ).strip()

            pil_img, src_name, src_errs = generate_real_ai_scene(
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
                emotion_label=scene_emo,
                seed_offset=batch_id * 100,
            )
            if src_errs:
                all_errors.extend([f"Scene {idx + 1}: {e}" for e in src_errs])

            if pil_img is not None:
                exact_img = ensure_exact_size(pil_img, width, height, bg_mode=st.session_state.inshot_bg_mode)
                img_path = f"scene_{idx}_b{batch_id}.png"
                exact_img.save(img_path, format="PNG")
                st.session_state.images.append(img_path)
                used_sources.append(src_name or "AI")

                scene_aud = (
                    st.session_state.audio_files[idx]
                    if idx < len(st.session_state.audio_files)
                    else None
                )
                clip_dur = per_scene_sec
                if scene_aud and os.path.exists(scene_aud):
                    try:
                        ac = AudioFileClip(scene_aud)
                        clip_dur = max(float(ac.duration or per_scene_sec) + 0.3, 1.0)
                        ac.close()
                    except Exception:
                        pass

                sub_line = scene_subtitles[idx] if idx < len(scene_subtitles) else ""
                vid_path = os.path.join("scene_videos", f"scene_{idx}_b{batch_id}.mp4")
                render_single_scene_video(
                    img_path=img_path,
                    out_path=vid_path,
                    target_w=width,
                    target_h=height,
                    duration=clip_dur,
                    motion_mode=scene_motion_mode,
                    scene_idx=idx,
                    subtitle_text=sub_line,
                    subtitle_lang=tab2_sub_lang,
                    enable_char_motion=True,
                    brightness=st.session_state.editor_brightness,
                    contrast=st.session_state.editor_contrast,
                    saturation=st.session_state.editor_saturation,
                    blur_radius=st.session_state.editor_blur,
                    video_fx=st.session_state.editor_video_fx,
                    bg_mode=st.session_state.inshot_bg_mode,
                )
                st.session_state.scene_videos.append(vid_path)

            progress.progress((idx + 1) / len(active_scenes))

        if st.session_state.scene_videos:
            st.success(
                f"Created {len(st.session_state.scene_videos)} scene videos at {width}x{height} "
                f"(Subtitles: {tab2_sub_lang}, Sources: {', '.join(set(used_sources))})."
            )
        else:
            st.error(
                "No scene videos were created. Enable local bypass or inspect API / Quota Details below."
            )

        if all_errors:
            with st.expander("API / Quota Details"):
                for err in all_errors:
                    st.write(err)

    if st.session_state.scene_videos:
        cols = st.columns(2)
        for idx, vid_path in enumerate(st.session_state.scene_videos):
            if os.path.exists(vid_path):
                emo_cap = (
                    st.session_state.scenes[idx].get("emotion", "")
                    if idx < len(st.session_state.scenes)
                    else ""
                )
                cap_txt = (
                    f"Scene {idx + 1} • {emo_cap} ({width}x{height})"
                    if emo_cap
                    else f"Scene {idx + 1} ({width}x{height})"
                )
                with cols[idx % 2]:
                    st.markdown(f"**{cap_txt}**")
                    st.video(vid_path)
                    with open(vid_path, "rb") as vf:
                        st.download_button(
                            label=f"⬇️ Scene {idx + 1} Video",
                            data=vf.read(),
                            file_name=f"scene_{idx + 1}.mp4",
                            mime="video/mp4",
                            key=f"dl_scene_vid_{idx}_b{st.session_state.gen_batch_id}",
                        )

# --- TAB 3: VOICE & SOUND ---
with tab3:
    st.subheader("Select Narrator Voice, Upload Audio & Script Language (English or Kannada)")
    voice_map = {
        "Warm Male Storyteller (US)": ("en-US-ChristopherNeural", "English"),
        "Expressive Female Storyteller (US)": ("en-US-AriaNeural", "English"),
        "Indian Male Narrator (English)": ("en-IN-PrabhatNeural", "English"),
        "Indian Female Narrator (English)": ("en-IN-NeerjaNeural", "English"),
        "Kannada Male Narrator (ಕನ್ನಡ - Gagan)": ("kn-IN-GaganNeural", "Kannada"),
        "Kannada Female Narrator (ಕನ್ನಡ - Sapna)": ("kn-IN-SapnaNeural", "Kannada"),
        "British Documentary Male": ("en-GB-RyanNeural", "English"),
        "My Own Voice (Upload Recording)": (None, "Custom"),
    }

    vc1, vc2 = st.columns(2)
    with vc1:
        selected_voice = st.selectbox("Voice Options", list(voice_map.keys()))
        voice_id, voice_lang = voice_map[selected_voice]
    with vc2:
        default_script_choice = 1 if voice_lang == "Kannada" else 0
        selected_script_lang = st.selectbox(
            "Script Language to Populate & Read",
            ["English", "Kannada"],
            index=default_script_choice,
        )

    if st.button("📥 Populate Selected Script Lines"):
        if not st.session_state.scenes:
            st.warning("Create or edit scenes in Tab 1 first.")
        else:
            with st.spinner(f"Populating {selected_script_lang} script lines..."):
                st.session_state.populated_script_lines = get_script_lines_for_language(
                    st.session_state.scenes,
                    script_lang=selected_script_lang,
                    gemini_key=api_key,
                )
                st.session_state.populated_script_lang = selected_script_lang
            st.success(f"Populated {len(st.session_state.populated_script_lines)} {selected_script_lang} script line(s).")

    active_script_lines = (
        st.session_state.populated_script_lines
        if (
            st.session_state.populated_script_lines
            and st.session_state.populated_script_lang == selected_script_lang
        )
        else get_script_lines_for_language(
            st.session_state.scenes,
            script_lang=selected_script_lang,
            gemini_key=api_key,
        )
    )

    if active_script_lines:
        st.markdown(f"**Active {selected_script_lang} Script Lines ({len(active_script_lines)} Scenes):**")
        for s_idx, s_line in enumerate(active_script_lines):
            st.write(f"**Scene {s_idx + 1} ({selected_script_lang}):** {s_line}")

    async def make_audio(text, voice, filename):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)

    if voice_id:
        if st.button("🎙️ Preview Sample Voice"):
            try:
                with st.spinner("Generating voice sample..."):
                    sample_text = (
                        active_script_lines[0]
                        if active_script_lines
                        else (
                            "ನೀವು ಬಯಸಿದ ಪ್ರತಿಯೊಂದು ವಿವರದೊಂದಿಗೆ ದೃಶ್ಯ ಜೀವಂತವಾಗುತ್ತಾ ಪ್ರಯಾಣ ಆರಂಭವಾಗುತ್ತದೆ."
                            if selected_script_lang == "Kannada"
                            else "The journey begins as the setting comes alive with every detail you commanded."
                        )
                    )
                    run_async(make_audio(sample_text, voice_id, "sample.mp3"))
                st.audio("sample.mp3")
            except Exception as exc:
                st.error(f"Could not generate sample audio: {exc}")

        if st.button("🔊 Generate Full Story Narration", type="primary"):
            if not active_script_lines:
                st.warning("Generate your script in Tab 1 first.")
            else:
                try:
                    st.session_state.audio_files = []
                    progress = st.progress(0)
                    for idx, line_text in enumerate(active_script_lines):
                        aud_path = f"voice_{idx}.mp3"
                        run_async(make_audio(line_text, voice_id, aud_path))
                        st.session_state.audio_files.append(aud_path)
                        progress.progress((idx + 1) / len(active_script_lines))
                    st.success(f"Generated {len(st.session_state.audio_files)} {selected_script_lang} voiceover(s)! Go to Tab 4 to render.")
                except Exception as exc:
                    st.error(f"Error generating narration audio: {exc}")
    else:
        st.info(
            "Upload your own recorded voice files below (either one audio file for the full story or separate files per scene)."
        )
        uploaded_voices = st.file_uploader(
            "Upload Your Voice Recording (.mp3, .wav, .m4a, .ogg)",
            type=["mp3", "wav", "m4a", "ogg"],
            accept_multiple_files=True,
        )
        if st.button("💾 Use My Uploaded Voice", type="primary"):
            if not uploaded_voices:
                st.error("Please upload at least one voice recording first.")
            else:
                os.makedirs("custom_voice", exist_ok=True)
                saved_paths = []
                for idx, up_file in enumerate(uploaded_voices):
                    ext = os.path.splitext(up_file.name)[1].lower() or ".mp3"
                    out_p = os.path.join("custom_voice", f"my_voice_{idx}{ext}")
                    with open(out_p, "wb") as f:
                        f.write(up_file.getbuffer())
                    saved_paths.append(out_p)

                scene_total = max(len(st.session_state.scenes), len(st.session_state.images), len(st.session_state.scene_videos), 1)
                if len(saved_paths) == 1 and scene_total > 1:
                    full_clip = AudioFileClip(saved_paths[0])
                    try:
                        total_dur = max(float(full_clip.duration or 4.0), 1.0)
                        seg_dur = total_dur / scene_total
                        split_paths = []
                        for s_i in range(scene_total):
                            start_t = s_i * seg_dur
                            end_t = min(total_dur, (s_i + 1) * seg_dur)
                            seg_clip = (
                                full_clip.subclipped(start_t, end_t)
                                if hasattr(full_clip, "subclipped")
                                else full_clip.subclip(start_t, end_t)
                            )
                            seg_path = os.path.join("custom_voice", f"scene_voice_{s_i}.mp3")
                            seg_clip.write_audiofile(seg_path, logger=None)
                            seg_clip.close()
                            split_paths.append(seg_path)
                        st.session_state.audio_files = split_paths
                    finally:
                        full_clip.close()
                else:
                    st.session_state.audio_files = saved_paths[:scene_total]

                st.success(f"Saved {len(st.session_state.audio_files)} voice track(s)! Go to Tab 4 to render.")

        if st.session_state.audio_files:
            st.audio(st.session_state.audio_files[0])

    # --- SEPARATE TABLE FOR MY OWN VOICE RECORDING & SAVING ---
    st.divider()
    st.markdown("### 📋 My Own Voice — Separate Script & Recording Table")
    st.caption(
        f"Currently showing **{selected_script_lang}** script lines to read. Record with your microphone (or upload a clip per row), preview it, and click Save My Voice Table."
    )

    table_count = max(
        len(active_script_lines),
        len(st.session_state.scenes),
        len(st.session_state.images),
        len(st.session_state.scene_videos),
        2,
    )
    os.makedirs("own_voice_table", exist_ok=True)
    draft_rows = []

    for idx in range(table_count):
        if idx < len(active_script_lines):
            line_to_read = active_script_lines[idx]
        elif idx < len(st.session_state.scenes):
            line_to_read = (
                st.session_state.scenes[idx].get("kannada_narration")
                if selected_script_lang == "Kannada"
                else st.session_state.scenes[idx].get("narration")
            ) or f"Scene {idx + 1}."
        else:
            line_to_read = f"Scene {idx + 1} script line to read aloud."

        row_c1, row_c2, row_c3, row_c4 = st.columns([0.9, 2.4, 2.1, 1.6])
        with row_c1:
            st.markdown(f"**Scene {idx + 1}**")
            existing_path = st.session_state.own_voice_table.get(idx, "")
            st.caption("✅ Saved" if existing_path and os.path.exists(existing_path) else "⏳ Pending")

        with row_c2:
            st.markdown(f"**Read This ({selected_script_lang}) Script:**")
            st.write(line_to_read)

        with row_c3:
            mic_val = None
            if hasattr(st, "audio_input"):
                mic_val = st.audio_input(f"Record Scene {idx + 1}", key=f"mic_scene_{idx}")
            up_row_val = st.file_uploader(
                f"Or upload Scene {idx + 1} audio",
                type=["mp3", "wav", "m4a", "ogg"],
                key=f"up_scene_{idx}",
            )

        with row_c4:
            saved_p = st.session_state.own_voice_table.get(idx, "")
            if mic_val is not None:
                st.audio(mic_val)
            elif up_row_val is not None:
                st.audio(up_row_val)
            elif saved_p and os.path.exists(saved_p):
                st.audio(saved_p)
            else:
                st.caption("No audio yet")

        draft_rows.append((idx, line_to_read, mic_val, up_row_val))

    if st.button("💾 Save My Voice Table", type="primary"):
        saved_audio_list = []
        missing_rows = []

        for idx, _, mic_val, up_row_val in draft_rows:
            out_path = os.path.join("own_voice_table", f"scene_{idx}_own_voice.wav")
            if mic_val is not None:
                with open(out_path, "wb") as f:
                    f.write(mic_val.getvalue())
                st.session_state.own_voice_table[idx] = out_path
                saved_audio_list.append(out_path)
            elif up_row_val is not None:
                ext = os.path.splitext(up_row_val.name)[1].lower() or ".mp3"
                alt_path = os.path.join("own_voice_table", f"scene_{idx}_own_voice{ext}")
                with open(alt_path, "wb") as f:
                    f.write(up_row_val.getbuffer())
                st.session_state.own_voice_table[idx] = alt_path
                saved_audio_list.append(alt_path)
            elif st.session_state.own_voice_table.get(idx) and os.path.exists(
                st.session_state.own_voice_table[idx]
            ):
                saved_audio_list.append(st.session_state.own_voice_table[idx])
            else:
                missing_rows.append(idx + 1)

        if saved_audio_list:
            st.session_state.audio_files = saved_audio_list
            st.success(
                f"Saved {len(saved_audio_list)} voice track(s) from your table! Ready for Tab 4 rendering."
            )
            if missing_rows:
                st.warning(f"No recording saved yet for Scene(s): {', '.join(map(str, missing_rows))}.")
        else:
            st.error("Please record or upload at least one scene voice clip in the table before saving.")

# --- TAB 4: RENDER & DOWNLOAD ---
with tab4:
    st.subheader("Final Timeline Assembly, Transitions, Sound FX, Subtitles & Exact-Size MP4 Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        free_motion_style = st.selectbox(
            "Camera Motion",
            ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"],
        )
    with c2:
        subtitle_lang = st.selectbox(
            "Subtitles Language",
            ["None", "English", "Kannada"],
            index=2,
        )
    with c3:
        enable_char_motion = st.checkbox("Character Movement", value=True)

    _, active_font_source = _find_font(28, lang=subtitle_lang if subtitle_lang != "None" else "English")
    st.caption(
        f"Target size: `{width}x{height}` ({ratio_str}) | Speed: `{st.session_state.inshot_speed}x` | "
        f"Transition: `{st.session_state.editor_transition}` | Video FX: `{st.session_state.editor_video_fx}` | Font: `{active_font_source}`"
    )

    if st.button("🎬 Render Final MP4 Video", type="primary"):
        has_visuals = bool(st.session_state.scene_videos or st.session_state.images)
        if not has_visuals:
            st.error("Generate scene videos/images in Tab 2 or upload video clips in the InShot Editing Deck first.")
        else:
            clips = []
            audio_clips = []
            source_video_clips = []
            final_video = None
            output_file = f"animated_story_b{st.session_state.gen_batch_id}.mp4"

            try:
                total_visual_slots = max(len(st.session_state.scene_videos), len(st.session_state.images))
                speed_mult = max(0.5, float(st.session_state.inshot_speed))
                with st.spinner(
                    f"Rendering {total_visual_slots}-clip {width}x{height} video at {speed_mult}x speed with {st.session_state.editor_transition} transitions..."
                ):
                    if subtitle_lang == "Kannada":
                        sub_lines = get_script_lines_for_language(
                            st.session_state.scenes,
                            script_lang="Kannada",
                            gemini_key=api_key,
                        )
                    elif subtitle_lang == "English":
                        sub_lines = get_script_lines_for_language(
                            st.session_state.scenes,
                            script_lang="English",
                            gemini_key=api_key,
                        )
                    else:
                        sub_lines = [""] * total_visual_slots

                    while len(sub_lines) < total_visual_slots:
                        sub_lines.append("")

                    for i in range(total_visual_slots):
                        aud_p = st.session_state.audio_files[i] if i < len(st.session_state.audio_files) else None
                        sfx_p = st.session_state.scene_audio_fx.get(i)

                        aud_clip = None
                        if aud_p and os.path.exists(aud_p):
                            aud_clip = AudioFileClip(aud_p)
                            audio_clips.append(aud_clip)

                        if sfx_p and os.path.exists(sfx_p):
                            sfx_clip = AudioFileClip(sfx_p)
                            audio_clips.append(sfx_clip)
                            if aud_clip is not None:
                                aud_clip = CompositeAudioClip([aud_clip, sfx_clip])
                            else:
                                aud_clip = sfx_clip

                        if aud_clip is not None:
                            target_duration = max((float(aud_clip.duration or 1.0) + 0.3) / speed_mult, 0.8)
                        else:
                            target_duration = max(4.0 / speed_mult, 0.8)

                        combined_sub = sub_lines[i] if i < len(sub_lines) else ""
                        if st.session_state.inshot_custom_text.strip():
                            combined_sub = (
                                f"{combined_sub} • {st.session_state.inshot_custom_text.strip()}"
                                if combined_sub
                                else st.session_state.inshot_custom_text.strip()
                            )

                        if i < len(st.session_state.scene_videos) and os.path.exists(st.session_state.scene_videos[i]):
                            src_vid = VideoFileClip(st.session_state.scene_videos[i])
                            source_video_clips.append(src_vid)
                            if (src_vid.duration or 0) < target_duration and hasattr(src_vid, "loop"):
                                motion_clip = apply_duration(src_vid.loop(duration=target_duration), target_duration)
                            else:
                                motion_clip = apply_duration(src_vid, target_duration)
                        else:
                            motion_clip = build_motion_clip(
                                st.session_state.images[i],
                                width,
                                height,
                                target_duration,
                                motion_mode=free_motion_style,
                                scene_idx=i,
                                subtitle_text=combined_sub,
                                subtitle_lang=subtitle_lang,
                                enable_char_motion=enable_char_motion,
                                brightness=st.session_state.editor_brightness,
                                contrast=st.session_state.editor_contrast,
                                saturation=st.session_state.editor_saturation,
                                blur_radius=st.session_state.editor_blur,
                                video_fx=st.session_state.editor_video_fx,
                                bg_mode=st.session_state.inshot_bg_mode,
                            )

                        if aud_clip is not None:
                            motion_clip = apply_audio(motion_clip, apply_duration(aud_clip, target_duration))
                        clips.append(motion_clip)

                    final_video = apply_clip_transition(
                        clips,
                        transition_name=st.session_state.editor_transition,
                        trans_sec=st.session_state.editor_transition_sec,
                    )
                    final_video.write_videofile(
                        output_file,
                        fps=24,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None,
                    )

                st.video(output_file)
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Video for Reels / YouTube",
                        data=f.read(),
                        file_name="my_animated_story.mp4",
                        mime="video/mp4",
                    )
            except Exception as exc:
                st.error(f"Video rendering failed: {exc}")
            finally:
                for c in clips:
                    try:
                        c.close()
                    except Exception:
                        pass
                for sv in source_video_clips:
                    try:
                        sv.close()
                    except Exception:
                        pass
                for a in audio_clips:
                    try:
                        a.close()
                    except Exception:
                        pass
                if final_video is not None:
                    try:
                        final_video.close()
                    except Exception:
                        pass
