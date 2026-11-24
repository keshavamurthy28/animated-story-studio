import asyncio
import base64
import json
import math
import os
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
    page_title="AI Cinema & Reels Studio Pro",
    page_icon="🎬",
    layout="wide",
)

# Responsive styling for both laptop and mobile screens
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
        gap: 10px;
        background: linear-gradient(90deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 14px;
    }
    .studio-pill {
        background: #ff2e63;
        color: #ffffff;
        font-weight: 700;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 999px;
        letter-spacing: 0.4px;
    }
    .studio-sub {
        color: #9da7b3;
        font-size: 13px;
    }
    .compact-preview-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #131822;
        border: 1px solid #283142;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
    }
    .compact-preview-box img,
    .compact-preview-box video,
    .compact-preview-box [data-testid="stImage"] img,
    .compact-preview-box [data-testid="stVideo"] video {
        max-width: 320px !important;
        max-height: 340px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 10px;
        margin: 0 auto !important;
        display: block !important;
    }
    .stButton > button {
        border-radius: 10px;
        padding: 0.55rem 0.95rem;
        font-weight: 600;
        min-height: 42px;
    }
    @media (max-width: 768px) {
        html, body, [class*="css"] {
            font-size: 14px;
        }
        .studio-header {
            padding: 10px 12px;
        }
        .compact-preview-box img,
        .compact-preview-box video,
        .compact-preview-box [data-testid="stImage"] img,
        .compact-preview-box [data-testid="stVideo"] video {
            max-width: 245px !important;
            max-height: 260px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_CHAR_PROMPT = (
    "Ancient Indian mythological hero and divine figures in traditional royal attire, crown, and ornaments; "
    "epic yuga setting with grand ancient architecture, glowing sun, planets, and sacred sky"
)

DEFAULT_SCENES = [
    {
        "narration": "In the dawn of an ancient yuga, celestial planets and a glowing sun illuminate the sacred realm.",
        "kannada_narration": "ಪ್ರಾಚೀನ ಯುಗದ ಉದಯದಲ್ಲಿ ಸೂರ್ಯ, ಗ್ರಹಗಳು ಮತ್ತು ದಿವ್ಯ ಬೆಳಕು ಪವಿತ್ರ ಲೋಕವನ್ನು ಬೆಳಗಿಸುತ್ತವೆ.",
        "emotion": "Sacred & Majestic",
        "motion": "Zoom In",
        "suggested_broll": "Wide celestial establishing shot with glowing sun, visible planets, and ancient yuga horizon",
        "suggested_fx": "Cinematic Vignette",
        "suggested_sfx": "🌲 Forest Birds",
        "visual": (
            "Grand ancient yuga vista with glowing sun, planets in the sky, and regal mythological figure "
            "in traditional crown, jewelry, and dhoti"
        ),
    },
    {
        "narration": "A moment of determination rises as the hero stands ready amid shifting cosmic light.",
        "kannada_narration": "ಬದಲಾಗುವ ದಿವ್ಯ ಬೆಳಕಿನ ನಡುವೆ ವೀರನು ದೃಢ ನಿರ್ಧಾರದೊಂದಿಗೆ ಸಜ್ಜಾಗುತ್ತಾನೆ.",
        "emotion": "Determination",
        "motion": "Pan Right",
        "suggested_broll": "Medium hero shot with dramatic rim lighting, ancient pillars, and planetary glow",
        "suggested_fx": "Zoom Punch",
        "suggested_sfx": "💥 Vine Boom (Meme)",
        "visual": (
            "Heroic mythological warrior in royal ornaments and traditional attire standing in an ancient "
            "courtyard with glowing sun and planets above"
        ),
    },
    {
        "narration": "Across sacred lands and timeless eras, divine presence guides every unfolding step.",
        "kannada_narration": "ಪವಿತ್ರ ಭೂಮಿ ಮತ್ತು ಕಾಲಾತೀತ ಯುಗಗಳ ನಡುವೆ ದಿವ್ಯ ಶಕ್ತಿಯು ಪ್ರತಿ ಹೆಜ್ಜೆಯನ್ನು ಮುನ್ನಡೆಸುತ್ತದೆ.",
        "emotion": "Reflective & Divine",
        "motion": "Zoom Out",
        "suggested_broll": "Sweeping pull-back shot showing ancient temple/forest scenery and celestial sky",
        "suggested_fx": "None",
        "suggested_sfx": "🌊 Ocean Waves",
        "visual": (
            "Ancient yuga landscape with mythological character, sacred architecture, glowing sun, "
            "and planets across a deep cosmic sky"
        ),
    },
    {
        "narration": "The scene culminates in a radiant vision uniting character, yuga heritage, and the cosmos.",
        "kannada_narration": "ಪಾತ್ರ, ಯುಗದ ಪರಂಪರೆ ಮತ್ತು ಬ್ರಹ್ಮಾಂಡವನ್ನು ಒಂದುಗೂಡಿಸುವ ತೇಜಸ್ವಿ ದೃಶ್ಯದಲ್ಲಿ ಕಥೆ ಸಾಗುತ್ತದೆ.",
        "emotion": "Radiant & Triumphant",
        "motion": "Pan Left",
        "suggested_broll": "Closing cinematic shot with golden rays, planets, and composed hero framing",
        "suggested_fx": "Cinematic Vignette",
        "suggested_sfx": "🔥 Campfire Crackle",
        "visual": (
            "Cinematic closing shot of ancient mythological character with crown and ornaments against "
            "yuga scenery, glowing sun, and planets"
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

VIDEO_FX_CHOICES = [
    "None",
    "Zoom Punch",
    "Shake",
    "Flash White",
    "Glitch RGB",
    "Vintage Sepia",
    "Cinematic Vignette",
]

MOTION_CHOICES = ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
SUBTITLE_POSITIONS = ["Bottom", "Center", "Top"]

for key, default in [
    ("scenes", [dict(s) for s in DEFAULT_SCENES]),
    ("images", []),
    ("scene_videos", []),
    ("uploaded_photo_paths", []),
    ("uploaded_video_clips", []),
    ("uploaded_audio_clips", []),
    ("arrangement_command", "Hero spotlight frame with glowing golden border, dynamic zoom, and bold reel wordings"),
    ("reel_wordings_input", ""),
    ("audio_files", []),
    ("bgm_audio_path", ""),
    ("bgm_volume", 0.22),
    ("bgm_duck_enabled", True),
    ("bgm_duck_volume", 0.08),
    ("voice_volume", 1.0),
    ("sfx_volume", 0.75),
    ("locked_character_prompt", DEFAULT_CHAR_PROMPT),
    ("exact_character_count", 4),
    ("lock_character_enabled", True),
    ("gen_batch_id", 0),
    ("selected_clip_idx", 0),
    ("format_preset", "9:16 Reels / Shorts"),
    ("bg_fit_mode", "Blur Canvas"),
    ("playback_speed", 1.0),
    ("custom_title_text", ""),
    ("custom_title_lang", "English"),
    ("subtitle_position", "Bottom"),
    ("subtitle_scale", 1.0),
    ("editor_brightness", 1.05),
    ("editor_contrast", 1.12),
    ("editor_saturation", 1.24),
    ("editor_blur", 0.0),
    ("editor_transition", "Crossfade"),
    ("editor_transition_sec", 0.5),
    ("editor_video_fx", "None"),
    ("scene_audio_fx", {}),
    ("scene_fx_overrides", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def sync_scene_widget_states(scenes_list):
    """Push scene list fields into Streamlit widget keys cleanly."""
    for idx, sc in enumerate(scenes_list):
        st.session_state[f"nar_{idx}"] = sc.get("narration", "")
        st.session_state[f"kn_script_{idx}"] = sc.get("kannada_narration", "")
        st.session_state[f"vis_{idx}"] = sc.get("visual", "")
        st.session_state[f"emo_{idx}"] = sc.get("emotion", "Expressive")
        st.session_state[f"mot_{idx}"] = sc.get("motion", "Auto Alternate")
        st.session_state[f"broll_{idx}"] = sc.get("suggested_broll", "Establishing cinematic B-roll shot")
        st.session_state[f"sfx_sug_{idx}"] = sc.get("suggested_sfx", "None")
        st.session_state[f"fx_sc_{idx}"] = st.session_state.scene_fx_overrides.get(
            idx, sc.get("suggested_fx", "None")
        )


sync_scene_widget_states(st.session_state.scenes)


def _ascii_safe(text):
    """Keep printable ASCII characters so basic bitmap fonts never render missing-glyph boxes."""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in (text or ""))
    return " ".join(cleaned.split())


def _extract_visual_command(text):
    """Clean conversational filler words while keeping concrete subjects, gods, yugas, and planets."""
    raw = " ".join((text or "").replace("\n", " ").split())
    for phrase in ("where required", "as commanded", "as requested", "follow the character through"):
        raw = raw.replace(phrase, " ").replace(phrase.title(), " ")
    return " ".join(raw.split())


def parse_arrangement_directives(arrangement_text):
    """Translate user arrangement wording ('how it should be arranged so that it pops') into layout/motion/pop rules."""
    low = (arrangement_text or "").lower()
    layout_mode = "Spotlight Card"
    if any(k in low for k in ["collage", "grid", "split", "multi", "side by side", "montage"]):
        layout_mode = "Split Pop"
    elif any(k in low for k in ["full", "edge to edge", "bleed", "wide"]):
        layout_mode = "Full Pop"

    motion_mode = "Zoom In"
    if any(k in low for k in ["punch", "bounce", "beat", "fast", "energetic", "reel"]):
        motion_mode = "Zoom Punch"
    elif any(k in low for k in ["pan", "slide", "sweep", "scroll"]):
        motion_mode = "Pan Right"
    elif any(k in low for k in ["zoom out", "pull back", "reveal"]):
        motion_mode = "Zoom Out"

    pop_boost = any(k in low for k in ["pop", "vibrant", "bright", "glow", "contrast", "bold", "highlight", "stand out"])
    return {
        "layout_mode": layout_mode,
        "motion_mode": motion_mode,
        "pop_boost": pop_boost,
        "raw_note": " ".join((arrangement_text or "").split()),
    }


def compose_user_photos_into_frame(
    photo_paths,
    target_w,
    target_h,
    scene_idx=0,
    arrangement_cmd="",
    caption_text="",
    caption_lang="English",
    bg_mode="Blur Canvas",
):
    """Arrange one or more user-uploaded photos into a crisp, high-pop composition with optional wordings."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    valid_paths = [p for p in (photo_paths or []) if p and os.path.exists(p)]
    if not valid_paths:
        return build_local_bypass_scene(arrangement_cmd, target_w, target_h, scene_idx)

    directives = parse_arrangement_directives(arrangement_cmd)
    primary_path = valid_paths[scene_idx % len(valid_paths)]
    secondary_path = valid_paths[(scene_idx + 1) % len(valid_paths)] if len(valid_paths) > 1 else None

    with Image.open(primary_path) as p_im:
        p_rgb = p_im.convert("RGB")

    # Background blurred/tinted canvas so the subject pops immediately
    bg_scale = max(target_w / max(1, p_rgb.width), target_h / max(1, p_rgb.height))
    bg_w = max(target_w, int(round(p_rgb.width * bg_scale)))
    bg_h = max(target_h, int(round(p_rgb.height * bg_scale)))
    bg = p_rgb.resize((bg_w, bg_h), resampling)
    bx = max(0, (bg_w - target_w) // 2)
    by = max(0, (bg_h - target_h) // 2)
    bg = bg.crop((bx, by, bx + target_w, by + target_h)).filter(ImageFilter.GaussianBlur(radius=22))
    bg = ImageEnhance.Brightness(bg).enhance(0.55 if directives["pop_boost"] else 0.65)
    bg = ImageEnhance.Color(bg).enhance(1.25)

    canvas = bg.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    # Subtle radial glow behind focal subject so it pops
    glow_layer = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow_layer)
    gx0 = int(target_w * 0.14)
    gy0 = int(target_h * 0.14)
    gx1 = int(target_w * 0.86)
    gy1 = int(target_h * 0.68)
    g_draw.ellipse([gx0, gy0, gx1, gy1], fill=(255, 195, 85, 58 if directives["pop_boost"] else 32))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=36))
    canvas = Image.alpha_composite(canvas, glow_layer)
    draw = ImageDraw.Draw(canvas)

    def _fit_box(src_img, box_w, box_h):
        sc = min(box_w / max(1, src_img.width), box_h / max(1, src_img.height))
        nw = max(1, int(round(src_img.width * sc)))
        nh = max(1, int(round(src_img.height * sc)))
        return src_img.resize((nw, nh), resampling)

    if directives["layout_mode"] == "Split Pop" and secondary_path:
        with Image.open(secondary_path) as s_im:
            s_rgb = s_im.convert("RGB")
        half_h = int(target_h * 0.36)
        box_w = int(target_w * 0.86)
        top_fg = _fit_box(p_rgb, box_w, half_h)
        bot_fg = _fit_box(s_rgb, box_w, half_h)

        top_x = (target_w - top_fg.width) // 2
        top_y = int(target_h * 0.10)
        bot_x = (target_w - bot_fg.width) // 2
        bot_y = int(target_h * 0.49)

        draw.rounded_rectangle(
            [top_x - 6, top_y - 6, top_x + top_fg.width + 6, top_y + top_fg.height + 6],
            radius=16,
            fill=(255, 195, 75, 220),
        )
        canvas.paste(top_fg, (top_x, top_y))

        draw.rounded_rectangle(
            [bot_x - 6, bot_y - 6, bot_x + bot_fg.width + 6, bot_y + bot_fg.height + 6],
            radius=16,
            fill=(255, 46, 99, 210),
        )
        canvas.paste(bot_fg, (bot_x, bot_y))
    else:
        max_fw = int(target_w * (0.90 if directives["layout_mode"] == "Full Pop" else 0.84))
        max_fh = int(target_h * (0.66 if directives["layout_mode"] == "Full Pop" else 0.58))
        fg = _fit_box(p_rgb, max_fw, max_fh)
        fx = (target_w - fg.width) // 2
        fy = int(target_h * 0.14)

        draw.rounded_rectangle(
            [fx - 6, fy - 6, fx + fg.width + 6, fy + fg.height + 6],
            radius=18,
            fill=(255, 200, 80, 225 if directives["pop_boost"] else 160),
        )
        canvas.paste(fg, (fx, fy))

    # Optional secondary mini-badge if user uploaded 3+ photos
    if len(valid_paths) >= 3:
        third_path = valid_paths[(scene_idx + 2) % len(valid_paths)]
        with Image.open(third_path) as t_im:
            t_thumb = _fit_box(t_im.convert("RGB"), int(target_w * 0.22), int(target_h * 0.14))
        tx = target_w - t_thumb.width - int(target_w * 0.06)
        ty = int(target_h * 0.15)
        draw.rounded_rectangle(
            [tx - 4, ty - 4, tx + t_thumb.width + 4, ty + t_thumb.height + 4],
            radius=10,
            fill=(255, 255, 255, 230),
        )
        canvas.paste(t_thumb, (tx, ty))

    out_rgb = canvas.convert("RGB")
    if directives["pop_boost"]:
        out_rgb = ImageEnhance.Contrast(out_rgb).enhance(1.16)
        out_rgb = ImageEnhance.Color(out_rgb).enhance(1.24)

    if caption_text and caption_text.strip() and caption_lang != "None":
        out_rgb = draw_subtitle_on_frame(
            out_rgb,
            subtitle_text=caption_text.strip(),
            target_w=target_w,
            target_h=target_h,
            lang=caption_lang,
            position="Bottom",
            scale=1.08,
        )

    return ensure_exact_size(out_rgb, target_w, target_h, bg_mode=bg_mode)


def swap_scenes_and_media(idx_a, idx_b):
    """Reorder two scenes along with their associated images, videos, audio, and FX."""
    n_sc = len(st.session_state.scenes)
    if not (0 <= idx_a < n_sc and 0 <= idx_b < n_sc and idx_a != idx_b):
        return

    st.session_state.scenes[idx_a], st.session_state.scenes[idx_b] = (
        st.session_state.scenes[idx_b],
        st.session_state.scenes[idx_a],
    )

    for list_key in ("images", "scene_videos", "audio_files"):
        lst = list(st.session_state.get(list_key) or [])
        if idx_a < len(lst) and idx_b < len(lst):
            lst[idx_a], lst[idx_b] = lst[idx_b], lst[idx_a]
            st.session_state[list_key] = lst

    sfx_map = dict(st.session_state.scene_audio_fx)
    sfx_a = sfx_map.get(idx_a)
    sfx_b = sfx_map.get(idx_b)
    if sfx_b is not None:
        sfx_map[idx_a] = sfx_b
    else:
        sfx_map.pop(idx_a, None)
    if sfx_a is not None:
        sfx_map[idx_b] = sfx_a
    else:
        sfx_map.pop(idx_b, None)
    st.session_state.scene_audio_fx = sfx_map

    fx_map = dict(st.session_state.scene_fx_overrides)
    fx_a = fx_map.get(idx_a)
    fx_b = fx_map.get(idx_b)
    if fx_b is not None:
        fx_map[idx_a] = fx_b
    else:
        fx_map.pop(idx_a, None)
    if fx_a is not None:
        fx_map[idx_b] = fx_a
    else:
        fx_map.pop(idx_b, None)
    st.session_state.scene_fx_overrides = fx_map

    sync_scene_widget_states(st.session_state.scenes)


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


def subclip_safe(clip, start_t, end_t):
    """Trim/split a MoviePy clip across v1 and v2 APIs."""
    s_val = max(0.0, float(start_t))
    e_val = max(s_val + 0.1, float(end_t))
    if clip.duration is not None:
        e_val = min(float(clip.duration), e_val)
        s_val = min(s_val, max(0.0, e_val - 0.1))
    if MOVIEPY_V2 and hasattr(clip, "subclipped"):
        return clip.subclipped(s_val, e_val)
    return clip.subclip(s_val, e_val)


def scale_audio_volume(clip, factor):
    if clip is None or abs(factor - 1.0) < 1e-3:
        return clip
    if MOVIEPY_V2 and hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(factor)
    if hasattr(clip, "volumex"):
        return clip.volumex(factor)
    return clip


def get_video_duration_safe(video_path):
    """Return video duration in seconds without leaking handles."""
    if not video_path or not os.path.exists(video_path):
        return 0.0
    vc = None
    try:
        vc = VideoFileClip(video_path)
        return float(vc.duration or 0.0)
    except Exception:
        return 0.0
    finally:
        if vc is not None:
            try:
                vc.close()
            except Exception:
                pass


def trim_video_clip_file(src_path, out_path, start_sec, end_sec):
    """Trim [start_sec, end_sec] from a video clip and save to out_path."""
    src = None
    seg = None
    try:
        src = VideoFileClip(src_path)
        seg = subclip_safe(src, start_sec, end_sec)
        seg.write_videofile(
            out_path,
            fps=24,
            codec="libx264",
            audio_codec="aac" if src.audio is not None else None,
            audio=src.audio is not None,
            logger=None,
        )
        return out_path
    finally:
        for obj in (seg, src):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


def split_video_clip_file(src_path, split_sec, part_a_path, part_b_path):
    """Split a video clip at split_sec into two separate MP4 files."""
    src = None
    part_a = None
    part_b = None
    try:
        src = VideoFileClip(src_path)
        dur = float(src.duration or 0.0)
        cut = max(0.2, min(dur - 0.2, float(split_sec)))
        if dur <= 0.5:
            raise ValueError("Clip is too short to split.")

        part_a = subclip_safe(src, 0.0, cut)
        part_a.write_videofile(
            part_a_path,
            fps=24,
            codec="libx264",
            audio_codec="aac" if src.audio is not None else None,
            audio=src.audio is not None,
            logger=None,
        )

        part_b = subclip_safe(src, cut, dur)
        part_b.write_videofile(
            part_b_path,
            fps=24,
            codec="libx264",
            audio_codec="aac" if src.audio is not None else None,
            audio=src.audio is not None,
            logger=None,
        )
        return part_a_path, part_b_path
    finally:
        for obj in (part_a, part_b, src):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


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
    if not fx_url or fx_name == "None":
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
    """AI/ML-style director for B-roll shot, motion, VFX, and SFX."""
    low = (text or "").lower()
    if any(w in low for w in ["funny", "joke", "oops", "slip", "laugh", "comedy", "silly", "meme"]):
        return (
            "Quick cut comedy B-roll reaction & expressive close-up with lively framing",
            "Zoom Punch",
            "Zoom Punch",
            "😂 Funny Laugh",
        )
    if any(w in low for w in ["rain", "storm", "thunder", "lightning"]):
        return (
            "Moody weather B-roll of falling rain, wet surfaces, and dramatic overcast light",
            "Pan Left",
            "Cinematic Vignette",
            "🌿 Rain & Thunder",
        )
    if any(w in low for w in ["ocean", "sea", "wave", "beach", "water", "river"]):
        return (
            "Sweeping coastal/water B-roll with gentle movement and natural horizon reflections",
            "Pan Right",
            "None",
            "🌊 Ocean Waves",
        )
    if any(w in low for w in ["fire", "flame", "camp", "warm", "torch"]):
        return (
            "Warm ember & light-flicker B-roll creating intimacy and depth",
            "Zoom In",
            "Cinematic Vignette",
            "🔥 Campfire Crackle",
        )
    if any(w in low for w in ["forest", "jungle", "tree", "bird", "nature", "morning"]):
        return (
            "Nature B-roll through foliage and sunlight shafts with calm pacing",
            "Pan Right",
            "None",
            "🌲 Forest Birds",
        )
    if any(w in low for w in ["shock", "twist", "sudden", "boom", "dramatic", "wow"]):
        return (
            "High-tension reveal B-roll with tight framing and punchy pacing",
            "Zoom Punch",
            "Shake",
            "💥 Vine Boom (Meme)",
        )
    motions = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    broll_shots = [
        "Establishing cinematic B-roll with glowing sun, planets, and ancient yuga horizon",
        "Medium character & B-roll cut showing traditional attire, ornaments, and sacred setting",
        "Cosmic & environmental B-roll highlighting depth, light rays, and mood",
        "Closing cinematic B-roll framing characters and ancient backdrop cleanly",
    ]
    return broll_shots[idx % len(broll_shots)], motions[idx % len(motions)], "None", "None"


def ai_suggest_broll_and_fx_for_scenes(gemini_key, scenes):
    """Generate scene-by-scene AI B-roll, motion, VFX, and SFX suggestions."""
    updated = []
    for idx, sc in enumerate(scenes):
        b_sug, m_sug, fx_sug, sfx_sug = infer_ml_scene_direction(
            f"{sc.get('narration', '')} {sc.get('visual', '')} {sc.get('emotion', '')}", idx
        )
        copy_sc = dict(sc)
        copy_sc["suggested_broll"] = sc.get("suggested_broll") or b_sug
        copy_sc["motion"] = sc.get("motion") or m_sug
        copy_sc["suggested_fx"] = sc.get("suggested_fx") or fx_sug
        copy_sc["suggested_sfx"] = sc.get("suggested_sfx") or sfx_sug
        updated.append(copy_sc)

    if gemini_key and gemini_key.strip() and updated:
        client = genai.Client(api_key=gemini_key.strip())
        compact_input = [
            {
                "scene": i + 1,
                "narration": s.get("narration", ""),
                "emotion": s.get("emotion", ""),
                "visual": s.get("visual", ""),
            }
            for i, s in enumerate(updated)
        ]
        prompt = (
            "You are a film & reels director. For each scene in the JSON input, suggest:\n"
            "- 'broll': concise 1-line B-roll shot idea\n"
            "- 'motion': one of ['Auto Alternate', 'Zoom In', 'Zoom Out', 'Pan Right', 'Pan Left']\n"
            "- 'fx': one of ['None', 'Zoom Punch', 'Shake', 'Flash White', 'Glitch RGB', 'Vintage Sepia', 'Cinematic Vignette']\n"
            "Return ONLY a valid JSON array matching the scene count, no markdown.\n"
            + json.dumps(compact_input, ensure_ascii=False)
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
                        ln for ln in raw.splitlines() if not ln.strip().startswith("```")
                    ).strip()
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for i, item in enumerate(parsed[: len(updated)]):
                        if isinstance(item, dict):
                            if item.get("broll"):
                                updated[i]["suggested_broll"] = str(item["broll"]).strip()
                            if item.get("motion") in MOTION_CHOICES:
                                updated[i]["motion"] = item["motion"]
                            if item.get("fx") in VIDEO_FX_CHOICES:
                                updated[i]["suggested_fx"] = item["fx"]
                    break
            except Exception:
                continue

    return updated


def build_fallback_scenes_from_rough_story(rough_text, count, char_prompt):
    """Build up to 10 scenes directly from story, character, god, yuga, and celestial elements."""
    target_count = max(1, min(10, int(count)))
    clean_story = _extract_visual_command(rough_text)
    clean_char = _extract_visual_command(char_prompt)
    sentences = [
        s.strip()
        for s in clean_story.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    beat_moods = [
        ("Opening & Celestial Setting", "wide view with glowing sun, planets, and ancient yuga sky"),
        ("Mythological Presence", "character in traditional attire and ornaments amidst ancient yuga setting"),
        ("Divine & Cosmic Shift", "dramatic light rays, sun, and planets across ancient sky"),
        ("Action & Journey", "dynamic movement through ancient yuga landscape and sacred environment"),
        ("Emotional Intensity", "expressive scene with rich golden lighting and clear character details"),
        ("Sacred Environment", "detailed ancient architecture, nature, and celestial sky"),
        ("Rising Drama", "bold framing with glowing sun, planets, and active scene"),
        ("Quiet Reflection", "serene atmospheric framing in ancient yuga twilight"),
        ("Turning Point", "striking visual contrast between characters, ancient setting, and planets"),
        ("Closing Vision", "grand closing shot with characters, ancient yuga backdrop, sun, and planets"),
    ]
    scenes = []
    combined_ctx = ", ".join(part for part in [clean_char, clean_story] if part)
    for idx in range(target_count):
        base_line = sentences[idx % len(sentences)] if sentences else (clean_story or "Ancient yuga scene")
        emo_label, beat_desc = beat_moods[idx % len(beat_moods)]
        broll_sug, motion_sug, fx_sug, sfx_sug = infer_ml_scene_direction(
            f"{base_line} {combined_ctx}", idx
        )
        scenes.append({
            "narration": f"{base_line}.",
            "kannada_narration": "",
            "emotion": emo_label,
            "motion": motion_sug,
            "suggested_broll": broll_sug,
            "suggested_fx": fx_sug,
            "suggested_sfx": sfx_sug,
            "visual": (
                f"{beat_desc}. Visual elements: {combined_ctx}. "
                f"Scene focus: {base_line}."
            ).strip(),
        })
    return scenes


def sync_scenes_to_count(existing_scenes, desired_count, rough_text, char_prompt):
    """Ensure st.session_state.scenes has the exact desired_count (up to 10 scenes)."""
    target_n = max(1, min(10, int(desired_count)))
    base_story = (
        rough_text.strip()
        if rough_text and rough_text.strip()
        else "Ancient yuga story with mythological characters, glowing sun, and planets"
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


def draw_subtitle_on_frame(
    frame_pil,
    subtitle_text,
    target_w,
    target_h,
    lang="English",
    position="Bottom",
    scale=1.0,
):
    """Draw a readable subtitle/caption banner at Bottom, Center, or Top with adjustable scale."""
    if not subtitle_text or not subtitle_text.strip() or lang == "None":
        return frame_pil

    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    safe_scale = max(0.65, min(1.6, float(scale or 1.0)))
    font_size = max(22, int(round(target_w * 0.036 * safe_scale)))
    font, _ = _find_font(font_size, lang=lang)
    lang_code = "kn" if lang == "Kannada" else "en"

    chars_per_line = max(12, int(target_w / max(10, font_size * 0.62)))
    wrapped_lines = textwrap.wrap(subtitle_text.strip(), width=chars_per_line)[:3]
    if not wrapped_lines:
        return frame_pil

    line_h = font_size + 14
    box_h = len(wrapped_lines) * line_h + 24
    pad_x = max(18, int(target_w * 0.04))

    if position == "Top":
        box_top = max(18, int(target_h * 0.06))
    elif position == "Center":
        box_top = max(18, (target_h - box_h) // 2)
    else:
        box_top = max(18, target_h - box_h - max(42, int(target_h * 0.07)))

    box_bot = min(target_h - 14, box_top + box_h)
    box_top = max(14, box_bot - box_h)

    draw.rounded_rectangle(
        [pad_x, box_top, target_w - pad_x, box_bot],
        radius=14,
        fill=(10, 14, 22, 218),
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
            (text_x + 2, cur_y + 2, (0, 0, 0, 245)),
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
    subtitle_pos="Bottom",
    subtitle_scale=1.0,
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
                position=subtitle_pos,
                scale=subtitle_scale,
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
    subtitle_pos="Bottom",
    subtitle_scale=1.0,
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
        subtitle_pos=subtitle_pos,
        subtitle_scale=subtitle_scale,
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


def build_uploaded_video_clip(
    vid_path,
    target_w,
    target_h,
    clip_dur,
    scene_idx=0,
    subtitle_text="",
    subtitle_lang="None",
    subtitle_pos="Bottom",
    subtitle_scale=1.0,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Process an existing video clip so canvas size, color/FX, and subtitles stay consistent."""
    src = None
    try:
        src = VideoFileClip(vid_path)
        native_dur = max(0.5, float(src.duration or clip_dur))
        if native_dur < clip_dur and hasattr(src, "loop"):
            timed = apply_duration(src.loop(duration=clip_dur), clip_dur)
        else:
            timed = subclip_safe(src, 0.0, min(native_dur, clip_dur))
            timed = apply_duration(timed, clip_dur)

        needs_visual_pass = (
            abs(brightness - 1.0) > 1e-3
            or abs(contrast - 1.0) > 1e-3
            or abs(saturation - 1.0) > 1e-3
            or (blur_radius and blur_radius > 0.05)
            or (video_fx and video_fx != "None")
            or (subtitle_lang != "None" and bool(subtitle_text))
            or (int(getattr(src, "w", target_w)) != target_w)
            or (int(getattr(src, "h", target_h)) != target_h)
        )

        if not needs_visual_pass:
            return timed, [src]

        os.makedirs("video_passes", exist_ok=True)
        step_dt = 0.25
        n_steps = max(4, min(40, int(math.ceil(clip_dur / step_dt))))
        actual_step = clip_dur / n_steps
        step_clips = []

        for idx_f in range(n_steps):
            sample_t = min(max(0.0, (idx_f + 0.5) * actual_step), max(0.0, float(timed.duration or clip_dur) - 0.02))
            frame_arr = timed.get_frame(sample_t)
            pil_f = Image.fromarray(np.asarray(frame_arr, dtype=np.uint8))
            pil_f = ensure_exact_size(pil_f, target_w, target_h, bg_mode=bg_mode)
            pil_f = apply_pil_effects(
                pil_f,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                blur_radius=blur_radius,
                video_fx=video_fx,
            )
            if subtitle_lang != "None" and subtitle_text:
                pil_f = draw_subtitle_on_frame(
                    pil_f,
                    subtitle_text=subtitle_text,
                    target_w=target_w,
                    target_h=target_h,
                    lang=subtitle_lang,
                    position=subtitle_pos,
                    scale=subtitle_scale,
                )
            p_f = os.path.join("video_passes", f"vp_{scene_idx}_{idx_f}.jpg")
            pil_f.save(p_f, format="JPEG", quality=90)
            step_clips.append(apply_duration(ImageClip(p_f), actual_step))

        merged = concatenate_videoclips(step_clips, method="chain")
        if timed.audio is not None:
            merged = apply_audio(merged, timed.audio)
        return merged, [src, timed]
    except Exception:
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        raise


def build_ducked_bgm_track(bgm_path, total_duration, voice_intervals, normal_vol=0.22, duck_vol=0.08):
    """Build a BGM clip across total_duration that automatically ducks during voiceover intervals."""
    raw = AudioFileClip(bgm_path)
    dur = max(0.5, float(total_duration))
    if (raw.duration or 0.0) < dur and hasattr(raw, "loop"):
        base_bgm = apply_duration(raw.loop(duration=dur), dur)
    else:
        base_bgm = apply_duration(raw, dur)

    if not voice_intervals:
        return scale_audio_volume(base_bgm, normal_vol), [raw, base_bgm]

    sorted_iv = sorted(
        (max(0.0, float(s)), min(dur, float(e)))
        for s, e in voice_intervals
        if float(e) > float(s)
    )
    merged_iv = []
    for s, e in sorted_iv:
        if not merged_iv or s > merged_iv[-1][1] + 0.05:
            merged_iv.append([s, e])
        else:
            merged_iv[-1][1] = max(merged_iv[-1][1], e)

    segments = []
    handles = [raw, base_bgm]
    cursor = 0.0
    for s, e in merged_iv:
        if s > cursor + 0.02:
            seg_n = subclip_safe(base_bgm, cursor, s)
            handles.append(seg_n)
            segments.append(scale_audio_volume(seg_n, normal_vol))
        if e > s + 0.02:
            seg_d = subclip_safe(base_bgm, s, e)
            handles.append(seg_d)
            segments.append(scale_audio_volume(seg_d, duck_vol))
        cursor = max(cursor, e)

    if cursor < dur - 0.02:
        seg_tail = subclip_safe(base_bgm, cursor, dur)
        handles.append(seg_tail)
        segments.append(scale_audio_volume(seg_tail, normal_vol))

    if not segments:
        return scale_audio_volume(base_bgm, duck_vol), handles

    return concatenate_videoclips(segments, method="chain"), handles


def build_local_bypass_scene(prompt_text, target_w, target_h, scene_idx, emotion_label=""):
    """Clean storyboard frame with sun/planet glow and ASCII-safe labels when APIs fail."""
    p_low = (prompt_text or "").lower()
    if any(k in p_low for k in ["sun", "planet", "yuga", "mahabharata", "ramayana", "god", "cosmic", "divine"]):
        palettes = [
            ((24, 28, 62), (168, 96, 44), (16, 18, 34)),
            ((18, 36, 68), (210, 134, 52), (14, 20, 38)),
            ((38, 22, 58), (186, 86, 64), (18, 14, 30)),
            ((20, 44, 62), (196, 142, 62), (12, 22, 34)),
        ]
    elif any(k in p_low for k in ["forest", "jungle", "tree", "woods", "nature", "mountain", "valley", "river"]):
        palettes = [
            ((18, 48, 38), (44, 96, 68), (12, 28, 22)),
            ((24, 58, 52), (58, 114, 92), (14, 32, 28)),
            ((34, 48, 38), (108, 92, 58), (18, 26, 18)),
            ((16, 42, 56), (46, 98, 94), (10, 24, 32)),
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

    sun_cx = target_w * (0.26 + 0.16 * (scene_idx % 4))
    sun_cy = target_h * (0.22 + 0.04 * (scene_idx % 3))
    dist_sq = (x_grid[:, :, 0] - sun_cx) ** 2 + (y_grid[:, :, 0] - sun_cy) ** 2
    glow_r = min(target_w, target_h) * 0.25
    glow = np.exp(-dist_sq / (2.0 * glow_r ** 2))[:, :, None]
    full_bg = np.clip(full_bg + glow * np.array([78.0, 58.0, 34.0], dtype=np.float32), 0, 255)

    img = Image.fromarray(full_bg.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    sun_r = max(18, int(min(target_w, target_h) * 0.045))
    draw.ellipse(
        [sun_cx - sun_r, sun_cy - sun_r, sun_cx + sun_r, sun_cy + sun_r],
        fill=(255, 210, 110, 210),
    )
    p1_x = int(target_w * 0.78)
    p1_y = int(target_h * 0.19)
    p1_r = max(10, int(min(target_w, target_h) * 0.022))
    draw.ellipse([p1_x - p1_r, p1_y - p1_r, p1_x + p1_r, p1_y + p1_r], fill=(170, 195, 245, 190))

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

    draw.polygon(
        [
            (cx - int(head_r * 0.85), head_cy - int(head_r * 0.75)),
            (cx - int(head_r * 0.45), head_cy - int(head_r * 1.35)),
            (cx, head_cy - int(head_r * 0.95)),
            (cx + int(head_r * 0.45), head_cy - int(head_r * 1.35)),
            (cx + int(head_r * 0.85), head_cy - int(head_r * 0.75)),
        ],
        fill=(245, 195, 75, 240),
    )

    draw.rounded_rectangle(
        [cx - int(head_r * 1.9), head_cy + head_r + 6, cx + int(head_r * 1.9), head_cy + head_r + int(head_r * 3.2)],
        radius=28,
        fill=(28, 36, 52, 230),
        outline=(235, 195, 115, 140),
        width=2,
    )
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=(214, 170, 140, 250))

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
    emo_badge = f" • {_ascii_safe(emotion_label).upper()}" if _ascii_safe(emotion_label) else ""
    draw.text(
        (cx0 + 18, cy0 + 12),
        f"SCENE {scene_idx + 1}{emo_badge}",
        font=title_font,
        fill=(255, 118, 154, 255),
    )

    clean_action = _ascii_safe(prompt_text)
    wrapped = textwrap.wrap(clean_action, width=max(26, int(card_w / max(10, int(target_w * 0.021) * 0.56))))[:3]
    cur_y = cy0 + 44
    for line in wrapped:
        draw.text((cx0 + 18, cur_y), line, font=body_font, fill=(242, 246, 252, 245))
        cur_y += int(target_w * 0.025)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    return ensure_exact_size(combined, target_w, target_h, bg_mode="Crop Fill")


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
    bounded_prompt = f"{_extract_visual_command(prompt_text)}, {aspect_hint}"[:2048]
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
    short_prompt = _extract_visual_command(prompt_text)[:280]
    encoded_prompt = urllib.parse.quote(short_prompt, safe="")
    endpoints = [
        f"https://gen.pollinations.ai/image/{encoded_prompt}?model=flux&width={w_p}&height={h_p}&seed={seed}",
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={w_p}&height={h_p}&seed={seed}&nologo=true",
    ]
    last_err = None
    for url in endpoints:
        try:
            resp = requests.get(url, timeout=45)
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
    ref_photo_paths=None,
    arrangement_cmd="",
    caption_text="",
    caption_lang="English",
    allow_local_bypass=True,
    emotion_label="",
    seed_offset=0,
    bg_mode="Blur Canvas",
):
    """Generate scene frames from AI engines when configured, or from user photos + arrangement/wordings."""
    errors = []
    scene_seed = 100 + scene_idx * 37 + seed_offset
    valid_photos = [p for p in (ref_photo_paths or []) if p and os.path.exists(p)]

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
                    prompt=_extract_visual_command(prompt_text),
                    model=model_id,
                    width=w_hf,
                    height=h_hf,
                )
                return ensure_exact_size(img.convert("RGB"), target_w, target_h, bg_mode=bg_mode), f"HF {model_id}"
            except Exception as exc:
                last_hf_err = exc
        raise RuntimeError(str(last_hf_err))

    def try_gemini():
        if not gemini_key.strip():
            raise RuntimeError("Missing Gemini API Key.")
        client = genai.Client(api_key=gemini_key.strip())
        clean_p = _extract_visual_command(prompt_text)
        contents = [clean_p]
        if valid_photos:
            with Image.open(valid_photos[scene_idx % len(valid_photos)]) as ref_im:
                contents = [ref_im.convert("RGB"), clean_p]
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
                            return ensure_exact_size(img, target_w, target_h, bg_mode=bg_mode), f"Gemini {model_id}"
                last_gem_err = RuntimeError(f"{model_id}: no image returned")
            except Exception as exc:
                last_gem_err = exc
        raise RuntimeError(str(last_gem_err))

    def try_pollinations():
        img = generate_pollinations_image(prompt_text, target_w, target_h, seed=200 + scene_seed)
        return img, "Pollinations AI Fallback"

    has_any_key = bool(
        (cf_account_id and cf_account_id.strip() and cf_api_token and cf_api_token.strip())
        or (hf_token and hf_token.strip())
        or (gemini_key and gemini_key.strip())
    )

    if has_any_key:
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
                if caption_text and caption_text.strip() and caption_lang != "None":
                    img = draw_subtitle_on_frame(
                        img,
                        subtitle_text=caption_text.strip(),
                        target_w=target_w,
                        target_h=target_h,
                        lang=caption_lang,
                        position="Bottom",
                        scale=1.08,
                    )
                return img, src_name, errors
            except Exception as exc:
                errors.append(f"{label}: {exc}")

    if valid_photos:
        composed = compose_user_photos_into_frame(
            photo_paths=valid_photos,
            target_w=target_w,
            target_h=target_h,
            scene_idx=scene_idx,
            arrangement_cmd=arrangement_cmd or prompt_text,
            caption_text=caption_text,
            caption_lang=caption_lang,
            bg_mode=bg_mode,
        )
        return composed, "Uploaded Photos Arrangement", errors

    if allow_local_bypass:
        base_sc = build_local_bypass_scene(
            arrangement_cmd or prompt_text, target_w, target_h, scene_idx, emotion_label=emotion_label
        )
        if caption_text and caption_text.strip() and caption_lang != "None":
            base_sc = draw_subtitle_on_frame(
                base_sc,
                subtitle_text=caption_text.strip(),
                target_w=target_w,
                target_h=target_h,
                lang=caption_lang,
                position="Bottom",
                scale=1.08,
            )
        return base_sc, "Local Storyboard Bypass", errors

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


def export_project_json():
    """Serialize studio state into a portable JSON project payload."""
    payload = {
        "version": "2.3-pro",
        "format_preset": st.session_state.format_preset,
        "bg_fit_mode": st.session_state.bg_fit_mode,
        "playback_speed": st.session_state.playback_speed,
        "custom_title_text": st.session_state.custom_title_text,
        "custom_title_lang": st.session_state.custom_title_lang,
        "subtitle_position": st.session_state.subtitle_position,
        "subtitle_scale": st.session_state.subtitle_scale,
        "editor_brightness": st.session_state.editor_brightness,
        "editor_contrast": st.session_state.editor_contrast,
        "editor_saturation": st.session_state.editor_saturation,
        "editor_blur": st.session_state.editor_blur,
        "editor_transition": st.session_state.editor_transition,
        "editor_transition_sec": st.session_state.editor_transition_sec,
        "editor_video_fx": st.session_state.editor_video_fx,
        "arrangement_command": st.session_state.arrangement_command,
        "reel_wordings_input": st.session_state.reel_wordings_input,
        "bgm_volume": st.session_state.bgm_volume,
        "bgm_duck_enabled": st.session_state.bgm_duck_enabled,
        "bgm_duck_volume": st.session_state.bgm_duck_volume,
        "voice_volume": st.session_state.voice_volume,
        "sfx_volume": st.session_state.sfx_volume,
        "locked_character_prompt": st.session_state.locked_character_prompt,
        "exact_character_count": st.session_state.exact_character_count,
        "lock_character_enabled": st.session_state.lock_character_enabled,
        "scenes": st.session_state.scenes,
        "scene_fx_overrides": {str(k): v for k, v in st.session_state.scene_fx_overrides.items()},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_project_json(raw_json_str):
    """Restore studio settings and scenes from a JSON project file."""
    data = json.loads(raw_json_str)
    if not isinstance(data, dict):
        raise ValueError("Invalid project JSON structure.")

    if data.get("format_preset") in FORMAT_PRESETS:
        st.session_state.format_preset = data["format_preset"]
    if data.get("bg_fit_mode") in ("Blur Canvas", "Crop Fill"):
        st.session_state.bg_fit_mode = data["bg_fit_mode"]
    if "playback_speed" in data:
        st.session_state.playback_speed = float(data["playback_speed"])
    if "custom_title_text" in data:
        st.session_state.custom_title_text = str(data["custom_title_text"])
    if data.get("custom_title_lang") in ("English", "Kannada"):
        st.session_state.custom_title_lang = data["custom_title_lang"]
    if data.get("subtitle_position") in SUBTITLE_POSITIONS:
        st.session_state.subtitle_position = data["subtitle_position"]
    if "subtitle_scale" in data:
        st.session_state.subtitle_scale = float(data["subtitle_scale"])
    for k in (
        "editor_brightness",
        "editor_contrast",
        "editor_saturation",
        "editor_blur",
        "editor_transition_sec",
        "bgm_volume",
        "bgm_duck_volume",
        "voice_volume",
        "sfx_volume",
    ):
        if k in data:
            st.session_state[k] = float(data[k])
    if "bgm_duck_enabled" in data:
        st.session_state.bgm_duck_enabled = bool(data["bgm_duck_enabled"])
    if data.get("editor_transition") in ("None", "Crossfade", "Fade", "Slide Left", "Slide Right"):
        st.session_state.editor_transition = data["editor_transition"]
    if data.get("editor_video_fx") in VIDEO_FX_CHOICES:
        st.session_state.editor_video_fx = data["editor_video_fx"]
    if "arrangement_command" in data:
        st.session_state.arrangement_command = str(data["arrangement_command"])
    if "reel_wordings_input" in data:
        st.session_state.reel_wordings_input = str(data["reel_wordings_input"])
    if "locked_character_prompt" in data:
        st.session_state.locked_character_prompt = str(data["locked_character_prompt"])
    if "exact_character_count" in data:
        st.session_state.exact_character_count = int(data["exact_character_count"])
    if "lock_character_enabled" in data:
        st.session_state.lock_character_enabled = bool(data["lock_character_enabled"])

    if isinstance(data.get("scene_fx_overrides"), dict):
        restored_fx = {}
        for k, v in data["scene_fx_overrides"].items():
            try:
                restored_fx[int(k)] = str(v)
            except Exception:
                pass
        st.session_state.scene_fx_overrides = restored_fx

    if isinstance(data.get("scenes"), list) and len(data["scenes"]) > 0:
        st.session_state.scenes = data["scenes"][:10]
        sync_scene_widget_states(st.session_state.scenes)


# --- TOP STUDIO BANNER ---
st.markdown(
    """
    <div class="studio-header">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="studio-pill">AI STUDIO PRO</span>
            <span style="font-weight:700;font-size:17px;">Movie, Shorts & Reels Creation Suite</span>
        </div>
        <div class="studio-sub">Unlimited Photos & Videos • Custom Arrangement & Wordings • Trim, Split & Reorder</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR: FORMAT, AI KEYS, CHARACTER COMMANDS & PROJECT SAVE/LOAD ---
with st.sidebar:
    st.header("🎬 Format & AI Setup")
    st.session_state.format_preset = st.selectbox(
        "Video Format Preset",
        list(FORMAT_PRESETS.keys()),
        index=list(FORMAT_PRESETS.keys()).index(st.session_state.format_preset),
    )
    width, height, ratio_str = FORMAT_PRESETS[st.session_state.format_preset]
    st.caption(f"Export Resolution: **{width} × {height}** (`{ratio_str}`)")

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
        height=105,
    )

    st.divider()
    st.subheader("💾 Project Save & Load")
    st.download_button(
        label="⬇️ Save Project (.json)",
        data=export_project_json(),
        file_name="ai_studio_project.json",
        mime="application/json",
        use_container_width=True,
    )
    proj_up = st.file_uploader("Load Project (.json)", type=["json"], key="proj_json_loader")
    if proj_up is not None and st.button("📂 Restore Project", key="btn_restore_proj", use_container_width=True):
        try:
            import_project_json(proj_up.getvalue().decode("utf-8"))
            st.success("Project settings and scenes restored!")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load project: {exc}")

# --- COMPACT LIVE PREVIEW STAGE, STUDIO CONTROLS & CLIP TRIM/SPLIT ---
with st.expander("🖥️ Compact Preview Stage, Studio Controls & Clip Trim/Split", expanded=True):
    stage_left, stage_right = st.columns([1.0, 1.85])

    with stage_left:
        avail_imgs = [p for p in st.session_state.images if os.path.exists(p)]
        if not avail_imgs:
            avail_imgs = [p for p in st.session_state.uploaded_photo_paths if os.path.exists(p)]
        avail_vids = [p for p in st.session_state.scene_videos if os.path.exists(p)]

        monitor_mode = st.radio(
            "Preview Mode",
            ["Live Frame + Effects", "Rendered Clip"],
            horizontal=True,
            key="top_monitor_mode",
        )

        preview_caption_txt = ""
        if st.session_state.custom_title_text.strip():
            preview_caption_txt = st.session_state.custom_title_text.strip()
        elif st.session_state.scenes:
            sc_cur_idx = min(int(st.session_state.selected_clip_idx), len(st.session_state.scenes) - 1)
            if st.session_state.custom_title_lang == "Kannada":
                preview_caption_txt = (st.session_state.scenes[sc_cur_idx].get("kannada_narration") or "").strip()
            else:
                preview_caption_txt = (st.session_state.scenes[sc_cur_idx].get("narration") or "").strip()

        st.markdown('<div class="compact-preview-box">', unsafe_allow_html=True)
        if monitor_mode == "Live Frame + Effects" and avail_imgs:
            c_idx = min(int(st.session_state.selected_clip_idx), len(avail_imgs) - 1)
            with Image.open(avail_imgs[c_idx]) as im_raw:
                prev_img = ensure_exact_size(im_raw, width, height, bg_mode=st.session_state.bg_fit_mode)
            scene_fx_choice = st.session_state.scene_fx_overrides.get(c_idx, st.session_state.editor_video_fx)
            prev_img = apply_pil_effects(
                prev_img,
                brightness=st.session_state.editor_brightness,
                contrast=st.session_state.editor_contrast,
                saturation=st.session_state.editor_saturation,
                blur_radius=st.session_state.editor_blur,
                video_fx=scene_fx_choice,
            )
            if st.session_state.custom_title_text.strip():
                prev_img = draw_subtitle_on_frame(
                    prev_img,
                    subtitle_text=preview_caption_txt,
                    target_w=width,
                    target_h=height,
                    lang=st.session_state.custom_title_lang,
                    position=st.session_state.subtitle_position,
                    scale=st.session_state.subtitle_scale,
                )
            st.image(
                prev_img,
                caption=f"Scene {c_idx + 1} • {width}x{height} • {scene_fx_choice}",
                width=270,
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
                    subtitle_text=preview_caption_txt,
                    target_w=width,
                    target_h=height,
                    lang=st.session_state.custom_title_lang,
                    position=st.session_state.subtitle_position,
                    scale=st.session_state.subtitle_scale,
                )
            st.image(
                fallback_img,
                caption=f"Preview ({width}x{height})",
                width=270,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with stage_right:
        ctrl_tabs = st.tabs([
            "🎨 Looks & FX",
            "✂️ Trim & Split",
            "🔊 Sound FX",
            "✍️ Text & Fit",
            "📁 Media",
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
                    VIDEO_FX_CHOICES,
                    index=VIDEO_FX_CHOICES.index(st.session_state.editor_video_fx),
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
            editable_vids = [p for p in st.session_state.scene_videos if os.path.exists(p)]
            if not editable_vids:
                st.info("Generate scene/reel videos in Tab 2 or upload video clips in 'Media' to trim or split.")
            else:
                clip_labels = [f"Clip {i + 1}: {os.path.basename(p)}" for i, p in enumerate(editable_vids)]
                target_v_idx = st.selectbox(
                    "Select Video Clip to Trim or Split",
                    options=list(range(len(editable_vids))),
                    format_func=lambda x: clip_labels[x],
                    index=min(int(st.session_state.selected_clip_idx), len(editable_vids) - 1),
                    key="trim_split_clip_selector",
                )
                target_v_path = editable_vids[target_v_idx]
                v_dur = max(0.5, round(get_video_duration_safe(target_v_path), 2))
                st.caption(f"Clip Duration: **{v_dur:.2f}s**")

                trim_c1, trim_c2 = st.columns(2)
                with trim_c1:
                    trim_start = st.number_input(
                        "Trim Start (sec)",
                        min_value=0.0,
                        max_value=max(0.1, v_dur - 0.2),
                        value=0.0,
                        step=0.1,
                        key="trim_start_val",
                    )
                with trim_c2:
                    trim_end = st.number_input(
                        "Trim End (sec)",
                        min_value=min(v_dur, trim_start + 0.2),
                        max_value=v_dur,
                        value=v_dur,
                        step=0.1,
                        key="trim_end_val",
                    )

                act_c1, act_c2 = st.columns(2)
                with act_c1:
                    if st.button("✂️ Trim Clip", key="btn_apply_trim", use_container_width=True):
                        if trim_end <= trim_start + 0.15:
                            st.error("Trim End must be at least 0.2s after Trim Start.")
                        else:
                            os.makedirs("edited_clips", exist_ok=True)
                            trimmed_out = os.path.join(
                                "edited_clips", f"trimmed_{target_v_idx}_{int(time.time())}.mp4"
                            )
                            with st.spinner("Trimming video clip..."):
                                trim_video_clip_file(target_v_path, trimmed_out, trim_start, trim_end)
                            st.session_state.scene_videos[target_v_idx] = trimmed_out
                            st.success(f"Trimmed Clip {target_v_idx + 1} to [{trim_start:.1f}s – {trim_end:.1f}s].")
                            st.rerun()

                with act_c2:
                    default_split = round(v_dur / 2.0, 2)
                    split_at = st.number_input(
                        "Split Point (sec)",
                        min_value=0.2,
                        max_value=max(0.3, v_dur - 0.2),
                        value=max(0.2, min(max(0.3, v_dur - 0.2), default_split)),
                        step=0.1,
                        key="split_point_val",
                    )
                    if st.button("🔀 Split Clip", key="btn_apply_split", use_container_width=True):
                        os.makedirs("edited_clips", exist_ok=True)
                        ts_now = int(time.time())
                        pa_path = os.path.join("edited_clips", f"split_{target_v_idx}_A_{ts_now}.mp4")
                        pb_path = os.path.join("edited_clips", f"split_{target_v_idx}_B_{ts_now}.mp4")
                        with st.spinner("Splitting clip into two parts..."):
                            split_video_clip_file(target_v_path, split_at, pa_path, pb_path)
                        new_list = list(st.session_state.scene_videos)
                        new_list[target_v_idx : target_v_idx + 1] = [pa_path, pb_path]
                        st.session_state.scene_videos = new_list
                        st.success(f"Split Clip {target_v_idx + 1} at {split_at:.1f}s into two clips!")
                        st.rerun()

        with ctrl_tabs[2]:
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
            if st.button("➕ Attach Sound Effect", key="attach_sfx_btn"):
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

        with ctrl_tabs[3]:
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
            tf_c1, tf_c2 = st.columns(2)
            with tf_c1:
                st.session_state.subtitle_position = st.selectbox(
                    "Subtitle Position",
                    SUBTITLE_POSITIONS,
                    index=SUBTITLE_POSITIONS.index(st.session_state.subtitle_position),
                    key="sub_pos_box",
                )
            with tf_c2:
                st.session_state.subtitle_scale = st.slider(
                    "Subtitle Size Scale",
                    0.7,
                    1.5,
                    float(st.session_state.subtitle_scale),
                    0.05,
                    key="sub_scale_sl",
                )

            st.session_state.custom_title_text = st.text_input(
                "Hook / Title Overlay Text",
                value=st.session_state.custom_title_text,
                placeholder="Example: Ancient Yuga Chronicle!",
                key="title_overlay_input",
            )
            st.session_state.custom_title_lang = st.selectbox(
                "Overlay / Preview Text Language",
                ["English", "Kannada"],
                index=["English", "Kannada"].index(st.session_state.custom_title_lang),
                key="title_overlay_lang",
            )

        with ctrl_tabs[4]:
            up_p_col, up_v_col, up_a_col = st.columns(3)
            with up_p_col:
                p_files_top = st.file_uploader(
                    "Upload Unlimited Photos",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key="up_photo_files_top",
                )
                if st.button("Add Photos", key="btn_add_photos_top") and p_files_top:
                    os.makedirs("uploaded_studio_media", exist_ok=True)
                    new_p = []
                    for i_p, pf in enumerate(p_files_top):
                        ext = os.path.splitext(pf.name)[1].lower() or ".jpg"
                        pp = os.path.join("uploaded_studio_media", f"photo_{int(time.time())}_{i_p}{ext}")
                        with open(pp, "wb") as f:
                            f.write(pf.getbuffer())
                        new_p.append(pp)
                    st.session_state.uploaded_photo_paths.extend(new_p)
                    st.success(f"Added {len(new_p)} photo(s). Total photos: {len(st.session_state.uploaded_photo_paths)}.")
                    st.rerun()

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

    clip_pool = st.session_state.scene_videos if st.session_state.scene_videos else st.session_state.images
    if clip_pool:
        st.markdown("**🎞️ Select Clip to Preview**")
        c_btns = st.columns(min(5, len(clip_pool)))
        for idx_c in range(min(5, len(clip_pool))):
            with c_btns[idx_c]:
                active_mark = "▶ " if idx_c == st.session_state.selected_clip_idx else ""
                if st.button(f"{active_mark}Clip {idx_c + 1}", key=f"pick_sc_{idx_c}", use_container_width=True):
                    st.session_state.selected_clip_idx = idx_c
                    st.rerun()

# --- MAIN WORKFLOW TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Story & AI Director",
    "2️⃣ Scene Videos & My Media Reels",
    "3️⃣ Voice, Music & SFX",
    "4️⃣ Final Movie / Reel Export",
])

# --- TAB 1: STORY & AI DIRECTOR ---
with tab1:
    st.markdown("### Write Your Command, Story, Nature Setting, or Reel Concept")
    rough_story = st.text_area(
        "Story / Reel / Movie Command:",
        value="Mahabharata, Ramayana, and ancient yugas with mythological heroes, gods, glowing sun, and planets across sacred landscapes.",
        height=105,
    )
    num_scenes = st.slider("Number of Scenes", 2, 10, max(2, min(10, len(st.session_state.scenes) or 4)))

    t1_b1, t1_b2, t1_b3 = st.columns(3)
    with t1_b1:
        gen_scenes_clicked = st.button("✨ Generate Smart Scenes", type="primary", use_container_width=True)
    with t1_b2:
        ai_sug_clicked = st.button("🤖 AI Suggest B-Roll & FX", use_container_width=True)
    with t1_b3:
        tr_all_clicked = st.button("🌐 Auto-Translate All to Kannada", use_container_width=True)

    if gen_scenes_clicked:
        if not rough_story.strip():
            st.error("Please enter a story or command first.")
        elif not api_key or not api_key.strip():
            st.session_state.scenes = build_fallback_scenes_from_rough_story(
                rough_story, num_scenes, st.session_state.locked_character_prompt
            )
            for idx, sc in enumerate(st.session_state.scenes):
                st.session_state.scene_fx_overrides[idx] = sc.get("suggested_fx", "None")
                if sc.get("suggested_sfx") and sc["suggested_sfx"] != "None":
                    p_auto_sfx = download_fx_audio(sc["suggested_sfx"], PRESET_AUDIO_EFFECTS.get(sc["suggested_sfx"], ""))
                    if p_auto_sfx:
                        st.session_state.scene_audio_fx[idx] = p_auto_sfx
            sync_scene_widget_states(st.session_state.scenes)
            st.success(f"Created {len(st.session_state.scenes)} smart scenes with AI B-roll, motion & FX suggestions.")
            st.rerun()
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
            Create a {num_scenes}-scene story script obeying every user command for characters, gods, yugas, sun, planets, and backgrounds.{char_context}
            Return ONLY valid JSON as a list of {num_scenes} objects, no markdown.
            Each object must have:
            - "narration": clear English voiceover (1-2 sentences)
            - "kannada_narration": natural Kannada voiceover/caption (1-2 sentences)
            - "emotion": mood or expression
            - "motion": one of ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
            - "suggested_broll": 1-line B-roll shot suggestion
            - "suggested_fx": one of ["None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"]
            - "visual": concrete visual scene description with character details, yuga/background, sun, and planets
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
                    b_sug, m_sug, fx_sug, sfx_sug = infer_ml_scene_direction(
                        f"{item.get('narration', '')} {item.get('visual', '')}", idx_p
                    )
                    chosen_m = str(item.get("motion", m_sug)).strip()
                    if chosen_m not in MOTION_CHOICES:
                        chosen_m = m_sug
                    chosen_fx = str(item.get("suggested_fx", fx_sug)).strip()
                    if chosen_fx not in VIDEO_FX_CHOICES:
                        chosen_fx = fx_sug
                    normalized.append({
                        "narration": str(item.get("narration", "")).strip(),
                        "kannada_narration": str(item.get("kannada_narration", "")).strip(),
                        "emotion": str(item.get("emotion", "Expressive")).strip(),
                        "motion": chosen_m,
                        "suggested_broll": str(item.get("suggested_broll", b_sug)).strip(),
                        "suggested_fx": chosen_fx,
                        "suggested_sfx": sfx_sug,
                        "visual": str(item.get("visual", "")).strip(),
                    })
                st.session_state.scenes = sync_scenes_to_count(
                    normalized, num_scenes, rough_story, st.session_state.locked_character_prompt
                )
                for idx, sc in enumerate(st.session_state.scenes):
                    st.session_state.scene_fx_overrides[idx] = sc.get("suggested_fx", "None")
                sync_scene_widget_states(st.session_state.scenes)
                st.success(f"Generated {len(st.session_state.scenes)} AI-directed scenes!")
                st.rerun()
            else:
                st.session_state.scenes = build_fallback_scenes_from_rough_story(
                    rough_story, num_scenes, st.session_state.locked_character_prompt
                )
                for idx, sc in enumerate(st.session_state.scenes):
                    st.session_state.scene_fx_overrides[idx] = sc.get("suggested_fx", "None")
                sync_scene_widget_states(st.session_state.scenes)
                st.warning(f"Used smart local scene builder ({last_exc}).")
                st.rerun()

    if ai_sug_clicked:
        with st.spinner("Analyzing scenes for AI B-roll, camera motion & video FX suggestions..."):
            suggested_list = ai_suggest_broll_and_fx_for_scenes(api_key, st.session_state.scenes)
            st.session_state.scenes = suggested_list
            for idx, sc in enumerate(suggested_list):
                st.session_state.scene_fx_overrides[idx] = sc.get("suggested_fx", "None")
                if sc.get("suggested_sfx") and sc["suggested_sfx"] != "None":
                    p_sfx = download_fx_audio(sc["suggested_sfx"], PRESET_AUDIO_EFFECTS.get(sc["suggested_sfx"], ""))
                    if p_sfx:
                        st.session_state.scene_audio_fx[idx] = p_sfx
            sync_scene_widget_states(suggested_list)
        st.success("Applied AI B-roll, motion, video FX & sound FX suggestions across all scenes!")
        st.rerun()

    if tr_all_clicked:
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

    if st.session_state.scenes:
        st.markdown(f"### Scene Script, Reorder, AI B-Roll & FX Cards ({len(st.session_state.scenes)} Scenes)")
        for i, scene in enumerate(st.session_state.scenes):
            with st.expander(f"Scene {i + 1} • {scene.get('emotion', 'Expressive')}", expanded=(i < 2)):
                mv1, mv2 = st.columns(2)
                with mv1:
                    if st.button("⬆️ Move Up", key=f"mv_up_{i}", disabled=(i == 0), use_container_width=True):
                        swap_scenes_and_media(i, i - 1)
                        st.rerun()
                with mv2:
                    if st.button(
                        "⬇️ Move Down",
                        key=f"mv_dn_{i}",
                        disabled=(i == len(st.session_state.scenes) - 1),
                        use_container_width=True,
                    ):
                        swap_scenes_and_media(i, i + 1)
                        st.rerun()

                c_left, c_right = st.columns(2)
                with c_left:
                    nar_v = st.text_area(f"English Narration (Scene {i + 1})", value=scene.get("narration", ""), key=f"nar_{i}")
                    emo_v = st.text_input(f"Mood / Emotion (Scene {i + 1})", value=scene.get("emotion", "Expressive"), key=f"emo_{i}")
                    mot_cur = st.session_state.get(f"mot_{i}", scene.get("motion", "Auto Alternate"))
                    mot_v = st.selectbox(
                        f"Camera Motion (Scene {i + 1})",
                        MOTION_CHOICES,
                        index=MOTION_CHOICES.index(mot_cur) if mot_cur in MOTION_CHOICES else 0,
                        key=f"mot_{i}",
                    )
                    fx_cur = st.session_state.scene_fx_overrides.get(i, scene.get("suggested_fx", "None"))
                    fx_v = st.selectbox(
                        f"Suggested / Active Video FX (Scene {i + 1})",
                        VIDEO_FX_CHOICES,
                        index=VIDEO_FX_CHOICES.index(fx_cur) if fx_cur in VIDEO_FX_CHOICES else 0,
                        key=f"fx_sc_{i}",
                    )
                    broll_v = st.text_input(
                        f"AI B-Roll Shot Suggestion (Scene {i + 1})",
                        value=scene.get("suggested_broll", ""),
                        key=f"broll_{i}",
                    )
                    vis_v = st.text_input(f"Visual / Background / Action Command (Scene {i + 1})", value=scene.get("visual", ""), key=f"vis_{i}")

                    scene["narration"] = nar_v
                    scene["emotion"] = emo_v
                    scene["motion"] = mot_v
                    scene["suggested_fx"] = fx_v
                    scene["suggested_broll"] = broll_v
                    scene["visual"] = vis_v
                    st.session_state.scene_fx_overrides[i] = fx_v
                with c_right:
                    kn_v = st.text_area(
                        f"Kannada Script / Subtitle (Scene {i + 1})",
                        value=scene.get("kannada_narration", ""),
                        key=f"kn_script_{i}",
                    )
                    scene["kannada_narration"] = kn_v
                    st.caption(f"Suggested Sound FX: **{scene.get('suggested_sfx', 'None')}**")

# --- TAB 2: SCENE VIDEOS & MY MEDIA REELS ---
with tab2:
    st.markdown("### Generate Scene Videos & Custom Reels From Your Photos / Videos")

    st.markdown("#### 1. Add Unlimited Photos & Videos (For Your Own Reels or Reference)")
    m_up1, m_up2 = st.columns(2)
    with m_up1:
        multi_photo_uploads = st.file_uploader(
            "Upload Unlimited Reference / Reel Photos",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="multi_ref_photos_tab2",
        )
        if st.button("➕ Save Uploaded Photos to Pool", key="btn_save_multi_photos_t2") and multi_photo_uploads:
            os.makedirs("uploaded_studio_media", exist_ok=True)
            saved_now = []
            for idx_u, uf in enumerate(multi_photo_uploads):
                ext = os.path.splitext(uf.name)[1].lower() or ".jpg"
                out_p = os.path.join("uploaded_studio_media", f"reel_photo_{int(time.time())}_{idx_u}{ext}")
                with open(out_p, "wb") as f:
                    f.write(uf.getbuffer())
                saved_now.append(out_p)
            st.session_state.uploaded_photo_paths.extend(saved_now)
            st.success(f"Added {len(saved_now)} photo(s). Total saved photos: {len(st.session_state.uploaded_photo_paths)}")
            st.rerun()

    with m_up2:
        multi_vid_uploads = st.file_uploader(
            "Upload Video Clips for Your Reel / Movie",
            type=["mp4", "mov", "webm", "mkv"],
            accept_multiple_files=True,
            key="multi_reel_videos_tab2",
        )
        if st.button("➕ Add Uploaded Videos to Timeline", key="btn_save_multi_vids_t2") and multi_vid_uploads:
            os.makedirs("uploaded_studio_media", exist_ok=True)
            saved_vids = []
            for idx_v, vf in enumerate(multi_vid_uploads):
                ext = os.path.splitext(vf.name)[1].lower() or ".mp4"
                out_v = os.path.join("uploaded_studio_media", f"reel_vid_{int(time.time())}_{idx_v}{ext}")
                with open(out_v, "wb") as f:
                    f.write(vf.getbuffer())
                saved_vids.append(out_v)
            st.session_state.uploaded_video_clips.extend(saved_vids)
            st.session_state.scene_videos.extend(saved_vids)
            st.success(f"Added {len(saved_vids)} video clip(s) to your clip list.")
            st.rerun()

    existing_photos = [p for p in st.session_state.uploaded_photo_paths if os.path.exists(p)]
    if existing_photos:
        st.caption(f"Saved Photo Pool ({len(existing_photos)} photo(s)):")
        thumb_cols = st.columns(min(6, len(existing_photos)))
        for idx_th, p_path in enumerate(existing_photos[:6]):
            with thumb_cols[idx_th]:
                st.image(p_path, caption=f"Photo {idx_th + 1}", width=110)

    st.markdown("#### 2. How Should Your Photos/Videos Be Arranged So That They Pop + Custom Script / Wordings")
    arr_col, word_col = st.columns(2)
    with arr_col:
        st.session_state.arrangement_command = st.text_area(
            "Arrangement & Pop Instructions (How it should be arranged so that it pops):",
            value=st.session_state.arrangement_command,
            height=90,
            placeholder="Example: Split pop layout with golden border, zoom punch, high contrast so subjects pop cleanly",
        )
    with word_col:
        st.session_state.reel_wordings_input = st.text_area(
            "Custom Script / Wordings on Clips (One line per scene/reel, or leave blank to use scene script):",
            value=st.session_state.reel_wordings_input,
            height=90,
            placeholder="Line 1: Sacred Dawn Begins\nLine 2: Rising Power & Light\nLine 3: Timeless Yuga Moment\nLine 4: Grand Finale",
        )

    art_style = st.selectbox(
        "Visual Style (When Using AI Image Engine)",
        [
            "Photorealistic cinematic film still, natural lighting matching the requested environment, lifelike human expressions, 35mm lens",
            "High-detail 3D CGI animation, expressive lifelike characters, rich environment lighting and depth",
            "Detailed storybook illustration, expressive characters, rich scenery and atmosphere",
        ],
    )

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        default_scene_n = max(2, min(10, max(len(st.session_state.scenes) or 4, len(existing_photos) or 4)))
        scene_count_slider = st.slider("Scene / Reel Count", 2, 10, default_scene_n)
    with opt2:
        per_scene_sec = st.slider("Base Clip Duration (sec)", 2.0, 10.0, 4.0, 0.5)
    with opt3:
        tab2_sub_lang = st.selectbox("Wordings / Subtitles Language", ["English", "Kannada", "None"], index=0)

    allow_local_bypass = st.checkbox(
        "Use local storyboard frames if no API keys and no uploaded photos are provided",
        value=True,
    )

    if st.button("🎬 Generate All Scene / Reel Videos", type="primary"):
        active_scenes = sync_scenes_to_count(
            st.session_state.scenes,
            scene_count_slider,
            rough_story,
            st.session_state.locked_character_prompt,
        )

        custom_lines = [
            ln.strip()
            for ln in (st.session_state.reel_wordings_input or "").splitlines()
            if ln.strip()
        ]
        directives = parse_arrangement_directives(st.session_state.arrangement_command)

        for idx, sc in enumerate(active_scenes):
            sc["narration"] = st.session_state.get(f"nar_{idx}", sc.get("narration", ""))
            sc["kannada_narration"] = st.session_state.get(f"kn_script_{idx}", sc.get("kannada_narration", ""))
            sc["visual"] = st.session_state.get(f"vis_{idx}", sc.get("visual", ""))
            sc["emotion"] = st.session_state.get(f"emo_{idx}", sc.get("emotion", "Expressive"))
            sc["motion"] = directives["motion_mode"] if st.session_state.arrangement_command.strip() else st.session_state.get(f"mot_{idx}", sc.get("motion", "Auto Alternate"))
            sc["suggested_broll"] = st.session_state.get(f"broll_{idx}", sc.get("suggested_broll", ""))

            if idx < len(custom_lines):
                if tab2_sub_lang == "Kannada":
                    sc["kannada_narration"] = custom_lines[idx]
                else:
                    sc["narration"] = custom_lines[idx]

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

        char_cmd = _extract_visual_command(st.session_state.locked_character_prompt)
        cast_guard = (
            f"Subjects and setting: {char_cmd} ({int(st.session_state.exact_character_count)} main characters)."
            if st.session_state.lock_character_enabled
            else char_cmd
        )

        prog = st.progress(0)
        for idx, scene in enumerate(active_scenes):
            broll_note = _extract_visual_command(scene.get("suggested_broll", ""))
            vis_note = _extract_visual_command(scene.get("visual", ""))
            arr_note = _extract_visual_command(st.session_state.arrangement_command)
            full_prompt = (
                f"{art_style}. {cast_guard} "
                f"Arrangement: {arr_note}. "
                f"Scene {idx + 1} mood: {scene.get('emotion', 'Expressive')}. "
                f"Scene visual: {vis_note}. "
                f"B-roll: {broll_note}."
            ).strip()

            sc_caption = scene_subtitles[idx] if idx < len(scene_subtitles) else ""

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
                ref_photo_paths=existing_photos,
                arrangement_cmd=arr_note,
                caption_text=sc_caption,
                caption_lang=tab2_sub_lang,
                allow_local_bypass=allow_local_bypass,
                emotion_label=scene.get("emotion", "Expressive"),
                seed_offset=batch_id * 100,
                bg_mode=st.session_state.bg_fit_mode,
            )

            if pil_img is None:
                pil_img = build_local_bypass_scene(
                    full_prompt, width, height, idx, emotion_label=scene.get("emotion", "Scene")
                )

            img_path = f"scene_{idx}_b{batch_id}.png"
            ensure_exact_size(pil_img, width, height, bg_mode=st.session_state.bg_fit_mode).save(img_path, format="PNG")
            st.session_state.images.append(img_path)

            sc_fx = st.session_state.scene_fx_overrides.get(idx, st.session_state.editor_video_fx)
            vid_path = os.path.join("scene_videos", f"scene_{idx}_b{batch_id}.mp4")
            render_single_scene_video(
                img_path=img_path,
                out_path=vid_path,
                target_w=width,
                target_h=height,
                duration=per_scene_sec,
                motion_mode=scene.get("motion", directives["motion_mode"]),
                scene_idx=idx,
                subtitle_text=sc_caption,
                subtitle_lang=tab2_sub_lang,
                subtitle_pos=st.session_state.subtitle_position,
                subtitle_scale=st.session_state.subtitle_scale,
                enable_char_motion=True,
                brightness=st.session_state.editor_brightness + (0.06 if directives["pop_boost"] else 0.0),
                contrast=st.session_state.editor_contrast + (0.10 if directives["pop_boost"] else 0.0),
                saturation=st.session_state.editor_saturation + (0.12 if directives["pop_boost"] else 0.0),
                blur_radius=st.session_state.editor_blur,
                video_fx=sc_fx,
                bg_mode=st.session_state.bg_fit_mode,
            )
            st.session_state.scene_videos.append(vid_path)
            prog.progress((idx + 1) / len(active_scenes))

        st.success(f"Generated all {len(st.session_state.scene_videos)} scene/reel videos at {width}x{height}!")

    if st.session_state.scene_videos:
        grid_cols = st.columns(2)
        for idx, vp in enumerate(st.session_state.scene_videos):
            if os.path.exists(vp):
                with grid_cols[idx % 2]:
                    st.markdown(f"**Clip {idx + 1} Video** ({get_video_duration_safe(vp):.1f}s)")
                    st.markdown('<div class="compact-preview-box">', unsafe_allow_html=True)
                    st.video(vp)
                    st.markdown("</div>", unsafe_allow_html=True)
                    with open(vp, "rb") as vf:
                        st.download_button(
                            label=f"⬇️ Download Clip {idx + 1}",
                            data=vf.read(),
                            file_name=f"clip_{idx + 1}.mp4",
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
    st.markdown("#### Mix Levels & Auto-Ducking (Voice, Background Music & Sound FX)")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.session_state.voice_volume = st.slider("Voice Volume", 0.0, 1.5, float(st.session_state.voice_volume), 0.05)
    with m2:
        st.session_state.bgm_volume = st.slider("Normal BGM Volume", 0.0, 0.8, float(st.session_state.bgm_volume), 0.02)
    with m3:
        st.session_state.sfx_volume = st.slider("Sound FX Volume", 0.0, 1.5, float(st.session_state.sfx_volume), 0.05)

    d1, d2 = st.columns(2)
    with d1:
        st.session_state.bgm_duck_enabled = st.checkbox(
            "Auto-duck BGM volume while voiceover speaks",
            value=bool(st.session_state.bgm_duck_enabled),
        )
    with d2:
        st.session_state.bgm_duck_volume = st.slider(
            "Ducked BGM Volume",
            0.0,
            0.4,
            float(st.session_state.bgm_duck_volume),
            0.02,
            disabled=not st.session_state.bgm_duck_enabled,
        )

    if st.session_state.bgm_audio_path and os.path.exists(st.session_state.bgm_audio_path):
        st.caption(f"Active BGM: `{os.path.basename(st.session_state.bgm_audio_path)}`")
        st.audio(st.session_state.bgm_audio_path)

# --- TAB 4: FINAL MOVIE / REEL EXPORT ---
with tab4:
    st.markdown("### Export Final Video (Movie, Shorts or Reels)")
    exp1, exp2 = st.columns(2)
    with exp1:
        final_sub_lang = st.selectbox("Final Subtitles", ["None", "English", "Kannada"], index=0, key="final_sub_box")
    with exp2:
        final_motion_mode = st.selectbox(
            "Fallback Motion Mode",
            MOTION_CHOICES,
            key="final_mot_box",
        )

    st.caption(
        f"Format: `{st.session_state.format_preset}` (`{width}x{height}`) • "
        f"Speed: `{st.session_state.playback_speed}x` • "
        f"Subtitles: `{final_sub_lang}` (`{st.session_state.subtitle_position}`, `{st.session_state.subtitle_scale}x`) • "
        f"BGM Ducking: `{'On' if st.session_state.bgm_duck_enabled else 'Off'}`"
    )

    if st.button("🎬 Render Final MP4 Video", type="primary"):
        has_media = bool(st.session_state.scene_videos or st.session_state.images)
        if not has_media:
            st.error("Generate scene/reel videos in Tab 2 or upload clips first.")
        else:
            clips = []
            src_vids = []
            aud_list = []
            bgm_handles = []
            final_vid = None
            out_mp4 = f"final_studio_b{st.session_state.gen_batch_id}_{int(time.time())}.mp4"

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

                voice_intervals = []
                running_t = 0.0

                with st.spinner("Rendering your final video with studio FX, subtitles & audio ducking..."):
                    for i in range(total_slots):
                        v_aud = st.session_state.audio_files[i] if i < len(st.session_state.audio_files) else None
                        s_sfx = st.session_state.scene_audio_fx.get(i)

                        mixed_parts = []
                        dur_candidates = [4.0]
                        vo_dur = 0.0
                        if v_aud and os.path.exists(v_aud):
                            va = AudioFileClip(v_aud)
                            aud_list.append(va)
                            vo_dur = float(va.duration or 3.0)
                            dur_candidates.append(vo_dur + 0.25)
                            mixed_parts.append(scale_audio_volume(va, st.session_state.voice_volume))

                        if s_sfx and os.path.exists(s_sfx):
                            sa = AudioFileClip(s_sfx)
                            aud_list.append(sa)
                            mixed_parts.append(scale_audio_volume(sa, st.session_state.sfx_volume))

                        clip_dur = max(max(dur_candidates) / spd, 0.8)

                        caption_txt = sub_lines[i]
                        if st.session_state.custom_title_text.strip():
                            caption_txt = (
                                f"{st.session_state.custom_title_text.strip()} — {caption_txt}"
                                if caption_txt
                                else st.session_state.custom_title_text.strip()
                            )

                        sc_fx = st.session_state.scene_fx_overrides.get(i, st.session_state.editor_video_fx)

                        if i < len(st.session_state.scene_videos) and os.path.exists(st.session_state.scene_videos[i]):
                            c_item, v_handles = build_uploaded_video_clip(
                                vid_path=st.session_state.scene_videos[i],
                                target_w=width,
                                target_h=height,
                                clip_dur=clip_dur,
                                scene_idx=i,
                                subtitle_text=caption_txt,
                                subtitle_lang=final_sub_lang,
                                subtitle_pos=st.session_state.subtitle_position,
                                subtitle_scale=st.session_state.subtitle_scale,
                                brightness=st.session_state.editor_brightness,
                                contrast=st.session_state.editor_contrast,
                                saturation=st.session_state.editor_saturation,
                                blur_radius=st.session_state.editor_blur,
                                video_fx=sc_fx,
                                bg_mode=st.session_state.bg_fit_mode,
                            )
                            src_vids.extend(v_handles)
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
                                subtitle_pos=st.session_state.subtitle_position,
                                subtitle_scale=st.session_state.subtitle_scale,
                                enable_char_motion=True,
                                brightness=st.session_state.editor_brightness,
                                contrast=st.session_state.editor_contrast,
                                saturation=st.session_state.editor_saturation,
                                blur_radius=st.session_state.editor_blur,
                                video_fx=sc_fx,
                                bg_mode=st.session_state.bg_fit_mode,
                            )

                        if mixed_parts:
                            trimmed = [apply_duration(mp, min(clip_dur, float(mp.duration or clip_dur))) for mp in mixed_parts]
                            c_item = apply_audio(c_item, CompositeAudioClip(trimmed))

                        if vo_dur > 0.05:
                            voice_intervals.append((running_t, running_t + min(clip_dur, vo_dur)))

                        clips.append(c_item)
                        running_t += clip_dur

                    final_vid = apply_clip_transition(
                        clips,
                        transition_name=st.session_state.editor_transition,
                        trans_sec=st.session_state.editor_transition_sec,
                    )

                    if st.session_state.bgm_audio_path and os.path.exists(st.session_state.bgm_audio_path) and st.session_state.bgm_volume > 0:
                        total_d = max(float(final_vid.duration or 1.0), 0.5)
                        active_duck_vol = (
                            st.session_state.bgm_duck_volume
                            if st.session_state.bgm_duck_enabled
                            else st.session_state.bgm_volume
                        )
                        bgm_mixed, bgm_handles = build_ducked_bgm_track(
                            bgm_path=st.session_state.bgm_audio_path,
                            total_duration=total_d,
                            voice_intervals=voice_intervals if st.session_state.bgm_duck_enabled else [],
                            normal_vol=st.session_state.bgm_volume,
                            duck_vol=active_duck_vol,
                        )
                        if final_vid.audio is not None:
                            final_vid = apply_audio(final_vid, CompositeAudioClip([final_vid.audio, bgm_mixed]))
                        else:
                            final_vid = apply_audio(final_vid, bgm_mixed)

                    final_vid.write_videofile(
                        out_mp4,
                        fps=24,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None,
                    )

                st.markdown('<div class="compact-preview-box">', unsafe_allow_html=True)
                st.video(out_mp4)
                st.markdown("</div>", unsafe_allow_html=True)
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
                for bh in bgm_handles:
                    try:
                        bh.close()
                    except Exception:
                        pass
                if final_vid is not None:
                    try:
                        final_vid.close()
                    except Exception:
                        pass
