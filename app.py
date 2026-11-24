import asyncio
import base64
import copy
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
        max-width: 340px !important;
        max-height: 360px !important;
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
            max-width: 250px !important;
            max-height: 270px !important;
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
    "Vivid Reels": {"b": 1.04, "c": 1.12, "s": 1.24, "fx": "None"},
    "Warm Golden": {"b": 1.05, "c": 1.06, "s": 1.15, "fx": "None"},
    "Cinema Moody": {"b": 0.97, "c": 1.14, "s": 0.88, "fx": "Cinematic Vignette"},
    "Vintage Film": {"b": 1.01, "c": 1.08, "s": 0.76, "fx": "Vintage Sepia"},
    "Comedy Punch": {"b": 1.05, "c": 1.15, "s": 1.20, "fx": "Zoom Punch"},
}

FORMAT_PRESETS = {
    "9:16 Reels / Shorts": (720, 1280, "9:16"),
    "16:9 Movie / YouTube": (1280, 720, "16:9"),
    "1:1 Square Post": (1080, 1080, "1:1"),
    "4:5 Portrait Feed": (1080, 1350, "4:5"),
    "3:4 Classic Portrait": (900, 1200, "3:4"),
}

ART_STYLE_OPTIONS = [
    "Photorealistic cinematic film still, natural lighting matching the requested environment, lifelike human expressions, 35mm lens",
    "High-detail 3D CGI animation, expressive lifelike characters, rich environment lighting and depth",
    "Detailed storybook illustration, expressive characters, rich scenery and atmosphere",
]

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
MAX_HISTORY_STEPS = 3
ANIMATION_FPS = 24

for key, default in [
    ("scenes", [dict(s) for s in DEFAULT_SCENES]),
    ("images", []),
    ("scene_videos", []),
    ("uploaded_photo_paths", []),
    ("uploaded_video_clips", []),
    ("uploaded_audio_clips", []),
    ("arrangement_command", "Smooth animated story flow with strong subject focus and clear captions"),
    ("reel_wordings_input", ""),
    ("video_links_input", ""),
    ("bgm_link_input", ""),
    ("voice_links_input", ""),
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
    ("subtitle_scale", 0.85),
    ("editor_brightness", 1.0),
    ("editor_contrast", 1.04),
    ("editor_saturation", 1.06),
    ("editor_blur", 0.0),
    ("editor_transition", "Crossfade"),
    ("editor_transition_sec", 0.4),
    ("editor_video_fx", "None"),
    ("scene_audio_fx", {}),
    ("scene_fx_overrides", {}),
    ("undo_stack", []),
    ("redo_stack", []),
    ("last_history_label", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _scene_widget_key(prefix, idx):
    return f"{prefix}_{idx}_b{st.session_state.gen_batch_id}"


def sync_scene_widget_states(scenes_list):
    for idx, sc in enumerate(scenes_list):
        for prefix, field, fallback in [
            ("nar", "narration", ""),
            ("kn_script", "kannada_narration", ""),
            ("vis", "visual", ""),
            ("emo", "emotion", "Expressive"),
            ("mot", "motion", "Auto Alternate"),
            ("broll", "suggested_broll", "Establishing cinematic B-roll shot"),
        ]:
            k = _scene_widget_key(prefix, idx)
            if k not in st.session_state:
                st.session_state[k] = sc.get(field, fallback)
        k_fx = _scene_widget_key("fx_sc", idx)
        if k_fx not in st.session_state:
            st.session_state[k_fx] = st.session_state.scene_fx_overrides.get(
                idx, sc.get("suggested_fx", "None")
            )


def force_sync_scene_widget_states(scenes_list):
    for idx, sc in enumerate(scenes_list):
        st.session_state[_scene_widget_key("nar", idx)] = sc.get("narration", "")
        st.session_state[_scene_widget_key("kn_script", idx)] = sc.get("kannada_narration", "")
        st.session_state[_scene_widget_key("vis", idx)] = sc.get("visual", "")
        st.session_state[_scene_widget_key("emo", idx)] = sc.get("emotion", "Expressive")
        st.session_state[_scene_widget_key("mot", idx)] = sc.get("motion", "Auto Alternate")
        st.session_state[_scene_widget_key("broll", idx)] = sc.get(
            "suggested_broll", "Establishing cinematic B-roll shot"
        )
        st.session_state[_scene_widget_key("fx_sc", idx)] = st.session_state.scene_fx_overrides.get(
            idx, sc.get("suggested_fx", "None")
        )


sync_scene_widget_states(st.session_state.scenes)


def _capture_snapshot(action_label="Edit"):
    return {
        "label": action_label,
        "scenes": copy.deepcopy(st.session_state.scenes),
        "images": list(st.session_state.images),
        "scene_videos": list(st.session_state.scene_videos),
        "uploaded_photo_paths": list(st.session_state.uploaded_photo_paths),
        "uploaded_video_clips": list(st.session_state.uploaded_video_clips),
        "arrangement_command": str(st.session_state.arrangement_command),
        "reel_wordings_input": str(st.session_state.reel_wordings_input),
        "video_links_input": str(st.session_state.video_links_input),
        "bgm_link_input": str(st.session_state.bgm_link_input),
        "voice_links_input": str(st.session_state.voice_links_input),
        "audio_files": list(st.session_state.audio_files),
        "bgm_audio_path": str(st.session_state.bgm_audio_path),
        "bgm_volume": float(st.session_state.bgm_volume),
        "bgm_duck_enabled": bool(st.session_state.bgm_duck_enabled),
        "bgm_duck_volume": float(st.session_state.bgm_duck_volume),
        "voice_volume": float(st.session_state.voice_volume),
        "sfx_volume": float(st.session_state.sfx_volume),
        "locked_character_prompt": str(st.session_state.locked_character_prompt),
        "exact_character_count": int(st.session_state.exact_character_count),
        "lock_character_enabled": bool(st.session_state.lock_character_enabled),
        "selected_clip_idx": int(st.session_state.selected_clip_idx),
        "format_preset": str(st.session_state.format_preset),
        "bg_fit_mode": str(st.session_state.bg_fit_mode),
        "playback_speed": float(st.session_state.playback_speed),
        "custom_title_text": str(st.session_state.custom_title_text),
        "custom_title_lang": str(st.session_state.custom_title_lang),
        "subtitle_position": str(st.session_state.subtitle_position),
        "subtitle_scale": float(st.session_state.subtitle_scale),
        "editor_brightness": float(st.session_state.editor_brightness),
        "editor_contrast": float(st.session_state.editor_contrast),
        "editor_saturation": float(st.session_state.editor_saturation),
        "editor_blur": float(st.session_state.editor_blur),
        "editor_transition": str(st.session_state.editor_transition),
        "editor_transition_sec": float(st.session_state.editor_transition_sec),
        "editor_video_fx": str(st.session_state.editor_video_fx),
        "scene_audio_fx": copy.deepcopy(st.session_state.scene_audio_fx),
        "scene_fx_overrides": copy.deepcopy(st.session_state.scene_fx_overrides),
    }


def _apply_snapshot(snap):
    for k, v in snap.items():
        if k == "label":
            continue
        st.session_state[k] = copy.deepcopy(v)
    force_sync_scene_widget_states(st.session_state.scenes)


def record_undo_step(action_label="Edit"):
    snap = _capture_snapshot(action_label=action_label)
    u_stack = list(st.session_state.undo_stack)
    u_stack.append(snap)
    if len(u_stack) > MAX_HISTORY_STEPS:
        u_stack = u_stack[-MAX_HISTORY_STEPS:]
    st.session_state.undo_stack = u_stack
    st.session_state.redo_stack = []
    st.session_state.last_history_label = action_label


def perform_undo():
    u_stack = list(st.session_state.undo_stack)
    if not u_stack:
        return False
    r_stack = list(st.session_state.redo_stack)
    r_stack.append(_capture_snapshot(action_label=st.session_state.last_history_label or "Redo Step"))
    if len(r_stack) > MAX_HISTORY_STEPS:
        r_stack = r_stack[-MAX_HISTORY_STEPS:]
    target_snap = u_stack.pop()
    st.session_state.undo_stack = u_stack
    st.session_state.redo_stack = r_stack
    st.session_state.last_history_label = f"Undid: {target_snap.get('label', 'Change')}"
    _apply_snapshot(target_snap)
    return True


def perform_redo():
    r_stack = list(st.session_state.redo_stack)
    if not r_stack:
        return False
    u_stack = list(st.session_state.undo_stack)
    u_stack.append(_capture_snapshot(action_label="Undo Step"))
    if len(u_stack) > MAX_HISTORY_STEPS:
        u_stack = u_stack[-MAX_HISTORY_STEPS:]
    target_snap = r_stack.pop()
    st.session_state.redo_stack = r_stack
    st.session_state.undo_stack = u_stack
    st.session_state.last_history_label = f"Redid: {target_snap.get('label', 'Change')}"
    _apply_snapshot(target_snap)
    return True


def _is_valid_media_url(raw_url):
    cleaned = (raw_url or "").strip()
    if not cleaned:
        return ""
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
        return cleaned
    raise ValueError(f"Unsupported or invalid URL: {cleaned}")


def _guess_media_ext(url, content_type, media_kind="video"):
    parsed = urllib.parse.urlparse(url)
    path_ext = os.path.splitext(parsed.path or "")[1].lower()
    if media_kind == "video" and path_ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        return path_ext
    if media_kind == "audio" and path_ext in (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"):
        return path_ext

    ct = (content_type or "").lower()
    if media_kind == "video":
        if "webm" in ct:
            return ".webm"
        if "quicktime" in ct or "mov" in ct:
            return ".mov"
        return ".mp4"
    else:
        if "wav" in ct:
            return ".wav"
        if "ogg" in ct:
            return ".ogg"
        if "mp4" in ct or "m4a" in ct or "aac" in ct:
            return ".m4a"
        return ".mp3"


def download_media_from_url(raw_url, media_kind="video", prefix="link_media", timeout=90):
    valid_url = _is_valid_media_url(raw_url)
    os.makedirs("uploaded_studio_media", exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AICinemaStudio/3.0)"}
    resp = requests.get(valid_url, headers=headers, stream=True, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} when fetching {valid_url}")

    ext = _guess_media_ext(valid_url, resp.headers.get("Content-Type", ""), media_kind=media_kind)
    out_path = os.path.join("uploaded_studio_media", f"{prefix}_{int(time.time() * 1000)}{ext}")

    total_bytes = 0
    max_bytes = 350 * 1024 * 1024 if media_kind == "video" else 100 * 1024 * 1024
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=131072):
            if chunk:
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise RuntimeError("Downloaded file exceeded size limit.")
                f.write(chunk)

    if total_bytes < 512 or not os.path.exists(out_path):
        raise RuntimeError("Downloaded file appears empty or incomplete.")
    return out_path


def _ascii_safe(text):
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in (text or ""))
    return " ".join(cleaned.split())


def _extract_visual_command(text):
    raw = " ".join((text or "").replace("\n", " ").split())
    for phrase in ("where required", "as commanded", "as requested", "follow the character through"):
        raw = raw.replace(phrase, " ").replace(phrase.title(), " ")
    return " ".join(raw.split())


def parse_arrangement_directives(arrangement_text):
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

    pop_boost = any(
        k in low for k in ["pop", "vibrant", "bright", "glow", "contrast", "bold", "highlight", "stand out"]
    )
    return {
        "layout_mode": layout_mode,
        "motion_mode": motion_mode,
        "pop_boost": pop_boost,
        "raw_note": " ".join((arrangement_text or "").split()),
    }


def swap_scenes_and_media(idx_a, idx_b):
    n_sc = len(st.session_state.scenes)
    if not (0 <= idx_a < n_sc and 0 <= idx_b < n_sc and idx_a != idx_b):
        return

    record_undo_step(f"Reorder Scene {idx_a + 1} & {idx_b + 1}")
    st.session_state.scenes[idx_a], st.session_state.scenes[idx_b] = (
        st.session_state.scenes[idx_b],
        st.session_state.scenes[idx_a],
    )

    for list_key in ("images", "scene_videos", "audio_files"):
        lst = list(st.session_state.get(list_key) or [])
        if idx_a < len(lst) and idx_b < len(lst):
            lst[idx_a], lst[idx_b] = lst[idx_b], lst[idx_a]
            st.session_state[list_key] = lst

    for map_key in ("scene_audio_fx", "scene_fx_overrides"):
        m = dict(st.session_state[map_key])
        va, vb = m.get(idx_a), m.get(idx_b)
        if vb is not None:
            m[idx_a] = vb
        else:
            m.pop(idx_a, None)
        if va is not None:
            m[idx_b] = va
        else:
            m.pop(idx_b, None)
        st.session_state[map_key] = m

    force_sync_scene_widget_states(st.session_state.scenes)


def run_async(coro):
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
    src = None
    seg = None
    try:
        src = VideoFileClip(src_path)
        seg = subclip_safe(src, start_sec, end_sec)
        seg.write_videofile(
            out_path,
            fps=ANIMATION_FPS,
            codec="libx264",
            audio_codec="aac" if src.audio is not None else None,
            audio=src.audio is not None,
            ffmpeg_params=["-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
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
            fps=ANIMATION_FPS,
            codec="libx264",
            audio_codec="aac" if src.audio is not None else None,
            audio=src.audio is not None,
            ffmpeg_params=["-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
            logger=None,
        )

        part_b = subclip_safe(src, cut, dur)
        part_b.write_videofile(
            part_b_path,
            fps=ANIMATION_FPS,
            codec="libx264",
            audio_codec="aac" if src.audio is not None else None,
            audio=src.audio is not None,
            ffmpeg_params=["-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
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
        bg = bg.crop((bx, by, bx + target_w, by + target_h)).filter(ImageFilter.GaussianBlur(radius=14))
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
        out = ImageEnhance.Brightness(out).enhance(1.30)
    elif video_fx == "Glitch RGB":
        r, g, b = out.split()
        r = ImageEnhance.Brightness(r).enhance(1.15)
        b = ImageEnhance.Contrast(b).enhance(1.20)
        out = Image.merge("RGB", (r, g, b))

    return out


@st.cache_data(show_spinner=False)
def download_fx_audio(fx_name, fx_url):
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


def _resolve_font_path(lang="English"):
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
                return path

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
                        return tmp_font
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
            return path
    return ""


@st.cache_resource(show_spinner=False)
def _cached_font(font_size: int, lang: str = "English"):
    font_path = _resolve_font_path(lang=lang)
    if font_path:
        try:
            return ImageFont.truetype(font_path, int(font_size)), font_path
        except Exception:
            pass
    return ImageFont.load_default(), "default_bitmap"


def _find_font(font_size, lang="English"):
    return _cached_font(int(font_size), lang=lang)


def translate_lines_to_kannada(gemini_key, lines):
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
    low = (text or "").lower()
    if any(w in low for w in ["funny", "joke", "oops", "slip", "laugh", "comedy", "silly", "meme"]):
        return (
            "Fast energetic comedy animation with expressive motion & punchy framing",
            "Zoom Punch",
            "Zoom Punch",
            "😂 Funny Laugh",
        )
    if any(w in low for w in ["rain", "storm", "thunder", "lightning"]):
        return (
            "Animated storm clouds, falling streaks, and dramatic lighting shifts",
            "Pan Left",
            "Cinematic Vignette",
            "🌿 Rain & Thunder",
        )
    if any(w in low for w in ["ocean", "sea", "wave", "beach", "water", "river"]):
        return (
            "Animated rolling waves, shimmering water light, and drifting horizon",
            "Pan Right",
            "None",
            "🌊 Ocean Waves",
        )
    if any(w in low for w in ["fire", "flame", "camp", "warm", "torch"]):
        return (
            "Animated warm firelight, rising embers, and glowing depth",
            "Zoom In",
            "Cinematic Vignette",
            "🔥 Campfire Crackle",
        )
    if any(w in low for w in ["forest", "jungle", "tree", "bird", "nature", "morning"]):
        return (
            "Animated sunlight beams, drifting foliage depth, and peaceful motion",
            "Pan Right",
            "None",
            "🌲 Forest Birds",
        )
    if any(w in low for w in ["shock", "twist", "sudden", "boom", "dramatic", "wow"]):
        return (
            "High-impact animated reveal with camera punch and dynamic light",
            "Zoom Punch",
            "Shake",
            "💥 Vine Boom (Meme)",
        )
    motions = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    broll_shots = [
        "Animated celestial dawn with glowing sun, orbiting planets, and yuga horizon",
        "Animated hero & story beat with dynamic camera pan, rays, and sacred setting",
        "Animated journey across sacred lands with drifting sky and planetary light",
        "Animated climactic rays and smooth pull-back uniting character and cosmos",
    ]
    return broll_shots[idx % len(broll_shots)], motions[idx % len(motions)], "None", "None"


def ai_suggest_broll_and_fx_for_scenes(gemini_key, scenes):
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
            "You are a film & reels animation director. For each scene in the JSON input, suggest:\n"
            "- 'broll': concise 1-line animated shot/motion idea matching the story\n"
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
    target_count = max(1, min(10, int(count)))
    clean_story = _extract_visual_command(rough_text)
    clean_char = _extract_visual_command(char_prompt)
    sentences = [
        s.strip()
        for s in clean_story.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    beat_moods = [
        ("Opening & Celestial Setting", "animated wide view with glowing sun, moving planets, and ancient yuga sky"),
        ("Mythological Presence", "animated hero in traditional attire and ornaments amidst ancient yuga setting"),
        ("Divine & Cosmic Shift", "animated light rays, sun, and planets shifting across ancient sky"),
        ("Action & Journey", "dynamic animated motion through ancient yuga landscape and sacred environment"),
        ("Emotional Intensity", "expressive animated scene with golden light rays and rich detail"),
        ("Sacred Environment", "animated ancient architecture, nature, and celestial sky"),
        ("Rising Drama", "bold animated framing with glowing sun, planets, and active story beat"),
        ("Quiet Reflection", "smooth animated atmospheric framing in ancient yuga twilight"),
        ("Turning Point", "striking animated contrast between characters, ancient setting, and planets"),
        ("Closing Vision", "grand animated closing shot with characters, yuga backdrop, sun, and planets"),
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
                f"{beat_desc}. Story elements: {combined_ctx}. "
                f"Action: {base_line}."
            ).strip(),
        })
    return scenes


def sync_scenes_to_count(existing_scenes, desired_count, rough_text, char_prompt):
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
    scale=0.85,
):
    if not subtitle_text or not subtitle_text.strip() or lang == "None":
        return frame_pil

    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    safe_scale = max(0.60, min(1.25, float(scale or 0.85)))
    font_size = max(18, int(round(min(target_w, target_h) * 0.028 * safe_scale)))
    font, _ = _find_font(font_size, lang=lang)
    lang_code = "kn" if lang == "Kannada" else "en"

    chars_per_line = max(18, int((target_w * 0.84) / max(9, font_size * 0.56)))
    wrapped_lines = textwrap.wrap(subtitle_text.strip(), width=chars_per_line)[:2]
    if not wrapped_lines:
        return frame_pil

    line_h = font_size + 8
    pad_y = 8
    box_h = len(wrapped_lines) * line_h + pad_y * 2
    pad_x = max(16, int(target_w * 0.05))

    if position == "Top":
        box_top = max(14, int(target_h * 0.04))
    elif position == "Center":
        box_top = max(14, (target_h - box_h) // 2)
    else:
        box_top = max(14, target_h - box_h - max(24, int(target_h * 0.04)))

    box_bot = min(target_h - 12, box_top + box_h)
    box_top = max(12, box_bot - box_h)

    draw.rounded_rectangle(
        [pad_x, box_top, target_w - pad_x, box_bot],
        radius=10,
        fill=(8, 11, 18, 195),
    )

    cur_y = box_top + pad_y
    for line in wrapped_lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font, language=lang_code)
        except TypeError:
            bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = max(pad_x + 10, (target_w - text_w) // 2)

        for ox, oy, col in [
            (text_x + 1, cur_y + 1, (0, 0, 0, 235)),
            (text_x, cur_y, (255, 252, 240, 255)),
        ]:
            try:
                draw.text((ox, oy), line, font=font, fill=col, language=lang_code)
            except TypeError:
                draw.text((ox, oy), line, font=font, fill=col)
        cur_y += line_h

    return img_rgba.convert("RGB")


def _render_story_animated_pil_frame(
    t,
    duration,
    target_w,
    target_h,
    scene_idx=0,
    story_context="",
    emotion_label="",
    motion_mode="Auto Alternate",
    base_img_path="",
    extra_photo_paths=None,
    brightness=1.0,
    contrast=1.04,
    saturation=1.06,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Build a smooth animated frame at time `t` driven by story context, motion, and media."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    safe_dur = max(0.5, float(duration or 4.0))
    u = max(0.0, min(1.0, float(t) / safe_dur))
    ease = u * u * (3.0 - 2.0 * u)

    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    low = (story_context or "").lower()
    is_cosmic = any(k in low for k in ["sun", "planet", "yuga", "mahabharata", "ramayana", "god", "cosmic", "divine"])
    is_nature = any(k in low for k in ["forest", "jungle", "tree", "nature", "mountain", "river", "ocean", "sea", "fire"])

    valid_photos = [p for p in (extra_photo_paths or []) if p and os.path.exists(p)]

    if base_img_path and os.path.exists(base_img_path):
        with Image.open(base_img_path) as im:
            src_rgb = im.convert("RGB")
        # If multiple photos were provided, blend smoothly toward the next photo across the second half
        if len(valid_photos) > 1 and u > 0.55:
            next_p = valid_photos[(scene_idx + 1) % len(valid_photos)]
            if os.path.exists(next_p) and next_p != base_img_path:
                with Image.open(next_p) as im2:
                    nxt_rgb = ensure_exact_size(im2, src_rgb.width, src_rgb.height, bg_mode=bg_mode)
                alpha_mix = min(0.85, (u - 0.55) / 0.45)
                src_rgb = Image.blend(src_rgb, nxt_rgb, alpha_mix)

        overscan = 1.14
        cw = int(round(target_w * overscan))
        ch = int(round(target_h * overscan))
        canvas = ensure_exact_size(src_rgb, cw, ch, bg_mode=bg_mode)

        if video_fx == "Zoom Punch":
            scale = 1.0 + 0.10 * math.sin(u * math.pi)
        elif active_mode == "Zoom In":
            scale = 1.0 + 0.11 * ease
        elif active_mode == "Zoom Out":
            scale = 1.12 - 0.11 * ease
        else:
            scale = 1.08

        win_w = max(2, int(round(target_w / scale)))
        win_h = max(2, int(round(target_h / scale)))

        if active_mode == "Pan Right":
            x0 = int(round(max(0, cw - win_w) * ease))
            y0 = max(0, (ch - win_h) // 2) + int(round(math.sin(u * math.pi * 2) * 8))
        elif active_mode == "Pan Left":
            x0 = int(round(max(0, cw - win_w) * (1.0 - ease)))
            y0 = max(0, (ch - win_h) // 2) + int(round(math.cos(u * math.pi * 2) * 8))
        else:
            x0 = (cw - win_w) // 2 + int(round(math.sin(u * math.pi * 2) * 10))
            y0 = (ch - win_h) // 2 + int(round(math.cos(u * math.pi * 2) * 7))

        if video_fx == "Shake":
            x0 += int(round(math.sin(t * 28.0) * 6))
            y0 += int(round(math.cos(t * 32.0) * 5))

        x0 = max(0, min(cw - win_w, x0))
        y0 = max(0, min(ch - win_h, y0))
        frame_pil = canvas.crop((x0, y0, x0 + win_w, y0 + win_h)).resize((target_w, target_h), resampling)
    else:
        # Story-driven multi-layer animated scene when no external image is available
        if is_cosmic:
            palettes = [
                ((20, 24, 58), (168, 92, 42), (14, 16, 30)),
                ((16, 34, 66), (208, 128, 48), (12, 18, 34)),
                ((36, 20, 56), (184, 84, 62), (16, 12, 28)),
                ((18, 42, 60), (194, 138, 58), (10, 20, 32)),
            ]
        elif is_nature:
            palettes = [
                ((16, 46, 36), (42, 96, 68), (10, 26, 20)),
                ((22, 56, 50), (56, 112, 90), (12, 30, 26)),
                ((32, 46, 36), (106, 90, 56), (16, 24, 16)),
                ((14, 40, 54), (44, 96, 92), (9, 22, 30)),
            ]
        else:
            palettes = [
                ((20, 32, 54), (42, 62, 92), (12, 18, 30)),
                ((34, 26, 52), (66, 52, 90), (16, 12, 26)),
                ((18, 32, 46), (94, 70, 54), (12, 18, 26)),
                ((24, 42, 40), (52, 82, 74), (12, 22, 20)),
            ]

        top_col, mid_col, bot_col = palettes[scene_idx % len(palettes)]
        y_grid = np.linspace(0.0, 1.0, target_h, dtype=np.float32)[:, None, None]
        x_grid = np.linspace(0.0, 1.0, target_w, dtype=np.float32)[None, :, None]
        c_top = np.array(top_col, dtype=np.float32)[None, None, :]
        c_mid = np.array(mid_col, dtype=np.float32)[None, None, :]
        c_bot = np.array(bot_col, dtype=np.float32)[None, None, :]

        b1 = np.clip(y_grid / 0.58, 0.0, 1.0)
        b2 = np.clip((y_grid - 0.58) / 0.42, 0.0, 1.0)
        bg_arr = (c_top * (1.0 - b1) + c_mid * b1) * (1.0 - b2) + c_bot * b2

        # Animated sun & planet motion across the scene
        sun_cx = target_w * (0.24 + 0.14 * (scene_idx % 4) + 0.12 * ease)
        sun_cy = target_h * (0.21 + 0.03 * math.sin(u * math.pi * 2 + scene_idx))
        dist_sq = (x_grid[:, :, 0] - sun_cx) ** 2 + (y_grid[:, :, 0] - sun_cy) ** 2
        glow_r = min(target_w, target_h) * (0.24 + 0.03 * math.sin(u * math.pi * 2))
        glow = np.exp(-dist_sq / (2.0 * glow_r ** 2))[:, :, None]
        bg_arr = np.clip(bg_arr + glow * np.array([84.0, 62.0, 36.0], dtype=np.float32), 0, 255)

        frame_pil = Image.fromarray(bg_arr.astype(np.uint8), mode="RGB").convert("RGBA")
        ov = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)

        sun_r = max(18, int(min(target_w, target_h) * 0.044))
        draw.ellipse(
            [sun_cx - sun_r, sun_cy - sun_r, sun_cx + sun_r, sun_cy + sun_r],
            fill=(255, 212, 112, 220),
        )

        # Orbiting planets
        for p_i in range(2):
            ang = u * math.pi * 1.4 + scene_idx * 1.1 + p_i * 2.2
            px = target_w * (0.72 + 0.08 * math.cos(ang))
            py = target_h * (0.18 + 0.05 * math.sin(ang) + p_i * 0.07)
            pr = max(9, int(min(target_w, target_h) * (0.022 - p_i * 0.005)))
            p_col = (175, 200, 248, 205) if p_i == 0 else (238, 158, 92, 195)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=p_col)

        # Animated drifting horizon / environment layers
        pan_shift = (ease - 0.5) * 90.0 if active_mode in ("Pan Right", "Zoom In") else (0.5 - ease) * 90.0
        horizon_y = int(target_h * 0.64)
        ridge_pts = [(0, target_h)]
        for x_s in range(0, target_w + 24, 24):
            wx = x_s + pan_shift * 0.45
            wave = (
                math.sin(wx * 0.008 + scene_idx * 1.2 + u * 2.0) * (target_h * 0.032)
                + math.cos(wx * 0.02 - scene_idx) * (target_h * 0.014)
            )
            ridge_pts.append((x_s, int(horizon_y + wave)))
        ridge_pts.append((target_w, target_h))
        draw.polygon(ridge_pts, fill=(14, 20, 30, 185))

        # Animated hero silhouette & crown/light motion
        bob = int(round(math.sin(u * math.pi * 2.0 + scene_idx) * 10))
        stride_x = int(round((ease - 0.5) * target_w * 0.08))
        cx = int(target_w * 0.50) + stride_x
        head_r = max(26, int(min(target_w, target_h) * 0.048))
        head_cy = int(target_h * 0.45) + bob

        draw.polygon(
            [
                (cx - int(head_r * 0.85), head_cy - int(head_r * 0.75)),
                (cx - int(head_r * 0.45), head_cy - int(head_r * 1.35)),
                (cx, head_cy - int(head_r * 0.95)),
                (cx + int(head_r * 0.45), head_cy - int(head_r * 1.35)),
                (cx + int(head_r * 0.85), head_cy - int(head_r * 0.75)),
            ],
            fill=(248, 198, 78, 245),
        )
        draw.rounded_rectangle(
            [cx - int(head_r * 1.95), head_cy + head_r + 6, cx + int(head_r * 1.95), head_cy + head_r + int(head_r * 3.3)],
            radius=28,
            fill=(28, 36, 54, 235),
            outline=(235, 195, 115, 150),
            width=2,
        )
        draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=(216, 172, 142, 250))

        eye_y = head_cy - 2
        eye_dx = max(9, int(head_r * 0.35))
        draw.ellipse([cx - eye_dx - 4, eye_y - 4, cx - eye_dx + 4, eye_y + 4], fill=(22, 24, 30, 255))
        draw.ellipse([cx + eye_dx - 4, eye_y - 4, cx + eye_dx + 4, eye_y + 4], fill=(22, 24, 30, 255))
        draw.arc([cx - 10, head_cy + 8, cx + 10, head_cy + 22], 15, 165, fill=(115, 58, 52, 255), width=2)

        # Story beat caption card
        card_w = int(target_w * 0.88)
        card_h = int(target_h * 0.18)
        cx0 = (target_w - card_w) // 2
        cy0 = int(target_h * 0.73)
        draw.rounded_rectangle(
            [cx0, cy0, cx0 + card_w, cy0 + card_h],
            radius=16,
            fill=(10, 14, 22, 212),
            outline=(255, 46, 99, 180),
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
        wrapped = textwrap.wrap(
            _ascii_safe(story_context),
            width=max(26, int(card_w / max(10, int(target_w * 0.021) * 0.56))),
        )[:3]
        cur_y = cy0 + 44
        for line in wrapped:
            draw.text((cx0 + 18, cur_y), line, font=body_font, fill=(242, 246, 252, 245))
            cur_y += int(target_w * 0.025)

        frame_pil = Image.alpha_composite(frame_pil, ov).convert("RGB")

    frame_pil = apply_pil_effects(
        frame_pil,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        blur_radius=blur_radius,
        video_fx=video_fx,
    )
    return ensure_exact_size(frame_pil, target_w, target_h, bg_mode="Crop Fill")


def build_animated_story_clip(
    target_w,
    target_h,
    duration,
    scene_idx=0,
    story_context="",
    emotion_label="",
    motion_mode="Auto Alternate",
    base_img_path="",
    extra_photo_paths=None,
    subtitle_text="",
    subtitle_lang="None",
    subtitle_pos="Bottom",
    subtitle_scale=0.85,
    brightness=1.0,
    contrast=1.04,
    saturation=1.06,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Build a smooth 24 FPS animated clip driven by story context, motion, and images."""
    safe_dur = max(float(duration), 0.5)
    has_sub = bool(subtitle_lang != "None" and subtitle_text and subtitle_text.strip())

    def make_frame(t):
        pil_f = _render_story_animated_pil_frame(
            t=t,
            duration=safe_dur,
            target_w=target_w,
            target_h=target_h,
            scene_idx=scene_idx,
            story_context=story_context,
            emotion_label=emotion_label,
            motion_mode=motion_mode,
            base_img_path=base_img_path,
            extra_photo_paths=extra_photo_paths,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            blur_radius=blur_radius,
            video_fx=video_fx,
            bg_mode=bg_mode,
        )
        if has_sub:
            pil_f = draw_subtitle_on_frame(
                pil_f,
                subtitle_text=subtitle_text,
                target_w=target_w,
                target_h=target_h,
                lang=subtitle_lang,
                position=subtitle_pos,
                scale=subtitle_scale,
            )
            pil_f = ensure_exact_size(pil_f, target_w, target_h, bg_mode="Crop Fill")
        return np.asarray(pil_f, dtype=np.uint8)

    if MOVIEPY_V2 and hasattr(ImageClip, "transform"):
        pass
    try:
        from moviepy.video.VideoClip import VideoClip as DynamicVideoClip
        return apply_duration(DynamicVideoClip(make_frame, duration=safe_dur), safe_dur)
    except Exception:
        steps = max(12, int(round(safe_dur * ANIMATION_FPS)))
        dt = safe_dur / steps
        parts = [
            apply_duration(ImageClip(make_frame(s * dt)), dt)
            for s in range(steps)
        ]
        return concatenate_videoclips(parts, method="chain")


def render_single_scene_video(
    img_path,
    out_path,
    target_w,
    target_h,
    duration=4.0,
    motion_mode="Auto Alternate",
    scene_idx=0,
    story_context="",
    emotion_label="",
    extra_photo_paths=None,
    subtitle_text="",
    subtitle_lang="None",
    subtitle_pos="Bottom",
    subtitle_scale=0.85,
    brightness=1.0,
    contrast=1.04,
    saturation=1.06,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    clip = build_animated_story_clip(
        target_w=target_w,
        target_h=target_h,
        duration=duration,
        scene_idx=scene_idx,
        story_context=story_context,
        emotion_label=emotion_label,
        motion_mode=motion_mode,
        base_img_path=img_path,
        extra_photo_paths=extra_photo_paths,
        subtitle_text=subtitle_text,
        subtitle_lang=subtitle_lang,
        subtitle_pos=subtitle_pos,
        subtitle_scale=subtitle_scale,
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
            fps=ANIMATION_FPS,
            codec="libx264",
            audio=False,
            ffmpeg_params=["-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
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
    subtitle_text="",
    subtitle_lang="None",
    subtitle_pos="Bottom",
    subtitle_scale=0.85,
    brightness=1.0,
    contrast=1.04,
    saturation=1.06,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Play real video clips smoothly without chopping them into low-FPS slide steps."""
    src = None
    try:
        src = VideoFileClip(vid_path)
        native_dur = max(0.5, float(src.duration or clip_dur))
        if native_dur < clip_dur and hasattr(src, "loop"):
            timed = apply_duration(src.loop(duration=clip_dur), clip_dur)
        else:
            timed = subclip_safe(src, 0.0, min(native_dur, clip_dur))
            timed = apply_duration(timed, clip_dur)

        has_sub = bool(subtitle_lang != "None" and subtitle_text and subtitle_text.strip())
        needs_fx = (
            abs(brightness - 1.0) > 1e-3
            or abs(contrast - 1.0) > 1e-3
            or abs(saturation - 1.0) > 1e-3
            or (blur_radius and blur_radius > 0.05)
            or (video_fx and video_fx != "None")
        )
        needs_resize = (int(getattr(timed, "w", target_w)) != target_w) or (
            int(getattr(timed, "h", target_h)) != target_h
        )

        if not needs_fx and not needs_resize and not has_sub:
            return timed, [src]

        def transform_frame(get_frame, t):
            raw_arr = get_frame(t)
            pil_f = Image.fromarray(np.asarray(raw_arr, dtype=np.uint8))
            if needs_resize:
                pil_f = ensure_exact_size(pil_f, target_w, target_h, bg_mode=bg_mode)
            if needs_fx:
                pil_f = apply_pil_effects(
                    pil_f,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    blur_radius=blur_radius,
                    video_fx=video_fx,
                )
            if has_sub:
                pil_f = draw_subtitle_on_frame(
                    pil_f,
                    subtitle_text=subtitle_text,
                    target_w=target_w,
                    target_h=target_h,
                    lang=subtitle_lang,
                    position=subtitle_pos,
                    scale=subtitle_scale,
                )
            pil_f = ensure_exact_size(pil_f, target_w, target_h, bg_mode="Crop Fill")
            return np.asarray(pil_f, dtype=np.uint8)

        if hasattr(timed, "transform"):
            transformed = timed.transform(lambda gf, t: transform_frame(gf, t))
            if timed.audio is not None:
                transformed = apply_audio(transformed, timed.audio)
            return transformed, [src, timed]
        elif hasattr(timed, "fl"):
            transformed = timed.fl(lambda gf, t: transform_frame(gf, t))
            if timed.audio is not None:
                transformed = apply_audio(transformed, timed.audio)
            return transformed, [src, timed]

        return timed, [src]
    except Exception:
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        raise


def build_ducked_bgm_track(bgm_path, total_duration, voice_intervals, normal_vol=0.22, duck_vol=0.08):
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


def generate_cloudflare_flux_image(cf_account_id, cf_api_token, prompt_text, target_w, target_h):
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
    resp = requests.post(url, headers=headers, json={"prompt": bounded_prompt}, timeout=90)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    if "image/" in resp.headers.get("Content-Type", ""):
        return ensure_exact_size(Image.open(BytesIO(resp.content)).convert("RGB"), target_w, target_h, bg_mode="Crop Fill")

    data = resp.json()
    b64_img = (data.get("result", {}) if isinstance(data, dict) else {}).get("image")
    if b64_img:
        return ensure_exact_size(
            Image.open(BytesIO(base64.b64decode(b64_img))).convert("RGB"),
            target_w,
            target_h,
            bg_mode="Crop Fill",
        )
    raise ValueError(f"Unexpected Cloudflare response: {data}")


def generate_pollinations_image(prompt_text, target_w, target_h, seed=42):
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
                return ensure_exact_size(Image.open(BytesIO(resp.content)).convert("RGB"), target_w, target_h, bg_mode="Crop Fill")
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
    caption_lang="None",
    allow_local_bypass=True,
    emotion_label="",
    seed_offset=0,
    bg_mode="Blur Canvas",
):
    errors = []
    scene_seed = 100 + scene_idx * 37 + seed_offset
    valid_photos = [p for p in (ref_photo_paths or []) if p and os.path.exists(p)]

    def try_cloudflare():
        if cf_account_id.strip() and cf_api_token.strip():
            return generate_cloudflare_flux_image(cf_account_id, cf_api_token, prompt_text, target_w, target_h), "Cloudflare FLUX.1-schnell"
        raise RuntimeError("Missing Cloudflare Account ID or API Token.")

    def try_huggingface():
        if not hf_token.strip():
            raise RuntimeError("Missing Hugging Face Token.")
        client = InferenceClient(token=hf_token.strip())
        w_hf = max(512, min(1024, (target_w // 8) * 8))
        h_hf = max(512, min(1024, (target_h // 8) * 8))
        last_hf_err = None
        for model_id in ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"]:
            try:
                img = client.text_to_image(prompt=_extract_visual_command(prompt_text), model=model_id, width=w_hf, height=h_hf)
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
                            return ensure_exact_size(Image.open(BytesIO(inline.data)).convert("RGB"), target_w, target_h, bg_mode=bg_mode), f"Gemini {model_id}"
                last_gem_err = RuntimeError(f"{model_id}: no image returned")
            except Exception as exc:
                last_gem_err = exc
        raise RuntimeError(str(last_gem_err))

    def try_pollinations():
        return generate_pollinations_image(prompt_text, target_w, target_h, seed=200 + scene_seed), "Pollinations AI Fallback"

    has_any_key = bool(
        (cf_account_id and cf_account_id.strip() and cf_api_token and cf_api_token.strip())
        or (hf_token and hf_token.strip())
        or (gemini_key and gemini_key.strip())
    )

    if has_any_key:
        if engine_choice == "Cloudflare Workers AI (Free Daily FLUX)":
            ordered_runners = [("Cloudflare", try_cloudflare), ("Hugging Face", try_huggingface), ("Gemini", try_gemini), ("Pollinations", try_pollinations)]
        elif engine_choice == "Hugging Face (FLUX / SDXL)":
            ordered_runners = [("Hugging Face", try_huggingface), ("Cloudflare", try_cloudflare), ("Gemini", try_gemini), ("Pollinations", try_pollinations)]
        else:
            ordered_runners = [("Gemini", try_gemini), ("Cloudflare", try_cloudflare), ("Hugging Face", try_huggingface), ("Pollinations", try_pollinations)]

        for label, runner in ordered_runners:
            try:
                img, src_name = runner()
                if caption_text and caption_text.strip() and caption_lang != "None":
                    img = draw_subtitle_on_frame(img, subtitle_text=caption_text.strip(), target_w=target_w, target_h=target_h, lang=caption_lang, position="Bottom", scale=0.85)
                return img, src_name, errors
            except Exception as exc:
                errors.append(f"{label}: {exc}")

    if valid_photos:
        p_path = valid_photos[scene_idx % len(valid_photos)]
        with Image.open(p_path) as p_im:
            composed = ensure_exact_size(p_im, target_w, target_h, bg_mode=bg_mode)
        if caption_text and caption_text.strip() and caption_lang != "None":
            composed = draw_subtitle_on_frame(composed, subtitle_text=caption_text.strip(), target_w=target_w, target_h=target_h, lang=caption_lang, position="Bottom", scale=0.85)
        return composed, "Uploaded Photo Base", errors

    if allow_local_bypass:
        base_sc = _render_story_animated_pil_frame(
            t=0.4,
            duration=4.0,
            target_w=target_w,
            target_h=target_h,
            scene_idx=scene_idx,
            story_context=arrangement_cmd or prompt_text,
            emotion_label=emotion_label,
            motion_mode="Zoom In",
            bg_mode=bg_mode,
        )
        if caption_text and caption_text.strip() and caption_lang != "None":
            base_sc = draw_subtitle_on_frame(base_sc, subtitle_text=caption_text.strip(), target_w=target_w, target_h=target_h, lang=caption_lang, position="Bottom", scale=0.85)
        return base_sc, "Story Animation Keyframe", errors

    return None, None, errors


def apply_clip_transition(clips, transition_name="Crossfade", trans_sec=0.4):
    if not clips:
        return None
    if len(clips) == 1 or transition_name == "None" or trans_sec <= 0.05:
        return concatenate_videoclips(clips, method="compose")

    positioned = []
    cur_start = 0.0
    for idx, clp in enumerate(clips):
        dur = max(float(clp.duration or 1.0), 0.2)
        t_dur = min(float(trans_sec), dur * 0.35)
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
    payload = {
        "version": "3.0-animated",
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
        "video_links_input": st.session_state.video_links_input,
        "bgm_link_input": st.session_state.bgm_link_input,
        "voice_links_input": st.session_state.voice_links_input,
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
    record_undo_step("Restore JSON Project")
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
    for sk in (
        "arrangement_command",
        "reel_wordings_input",
        "video_links_input",
        "bgm_link_input",
        "voice_links_input",
        "locked_character_prompt",
    ):
        if sk in data:
            st.session_state[sk] = str(data[sk])
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
        force_sync_scene_widget_states(st.session_state.scenes)


# --- TOP STUDIO BANNER + 3-STEP UNDO / REDO BAR ---
st.markdown(
    """
    <div class="studio-header">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="studio-pill">AI STUDIO PRO</span>
            <span style="font-weight:700;font-size:17px;">Story-Driven Animation, Movie & Reels Suite</span>
        </div>
        <div class="studio-sub">24 FPS Story Animation • Real Video & Photo Priority • Links, Voice & BGM Ducking</div>
    </div>
    """,
    unsafe_allow_html=True,
)

ur_col1, ur_col2, ur_col3 = st.columns([1.0, 1.0, 2.2])
with ur_col1:
    undo_count = len(st.session_state.undo_stack)
    if st.button(
        f"↩️ Undo ({undo_count}/{MAX_HISTORY_STEPS})",
        key="btn_undo_3",
        disabled=(undo_count == 0),
        use_container_width=True,
    ):
        if perform_undo():
            st.rerun()
with ur_col2:
    redo_count = len(st.session_state.redo_stack)
    if st.button(
        f"↪️ Redo ({redo_count}/{MAX_HISTORY_STEPS})",
        key="btn_redo_3",
        disabled=(redo_count == 0),
        use_container_width=True,
    ):
        if perform_redo():
            st.rerun()
with ur_col3:
    status_txt = st.session_state.last_history_label or "Ready — up to 3 Undo & 3 Redo steps available"
    st.caption(f"History: **{status_txt}** • Undo left: `{undo_count}` | Redo left: `{redo_count}`")

# --- SIDEBAR ---
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

        sc_cur_idx = (
            min(int(st.session_state.selected_clip_idx), len(st.session_state.scenes) - 1)
            if st.session_state.scenes
            else 0
        )
        cur_scene_obj = st.session_state.scenes[sc_cur_idx] if st.session_state.scenes else {}
        cur_story_ctx = (
            f"{cur_scene_obj.get('visual', '')} {cur_scene_obj.get('narration', '')}".strip()
            or st.session_state.locked_character_prompt
        )

        preview_caption_txt = ""
        if st.session_state.custom_title_text.strip():
            preview_caption_txt = st.session_state.custom_title_text.strip()
        elif st.session_state.scenes:
            if st.session_state.custom_title_lang == "Kannada":
                preview_caption_txt = (cur_scene_obj.get("kannada_narration") or "").strip()
            else:
                preview_caption_txt = (cur_scene_obj.get("narration") or "").strip()

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
            st.image(prev_img, caption=f"Scene {c_idx + 1} • {width}x{height} • {scene_fx_choice}", width=270)
        elif monitor_mode == "Rendered Clip" and avail_vids:
            v_idx = min(int(st.session_state.selected_clip_idx), len(avail_vids) - 1)
            st.video(avail_vids[v_idx])
        else:
            fallback_img = _render_story_animated_pil_frame(
                t=0.5,
                duration=4.0,
                target_w=width,
                target_h=height,
                scene_idx=sc_cur_idx,
                story_context=cur_story_ctx,
                emotion_label=cur_scene_obj.get("emotion", "Story Preview"),
                motion_mode=cur_scene_obj.get("motion", "Zoom In"),
                brightness=st.session_state.editor_brightness,
                contrast=st.session_state.editor_contrast,
                saturation=st.session_state.editor_saturation,
                blur_radius=st.session_state.editor_blur,
                video_fx=st.session_state.editor_video_fx,
                bg_mode=st.session_state.bg_fit_mode,
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
            st.image(fallback_img, caption=f"Story Preview ({width}x{height})", width=270)
        st.markdown("</div>", unsafe_allow_html=True)

    with stage_right:
        ctrl_tabs = st.tabs([
            "🎨 Looks & FX",
            "✂️ Trim & Split",
            "🔊 Sound FX",
            "✍️ Text & Fit",
            "📁 Media & Links",
        ])

        with ctrl_tabs[0]:
            mood_sel = st.selectbox("One-Tap Color Mood", list(MOOD_FILTERS.keys()), key="mood_sel_box")
            if st.button("Apply Color Mood", key="apply_mood_btn"):
                record_undo_step(f"Color Mood: {mood_sel}")
                m = MOOD_FILTERS[mood_sel]
                st.session_state.editor_brightness = m["b"]
                st.session_state.editor_contrast = m["c"]
                st.session_state.editor_saturation = m["s"]
                st.session_state.editor_video_fx = m["fx"]
                st.rerun()

            sl1, sl2 = st.columns(2)
            with sl1:
                b_val = st.slider("Brightness", 0.5, 1.6, float(st.session_state.editor_brightness), 0.05, key="sl_b")
                c_val = st.slider("Contrast", 0.5, 1.8, float(st.session_state.editor_contrast), 0.05, key="sl_c")
            with sl2:
                s_val = st.slider("Saturation", 0.0, 2.0, float(st.session_state.editor_saturation), 0.05, key="sl_s")
                blur_val = st.slider("Soft Blur", 0.0, 6.0, float(st.session_state.editor_blur), 0.25, key="sl_blur")

            if (
                abs(b_val - float(st.session_state.editor_brightness)) > 1e-4
                or abs(c_val - float(st.session_state.editor_contrast)) > 1e-4
                or abs(s_val - float(st.session_state.editor_saturation)) > 1e-4
                or abs(blur_val - float(st.session_state.editor_blur)) > 1e-4
            ):
                record_undo_step("Adjust Color & Blur")
                st.session_state.editor_brightness = b_val
                st.session_state.editor_contrast = c_val
                st.session_state.editor_saturation = s_val
                st.session_state.editor_blur = blur_val

            fx1, fx2, fx3 = st.columns(3)
            with fx1:
                vfx_val = st.selectbox(
                    "Video Effect",
                    VIDEO_FX_CHOICES,
                    index=VIDEO_FX_CHOICES.index(st.session_state.editor_video_fx),
                    key="sel_vfx",
                )
            with fx2:
                trans_val = st.selectbox(
                    "Transition",
                    ["None", "Crossfade", "Fade", "Slide Left", "Slide Right"],
                    index=["None", "Crossfade", "Fade", "Slide Left", "Slide Right"].index(
                        st.session_state.editor_transition
                    ),
                    key="sel_trans",
                )
            with fx3:
                tsec_val = st.slider(
                    "Trans (sec)", 0.1, 1.5, float(st.session_state.editor_transition_sec), 0.1, key="sl_tsec"
                )

            if (
                vfx_val != st.session_state.editor_video_fx
                or trans_val != st.session_state.editor_transition
                or abs(tsec_val - float(st.session_state.editor_transition_sec)) > 1e-4
            ):
                record_undo_step("Update Video FX / Transition")
                st.session_state.editor_video_fx = vfx_val
                st.session_state.editor_transition = trans_val
                st.session_state.editor_transition_sec = tsec_val

        with ctrl_tabs[1]:
            editable_vids = [p for p in st.session_state.scene_videos if os.path.exists(p)]
            if not editable_vids:
                st.info("Generate animated story clips in Tab 2 or upload/link video clips to trim or split.")
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
                            record_undo_step(f"Trim Clip {target_v_idx + 1}")
                            os.makedirs("edited_clips", exist_ok=True)
                            trimmed_out = os.path.join("edited_clips", f"trimmed_{target_v_idx}_{int(time.time())}.mp4")
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
                        record_undo_step(f"Split Clip {target_v_idx + 1}")
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
                record_undo_step(f"Sound FX Scene {sc_target + 1}")
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
            bg_fit_val = st.selectbox(
                "Canvas Framing Mode",
                ["Blur Canvas", "Crop Fill"],
                index=["Blur Canvas", "Crop Fill"].index(st.session_state.bg_fit_mode),
                key="bg_fit_box",
            )
            spd_val = st.select_slider(
                "Video Speed",
                options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                value=float(st.session_state.playback_speed),
                key="speed_slider",
            )
            tf_c1, tf_c2 = st.columns(2)
            with tf_c1:
                sub_pos_val = st.selectbox(
                    "Subtitle Position",
                    SUBTITLE_POSITIONS,
                    index=SUBTITLE_POSITIONS.index(st.session_state.subtitle_position),
                    key="sub_pos_box",
                )
            with tf_c2:
                sub_scale_val = st.slider(
                    "Subtitle Size Scale",
                    0.7,
                    1.5,
                    float(st.session_state.subtitle_scale),
                    0.05,
                    key="sub_scale_sl",
                )

            title_txt_val = st.text_input(
                "Hook / Title Overlay Text",
                value=st.session_state.custom_title_text,
                placeholder="Example: Ancient Yuga Chronicle!",
                key="title_overlay_input",
            )
            title_lang_val = st.selectbox(
                "Overlay / Preview Text Language",
                ["English", "Kannada"],
                index=["English", "Kannada"].index(st.session_state.custom_title_lang),
                key="title_overlay_lang",
            )

            if (
                bg_fit_val != st.session_state.bg_fit_mode
                or abs(spd_val - float(st.session_state.playback_speed)) > 1e-4
                or sub_pos_val != st.session_state.subtitle_position
                or abs(sub_scale_val - float(st.session_state.subtitle_scale)) > 1e-4
                or title_txt_val != st.session_state.custom_title_text
                or title_lang_val != st.session_state.custom_title_lang
            ):
                record_undo_step("Text & Framing Update")
                st.session_state.bg_fit_mode = bg_fit_val
                st.session_state.playback_speed = spd_val
                st.session_state.subtitle_position = sub_pos_val
                st.session_state.subtitle_scale = sub_scale_val
                st.session_state.custom_title_text = title_txt_val
                st.session_state.custom_title_lang = title_lang_val

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
                    record_undo_step("Add Photos")
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
                    record_undo_step("Add Videos")
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
                    record_undo_step("Load BGM Track")
                    os.makedirs("uploaded_studio_media", exist_ok=True)
                    af = a_files[0]
                    ext = os.path.splitext(af.name)[1].lower() or ".mp3"
                    bp = os.path.join("uploaded_studio_media", f"bgm_{int(time.time())}{ext}")
                    with open(bp, "wb") as f:
                        f.write(af.getbuffer())
                    st.session_state.bgm_audio_path = bp
                    st.success("Background music (BGM) track loaded.")
                    st.rerun()

            st.divider()
            st.markdown("**🔗 Or Load Video, Voiceover & BGM Directly From Links (URLs)**")
            lnk_v_col, lnk_bgm_col = st.columns(2)
            with lnk_v_col:
                v_links_val = st.text_area(
                    "Video Clip URLs (one URL per line)",
                    value=st.session_state.video_links_input,
                    height=75,
                    placeholder="https://example.com/clip1.mp4\nhttps://example.com/clip2.mp4",
                    key="media_tab_video_links",
                )
                st.session_state.video_links_input = v_links_val
                if st.button("🌐 Fetch Video Links", key="btn_fetch_vid_links_media", use_container_width=True):
                    urls = [ln.strip() for ln in v_links_val.splitlines() if ln.strip()]
                    if not urls:
                        st.warning("Paste at least one direct video URL first.")
                    else:
                        record_undo_step("Fetch Video Links")
                        fetched = []
                        errs = []
                        with st.spinner(f"Fetching {len(urls)} video link(s)..."):
                            for idx_l, u in enumerate(urls):
                                try:
                                    p_vid = download_media_from_url(u, media_kind="video", prefix=f"link_vid_{idx_l}")
                                    fetched.append(p_vid)
                                except Exception as exc:
                                    errs.append(f"Link {idx_l + 1}: {exc}")
                        if fetched:
                            st.session_state.uploaded_video_clips.extend(fetched)
                            st.session_state.scene_videos.extend(fetched)
                            st.success(f"Loaded {len(fetched)} video clip(s) from link(s).")
                        for e in errs:
                            st.error(e)
                        if fetched:
                            st.rerun()

            with lnk_bgm_col:
                bgm_url_val = st.text_input(
                    "Background Music / BGM Link (MP3 / WAV / M4A / OGG)",
                    value=st.session_state.bgm_link_input,
                    placeholder="https://example.com/background-music.mp3",
                    key="media_tab_bgm_link",
                )
                st.session_state.bgm_link_input = bgm_url_val
                if st.button("🎵 Load BGM From Link", key="btn_fetch_bgm_link_media", use_container_width=True):
                    if not bgm_url_val.strip():
                        st.warning("Paste a BGM audio link first.")
                    else:
                        record_undo_step("Load BGM From Link")
                        try:
                            with st.spinner("Downloading BGM track..."):
                                bp = download_media_from_url(bgm_url_val, media_kind="audio", prefix="link_bgm")
                            st.session_state.bgm_audio_path = bp
                            st.success(f"Loaded BGM from link: `{os.path.basename(bp)}`")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not fetch BGM link: {exc}")

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
    "2️⃣ Animated Story Clips & Media",
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
        key="rough_story_input",
    )
    num_scenes = st.slider(
        "Number of Scenes",
        2,
        10,
        max(2, min(10, len(st.session_state.scenes) or 4)),
        key="tab1_num_scenes",
    )

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
            record_undo_step("Generate Smart Scenes")
            st.session_state.scenes = build_fallback_scenes_from_rough_story(
                rough_story, num_scenes, st.session_state.locked_character_prompt
            )
            for idx, sc in enumerate(st.session_state.scenes):
                st.session_state.scene_fx_overrides[idx] = sc.get("suggested_fx", "None")
                if sc.get("suggested_sfx") and sc["suggested_sfx"] != "None":
                    p_auto_sfx = download_fx_audio(
                        sc["suggested_sfx"], PRESET_AUDIO_EFFECTS.get(sc["suggested_sfx"], "")
                    )
                    if p_auto_sfx:
                        st.session_state.scene_audio_fx[idx] = p_auto_sfx
            force_sync_scene_widget_states(st.session_state.scenes)
            st.success(f"Created {len(st.session_state.scenes)} story-driven scenes with animation & FX direction.")
            st.rerun()
        else:
            record_undo_step("Generate Smart Scenes")
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
            Create a {num_scenes}-scene animated story script obeying every user command for characters, gods, yugas, sun, planets, and backgrounds.{char_context}
            Return ONLY valid JSON as a list of {num_scenes} objects, no markdown.
            Each object must have:
            - "narration": clear English voiceover (1-2 sentences)
            - "kannada_narration": natural Kannada voiceover/caption (1-2 sentences)
            - "emotion": mood or expression
            - "motion": one of ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
            - "suggested_broll": 1-line animated shot/motion suggestion matching the story
            - "suggested_fx": one of ["None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"]
            - "visual": concrete animated visual description with character details, yuga/background, sun, and planets
            User command: {rough_story.strip()}
            """
            client = genai.Client(api_key=api_key.strip())
            parsed_scenes = None
            last_exc = None

            with st.spinner(f"Generating {num_scenes} AI-directed story scenes..."):
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
                force_sync_scene_widget_states(st.session_state.scenes)
                st.success(f"Generated {len(st.session_state.scenes)} AI-directed story scenes!")
                st.rerun()
            else:
                st.session_state.scenes = build_fallback_scenes_from_rough_story(
                    rough_story, num_scenes, st.session_state.locked_character_prompt
                )
                for idx, sc in enumerate(st.session_state.scenes):
                    st.session_state.scene_fx_overrides[idx] = sc.get("suggested_fx", "None")
                force_sync_scene_widget_states(st.session_state.scenes)
                st.warning(f"Used story-driven scene builder ({last_exc}).")
                st.rerun()

    if ai_sug_clicked:
        record_undo_step("AI Suggest B-Roll & FX")
        with st.spinner("Analyzing scenes for story motion, B-roll & video FX..."):
            suggested_list = ai_suggest_broll_and_fx_for_scenes(api_key, st.session_state.scenes)
            st.session_state.scenes = suggested_list
            for idx, sc in enumerate(suggested_list):
                st.session_state.scene_fx_overrides[idx] = sc.get("suggested_fx", "None")
                if sc.get("suggested_sfx") and sc["suggested_sfx"] != "None":
                    p_sfx = download_fx_audio(
                        sc["suggested_sfx"], PRESET_AUDIO_EFFECTS.get(sc["suggested_sfx"], "")
                    )
                    if p_sfx:
                        st.session_state.scene_audio_fx[idx] = p_sfx
            force_sync_scene_widget_states(suggested_list)
        st.success("Applied story animation, motion, video FX & sound FX suggestions across all scenes!")
        st.rerun()

    if tr_all_clicked:
        if not api_key or not api_key.strip():
            st.error("Add your Gemini API Key in the sidebar to translate with AI.")
        else:
            record_undo_step("Translate All to Kannada")
            en_list = [
                (
                    st.session_state.get(_scene_widget_key("nar", i), sc.get("narration", ""))
                    or f"Scene {i + 1}."
                ).strip()
                for i, sc in enumerate(st.session_state.scenes)
            ]
            with st.spinner("Translating scenes to Kannada..."):
                kn_list = translate_lines_to_kannada(api_key, en_list)
            for i, kn_val in enumerate(kn_list):
                st.session_state.scenes[i]["kannada_narration"] = kn_val
                st.session_state[_scene_widget_key("kn_script", i)] = kn_val
            st.success("Kannada script updated for all scenes.")
            st.rerun()

    if st.session_state.scenes:
        st.markdown(f"### Scene Script, Reorder, Animation & FX Cards ({len(st.session_state.scenes)} Scenes)")
        for i, scene in enumerate(st.session_state.scenes):
            with st.expander(f"Scene {i + 1} • {scene.get('emotion', 'Expressive')}", expanded=(i < 2)):
                mv1, mv2 = st.columns(2)
                with mv1:
                    if st.button(
                        "⬆️ Move Up",
                        key=f"mv_up_{i}_b{st.session_state.gen_batch_id}",
                        disabled=(i == 0),
                        use_container_width=True,
                    ):
                        swap_scenes_and_media(i, i - 1)
                        st.rerun()
                with mv2:
                    if st.button(
                        "⬇️ Move Down",
                        key=f"mv_dn_{i}_b{st.session_state.gen_batch_id}",
                        disabled=(i == len(st.session_state.scenes) - 1),
                        use_container_width=True,
                    ):
                        swap_scenes_and_media(i, i + 1)
                        st.rerun()

                c_left, c_right = st.columns(2)
                with c_left:
                    nar_v = st.text_area(
                        f"English Narration (Scene {i + 1})",
                        value=scene.get("narration", ""),
                        key=_scene_widget_key("nar", i),
                    )
                    emo_v = st.text_input(
                        f"Mood / Emotion (Scene {i + 1})",
                        value=scene.get("emotion", "Expressive"),
                        key=_scene_widget_key("emo", i),
                    )
                    mot_cur = st.session_state.get(
                        _scene_widget_key("mot", i), scene.get("motion", "Auto Alternate")
                    )
                    mot_v = st.selectbox(
                        f"Animation Motion (Scene {i + 1})",
                        MOTION_CHOICES,
                        index=MOTION_CHOICES.index(mot_cur) if mot_cur in MOTION_CHOICES else 0,
                        key=_scene_widget_key("mot", i),
                    )
                    fx_cur = st.session_state.scene_fx_overrides.get(i, scene.get("suggested_fx", "None"))
                    fx_v = st.selectbox(
                        f"Suggested / Active Video FX (Scene {i + 1})",
                        VIDEO_FX_CHOICES,
                        index=VIDEO_FX_CHOICES.index(fx_cur) if fx_cur in VIDEO_FX_CHOICES else 0,
                        key=_scene_widget_key("fx_sc", i),
                    )
                    broll_v = st.text_input(
                        f"Animated Shot / B-Roll Direction (Scene {i + 1})",
                        value=scene.get("suggested_broll", ""),
                        key=_scene_widget_key("broll", i),
                    )
                    vis_v = st.text_input(
                        f"Story Visual / Action / Setting (Scene {i + 1})",
                        value=scene.get("visual", ""),
                        key=_scene_widget_key("vis", i),
                    )

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
                        key=_scene_widget_key("kn_script", i),
                    )
                    scene["kannada_narration"] = kn_v
                    st.caption(f"Suggested Sound FX: **{scene.get('suggested_sfx', 'None')}**")

# --- TAB 2: ANIMATED STORY CLIPS & MEDIA ---
with tab2:
    st.markdown("### Build Smooth 24 FPS Story Animations From Your Story, Photos, Videos & Links")

    st.markdown("#### 1. Priority Media: Uploaded/Linked Videos & Photos")
    m_up1, m_up2 = st.columns(2)
    with m_up1:
        multi_photo_uploads = st.file_uploader(
            "Upload Reference / Story Photos",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="multi_ref_photos_tab2",
        )
        if st.button("➕ Save Uploaded Photos to Pool", key="btn_save_multi_photos_t2") and multi_photo_uploads:
            record_undo_step("Save Story Photos")
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
            "Upload Video Clips (Used Directly for Matching Scenes)",
            type=["mp4", "mov", "webm", "mkv"],
            accept_multiple_files=True,
            key="multi_reel_videos_tab2",
        )
        if st.button("➕ Add Uploaded Videos to Clips", key="btn_save_multi_vids_t2") and multi_vid_uploads:
            record_undo_step("Add Reel Videos")
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
            st.success(f"Added {len(saved_vids)} video clip(s).")
            st.rerun()

    t2_v_links = st.text_area(
        "Direct Video Clip Links (One video URL per line — MP4 / MOV / WEBM / MKV):",
        value=st.session_state.video_links_input,
        height=75,
        placeholder="https://example.com/scene1.mp4\nhttps://example.com/scene2.mp4",
        key="tab2_video_links_area",
    )
    st.session_state.video_links_input = t2_v_links
    if st.button("🌐 Download & Add Video Links to Clips", key="btn_add_vid_links_t2"):
        urls = [ln.strip() for ln in t2_v_links.splitlines() if ln.strip()]
        if not urls:
            st.warning("Enter at least one video URL first.")
        else:
            record_undo_step("Add Video Links")
            added = []
            errs = []
            with st.spinner(f"Fetching {len(urls)} video link(s)..."):
                for idx_l, u in enumerate(urls):
                    try:
                        vp = download_media_from_url(u, media_kind="video", prefix=f"t2_vid_{idx_l}")
                        added.append(vp)
                    except Exception as exc:
                        errs.append(f"URL {idx_l + 1}: {exc}")
            if added:
                st.session_state.uploaded_video_clips.extend(added)
                st.session_state.scene_videos.extend(added)
                st.success(f"Added {len(added)} linked video clip(s).")
            for e in errs:
                st.error(e)
            if added:
                st.rerun()

    existing_photos = [p for p in st.session_state.uploaded_photo_paths if os.path.exists(p)]
    if existing_photos:
        st.caption(f"Saved Photo Pool ({len(existing_photos)} photo(s)):")
        thumb_cols = st.columns(min(6, len(existing_photos)))
        for idx_th, p_path in enumerate(existing_photos[:6]):
            with thumb_cols[idx_th]:
                st.image(p_path, caption=f"Photo {idx_th + 1}", width=110)

    st.markdown("#### 2. Story Pacing, Animation Style & Custom Wordings")
    arr_col, word_col = st.columns(2)
    with arr_col:
        arr_input = st.text_area(
            "Story Animation & Motion Direction:",
            value=st.session_state.arrangement_command,
            height=90,
            placeholder="Example: Smooth animated zoom and pan following each story beat with vivid lighting",
            key="arr_command_input",
        )
    with word_col:
        word_input = st.text_area(
            "Custom Script / Wordings on Clips (One line per scene, or leave blank to use scene script):",
            value=st.session_state.reel_wordings_input,
            height=90,
            placeholder="Line 1: Sacred Dawn Begins\nLine 2: Rising Power & Light\nLine 3: Timeless Yuga Moment\nLine 4: Grand Finale",
            key="reel_wordings_area",
        )

    if arr_input != st.session_state.arrangement_command or word_input != st.session_state.reel_wordings_input:
        record_undo_step("Update Animation / Wordings")
        st.session_state.arrangement_command = arr_input
        st.session_state.reel_wordings_input = word_input

    art_style = st.selectbox(
        "Visual Style (When Using AI Image Engine)",
        ART_STYLE_OPTIONS,
        key="visual_style_selectbox",
    )

    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        default_scene_n = max(2, min(10, max(len(st.session_state.scenes) or 4, len(existing_photos) or 4)))
        scene_count_slider = st.slider("Scene / Clip Count", 2, 10, default_scene_n, key="tab2_scene_count")
    with opt2:
        per_scene_sec = st.slider("Clip Duration (sec)", 2.0, 10.0, 4.0, 0.5, key="tab2_clip_dur")
    with opt3:
        tab2_sub_lang = st.selectbox(
            "Subtitles Language", ["None", "English", "Kannada"], index=0, key="tab2_sub_lang_box"
        )

    preserve_existing_vids = st.checkbox(
        "Keep uploaded/linked video clips for their scene slots instead of overwriting with newly animated images",
        value=True,
        key="tab2_keep_vids_chk",
    )

    if st.button("🎬 Generate Animated Story Clips", type="primary", key="btn_gen_all_scene_videos"):
        record_undo_step("Generate Animated Story Clips")
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
            sc["narration"] = st.session_state.get(_scene_widget_key("nar", idx), sc.get("narration", ""))
            sc["kannada_narration"] = st.session_state.get(
                _scene_widget_key("kn_script", idx), sc.get("kannada_narration", "")
            )
            sc["visual"] = st.session_state.get(_scene_widget_key("vis", idx), sc.get("visual", ""))
            sc["emotion"] = st.session_state.get(_scene_widget_key("emo", idx), sc.get("emotion", "Expressive"))
            sc["motion"] = (
                directives["motion_mode"]
                if st.session_state.arrangement_command.strip()
                and st.session_state.get(_scene_widget_key("mot", idx), "Auto Alternate") == "Auto Alternate"
                else st.session_state.get(_scene_widget_key("mot", idx), sc.get("motion", "Auto Alternate"))
            )
            sc["suggested_broll"] = st.session_state.get(
                _scene_widget_key("broll", idx), sc.get("suggested_broll", "")
            )

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

        prior_vids = [p for p in st.session_state.scene_videos if os.path.exists(p)] if preserve_existing_vids else []
        st.session_state.gen_batch_id += 1
        batch_id = st.session_state.gen_batch_id
        st.session_state.images = []
        new_scene_videos = []
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
            nar_note = _extract_visual_command(scene.get("narration", ""))
            arr_note = _extract_visual_command(st.session_state.arrangement_command)
            story_ctx = " | ".join(x for x in [nar_note, vis_note, broll_note, char_cmd] if x)
            sc_caption = scene_subtitles[idx] if idx < len(scene_subtitles) else ""
            sc_fx = st.session_state.scene_fx_overrides.get(idx, st.session_state.editor_video_fx)
            b_adj = st.session_state.editor_brightness + (0.03 if directives["pop_boost"] else 0.0)
            c_adj = st.session_state.editor_contrast + (0.04 if directives["pop_boost"] else 0.0)
            s_adj = st.session_state.editor_saturation + (0.05 if directives["pop_boost"] else 0.0)

            # Priority 1: Use existing uploaded/linked video clip for this scene if enabled
            if preserve_existing_vids and idx < len(prior_vids) and os.path.exists(prior_vids[idx]):
                raw_vid = prior_vids[idx]
                out_v = os.path.join("scene_videos", f"scene_{idx}_vid_b{batch_id}.mp4")
                c_up, h_up = build_uploaded_video_clip(
                    vid_path=raw_vid,
                    target_w=width,
                    target_h=height,
                    clip_dur=per_scene_sec,
                    subtitle_text=sc_caption,
                    subtitle_lang=tab2_sub_lang,
                    subtitle_pos=st.session_state.subtitle_position,
                    subtitle_scale=st.session_state.subtitle_scale,
                    brightness=b_adj,
                    contrast=c_adj,
                    saturation=s_adj,
                    blur_radius=st.session_state.editor_blur,
                    video_fx=sc_fx,
                    bg_mode=st.session_state.bg_fit_mode,
                )
                try:
                    c_up.write_videofile(
                        out_v,
                        fps=ANIMATION_FPS,
                        codec="libx264",
                        audio_codec="aac" if c_up.audio is not None else None,
                        audio=c_up.audio is not None,
                        ffmpeg_params=["-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
                        logger=None,
                    )
                finally:
                    try:
                        c_up.close()
                    except Exception:
                        pass
                    for hh in h_up:
                        try:
                            hh.close()
                        except Exception:
                            pass
                new_scene_videos.append(out_v)
                st.session_state.images.append(existing_photos[idx % len(existing_photos)] if existing_photos else "")
                prog.progress((idx + 1) / len(active_scenes))
                continue

            # Priority 2: AI image (when API keys exist) or user photo / story animation
            full_prompt = (
                f"{art_style}. {cast_guard} "
                f"Story beat: {nar_note}. "
                f"Visual & setting: {vis_note}. "
                f"Motion/B-roll: {broll_note}. "
                f"Direction: {arr_note}."
            ).strip()

            base_img_path = ""
            try:
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
                    arrangement_cmd=story_ctx,
                    caption_text="",
                    caption_lang="None",
                    allow_local_bypass=False,
                    emotion_label=scene.get("emotion", "Expressive"),
                    seed_offset=batch_id * 100,
                    bg_mode=st.session_state.bg_fit_mode,
                )
                if pil_img is not None:
                    base_img_path = f"scene_{idx}_b{batch_id}.png"
                    ensure_exact_size(pil_img, width, height, bg_mode=st.session_state.bg_fit_mode).save(
                        base_img_path, format="PNG"
                    )
            except Exception:
                base_img_path = ""

            if not base_img_path and existing_photos:
                base_img_path = existing_photos[idx % len(existing_photos)]

            st.session_state.images.append(base_img_path)

            vid_path = os.path.join("scene_videos", f"scene_{idx}_b{batch_id}.mp4")
            render_single_scene_video(
                img_path=base_img_path,
                out_path=vid_path,
                target_w=width,
                target_h=height,
                duration=per_scene_sec,
                motion_mode=scene.get("motion", "Auto Alternate"),
                scene_idx=idx,
                story_context=story_ctx,
                emotion_label=scene.get("emotion", "Expressive"),
                extra_photo_paths=existing_photos,
                subtitle_text=sc_caption,
                subtitle_lang=tab2_sub_lang,
                subtitle_pos=st.session_state.subtitle_position,
                subtitle_scale=st.session_state.subtitle_scale,
                brightness=b_adj,
                contrast=c_adj,
                saturation=s_adj,
                blur_radius=st.session_state.editor_blur,
                video_fx=sc_fx,
                bg_mode=st.session_state.bg_fit_mode,
            )
            new_scene_videos.append(vid_path)
            prog.progress((idx + 1) / len(active_scenes))

        st.session_state.scene_videos = new_scene_videos
        st.success(f"Generated {len(st.session_state.scene_videos)} smooth 24 FPS story-driven clips at {width}x{height}!")

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
        v_choice = st.selectbox("AI Narrator Voice", list(voice_map.keys()), key="tab3_voice_choice")
        v_id, v_lang = voice_map[v_choice]
    with vc2:
        s_lang = st.selectbox(
            "Script Language", ["English", "Kannada"], index=1 if v_lang == "Kannada" else 0, key="tab3_script_lang"
        )

    lines_to_speak = get_script_lines_for_language(st.session_state.scenes, script_lang=s_lang, gemini_key=api_key)

    async def synth_tts(txt, v_name, out_file):
        comm = edge_tts.Communicate(txt, v_name)
        await comm.save(out_file)

    async def synth_tts_batch(lines, v_name):
        out_paths = [f"voice_{idx}.mp3" for idx in range(len(lines))]
        tasks = [synth_tts(line, v_name, out_paths[idx]) for idx, line in enumerate(lines)]
        await asyncio.gather(*tasks)
        return out_paths

    if st.button("🔊 Generate Voiceover for All Scenes", type="primary", key="btn_gen_voiceover_all"):
        record_undo_step("Generate Voiceover")
        try:
            with st.spinner(f"Synthesizing {len(lines_to_speak)} voiceovers in parallel..."):
                generated_paths = run_async(synth_tts_batch(lines_to_speak, v_id))
            st.session_state.audio_files = generated_paths
            st.success(f"Generated {len(st.session_state.audio_files)} voiceover clip(s) in parallel.")
        except Exception as exc:
            st.error(f"Voiceover generation error: {exc}")

    st.divider()
    st.markdown("#### 🔗 Load Voiceover or Background Music (BGM) From Links")
    au_lnk1, au_lnk2 = st.columns(2)
    with au_lnk1:
        t3_voice_links = st.text_area(
            "Voiceover Audio Links (One URL per line — Scene 1, Scene 2, ...):",
            value=st.session_state.voice_links_input,
            height=75,
            placeholder="https://example.com/vo-scene1.mp3\nhttps://example.com/vo-scene2.mp3",
            key="tab3_voice_links_area",
        )
        st.session_state.voice_links_input = t3_voice_links
        if st.button("🎙️ Fetch Voiceover Links", key="btn_fetch_voice_links_t3", use_container_width=True):
            v_urls = [ln.strip() for ln in t3_voice_links.splitlines() if ln.strip()]
            if not v_urls:
                st.warning("Paste at least one voiceover URL first.")
            else:
                record_undo_step("Fetch Voiceover Links")
                loaded_vos = []
                errs = []
                with st.spinner(f"Fetching {len(v_urls)} voiceover track(s)..."):
                    for idx_v, vu in enumerate(v_urls):
                        try:
                            vp = download_media_from_url(vu, media_kind="audio", prefix=f"t3_vo_{idx_v}")
                            loaded_vos.append(vp)
                        except Exception as exc:
                            errs.append(f"VO {idx_v + 1}: {exc}")
                if loaded_vos:
                    st.session_state.audio_files = loaded_vos
                    st.success(f"Assigned {len(loaded_vos)} voiceover track(s) by scene order.")
                for e in errs:
                    st.error(e)

    with au_lnk2:
        t3_bgm_link = st.text_input(
            "Background Music / BGM Link (MP3 / WAV / M4A / OGG):",
            value=st.session_state.bgm_link_input,
            placeholder="https://example.com/bgm-score.mp3",
            key="tab3_bgm_link_input",
        )
        st.session_state.bgm_link_input = t3_bgm_link
        if st.button("🎶 Fetch BGM Link", key="btn_fetch_bgm_link_t3", use_container_width=True):
            if not t3_bgm_link.strip():
                st.warning("Paste a BGM URL first.")
            else:
                record_undo_step("Fetch BGM Link")
                try:
                    with st.spinner("Downloading BGM link..."):
                        bp = download_media_from_url(t3_bgm_link, media_kind="audio", prefix="t3_bgm")
                    st.session_state.bgm_audio_path = bp
                    st.success(f"Set BGM track to `{os.path.basename(bp)}`.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not fetch BGM link: {exc}")

    st.divider()
    st.markdown("#### Mix Levels & Auto-Ducking (Voice, Background Music & Sound FX)")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.session_state.voice_volume = st.slider(
            "Voice Volume", 0.0, 1.5, float(st.session_state.voice_volume), 0.05, key="sl_voice_vol"
        )
    with m2:
        st.session_state.bgm_volume = st.slider(
            "Normal BGM Volume", 0.0, 0.8, float(st.session_state.bgm_volume), 0.02, key="sl_bgm_vol"
        )
    with m3:
        st.session_state.sfx_volume = st.slider(
            "Sound FX Volume", 0.0, 1.5, float(st.session_state.sfx_volume), 0.05, key="sl_sfx_vol"
        )

    d1, d2 = st.columns(2)
    with d1:
        st.session_state.bgm_duck_enabled = st.checkbox(
            "Auto-duck BGM volume while voiceover speaks",
            value=bool(st.session_state.bgm_duck_enabled),
            key="chk_bgm_duck",
        )
    with d2:
        st.session_state.bgm_duck_volume = st.slider(
            "Ducked BGM Volume",
            0.0,
            0.4,
            float(st.session_state.bgm_duck_volume),
            0.02,
            disabled=not st.session_state.bgm_duck_enabled,
            key="sl_bgm_duck_vol",
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
            "Fallback Animation Motion",
            MOTION_CHOICES,
            key="final_mot_box",
        )

    st.caption(
        f"Format: `{st.session_state.format_preset}` (`{width}x{height}`) • "
        f"Speed: `{st.session_state.playback_speed}x` • "
        f"Subtitles: `{final_sub_lang}` (`{st.session_state.subtitle_position}`, `{st.session_state.subtitle_scale}x`) • "
        f"BGM Ducking: `{'On' if st.session_state.bgm_duck_enabled else 'Off'}`"
    )

    if st.button("🎬 Render Final MP4 Video", type="primary", key="btn_render_final_mp4"):
        has_media = bool(st.session_state.scene_videos or st.session_state.images or st.session_state.scenes)
        if not has_media:
            st.error("Add a story or generate/upload clips first.")
        else:
            clips = []
            src_vids = []
            aud_list = []
            bgm_handles = []
            final_vid = None
            out_mp4 = f"final_studio_b{st.session_state.gen_batch_id}_{int(time.time())}.mp4"

            try:
                total_slots = max(
                    len(st.session_state.scene_videos),
                    len(st.session_state.images),
                    len(st.session_state.scenes),
                )
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
                char_cmd = _extract_visual_command(st.session_state.locked_character_prompt)
                pool_photos = [p for p in st.session_state.uploaded_photo_paths if os.path.exists(p)]

                with st.spinner("Rendering your final animated video with story motion, subtitles & audio ducking..."):
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

                        sc_obj = st.session_state.scenes[i] if i < len(st.session_state.scenes) else {}
                        sc_fx = st.session_state.scene_fx_overrides.get(i, st.session_state.editor_video_fx)
                        story_ctx = " | ".join(
                            x
                            for x in [
                                _extract_visual_command(sc_obj.get("narration", "")),
                                _extract_visual_command(sc_obj.get("visual", "")),
                                _extract_visual_command(sc_obj.get("suggested_broll", "")),
                                char_cmd,
                            ]
                            if x
                        )

                        if i < len(st.session_state.scene_videos) and os.path.exists(st.session_state.scene_videos[i]):
                            c_item, v_handles = build_uploaded_video_clip(
                                vid_path=st.session_state.scene_videos[i],
                                target_w=width,
                                target_h=height,
                                clip_dur=clip_dur,
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
                            img_p = (
                                st.session_state.images[i]
                                if i < len(st.session_state.images) and os.path.exists(st.session_state.images[i])
                                else (pool_photos[i % len(pool_photos)] if pool_photos else "")
                            )
                            c_item = build_animated_story_clip(
                                target_w=width,
                                target_h=height,
                                duration=clip_dur,
                                scene_idx=i,
                                story_context=story_ctx,
                                emotion_label=sc_obj.get("emotion", "Expressive"),
                                motion_mode=sc_obj.get("motion", final_motion_mode),
                                base_img_path=img_p,
                                extra_photo_paths=pool_photos,
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

                        if mixed_parts:
                            trimmed = [
                                apply_duration(mp, min(clip_dur, float(mp.duration or clip_dur)))
                                for mp in mixed_parts
                            ]
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

                    if (
                        st.session_state.bgm_audio_path
                        and os.path.exists(st.session_state.bgm_audio_path)
                        and st.session_state.bgm_volume > 0
                    ):
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
                            final_vid = apply_audio(
                                final_vid, CompositeAudioClip([final_vid.audio, bgm_mixed])
                            )
                        else:
                            final_vid = apply_audio(final_vid, bgm_mixed)

                    final_vid.write_videofile(
                        out_mp4,
                        fps=ANIMATION_FPS,
                        codec="libx264",
                        audio_codec="aac",
                        ffmpeg_params=["-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
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
                        key=f"dl_final_mp4_{st.session_state.gen_batch_id}",
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
