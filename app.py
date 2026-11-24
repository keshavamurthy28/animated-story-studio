import asyncio
import base64
import copy
import json
import math
import os
import random
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
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.VideoClip import VideoClip as DynamicVideoClip

    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.VideoClip import VideoClip as DynamicVideoClip

    MOVIEPY_V2 = True

st.set_page_config(
    page_title="AI Cinema & Reels Studio Pro (Full Edition)",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 15px; }
    .stApp { background-color: #0b0f17; color: #f0f6fc; }
    .studio-header {
        display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
        gap: 10px; background: linear-gradient(90deg, #131926 0%, #1d263b 100%);
        border: 1px solid #30363d; border-radius: 14px; padding: 14px 18px; margin-bottom: 14px;
    }
    .studio-pill {
        background: linear-gradient(90deg, #ff2e63, #ff6b35); color: #ffffff;
        font-weight: 700; font-size: 12px; padding: 4px 12px; border-radius: 999px;
    }
    .studio-sub { color: #9da7b3; font-size: 13px; }
    .stButton > button { border-radius: 10px; padding: 0.55rem 0.95rem; font-weight: 600; min-height: 42px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- PRESETS & CONSTANTS ---
FORMAT_PRESETS = {
    "9:16 Reels / Shorts / Vertical Drama": (720, 1280, "9:16"),
    "16:9 Movie / YouTube": (1280, 720, "16:9"),
    "1:1 Square Post": (1080, 1080, "1:1"),
    "4:5 Portrait Feed": (1080, 1350, "4:5"),
}

ANIMATION_STYLES = {
    "Photorealistic AI Cinema Drama (Like Vertical Series)": (
        "Photorealistic 35mm cinema film still, anamorphic shallow depth of field, moody dramatic studio lighting, "
        "sharp facial detail, realistic skin texture, luxury film set, 8k raw cinema look, no text, no watermarks"
    ),
    "3D Pixar / Disney Feature Film": (
        "3D animated feature film style like Pixar and DreamWorks, expressive faces, "
        "volumetric global illumination, rich textures, dynamic action camera, 8k render, no text"
    ),
    "2D Warm Illustrated Reel": (
        "Warm modern 2D animated illustration style, expressive cartoon characters with clean dark outlines, "
        "soft golden ambient lighting, cozy expressive faces, clean digital art, no text"
    ),
    "Epic 3D Unreal Engine 5 Mythological": (
        "Unreal Engine 5 cinematic 3D animation, hyper-detailed armor and divine aura, "
        "dramatic rim lighting, volumetric fog, particles, epic movie framing, no text"
    ),
    "High-Budget Anime (Ufotable / Makoto Shinkai)": (
        "High-budget theatrical anime animation, vibrant glowing effects, dynamic motion lines, "
        "detailed painted backgrounds, expressive character acting, dramatic sky, no text"
    ),
}

VIDEO_FX_CHOICES = [
    "None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"
]
MOTION_CHOICES = ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
SUBTITLE_POSITIONS = ["Bottom", "Center", "Top"]
ANIMATION_FPS = 24
MAX_HISTORY_STEPS = 5

TTS_VOICES = {
    "English (Male - Deep Boss / Storyteller)": "en-US-ChristopherNeural",
    "English (Male - Young / Guard)": "en-US-GuyNeural",
    "English (Female - Warm / Lead)": "en-US-AriaNeural",
    "English (Female - Dramatic)": "en-US-JennyNeural",
    "English India (Male - Prabhat)": "en-IN-PrabhatNeural",
    "English India (Female - Neerja)": "en-IN-NeerjaNeural",
    "Kannada (Male - Gagan)": "kn-IN-GaganNeural",
    "Kannada (Female - Sapna)": "kn-IN-SapnaNeural",
    "Hindi (Male - Madhur)": "hi-IN-MadhurNeural",
    "Hindi (Female - Swara)": "hi-IN-SwaraNeural",
}

PRESET_AUDIO_EFFECTS = {
    "None": "",
    "⚡ Thunder Strike": "https://assets.mixkit.co/active_storage/sfx/1271/1271-preview.mp3",
    "🔥 Lighter / Fire Flicker": "https://assets.mixkit.co/active_storage/sfx/2435/2435-preview.mp3",
    "💨 Wind / Whoosh": "https://assets.mixkit.co/active_storage/sfx/2673/2673-preview.mp3",
    "🌊 Ocean / River Waves": "https://assets.mixkit.co/active_storage/sfx/1188/1188-preview.mp3",
    "🚪 Heavy Door / Footsteps": "https://assets.mixkit.co/active_storage/sfx/2578/2578-preview.mp3",
}

DEFAULT_CAST = [
    {
        "name": "Don Julian (Lead Male)",
        "appearance": "Handsome 30-year-old Italian man with slicked-back dark wavy hair, sharp jawline, intense dark eyes, trimmed beard, wearing an all-black tailored suit and black silk shirt",
        "voice": "English (Male - Deep Boss / Storyteller)",
        "ref_photo": "",
    },
    {
        "name": "Marco (Bodyguard)",
        "appearance": "34-year-old loyal bodyguard with short dark hair, clean-shaven, wearing a formal black suit, crisp white shirt, and black tie",
        "voice": "English (Male - Young / Guard)",
        "ref_photo": "",
    },
    {
        "name": "Elena (Female Lead)",
        "appearance": "26-year-old graceful woman with long dark wavy hair and bangs, expressive dark eyes, wearing an elegant black evening dress",
        "voice": "English (Female - Warm / Lead)",
        "ref_photo": "",
    },
]

DEFAULT_SCENES = [
    {
        "speaker": "Don Julian (Lead Male)",
        "narration": "Keep your voice down, and the truth will come right out.",
        "kannada_narration": "ನಿನ್ನ ಧ್ವನಿಯನ್ನು ತಗ್ಗಿಸು, ಸತ್ಯ ತಾನಾಗಿಯೇ ಹೊರಬರುತ್ತದೆ.",
        "emotion": "Cold & Commanding",
        "motion": "Zoom In",
        "action_motion": "Close-up shot, man slowly lighting a cigar with a silver lighter, subtle smoke rising, intense eye contact, slow cinematic push-in",
        "suggested_fx": "Cinematic Vignette",
        "suggested_sfx": "🔥 Lighter / Fire Flicker",
        "visual": "Close-up portrait of Don Julian sitting in a dark mahogany library lighting a cigar with a metallic lighter, warm flame illuminating his face, blurred bookshelves behind",
    },
    {
        "speaker": "Marco (Bodyguard)",
        "narration": "Sir, the school called. Young Master Nino skipped class again!",
        "kannada_narration": "ಸರ್, ಶಾಲೆಯಿಂದ ಕರೆ ಬಂದಿತ್ತು. ಯಂಗ್ ಮಾಸ್ಟರ್ ನಿನೋ ಮತ್ತೆ ತರಗತಿಗೆ ಹೋಗಿಲ್ಲ!",
        "emotion": "Urgent & Nervous",
        "motion": "Pan Right",
        "action_motion": "Bodyguard rushing through tall carved wooden double doors into a luxury study, speaking urgently with worried expression",
        "suggested_fx": "None",
        "suggested_sfx": "🚪 Heavy Door / Footsteps",
        "visual": "Medium shot of Marco in a black suit and white shirt rushing through large wooden doors into a dimly lit classic mansion study, urgent expression",
    },
    {
        "speaker": "Don Julian (Lead Male)",
        "narration": "He must have gone to the casino again. Take some men and bring him back to me.",
        "kannada_narration": "ಅವನು ಮತ್ತೆ ಕ್ಯಾಸಿನೊಗೆ ಹೋಗಿರಬೇಕು. ಹುಡುಗರನ್ನು ಕರೆದುಕೊಂಡು ಹೋಗಿ ಅವನನ್ನು ವಾಪಸ್ ಕರೆತಾ.",
        "emotion": "Furious & Dominant",
        "motion": "Zoom In",
        "action_motion": "Man leaning forward over a wooden desk holding a cigar, speaking firmly with narrowed eyes, dramatic side lighting",
        "suggested_fx": "Zoom Punch",
        "suggested_sfx": "None",
        "visual": "Medium close-up of Don Julian seated behind a heavy mahogany desk holding a cigar, stern expression, silver pistol resting on the desk in foreground",
    },
    {
        "speaker": "Marco (Bodyguard)",
        "narration": "The last men who went after him were thrown out on the street. He warned us not to bother him.",
        "kannada_narration": "ಹಿಂದೆ ಅವನನ್ನು ಹಿಡಿಯಲು ಹೋದವರನ್ನು ಬೀದಿಗೆ ಎಸೆಯಲಾಗಿತ್ತು. ತನ್ನ ತಂಟೆಗೆ ಬರಬೇಡಿ ಎಂದು ಎಚ್ಚರಿಸಿದ್ದಾನೆ.",
        "emotion": "Fearful & Hesitant",
        "motion": "Zoom Out",
        "action_motion": "Bodyguard swallowing nervously, wide worried eyes, speaking hesitantly in close-up portrait",
        "suggested_fx": "None",
        "suggested_sfx": "None",
        "visual": "Close-up portrait of Marco in black suit and tie looking anxious and hesitant while reporting news in a moody warm-lit study",
    },
]


def _get_secret(key_name: str) -> str:
    try:
        if key_name in st.secrets:
            return str(st.secrets[key_name] or "")
    except Exception:
        pass
    return os.environ.get(key_name, "")


# --- SESSION STATE INIT ---
for key, default in [
    ("cast", copy.deepcopy(DEFAULT_CAST)),
    ("scenes", copy.deepcopy(DEFAULT_SCENES)),
    ("images", []),
    ("scene_videos", []),
    ("uploaded_photo_paths", []),
    ("audio_files", []),
    ("bgm_audio_path", ""),
    ("bgm_volume", 0.18),
    ("bgm_duck_enabled", True),
    ("bgm_duck_volume", 0.06),
    ("voice_volume", 1.0),
    ("sfx_volume", 0.70),
    ("lock_character_enabled", True),
    ("animation_style_name", "Photorealistic AI Cinema Drama (Like Vertical Series)"),
    ("gen_batch_id", 0),
    ("format_preset", "9:16 Reels / Shorts / Vertical Drama"),
    ("bg_fit_mode", "Crop Fill"),
    ("top_hook_enabled", False),
    ("top_hook_text", ""),
    ("top_hook_highlights", ""),
    ("subtitle_lang", "English"),
    ("subtitle_position", "Bottom"),
    ("subtitle_scale", 0.90),
    ("editor_brightness", 1.0),
    ("editor_contrast", 1.06),
    ("editor_saturation", 1.08),
    ("editor_blur", 0.0),
    ("editor_video_fx", "None"),
    ("scene_audio_fx", {}),
    ("scene_fx_overrides", {}),
    ("undo_stack", []),
    ("redo_stack", []),
    ("last_history_label", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- HISTORY & WIDGET SYNC ---
def _scene_widget_key(prefix, idx):
    return f"{prefix}_{idx}_b{st.session_state.gen_batch_id}"


def force_sync_scene_widget_states(scenes_list):
    for idx, sc in enumerate(scenes_list):
        st.session_state[_scene_widget_key("nar", idx)] = sc.get("narration", "")
        st.session_state[_scene_widget_key("kn_script", idx)] = sc.get("kannada_narration", "")
        st.session_state[_scene_widget_key("vis", idx)] = sc.get("visual", "")
        st.session_state[_scene_widget_key("act", idx)] = sc.get("action_motion", "")
        st.session_state[_scene_widget_key("emo", idx)] = sc.get("emotion", "Dramatic")
        st.session_state[_scene_widget_key("mot", idx)] = sc.get("motion", "Zoom In")


def record_undo_step(action_label="Edit"):
    snap = {
        "label": action_label,
        "cast": copy.deepcopy(st.session_state.cast),
        "scenes": copy.deepcopy(st.session_state.scenes),
        "images": list(st.session_state.images),
        "scene_videos": list(st.session_state.scene_videos),
        "audio_files": list(st.session_state.audio_files),
    }
    u_stack = list(st.session_state.undo_stack)
    u_stack.append(snap)
    st.session_state.undo_stack = u_stack[-MAX_HISTORY_STEPS:]
    st.session_state.redo_stack = []
    st.session_state.last_history_label = action_label


def perform_undo():
    if not st.session_state.undo_stack:
        return False
    snap = st.session_state.undo_stack.pop()
    st.session_state.redo_stack.append({
        "label": "Redo",
        "cast": copy.deepcopy(st.session_state.cast),
        "scenes": copy.deepcopy(st.session_state.scenes),
        "images": list(st.session_state.images),
        "scene_videos": list(st.session_state.scene_videos),
        "audio_files": list(st.session_state.audio_files),
    })
    for k in ("cast", "scenes", "images", "scene_videos", "audio_files"):
        st.session_state[k] = copy.deepcopy(snap[k])
    force_sync_scene_widget_states(st.session_state.scenes)
    st.session_state.last_history_label = f"Undid: {snap['label']}"
    return True


def perform_redo():
    if not st.session_state.redo_stack:
        return False
    snap = st.session_state.redo_stack.pop()
    for k in ("cast", "scenes", "images", "scene_videos", "audio_files"):
        st.session_state[k] = copy.deepcopy(snap[k])
    force_sync_scene_widget_states(st.session_state.scenes)
    st.session_state.last_history_label = "Redid change"
    return True


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


# --- MOVIEPY COMPATIBILITY ---
def apply_duration(clip, duration):
    return clip.with_duration(duration) if MOVIEPY_V2 else clip.set_duration(duration)


def apply_audio(clip, audio):
    return clip.with_audio(audio) if MOVIEPY_V2 else clip.set_audio(audio)


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


def loop_clip_safe(clip, target_duration):
    cur = max(0.1, float(clip.duration or 0.1))
    if target_duration <= cur:
        return apply_duration(subclip_safe(clip, 0.0, target_duration), target_duration)
    if hasattr(clip, "loop"):
        return apply_duration(clip.loop(duration=target_duration), target_duration)
    reps = int(math.ceil(target_duration / cur))
    return apply_duration(concatenate_videoclips([clip] * reps), target_duration)


# --- IMAGE PROCESSING & COLOR GRADING ---
def ensure_exact_size(pil_img, target_w, target_h, bg_mode="Crop Fill"):
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    rgb = pil_img.convert("RGB")
    if rgb.size == (target_w, target_h):
        return rgb
    if bg_mode == "Blur Canvas":
        bg_scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
        bg = rgb.resize((int(rgb.width * bg_scale), int(rgb.height * bg_scale)), resampling)
        bx = max(0, (bg.width - target_w) // 2)
        by = max(0, (bg.height - target_h) // 2)
        bg = bg.crop((bx, by, bx + target_w, by + target_h)).filter(ImageFilter.GaussianBlur(radius=16))
        bg = ImageEnhance.Brightness(bg).enhance(0.55)
        fg_scale = min(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
        fg = rgb.resize((max(1, int(rgb.width * fg_scale)), max(1, int(rgb.height * fg_scale))), resampling)
        bg.paste(fg, ((target_w - fg.width) // 2, (target_h - fg.height) // 2))
        return bg

    scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
    nw = max(target_w, int(round(rgb.width * scale)))
    nh = max(target_h, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), resampling)
    left = max(0, (nw - target_w) // 2)
    top = max(0, (nh - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h)).resize((target_w, target_h), resampling)


def apply_pil_effects(pil_img, brightness=1.0, contrast=1.0, saturation=1.0, blur_radius=0.0, video_fx="None"):
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
        sepia = (0.393, 0.769, 0.189, 0, 0.349, 0.686, 0.168, 0, 0.272, 0.534, 0.131, 0)
        out = out.convert("RGB", sepia)
    elif video_fx == "Cinematic Vignette":
        w, h = out.size
        vig = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(vig)
        draw.ellipse([-w * 0.18, -h * 0.18, w * 1.18, h * 1.18], fill=0)
        vig = vig.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.22))
        out = Image.composite(out, Image.new("RGB", (w, h), (0, 0, 0)), vig)
    elif video_fx == "Flash White":
        out = ImageEnhance.Brightness(out).enhance(1.25)
    elif video_fx == "Glitch RGB":
        r, g, b = out.split()
        r = ImageEnhance.Brightness(r).enhance(1.16)
        b = ImageEnhance.Contrast(b).enhance(1.20)
        out = Image.merge("RGB", (r, g, b))
    return out


# --- YOUTUBE & DIRECT MEDIA DOWNLOADER ---
def download_media_or_youtube(raw_url, media_kind="video", prefix="media"):
    cleaned = (raw_url or "").strip()
    if not cleaned:
        raise ValueError("Empty URL provided.")
    os.makedirs("uploaded_studio_media", exist_ok=True)
    is_yt = any(d in cleaned.lower() for d in ("youtube.com", "youtu.be"))

    if is_yt:
        import yt_dlp

        out_tmpl = os.path.join("uploaded_studio_media", f"{prefix}_yt_{int(time.time())}.%(ext)s")
        ydl_opts = {
            "outtmpl": out_tmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best"
            if media_kind == "video"
            else "bestaudio[ext=m4a]/bestaudio/best",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(cleaned, download=True)
            downloaded_path = ydl.prepare_filename(info)
            if not os.path.exists(downloaded_path):
                base_p = os.path.splitext(downloaded_path)[0]
                for ext in (".mp4", ".webm", ".mkv", ".m4a", ".mp3"):
                    if os.path.exists(base_p + ext):
                        downloaded_path = base_p + ext
                        break
            if os.path.exists(downloaded_path) and os.path.getsize(downloaded_path) > 4096:
                return downloaded_path
            raise RuntimeError("YouTube download produced an empty file.")

    resp = requests.get(cleaned, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} fetching {cleaned}")
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ct:
        raise RuntimeError("URL points to a webpage (HTML), not a direct MP4/MP3 file.")
    ext = ".mp4" if media_kind == "video" else ".mp3"
    out_path = os.path.join("uploaded_studio_media", f"{prefix}_{int(time.time() * 1000)}{ext}")
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=131072):
            if chunk:
                f.write(chunk)
    return out_path


# --- MULTI-ENGINE AI CHARACTER KEYFRAME GENERATOR ---
def generate_ai_keyframe(
    engine_choice,
    cf_account_id,
    cf_api_token,
    hf_token,
    gemini_key,
    prompt_text,
    target_w,
    target_h,
    ref_photo_path=None,
    seed=42,
):
    errors = []
    clean_p = " ".join((prompt_text or "").split())[:1500]

    def run_gemini():
        if not gemini_key or not gemini_key.strip():
            raise RuntimeError("Gemini API key missing")
        client = genai.Client(api_key=gemini_key.strip())
        for model_id in ["gemini-2.5-flash-image", "gemini-2.0-flash-exp-image-generation"]:
            try:
                contents = [clean_p]
                if ref_photo_path and os.path.exists(ref_photo_path):
                    with Image.open(ref_photo_path) as ref_im:
                        contents = [
                            ref_im.convert("RGB"),
                            f"Preserve the exact facial features, hair, and clothing of the character in this reference photo. {clean_p}",
                        ]
                resp = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
                for cand in getattr(resp, "candidates", None) or []:
                    for part in getattr(getattr(cand, "content", None), "parts", None) or []:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            return ensure_exact_size(Image.open(BytesIO(inline.data)), target_w, target_h), f"Gemini {model_id}"
            except Exception as e:
                last_e = e
        raise RuntimeError(str(last_e))

    def run_hf():
        if not hf_token or not hf_token.strip():
            raise RuntimeError("Hugging Face token missing")
        client = InferenceClient(token=hf_token.strip())
        w_hf = max(512, min(1024, (target_w // 8) * 8))
        h_hf = max(512, min(1024, (target_h // 8) * 8))
        last_e = "HF error"
        for model_id in ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"]:
            try:
                img = client.text_to_image(prompt=clean_p, model=model_id, width=w_hf, height=h_hf)
                return ensure_exact_size(img, target_w, target_h), f"HF {model_id}"
            except Exception as e:
                last_e = e
        raise RuntimeError(str(last_e))

    def run_cf():
        if not (cf_account_id and cf_account_id.strip() and cf_api_token and cf_api_token.strip()):
            raise RuntimeError("Cloudflare credentials missing")
        url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id.strip()}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        headers = {"Authorization": f"Bearer {cf_api_token.strip()}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"prompt": clean_p}, timeout=90)
        if not resp.ok:
            raise RuntimeError(f"CF HTTP {resp.status_code}")
        if "image/" in resp.headers.get("Content-Type", ""):
            return ensure_exact_size(Image.open(BytesIO(resp.content)), target_w, target_h), "Cloudflare FLUX.1"
        b64 = resp.json().get("result", {}).get("image")
        if b64:
            return ensure_exact_size(Image.open(BytesIO(base64.b64decode(b64))), target_w, target_h), "Cloudflare FLUX.1"
        raise RuntimeError("Cloudflare returned no image")

    def run_pollinations():
        w_p = max(512, min(1024, (target_w // 8) * 8))
        h_p = max(512, min(1024, (target_h // 8) * 8))
        enc = urllib.parse.quote(clean_p[:350], safe="")
        for attempt in range(2):
            url = f"https://image.pollinations.ai/prompt/{enc}?width={w_p}&height={h_p}&seed={seed + attempt}&model=flux&nologo=true"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=50)
            if r.ok and "image/" in r.headers.get("Content-Type", ""):
                return ensure_exact_size(Image.open(BytesIO(r.content)), target_w, target_h), "Pollinations FLUX"
            time.sleep(1.5)
        raise RuntimeError(f"Pollinations HTTP {r.status_code}")

    order = [
        ("Gemini", run_gemini),
        ("Hugging Face", run_hf),
        ("Cloudflare", run_cf),
        ("Pollinations", run_pollinations),
    ]
    if "Hugging Face" in engine_choice:
        order = [("Hugging Face", run_hf), ("Gemini", run_gemini), ("Cloudflare", run_cf), ("Pollinations", run_pollinations)]
    elif "Cloudflare" in engine_choice:
        order = [("Cloudflare", run_cf), ("Gemini", run_gemini), ("Hugging Face", run_hf), ("Pollinations", run_pollinations)]
    elif "Pollinations" in engine_choice:
        order = [("Pollinations", run_pollinations), ("Gemini", run_gemini), ("Hugging Face", run_hf), ("Cloudflare", run_cf)]

    for name, fn in order:
        try:
            img, src = fn()
            return img, src, errors
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if ref_photo_path and os.path.exists(ref_photo_path):
        with Image.open(ref_photo_path) as im:
            return ensure_exact_size(im, target_w, target_h), "Uploaded Character Reference", errors

    return None, None, errors


# --- TRUE AI VIDEO GENERATION (IMAGE-TO-VIDEO & TEXT-TO-VIDEO) ---
def generate_true_ai_video_clip(
    video_engine,
    hf_token,
    gemini_key,
    fal_key,
    prompt_text,
    action_motion,
    keyframe_path,
    out_mp4_path,
    target_w,
    target_h,
):
    errors = []
    full_motion_prompt = f"{prompt_text}. Character action & camera movement: {action_motion}. Smooth 24fps cinema film."[:1000]

    # 1. Google Veo 2.0 (via Gemini API Key)
    if gemini_key and gemini_key.strip() and ("Veo" in video_engine or "Auto" in video_engine):
        try:
            client = genai.Client(api_key=gemini_key.strip())
            aspect = "9:16" if target_h > target_w else "16:9"
            op = client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=full_motion_prompt,
                config=types.GenerateVideosConfig(aspect_ratio=aspect, person_generation="allow_adult"),
            )
            waited = 0
            while not op.done and waited < 180:
                time.sleep(6)
                waited += 6
                op = client.operations.get(op)
            if op.done and getattr(op.response, "generated_videos", None):
                gen_vid = op.response.generated_videos[0]
                client.files.download(file=gen_vid.video)
                gen_vid.video.save(out_mp4_path)
                if os.path.exists(out_mp4_path) and os.path.getsize(out_mp4_path) > 5000:
                    return out_mp4_path, "Google Veo 2.0", errors
        except Exception as exc:
            errors.append(f"Google Veo: {exc}")

    # 2. Hugging Face Serverless Video (Wan 2.1 / LTX-Video)
    if hf_token and hf_token.strip() and ("Hugging Face" in video_engine or "Auto" in video_engine):
        client = InferenceClient(token=hf_token.strip())
        if keyframe_path and os.path.exists(keyframe_path) and hasattr(client, "image_to_video"):
            with open(keyframe_path, "rb") as f:
                img_bytes = f.read()
            for i2v_model in ["Wan-AI/Wan2.1-I2V-14B-480P", "Lightricks/LTX-Video"]:
                try:
                    vid_bytes = client.image_to_video(image=img_bytes, prompt=full_motion_prompt, model=i2v_model)
                    if vid_bytes and len(vid_bytes) > 5000:
                        with open(out_mp4_path, "wb") as out_f:
                            out_f.write(vid_bytes)
                        return out_mp4_path, f"HF I2V ({i2v_model})", errors
                except Exception as exc:
                    errors.append(f"HF I2V {i2v_model}: {exc}")

    # 3. fal.ai Fast Image-to-Video (Wan / Kling)
    if fal_key and fal_key.strip() and ("fal.ai" in video_engine or "Auto" in video_engine):
        try:
            headers = {"Authorization": f"Key {fal_key.strip()}", "Content-Type": "application/json"}
            payload = {"prompt": full_motion_prompt}
            if keyframe_path and os.path.exists(keyframe_path):
                with open(keyframe_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                payload["image_url"] = f"data:image/png;base64,{b64}"
                endpoint = "https://fal.run/fal-ai/wan-i2v"
            else:
                endpoint = "https://fal.run/fal-ai/wan-t2v"
            r = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            if r.ok:
                vid_url = r.json().get("video", {}).get("url")
                if vid_url:
                    vr = requests.get(vid_url, timeout=90)
                    with open(out_mp4_path, "wb") as f:
                        f.write(vr.content)
                    return out_mp4_path, "fal.ai Wan Video", errors
            else:
                errors.append(f"fal.ai HTTP {r.status_code}")
        except Exception as exc:
            errors.append(f"fal.ai: {exc}")

    return None, None, errors


# --- 2.5D DEPTH PARALLAX + CINEMA MOTION FALLBACK ---
def build_25d_parallax_clip(
    img_path,
    out_path,
    target_w,
    target_h,
    duration=4.0,
    motion_mode="Zoom In",
    scene_idx=0,
    brightness=1.0,
    contrast=1.06,
    saturation=1.08,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Crop Fill",
):
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    safe_dur = max(1.5, float(duration))
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    with Image.open(img_path) as im:
        base_rgb = ensure_exact_size(im.convert("RGB"), target_w, target_h, bg_mode=bg_mode)
    base_rgb = apply_pil_effects(base_rgb, brightness, contrast, saturation, blur_radius, video_fx)

    overscan = 1.16
    cw, ch = int(target_w * overscan), int(target_h * overscan)
    fg_canvas = base_rgb.resize((cw, ch), resampling)
    bg_canvas = fg_canvas.filter(ImageFilter.GaussianBlur(radius=5))

    mask = Image.new("L", (cw, ch), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.ellipse([int(cw * 0.14), int(ch * 0.12), int(cw * 0.86), int(ch * 0.88)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=int(min(cw, ch) * 0.12)))
    layered_canvas = Image.composite(fg_canvas, bg_canvas, mask)

    def make_frame(t):
        u = max(0.0, min(1.0, float(t) / safe_dur))
        ease = u * u * (3.0 - 2.0 * u)

        if video_fx == "Zoom Punch":
            scale = 1.0 + 0.12 * math.sin(u * math.pi)
        elif active_mode == "Zoom In":
            scale = 1.0 + 0.13 * ease
        elif active_mode == "Zoom Out":
            scale = 1.14 - 0.13 * ease
        else:
            scale = 1.09

        win_w = max(2, int(round(target_w / scale)))
        win_h = max(2, int(round(target_h / scale)))

        if active_mode == "Pan Right":
            x0 = int(round((cw - win_w) * ease))
            y0 = (ch - win_h) // 2 + int(round(math.sin(u * math.pi * 2) * 6))
        elif active_mode == "Pan Left":
            x0 = int(round((cw - win_w) * (1.0 - ease)))
            y0 = (ch - win_h) // 2 + int(round(math.cos(u * math.pi * 2) * 6))
        else:
            x0 = (cw - win_w) // 2 + int(round(math.sin(u * math.pi * 2) * 8))
            y0 = (ch - win_h) // 2 + int(round(math.cos(u * math.pi * 2) * 6))

        if video_fx == "Shake":
            x0 += int(round(math.sin(t * 32.0) * 7))
            y0 += int(round(math.cos(t * 36.0) * 5))

        x0 = max(0, min(cw - win_w, x0))
        y0 = max(0, min(ch - win_h, y0))
        frame = layered_canvas.crop((x0, y0, x0 + win_w, y0 + win_h)).resize((target_w, target_h), resampling)
        return np.asarray(frame, dtype=np.uint8)

    clip = apply_duration(DynamicVideoClip(make_frame, duration=safe_dur), safe_dur)
    try:
        clip.write_videofile(
            out_path,
            fps=ANIMATION_FPS,
            codec="libx264",
            audio=False,
            ffmpeg_params=["-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
            logger=None,
        )
        return out_path
    finally:
        try:
            clip.close()
        except Exception:
            pass


# --- SUBTITLES & OPTIONAL TOP HOOK BANNER ---
@st.cache_resource(show_spinner=False)
def _cached_font(font_size: int, lang: str = "English"):
    if lang == "Kannada":
        tmp_font = "/tmp/NotoSansKannada-Bold.ttf"
        if not os.path.exists(tmp_font):
            try:
                r = requests.get(
                    "https://github.com/google/fonts/raw/main/ofl/notosanskannada/static/NotoSansKannada-Bold.ttf",
                    timeout=20,
                )
                if r.ok and len(r.content) > 40000:
                    with open(tmp_font, "wb") as f:
                        f.write(r.content)
            except Exception:
                pass
        if os.path.exists(tmp_font):
            return ImageFont.truetype(tmp_font, int(font_size))
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, int(font_size))
    return ImageFont.load_default()


def render_cinema_captions_and_hook(
    frame_pil,
    subtitle_text,
    target_w,
    target_h,
    lang="English",
    position="Bottom",
    scale=0.90,
    hook_enabled=False,
    hook_text="",
    hook_highlights="",
):
    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    # Optional Top Hook Banner
    if hook_enabled and hook_text.strip():
        hi_set = {w.strip().lower() for w in (hook_highlights or "").split(",") if w.strip()}
        h_font = _cached_font(max(22, int(target_w * 0.042)), lang="English")
        y_cur = max(28, int(target_h * 0.05))
        for raw_line in textwrap.wrap(hook_text.strip(), width=28)[:2]:
            words = raw_line.split()
            sp_w = draw.textbbox((0, 0), " ", font=h_font)[2] - draw.textbbox((0, 0), " ", font=h_font)[0]
            widths = [draw.textbbox((0, 0), w, font=h_font)[2] - draw.textbbox((0, 0), w, font=h_font)[0] for w in words]
            tot_w = sum(widths) + max(0, len(words) - 1) * sp_w
            x_cur = max(16, (target_w - tot_w) // 2)
            for w, ww in zip(words, widths):
                clean_w = w.strip(".,!?:;\"'()").lower()
                fill_c = (255, 222, 89, 255) if clean_w in hi_set else (255, 255, 255, 255)
                draw.text((x_cur + 2, y_cur + 2), w, font=h_font, fill=(0, 0, 0, 240))
                draw.text((x_cur, y_cur), w, font=h_font, fill=fill_c)
                x_cur += ww + sp_w
            y_cur += int(target_w * 0.052)

    # Clean Cinema Subtitle (Matches video_1.mp4 crisp white subtitle style)
    if subtitle_text and subtitle_text.strip() and lang != "None":
        font_size = max(20, int(round(min(target_w, target_h) * 0.033 * float(scale))))
        font = _cached_font(font_size, lang=lang)
        chars_per_line = max(20, int((target_w * 0.84) / max(9, font_size * 0.54)))
        lines = textwrap.wrap(subtitle_text.strip(), width=chars_per_line)[:2]
        if lines:
            line_h = font_size + 8
            box_h = len(lines) * line_h + 12
            if position == "Top":
                box_top = max(20, int(target_h * 0.06))
            elif position == "Center":
                box_top = (target_h - box_h) // 2
            else:
                box_top = target_h - box_h - max(130, int(target_h * 0.14))

            cur_y = box_top
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                tx = max(20, (target_w - (bbox[2] - bbox[0])) // 2)
                # Crisp dark outline for cinema readability
                for ox, oy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 2)]:
                    draw.text((tx + ox, cur_y + oy), line, font=font, fill=(0, 0, 0, 235))
                draw.text((tx, cur_y), line, font=font, fill=(255, 255, 255, 255))
                cur_y += line_h

    return img_rgba.convert("RGB")


# --- AI MULTI-CHARACTER SCRIPT DIRECTOR ---
def generate_ai_script(gemini_key, story_cmd, num_scenes, cast_list, style_name):
    if not gemini_key or not gemini_key.strip():
        return None, "Add your Gemini API key in the sidebar to generate custom scripts."
    client = genai.Client(api_key=gemini_key.strip())
    cast_summary = "\n".join(f"- {c['name']}: {c['appearance']}" for c in cast_list)
    cast_names = [c["name"] for c in cast_list]

    prompt = f"""
Create a {num_scenes}-scene dramatic vertical series / animated reel script with dialogue between characters.
Visual Style: {style_name}
Available Cast Members:
{cast_summary}

User Story Concept: {story_cmd}

Return ONLY a valid JSON array of {num_scenes} objects (no markdown fences) with keys:
- "speaker": exact name from {cast_names} who is speaking in this scene
- "narration": crisp spoken dialogue line in English (1-2 short sentences)
- "kannada_narration": natural Kannada translation of the dialogue
- "emotion": dramatic emotion label (e.g., "Cold & Commanding", "Urgent & Nervous")
- "motion": one of ["Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
- "action_motion": physical character acting, facial expression, lip movement, and camera shot for AI video generation
- "suggested_fx": one of ["None", "Zoom Punch", "Shake", "Flash White", "Cinematic Vignette"]
- "visual": detailed visual prompt describing the exact character(s) on screen, camera framing (close-up / medium shot), lighting, and set background
"""
    last_err = "Unknown error"
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            raw = (resp.text or "").strip()
            if raw.startswith("```"):
                raw = "\n".join(ln for ln in raw.splitlines() if not ln.strip().startswith("```")).strip()
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data[:num_scenes], None
        except Exception as exc:
            last_err = str(exc)
    return None, last_err


# --- HEADER + UNDO/REDO BAR ---
st.markdown(
    """
    <div class="studio-header">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="studio-pill">AI CINEMA & REELS STUDIO PRO</span>
            <span style="font-weight:700;font-size:17px;">Multi-Character Vertical Drama, 3D/2D Animation & Reels Studio</span>
        </div>
        <div class="studio-sub">Multi-Character Cast Lock • Per-Character Voices • Veo / Wan 2.1 / Gemini / FLUX • Color Grading, SFX & BGM Ducking</div>
    </div>
    """,
    unsafe_allow_html=True,
)

u_col1, u_col2, u_col3 = st.columns([1, 1, 4])
with u_col1:
    if st.button("↩️ Undo", disabled=not st.session_state.undo_stack, use_container_width=True):
        if perform_undo():
            st.rerun()
with u_col2:
    if st.button("↪️ Redo", disabled=not st.session_state.redo_stack, use_container_width=True):
        if perform_redo():
            st.rerun()
with u_col3:
    if st.session_state.last_history_label:
        st.caption(f"Last action: **{st.session_state.last_history_label}**")

# --- SIDEBAR: FORMAT, ENGINES & VISUAL EDITOR ---
with st.sidebar:
    st.header("🎬 Format & Visual Style")
    st.session_state.format_preset = st.selectbox(
        "Video Format Preset",
        list(FORMAT_PRESETS.keys()),
        index=list(FORMAT_PRESETS.keys()).index(st.session_state.format_preset),
    )
    width, height, ratio_str = FORMAT_PRESETS[st.session_state.format_preset]
    st.caption(f"Resolution: **{width} × {height}** (`{ratio_str}`)")

    st.session_state.animation_style_name = st.selectbox(
        "Cinema / Animation Style",
        list(ANIMATION_STYLES.keys()),
        index=list(ANIMATION_STYLES.keys()).index(st.session_state.animation_style_name),
    )

    with st.expander("🔑 AI Video & Image API Keys", expanded=True):
        gemini_key = st.text_input(
            "Gemini API Key (Script + Imagen + Veo 2)",
            value=_get_secret("GEMINI_API_KEY"),
            type="password",
        )
        hf_token = st.text_input(
            "Hugging Face Token (FLUX + Wan 2.1 Video)",
            value=_get_secret("HF_TOKEN"),
            type="password",
        )
        cf_account_id = st.text_input(
            "Cloudflare Account ID (Free Daily FLUX)",
            value=_get_secret("CF_ACCOUNT_ID"),
            type="password",
        )
        cf_api_token = st.text_input(
            "Cloudflare API Token",
            value=_get_secret("CF_API_TOKEN"),
            type="password",
        )
        fal_key = st.text_input(
            "fal.ai API Key (Optional Fast Wan/Kling I2V)",
            value=_get_secret("FAL_KEY"),
            type="password",
        )

    with st.expander("🎨 Color Grading, Video FX & Hook Banner", expanded=False):
        st.session_state.editor_brightness = st.slider("Brightness", 0.6, 1.5, st.session_state.editor_brightness, 0.02)
        st.session_state.editor_contrast = st.slider("Contrast", 0.6, 1.6, st.session_state.editor_contrast, 0.02)
        st.session_state.editor_saturation = st.slider("Saturation", 0.5, 1.8, st.session_state.editor_saturation, 0.02)
        st.session_state.editor_video_fx = st.selectbox("Global Video FX", VIDEO_FX_CHOICES)
        st.session_state.top_hook_enabled = st.checkbox("Add Top Hook Text Banner", value=st.session_state.top_hook_enabled)
        if st.session_state.top_hook_enabled:
            st.session_state.top_hook_text = st.text_input("Top Hook Text", value=st.session_state.top_hook_text)
            st.session_state.top_hook_highlights = st.text_input("Yellow Highlight Words", value=st.session_state.top_hook_highlights)

# --- ALL 5 STUDIO TABS ---
tab_cast, tab_script, tab_video, tab_audio, tab_export = st.tabs([
    "1️⃣ Character Cast & Photos",
    "2️⃣ Story & Dialogue Script",
    "3️⃣ AI Video & Media Studio",
    "4️⃣ Multi-Voice, SFX & BGM",
    "5️⃣ Render Final Movie",
])

# === TAB 1: MULTI-CHARACTER CAST & REFERENCE PHOTOS ===
with tab_cast:
    st.markdown("### 1. Define Your Cast (Keeps Every Character's Face & Outfit Consistent)")
    st.session_state.lock_character_enabled = st.checkbox(
        "Enforce Character Cast Appearance Lock in Every Scene", value=st.session_state.lock_character_enabled
    )

    if st.button("➕ Add New Character to Cast"):
        record_undo_step("Add Cast Member")
        st.session_state.cast.append({
            "name": f"Character {len(st.session_state.cast) + 1}",
            "appearance": "Detailed appearance, hair, age, and clothing description",
            "voice": list(TTS_VOICES.keys())[0],
            "ref_photo": "",
        })
        st.rerun()

    for c_idx, member in enumerate(st.session_state.cast):
        with st.expander(f"🎭 {member['name']}", expanded=True):
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                member["name"] = st.text_input("Character Name", value=member["name"], key=f"cname_{c_idx}")
                member["appearance"] = st.text_area(
                    "Exact Visual Appearance (Face, Hair, Suit/Outfit)",
                    value=member["appearance"],
                    height=80,
                    key=f"capp_{c_idx}",
                )
                v_keys = list(TTS_VOICES.keys())
                v_idx = v_keys.index(member["voice"]) if member.get("voice") in v_keys else 0
                member["voice"] = st.selectbox("Character Voice", v_keys, index=v_idx, key=f"cvoice_{c_idx}")
            with cc2:
                ref_up = st.file_uploader(
                    f"Reference Face Photo ({member['name']})",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"cref_up_{c_idx}",
                )
                if ref_up is not None:
                    os.makedirs("uploaded_studio_media", exist_ok=True)
                    rp = os.path.join("uploaded_studio_media", f"cast_{c_idx}_{int(time.time())}.png")
                    Image.open(ref_up).convert("RGB").save(rp)
                    member["ref_photo"] = rp
                if member.get("ref_photo") and os.path.exists(member["ref_photo"]):
                    st.image(member["ref_photo"], caption="Active Reference", width=150)

# === TAB 2: STORY & DIALOGUE SCRIPT ===
with tab_script:
    st.markdown("### 2. Write or Generate Multi-Character Scene Dialogue")
    story_concept = st.text_area(
        "Series / Drama Storyline:",
        value="A powerful Mafia Don in his study gets an urgent report from his loyal bodyguard that his rebellious younger brother skipped school and went to a high-stakes casino.",
        height=85,
    )
    sc_col1, sc_col2 = st.columns([2, 1])
    with sc_col1:
        num_scenes = st.slider("Number of Scenes", 2, 10, len(st.session_state.scenes))
    with sc_col2:
        if st.button("➕ Add Blank Scene", use_container_width=True):
            record_undo_step("Add Scene")
            st.session_state.scenes.append({
                "speaker": st.session_state.cast[0]["name"],
                "narration": "New dramatic dialogue line.",
                "kannada_narration": "ಹೊಸ ಸಂಭಾಷಣೆ.",
                "emotion": "Dramatic",
                "motion": "Zoom In",
                "action_motion": "Close-up character speaking dramatically with subtle head movement",
                "suggested_fx": "None",
                "suggested_sfx": "None",
                "visual": "Dramatic cinema portrait in luxury study",
            })
            st.rerun()

    if st.button("✨ Generate Multi-Character Dialogue Script (Gemini)", type="primary", use_container_width=True):
        record_undo_step("Generate AI Script")
        with st.spinner("Writing dramatic dialogue, camera cuts, and action prompts..."):
            scenes_out, err = generate_ai_script(
                gemini_key,
                story_concept,
                num_scenes,
                st.session_state.cast,
                st.session_state.animation_style_name,
            )
            if scenes_out:
                st.session_state.scenes = scenes_out
                st.session_state.gen_batch_id += 1
                force_sync_scene_widget_states(scenes_out)
                st.success(f"Generated {len(scenes_out)} dramatic scenes!")
                st.rerun()
            else:
                st.warning(f"Could not call Gemini ({err}). Edit the pre-loaded scenes below directly!")

    cast_names = [c["name"] for c in st.session_state.cast]
    for i, sc in enumerate(st.session_state.scenes):
        with st.expander(f"Scene {i + 1} — 🗣️ {sc.get('speaker', cast_names[0])} ({sc.get('emotion', 'Dramatic')})", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                cur_sp = sc.get("speaker", cast_names[0])
                sp_idx = cast_names.index(cur_sp) if cur_sp in cast_names else 0
                sc["speaker"] = st.selectbox("Speaking Character", cast_names, index=sp_idx, key=_scene_widget_key("spk", i))
                sc["narration"] = st.text_area(
                    "English Dialogue / Caption",
                    value=sc.get("narration", ""),
                    key=_scene_widget_key("nar", i),
                )
                sc["kannada_narration"] = st.text_area(
                    "Kannada Dialogue / Caption",
                    value=sc.get("kannada_narration", ""),
                    key=_scene_widget_key("kn_script", i),
                )
            with c2:
                sc["visual"] = st.text_area(
                    "Visual Shot & Lighting Prompt",
                    value=sc.get("visual", ""),
                    key=_scene_widget_key("vis", i),
                )
                sc["action_motion"] = st.text_input(
                    "AI Video Acting & Camera Motion",
                    value=sc.get("action_motion", "Close-up character speaking with realistic facial expressions"),
                    key=_scene_widget_key("act", i),
                )
                m_c1, m_c2 = st.columns(2)
                with m_c1:
                    cur_m = sc.get("motion", "Zoom In")
                    sc["motion"] = st.selectbox(
                        "Camera Move",
                        MOTION_CHOICES,
                        index=MOTION_CHOICES.index(cur_m) if cur_m in MOTION_CHOICES else 0,
                        key=_scene_widget_key("mot", i),
                    )
                with m_c2:
                    sfx_keys = list(PRESET_AUDIO_EFFECTS.keys())
                    cur_sfx = sc.get("suggested_sfx", "None")
                    sc["suggested_sfx"] = st.selectbox(
                        "Scene SFX",
                        sfx_keys,
                        index=sfx_keys.index(cur_sfx) if cur_sfx in sfx_keys else 0,
                        key=_scene_widget_key("sfx", i),
                    )

# === TAB 3: AI VIDEO & MEDIA STUDIO ===
with tab_video:
    st.markdown("### 3. Generate Character Keyframes & AI Animated Video Clips")

    imp_c1, imp_c2 = st.columns(2)
    with imp_c1:
        yt_or_mp4 = st.text_input(
            "Import Clip from YouTube URL or Direct MP4 Link:",
            placeholder="https://www.youtube.com/watch?v=... or https://.../scene.mp4",
        )
        if st.button("🌐 Download & Append Video Clip", use_container_width=True) and yt_or_mp4.strip():
            with st.spinner("Downloading clip..."):
                try:
                    vp = download_media_or_youtube(yt_or_mp4, media_kind="video", prefix="yt_clip")
                    st.session_state.scene_videos.append(vp)
                    st.success(f"Added `{os.path.basename(vp)}`")
                except Exception as exc:
                    st.error(f"Import failed: {exc}")
    with imp_c2:
        up_vids = st.file_uploader("Or Upload Existing MP4 Clips (from Veo / Kling / Runway)", type=["mp4", "mov"], accept_multiple_files=True)
        if st.button("📥 Use Uploaded MP4 Clips", use_container_width=True) and up_vids:
            os.makedirs("uploaded_studio_media", exist_ok=True)
            paths = []
            for idx_v, vf in enumerate(up_vids):
                vp = os.path.join("uploaded_studio_media", f"up_vid_{int(time.time())}_{idx_v}.mp4")
                with open(vp, "wb") as f:
                    f.write(vf.getbuffer())
                paths.append(vp)
            st.session_state.scene_videos = paths
            st.success(f"Loaded {len(paths)} video clip(s)!")

    st.divider()
    eng_c1, eng_c2, eng_c3 = st.columns(3)
    with eng_c1:
        img_engine = st.selectbox(
            "Character Keyframe Image Engine",
            [
                "Gemini Image (Best Character Consistency)",
                "Hugging Face (FLUX.1-schnell)",
                "Cloudflare Workers AI (FLUX)",
                "Pollinations (Free No-Key)",
            ],
        )
    with eng_c2:
        vid_engine = st.selectbox(
            "AI Video Animation Engine",
            [
                "Auto (Google Veo 2 -> HF Wan 2.1 -> fal.ai -> 2.5D Cinema Fallback)",
                "Google Veo 2.0",
                "Hugging Face Video (Wan 2.1 I2V)",
                "fal.ai Image-to-Video",
                "2.5D Depth Parallax Cinema (Instant Free)",
            ],
        )
    with eng_c3:
        clip_sec = st.slider("Default Clip Duration (sec)", 2.5, 6.0, 4.0, 0.5)

    if st.button("🚀 Generate All Character Visuals & Video Clips", type="primary", use_container_width=True):
        record_undo_step("Generate Scene Videos")
        os.makedirs("generated_keyframes", exist_ok=True)
        os.makedirs("generated_videos", exist_ok=True)

        style_desc = ANIMATION_STYLES[st.session_state.animation_style_name]
        cast_lookup = {c["name"]: c for c in st.session_state.cast}

        new_images, new_videos = [], []
        prog = st.progress(0.0)
        status = st.empty()

        for idx, sc in enumerate(st.session_state.scenes):
            speaker_name = sc.get("speaker", "")
            char_obj = cast_lookup.get(speaker_name, st.session_state.cast[0])
            char_desc = char_obj["appearance"] if st.session_state.lock_character_enabled else ""
            char_ref = char_obj.get("ref_photo", "")

            status.info(f"Scene {idx + 1}/{len(st.session_state.scenes)} ({speaker_name}): Rendering character keyframe & video...")
            full_prompt = f"{style_desc}. Character on screen: {char_desc}. Scene: {sc.get('visual', '')}"

            kf_path = os.path.join("generated_keyframes", f"kf_{idx}_{int(time.time())}.png")
            pil_kf, img_src, img_errs = generate_ai_keyframe(
                engine_choice=img_engine,
                cf_account_id=cf_account_id,
                cf_api_token=cf_api_token,
                hf_token=hf_token,
                gemini_key=gemini_key,
                prompt_text=full_prompt,
                target_w=width,
                target_h=height,
                ref_photo_path=char_ref,
                seed=800 + idx * 29,
            )

            if pil_kf is not None:
                pil_kf.save(kf_path)
                new_images.append(kf_path)
            else:
                kf_path = ""
                st.error(f"Scene {idx + 1} image failed ({'; '.join(img_errs)}). Add a Gemini or Hugging Face key in the sidebar!")

            vid_out = os.path.join("generated_videos", f"scene_{idx}_{int(time.time())}.mp4")
            gen_mp4 = None
            vid_src = ""

            if "2.5D Depth Parallax" not in vid_engine:
                gen_mp4, vid_src, vid_errs = generate_true_ai_video_clip(
                    video_engine=vid_engine,
                    hf_token=hf_token,
                    gemini_key=gemini_key,
                    fal_key=fal_key,
                    prompt_text=full_prompt,
                    action_motion=sc.get("action_motion", "Cinematic acting and camera motion"),
                    keyframe_path=kf_path,
                    out_mp4_path=vid_out,
                    target_w=width,
                    target_h=height,
                )
                if not gen_mp4 and vid_errs:
                    st.caption(f"Scene {idx + 1} video API fallback: {vid_errs[0]}")

            if gen_mp4 and os.path.exists(gen_mp4):
                new_videos.append(gen_mp4)
                st.toast(f"Scene {idx + 1}: Generated via {vid_src}")
            elif kf_path and os.path.exists(kf_path):
                build_25d_parallax_clip(
                    img_path=kf_path,
                    out_path=vid_out,
                    target_w=width,
                    target_h=height,
                    duration=clip_sec,
                    motion_mode=sc.get("motion", "Zoom In"),
                    scene_idx=idx,
                    brightness=st.session_state.editor_brightness,
                    contrast=st.session_state.editor_contrast,
                    saturation=st.session_state.editor_saturation,
                    video_fx=sc.get("suggested_fx", st.session_state.editor_video_fx),
                    bg_mode=st.session_state.bg_fit_mode,
                )
                new_videos.append(vid_out)

            prog.progress((idx + 1) / len(st.session_state.scenes))

        st.session_state.images = new_images
        st.session_state.scene_videos = new_videos
        if new_videos:
            status.success(f"Generated {len(new_videos)} scene clips!")

    if st.session_state.scene_videos:
        st.markdown("#### Generated Scene Previews")
        cols = st.columns(min(4, len(st.session_state.scene_videos)))
        for idx, vp in enumerate(st.session_state.scene_videos):
            if os.path.exists(vp):
                with cols[idx % len(cols)]:
                    sp_lbl = st.session_state.scenes[idx].get("speaker", f"Scene {idx + 1}") if idx < len(st.session_state.scenes) else f"Scene {idx + 1}"
                    st.caption(f"Scene {idx + 1}: {sp_lbl}")
                    st.video(vp)

# === TAB 4: MULTI-VOICE, SFX & BGM ===
with tab_audio:
    st.markdown("### 4. Multi-Character Neural Voiceover, Sound Effects & BGM")
    a_c1, a_c2, a_c3 = st.columns(3)
    with a_c1:
        st.session_state.subtitle_lang = st.selectbox("Dialogue & Subtitle Language", ["English", "Kannada", "None"])
    with a_c2:
        st.session_state.subtitle_position = st.selectbox("Subtitle Position", SUBTITLE_POSITIONS)
    with a_c3:
        st.session_state.subtitle_scale = st.slider("Subtitle Size", 0.6, 1.3, st.session_state.subtitle_scale, 0.05)

    st.info("Each scene automatically uses the voice assigned to its **Speaking Character** in Tab 1!")
    if st.button("🎙️ Generate Multi-Character Dialogue Voices (edge-tts)", type="primary", use_container_width=True):
        os.makedirs("generated_audio", exist_ok=True)
        cast_lookup = {c["name"]: c for c in st.session_state.cast}
        aud_paths = []
        with st.spinner("Synthesizing character dialogue..."):
            for idx, sc in enumerate(st.session_state.scenes):
                char_obj = cast_lookup.get(sc.get("speaker", ""), st.session_state.cast[0])
                v_label = char_obj.get("voice", list(TTS_VOICES.keys())[0])
                v_code = TTS_VOICES.get(v_label, "en-US-ChristopherNeural")
                if st.session_state.subtitle_lang == "Kannada":
                    v_code = "kn-IN-SapnaNeural" if "Female" in v_label else "kn-IN-GaganNeural"

                spoken = (
                    sc.get("kannada_narration")
                    if st.session_state.subtitle_lang == "Kannada" and sc.get("kannada_narration")
                    else sc.get("narration", f"Scene {idx + 1}")
                )
                out_mp3 = os.path.join("generated_audio", f"dialogue_{idx}_{int(time.time())}.mp3")
                run_async(edge_tts.Communicate(spoken, v_code).save(out_mp3))
                aud_paths.append(out_mp3)
        st.session_state.audio_files = aud_paths
        st.success(f"Generated {len(aud_paths)} character dialogue audio tracks!")

    st.divider()
    bgm_c1, bgm_c2 = st.columns(2)
    with bgm_c1:
        bgm_file = st.file_uploader("Upload Dramatic Background Music (MP3 / WAV)", type=["mp3", "wav", "m4a"])
        if bgm_file:
            os.makedirs("uploaded_studio_media", exist_ok=True)
            bp = os.path.join("uploaded_studio_media", f"bgm_{int(time.time())}.mp3")
            with open(bp, "wb") as f:
                f.write(bgm_file.getbuffer())
            st.session_state.bgm_audio_path = bp
            st.success("Background music loaded!")
    with bgm_c2:
        st.session_state.bgm_volume = st.slider("BGM Volume", 0.0, 0.5, st.session_state.bgm_volume, 0.02)
        st.session_state.bgm_duck_enabled = st.checkbox("Auto-Lower BGM During Dialogue (Audio Ducking)", value=st.session_state.bgm_duck_enabled)
        st.session_state.sfx_volume = st.slider("Scene SFX Volume", 0.0, 1.0, st.session_state.sfx_volume, 0.05)

# === TAB 5: RENDER FINAL MOVIE ===
with tab_export:
    st.markdown("### 5. Assemble & Export Final Cinema Drama / Reel MP4")
    if st.button("🎬 Render Final Full-Resolution MP4", type="primary", use_container_width=True):
        vids = [p for p in st.session_state.scene_videos if os.path.exists(p)]
        if not vids:
            st.error("Generate or upload scene clips in Tab 3 first.")
        else:
            os.makedirs("final_exports", exist_ok=True)
            final_mp4 = os.path.join("final_exports", f"AI_Cinema_Series_{int(time.time())}.mp4")
            assembled_clips, handles = [], []

            with st.spinner("Compositing characters, dialogue voices, SFX, subtitles, and BGM..."):
                for idx, vp in enumerate(vids):
                    vc = VideoFileClip(vp)
                    handles.append(vc)

                    sc_obj = st.session_state.scenes[idx] if idx < len(st.session_state.scenes) else {}
                    aud_tracks = []

                    # Dialogue voiceover
                    if idx < len(st.session_state.audio_files) and os.path.exists(st.session_state.audio_files[idx]):
                        vo = AudioFileClip(st.session_state.audio_files[idx])
                        handles.append(vo)
                        aud_tracks.append(scale_audio_volume(vo, st.session_state.voice_volume))
                        target_dur = max(float(vc.duration or 3.0), float(vo.duration or 3.0) + 0.35)
                    else:
                        target_dur = float(vc.duration or 4.0)

                    # Optional Scene SFX
                    sfx_name = sc_obj.get("suggested_sfx", "None")
                    sfx_url = PRESET_AUDIO_EFFECTS.get(sfx_name, "")
                    if sfx_url:
                        try:
                            sfx_p = download_media_or_youtube(sfx_url, media_kind="audio", prefix=f"sfx_{idx}")
                            sfx_c = AudioFileClip(sfx_p)
                            handles.append(sfx_c)
                            sfx_trim = subclip_safe(sfx_c, 0.0, min(target_dur, float(sfx_c.duration or 2.0)))
                            aud_tracks.append(scale_audio_volume(sfx_trim, st.session_state.sfx_volume))
                        except Exception:
                            pass

                    vc_fit = loop_clip_safe(vc, target_dur)

                    sub_txt = (
                        sc_obj.get("kannada_narration", "")
                        if st.session_state.subtitle_lang == "Kannada"
                        else sc_obj.get("narration", "")
                    )

                    def frame_pipeline(gf, t, s_txt=sub_txt):
                        pil = ensure_exact_size(Image.fromarray(gf(t)), width, height, st.session_state.bg_fit_mode)
                        pil = render_cinema_captions_and_hook(
                            frame_pil=pil,
                            subtitle_text=s_txt,
                            target_w=width,
                            target_h=height,
                            lang=st.session_state.subtitle_lang,
                            position=st.session_state.subtitle_position,
                            scale=st.session_state.subtitle_scale,
                            hook_enabled=st.session_state.top_hook_enabled,
                            hook_text=st.session_state.top_hook_text,
                            hook_highlights=st.session_state.top_hook_highlights,
                        )
                        return np.asarray(pil, dtype=np.uint8)

                    styled_vc = vc_fit.transform(frame_pipeline) if hasattr(vc_fit, "transform") else vc_fit.fl(frame_pipeline)
                    if aud_tracks:
                        styled_vc = apply_audio(styled_vc, CompositeAudioClip(aud_tracks))
                    assembled_clips.append(styled_vc)

                final_video = concatenate_videoclips(assembled_clips, method="compose")

                # Mix BGM with optional ducking
                if st.session_state.bgm_audio_path and os.path.exists(st.session_state.bgm_audio_path):
                    bgm_raw = AudioFileClip(st.session_state.bgm_audio_path)
                    handles.append(bgm_raw)
                    tot_d = float(final_video.duration or 4.0)
                    bgm_loop = loop_clip_safe(bgm_raw, tot_d)
                    effective_bgm_vol = (
                        st.session_state.bgm_duck_volume
                        if (st.session_state.bgm_duck_enabled and final_video.audio is not None)
                        else st.session_state.bgm_volume
                    )
                    bgm_scaled = scale_audio_volume(bgm_loop, effective_bgm_vol)
                    if final_video.audio is not None:
                        final_video = apply_audio(final_video, CompositeAudioClip([final_video.audio, bgm_scaled]))
                    else:
                        final_video = apply_audio(final_video, bgm_scaled)

                final_video.write_videofile(
                    final_mp4,
                    fps=ANIMATION_FPS,
                    codec="libx264",
                    audio_codec="aac",
                    ffmpeg_params=["-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
                    logger=None,
                )
                for h in handles:
                    try:
                        h.close()
                    except Exception:
                        pass

            st.success("Final Cinema Drama / Reel rendered!")
            st.video(final_mp4)
            with open(final_mp4, "rb") as f:
                st.download_button(
                    "⬇️ Download Final Movie MP4",
                    data=f.read(),
                    file_name=os.path.basename(final_mp4),
                    mime="video/mp4",
                    use_container_width=True,
                )
