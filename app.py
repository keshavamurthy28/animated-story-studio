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
    from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
    MOVIEPY_V2 = True

st.set_page_config(
    page_title="AI Animated Story Studio",
    page_icon="🎬",
    layout="wide",
)

DEFAULT_CHAR_PROMPT = (
    "Single 35-year-old man with short neat dark hair, subtle stubble, wearing a crisp "
    "charcoal blazer over a light shirt and dark trousers, expressive eyes and body language, "
    "full upper-body or medium shot inside a modern corporate office"
)

DEFAULT_CORPORATE_SCENES = [
    {
        "narration": "At thirty-five, every deadline in the corporate office feels heavier than the last, pulling him deep into quiet pressure.",
        "kannada_narration": "ಮೂವತ್ತೈದರ ವಯಸ್ಸಿನಲ್ಲಿ, ಕಚೇರಿಯ ಪ್ರತಿಯೊಂದು ಗಡುವು ಮೊದಲಿಗಿಂತ ಹೆಚ್ಚು ಭಾರವಾಗಿ ಭಾಸವಾಗುತ್ತದೆ, ಆಳವಾದ ಒತ್ತಡಕ್ಕೆ ದೂಡುತ್ತದೆ.",
        "emotion": "Heavy Pressure & Quiet Strain",
        "visual": (
            "Medium shot at a corporate desk with dual monitors and spreadsheets; 35-year-old man "
            "rubbing his temples with furrowed brow and tense shoulders under cool office lights"
        ),
    },
    {
        "narration": "A sudden shift in the meeting room leaves him caught between sharp expectation and silent disbelief.",
        "kannada_narration": "ಸಭಾಂಗಣದಲ್ಲಿನ ಹಠಾತ್ ಬದಲಾವಣೆ ಅವನನ್ನು ನಿರೀಕ್ಷೆ ಮತ್ತು ಮೌನ ಅಚ್ಚರಿಯ ನಡುವೆ ಸಿಲುಕಿಸುತ್ತದೆ.",
        "emotion": "Shock & Silent Disbelief",
        "visual": (
            "Wide-medium shot inside a glass-walled corporate conference room with a long table and notes; "
            "35-year-old man frozen mid-thought with wide eyes and jaw tightened"
        ),
    },
    {
        "narration": "Standing beside the tall office window as city lights flicker, waves of frustration and vulnerability wash over him.",
        "kannada_narration": "ಕಚೇರಿಯ ಎತ್ತರದ ಕಿಟಕಿಯ ಬಳಿ ನಿಂತಾಗ, ಹತಾಶೆ ಮತ್ತು ಭಾವುಕತೆಯ ಅಲೆಗಳು ಅವನನ್ನು ಆವರಿಸುತ್ತವೆ.",
        "emotion": "Frustration & Vulnerability",
        "visual": (
            "Medium-wide shot near floor-to-ceiling office windows overlooking twilight city towers; "
            "35-year-old man leaning slightly against the frame with a tired sigh and downcast gaze"
        ),
    },
    {
        "narration": "Taking a slow breath back at his workstation, he gathers resolve and steadies his focus for what comes next.",
        "kannada_narration": "ತನ್ನ ಡೆಸ್ಕ್ ಬಳಿ ನಿಧಾನವಾಗಿ ಉಸಿರು ಎಳೆದುಕೊಂಡು, ಮುಂದಿನ ಸವಾಲಿಗೆ ಧೈರ್ಯ ಮತ್ತು ಗಮನವನ್ನು ಒಗ್ಗೂಡಿಸುತ್ತಾನೆ.",
        "emotion": "Inner Resolve & Composure",
        "visual": (
            "Medium shot at a clean corporate workstation with a notebook and coffee mug; "
            "35-year-old man sitting upright with a calm, determined expression and focused eyes"
        ),
    },
]

for key, default in [
    ("scenes", [dict(s) for s in DEFAULT_CORPORATE_SCENES]),
    ("images", []),
    ("scene_videos", []),
    ("tab2_baked_subtitles", False),
    ("audio_files", []),
    ("custom_audio_paths", []),
    ("own_voice_table", {}),
    ("locked_character_prompt", DEFAULT_CHAR_PROMPT),
    ("exact_character_count", 1),
    ("lock_character_enabled", True),
    ("kannada_subtitles", {}),
    ("populated_script_lines", []),
    ("populated_script_lang", "English"),
    ("gen_batch_id", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

for idx, sc in enumerate(st.session_state.scenes):
    st.session_state.setdefault(f"nar_{idx}", sc.get("narration", ""))
    st.session_state.setdefault(f"kn_script_{idx}", sc.get("kannada_narration", ""))
    st.session_state.setdefault(f"vis_{idx}", sc.get("visual", ""))
    st.session_state.setdefault(f"emo_{idx}", sc.get("emotion", "Dense Emotion"))


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


def apply_audio(clip, audio):
    return clip.with_audio(audio) if MOVIEPY_V2 else clip.set_audio(audio)


def ensure_exact_size(pil_img, target_w, target_h):
    """Force any image onto an exact (target_w, target_h) canvas without aspect distortion."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    rgb = pil_img.convert("RGB")
    if rgb.size == (target_w, target_h):
        return rgb
    scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
    nw = max(target_w, int(round(rgb.width * scale)))
    nh = max(target_h, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), resampling)
    left = max(0, (nw - target_w) // 2)
    top = max(0, (nh - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h)).resize(
        (target_w, target_h), resampling
    )


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
    """Build up to 10 distinct emotional corporate scenes when Gemini is unavailable or has no quota."""
    target_count = max(1, min(10, int(count)))
    clean_story = " ".join(rough_text.strip().split())
    sentences = [
        s.strip()
        for s in clean_story.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    emotions_and_settings = [
        ("Overwhelmed & Tense", "at a glowing computer desk surrounded by spreadsheets inside a corporate office"),
        ("Shocked & Guarded", "inside a glass-walled corporate meeting room with a conference table and presentation screen"),
        ("Sorrowful & Drained", "standing beside tall corporate office windows overlooking city buildings at dusk"),
        ("Angry & Restrained", "by a modern office cubicle partition holding files with tight posture"),
        ("Anxious & Restless", "pacing near the office printer and coffee station under bright fluorescent lights"),
        ("Vulnerable & Quiet", "sitting alone in a corner lounge area of the corporate office looking down"),
        ("Defiant & Focused", "reviewing urgent contracts at his desk with a sharp, steady gaze"),
        ("Exhausted & Reflective", "leaning back in his office chair during late evening hours with soft desk lamp light"),
        ("Hopeful & Renewed", "standing near bright morning windows inside the corporate workspace"),
        ("Resolute & Composed", "seated upright at his corporate workstation with a notebook and coffee cup, ready for action"),
    ]
    scenes = []
    for idx in range(target_count):
        base_line = sentences[idx % len(sentences)] if sentences else clean_story
        emo_label, setting_desc = emotions_and_settings[idx % len(emotions_and_settings)]
        scenes.append({
            "narration": f"Scene {idx + 1}: {base_line}.",
            "kannada_narration": "",
            "emotion": emo_label,
            "visual": (
                f"Medium shot {setting_desc}; {char_prompt}, showing {emo_label.lower()} "
                f"facial reaction and body posture tied to: {base_line}"
            ),
        })
    return scenes


def sync_scenes_to_count(existing_scenes, desired_count, rough_text, char_prompt):
    """Ensure st.session_state.scenes has the exact desired_count (up to 10 scenes)."""
    target_n = max(1, min(10, int(desired_count)))
    base_story = (
        rough_text.strip()
        if rough_text and rough_text.strip()
        else "A 35-year-old man in a corporate office experiences dense emotions and tense reactions"
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


def compute_audio_envelope(audio_path, num_steps):
    """Compute normalized speech energy (0.0 to 1.0) across steps for lip-sync cadence."""
    if not audio_path or not os.path.exists(audio_path):
        return [0.45 + 0.35 * abs(math.sin(i * 1.3)) for i in range(num_steps)]
    try:
        clip = AudioFileClip(audio_path)
        try:
            samples = clip.to_soundarray(fps=4000)
            if samples is None or len(samples) == 0:
                raise ValueError("Empty audio samples")
            mono = np.mean(np.abs(samples), axis=1) if samples.ndim > 1 else np.abs(samples)
            peak = float(np.max(mono)) if len(mono) > 0 else 0.0
            if peak <= 1e-6:
                return [0.15] * num_steps

            env = []
            chunk_len = max(1, len(mono) // num_steps)
            for idx in range(num_steps):
                start = idx * chunk_len
                end = len(mono) if idx == num_steps - 1 else min(len(mono), (idx + 1) * chunk_len)
                seg_rms = float(np.sqrt(np.mean(mono[start:end] ** 2))) if end > start else 0.0
                norm_val = min(1.0, max(0.08, (seg_rms / peak) * 1.45))
                env.append(norm_val)
            return env
        finally:
            clip.close()
    except Exception:
        return [0.35 + 0.45 * abs(math.sin(i * 1.2)) for i in range(num_steps)]


def draw_lipsync_indicator(frame_pil, speech_energy, target_w, target_h, enable_lipsync=True):
    """Draw a subtle animated mouth/lip-sync pulse indicator synced to voiceover energy."""
    if not enable_lipsync:
        return frame_pil

    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    mouth_open = max(2, int(round(4 + speech_energy * 14)))
    mouth_w = max(18, int(round(24 + speech_energy * 12)))
    cx = target_w // 2
    cy = int(round(target_h * 0.72))

    badge_w = mouth_w + 26
    badge_h = max(26, mouth_open + 16)
    draw.rounded_rectangle(
        [cx - badge_w // 2, cy - badge_h // 2, cx + badge_w // 2, cy + badge_h // 2],
        radius=badge_h // 2,
        fill=(15, 20, 30, 145),
        outline=(255, 210, 120, 180),
        width=1,
    )
    draw.ellipse(
        [cx - mouth_w // 2, cy - mouth_open // 2, cx + mouth_w // 2, cy + mouth_open // 2],
        fill=(220, 85, 95, 225) if speech_energy > 0.22 else (140, 95, 100, 170),
    )
    return img_rgba.convert("RGB")


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
    audio_path=None,
    enable_lipsync=True,
    enable_char_motion=True,
):
    """Create animated scene frames with exact target_w x target_h dimensions."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    overscan = 1.16
    canvas_w = int(round(target_w * overscan))
    canvas_h = int(round(target_h * overscan))

    with Image.open(img_path) as im:
        exact_base = ensure_exact_size(im, canvas_w, canvas_h)
        canvas_np = np.asarray(exact_base, dtype=np.uint8)

    safe_dur = max(float(duration), 0.5)
    num_steps = max(6, min(20, int(round(safe_dur * 4))))
    step_dur = safe_dur / num_steps
    speech_env = compute_audio_envelope(audio_path, num_steps)
    subclips = []

    os.makedirs("motion_frames", exist_ok=True)
    for s_idx in range(num_steps):
        progress = s_idx / max(1, num_steps - 1)
        ease = progress * progress * (3.0 - 2.0 * progress)

        char_sway_x = int(round(math.sin(s_idx * 1.1 + scene_idx) * 6)) if enable_char_motion else 0
        char_bob_y = int(round(abs(math.cos(s_idx * 1.4 + scene_idx)) * 5)) if enable_char_motion else 0

        if active_mode == "Zoom In":
            scale = 1.0 + 0.12 * ease
            win_w = max(2, int(round(target_w / scale)))
            win_h = max(2, int(round(target_h / scale)))
            x0 = (canvas_w - win_w) // 2 + char_sway_x
            y0 = (canvas_h - win_h) // 2 + char_bob_y
        elif active_mode == "Zoom Out":
            scale = 1.12 - 0.12 * ease
            win_w = max(2, int(round(target_w / scale)))
            win_h = max(2, int(round(target_h / scale)))
            x0 = (canvas_w - win_w) // 2 + char_sway_x
            y0 = (canvas_h - win_h) // 2 + char_bob_y
        elif active_mode == "Pan Right":
            win_w = target_w
            win_h = target_h
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * ease)) + char_sway_x
            y0 = max(0, (canvas_h - win_h) // 2) + char_bob_y
        else:
            win_w = target_w
            win_h = target_h
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * (1.0 - ease))) + char_sway_x
            y0 = max(0, (canvas_h - win_h) // 2) + char_bob_y

        x0 = min(max(0, x0), max(0, canvas_w - win_w))
        y0 = min(max(0, y0), max(0, canvas_h - win_h))
        cropped = canvas_np[y0 : y0 + win_h, x0 : x0 + win_w]

        frame_pil = Image.fromarray(cropped)
        if frame_pil.size != (target_w, target_h):
            frame_pil = frame_pil.resize((target_w, target_h), resampling)

        if enable_lipsync:
            frame_pil = draw_lipsync_indicator(
                frame_pil,
                speech_energy=speech_env[s_idx],
                target_w=target_w,
                target_h=target_h,
                enable_lipsync=enable_lipsync,
            )

        if subtitle_lang != "None" and subtitle_text:
            frame_pil = draw_subtitle_on_frame(
                frame_pil,
                subtitle_text=subtitle_text,
                target_w=target_w,
                target_h=target_h,
                lang=subtitle_lang,
            )

        frame_pil = ensure_exact_size(frame_pil, target_w, target_h)
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
    audio_path=None,
    enable_lipsync=True,
    enable_char_motion=True,
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
        audio_path=audio_path,
        enable_lipsync=enable_lipsync,
        enable_char_motion=enable_char_motion,
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
    """Distinct corporate scene storyboard frame when external AI image APIs fail/quota-out."""
    palettes = [
        ((32, 46, 72), (56, 76, 108), (20, 26, 38)),
        ((46, 38, 68), (82, 66, 106), (24, 20, 34)),
        ((28, 44, 62), (118, 86, 68), (18, 24, 34)),
        ((58, 34, 42), (96, 54, 62), (24, 18, 24)),
        ((34, 48, 64), (72, 92, 114), (20, 28, 36)),
        ((28, 56, 68), (64, 104, 112), (18, 32, 38)),
        ((52, 40, 34), (94, 72, 58), (22, 18, 16)),
        ((26, 38, 58), (52, 74, 102), (16, 22, 32)),
        ((36, 56, 52), (68, 102, 92), (18, 30, 28)),
        ((42, 46, 76), (78, 84, 126), (20, 22, 38)),
    ]
    top_col, mid_col, bot_col = palettes[scene_idx % len(palettes)]

    y_grid = np.linspace(0.0, 1.0, target_h, dtype=np.float32)[:, None, None]
    x_grid = np.linspace(0.0, 1.0, target_w, dtype=np.float32)[None, :, None]
    c_top = np.array(top_col, dtype=np.float32)[None, None, :]
    c_mid = np.array(mid_col, dtype=np.float32)[None, None, :]
    c_bot = np.array(bot_col, dtype=np.float32)[None, None, :]

    blend_upper = np.clip(y_grid / 0.56, 0.0, 1.0)
    blend_lower = np.clip((y_grid - 0.56) / 0.44, 0.0, 1.0)
    sky_wall = c_top * (1.0 - blend_upper) + c_mid * blend_upper
    full_bg = sky_wall * (1.0 - blend_lower) + c_bot * blend_lower

    lamp_cx = target_w * (0.22 + 0.14 * (scene_idx % 5))
    lamp_cy = target_h * (0.22 + 0.04 * (scene_idx % 3))
    dist_sq = (x_grid[:, :, 0] - lamp_cx) ** 2 + (y_grid[:, :, 0] - lamp_cy) ** 2
    lamp_r = min(target_w, target_h) * 0.18
    glow = np.exp(-dist_sq / (2.0 * lamp_r ** 2))[:, :, None]
    full_bg = np.clip(full_bg + glow * np.array([52.0, 46.0, 38.0], dtype=np.float32), 0, 255)

    img = Image.fromarray(full_bg.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    floor_y = int(target_h * 0.64)
    variant = scene_idx % 4
    if variant == 0:
        draw.rectangle([int(target_w * 0.08), floor_y, int(target_w * 0.92), int(target_h * 0.72)], fill=(22, 28, 38, 220))
        for mx in [0.16, 0.66]:
            draw.rounded_rectangle(
                [int(target_w * mx), int(target_h * 0.44), int(target_w * (mx + 0.18)), int(target_h * 0.58)],
                radius=8,
                fill=(45, 78, 112, 200),
                outline=(120, 175, 220, 180),
                width=2,
            )
    elif variant == 1:
        for gx in [0.12, 0.36, 0.64, 0.88]:
            draw.line([(int(target_w * gx), int(target_h * 0.14)), (int(target_w * gx), floor_y)], fill=(150, 180, 210, 110), width=3)
        draw.polygon(
            [(int(target_w * 0.12), floor_y + 15), (int(target_w * 0.88), floor_y + 15), (target_w, int(target_h * 0.76)), (0, int(target_h * 0.76))],
            fill=(30, 36, 48, 215),
        )
    elif variant == 2:
        wx0, wy0, wx1, wy1 = int(target_w * 0.54), int(target_h * 0.16), int(target_w * 0.92), floor_y
        draw.rectangle([wx0, wy0, wx1, wy1], fill=(24, 34, 48, 185), outline=(140, 165, 195, 160), width=2)
        for bx in range(1, 4):
            px = wx0 + int((wx1 - wx0) * bx / 4)
            draw.line([(px, wy0), (px, wy1)], fill=(120, 145, 175, 120), width=2)
    else:
        for px in [0.25, 0.75]:
            draw.line([(int(target_w * px), 0), (int(target_w * px), int(target_h * 0.12))], fill=(180, 190, 205, 140), width=2)
            draw.ellipse([int(target_w * px) - 24, int(target_h * 0.12) - 6, int(target_w * px) + 24, int(target_h * 0.12) + 8], fill=(255, 225, 165, 190))

    cx = int(target_w * (0.44 + 0.08 * math.sin(scene_idx * 1.7)))
    head_r = max(28, int(min(target_w, target_h) * 0.055))
    slump_y = int(round(math.sin(scene_idx * 1.3) * 12))
    head_cy = int(target_h * 0.43) + slump_y

    sh_w = int(head_r * 2.5)
    sh_h = int(head_r * 3.1)
    draw.rounded_rectangle(
        [cx - sh_w, head_cy + head_r + 6, cx + sh_w, head_cy + head_r + 6 + sh_h],
        radius=26,
        fill=(34, 40, 52, 235),
        outline=(95, 112, 138, 160),
        width=2,
    )
    draw.polygon(
        [(cx - 18, head_cy + head_r + 8), (cx + 18, head_cy + head_r + 8), (cx, head_cy + head_r + 56)],
        fill=(220, 228, 238, 220),
    )

    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=(206, 162, 132, 245))
    draw.pieslice([cx - head_r, head_cy - head_r - 4, cx + head_r, head_cy + head_r - 8], 185, 355, fill=(38, 32, 28, 255))

    eye_y = head_cy - 2
    eye_dx = max(10, int(head_r * 0.36))
    emo_txt = (emotion_label or "").lower()
    if any(k in emo_txt for k in ["shock", "disbelief", "surpris"]):
        draw.ellipse([cx - eye_dx - 6, eye_y - 6, cx - eye_dx + 6, eye_y + 6], fill=(250, 250, 250, 255))
        draw.ellipse([cx + eye_dx - 6, eye_y - 6, cx + eye_dx + 6, eye_y + 6], fill=(250, 250, 250, 255))
        draw.ellipse([cx - eye_dx - 3, eye_y - 3, cx - eye_dx + 3, eye_y + 3], fill=(25, 25, 30, 255))
        draw.ellipse([cx + eye_dx - 3, eye_y - 3, cx + eye_dx + 3, eye_y + 3], fill=(25, 25, 30, 255))
        draw.ellipse([cx - 8, head_cy + 14, cx + 8, head_cy + 26], outline=(110, 58, 52, 255), width=2)
    elif any(k in emo_txt for k in ["sorrow", "sad", "drain", "vulnerab", "heavy"]):
        draw.line([(cx - eye_dx - 7, eye_y - 6), (cx - eye_dx + 6, eye_y + 1)], fill=(45, 36, 32, 255), width=2)
        draw.line([(cx + eye_dx - 6, eye_y + 1), (cx + eye_dx + 7, eye_y - 6)], fill=(45, 36, 32, 255), width=2)
        draw.arc([cx - 12, head_cy + 16, cx + 12, head_cy + 28], 190, 350, fill=(105, 56, 52, 255), width=2)
    else:
        draw.line([(cx - eye_dx - 7, eye_y + 2), (cx - eye_dx + 6, eye_y - 5)], fill=(45, 36, 32, 255), width=2)
        draw.line([(cx + eye_dx - 6, eye_y - 5), (cx + eye_dx + 7, eye_y + 2)], fill=(45, 36, 32, 255), width=2)
        draw.line([(cx - 10, head_cy + 18), (cx + 10, head_cy + 18)], fill=(105, 56, 52, 255), width=2)

    card_w = int(target_w * 0.88)
    card_h = int(target_h * 0.19)
    cx0 = (target_w - card_w) // 2
    cy0 = int(target_h * 0.72)
    draw.rounded_rectangle(
        [cx0, cy0, cx0 + card_w, cy0 + card_h],
        radius=18,
        fill=(10, 15, 24, 205),
        outline=(235, 195, 115, 185),
        width=2,
    )

    title_font, _ = _find_font(max(20, int(target_w * 0.028)), lang="English")
    body_font, _ = _find_font(max(17, int(target_w * 0.023)), lang="English")
    emo_badge = f" • {emotion_label.upper()}" if emotion_label else ""
    draw.text(
        (cx0 + 18, cy0 + 12),
        f"SCENE {scene_idx + 1}{emo_badge}",
        font=title_font,
        fill=(255, 218, 138, 255),
    )

    clean_action = " ".join((prompt_text or "").split())
    wrapped = textwrap.wrap(clean_action, width=max(26, int(card_w / max(10, int(target_w * 0.023) * 0.56))))[:3]
    cur_y = cy0 + 46
    for line in wrapped:
        draw.text((cx0 + 18, cur_y), line, font=body_font, fill=(238, 244, 250, 245))
        cur_y += int(target_w * 0.027)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    return ensure_exact_size(combined, target_w, target_h)


def modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, art_style):
    """Adapt an uploaded reference photo/image into an exact target_w x target_h scene frame."""
    base = ensure_exact_size(ref_pil, target_w, target_h)
    style_lower = (art_style or "").lower()
    if "ghibli" in style_lower or "watercolor" in style_lower:
        framed = base.filter(ImageFilter.SMOOTH_MORE)
        framed = ImageEnhance.Color(framed).enhance(1.18)
    else:
        framed = ImageEnhance.Contrast(base).enhance(1.12)
        framed = ImageEnhance.Color(framed).enhance(1.14)

    mood_factors = [1.0, 1.05, 0.96, 1.03, 0.98, 1.05, 1.02, 0.95, 1.06, 1.01]
    framed = ImageEnhance.Brightness(framed).enhance(mood_factors[scene_idx % len(mood_factors)])
    return ensure_exact_size(framed, target_w, target_h)


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
        return ensure_exact_size(img, target_w, target_h)

    data = resp.json()
    result = data.get("result", {}) if isinstance(data, dict) else {}
    b64_img = result.get("image")
    if b64_img:
        raw_bytes = base64.b64decode(b64_img)
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        return ensure_exact_size(img, target_w, target_h)

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
                return ensure_exact_size(img, target_w, target_h)
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
                return ensure_exact_size(img.convert("RGB"), target_w, target_h), f"HF {model_id}"
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
                            return ensure_exact_size(img, target_w, target_h), f"Gemini {model_id}"
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


st.title("🎬 AI Animated Story Studio")
st.caption(
    "Up to 10 scenes, multi-engine AI scene failover, local bypass fallback, locked character identity, interactive English & Kannada story editor, per-scene video creation with subtitles & lip-sync, voiceovers, and exact-size MP4 export."
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
    aspect_ratio = st.selectbox("Video Format", ["9:16 (Reels / Shorts)", "16:9 (YouTube)"])
    if "9:16" in aspect_ratio:
        width, height, ratio_str = 720, 1280, "9:16"
    else:
        width, height, ratio_str = 1280, 720, "16:9"

    st.divider()
    st.subheader("🔒 Locked Character & Cast Identity")
    st.session_state.lock_character_enabled = st.checkbox(
        "Lock character count, faces & outfits across scenes",
        value=st.session_state.lock_character_enabled,
    )
    st.session_state.exact_character_count = st.number_input(
        "Exact Main Character Count",
        min_value=1,
        max_value=8,
        value=int(st.session_state.exact_character_count),
        step=1,
    )
    st.session_state.locked_character_prompt = st.text_area(
        "Locked Character & Face Prompt",
        value=st.session_state.locked_character_prompt,
        height=110,
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Rough Story & Editor",
    "2️⃣ Scene Videos",
    "3️⃣ Voice & Sound",
    "4️⃣ Render & Download",
])

# --- TAB 1: SCRIPT WRITER & INTERACTIVE EDITING AREA (ENGLISH + KANNADA) ---
with tab1:
    st.subheader("Write Your Rough Idea & Edit English / Kannada Script")
    rough_story = st.text_area(
        "Type your rough story in any style (even broken sentences):",
        value="A 35-year-old man in a corporate office experiences dense emotions and tense reactions through high-stakes work situations, sudden setbacks, and quiet inner resolve.",
        height=110,
    )
    num_scenes = st.slider("Number of Scenes", 2, 10, max(2, min(10, len(st.session_state.scenes) or 6)))

    if st.button("✨ Polish Story & Create Scenes", type="primary"):
        if not rough_story.strip():
            st.error("Please write a rough story first.")
        elif not api_key or not api_key.strip():
            st.session_state.scenes = build_fallback_scenes_from_rough_story(
                rough_story, num_scenes, st.session_state.locked_character_prompt
            )
            for idx, sc in enumerate(st.session_state.scenes):
                st.session_state[f"nar_{idx}"] = sc["narration"]
                st.session_state[f"kn_script_{idx}"] = sc["kannada_narration"]
                st.session_state[f"vis_{idx}"] = sc["visual"]
                st.session_state[f"emo_{idx}"] = sc.get("emotion", "Dense Emotion")
            st.warning(f"No Gemini API Key provided—created {len(st.session_state.scenes)} distinct emotional corporate scenes directly from your rough story.")
        else:
            fallback_models = [
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
            ]
            char_context = (
                f"\nStrict cast lock: Keep exact character count ({st.session_state.exact_character_count} main character(s)) and consistent identity/outfit across every scene: {st.session_state.locked_character_prompt}"
                if st.session_state.lock_character_enabled
                else ""
            )
            prompt = f"""
            Fix and rewrite this rough story into an engaging {num_scenes}-scene story script with distinct emotional reactions and corporate office settings per scene.{char_context}
            Return ONLY valid JSON as a list of {num_scenes} objects, with no markdown formatting.
            Each object must have:
            - "narration": clear, emotional English voiceover text (1-2 sentences)
            - "kannada_narration": natural, clear Kannada translation/script of that scene's voiceover (1-2 sentences)
            - "emotion": specific emotion & facial reaction for that scene (e.g., "Overwhelmed & Tense", "Shocked & Guarded", "Sorrowful & Drained", "Resolute & Composed")
            - "visual": distinct medium or wide corporate office visual description (specific office spot, camera framing, lighting, posture, and facial expression)
            Rough story: {rough_story.strip()}
            """
            client = genai.Client(api_key=api_key.strip())
            last_exc = None
            parsed_scenes = None

            with st.spinner(f"Polishing {num_scenes} English & Kannada story scenes..."):
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
                        "emotion": str(item.get("emotion", "Dense Emotion")).strip(),
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
                    st.session_state[f"emo_{idx}"] = sc.get("emotion", "Dense Emotion")
                st.warning(f"Gemini quota/error hit ({last_exc})—created {len(st.session_state.scenes)} distinct fallback scenes.")

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
                        f"Scene {i + 1} Emotion / Facial Reaction",
                        value=scene.get("emotion", "Dense Emotion"),
                        key=f"emo_{i}",
                    )
                    scene["emotion"] = emo_val
                    vis_val = st.text_input(
                        f"Scene {i + 1} Action / Camera Visual",
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

# --- TAB 2: SCENE-WISE VIDEO CREATOR (WITH SUBTITLES & LIP-SYNC) ---
with tab2:
    st.subheader("Create Scene-by-Scene Videos (Subtitles & Lip-Sync)")
    art_style = st.selectbox(
        "Choose Visual Style",
        [
            "Photorealistic cinematic film still, medium or wide framing showing full head and shoulders, natural corporate lighting, lifelike human expressions, 35mm lens",
            "High-detail 3D CGI animation, expressive lifelike characters, rich corporate office lighting and depth",
            "Detailed storybook illustration, expressive characters, modern corporate office scenery",
        ],
    )

    ref_col1, ref_col2 = st.columns([1.4, 1])
    with ref_col1:
        active_char_prompt = st.text_area(
            "Active Character & Face Prompt (Synced with Locked Character Prompt)",
            value=st.session_state.locked_character_prompt,
            height=90,
        )
        if st.session_state.lock_character_enabled:
            st.session_state.locked_character_prompt = active_char_prompt
    with ref_col2:
        ref_file = st.file_uploader(
            "Character Reference Image (Optional)",
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

    sub_lip_col1, sub_lip_col2, sub_lip_col3 = st.columns([1.2, 1.1, 1.7])
    with sub_lip_col1:
        tab2_sub_lang = st.selectbox(
            "Scene Video Subtitles",
            ["None", "English", "Kannada"],
            index=2,
            key="tab2_subtitle_lang",
        )
    with sub_lip_col2:
        tab2_lipsync = st.checkbox(
            "Scene Lip-Sync Cadence",
            value=True,
            key="tab2_enable_lipsync",
        )
    with sub_lip_col3:
        allow_local_bypass = st.checkbox(
            "Enable zero-API local bypass frames if all external image APIs fail/quota-out",
            value=True,
        )

    st.session_state["tab2_baked_subtitles"] = tab2_sub_lang != "None"

    ref_pil = None
    if ref_file is not None:
        ref_pil = Image.open(ref_file).convert("RGB")
        st.image(ref_pil, caption="Uploaded Character Reference Image", width=260)

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
            st.session_state.setdefault(f"emo_{idx}", sc.get("emotion", "Dense Emotion"))

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

        char_to_use = (
            st.session_state.locked_character_prompt
            if st.session_state.lock_character_enabled
            else active_char_prompt
        )
        cast_guard = (
            f"Exact character count: {int(st.session_state.exact_character_count)} main subject. "
            f"Keep consistent identity and outfit from {char_to_use}, framed with headroom so face and eyes are never cropped."
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
            scene_emo = st.session_state.get(f"emo_{idx}", scene.get("emotion", "Dense Emotion"))
            full_prompt = (
                f"{art_style}. {cast_guard} "
                f"Scene {idx + 1} emotion: {scene_emo}. "
                f"Scene {idx + 1} action & office setting: {scene_vis}"
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
                exact_img = ensure_exact_size(pil_img, width, height)
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
                    audio_path=scene_aud,
                    enable_lipsync=tab2_lipsync,
                    enable_char_motion=True,
                )
                st.session_state.scene_videos.append(vid_path)

            progress.progress((idx + 1) / len(active_scenes))

        if st.session_state.scene_videos:
            st.success(
                f"Created {len(st.session_state.scene_videos)} scene videos at {width}x{height} "
                f"(Subtitles: {tab2_sub_lang}, Lip-Sync: {'On' if tab2_lipsync else 'Off'}, Sources: {', '.join(set(used_sources))})."
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
    st.subheader("Select Narrator Voice & Script Language (English or Kannada)")
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
                            "ಮೂವತ್ತೈದರ ವಯಸ್ಸಿನಲ್ಲಿ, ಕಚೇರಿಯ ಪ್ರತಿಯೊಂದು ಗಡುವು ಮೊದಲಿಗಿಂತ ಹೆಚ್ಚು ಭಾರವಾಗಿ ಭಾಸವಾಗುತ್ತದೆ."
                            if selected_script_lang == "Kannada"
                            else "At thirty-five, every deadline in the corporate office feels heavier than the last."
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
    st.subheader("Character Animation, Lip-Sync, Subtitles & Exact-Size MP4 Export")
    c1, c2, c3, c4 = st.columns(4)
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
    with c4:
        enable_lipsync = st.checkbox("Voice Lip-Sync Cadence", value=True)

    _, active_font_source = _find_font(28, lang=subtitle_lang if subtitle_lang != "None" else "English")
    st.caption(
        f"Target output size: `{width}x{height}` ({ratio_str}) | Scene Videos ready: `{len(st.session_state.scene_videos)}` | Scene Images ready: `{len(st.session_state.images)}` | Active font: `{active_font_source}`"
    )

    if st.button("🎬 Render Final MP4 Video", type="primary"):
        has_visuals = bool(st.session_state.scene_videos or st.session_state.images)
        if not has_visuals or not st.session_state.audio_files:
            st.error("Make sure both Scene Videos (Tab 2) and Narration (Tab 3) are generated or saved.")
        else:
            clips = []
            audio_clips = []
            source_video_clips = []
            final_video = None
            output_file = f"animated_story_b{st.session_state.gen_batch_id}.mp4"

            try:
                with st.spinner(
                    f"Stitching {max(len(st.session_state.scene_videos), len(st.session_state.images))}-scene {width}x{height} video with voiceovers and {subtitle_lang} subtitles..."
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
                        sub_lines = [""] * len(st.session_state.audio_files)

                    while len(sub_lines) < len(st.session_state.audio_files):
                        sub_lines.append("")

                    scene_total = max(len(st.session_state.scene_videos), len(st.session_state.images))
                    for i, aud_p in enumerate(st.session_state.audio_files):
                        if i >= scene_total:
                            break
                        audio_clip = AudioFileClip(aud_p)
                        audio_clips.append(audio_clip)
                        target_duration = max(float(audio_clip.duration or 1.0) + 0.3, 1.0)

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
                                subtitle_text=sub_lines[i] if i < len(sub_lines) else "",
                                subtitle_lang=subtitle_lang,
                                audio_path=aud_p,
                                enable_lipsync=enable_lipsync,
                                enable_char_motion=enable_char_motion,
                            )

                        scene_clip = apply_audio(motion_clip, audio_clip)
                        clips.append(scene_clip)

                    final_video = concatenate_videoclips(clips, method="compose")
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
