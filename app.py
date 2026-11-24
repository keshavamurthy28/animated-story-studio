def draw_subtitle_on_frame(
    frame_pil,
    subtitle_text,
    target_w,
    target_h,
    lang="English",
    position="Bottom",
    scale=0.85,
):
    """Draw a compact, high-legibility subtitle strip without covering the main subject."""
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


def compose_user_photos_into_frame(
    photo_paths,
    target_w,
    target_h,
    scene_idx=0,
    arrangement_cmd="",
    caption_text="",
    caption_lang="None",
    bg_mode="Blur Canvas",
):
    """Compose user photos cleanly without heavy borders or harsh color shifts."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    valid_paths = [p for p in (photo_paths or []) if p and os.path.exists(p)]
    if not valid_paths:
        return build_local_bypass_scene(arrangement_cmd, target_w, target_h, scene_idx)

    directives = parse_arrangement_directives(arrangement_cmd)
    primary_path = valid_paths[scene_idx % len(valid_paths)]
    secondary_path = valid_paths[(scene_idx + 1) % len(valid_paths)] if len(valid_paths) > 1 else None

    with Image.open(primary_path) as p_im:
        p_rgb = p_im.convert("RGB")

    # Clean blurred backdrop preserving natural colors
    bg_scale = max(target_w / max(1, p_rgb.width), target_h / max(1, p_rgb.height))
    bg_w = max(target_w, int(round(p_rgb.width * bg_scale)))
    bg_h = max(target_h, int(round(p_rgb.height * bg_scale)))
    bg = p_rgb.resize((bg_w, bg_h), resampling)
    bx = max(0, (bg_w - target_w) // 2)
    by = max(0, (bg_h - target_h) // 2)
    bg = bg.crop((bx, by, bx + target_w, by + target_h)).filter(ImageFilter.GaussianBlur(radius=16))
    bg = ImageEnhance.Brightness(bg).enhance(0.58)
    canvas = bg.copy()

    def _fit_box(src_img, box_w, box_h):
        sc = min(box_w / max(1, src_img.width), box_h / max(1, src_img.height))
        nw = max(1, int(round(src_img.width * sc)))
        nh = max(1, int(round(src_img.height * sc)))
        return src_img.resize((nw, nh), resampling)

    if directives["layout_mode"] == "Split Pop" and secondary_path:
        with Image.open(secondary_path) as s_im:
            s_rgb = s_im.convert("RGB")
        half_h = int(target_h * 0.40)
        box_w = int(target_w * 0.90)
        top_fg = _fit_box(p_rgb, box_w, half_h)
        bot_fg = _fit_box(s_rgb, box_w, half_h)
        gap = max(12, int(target_h * 0.02))
        total_stack_h = top_fg.height + gap + bot_fg.height
        top_y = max(16, (target_h - total_stack_h) // 2)
        bot_y = top_y + top_fg.height + gap
        canvas.paste(top_fg, ((target_w - top_fg.width) // 2, top_y))
        canvas.paste(bot_fg, ((target_w - bot_fg.width) // 2, bot_y))
    else:
        max_fw = int(target_w * 0.94)
        max_fh = int(target_h * 0.82)
        fg = _fit_box(p_rgb, max_fw, max_fh)
        fx = (target_w - fg.width) // 2
        fy = max(16, int(target_h * 0.07))
        canvas.paste(fg, (fx, fy))

    if caption_text and caption_text.strip() and caption_lang != "None":
        canvas = draw_subtitle_on_frame(
            canvas,
            subtitle_text=caption_text.strip(),
            target_w=target_w,
            target_h=target_h,
            lang=caption_lang,
            position="Bottom",
            scale=0.85,
        )

    return ensure_exact_size(canvas, target_w, target_h, bg_mode="Crop Fill")


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
    subtitle_scale=0.85,
    enable_char_motion=False,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Smooth, high-sharpness 24-fps motion clip using lossless PNG frames."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    overscan = 1.08
    canvas_w = int(round(target_w * overscan))
    canvas_h = int(round(target_h * overscan))

    with Image.open(img_path) as im:
        exact_base = ensure_exact_size(im, canvas_w, canvas_h, bg_mode=bg_mode)
        canvas_np = np.asarray(exact_base, dtype=np.uint8)

    safe_dur = max(float(duration), 0.5)
    # Render at smooth ~12–24 steps per clip with PNG so photo sharpness is preserved
    num_steps = max(12, min(48, int(round(safe_dur * 12))))
    step_dur = safe_dur / num_steps
    subclips = []

    os.makedirs("motion_frames", exist_ok=True)
    for s_idx in range(num_steps):
        progress = s_idx / max(1, num_steps - 1)
        ease = progress * progress * (3.0 - 2.0 * progress)

        if video_fx == "Zoom Punch":
            scale = 1.0 + 0.08 * math.sin(progress * math.pi)
        elif active_mode == "Zoom In":
            scale = 1.0 + 0.06 * ease
        elif active_mode == "Zoom Out":
            scale = 1.06 - 0.06 * ease
        else:
            scale = 1.04

        win_w = max(2, int(round(target_w / scale)))
        win_h = max(2, int(round(target_h / scale)))

        if active_mode == "Pan Right":
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * ease))
            y0 = max(0, (canvas_h - win_h) // 2)
        elif active_mode == "Pan Left":
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * (1.0 - ease)))
            y0 = max(0, (canvas_h - win_h) // 2)
        else:
            x0 = (canvas_w - win_w) // 2
            y0 = (canvas_h - win_h) // 2

        if video_fx == "Shake":
            x0 += int(round(math.sin(s_idx * 2.2) * 4))
            y0 += int(round(math.cos(s_idx * 2.7) * 3))

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
        frame_path = os.path.join("motion_frames", f"s{scene_idx}_f{s_idx}.png")
        frame_pil.save(frame_path, format="PNG")
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
    subtitle_scale=0.85,
    enable_char_motion=False,
    brightness=1.0,
    contrast=1.0,
    saturation=1.0,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Blur Canvas",
):
    """Render a crisp 24-fps MP4 scene clip."""
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
            ffmpeg_params=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"],
            logger=None,
        )
        return out_path
    finally:
        try:
            clip.close()
        except Exception:
            pass
