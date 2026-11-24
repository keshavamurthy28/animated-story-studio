def draw_illustrated_characters(draw, target_w, target_h, scene_idx, char_desc):
    """Draw stylized character silhouettes/figures (4 boys + giant bear) so scenes show characters."""
    desc_l = (char_desc or "").lower()
    ground_y = int(target_h * 0.76)

    # Draw 4 boys with distinct builds (slim, lean, taller/medium, chubby/fat)
    boy_specs = [
        ("slim", int(target_w * 0.16), 22, 118, (66, 135, 245)),
        ("lean", int(target_w * 0.27), 24, 124, (46, 184, 138)),
        ("medium", int(target_w * 0.38), 26, 128, (235, 162, 52)),
        ("fat", int(target_w * 0.50), 40, 122, (219, 92, 122)),
    ]

    # If scene is about climbing a tree, raise two boys higher
    climbing = "climb" in desc_l or scene_idx == 2
    lying_down = scene_idx == 2

    for i, (build, bx, half_w, height_px, shirt_col) in enumerate(boy_specs):
        by = ground_y
        if climbing and i in (0, 1):
            by = int(target_h * 0.48) - i * 35
        elif lying_down and i == 3:
            # Boy lying flat on ground
            draw.rounded_rectangle(
                [bx - 45, ground_y - 24, bx + 55, ground_y],
                radius=10,
                fill=shirt_col,
                outline=(25, 25, 35),
                width=2,
            )
            draw.ellipse(
                [bx + 45, ground_y - 26, bx + 73, ground_y + 2],
                fill=(245, 196, 155),
                outline=(25, 25, 35),
                width=2,
            )
            continue

        torso_top = by - height_px + 34
        # Legs
        draw.rectangle([bx - half_w + 4, by - 42, bx - 3, by], fill=(42, 54, 78))
        draw.rectangle([bx + 3, by - 42, bx + half_w - 4, by], fill=(42, 54, 78))
        # Torso / shirt
        draw.rounded_rectangle(
            [bx - half_w, torso_top, bx + half_w, by - 38],
            radius=12,
            fill=shirt_col,
            outline=(20, 24, 32),
            width=2,
        )
        # Head & hair
        head_r = 17
        head_cy = torso_top - head_r + 2
        draw.ellipse(
            [bx - head_r, head_cy - head_r, bx + head_r, head_cy + head_r],
            fill=(245, 196, 155),
            outline=(20, 24, 32),
            width=2,
        )
        draw.pieslice(
            [bx - head_r, head_cy - head_r - 3, bx + head_r, head_cy + 4],
            start=180,
            end=360,
            fill=(38, 28, 22),
        )

    # Draw giant bulky bear on the right side
    bear_cx = int(target_w * 0.78)
    bear_cy = ground_y - 78
    draw.ellipse(
        [bear_cx - 68, bear_cy - 56, bear_cx + 68, bear_cy + 78],
        fill=(104, 62, 36),
        outline=(35, 20, 12),
        width=3,
    )
    # Bear head & ears
    draw.ellipse(
        [bear_cx - 44, bear_cy - 102, bear_cx + 24, bear_cy - 34],
        fill=(116, 70, 40),
        outline=(35, 20, 12),
        width=3,
    )
    draw.ellipse([bear_cx - 42, bear_cy - 114, bear_cx - 16, bear_cy - 88], fill=(92, 52, 28))
    draw.ellipse([bear_cx - 4, bear_cy - 114, bear_cx + 22, bear_cy - 88], fill=(92, 52, 28))
    # Eyes
    draw.ellipse([bear_cx - 26, bear_cy - 78, bear_cx - 16, bear_cy - 68], fill=(255, 255, 255))
    draw.ellipse([bear_cx - 23, bear_cy - 75, bear_cx - 18, bear_cy - 70], fill=(15, 15, 15))


def generate_direct_scene_image(gemini_key, prompt_text, ratio_str, target_w, target_h, scene_idx, ref_pil=None, char_desc=""):
    """Generate a scene image directly from description (and optional reference image), with character scene fallback."""
    if gemini_key and gemini_key.strip():
        try:
            client = genai.Client(api_key=gemini_key.strip())
            contents = [ref_pil.convert("RGB"), prompt_text] if ref_pil is not None else prompt_text
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
                                return Image.open(BytesIO(inline.data)).convert("RGB")
                except Exception:
                    continue
        except Exception:
            pass

    if ref_pil is not None:
        return modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, prompt_text)

    palettes = [
        ((46, 114, 158), (68, 148, 86), (250, 206, 104)),
        ((34, 72, 118), (52, 124, 74), (244, 166, 86)),
        ((58, 96, 134), (62, 134, 82), (248, 198, 118)),
        ((42, 88, 128), (58, 132, 78), (252, 212, 126)),
    ]
    c_sky, c_ground, c_sun = palettes[scene_idx % len(palettes)]
    img = Image.new("RGB", (target_w, target_h), c_sky)
    draw = ImageDraw.Draw(img)

    ground_y = int(target_h * 0.76)
    draw.ellipse([target_w - 150, 50, target_w - 50, 150], fill=c_sun)
    # Background trees
    for tx in range(30, target_w, 130):
        draw.rectangle([tx + 24, ground_y - 190, tx + 44, ground_y], fill=(78, 50, 32))
        draw.ellipse([tx - 15, ground_y - 260, tx + 83, ground_y - 140], fill=(42, 108, 62))

    draw.rectangle([0, ground_y, target_w, target_h], fill=c_ground)
    draw_illustrated_characters(draw, target_w, target_h, scene_idx, char_desc or prompt_text)

    draw.rectangle([24, target_h - 115, target_w - 24, target_h - 24], fill=(16, 22, 32))
    draw.text((42, target_h - 95), f"Scene {scene_idx + 1}", fill=c_sun)
    draw.text((42, target_h - 65), (char_desc or prompt_text)[:75], fill=(242, 246, 250))
    return img
