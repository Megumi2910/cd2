"""Generate high-resolution (512x512) Vietnamese traffic sign PNG textures (QCVN 41:2019/BGTVT).

Outputs to: webots_project/textures/signs/<class_name>.png
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "webots_project" / "textures" / "signs"

SIZE = 512
CENTER = SIZE // 2
BG_COLOR = (225, 228, 232, 255)  # Neutral metallic sign plate background for corners
RED = (212, 28, 28, 255)
BLUE = (18, 92, 188, 255)
WHITE = (250, 250, 250, 255)
BLACK = (22, 22, 26, 255)


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    return img, draw


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font, fill, y_offset: int = 0) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = CENTER - tw / 2 - bbox[0]
    y = CENTER - th / 2 - bbox[1] + y_offset
    draw.text((x, y), text, font=font, fill=fill)


def _octagon_points(radius: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(8):
        angle = math.radians(22.5 + i * 45.0)
        pts.append((CENTER + radius * math.cos(angle), CENTER + radius * math.sin(angle)))
    return pts


def _draw_stop() -> Image.Image:
    img, draw = _new_canvas()
    draw.polygon(_octagon_points(238), fill=RED)
    draw.polygon(_octagon_points(222), fill=WHITE)
    draw.polygon(_octagon_points(204), fill=RED)
    font = _load_bold_font(124)
    _draw_centered_text(draw, "STOP", font, WHITE)
    return img.convert("RGB")


def _draw_blue_circle_base() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img, draw = _new_canvas()
    draw.ellipse((14, 14, SIZE - 14, SIZE - 14), fill=BLUE)
    draw.ellipse((28, 28, SIZE - 28, SIZE - 28), outline=WHITE, width=10)
    return img, draw


def _draw_prohibition_base() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img, draw = _new_canvas()
    draw.ellipse((14, 14, SIZE - 14, SIZE - 14), fill=RED)
    draw.ellipse((74, 74, SIZE - 74, SIZE - 74), fill=WHITE)
    return img, draw


def _draw_bent_arrow(draw: ImageDraw.ImageDraw, direction: str, color, scale: float = 1.0) -> None:
    """Draw a clean 90-degree bent arrow pointing left or right."""
    s = scale
    cx, cy = CENTER, CENTER
    stem_w = int(42 * s)
    if direction == "left":
        # Vertical stem from bottom up to bend
        stem_x = cx + int(48 * s)
        bend_y = cy - int(32 * s)
        draw.rectangle(
            (stem_x - stem_w // 2, bend_y, stem_x + stem_w // 2, cy + int(125 * s)),
            fill=color,
        )
        # Rounded elbow
        draw.ellipse(
            (stem_x - stem_w // 2, bend_y - stem_w // 2, stem_x + stem_w // 2, bend_y + stem_w // 2),
            fill=color,
        )
        # Horizontal arm left
        arm_left = cx - int(60 * s)
        draw.rectangle(
            (arm_left, bend_y - stem_w // 2, stem_x, bend_y + stem_w // 2),
            fill=color,
        )
        # Arrowhead pointing left
        head = [
            (arm_left - int(78 * s), bend_y),
            (arm_left + int(6 * s), bend_y - int(66 * s)),
            (arm_left + int(6 * s), bend_y + int(66 * s)),
        ]
        draw.polygon(head, fill=color)
    else:
        stem_x = cx - int(48 * s)
        bend_y = cy - int(32 * s)
        draw.rectangle(
            (stem_x - stem_w // 2, bend_y, stem_x + stem_w // 2, cy + int(125 * s)),
            fill=color,
        )
        draw.ellipse(
            (stem_x - stem_w // 2, bend_y - stem_w // 2, stem_x + stem_w // 2, bend_y + stem_w // 2),
            fill=color,
        )
        arm_right = cx + int(60 * s)
        draw.rectangle(
            (stem_x, bend_y - stem_w // 2, arm_right, bend_y + stem_w // 2),
            fill=color,
        )
        head = [
            (arm_right + int(78 * s), bend_y),
            (arm_right - int(6 * s), bend_y - int(66 * s)),
            (arm_right - int(6 * s), bend_y + int(66 * s)),
        ]
        draw.polygon(head, fill=color)


def _draw_turn_left() -> Image.Image:
    img, draw = _draw_blue_circle_base()
    _draw_bent_arrow(draw, "left", WHITE, scale=1.0)
    return img.convert("RGB")


def _draw_turn_right() -> Image.Image:
    img, draw = _draw_blue_circle_base()
    _draw_bent_arrow(draw, "right", WHITE, scale=1.0)
    return img.convert("RGB")


def _draw_go_straight() -> Image.Image:
    img, draw = _draw_blue_circle_base()
    cx, cy = CENTER, CENTER
    draw.rectangle((cx - 24, cy - 28, cx + 24, cy + 145), fill=WHITE)
    head = [
        (cx, cy - 148),
        (cx - 88, cy - 20),
        (cx + 88, cy - 20),
    ]
    draw.polygon(head, fill=WHITE)
    return img.convert("RGB")


def _draw_no_turn(direction: str) -> Image.Image:
    img, draw = _draw_prohibition_base()
    _draw_bent_arrow(draw, direction, BLACK, scale=0.80)
    # Red diagonal prohibition slash (top-left to bottom-right)
    draw.line((105, 105, SIZE - 105, SIZE - 105), fill=RED, width=38)
    return img.convert("RGB")


def _draw_speed_limit(value: str) -> Image.Image:
    img, draw = _draw_prohibition_base()
    font = _load_bold_font(200)
    _draw_centered_text(draw, value, font, BLACK, y_offset=-4)
    return img.convert("RGB")


def generate_all(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generators = {
        "stop": _draw_stop,
        "turn_left": _draw_turn_left,
        "turn_right": _draw_turn_right,
        "go_straight": _draw_go_straight,
        "no_left_turn": lambda: _draw_no_turn("left"),
        "no_right_turn": lambda: _draw_no_turn("right"),
        "speed_limit_30": lambda: _draw_speed_limit("30"),
        "speed_limit_50": lambda: _draw_speed_limit("50"),
    }
    created: list[Path] = []
    for name, fn in generators.items():
        out_path = output_dir / f"{name}.png"
        fn().save(out_path, format="PNG")
        created.append(out_path)
    return created


if __name__ == "__main__":
    files = generate_all()
    print(f"Generated {len(files)} Vietnamese traffic sign textures in {OUTPUT_DIR}:")
    for f in files:
        print(f"  - {f.name}")
