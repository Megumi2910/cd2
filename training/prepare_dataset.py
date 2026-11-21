"""Dataset preparation & domain-randomized synthetic generator for 8 VN traffic signs.

Supports two workflows:
1. Synthetic bootstrap (`--synthetic-per-class 250`): composites exact Webots sign textures
   onto diverse road-like backgrounds with perspective warp, lighting, blur, and occlusion
   so you can train a strong baseline model immediately.
2. Webots real capture merge (`--raw-dir datasets/raw_webots`): merges frames automatically
   collected via `dataset_collector` in Webots and performs stratified 70/20/10 train/val/test split.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXTURES_DIR = PROJECT_ROOT / "webots_project" / "textures" / "signs"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "vn_signs"

CLASS_NAMES = [
    "stop",
    "turn_left",
    "turn_right",
    "go_straight",
    "no_left_turn",
    "no_right_turn",
    "speed_limit_30",
    "speed_limit_50",
]


def _make_road_background(w: int = 640, h: int = 480) -> np.ndarray:
    """Create a varied synthetic outdoor road background matching Webots appearance."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = random.randint(int(h * 0.36), int(h * 0.50))

    # Sky gradient (daylight / overcast / sunset tint)
    sky_top = np.array(
        [random.randint(175, 235), random.randint(155, 210), random.randint(110, 175)],
        dtype=np.float32,
    )
    sky_bot = np.array(
        [random.randint(205, 248), random.randint(200, 240), random.randint(180, 230)],
        dtype=np.float32,
    )
    for y in range(horizon):
        t = y / max(1, horizon - 1)
        bg[y, :] = (sky_top * (1 - t) + sky_bot * t).astype(np.uint8)

    # Grass/ground shoulders
    grass_color = (
        random.randint(35, 75),
        random.randint(85, 135),
        random.randint(40, 85),
    )
    bg[horizon:, :] = grass_color

    # Asphalt road trapezoid
    vanish_x = random.randint(int(w * 0.35), int(w * 0.65))
    road_top_w = random.randint(30, 80)
    road_pts = np.array(
        [
            [vanish_x - road_top_w, horizon],
            [vanish_x + road_top_w, horizon],
            [w + 80, h],
            [-80, h],
        ],
        dtype=np.int32,
    )
    asphalt_val = random.randint(52, 92)
    cv2.fillPoly(bg, [road_pts], (asphalt_val, asphalt_val, asphalt_val + random.randint(-4, 4)))

    # Random road lane markings (white edge + yellow center)
    cv2.line(bg, (vanish_x, horizon), (int(w * 0.48), h), (35, 200, 235), random.randint(2, 5))
    cv2.line(bg, (vanish_x + road_top_w - 8, horizon), (int(w * 0.92), h), (235, 235, 235), random.randint(2, 5))

    # Random distant trees/buildings rectangles for clutter
    for _ in range(random.randint(3, 8)):
        bx = random.randint(0, w - 40)
        bw = random.randint(20, 75)
        bh = random.randint(25, 110)
        by = max(0, horizon - bh + random.randint(-5, 15))
        color = (
            random.randint(45, 150),
            random.randint(55, 155),
            random.randint(50, 145),
        )
        cv2.rectangle(bg, (bx, by), (min(w - 1, bx + bw), horizon + 8), color, -1)

    # Subtle sensor noise
    noise = np.random.normal(0, random.uniform(1.5, 5.0), bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def _warp_and_paste_sign(
    bg: np.ndarray, sign_bgr: np.ndarray
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Apply perspective tilt, lighting shift, pole, and paste sign onto background."""
    h, w = bg.shape[:2]
    target_size = random.randint(28, 128)
    resized = cv2.resize(sign_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Random perspective tilt (simulates camera viewing angle)
    s = float(target_size)
    jitter = s * random.uniform(0.02, 0.14)
    src = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
    dst = np.array(
        [
            [random.uniform(0, jitter), random.uniform(0, jitter)],
            [s - random.uniform(0, jitter), random.uniform(0, jitter)],
            [s - random.uniform(0, jitter), s - random.uniform(0, jitter)],
            [random.uniform(0, jitter), s - random.uniform(0, jitter)],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(resized, M, (target_size, target_size), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full((target_size, target_size), 255, dtype=np.uint8),
        M,
        (target_size, target_size),
        borderValue=0,
    )

    # Brightness / contrast jitter
    alpha = random.uniform(0.68, 1.25)
    beta = random.uniform(-25, 20)
    warped = np.clip(warped.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Optional slight blur (distance/motion)
    if random.random() < 0.35:
        k = random.choice([3, 5])
        warped = cv2.GaussianBlur(warped, (k, k), 0)

    # Placement (biased toward right roadside, plus full-width diversity)
    if random.random() < 0.7:
        px = random.randint(int(w * 0.48), max(int(w * 0.50), w - target_size - 8))
    else:
        px = random.randint(8, max(10, w - target_size - 8))
    py = random.randint(int(h * 0.12), max(int(h * 0.14), int(h * 0.58) - target_size))

    # Draw metallic sign pole below the sign
    pole_w = max(2, target_size // 12)
    pole_x = px + target_size // 2
    pole_bottom = min(h - 1, py + target_size + int(target_size * random.uniform(0.8, 2.2)))
    cv2.rectangle(
        bg,
        (pole_x - pole_w // 2, py + target_size - 2),
        (pole_x + pole_w // 2, pole_bottom),
        (140, 145, 150),
        -1,
    )

    # Composite warped sign
    roi = bg[py : py + target_size, px : px + target_size]
    m_bool = mask > 127
    roi[m_bool] = warped[m_bool]
    bg[py : py + target_size, px : px + target_size] = roi

    ys, xs = np.where(m_bool)
    x1 = px + int(xs.min())
    y1 = py + int(ys.min())
    x2 = px + int(xs.max())
    y2 = py + int(ys.max())
    return bg, (x1, y1, x2, y2)


def _bbox_to_yolo(bbox: Tuple[int, int, int, int], w: int, h: int) -> str:
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def generate_synthetic_samples(
    out_raw_dir: Path, per_class: int, seed: int = 42
) -> List[Tuple[Path, Path, int]]:
    """Generate synthetic images + YOLO labels using the exact Webots sign textures."""
    random.seed(seed)
    np.random.seed(seed)

    img_dir = out_raw_dir / "images"
    lbl_dir = out_raw_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    textures = {}
    for cls_id, name in enumerate(CLASS_NAMES):
        tex_path = TEXTURES_DIR / f"{name}.png"
        if not tex_path.exists():
            raise FileNotFoundError(
                f"Missing texture {tex_path}. Run `python training/generate_sign_textures.py` first."
            )
        textures[cls_id] = cv2.imread(str(tex_path))

    pairs: List[Tuple[Path, Path, int]] = []
    for cls_id, name in enumerate(CLASS_NAMES):
        for i in range(per_class):
            bg = _make_road_background(640, 480)
            lines = []

            # Primary sign of target class
            bg, bbox = _warp_and_paste_sign(bg, textures[cls_id])
            lines.append(f"{cls_id} {_bbox_to_yolo(bbox, 640, 480)}")

            # 20% chance of adding a second smaller sign of another class (multi-sign frame)
            if random.random() < 0.20:
                other_id = random.choice([c for c in range(len(CLASS_NAMES)) if c != cls_id])
                bg, bbox2 = _warp_and_paste_sign(bg, textures[other_id])
                # Avoid heavy overlap with first box
                ix1 = max(bbox[0], bbox2[0])
                iy1 = max(bbox[1], bbox2[1])
                ix2 = min(bbox[2], bbox2[2])
                iy2 = min(bbox[3], bbox2[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    lines.append(f"{other_id} {_bbox_to_yolo(bbox2, 640, 480)}")

            stem = f"syn_{name}_{i:04d}"
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"
            cv2.imwrite(str(img_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            pairs.append((img_path, lbl_path, cls_id))

    # Add ~6% pure negative (background-only) frames to suppress false positives
    num_neg = max(20, int(len(pairs) * 0.06))
    for i in range(num_neg):
        bg = _make_road_background(640, 480)
        stem = f"syn_negative_{i:04d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"
        cv2.imwrite(str(img_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        lbl_path.write_text("", encoding="utf-8")
        pairs.append((img_path, lbl_path, -1))

    return pairs


def collect_raw_pairs(raw_dir: Path) -> List[Tuple[Path, Path, int]]:
    img_dir = raw_dir / "images"
    lbl_dir = raw_dir / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        return []
    pairs: List[Tuple[Path, Path, int]] = []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        text = lbl_path.read_text(encoding="utf-8").strip()
        if not text:
            primary_cls = -1
        else:
            primary_cls = int(text.splitlines()[0].split()[0])
        pairs.append((img_path, lbl_path, primary_cls))
    return pairs


def build_yolo_dataset(
    pairs: List[Tuple[Path, Path, int]],
    out_dir: Path = DEFAULT_OUT_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Path:
    """Perform stratified train/val/test split and write Ultralytics dataset.yaml."""
    rng = random.Random(seed)
    by_class: dict[int, List[Tuple[Path, Path, int]]] = {}
    for item in pairs:
        by_class.setdefault(item[2], []).append(item)

    splits = {"train": [], "val": [], "test": []}
    for cls_id, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for split_name, items in splits.items():
        s_img = out_dir / "images" / split_name
        s_lbl = out_dir / "labels" / split_name
        if s_img.exists():
            shutil.rmtree(s_img)
        if s_lbl.exists():
            shutil.rmtree(s_lbl)
        s_img.mkdir(parents=True, exist_ok=True)
        s_lbl.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path, _ in items:
            shutil.copy2(img_path, s_img / img_path.name)
            shutil.copy2(lbl_path, s_lbl / lbl_path.name)

    yaml_path = out_dir / "dataset.yaml"
    data_cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)

    print(
        f"Prepared dataset at {out_dir}: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset for 8 VN traffic signs.")
    parser.add_argument(
        "--synthetic-per-class",
        type=int,
        default=200,
        help="Number of synthetic bootstrap images per sign class (0 to disable).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw_webots",
        help="Optional directory with Webots auto-captured images/ and labels/.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_pairs: List[Tuple[Path, Path, int]] = []
    if args.synthetic_per_class > 0:
        syn_dir = PROJECT_ROOT / "datasets" / "raw_synthetic"
        if syn_dir.exists():
            shutil.rmtree(syn_dir)
        all_pairs.extend(
            generate_synthetic_samples(syn_dir, per_class=args.synthetic_per_class, seed=args.seed)
        )

    webots_pairs = collect_raw_pairs(args.raw_dir)
    if webots_pairs:
        print(f"Found {len(webots_pairs)} Webots-captured frames in {args.raw_dir}")
        all_pairs.extend(webots_pairs)

    if not all_pairs:
        raise RuntimeError("No dataset samples found. Enable --synthetic-per-class or run Webots collector.")

    build_yolo_dataset(all_pairs, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
