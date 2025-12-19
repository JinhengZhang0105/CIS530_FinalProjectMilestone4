
import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(directory: Path):
    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def resize_lowres(img: Image.Image, short_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) <= short_side:
        return img

    if w <= h:
        new_w = short_side
        new_h = int(h * short_side / w)
    else:
        new_h = short_side
        new_w = int(w * short_side / h)

    small = img.resize((new_w, new_h), Image.BICUBIC)
    up = small.resize((w, h), Image.BICUBIC)
    return up


def process_directory(input_dir: Path, output_dir: Path, short_side: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_dir)

    if not images:
        raise RuntimeError(f"No input images found in {input_dir}")

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Saving low-res reconstructions to {output_dir}")

    for idx, in_path in enumerate(images, start=1):
        img = Image.open(in_path)
        recon = resize_lowres(img, short_side=short_side)

        out_path = output_dir / in_path.name
        recon.save(out_path)

        if idx % 50 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)} images.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple low-res resize baseline for semantic image reconstruction."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input/reference images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save reconstructed images.",
    )
    parser.add_argument(
        "--short_side",
        type=int,
        default=32,
        help="Target size for shorter side during downsampling (default: 32).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist or is not a directory: {input_dir}")

    process_directory(input_dir, output_dir, args.short_side)


if __name__ == "__main__":
    main()
