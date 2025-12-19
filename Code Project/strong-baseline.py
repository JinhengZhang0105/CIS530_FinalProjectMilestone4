
import argparse
from pathlib import Path
from typing import List

from PIL import Image
import csv
import os


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(directory: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def compress_jpeg(img: Image.Image, quality: int) -> Image.Image:
    return img.convert("RGB")


def process_directory(input_dir: Path, output_dir: Path, quality: int, log_bpp: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_dir)

    if not images:
        raise RuntimeError(f"No input images found in {input_dir}")

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Saving JPEG(q={quality}) reconstructions to {output_dir}")

    csv_path = output_dir / "jpeg_stats.csv"
    csv_file = None
    writer = None

    if log_bpp:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "height", "width", "file_size_bytes", "bpp"])

    try:
        for idx, in_path in enumerate(images, start=1):
            img = Image.open(in_path)
            img = compress_jpeg(img, quality=quality)

            stem = in_path.stem
            out_path = output_dir / f"{stem}.jpg"

            img.save(
                out_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                subsampling=0,
            )

            if log_bpp:
                file_size = os.path.getsize(out_path)
                w, h = img.size
                bpp = 8.0 * file_size / (w * h)
                writer.writerow([out_path.name, h, w, file_size, bpp])

            if idx % 50 == 0 or idx == len(images):
                print(f"Processed {idx}/{len(images)} images.")

    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"Wrote JPEG stats to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="JPEG-based strong baseline for semantic image reconstruction."
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
        help="Directory to save JPEG-compressed images.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=10,
        help="JPEG quality parameter (1-95, lower = stronger compression). Default: 10.",
    )
    parser.add_argument(
        "--log_bpp",
        action="store_true",
        help="If set, log per-image bits-per-pixel statistics to jpeg_stats.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist or is not a directory: {input_dir}")

    if not (1 <= args.quality <= 95):
        raise ValueError("JPEG quality should be between 1 and 95.")

    process_directory(input_dir, output_dir, args.quality, args.log_bpp)


if __name__ == "__main__":
    main()
