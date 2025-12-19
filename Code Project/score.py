
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
import clip


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(directory: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def build_stem_index(paths: List[Path]) -> dict:
    index = {}
    for p in paths:
        index.setdefault(p.stem, []).append(p)
    return index


def match_image_pairs(pred_dir: Path, gold_dir: Path) -> List[Tuple[Path, Path]]:
    gold_paths = list_images(gold_dir)
    pred_paths = list_images(pred_dir)

    pred_index = build_stem_index(pred_paths)

    pairs = []
    missing = []

    for gold_path in gold_paths:
        stem = gold_path.stem
        candidates = pred_index.get(stem, [])
        if not candidates:
            missing.append(stem)
            continue
        if len(candidates) > 1:
            candidates = sorted(candidates)
        pred_path = candidates[0]
        pairs.append((pred_path, gold_path))

    if missing:
        print(f"[WARN] No prediction found for {len(missing)} gold images (by stem).")
        print("Missing stems (first 10):", missing[:10])

    if not pairs:
        raise RuntimeError(
            f"No matched prediction/gold image pairs found between "
            f"{pred_dir} and {gold_dir}."
        )

    return pairs


def load_image(path: Path, preprocess) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return preprocess(img)


def compute_clip_similarity(
    pairs: List[Tuple[Path, Path]],
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[float, float]:
    sims = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            if not batch:
                continue

            preds = torch.stack(
                [load_image(pred, preprocess) for pred, _ in batch], dim=0
            )
            golds = torch.stack(
                [load_image(gold, preprocess) for _, gold in batch], dim=0
            )

            preds = preds.to(device)
            golds = golds.to(device)

            feats_pred = model.encode_image(preds)
            feats_gold = model.encode_image(golds)

            feats_pred = feats_pred / feats_pred.norm(dim=-1, keepdim=True)
            feats_gold = feats_gold / feats_gold.norm(dim=-1, keepdim=True)

            batch_sims = (feats_pred * feats_gold).sum(dim=-1)  # [B]
            sims.append(batch_sims.cpu())

    sims = torch.cat(sims, dim=0)
    mean_sim = sims.mean().item()
    std_sim = sims.std(unbiased=False).item()
    return mean_sim, std_sim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CLIP-based semantic reconstruction score."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory with predicted / reconstructed images.",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        required=True,
        help="Directory with reference (gold) images.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP model variant (default: ViT-B/32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run CLIP on (default: cuda).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for CLIP encoding (default: 32).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print diagnostic information in addition to the raw score.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    gold_dir = Path(args.gold_dir)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_dir does not exist or is not a directory: {pred_dir}")
    if not gold_dir.is_dir():
        raise FileNotFoundError(f"gold_dir does not exist or is not a directory: {gold_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")

    print(f"Loading CLIP model {args.model_name} on {device}...", flush=True)
    model, preprocess = clip.load(args.model_name, device=device)

    pairs = match_image_pairs(pred_dir, gold_dir)
    mean_sim, std_sim = compute_clip_similarity(pairs, model, preprocess, device, args.batch_size)

    print(f"{mean_sim:.6f}")

    if args.verbose:
        print(f"Evaluated {len(pairs)} image pairs.")
        print(f"Mean CLIP cosine similarity: {mean_sim:.6f}")
        print(f"Std of per-image similarity: {std_sim:.6f}")


if __name__ == "__main__":
    main()
