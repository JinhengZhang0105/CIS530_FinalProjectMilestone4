'''
python refinement.py \
    --ref_dir data/refs \
    --captions data/captions.json \
    --output_dir runs/city_complex_run1 \
    --iterations 4 --M 3 --K 4 \
    --image_model dall-e-3 --image_quality hd
'''

import argparse
import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from openai import OpenAI

import clip


def list_image_files(ref_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in ref_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(files)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_captions(captions_path: Optional[Path]) -> Dict[str, str]:
    if captions_path is None:
        logging.warning("No captions JSON provided; falling back to generic captions.")
        return {}
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")
    with open(captions_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    captions = {}
    for k, v in raw.items():
        stem = Path(k).stem
        captions[stem] = v
    logging.info(f"Loaded {len(captions)} captions from {captions_path}")
    return captions


def default_caption_for_stem(stem: str) -> str:
    return (
        f"An image named {stem}. You do NOT know its content; "
        f"write a plausible, detailed photographic scene description in English."
    )

def load_clip_model(model_name: str, device: torch.device):
    logging.info(f"Loading CLIP model {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def encode_ref_image(
    clip_model, preprocess, image_path: Path, device: torch.device
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(image_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def compute_prompt_clip_scores(
    prompts: List[str],
    ref_feat: torch.Tensor,
    clip_model,
    device: torch.device,
) -> List[float]:
    if not prompts:
        return []

    tokens = clip.tokenize(prompts, truncate=True).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (ref_feat @ text_features.T).squeeze(0)

    scores = sims.detach().cpu().numpy().tolist()
    return [float(s) for s in scores]



def _prompt_fits_clip_context(prompt: str) -> bool:
    try:
        clip.tokenize([prompt], truncate=False)
        return True
    except Exception:
        return False


def split_text_into_clip_chunks(text: str) -> List[str]:
    words = text.strip().split()
    if not words:
        return [""]
    chunks: List[str] = []
    cur_words: List[str] = []
    for w in words:
        candidate = " ".join(cur_words + [w]).strip()
        if _prompt_fits_clip_context(candidate):
            cur_words.append(w)
        else:
            if cur_words:
                chunks.append(" ".join(cur_words))
                cur_words = [w]
            else:
                chunks.append(w)
                cur_words = []
    if cur_words:
        chunks.append(" ".join(cur_words))
    return [c for c in (ch.strip() for ch in chunks) if c] or [text.strip()]


def compute_prompt_clip_scores_chunked_mean(
    prompts: List[str],
    ref_feat: torch.Tensor,
    clip_model,
    device: torch.device,
) -> List[float]:
    
    if not prompts:
        return []

    all_chunks: List[str] = []
    spans: List[Tuple[int, int]] = []
    for p in prompts:
        ch = split_text_into_clip_chunks(p)
        s = len(all_chunks)
        all_chunks.extend(ch)
        e = len(all_chunks)
        spans.append((s, e))

    tokens = clip.tokenize(all_chunks, truncate=False).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (ref_feat @ text_features.T).squeeze(0)

    sims_np = sims.detach().cpu().numpy()

    prompt_scores: List[float] = []
    for (s, e) in spans:
        chunk_scores = sims_np[s:e]
        prompt_scores.append(float(np.mean(chunk_scores)) if len(chunk_scores) > 0 else float("-inf"))

    return prompt_scores


def compute_prompt_clip_scores_auto(
    prompts: List[str],
    ref_feat: torch.Tensor,
    clip_model,
    device: torch.device,
) -> List[float]:
    
    use_chunked = any(not _prompt_fits_clip_context(p) for p in prompts)
    if use_chunked:
        logging.info("  Using chunked CLIP prompt scoring (mean over chunks) due to long prompt(s).")
        return compute_prompt_clip_scores_chunked_mean(
            prompts=prompts, ref_feat=ref_feat, clip_model=clip_model, device=device
        )
    return compute_prompt_clip_scores(
        prompts=prompts, ref_feat=ref_feat, clip_model=clip_model, device=device
    )

def compute_image_clip_scores(
    image_paths: List[Path],
    ref_feat: torch.Tensor,
    clip_model,
    preprocess,
    device: torch.device,
    batch_size: int = 8,
) -> List[float]:
    scores: List[float] = []

    if not image_paths:
        return scores

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            images = [
                preprocess(Image.open(p).convert("RGB")) for p in batch
            ]
            image_tensor = torch.stack(images).to(device)
            feats = clip_model.encode_image(image_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sims = (feats @ ref_feat.T).squeeze(1)  # (batch,)
            scores.extend(sims.detach().cpu().numpy().tolist())

    return [float(s) for s in scores]

def build_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)
    return client


def _extract_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        snippet = text[start : end + 1]
        return json.loads(snippet)
    except Exception:
        return None


def generate_prompts_with_gpt41mini(
    client: OpenAI,
    caption: str,
    base_prompt: Optional[str],
    num_prompts: int,
    mode: str = "initial",
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> List[str]:
    assert mode in {"initial", "variation"}

    system_msg = (
        "You are an expert visual prompt engineer for state-of-the-art "
        "text-to-image models (such as GPT-4o image generation). "
        "You write precise, concise English prompts (< 200 words) that fully "
        "describe a scene for high-fidelity image synthesis."
    )

    if mode == "initial":
        user_msg = f"""
We want to reconstruct a reference photograph described by this caption:

\"\"\"{caption}\"\"\".

Generate {num_prompts} diverse candidate prompts that could reproduce this scene.
Each prompt should be self-contained and suitable to send directly to an image
generator. Focus on global scene, key objects with attributes, background,
lighting, composition, and style. Avoid mentioning 'reference' or 'caption'.

Return ONLY a JSON object of the form:
{{
  "prompts": ["...", "...", ...]
}}
with exactly {num_prompts} strings.
"""
    else:
        base_prompt_text = base_prompt or ""
        user_msg = f"""
We want to refine a current best text-to-image prompt to better match the
following target scene description:

TARGET CAPTION:
\"\"\"{caption}\"\"\".

CURRENT BEST PROMPT:
\"\"\"{base_prompt_text}\"\"\".

Generate {num_prompts} small variations of the BEST prompt that might improve
alignment with the target scene. Make MINIMAL edits: add or adjust details,
clarify composition, or slightly tweak style. Do NOT drastically change the
scene or introduce new objects not implied by the caption.

Return ONLY a JSON object of the form:
{{
  "prompts": ["...", "...", ...]
}}
with exactly {num_prompts} strings.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = response.choices[0].message.content or ""
    data = _extract_json_object(text)
    prompts: List[str] = []

    if data and isinstance(data, dict) and "prompts" in data:
        raw = data["prompts"]
        if isinstance(raw, list):
            prompts = [str(p).strip() for p in raw if str(p).strip()]

    if not prompts:
        logging.warning("Failed to parse JSON 'prompts'; using line-based fallback.")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if ln[0].isdigit() and "." in ln:
                ln = ln.split(".", 1)[1].strip()
            if ln.startswith("- "):
                ln = ln[2:].strip()
            if ln:
                prompts.append(ln)

    seen = set()
    unique_prompts = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
        if len(unique_prompts) >= num_prompts:
            break

    if not unique_prompts:
        logging.error("Could not recover any prompts from GPT-4.1-mini; falling back.")
        if base_prompt:
            unique_prompts = [base_prompt] * num_prompts
        else:
            unique_prompts = [caption] * num_prompts

    while len(unique_prompts) < num_prompts:
        unique_prompts.append(unique_prompts[-1])

    return unique_prompts


def generate_images_gpt_image_1(
    client: OpenAI,
    prompt: str,
    image_model: str,
    out_dir: Path,
    stem: str,
    iter_idx: int,
    K: int,
    size: str = "1024x1024",
    quality: str = "high",
) -> List[Path]:
    ensure_dir(out_dir)
    image_paths: List[Path] = []

    logging.info(
        f"  [{image_model}] Generating {K} images for iter {iter_idx} prompt (len={len(prompt)})"
    )

    if image_model == "gpt-image-1":
        resp = client.images.generate(
            model=image_model,
            prompt=prompt,
            n=K,
            size=size,
            quality=quality,
        )
        for j, d in enumerate(resp.data):
            b64 = d.b64_json
            img_bytes = base64.b64decode(b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            fname = out_dir / f"{stem}_iter{iter_idx:02d}_sample{j+1:02d}.png"
            img.save(fname)
            image_paths.append(fname)

    else:
        for j in range(K):
            resp = client.images.generate(
                model=image_model,
                prompt=prompt,
                n=1,
                size=size,
                quality=quality,
                style="natural", 
                response_format="b64_json",
            )
            d = resp.data[0]
            b64 = d.b64_json
            img_bytes = base64.b64decode(b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            fname = out_dir / f"{stem}_iter{iter_idx:02d}_sample{j+1:02d}.png"
            img.save(fname)
            image_paths.append(fname)

    return image_paths

def run_multi_prompt_loop_for_image(
    client: OpenAI,
    clip_model,
    preprocess,
    device: torch.device,
    image_path: Path,
    caption: str,
    out_root: Path,
    iterations: int,
    M: int,
    K: int,
    refiner_model: str,
    image_model: str,
    image_size: str,
    image_quality: str,
    accept_threshold: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
) -> Tuple[str, Path, float]:
    stem = image_path.stem
    example_dir = out_root / stem
    images_dir = example_dir / "images"
    ensure_dir(example_dir)
    ensure_dir(images_dir)

    ref_img = Image.open(image_path).convert("RGB")
    ref_copy_path = example_dir / "ref.png"
    ref_img.save(ref_copy_path)

    ref_feat = encode_ref_image(clip_model, preprocess, image_path, device)

    logs: List[dict] = []

    global_best_image_score = -1e9
    global_best_prompt: Optional[str] = None
    global_best_image_path: Optional[Path] = None

    last_improvement_score = -1e9
    no_improve_steps = 0

    logging.info(f"Processing {stem} with caption: {caption[:80]}...")

    for iter_idx in range(1, iterations + 1):
        logging.info(f"[{stem}] Iteration {iter_idx}/{iterations}")

        if iter_idx == 1 or global_best_prompt is None:
            prompts = generate_prompts_with_gpt41mini(
                client=client,
                caption=caption,
                base_prompt=None,
                num_prompts=M,
                mode="initial",
                model_name=refiner_model,
            )
        else:
            prompts = generate_prompts_with_gpt41mini(
                client=client,
                caption=caption,
                base_prompt=global_best_prompt,
                num_prompts=M,
                mode="variation",
                model_name=refiner_model,
            )

        prompt_scores = compute_prompt_clip_scores_auto(
            prompts=prompts,
            ref_feat=ref_feat,
            clip_model=clip_model,
            device=device,
        )
        assert len(prompts) == len(prompt_scores)
        best_prompt_idx = int(np.argmax(prompt_scores))
        current_prompt = prompts[best_prompt_idx]
        logging.info(
            f"  Best prompt idx={best_prompt_idx} CLIP(text,img)={prompt_scores[best_prompt_idx]:.4f}"
        )

        batch_image_paths = generate_images_gpt_image_1(
            client=client,
            prompt=current_prompt,
            image_model=image_model,
            out_dir=images_dir,
            stem=stem,
            iter_idx=iter_idx,
            K=K,
            size=image_size,
            quality=image_quality,
        )

        image_scores = compute_image_clip_scores(
            image_paths=batch_image_paths,
            ref_feat=ref_feat,
            clip_model=clip_model,
            preprocess=preprocess,
            device=device,
        )
        assert len(batch_image_paths) == len(image_scores)
        iter_best_idx = int(np.argmax(image_scores))
        iter_best_score = float(image_scores[iter_best_idx])
        iter_best_image_path = batch_image_paths[iter_best_idx]

        epoch_rec_dir = out_root / f"epoch{iter_idx:02d}_rec"
        ensure_dir(epoch_rec_dir)
        epoch_best_path = epoch_rec_dir / f"{stem}.png"
        Image.open(iter_best_image_path).save(epoch_best_path)

        logging.info(
            f"  Best image score this iter: {iter_best_score:.4f} "
            f"(sample {iter_best_idx}, file={iter_best_image_path.name})"
        )

        improved = False
        if iter_best_score > global_best_image_score + accept_threshold:
            improved = True
            global_best_image_score = iter_best_score
            global_best_prompt = current_prompt
            global_best_image_path = iter_best_image_path
            logging.info(
                f"  [IMPROVED] New global best score: {global_best_image_score:.4f}"
            )

        logs.append(
            {
                "iter": iter_idx,
                "prompts": prompts,
                "prompt_scores": prompt_scores,
                "chosen_prompt_idx": best_prompt_idx,
                "chosen_prompt": current_prompt,
                "image_paths": [str(p) for p in batch_image_paths],
                "image_scores": image_scores,
                "iter_best_idx": iter_best_idx,
                "iter_best_score": iter_best_score,
                "global_best_score": global_best_image_score,
                "improved": improved,
            }
        )

        if improved:
            if global_best_image_score > last_improvement_score + early_stop_min_delta:
                last_improvement_score = global_best_image_score
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        if (
            early_stop_patience > 0
            and no_improve_steps >= early_stop_patience
        ):
            logging.info(
                f"  Early stopping after {iter_idx} iterations "
                f"(no significant improvement for {no_improve_steps} steps)."
            )
            break

    if global_best_prompt is None or global_best_image_path is None:
        logging.warning(
            f"[{stem}] No improvement found; falling back to last iteration's best."
        )
        global_best_prompt = current_prompt
        global_best_image_path = iter_best_image_path
        global_best_image_score = iter_best_score

    with open(example_dir / "final_prompt.txt", "w", encoding="utf-8") as f:
        f.write(global_best_prompt)

    summary = {
        "image_id": stem,
        "caption": caption,
        "final_prompt": global_best_prompt,
        "final_image": str(global_best_image_path),
        "final_clip_score": global_best_image_score,
        "iterations_run": len(logs),
        "logs": logs,
    }
    with open(example_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"[{stem}] Finished. Global best CLIPSim={global_best_image_score:.4f}, "
        f"image={global_best_image_path.name}"
    )

    return global_best_prompt, global_best_image_path, global_best_image_score

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-prompt CLIP-guided refinement with GPT-4.1-mini (refiner) "
            "and GPT-4o image generation (gpt-image-1)."
        )
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        help="Directory containing reference images (PNG/JPG).",
    )
    parser.add_argument(
        "--captions",
        type=str,
        default=None,
        help="Optional JSON file mapping image stems to captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where refinement runs will be stored.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="Max number of refinement iterations per image (N).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=4,
        help="Number of images per iteration (image batch size).",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=4,
        help="Number of candidate prompts per iteration (prompt batch size).",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP model name to use (default: ViT-B/32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device for CLIP: "auto", "cpu", "cuda", or "mps".',
    )
    parser.add_argument(
        "--refiner_model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name for the refiner LLM (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--image_model",
        type=str,
        default="dall-e-3",
        help="OpenAI model name for image generation.",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="1024x1024",
        help='Image size (e.g., "1024x1024", "1536x1024", "1024x1536").',
    )
    parser.add_argument(
        "--image_quality",
        type=str,
        default="hd",
        help='Image quality ("standard" or "hd").',
    )
    parser.add_argument(
        "--accept_threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum CLIPSim improvement required to accept a new global best "
            "(hill-climbing). Default: 0.0."
        ),
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help=(
            "If > 0, stop if there is no significant improvement for this many "
            "consecutive iterations (per image)."
        ),
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.001,
        help="Minimum improvement in CLIPSim considered significant for early stopping.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="Optionally limit the number of reference images processed (for debugging).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    ref_dir = Path(args.ref_dir)
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")

    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    captions_path = Path(args.captions) if args.captions else None
    captions = load_captions(captions_path)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            logging.warning(
                "CUDA requested but not available; falling back to CPU."
            )
            device = torch.device("cpu")
        elif args.device == "mps" and not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            logging.warning(
                "MPS requested but not available; falling back to CPU."
            )
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)

    logging.info(f"Using device: {device}")

    client = build_openai_client()

    clip_model, preprocess = load_clip_model(args.clip_model, device=device)

    ref_images = list_image_files(ref_dir)
    if args.max_images > 0:
        ref_images = ref_images[: args.max_images]

    if not ref_images:
        logging.error(f"No images found in {ref_dir}")
        return

    logging.info(f"Found {len(ref_images)} reference images in {ref_dir}")

    final_rec_dir = out_root / "final_rec"
    ensure_dir(final_rec_dir)

    best_scores: List[float] = []

    for image_path in ref_images:
        stem = image_path.stem
        caption = captions.get(stem, default_caption_for_stem(stem))

        try:
            best_prompt, best_image_path, best_score = run_multi_prompt_loop_for_image(
                client=client,
                clip_model=clip_model,
                preprocess=preprocess,
                device=device,
                image_path=image_path,
                caption=caption,
                out_root=out_root,
                iterations=args.iterations,
                M=args.M,
                K=args.K,
                refiner_model=args.refiner_model,
                image_model=args.image_model,
                image_size=args.image_size,
                image_quality=args.image_quality,
                accept_threshold=args.accept_threshold,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
            )
        except Exception as e:
            logging.exception(
                f"Error processing {stem}; skipping this image. Error: {e}"
            )
            continue

        final_path = final_rec_dir / f"{stem}.png"
        Image.open(best_image_path).save(final_path)

        best_scores.append(best_score)

    if best_scores:
        mean_clip = float(np.mean(best_scores))
        logging.info(
            f"Finished all images. Mean best CLIPSim over dataset: {mean_clip:.4f}"
        )
    else:
        logging.warning("No images successfully processed; no scores to report.")


if __name__ == "__main__":
    main()
