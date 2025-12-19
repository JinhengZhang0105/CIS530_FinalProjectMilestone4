# LLM-Guided Semantic Image Refinement (`refinement.py`)

**IMPORTANT**: Due to the size limit of Canvas uploader, the image results are removed from the folder for submission. You may check the slides or the report for results summary and sample reconstruction, including the evolution of CLIP scores.

This document describes how to use `refinement.py` to run our LLM-guided semantic image compression experiments, on top of the evaluation and baselines from Milestone 2.

**IMPORTANT**: To run this code, you will need an API key set in the environment, while the one that the author used will not be provided
here since it is private. You may use the one shared from the lab or from yourself to validate the code. Thank you.

The method implements a practical version of the multi-prompt, CLIP-guided refinement loop using:

- **GPT‑4.1‑mini** as the refiner LLM (prompt generation / variation),
- **GPT‑4o image generation** via the `gpt-image-1` model as the decoder,
- **CLIP** as the semantic similarity metric, consistent with our CLIPSim definition and `score.py`.   

---

## Directory Layout

We assume the following structure:

```text
project_root/
  refinement.py

  score.py
  simple-baseline.py
  strong-baseline.py

  README.md

  data/
      refs/
        1.jpg
        2.jpg
        ...
      captions.json

  runs/
    ...
```

## Usage Example
```
  python refinement.py \
  --ref_dir data/refs \
  --captions data/captions.json \
  --output_dir runs/city_complex_run1 \
  --iterations 4 \
  --K 4 \
  --M 4
```

The output directory is "/runs". Thank you for reviewing our code!