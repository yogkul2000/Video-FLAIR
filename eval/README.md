# Evaluation

Evaluation runs on [lmms-eval](https://github.com/evolvinglmms-lab/lmms-eval). `overlay/` holds the files that add the Video-FLAIR model wrapper and the FLAIR answer format to the benchmarks used in the paper. `setup.sh` clones lmms-eval at a fixed commit and copies the overlay on top.

## Setup

Install the training environment from the main README first (the model wrapper uses ms-swift's vLLM engine). Then, from this folder:

```bash
pip install uv
bash setup.sh
```

## Run

Download a checkpoint and point the script at it:

```bash
bash run_eval.sh /path/to/downloaded/checkpoint results
```

This runs all 11 benchmarks. To run a subset, list the task names after the output folder:

```bash
bash run_eval.sh /path/to/downloaded/checkpoint results video_holmes vsibench
```

Each task writes its scores and per-sample outputs to `results/<task>/`.

| Benchmark | Task name |
|---|---|
| EMMA | `emma` |
| MMMU | `mmmu_val` |
| MathVista | `mathvista_testmini_format` |
| HallusionBench | `hallusion_bench_image` |
| AI2D | `ai2d` |
| MM-Vet v2 | `mmvetv2` |
| Video-Holmes | `video_holmes` |
| Video-TT | `videott_single_mc` |
| VSI-Bench | `vsibench` |
| SciVideoBench | `scivideobench` |
| Video-MMMU | `video_mmmu_flair` |

## Settings

- The model receives the ADAPTIVE system prompt and picks its own mode. Set `MODE=DIRECT`, `MODE=CONCISE` or `MODE=DEEP` to force one mode.
- Decoding is greedy with up to 2048 new tokens.
- Videos are sampled at 1 fps, up to 128 frames (32 frames for Video-MMMU).
- `LIMIT=N` evaluates only the first N samples of each task.

## LLM judge

MathVista, HallusionBench, MM-Vet v2 and EMMA use an LLM judge through an OpenAI-compatible API:

```bash
export OPENAI_API_URL=https://your-endpoint/v1
export OPENAI_API_KEY=your-key
export FLAIR_JUDGE_MODEL=gemini-3.8-flash
```

`FLAIR_JUDGE_MODEL` selects the judge model (default `gemini-3.8-flash`). The other benchmarks are scored by exact match and do not need an API key.
