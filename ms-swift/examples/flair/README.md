# Video-FLAIR on ms-swift

Mode-conditioned reasoning (DIRECT / CONCISE / DEEP / ADAPTIVE) for Qwen2.5-VL and Qwen3-VL:
SFT warm start, then slot-structured GRPO with 7 rewards, GDPO normalization, DAPO loss,
token-level span credit, prioritized replay with clipped importance sampling, and an online Qwen3-VL verifier that is refreshed with DPO and hot-swapped during training.

Every path comes from arguments or environment variables. The scripts in `qwen2.5vl/` and `qwen3vl/` are identical apart from `--model_type`.

## Layout

| path | role |
|---|---|
| `flair/prompts.py` | mode system prompts, slot schedule (two slots each for DIRECT, CONCISE, DEEP, ADAPTIVE) |
| `flair/parsing.py` | `<think>/<answer>` parsing, grounded spans `<obj><box><t>` |
| `flair/answers.py` | answer typing and scoring (MC, numeric, open, temporal IoU) |
| `flair/verifier.py` | OpenAI-compatible verifier client, verdict parsing, hot-swap by polling the server for the newest adapter |
| `flair/scorer.py` | per-group reward components, utility-based mode selection, replay, pair dumping |
| `flair/rewards.py` | ORMs `flair_ans flair_format flair_comply flair_cost flair_balance flair_select flair_verifier` |
| `flair/replay.py` | prioritized prompt replay, reuse of stored completions, clipped IS weights |
| `flair/trainer_patch.py` | GRPO patches: slot/mode assignment, replay mixing, token credit, frame-list videos, fps scalar |
| `flair/plugin.py` | `--external_plugins` entry point |
| `tools/prepare_sft.py` | FLAIR traces to swift SFT jsonl |
| `tools/prepare_rl.py` | RL prompts with `solution`, `answer_type`, `sid` |
| `tools/difficulty_filter.py` | `expand` / `select` for the pass-rate filter |
| `tools/build_verifier_dpo.py` | verifier DPO pairs from GRPO groups |
| `tools/verifier_refresh.py` | watches GRPO step, runs DPO, hot-loads the LoRA into the verifier server |

## Pipeline

```bash
cd ms-swift
S=examples/flair/qwen2.5vl

python examples/flair/tools/prepare_sft.py --input $TRACES --media_root $MEDIA --output_dir $DATA/sft
MODEL=$BASE DATA_DIR=$DATA/sft OUTPUT_DIR=$OUT/sft bash $S/sft.sh
ADAPTER=$OUT/sft/<run>/checkpoint-N OUTPUT_DIR=$OUT/sft_merged bash $S/merge.sh

python examples/flair/tools/prepare_rl.py --input $TRACES --media_root $MEDIA --output $DATA/rl/rl_all.jsonl --exclude $DATA/sft/sft_train.jsonl
MODEL=$OUT/sft_merged RL_DATA=$DATA/rl/rl_all.jsonl OUTPUT_DIR=$OUT/difficulty bash $S/difficulty.sh

CUDA_VISIBLE_DEVICES=0 VERIFIER_MODEL=$QWEN3VL_8B PORT=8100 bash $S/verifier_server.sh
CUDA_VISIBLE_DEVICES=1 VERIFIER_MODEL=$QWEN3VL_8B python examples/flair/tools/verifier_refresh.py \
    --pair_dir $SHARED/grpo_pairs --work_dir $OUT/verifier_dpo \
    --server_url http://127.0.0.1:8100/v1 --dpo_script $S/verifier_dpo.sh --every 100

FLAIR_VERIFIER_URL=http://$VERIFIER_HOST:8100/v1 FLAIR_PAIR_DIR=$SHARED/grpo_pairs \
MODEL=$OUT/sft_merged DATASET=$OUT/difficulty/rl_train.jsonl OUTPUT_DIR=$OUT/grpo bash $S/grpo.sh
```

The verifier server and refresher can run on a different machine from the policy. The server binds to `0.0.0.0` (override with `HOST`). `verifier_dpo.sh` uses torchrun port `DPO_MASTER_PORT` (default `29611`) so it does not collide with a GRPO run on the same machine. The client sends frames inline as base64, so the verifier host needs no media. `$VERIFIER_HOST` is the verifier machine's IP. `$SHARED` must be a directory that both machines can see: the trainer writes groups and its current step there, and the refresher reads them. After each DPO round the refresher loads `verifier_vN` into the server. The trainer polls `/v1/models` every `FLAIR_VERIFIER_POLL` seconds and switches to the highest `verifier_vN`. Pass `--model_file` / `FLAIR_VERIFIER_MODEL_FILE` to swap through a shared file instead.

The GRPO generation batch (`BATCH_SIZE x NPROC_PER_NODE x GRAD_ACC`) must be divisible by 8, the group size K.

## Main knobs

| variable | default | meaning |
|---|---|---|
| `FLAIR_REWARD_WEIGHTS` | `1,.25,.6,.2,.2,.4,.35` | weights for the 7 rewards (also `REWARD_WEIGHTS` in `grpo.sh`) |
| `FLAIR_SLOT_MODES` | `DIRECT,DIRECT,CONCISE,CONCISE,DEEP,DEEP,ADAPTIVE,ADAPTIVE` | slot-to-mode schedule within a group |
| `FLAIR_TOKEN_CREDIT` | `1` | span-level advantage reweighting on/off |
| `FLAIR_CREDIT_FLOOR` | `0.05` | lower floor on raw per-token credit |
| `FLAIR_CREDIT_CLIP` | `4` | upper clip on per-token credit after normalizing to mean 1 |
| `FLAIR_VERIFIER_URL` | empty | verifier endpoint, local or remote; when empty the heuristic verifier is used |
| `FLAIR_VERIFIER_POLL` | `60` | seconds between `/v1/models` polls for a newer `verifier_vN` |
| `FLAIR_VERIFIER_ENABLE_STEP` | `100` | step at which verifier rewards are enabled |
| `FLAIR_VERIFIER_MODEL_FILE` | empty | optional shared file written by the refresher; overrides polling |
| `FLAIR_PAIR_DIR` | `$OUTPUT_DIR/verifier_pairs` | where groups are dumped for verifier DPO |
| `FLAIR_REPLAY_FRAC` | `0.125` | fraction of prompt groups per generation batch replaced by high-priority buffer prompts (`0` disables replay mixing) |
| `FLAIR_REPLAY_REUSE` | `0.5` | fraction of slots in a replayed group that reuse the stored completion instead of the fresh one |
| `FLAIR_REPLAY_IS_MIN` / `FLAIR_REPLAY_IS_MAX` | `0.5` / `2.0` | clip range of the IS weight applied to reused completions |
| `FLAIR_REPLAY_COOLDOWN` | `20` | generation rounds before a prompt can be replayed again |
| `FLAIR_REPLAY_MAX_VISITS` | `2` | maximum replays per prompt |
| `FLAIR_REPLAY_STORE` | `512` | number of top-priority prompts whose completions are kept for reuse |
| `FLAIR_<CFG_KEY>` | see `scorer.CFG` | overrides any scorer constant, e.g. `FLAIR_LAMBDA_L` |

Token credit and the replay IS correction are turned off automatically when `--padding_free` is enabled.

The replay buffer is filled once verifier rewards are on, with priority p = |S_verifier| + (1 - R_ans). Each generation round, high-priority prompts are swapped in as whole groups. In a replayed group, a subset of slots reuse that slot's stored completion and its vLLM behavior logprobs. Their advantages are scaled by clip(w, w_min, w_max), where w = exp(mean_t[log pi_theta - log pi_behavior]) is the length-normalized sequence ratio.
