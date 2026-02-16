# play_manic_miner

RL training setup for **Manic Miner** using a custom **Fuse ML socket bridge** and Python PPO agent.

## What this project does

This folder contains the Python side of the training loop:

- `play_manic_miner/train.py`: PPO training entrypoint.
- `play_manic_miner/manic_env.py`: Gymnasium env orchestration and bridge calls.
- `play_manic_miner/manic_data.py`: stable game memory map + data interpretation.
- `play_manic_miner/manic_play.py`: gameplay logic and reward shaping.
- `play_manic_miner/ml_client.py`: low-level UNIX socket client.

Design intent:

1. `manic_data.py` should remain mostly stable (addresses/decoders).
2. `manic_play.py` can be swapped to compare policies/rewards.

## Required Fuse emulator build

You need a Fuse build with the ML socket bridge enabled.

In this workspace, that is:

- Fuse version forked by Coo Chew Games: **1.6.0** (`fuse --version`)
- With ML bridge sources present:
  - `fuse/ml_bridge.c`
  - `fuse/ml_game_adapter.c`

This project expects the socket command set including:

- `PING`, `RESET`, `READ`, `GETINFO`, `MODE`
- `EPISODE_STEP`
- `EPISODE_STEP_KEYS` (preferred)

Notes:

- `manic_env.py` will try `EPISODE_STEP_KEYS` first and automatically fall back to `EPISODE_STEP` if unsupported.
- Without the ML bridge, training will not run.

If you are setting this up on another machine, build Fuse from the ML-enabled source tree (the one containing `ml_bridge.c` and `ml_game_adapter.c`), for example:

```bash
cd /path/to/fuse/source
./configure
make -j
```

## Python requirements

Use Python 3.9+.

Typical dependencies:

- `stable-baselines3`
- `gymnasium` (or `gym` fallback)
- `numpy`
- `matplotlib` (pulled by SB3 plotting stack in many setups)
- `tensorboard` (optional but recommended)

Example setup:

```bash
cd play_manic_miner
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install stable-baselines3 gymnasium numpy matplotlib tensorboard
```

## Start Fuse in ML mode

Before running training, launch Fuse with ML env vars.

Minimum required env vars:

- `FUSE_ML_MODE=1`
- `FUSE_ML_SOCKET=/tmp/fuse-ml.sock`
- `FUSE_ML_RESET_SNAPSHOT=/absolute/path/to/manicminer.szx`

Additional optional env vars supported by the Fuse ML bridge:

- `FUSE_ML_VISUAL=1`: start bridge in visual mode.
- `FUSE_ML_VISUAL_PACE_MS=<ms>`: pacing delay per rendered frame.
- `FUSE_ML_ACTION_KEYS=...`: custom action-id to key mapping for `EPISODE_STEP`.
- `FUSE_ML_REWARD_ADDR=<addr>`: bridge-side reward byte source.
- `FUSE_ML_DONE_ADDR=<addr>` and `FUSE_ML_DONE_VALUE=<value>`: bridge-side done condition.

Example:

```bash
FUSE_ML_MODE=1 \
FUSE_ML_SOCKET=/tmp/fuse-ml.sock \
FUSE_ML_RESET_SNAPSHOT=play_manic_miner/manicminer.szx \
fuse --speed 300
```

## Run training

Example command:

```bash
play_manic_miner/.venv/bin/python play_manic_miner/train.py \
  --visual \
  --socket /tmp/fuse-ml.sock \
  --timesteps 300000 \
  --status-window --status-refresh-steps 1 \
  --pathing-reward 1.0
```

Model output:

- Default model path prefix: `ppo_manic_miner` (SB3 saves `ppo_manic_miner.zip`)

TensorBoard logs:

- Default: `./tb_logs`

## CLI parameters

From `train.py --help`:

- `--socket`: UNIX socket path (default `/tmp/fuse-ml.sock`)
- `--socket-timeout-s`: per-command timeout seconds
- `--timesteps`: total training timesteps
- `--frames-per-action`: frames per env action
- `--max-steps`: max env steps per episode
- `--random-action-prob`: action replacement probability
- `--reset-random-action-steps`: random warmup actions after reset
- `--model-out`: output model path prefix
- `--load-model`: existing `.zip` to continue training
- `--tensorboard-log`: TensorBoard log dir
- `--log-level`: logging level
- `--ent-coef`: PPO entropy coefficient
- `--visual`: run Fuse in visual mode
- `--visual-pace-ms`: visual frame pace
- `--visual-steps`: switch to headless after N timesteps
- `--include-bridge-reward`: include bridge reward in shaped reward
- `--pathing-reward`: reward for newly visited cells
- `--first-key-mode`: terminate episode on first key
- `--first-key-success-bonus`: first-key bonus
- `--disable-safety-shield`: disable lethal-action blocking
- `--status-window`: fixed terminal status panel
- `--status-window-title`: status panel title
- `--status-refresh-steps`: status update interval

## Switching gameplay scripts for comparison

`manic_env.py` supports selecting gameplay logic module via:

- `MANIC_PLAY_MODULE` (default: `manic_play`)

Example:

```bash
MANIC_PLAY_MODULE=manic_play_variant_a \
play_manic_miner/.venv/bin/python play_manic_miner/train.py --timesteps 100000
```

Create additional variants beside `manic_play.py` to compare reward/action policies with the same data layer.

## Common issues

### `ERR reset failed`

Usually `FUSE_ML_RESET_SNAPSHOT` points to a missing/unreadable file.

Check path and permissions. `train.py` performs a startup preflight warning for this.

### `ModuleNotFoundError: stable_baselines3`

Use the project virtualenv Python and install dependencies.

### Socket connection failures

Verify Fuse is running with `FUSE_ML_MODE=1` and socket path matches `--socket`.

### Bridge command mismatch

If using an older Fuse ML build, `EPISODE_STEP_KEYS` may be unavailable; env falls back to `EPISODE_STEP` automatically.

## Safety and reproducibility notes

- Keep the same `.szx` reset snapshot when comparing policy variants.
- Keep Fuse binary/build constant across experiments.
- Record train flags and `MANIC_PLAY_MODULE` for each run.
