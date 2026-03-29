# play_manic_miner

RL training setup for **Manic Miner** (ZX Spectrum, 1983) using a custom **Fuse ML socket bridge** and a Python PPO agent.

---

## Project summary

This project attempted to train a reinforcement learning agent to play the Central Cavern level of Manic Miner from scratch, using PPO (Stable Baselines3) with a semantic CNN observation of the game grid. What follows is a record of the approaches taken, the results of each, and the conclusions reached.

---

## Approach 1 — Visual pixel observation (abandoned)

**Intent:** Feed raw ZX Spectrum screen pixels into a standard CNN policy (NatureCNN).

**Result:** NatureCNN's convolutional layers, designed for 84×84 Atari frames, reduced the 16×32 Spectrum attribute grid to zero spatial dimensions. The architecture was incompatible with the game's resolution.

**Conclusion:** A custom CNN suited to the 16×32 grid was required.

---

## Approach 2 — Semantic grid observation with flat feature vector

**Intent:** Replace pixels with a structured observation: a flat vector encoding Willy's position, key positions, guardian position, and BFS distance to the nearest key.

**Result:** The agent learned to collect the first key but made no further progress. Score=100 appeared reliably; score=200 appeared in ~1,368 episodes but was a false positive — Willy clipped a key sprite mid-death jump rather than collecting it intentionally.

**Issues identified:**
- The flat feature vector gave the agent no spatial context for navigation.
- BFS distance to nearest key did not encode whether that key was reachable without stranding others.

**Conclusion:** A spatial observation was needed. The vector approach was replaced.

---

## Approach 3 — Semantic CNN with 7-channel grid observation

**Intent:** Replace the flat vector with a 7-channel 16×32 binary grid fed into a custom CNN (AttrGridCNN), with channels for: solid tiles, nasty tiles, Willy, guardian, uncollected keys, portal, and a BFS waypoint.

**Implementation:**
- `manic_env_semantic.py`: reads all game state directly from ZX Spectrum memory via the Fuse ML socket bridge; builds static solid/nasty/conveyor grids from ROM tile definitions; tracks key collection via score delta.
- `pathfinder.py`: BFS movement graph modelling Manic Miner physics — walk, fall, jump (dy up to 4 cells to reach ceiling keys), and conveyor belt restrictions (leftward movement only from conveyor surfaces).
- `train.py`: PPO with `AttrGridCNN`, VecFrameStack (4 frames), pathing reward for newly visited cells, death penalty.
- `run_loop.py`: orchestrates Fuse launch, training iterations, and analysis.

**Issues found and fixed:**
- Conveyor belt cells were initially marked as nasty (impassable). Fixed: tracked separately; movement restricted instead.
- Jump model limited to dy=3 could not reach keys at y=0 above the all-solid y=5 platform row. Fixed: extended to dy=4.
- Key sprite attribute bytes collided with conveyor tile attribute bytes, causing false conveyor detection at key positions. Fixed: key cells cleared from all grids at reset.
- `check_bfs_routing.py` written to verify BFS routing through all 5 keys and visualise waypoint progression.

**Critical topology problem discovered:**

The level is divided by a fully solid row at y=5. Semi-solid platforms allow jumping up through them but not falling back down. Once Willy crosses into the upper half (y=0–4), he cannot return to the lower half (y=6+). Key 5 at (30,6) is in the lower half and must be collected before any upper-half key. The BFS nearest-key routing violated this constraint, permanently stranding K5.

**Fix:** Priority key system — scored each remaining key by how many other keys remain reachable from it. K5 scores 4 (can reach all others from the lower half); K1–K4 score 2–3. The waypoint always targets the highest-scoring key first, ensuring K5 is collected before Willy crosses y=5.

**Training result:** 6M timesteps, 2,407 episodes, score=0 throughout.

**Conclusion:** The priority routing was correct, but the agent never collected a key. The waypoint channel pointed to K5 at BFS distance 17 from Willy's start — a significantly harder first target than the previous nearest-key approach (distance 5). With a fresh model and sparse reward, 6M timesteps was insufficient for the agent to discover the path.

---

## Fundamental conclusion

Standard model-free RL (PPO) is the wrong algorithm for Manic Miner. The game is a **planning problem**, not a control problem:

- **Conjunctive completion:** all keys must be collected AND the portal reached AND air remaining. Partial progress has zero value at episode end. Reward cannot be summed across independent events that are actually interdependent.
- **Irreversible decisions:** crossing y=5 before collecting K5 ends any chance of level completion, with no immediate signal that the decision was wrong. PPO's temporal credit assignment cannot bridge this gap.
- **Long planning horizon:** the consequence of a bad route choice appears hundreds of steps later, well beyond what discounted reward can attribute correctly.
- **Global route validity:** the relevant question at every step is not "where is the nearest key?" but "does a valid completion route still exist given the current level state?" A CNN observing a snapshot cannot answer this.

Adding BFS waypoints to the observation did not solve the problem — it replaced it. The agent was being told how to play rather than learning to play. This is the core contradiction the project ultimately arrived at.

---

## What would actually work

The infrastructure built here is the right foundation for a different approach:

- **Route planner:** extend the BFS graph nodes from `(x, y)` to `(x, y, remaining_keys)`. A single BFS traversal over this state space finds the optimal complete route — all keys in the correct order, ending at the portal. This is tractable for a single level.
- **Learned executor:** once a complete route is planned, the agent's task reduces to short-horizon motor control between waypoints — a problem PPO handles well.

The planning and execution problems are separable. Solving them together with a single reward signal is what makes standard RL fail here.

---

## Files

| File | Purpose |
|------|---------|
| `manic_env_semantic.py` | Gymnasium environment; 7-channel semantic grid observation; key tracking; priority routing |
| `pathfinder.py` | BFS movement graph with Manic Miner physics |
| `train.py` | PPO training entry point; AttrGridCNN; episode logger |
| `run_loop.py` | Orchestrates Fuse launch + training iterations |
| `manic_data.py` | ZX Spectrum memory map constants |
| `ml_client.py` | UNIX socket client for the Fuse ML bridge |
| `check_bfs_routing.py` | Diagnostic: verifies BFS routing through all 5 keys |
| `check_semantic_obs.py` | Diagnostic: prints live 7-channel observation as ASCII |
| `analyse_run.py` | Analyses episode log; score distribution and statistics |

---

## Setup

**Requirements:** Python 3.9+, Fuse with ML bridge (`FUSE_ML_MODE`, `FUSE_ML_SOCKET`, `FUSE_ML_RESET_SNAPSHOT`).

```bash
cd play_manic_miner
python3 -m venv .venv
source .venv/bin/activate
pip install stable-baselines3 gymnasium numpy
```

**Run a training loop:**

```bash
.venv/bin/python run_loop.py \
  --iterations 20 \
  --timesteps 100000 \
  --train-args "--pathing-reward 0.01 --death-penalty 1.0 --ent-coef 0.02"
```

**Continue from a saved model:**

```bash
.venv/bin/python run_loop.py \
  --iterations 20 \
  --timesteps 100000 \
  --continue-model \
  --train-args "--pathing-reward 0.01 --death-penalty 1.0 --ent-coef 0.02"
```

**Verify BFS routing:**

```bash
.venv/bin/python check_bfs_routing.py
```
