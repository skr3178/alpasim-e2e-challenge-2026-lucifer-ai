# AlpaSim E2E Closed Loop Challenge 2026: Team Lucifer AI solution (nuPlan track)

**Lucifer-LTF**: route-aligned inference and a bend veto for end-to-end autonomous driving in NVIDIA AlpaSim closed-loop simulation.

Team Lucifer AI's solution to the [NVIDIA AlpaSim E2E Closed Loop Challenge 2026](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026), nuPlan track.
**4th of 56 teams** on the final public leaderboard and 4th in the organisers' finals re-run.

Both entries use the organisers' released checkpoint **unchanged**. Everything here acts at inference time: one fix to how
the route is handed to the policy, and one rule that corrects plans heading off the road on bends.

Technical report: [`report/lucifer_ai_report.pdf`](report/lucifer_ai_report.pdf).

## Results (official, nuPlan track)

| Entry | What it is | PCS | Avg scene score | km per at-fault incident |
|---|---|---:|---:|---:|
| Host baseline (`093-nuplan-v1`) | Released LTF waypoints checkpoint, seed 1, stock driver | 2048* | 0.9437 | – |
| **lf-009** | + route-interface patch | 2514 (finals, 3-run mean: 2513) | 0.9454 | 1.039 |
| **lf-013** | lf-009 + bend veto (10°) | **2528** | **0.9494** | **1.364** |

PCS = Policy Capability Score, a Drive-IRT fit over all entries, so values shift slightly as entries are added.
\*Host baseline PCS from the 31 Oct 2026 snapshot, when lf-009 scored 2054 on the same fit. The top four teams' PCS
intervals (about ±170) overlap.

## Method

### 1. Route-interface patch (lf-009): [`patch/exact-targets-keep-route-whole.patch`](patch/exact-targets-keep-route-whole.patch)

Two opt-in driver settings for [py123d_garage](https://github.com/kesai-labs/py123d_garage) v0.2.2:

- **`exact_route_targets`**: the policy is conditioned on target points 40/60/80 m ahead. The stock driver placed them on
  the straight walk onto the served route; this places each one at its arc distance along the route.
- **`keep_served_route_whole`**: the stock driver cut the served route at its point nearest the ego. On tight bends that
  point can sit near the route's end, so the cut dropped every target point (16 of 1,473 public scenes).

Local effect on 1,473 public scenes, paired against the unchanged host model: mean scene score 0.9079 → 0.9133
(+0.0054 [+0.0008, +0.0104]), hard failures 62 → 52, all four cities positive.

### 2. Bend veto (lf-013): [`bendveto/`](bendveto/)

A pure-numpy rule over the model's own outputs (BEV semantics, vehicle boxes, plan). No retraining.

- Acts only on a bend (plan heading change ≥ 10° between 1.0 and 2.5 s), and never on a session's first call.
- Flags the plan when the predicted drivable surface under the ego footprint (+0.35 m) drops below 0.5 at the
  1.0 / 1.5 / 2.0 / 2.5 s poses.
- Tries lateral ramps of ±0.5 / ±1.0 m, drops any that touch a confident vehicle box, and takes the smallest with road
  score ≥ 0.7. Otherwise it keeps the model's plan.

`bendveto_hook.py` wraps the stock driver and only replaces the returned plan when the veto applies
(`BENDVETO_ARM=off|shadow|active`). Local closed-loop effect on all 1,472 public scenes: +0.0086 [+0.0029, +0.0154],
at-fault incidents 52 → 42, no perfect scene lost, no new at-fault incident.

The submitted lf-013 image ran the same rule through a combined hook that also carried a disabled speed rule; that hook's
output matched this stand-alone hook on all 770 saved planning calls checked offline.

## Build

Requires git, Docker with buildx, `uv`, and curl.

```bash
bash build.sh            # -> py123d-garage-ltfv6:lf-009 and py123d-garage-ltfv6:lf-013
```

`build.sh` clones py123d_garage v0.2.2 (`ad6cb51`), applies the patch and runs the host's setup and image build. It
downloads the released checkpoint
[`v0.2.0/ltf-base-waypoints-68M/posttrain/model_1.pth`](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints)
and checks its hash (`ae1619d5…`). If KE:SAI's repo is unavailable, it falls back to our unmodified mirror,
[skr3178/alpasim-e2e-challenge-2026-lucifer-ai](https://huggingface.co/skr3178/alpasim-e2e-challenge-2026-lucifer-ai). It then builds lf-013 on top of lf-009. To evaluate locally, follow the
[py123d_garage AlpaSim docs](https://github.com/kesai-labs/py123d_garage/blob/main/docs/evaluation/alpasim.md).

## What did not transfer

The report also covers entries that improved local scores but not official ones: a decoder fine-tune on synthetic recovery
data (lf-010) and a lead-departure speed rule (lf-012). Its experiment-history appendix lists them.

## Licences and credits

- Our code (`patch/`, `bendveto/`, `build.sh`): MIT, see [LICENSE](LICENSE).
- The patch modifies [py123d_garage](https://github.com/kesai-labs/py123d_garage) by KE:SAI (Apache-2.0).
- Checkpoint: [kesai-labs/py123d_garage_pretrained_checkpoints](https://huggingface.co/kesai-labs/py123d_garage_pretrained_checkpoints)
  (CC BY-NC-SA 4.0). An unmodified copy, under the same licence, is kept at
  [skr3178/alpasim-e2e-challenge-2026-lucifer-ai](https://huggingface.co/skr3178/alpasim-e2e-challenge-2026-lucifer-ai).
- Simulator and challenge: [NVlabs/alpasim](https://github.com/NVlabs/alpasim) (`e2e_challenge` branch).
- Report layout: `report/neurips_2026.sty`, an AlpaSim adaptation of the NeurIPS 2026 template (CC BY 4.0, see
  [`report/LICENSE-template-CC-BY-4.0.md`](report/LICENSE-template-CC-BY-4.0.md)). Camera frames in `report/figures/`
  come from nuPlan-based simulation and are subject to the nuPlan licence.

## Citation

```bibtex
@techreport{rout2026luciferltf,
  title       = {Lucifer-LTF: Route-Aligned Inference and Conservative Adaptation for the AlpaSim Challenge},
  author      = {Rout, Sangram},
  institution = {Team Lucifer AI},
  year        = {2026}
}
```
