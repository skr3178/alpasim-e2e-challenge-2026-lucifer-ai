#!/usr/bin/env bash
# Build the Team Lucifer AI images for the AlpaSim E2E Closed Loop Challenge 2026 (nuPlan track).
#
#   lf-009  = host py123d_garage v0.2.2 + the released LTF waypoints checkpoint (seed 1), unchanged,
#             + our route-interface patch (exact arc targets, whole served route)
#   lf-013  = lf-009 + the bend veto (an inference-time rule; no weights change)
#
# Usage:  bash build.sh [workdir]        (needs git, docker buildx and huggingface-cli or curl)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$HERE/work}"
GARAGE_TAG=v0.2.2                 # commit ad6cb515c83f61207d2b83ae3d7faca1ba7ce99a
CKPT_REPO=kesai-labs/py123d_garage_pretrained_checkpoints
CKPT_PATH=v0.2.0/ltf-base-waypoints-68M/posttrain/model_1.pth
CKPT_SHA256_PREFIX=ae1619d546e5aacf

mkdir -p "$WORK"

# 1. Host code at the exact release we built on, plus our patch.
if [[ ! -d "$WORK/py123d_garage" ]]; then
	git clone --depth 1 --branch "$GARAGE_TAG" https://github.com/kesai-labs/py123d_garage.git "$WORK/py123d_garage"
	git -C "$WORK/py123d_garage" apply "$HERE/patch/exact-targets-keep-route-whole.patch"
fi
# The host's setup clones NVlabs/alpasim (e2e_challenge, pinned commit e121e59) into lib/alpasim/alpasim;
# the image build needs its gRPC package. Needs uv.
[[ -d "$WORK/py123d_garage/lib/alpasim/alpasim/.git" ]] || bash "$WORK/py123d_garage/lib/alpasim/tools/setup_alpasim.sh"

# 2. The host's released checkpoint (CC BY-NC-SA 4.0; not redistributed here).
CKPT="$WORK/checkpoints/$CKPT_PATH"
if [[ ! -f "$CKPT" ]]; then
	mkdir -p "$(dirname "$CKPT")"
	curl -L --fail -o "$CKPT" "https://huggingface.co/$CKPT_REPO/resolve/main/$CKPT_PATH"
fi
sha256sum "$CKPT" | grep -q "^$CKPT_SHA256_PREFIX" || { echo "checkpoint hash mismatch" >&2; exit 1; }
# The host build script also reads config.yaml and the nuPlan sensor rig next to the checkpoint.
for f in config.yaml sensor_rig_nuplan.yaml; do
	[[ -f "$(dirname "$CKPT")/$f" ]] ||
		curl -L --fail -o "$(dirname "$CKPT")/$f" "https://huggingface.co/$CKPT_REPO/resolve/main/$(dirname "$CKPT_PATH")/$f"
done

# 3. lf-009: the host's own build script with our two driver settings switched on.
(
	cd "$WORK/py123d_garage"
	TRACK=nuplan TAG=lf-009 CHECKPOINT_FILE="$CKPT" \
		DRIVER_OVERRIDES="exact_route_targets=true keep_served_route_whole=true" \
		bash scripts/alpasim_submission/build_docker_image.sh
)

# 4. lf-013: lf-009 + bend veto.
docker build -t py123d-garage-ltfv6:lf-013 -f "$HERE/bendveto/Dockerfile" "$HERE/bendveto"

echo "Built py123d-garage-ltfv6:lf-009 and py123d-garage-ltfv6:lf-013"
