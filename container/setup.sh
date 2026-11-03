#!/usr/bin/env bash
# Build the atlas-torch dev image and create the distrobox. Run on the host (SteamOS).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
podman build -t localhost/atlas-torch-dev:latest -f "$here/Containerfile" "$here"
distrobox create --yes --name atlas-dev --image localhost/atlas-torch-dev:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri --group-add keep-groups" || true
echo "Enter with: distrobox enter atlas-dev   (then: source /opt/intel/oneapi/setvars.sh)"
