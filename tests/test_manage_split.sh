#!/usr/bin/env bash
# Smoke test the real background manager using an exported mock Python command.
# Usage: bash tests/test_manage_split.sh <temporary-directory>
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="${1:?temporary directory required}"
mkdir -p "$scratch/import root" "$scratch/A" "$scratch/B"
# Git Bash lacks setsid; this test checks forwarding/completion, not isolation.
if ! command -v setsid >/dev/null; then
    mkdir -p "$scratch/bin"
    printf '#!/usr/bin/env bash\nexec "$@"\n' > "$scratch/bin/setsid"
    chmod +x "$scratch/bin/setsid"
    export PATH="$scratch/bin:$PATH"
fi
printf '{}\n' > "$scratch/import root/transfer_manifest.json"
touch "$scratch/A/batch.py"
printf 'data_root = "../import root"\n' > "$scratch/B/batch.py"
python() { printf '%s\n' "$@" > "$BATCH_DATA_DIR/arguments.txt"; }
export -f python
export BATCH_DATA_DIR="$scratch/A" BATCH_PHASE=lammps
bash "$repo/manage_batch.sh" start
grep -qx lammps "$scratch/A/arguments.txt"
grep -q 'phase=lammps result=success' "$scratch/A/batch_generate.status"
export BATCH_DATA_DIR="$scratch/B" BATCH_PHASE=dpnegf
bash "$repo/manage_batch.sh" start
grep -qx dpnegf "$scratch/B/arguments.txt"
if grep -qx -- '--root' "$scratch/B/arguments.txt"; then
    echo 'Unexpected root override'; exit 1
fi
grep -q 'phase=dpnegf result=success' "$scratch/B/batch_generate.status"
echo 'Manager phase forwarding and status checks passed'
