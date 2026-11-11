#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="${1:?temporary directory required}"
source_root="$scratch/source data"
capture="$scratch/captured-files.txt"
fake_put="$scratch/fake_bscc_put.sh"

if ! command -v rsync >/dev/null 2>&1; then
    mkdir -p "$scratch/bin"
    fake_rsync="$scratch/bin/rsync"
    cat > "$fake_rsync" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
source_dir="${@: -2:1}"
destination="${@: -1}"
while IFS= read -r -d '' relative; do
    mkdir -p "$destination/$(dirname "$relative")"
    cp -p "$source_dir/$relative" "$destination/$relative"
done
EOF
    chmod +x "$fake_rsync"
    export PATH="$scratch/bin:$PATH"
fi

mkdir -p "$source_root/case/replica_001/dpnegf/40000"
printf 'config\n' > "$source_root/case/replica_001/workflow_config.json"
printf 'fdf\n' > "$source_root/case/replica_001/dpnegf/40000/STRUCT.fdf"
printf 'xyz\n' > "$source_root/case/replica_001/dpnegf/40000/5_5.xyz"
printf 'large dump must stay on A\n' > "$source_root/case/replica_001/traj.dump"

python_bin="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"$python_bin" - "$source_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = [
    "case/replica_001/workflow_config.json",
    "case/replica_001/dpnegf/40000/STRUCT.fdf",
    "case/replica_001/dpnegf/40000/5_5.xyz",
]
manifest = {
    "version": 1,
    "replicas": [{"replica": "case/replica_001", "configuration": "case"}],
    "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
}
(root / "transfer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
PY

cat > "$fake_put" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
find "$1" -type f -printf '%P\n' | LC_ALL=C sort > "${CAPTURE_FILE:?}"
printf '%s\n' "$2" > "${REMOTE_CAPTURE:?}"
EOF
chmod +x "$fake_put"

export CAPTURE_FILE="$capture"
export REMOTE_CAPTURE="$scratch/remote.txt"
bash "$repo/bscc_put_dpnegf.sh" "$source_root" "/remote/data root/" "$fake_put"

expected="$scratch/expected.txt"
printf '%s\n' \
    'case/replica_001/dpnegf/40000/5_5.xyz' \
    'case/replica_001/dpnegf/40000/STRUCT.fdf' \
    'case/replica_001/workflow_config.json' \
    'transfer_manifest.json' > "$expected"
diff -u "$expected" "$capture"
grep -Fxq '/remote/data root/' "$scratch/remote.txt"
if grep -q 'traj.dump' "$capture"; then
    echo 'dump was unexpectedly staged' >&2
    exit 1
fi

# 篡改结构后必须在调用上传脚本前失败。
printf 'changed\n' >> "$source_root/case/replica_001/dpnegf/40000/STRUCT.fdf"
rm -f "$capture"
if bash "$repo/bscc_put_dpnegf.sh" "$source_root" /remote/ "$fake_put"; then
    echo 'tampered input was unexpectedly accepted' >&2
    exit 1
fi
if [[ -e "$capture" ]]; then
    echo 'uploader was called after checksum failure' >&2
    exit 1
fi

echo 'DPNEGF transfer wrapper checks passed'
