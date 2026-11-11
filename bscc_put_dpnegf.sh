#!/usr/bin/env bash

set -Eeuo pipefail

# 根据 LAMMPS 阶段生成的 transfer_manifest.json，只上传 DPNEGF 所需文件。
# 实际跨集群传输复用 ~/bscc_put.sh，因此 SSH 账号、端口和断点续传参数
# 仍只在 bscc_put.sh 中维护。

show_help() {
    cat <<'EOF'
用法:
  ./bscc_put_dpnegf.sh 数据根目录 BSCC远端数据根目录 [bscc_put.sh路径]

示例:
  ./bscc_put_dpnegf.sh ./data_tb /public5/home/t6s008517/data_tb/
  ./bscc_put_dpnegf.sh ./data_tb /public5/home/t6s008517/data_tb/ ~/bscc_put.sh

说明:
  - 数据根目录中必须存在 transfer_manifest.json
  - 上传前严格校验清单内每个文件的 SHA-256
  - 只上传清单中的 workflow_config.json、STRUCT.fdf、XYZ、外部 UC
    以及 transfer_manifest.json
  - 不上传 LAMMPS dump、日志、flag、job_id、模型或计算输出
  - 远端路径表示 B 集群 batch.py 中 data_root 对应的目录
  - B 集群自己的 batch.py、manage.sh 和输入模型需提前准备
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    show_help
    exit 2
fi

SOURCE_ARG="$1"
REMOTE_PATH="$2"
PUT_SCRIPT="${3:-${HOME}/bscc_put.sh}"

if [[ ! -d "$SOURCE_ARG" ]]; then
    echo "[ERROR] 数据根目录不存在: $SOURCE_ARG" >&2
    exit 1
fi

DATA_ROOT="$(cd "$SOURCE_ARG" && pwd -P)"
MANIFEST="$DATA_ROOT/transfer_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[ERROR] 找不到传输清单: $MANIFEST" >&2
    echo "[ERROR] 请先在 A 集群完成 BATCH_PHASE=lammps。" >&2
    exit 1
fi

if [[ ! -x "$PUT_SCRIPT" ]]; then
    echo "[ERROR] 上传脚本不存在或不可执行: $PUT_SCRIPT" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "[ERROR] 未找到 rsync。" >&2
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "[ERROR] PYTHON_BIN 不可执行: $PYTHON_BIN" >&2
        exit 1
    fi
    PYTHON_CMD="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python)"
else
    echo "[ERROR] 未找到 python3 或 python。" >&2
    exit 1
fi

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cnt-dpnegf-transfer.XXXXXX")"

cleanup() {
    if [[ -n "${STAGING_DIR:-}" && -d "$STAGING_DIR" ]]; then
        case "$(basename "$STAGING_DIR")" in
            cnt-dpnegf-transfer.*) rm -rf -- "$STAGING_DIR" ;;
            *) echo "[WARN] 未清理非预期临时目录: $STAGING_DIR" >&2 ;;
        esac
    fi
}
trap cleanup EXIT INT TERM

echo "[INFO] 数据根目录: $DATA_ROOT"
echo "[INFO] 远端数据根目录: $REMOTE_PATH"
echo "[INFO] 校验传输清单并准备轻量文件..."

# Python 向 stdout 输出 NUL 分隔的相对路径；rsync 按该列表保持目录结构。
"$PYTHON_CMD" - "$DATA_ROOT" <<'PY' |
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve()
manifest_path = root / "transfer_manifest.json"

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"[ERROR] 无法读取传输清单: {exc}")

if manifest.get("version") != 1:
    raise SystemExit(f"[ERROR] 不支持的传输清单版本: {manifest.get('version')!r}")

files = manifest.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("[ERROR] 传输清单 files 为空或格式错误")

validated = []
total_bytes = 0
hash_pattern = re.compile(r"[0-9a-fA-F]{64}")

for raw_name, expected_hash in sorted(files.items()):
    if not isinstance(raw_name, str) or not isinstance(expected_hash, str):
        raise SystemExit("[ERROR] 清单路径或 SHA-256 不是字符串")
    relative = PurePosixPath(raw_name)
    if relative.is_absolute() or not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise SystemExit(f"[ERROR] 清单包含不安全路径: {raw_name!r}")
    if not hash_pattern.fullmatch(expected_hash):
        raise SystemExit(f"[ERROR] 清单 SHA-256 格式错误: {raw_name}")

    lexical_path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"[ERROR] 清单文件不能是软链接或经过软链接目录: {raw_name}")

    try:
        source = lexical_path.resolve(strict=True)
        source.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[ERROR] 清单文件越界或不存在: {raw_name}: {exc}")
    if not source.is_file():
        raise SystemExit(f"[ERROR] 清单条目不是普通文件: {raw_name}")

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual_hash = digest.hexdigest()
    if actual_hash.lower() != expected_hash.lower():
        raise SystemExit(
            f"[ERROR] SHA-256 不一致: {raw_name}\n"
            f"        expected={expected_hash.lower()}\n"
            f"        actual  ={actual_hash}"
        )
    validated.append(raw_name)
    total_bytes += source.stat().st_size

print(f"[OK] 已校验 {len(validated)} 个清单文件，共 {total_bytes} bytes", file=sys.stderr)
for name in ["transfer_manifest.json", *validated]:
    sys.stdout.buffer.write(os.fsencode(name) + b"\0")
PY
rsync -a --from0 --files-from=- --prune-empty-dirs "$DATA_ROOT/" "$STAGING_DIR/"

echo "[INFO] 临时传输包: $STAGING_DIR"
echo "[INFO] 调用: $PUT_SCRIPT"

# 源目录末尾的 / 表示把临时包内容放到远端数据根目录中。
"$PUT_SCRIPT" "$STAGING_DIR/" "$REMOTE_PATH"

echo "[OK] DPNEGF 交接文件上传完成"
