"""下载 Hugging Face 模型到本地 models/base 目录。

示例：
    python dev_tools/download_hf_model.py \
      --repo-id Qwen/Qwen3-1.7B-Base \
      --local-dir models/base/qwen3_1d7B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


YHY_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="下载 Hugging Face 模型快照到本地目录。")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id，例如 Qwen/Qwen3-1.7B-Base。")
    parser.add_argument("--local-dir", type=Path, required=True, help="本地保存目录，可以是相对 yhy/ 的路径。")
    parser.add_argument(
        "--revision",
        default=None,
        help="可选模型版本、分支或 commit hash；不传时使用 Hugging Face 默认版本。",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="保留兼容参数；huggingface_hub 新版本默认支持断点续传。",
    )
    return parser.parse_args()


def main() -> None:
    """执行模型下载。"""
    args = parse_args()
    local_dir = args.local_dir
    if not local_dir.is_absolute():
        local_dir = YHY_DIR / local_dir
    local_dir = local_dir.resolve()

    print("repo_id:", args.repo_id)
    print("revision:", args.revision or "<default>")
    print("local_dir:", local_dir)

    # snapshot_download 会读取 Hugging Face 仓库的文件清单，
    # 然后把 config、tokenizer、safetensors 权重等文件下载到 local_dir。
    # local_dir_use_symlinks=False 表示在目标目录中保存真实文件，
    # 这样后续 transformers.from_pretrained(local_dir) 可以直接离线加载。
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    print("downloaded to:", path)


if __name__ == "__main__":
    main()
