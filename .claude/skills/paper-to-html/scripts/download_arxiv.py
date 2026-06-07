#!/usr/bin/env python3
"""
arXiv论文下载脚本
从arXiv链接下载PDF文件
"""

import sys
import re
import urllib.request
from pathlib import Path


def extract_arxiv_id(url_or_id):
    """
    从URL或ID中提取arXiv ID

    支持格式:
    - https://arxiv.org/abs/2301.12345
    - https://arxiv.org/pdf/2301.12345.pdf
    - 2301.12345
    - arxiv:2301.12345
    """
    # 如果是完整URL
    match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url_or_id)
    if match:
        return match.group(1)

    # 如果是arxiv:ID格式
    match = re.search(r'arxiv:(\d+\.\d+)', url_or_id, re.IGNORECASE)
    if match:
        return match.group(1)

    # 如果直接是ID
    match = re.match(r'(\d+\.\d+)', url_or_id)
    if match:
        return match.group(1)

    return None


def download_arxiv_paper(arxiv_id, output_path=None):
    """
    下载arXiv论文PDF

    Args:
        arxiv_id: arXiv ID (如 2301.12345)
        output_path: 输出文件路径，默认为当前目录下的 {arxiv_id}.pdf

    Returns:
        str: 下载的文件路径
    """
    if output_path is None:
        output_path = f"{arxiv_id}.pdf"

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    print(f"正在下载: {pdf_url}")

    try:
        urllib.request.urlretrieve(pdf_url, output_path)
        print(f"下载成功: {output_path}")
        return output_path
    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python download_arxiv.py <arxiv_url_or_id> [output_path]")
        print("\n示例:")
        print("  python download_arxiv.py https://arxiv.org/abs/2301.12345")
        print("  python download_arxiv.py 2301.12345")
        print("  python download_arxiv.py arxiv:2301.12345 paper.pdf")
        sys.exit(1)

    url_or_id = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    arxiv_id = extract_arxiv_id(url_or_id)
    if not arxiv_id:
        print(f"错误: 无法从 '{url_or_id}' 中提取arXiv ID", file=sys.stderr)
        sys.exit(1)

    download_arxiv_paper(arxiv_id, output_path)
