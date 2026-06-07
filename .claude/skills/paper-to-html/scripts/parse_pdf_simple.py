#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的PDF文本提取脚本
使用pdfplumber库（如果可用）或者直接读取PDF结构
"""

import sys
import os

def extract_text_from_pdf(pdf_path):
    """从PDF中提取文本"""
    try:
        # 尝试使用pdfplumber
        import pdfplumber

        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            print(f"PDF总页数: {len(pdf.pages)}", file=sys.stderr)
            for i, page in enumerate(pdf.pages[:30]):  # 只读前30页
                print(f"正在处理第 {i+1} 页...", file=sys.stderr)
                text = page.extract_text()
                if text:
                    text_content.append(f"\n\n=== 第 {i+1} 页 ===\n\n")
                    text_content.append(text)

        return ''.join(text_content)

    except ImportError:
        print("pdfplumber未安装，尝试使用pypdf...", file=sys.stderr)
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            text_content = []
            print(f"PDF总页数: {len(reader.pages)}", file=sys.stderr)

            for i, page in enumerate(reader.pages[:30]):  # 只读前30页
                print(f"正在处理第 {i+1} 页...", file=sys.stderr)
                text = page.extract_text()
                if text:
                    text_content.append(f"\n\n=== 第 {i+1} 页 ===\n\n")
                    text_content.append(text)

            return ''.join(text_content)

        except ImportError:
            return "错误: 需要安装 pdfplumber 或 pypdf 库"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python parse_pdf_simple.py <pdf文件路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在: {pdf_path}")
        sys.exit(1)

    text = extract_text_from_pdf(pdf_path)
    print(text)
