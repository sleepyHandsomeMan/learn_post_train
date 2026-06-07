#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用纯Python方法提取PDF文本
"""

import sys
import struct
import zlib
import re

def extract_text_objects(pdf_path):
    """从PDF中提取文本对象"""
    with open(pdf_path, 'rb') as f:
        content = f.read()

    # 查找所有文本流
    text_parts = []

    # 查找所有stream对象
    stream_pattern = rb'stream\s*\n(.*?)\nendstream'
    streams = re.findall(stream_pattern, content, re.DOTALL)

    for stream_data in streams[:50]:  # 只处理前50个流
        try:
            # 尝试解压缩
            decompressed = zlib.decompress(stream_data)
            # 提取文本内容
            text = extract_text_from_stream(decompressed)
            if text:
                text_parts.append(text)
        except:
            # 如果不是压缩的，直接提取
            text = extract_text_from_stream(stream_data)
            if text:
                text_parts.append(text)

    return '\n\n'.join(text_parts)

def extract_text_from_stream(data):
    """从PDF流中提取文本"""
    try:
        # 查找Tj和TJ操作符（显示文本）
        text_parts = []

        # 匹配 (text) Tj 或 [(text)] TJ
        pattern1 = rb'\((.*?)\)\s*Tj'
        pattern2 = rb'\[(.*?)\]\s*TJ'

        matches1 = re.findall(pattern1, data)
        matches2 = re.findall(pattern2, data)

        for match in matches1:
            try:
                text = match.decode('utf-8', errors='ignore')
                if text.strip():
                    text_parts.append(text)
            except:
                pass

        for match in matches2:
            try:
                text = match.decode('utf-8', errors='ignore')
                if text.strip():
                    text_parts.append(text)
            except:
                pass

        return ' '.join(text_parts)
    except:
        return ''

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_pdf_text.py <pdf文件路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    text = extract_text_objects(pdf_path)

    if text:
        print(text)
    else:
        print("无法提取文本，PDF可能使用了复杂的编码")
