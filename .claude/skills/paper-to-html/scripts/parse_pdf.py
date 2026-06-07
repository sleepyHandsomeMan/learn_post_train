#!/usr/bin/env python3
"""
PDF论文解析脚本
提取论文的文本、结构、公式、图表等信息
"""

import sys
import json
from pathlib import Path

def parse_pdf(pdf_path):
    """
    解析PDF文件，提取论文内容

    Args:
        pdf_path: PDF文件路径

    Returns:
        dict: 包含论文结构化信息的字典
    """
    try:
        import pdfplumber
    except ImportError:
        print("警告: pdfplumber未安装，尝试使用PyPDF2", file=sys.stderr)
        try:
            import PyPDF2
            return parse_with_pypdf2(pdf_path)
        except ImportError:
            print("错误: 需要安装pdfplumber或PyPDF2", file=sys.stderr)
            print("运行: pip install pdfplumber", file=sys.stderr)
            sys.exit(1)

    result = {
        "title": "",
        "authors": [],
        "abstract": "",
        "sections": [],
        "figures": [],
        "tables": [],
        "references": [],
        "full_text": ""
    }

    with pdfplumber.open(pdf_path) as pdf:
        full_text = []

        for page_num, page in enumerate(pdf.pages, 1):
            # 提取文本
            text = page.extract_text()
            if text:
                full_text.append(text)

            # 提取表格
            tables = page.extract_tables()
            for table in tables:
                result["tables"].append({
                    "page": page_num,
                    "data": table
                })

        result["full_text"] = "\n\n".join(full_text)

        # 简单的章节识别（基于常见模式）
        lines = result["full_text"].split("\n")
        current_section = None
        section_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 识别章节标题（简单启发式）
            if any(keyword in line.lower() for keyword in
                   ["abstract", "introduction", "related work", "method",
                    "experiment", "result", "conclusion", "reference"]):
                if current_section:
                    result["sections"].append({
                        "title": current_section,
                        "content": "\n".join(section_text)
                    })
                current_section = line
                section_text = []
            else:
                section_text.append(line)

        # 添加最后一个章节
        if current_section:
            result["sections"].append({
                "title": current_section,
                "content": "\n".join(section_text)
            })

    return result


def parse_with_pypdf2(pdf_path):
    """使用PyPDF2作为备选方案"""
    import PyPDF2

    result = {
        "title": "",
        "authors": [],
        "abstract": "",
        "sections": [],
        "figures": [],
        "tables": [],
        "references": [],
        "full_text": ""
    }

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

        result["full_text"] = "\n\n".join(full_text)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python parse_pdf.py <pdf_path> [output_json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = parse_pdf(pdf_path)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"解析结果已保存到: {output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
