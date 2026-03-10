import os
import json
import pandas as pd
from pypdf import PdfReader
from docx import Document


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_md(file_path: str) -> str:
    return load_txt(file_path)


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n--- Page {i + 1} ---\n{text}")
    return "\n".join(pages)


def load_docx(file_path: str) -> str:
    doc = Document(file_path)
    lines = []
    for para in doc.paragraphs:
        if para.text.strip():
            lines.append(para.text)
    return "\n".join(lines)


def load_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    summary = [
        f"Rows: {len(df)}",
        f"Columns: {', '.join(df.columns.astype(str).tolist())}",
        "",
        df.head(100).to_csv(index=False)
    ]
    return "\n".join(summary)


def load_xlsx(file_path: str) -> str:
    xl = pd.ExcelFile(file_path)
    sections = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        sections.append(f"\n--- Sheet: {sheet_name} ---")
        sections.append(f"Rows: {len(df)}")
        sections.append(f"Columns: {', '.join(df.columns.astype(str).tolist())}")
        sections.append(df.head(100).to_csv(index=False))
    return "\n".join(sections)


def load_json(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False)


def load_file_content(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".docx":
        return load_docx(file_path)
    if ext == ".txt":
        return load_txt(file_path)
    if ext == ".csv":
        return load_csv(file_path)
    if ext in [".xlsx", ".xls"]:
        return load_xlsx(file_path)
    if ext == ".json":
        return load_json(file_path)
    if ext == ".md":
        return load_md(file_path)

    raise ValueError(f"Unsupported file type: {ext}")