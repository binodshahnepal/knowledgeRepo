import pandas as pd


def summarize_file_metadata(file_name: str, content: str) -> dict:
    lines = content.splitlines()
    words = len(content.split())

    return {
        "file_name": file_name,
        "characters": len(content),
        "lines": len(lines),
        "words": words,
        "preview": content[:500]
    }


def dataframe_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns]
    }