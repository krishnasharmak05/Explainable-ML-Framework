from typing import List, Optional
import pandas as pd


def format_results_markdown_table(
    df: pd.DataFrame,
    headers: Optional[List[str]] = None,
) -> str:
    """
    Renders a pandas DataFrame as a GitHub-flavored Markdown table natively
    without requiring external dependencies like tabulate.
    """
    if df.empty:
        return ""

    cols = headers if (headers and len(headers) == len(df.columns)) else [str(c) for c in df.columns]
    header_line = "| " + " | ".join(cols) + " |"
    sep_line = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        rows.append(row_str)

    return "\n".join([header_line, sep_line] + rows)


def format_latex_table(
    df: pd.DataFrame,
    caption: str = "Experimental Results",
    label: str = "tab:results",
) -> str:
    """
    Renders a pandas DataFrame as a publication-quality LaTeX table natively
    without requiring jinja2.
    """
    if df.empty:
        return ""

    cols = [str(c).replace("_", "\\_") for c in df.columns]
    col_spec = "l" + "r" * (len(cols) - 1)
    header = " & ".join(cols) + " \\\\"

    rows = []
    for _, row in df.iterrows():
        row_vals = [str(val).replace("_", "\\_") for val in row.values]
        rows.append(" & ".join(row_vals) + " \\\\")

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header,
        "\\hline",
    ] + rows + [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)
