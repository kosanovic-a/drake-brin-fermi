from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib as mpl


_LATEX_PREAMBLE = r"""
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{amssymb}
"""


_CONFIGURED = False


def latex_available() -> bool:
    return shutil.which("latex") is not None


def configure_matplotlib_for_latex(force: bool | None = None) -> bool:
    global _CONFIGURED

    env_use_tex = os.environ.get("CIV_USE_TEX", "1").strip().lower()
    requested = env_use_tex not in {"0", "false", "no", "off"}
    if force is not None:
        requested = bool(force)

    use_tex = requested and latex_available()

    common = {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }

    if use_tex:
        mpl.rcParams.update(common | {
            "text.usetex": True,
            "text.latex.preamble": _LATEX_PREAMBLE,
        })
    else:
        mpl.rcParams.update(common | {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
        })
        

    _CONFIGURED = True
    return use_tex


def savefig_publication(fig, path, *, dpi: int | None = None, bbox_inches: str = "tight", also_pdf: bool | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")

    if dpi is None:
        dpi = int(mpl.rcParams.get("savefig.dpi", 600))

    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)
