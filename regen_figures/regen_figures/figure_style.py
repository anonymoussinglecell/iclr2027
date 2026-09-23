#!/usr/bin/env python
"""Vector-output settings shared by the figures.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAHOE = # set path
if TAHOE not in sys.path:
    sys.path.append(TAHOE)
from report_common import save_fig 

plt.rcParams.update({
    "pdf.fonttype": 42,       # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "svg.fonttype": "none",   # keep SVG text as text, editable 
})

FORMATS = ("png", "pdf", "svg")


def save(fig, stem, formats=FORMATS, dpi=300, close=True, **kw):
    """Write fig to <stem>.<ext> for each format. Returns the stem."""
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in formats:
        save_fig(fig, f"{stem}.{ext}", dpi=dpi, **kw)
    if close:
        plt.close(fig)
    return stem
