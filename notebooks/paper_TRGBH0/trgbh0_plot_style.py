"""Shared plotting style for the TRGBH0 paper figures."""

from matplotlib.colors import LinearSegmentedColormap


TRGBH0_COLOURS = [
    "#ef476f",
    "#473198",
    "#a8c256",
    "#5adbff",
    "#fe9000",
]


def trgbh0_cmap(name="trgbh0"):
    return LinearSegmentedColormap.from_list(name, TRGBH0_COLOURS)
