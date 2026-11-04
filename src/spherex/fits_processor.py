from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

LOG = logging.getLogger(__name__)


def classify_extension(extname: str) -> str:
    name = extname.upper()
    if any(token in name for token in ("VAR", "UNC", "ERROR", "SIGMA")):
        return "variance_or_uncertainty"
    if any(token in name for token in ("MASK", "FLAG", "DQ")):
        return "mask_or_quality"
    if "PSF" in name:
        return "psf"
    if any(token in name for token in ("WAVE", "LAMBDA")):
        return "wavelength_calibration"
    if "ZODI" in name:
        return "background_model"
    if any(token in name for token in ("SCI", "IMAGE", "FLUX")):
        return "science_candidate"
    return "unknown"


def inspect_fits(path: Path, plot_dir: Path) -> list[dict[str, object]]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with fits.open(path, memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            name = hdu.header.get("EXTNAME", "PRIMARY" if index == 0 else f"HDU{index}")
            data = hdu.data
            kind = classify_extension(str(name))
            finite_fraction = np.nan
            if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number) and data.size:
                finite_fraction = float(np.isfinite(data).sum() / data.size)
            row = {
                "filename": path.name,
                "hdu_index": index,
                "extname": name,
                "classification": kind,
                "shape": str(getattr(data, "shape", None)),
                "dtype": str(getattr(data, "dtype", None)),
                "finite_fraction": finite_fraction,
                "bunit": hdu.header.get("BUNIT"),
                "has_celestial_wcs": all(key in hdu.header for key in ("CTYPE1", "CTYPE2")),
            }
            rows.append(row)

            # Plot only a two-dimensional science candidate.  The plot is an
            # image diagnostic, never labelled or treated as a source spectrum.
            if kind == "science_candidate" and isinstance(data, np.ndarray) and data.ndim == 2 and data.size:
                try:
                    fig, ax = plt.subplots(figsize=(7, 6))
                    norm = ImageNormalize(data, interval=PercentileInterval(99), stretch=AsinhStretch())
                    image = ax.imshow(data, origin="lower", cmap="viridis", norm=norm)
                    short_name = path.stem if len(path.stem) <= 48 else f"{path.stem[:45]}…"
                    ax.set_title(f"{short_name}\n{name} (image diagnostic)", fontsize=10)
                    ax.set(xlabel="Pixel X", ylabel="Pixel Y")
                    fig.colorbar(image, ax=ax, label=hdu.header.get("BUNIT", "pixel value"))
                    fig.tight_layout()
                    fig.savefig(plot_dir / f"{path.stem}_hdu{index}.png", dpi=140)
                    plt.close(fig)
                except Exception as exc:
                    LOG.warning("Could not plot %s HDU %s: %s", path, index, exc)
    return rows


def inspect_directory(raw_dir: Path, plot_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.fits")):
        try:
            rows.extend(inspect_fits(path, plot_dir))
        except Exception as exc:
            rows.append({"filename": path.name, "hdu_index": None, "extname": None, "classification": "invalid", "error": str(exc)})
    return pd.DataFrame(rows)
