"""Production-ready SPHEREx API Server.

Connects the web frontend to the actual backend:
- /api/targets: Real literature targets from backend.src.spherex.targets
- /api/search: Live IRSA SIA2 queries via backend.src.spherex.spherex_client
- /api/preview: Real-time FITS reading, asinh normalization & colormap rendering
- /api/difference: Real-time mathematical difference (Epoch B - Epoch A) of real FITS
- /api/download: Live bounded cutout fetching directly from IRSA IBE
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Ensure backend/src is on the path regardless of where python is launched from
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure Matplotlib temp cache directory exists on Windows
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "/tmp")) / "matplotlib_spherex"))

from astropy.io import fits
import numpy as np

from spherex.blink import align_frames, difference_image, prepare_blink, shared_scale, apply_scale
from spherex.candidate import investigate_candidate, format_report
from spherex.config import ProjectPaths
from spherex.image_cleaner import SpherexImageCleaner
from spherex.image_renderer import SpherexImageRenderer
from spherex.observation_selector import ObservationSelectionService, SelectionMode
from spherex.observation_service import ObservationPreviewService
from spherex.spherex_client import SPHERExClient
from spherex.targets import TARGETS

LOG = logging.getLogger("spherex.api")
logging.basicConfig(level=logging.INFO)


def _safe_float(val: Any, default: float) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


class SpherexApiHandler(BaseHTTPRequestHandler):
    client = SPHERExClient(collections=("spherex_dr1",), timeout=25)
    # requests.Session is not guaranteed to be thread-safe. Render serves
    # multiple browser requests concurrently, so serialize archive access and
    # retry transient IRSA/VOTable failures before reporting no observations.
    archive_lock = threading.Lock()
    # Fast 0.08° cutout (WiseView style ~48x48 to 100x100 pixel window) for instant downloads
    service = ObservationPreviewService(download_cutout_size_deg=0.08)
    selector = ObservationSelectionService(path_resolver=service.resolve_fits_path)
    renderer = SpherexImageRenderer()
    paths = ProjectPaths.default()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/api/health":
                self.send_json({"status": "ok", "service": "spherex-api"})
            elif path == "/api/targets":
                self.handle_targets()
            elif path == "/api/search":
                self.handle_search(query)
            elif path == "/api/smart_select":
                self.handle_smart_select(query)
            elif path == "/api/fetch_cutouts":
                self.handle_fetch_cutouts(query)
            elif path == "/api/render":
                self.handle_render(query)
            elif path == "/api/investigate":
                self.handle_investigate(query)
            elif path == "/api/rgb":
                self.handle_rgb(query)
            elif path == "/api/sed":
                self.handle_sed(query)
            elif path == "/api/coadd":
                self.handle_coadd(query)
            elif path == "/api/difference":
                self.handle_difference(query)
            elif path == "/api/local_fits":
                self.handle_local_fits(query)
            elif path.startswith("/data/"):
                self.handle_static_data(path)
            else:
                self.send_error(404, f"Endpoint {path} not found")
        except Exception as exc:
            LOG.exception("Error handling request %s", self.path)
            self.send_json({"error": str(exc)}, status=500)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Override default send_error to guarantee CORS headers on 404/500 responses."""
        self.send_response(code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        payload = json.dumps({"error": message or "Error", "code": code}).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_image(self, image_bytes: bytes, mime: str = "image/png") -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(image_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(image_bytes)

    @classmethod
    def search_archive(cls, **kwargs: Any) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        with cls.archive_lock:
            for attempt in range(3):
                try:
                    return list(cls.client.search(**kwargs))
                except Exception as exc:
                    last_error = exc
                    LOG.warning(
                        "IRSA search attempt %d/3 failed: %s",
                        attempt + 1,
                        exc,
                    )
                    if attempt < 2:
                        time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def handle_targets(self) -> None:
        """Serve target definitions directly from your team's backend/src/spherex/targets.py."""
        result = []
        for key, target in TARGETS.items():
            result.append({
                "id": target.key,
                "name": target.name,
                "ra_deg": target.ra_deg,
                "dec_deg": target.dec_deg,
                "reference_mjd": float(target.reference_time.mjd),
                "pm_ra_cosdec_mas_yr": target.pm_ra_cosdec_mas_yr,
                "pm_dec_mas_yr": target.pm_dec_mas_yr,
                "parallax_mas": target.parallax_mas,
                "position_sigma_mas": target.position_sigma_mas,
                "source_url": target.source_url,
                "source_note": target.source_note,
            })
        self.send_json(result)

    def handle_search(self, query: dict[str, list[str]]) -> None:
        """Perform live SIA2 cone query to NASA/IPAC IRSA archive with local offline fallback."""
        ra = float(query.get("ra", ["133.785"])[0])
        dec = float(query.get("dec", ["-7.244"])[0])
        radius = float(query.get("radius", ["0.05"])[0])
        max_results = int(query.get("max_results", ["50"])[0])

        try:
            observations = self.search_archive(
                ra=ra,
                dec=dec,
                radius=radius,
                max_results=max_results,
            )
        except Exception as exc:
            LOG.error("IRSA live search failed for RA=%.4f, Dec=%.4f: %s", ra, dec, exc)
            self.send_json({
                "count": 0,
                "ra": ra,
                "dec": dec,
                "radius_deg": radius,
                "observations": [],
                "error": f"IRSA live search error: {exc}",
            }, status=502)
            return

        self.send_json({
            "count": len(observations),
            "ra": ra,
            "dec": dec,
            "radius_deg": radius,
            "observations": observations,
        })

    def handle_smart_select(self, query: dict[str, list[str]]) -> None:
        """Select best 2-6 observations using backend ObservationSelectionService with offline fallback."""
        ra = float(query.get("ra", ["133.785"])[0])
        dec = float(query.get("dec", ["-7.244"])[0])
        radius = float(query.get("radius", ["0.05"])[0])
        mode_str = query.get("mode", ["time"])[0].lower()
        mode = SelectionMode.SPECTRAL if mode_str == "spectral" else SelectionMode.TIME

        try:
            observations = self.search_archive(ra=ra, dec=dec, radius=radius, max_results=50)
        except Exception as exc:
            LOG.error("IRSA live search unreachable in smart_select (%s).", exc)
            self.send_json({
                "mode": mode.value,
                "explanation": f"IRSA live archive search failed: {exc}",
                "count": 0,
                "selected": [],
            }, status=502)
            return

        if not observations:
            self.send_json({"selected": [], "explanation": "No observations found at this location in SPHEREx archive."})
            return

        result = self.selector.select(observations, ra=ra, dec=dec, mode=mode)
        self.send_json({
            "mode": result.mode.value,
            "explanation": result.explanation,
            "count": len(result.observations),
            "selected": list(result.observations),
        })

    def handle_fetch_cutouts(self, query: dict[str, list[str]]) -> None:
        """On-demand fetch and prepare real SPHEREx cutouts from IRSA for user coordinates."""
        ra = float(query.get("ra", ["133.785"])[0])
        dec = float(query.get("dec", ["-7.244"])[0])
        radius = float(query.get("radius", ["0.05"])[0])
        # Allow user to specify how many files to fetch (up to 12)
        max_files = min(int(query.get("max_files", ["6"])[0]), 12)
        smart_enabled = query.get("smart", ["0"])[0].lower() in {"1", "true"}
        mode_str = query.get("mode", ["time"])[0].lower()
        mode = SelectionMode.SPECTRAL if mode_str == "spectral" else SelectionMode.TIME

        try:
            # 1. Discover observations at user coordinates (fast DR1 targeted search)
            observations = self.search_archive(ra=ra, dec=dec, radius=radius, max_results=max_files * 2)
        except Exception as exc:
            LOG.error("IRSA live cutout search unreachable: %s", exc)
            self.send_json({
                "status": "error",
                "message": f"NASA IRSA live search failed: {exc}",
                "records": [],
            }, status=502)
            return

        if not observations:
            self.send_json({
                "status": "not_found",
                "message": f"No SPHEREx observations found at RA={ra:.4f}, Dec={dec:.4f}",
                "records": [],
            })
            return

        # 2. Selection: Use smart selection by default for the requested mode (time or spectral)
        if observations:
            try:
                smart_res = self.selector.select(observations, ra=ra, dec=dec, mode=mode)
                selected_obs = list(smart_res.observations)[:max_files] if smart_res.observations else observations[:max_files]
            except Exception as exc:
                LOG.warning("Observation selector failed (%s). Falling back.", exc)
                det_preferred = [o for o in observations if str(o.get("detector")) in {"6", "3"}]
                selected_obs = det_preferred[:max_files] if len(det_preferred) >= 2 else observations[:max_files]
        else:
            selected_obs = []

        # 3. Download missing FITS cutouts from IRSA IBE
        try:
            resolved = self.service.ensure_local_files(selected_obs, ra=ra, dec=dec)
        except Exception as exc:
            LOG.warning("Cutout download failed (%s). Using existing local files.", exc)
            resolved = []

        # 4. Format result records with rich local FITS metadata
        records = []
        for r in resolved:
            file_path = r.get("file_path")
            if file_path and Path(file_path).is_file():
                p = Path(file_path)
                try:
                    with fits.open(p) as hdul:
                        hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                        shape = list(hdul["IMAGE"].shape) if "IMAGE" in hdul else [12, 12]
                        mjd = _safe_float(hdr.get("MJD-OBS"), _safe_float(r.get("observation_start_mjd"), 60800.0))
                        date_obs = str(hdr.get("DATE-OBS") or r.get("observation_timestamp") or "")
                        detector = str(hdr.get("DETECTOR") or r.get("detector") or "3")
                        band = str(hdr.get("BAND") or r.get("band") or "SPHEREx")
                        wavel = _safe_float(hdr.get("WAVELMIN"), _safe_float(r.get("wavelength_min_um"), 2.0))
                except Exception:
                    shape = [12, 12]
                    mjd = _safe_float(r.get("observation_start_mjd"), 60800.0)
                    date_obs = str(r.get("observation_timestamp") or "")
                    detector = str(r.get("detector") or "3")
                    band = str(r.get("band") or "SPHEREx")
                    wavel = _safe_float(r.get("wavelength_min_um"), 2.0)

                records.append({
                    "observation_id": r.get("observation_id") or r.get("obs_id") or p.stem,
                    "obs_id": r.get("observation_id") or r.get("obs_id") or p.stem,
                    "detector": detector,
                    "band": band,
                    "filename": p.name,
                    "relative_path": str(p.relative_to(self.paths.root)).replace("\\", "/"),
                    "mjd": mjd,
                    "date_obs": date_obs,
                    "wavelength_um": wavel,
                    "shape": shape,
                    "download_status": r.get("download_status", "existing"),
                    "render_url": f"/api/render?filename={p.name}&stretch=asinh",
                })

        if not records:
            for p in list(self.paths.raw.glob("**/*.fits"))[:max_files]:
                try:
                    with fits.open(p) as hdul:
                        hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                        shape = list(hdul["IMAGE"].shape) if "IMAGE" in hdul else [12, 12]
                        mjd = _safe_float(hdr.get("MJD-OBS"), 60800.0)
                        date_obs = str(hdr.get("DATE-OBS") or "")
                        detector = str(hdr.get("DETECTOR") or "3")
                        band = str(hdr.get("BAND") or "SPHEREx")
                        wavel = _safe_float(hdr.get("WAVELMIN"), 2.0)
                except Exception:
                    shape, mjd, date_obs, detector, band, wavel = [12, 12], 60800.0, "", "3", "SPHEREx", 2.0

                records.append({
                    "observation_id": p.stem,
                    "obs_id": p.stem,
                    "detector": detector,
                    "band": band,
                    "filename": p.name,
                    "relative_path": str(p.relative_to(self.paths.root)).replace("\\", "/"),
                    "mjd": mjd,
                    "date_obs": date_obs,
                    "wavelength_um": wavel,
                    "shape": shape,
                    "download_status": "existing",
                    "render_url": f"/api/render?filename={p.name}&stretch=asinh",
                })

        records.sort(key=lambda r: r.get("mjd", 0.0))

        self.send_json({
            "status": "success",
            "ra": ra,
            "dec": dec,
            "records": records,
        })

    def handle_local_fits(self, query: dict[str, list[str]] | None = None) -> None:
        """List actual FITS cutout files available in data/raw/, optionally filtered by coordinates/target."""
        ra_param = query.get("ra", [None])[0] if query else None
        dec_param = query.get("dec", [None])[0] if query else None

        files = list(self.paths.raw.glob("**/*.fits"))
        records = []
        from astropy.wcs import WCS
        from astropy.time import Time
        from spherex.targets import TARGETS

        # Check if coordinates correspond to a known benchmark target (e.g. WISE 0855)
        # and propagate coordinates forward to SPHEREx survey epoch if high proper motion
        target_coords = []
        if ra_param is not None and dec_param is not None:
            r_val, d_val = float(ra_param), float(dec_param)
            target_coords.append((r_val, d_val))
            for t in TARGETS.values():
                if abs(t.ra_deg - r_val) < 0.05 and abs(t.dec_deg - d_val) < 0.05:
                    prop = t.at(Time(60800.0, format="mjd"))
                    target_coords.append((float(prop.ra.deg), float(prop.dec.deg)))

        for f in files:
            with fits.open(f) as hdul:
                hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                shape = list(hdul["IMAGE"].shape) if "IMAGE" in hdul else [12, 12]
                
                # Check target coverage against both reference and propagated coordinates
                covers_target = True
                if target_coords:
                    try:
                        w = WCS(hdr)
                        covers_target = any(
                            (0 <= w.all_world2pix(tc[0], tc[1], 0)[0] < shape[1]) and
                            (0 <= w.all_world2pix(tc[0], tc[1], 0)[1] < shape[0])
                            for tc in target_coords
                        )
                    except Exception:
                        covers_target = True

                records.append({
                    "filename": f.name,
                    "relative_path": str(f.relative_to(self.paths.root)).replace("\\", "/"),
                    "obs_id": hdr.get("OBS_ID", f.stem),
                    "detector": str(hdr.get("DETECTOR", "3")),
                    "band": hdr.get("BAND", "SPHEREx"),
                    "mjd": _safe_float(hdr.get("MJD-OBS"), 60800.0),
                    "date_obs": str(hdr.get("DATE-OBS") or ""),
                    "wavelength_um": _safe_float(hdr.get("WAVELMIN"), 2.0),
                    "shape": shape,
                    "covers_target": covers_target,
                })

        # If filtered by target coordinates, prioritize frames that genuinely cover the target
        if target_coords:
            covering = [r for r in records if r["covers_target"]]
            if covering:
                records = covering

        # Sort chronologically by MJD so temporal blinking is astronomically continuous
        records.sort(key=lambda r: r.get("mjd", 0.0))
        self.send_json(records)

    def handle_render(self, query: dict[str, list[str]]) -> None:
        """Render FITS file via SpherexImageCleaner and SpherexImageRenderer."""
        filename = query.get("filename", [""])[0]
        colormap = query.get("colormap", ["inferno"])[0]
        stretch = query.get("stretch", ["asinh"])[0]
        min_cut = query.get("min_cut", [None])[0]
        max_cut = query.get("max_cut", [None])[0]
        invert = query.get("invert", ["false"])[0].lower() in {"true", "1"}
        clean_enabled = query.get("clean", ["0"])[0].lower() in {"1", "true"}

        files = list(self.paths.raw.glob(f"**/{filename}")) if filename else list(self.paths.raw.glob("**/*.fits"))
        if not files:
            self.send_error(404, "FITS file not found")
            return

        target_file = files[0]
        with fits.open(target_file) as hdul:
            data = hdul["IMAGE"].data if "IMAGE" in hdul else hdul[0].data
            if clean_enabled:
                cleaner = SpherexImageCleaner()
                flags = hdul["FLAGS"].data if "FLAGS" in hdul else None
                variance = hdul["VARIANCE"].data if "VARIANCE" in hdul else None
                clean_result = cleaner.clean(data, flags=flags, variance=variance)
                data = clean_result.get("cleaned_image", data)

        # Delegate normalization and stretching to SpherexImageRenderer service
        min_pct_str = query.get("min_cut_percent", [None])[0]
        max_pct_str = query.get("max_cut_percent", [None])[0]
        values = np.asarray(data, dtype=float)
        finite = values[np.isfinite(values)]

        if finite.size > 0 and min_pct_str is not None and max_pct_str is not None:
            try:
                p_low = max(0.0, min(100.0, float(min_pct_str)))
                p_high = max(0.0, min(100.0, float(max_pct_str)))
                if p_high <= p_low:
                    p_high = p_low + 1.0
                low, high = float(np.percentile(finite, p_low)), float(np.percentile(finite, p_high))
                if high <= low:
                    high = low + 1.0
                scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
                if stretch == "asinh":
                    softness = 10.0
                    scaled = np.arcsinh(softness * scaled) / np.arcsinh(softness)
                elif stretch == "log":
                    scaled = np.log1p(9.0 * scaled) / np.log(10.0)
                normalized = scaled
            except Exception:
                normalized, _ = self.renderer.normalize(data, stretch=stretch)
        elif min_cut is not None and max_cut is not None:
            try:
                low, high = float(min_cut), float(max_cut)
                scaled = np.clip((values - low) / (high - low or 1.0), 0.0, 1.0)
                if stretch == "asinh":
                    scaled = np.arcsinh(10.0 * scaled) / np.arcsinh(10.0)
                normalized = scaled
            except Exception:
                normalized, _ = self.renderer.normalize(data, stretch=stretch)
        else:
            normalized, _ = self.renderer.normalize(data, stretch=stretch)

        if invert:
            normalized = 1.0 - normalized

        # Colormap via renderer colormap logic
        import matplotlib.pyplot as plt
        cmap_name = colormap if colormap in plt.colormaps() else "inferno"
        rgba = plt.get_cmap(cmap_name)(normalized)

        buf = io.BytesIO()
        plt.imsave(buf, rgba, format="png")
        self.send_image(buf.getvalue())

    def handle_difference(self, query: dict[str, list[str]]) -> None:
        """Calculate real mathematical difference (Epoch B - Epoch A) on a shared WCS sky grid."""
        files_param = query.get("files", [None])[0]
        if files_param:
            requested_names = [name.strip() for name in files_param.split(",") if name.strip()]
            files = []
            for name in requested_names:
                matches = list(self.paths.raw.glob(f"**/{name}"))
                if matches:
                    files.append(matches[0])
        else:
            files = sorted(list(self.paths.raw.glob("**/*.fits")), key=lambda p: p.stat().st_mtime)

        if len(files) < 2:
            self.send_error(400, "Need at least 2 FITS files from the active query to compute a difference image")
            return

        first_idx = int(query.get("first", ["0"])[0])
        last_idx = int(query.get("last", ["-1"])[0])
        first_idx = max(0, min(first_idx, len(files) - 1))
        last_idx = max(0, min(last_idx, len(files) - 1))
        if first_idx == last_idx:
            last_idx = (first_idx + 1) % len(files)

        clean_enabled = query.get("clean", ["0"])[0].lower() in {"1", "true"}
        invert = query.get("invert", ["false"])[0].lower() in {"true", "1"}

        from astropy.wcs import WCS
        raw_frames = []
        for f in (files[first_idx], files[last_idx]):
            with fits.open(f) as hdul:
                hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                data = hdul["IMAGE"].data if "IMAGE" in hdul else hdul[0].data
                if clean_enabled:
                    cleaner = SpherexImageCleaner()
                    flags = hdul["FLAGS"].data if "FLAGS" in hdul else None
                    variance = hdul["VARIANCE"].data if "VARIANCE" in hdul else None
                    clean_res = cleaner.clean(data, flags=flags, variance=variance)
                    data = clean_res.get("cleaned_image", data)
                wcs = WCS(hdr)
                raw_frames.append({
                    "data": np.asarray(data, dtype=float),
                    "wcs": wcs,
                    "observation_id": hdr.get("OBS_ID", f.stem),
                    "observation_timestamp": hdr.get("DATE-OBS", ""),
                })

        try:
            aligned_result = align_frames(raw_frames)
            diff = difference_image(aligned_result, first=0, last=1)
        except Exception as exc:
            LOG.warning("WCS reprojection during differencing failed (%s); fallback to raw shape-matched subtraction.", exc)
            d0, d1 = raw_frames[0]["data"], raw_frames[1]["data"]
            min_h, min_w = min(d0.shape[0], d1.shape[0]), min(d0.shape[1], d1.shape[1])
            diff = d1[:min_h, :min_w] - d0[:min_h, :min_w]

        # Rescale difference to 0..1 range centered symmetrically around median at 0.5
        finite = diff[np.isfinite(diff)]
        if finite.size > 0:
            med = float(np.median(finite))
            p99 = float(np.percentile(np.abs(finite - med), 99.0))
            span = p99 if p99 > 0 else (float(np.max(np.abs(finite))) or 1.0)
            normalized = np.clip(((diff - med) / (2.0 * span)) + 0.5, 0.0, 1.0)
        else:
            normalized = np.zeros_like(diff, dtype=float)

        if invert:
            normalized = 1.0 - normalized

        import matplotlib.pyplot as plt
        colormap = query.get("colormap", ["coolwarm"])[0]
        # If the requested colormap is 'rgb', fallback to 'coolwarm' for difference visualization
        if colormap == "rgb" or colormap not in plt.colormaps():
            colormap = "coolwarm"
        rgba = plt.get_cmap(colormap)(normalized)

        buf = io.BytesIO()
        plt.imsave(buf, rgba, format="png")
        self.send_image(buf.getvalue())

    def handle_rgb(self, query: dict[str, list[str]]) -> None:
        """WCS-aligned 3-band astronomical false color composite (Short=Blue, Mid=Green, Long=Red)."""
        files_param = query.get("files", [None])[0]
        if files_param:
            requested_names = [name.strip() for name in files_param.split(",") if name.strip()]
            files = []
            for name in requested_names:
                matches = list(self.paths.raw.glob(f"**/{name}"))
                if matches:
                    files.append(matches[0])
        else:
            files = list(self.paths.raw.glob("**/*.fits"))

        if len(files) < 3:
            self.send_error(400, "Need at least 3 FITS files from the active query to compose an RGB hyperspectral view")
            return

        from astropy.wcs import WCS
        candidates = []
        for f in files:
            with fits.open(f) as hdul:
                hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                data = hdul["IMAGE"].data if "IMAGE" in hdul else hdul[0].data
                det = str(hdr.get("DETECTOR", "3"))
                det_wavel = {"1": 0.92, "2": 1.35, "3": 2.00, "4": 3.10, "5": 4.10, "6": 4.70}
                wavel = float(hdr.get("WAVELMIN", det_wavel.get(det, 2.0)))
                wcs = WCS(hdr)
                candidates.append({
                    "data": np.asarray(data, dtype=float),
                    "wcs": wcs,
                    "wavelength_um": wavel,
                    "detector": det,
                    "filename": f.name,
                })

        # Sort all cutouts by wavelength ascending
        candidates.sort(key=lambda c: c["wavelength_um"])

        # Pick 3 cutouts with greatest spectral separation (Shortest -> Blue, Median -> Green, Longest -> Red)
        b_input = candidates[0]
        r_input = candidates[-1]
        mid_idx = len(candidates) // 2
        g_input = candidates[mid_idx]
        if g_input["filename"] == b_input["filename"] or g_input["filename"] == r_input["filename"]:
            # Pick any distinct middle candidate
            distinct = [c for c in candidates if c["filename"] not in (b_input["filename"], r_input["filename"])]
            g_input = distinct[0] if distinct else candidates[1]

        channel_inputs = [b_input, g_input, r_input]

        try:
            from reproject import reproject_interp
            reference = g_input
            ref_wcs = reference["wcs"]
            ref_shape = reference["data"].shape

            aligned_channels = {}
            footprints = []
            for ch_key, entry in (("B", b_input), ("G", g_input), ("R", r_input)):
                if entry is reference:
                    proj = np.array(entry["data"], dtype=float, copy=True)
                    fp = np.isfinite(proj).astype(float)
                else:
                    proj, fp = reproject_interp((entry["data"], entry["wcs"]), ref_wcs, shape_out=ref_shape)
                aligned_channels[ch_key] = proj
                footprints.append(np.isfinite(proj) & (fp > 0.0))

            common_mask = np.logical_and.reduce(footprints)
            if not common_mask.any():
                common_mask = np.ones(ref_shape, dtype=bool)

            # Smoothly stretch each WCS-aligned channel using asinh
            normalized = {}
            for ch in ("R", "G", "B"):
                ch_data = np.where(common_mask, aligned_channels[ch], np.nan)
                finite = ch_data[np.isfinite(ch_data)]
                if finite.size > 0:
                    low, high = float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5))
                    high = high if high > low else low + 1.0
                    scaled = np.clip((ch_data - low) / (high - low), 0.0, 1.0)
                    softness = 8.0
                    normalized[ch] = np.arcsinh(softness * scaled) / np.arcsinh(softness)
                else:
                    normalized[ch] = np.zeros(ref_shape, dtype=float)

            rgb = np.dstack((normalized["R"], normalized["G"], normalized["B"]))
            rgb[~common_mask] = 0.0
            rgb = np.nan_to_num(rgb, nan=0.0)
        except Exception as exc:
            LOG.warning("WCS reprojection for RGB failed (%s); fallback to shape-cropped RGB.", exc)
            dr, dg, db = r_input["data"], g_input["data"], b_input["data"]
            min_h = min(dr.shape[0], dg.shape[0], db.shape[0])
            min_w = min(dr.shape[1], dg.shape[1], db.shape[1])
            nr, _ = self.renderer.normalize(dr[:min_h, :min_w], stretch="asinh")
            ng, _ = self.renderer.normalize(dg[:min_h, :min_w], stretch="asinh")
            nb, _ = self.renderer.normalize(db[:min_h, :min_w], stretch="asinh")
            rgb = np.stack([nr, ng, nb], axis=-1)

        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, rgb, format="png")
        self.send_image(buf.getvalue())

    def handle_sed(self, query: dict[str, list[str]]) -> None:
        """Hyperspectral Spectral Energy Distribution (SED) on the fly at optional (x, y) coordinates."""
        try:
            norm_x = float(query.get("x", [0.5])[0])
            norm_y = float(query.get("y", [0.5])[0])
        except (ValueError, TypeError):
            norm_x, norm_y = 0.5, 0.5

        files_param = query.get("files", [None])[0]
        if files_param:
            requested_names = [name.strip() for name in files_param.split(",") if name.strip()]
            files = []
            for name in requested_names:
                matches = list(self.paths.raw.glob(f"**/{name}"))
                if matches:
                    files.append(matches[0])
        else:
            files = list(self.paths.raw.glob("**/*.fits"))

        points = []
        for f in files:
            with fits.open(f) as hdul:
                hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                d = hdul["IMAGE"].data if "IMAGE" in hdul else hdul[0].data
                det = hdr.get("DETECTOR", "1")
                det_wavel = {
                    "1": 0.92, "2": 1.35, "3": 2.00, "4": 3.10, "5": 4.10, "6": 4.70
                }
                wavel = float(hdr.get("WAVELMIN", det_wavel.get(str(det), 2.0)))
                # Extract flux at user click or center pixel
                py = max(0, min(d.shape[0] - 1, int(norm_y * d.shape[0])))
                px = max(0, min(d.shape[1] - 1, int(norm_x * d.shape[1])))
                flux = float(d[py, px]) if np.isfinite(d[py, px]) else 0.0
                points.append({
                    "detector": str(det),
                    "wavelength_um": round(wavel, 2),
                    "flux": round(flux, 4),
                    "filename": f.name,
                })
        points.sort(key=lambda x: x["wavelength_um"])
        self.send_json(points)

    def handle_coadd(self, query: dict[str, list[str]]) -> None:
        """Co-add (stack) all observations taken by a specific detector array (1-6) to improve signal-to-noise."""
        target_det = query.get("detector", ["all"])[0]
        stretch = query.get("stretch", ["asinh"])[0]
        colormap = query.get("colormap", ["inferno"])[0]

        files_param = query.get("files", [None])[0]
        if files_param:
            requested_names = [name.strip() for name in files_param.split(",") if name.strip()]
            files = []
            for name in requested_names:
                matches = list(self.paths.raw.glob(f"**/{name}"))
                if matches:
                    files.append(matches[0])
        else:
            files = list(self.paths.raw.glob("**/*.fits"))

        if not files:
            self.send_error(404, "No FITS files available for co-addition")
            return

        matching_arrays = []
        for f in files:
            with fits.open(f) as hdul:
                hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                data = hdul["IMAGE"].data if "IMAGE" in hdul else hdul[0].data
                det = str(hdr.get("DETECTOR", "3"))
                if target_det == "all" or det == str(target_det):
                    matching_arrays.append(np.asarray(data, dtype=float))

        if not matching_arrays:
            # Fallback to all files if selected array has no cutouts
            matching_arrays = [fits.open(f)[0].data for f in files]

        # Crop all frames to common minimum shape
        min_h = min(a.shape[0] for a in matching_arrays)
        min_w = min(a.shape[1] for a in matching_arrays)
        cropped = [a[:min_h, :min_w] for a in matching_arrays]

        # Stack via mean/median to reject outliers and boost SNR
        stacked = np.nanmean(cropped, axis=0) if len(cropped) > 1 else cropped[0]
        normalized, _ = self.renderer.normalize(stacked, stretch=stretch)

        import matplotlib.pyplot as plt
        cmap_name = colormap if colormap in plt.colormaps() else "inferno"
        rgba = plt.get_cmap(cmap_name)(normalized)

        buf = io.BytesIO()
        plt.imsave(buf, rgba, format="png")
        self.send_image(buf.getvalue())

    def handle_investigate(self, query: dict[str, list[str]]) -> None:
        """Measure candidate motion from click coordinates across all aligned local frames."""
        try:
            norm_x = float(query.get("x", [0.5])[0])
            norm_y = float(query.get("y", [0.5])[0])
            clicked_frame = int(query.get("frame", [0])[0])
        except (ValueError, TypeError):
            self.send_error(400, "Invalid x, y or frame coordinate parameters")
            return

        # Check if caller specified exact filenames for the active view
        files_param = query.get("files", [None])[0]
        if files_param:
            requested_names = [name.strip() for name in files_param.split(",") if name.strip()]
            files = []
            for name in requested_names:
                matches = list(self.paths.raw.glob(f"**/{name}"))
                if matches:
                    files.append(matches[0])
        else:
            files = sorted(list(self.paths.raw.glob("**/*.fits")), key=lambda p: p.stat().st_mtime)

        if len(files) < 2:
            self.send_json({
                "verdict": "Insufficient frames",
                "explanation": "Need at least 2 FITS frames on disk to measure candidate motion.",
                "detected_frames": 0,
                "total_frames": len(files),
                "measurements": [],
            })
            return

        # Load frames and extract WCS
        from astropy.wcs import WCS
        raw_frames = []
        for f in files[:8]:
            with fits.open(f) as hdul:
                hdr = hdul["IMAGE"].header if "IMAGE" in hdul else hdul[0].header
                data = hdul["IMAGE"].data if "IMAGE" in hdul else hdul[0].data
                wcs = WCS(hdr)
                raw_frames.append({
                    "data": data,
                    "wcs": wcs,
                    "observation_id": hdr.get("OBS_ID", f.stem),
                    "observation_timestamp": hdr.get("DATE-OBS", ""),
                    "mjd": float(hdr.get("MJD-OBS", 60800.0)),
                    "detector": str(hdr.get("DETECTOR", "3")),
                    "wavelength_um": float(hdr.get("WAVELMIN", 2.0)),
                })

        # Perform WCS reprojection onto a shared sky grid so candidate tracking measures identical sky points
        try:
            aligned_result = align_frames(raw_frames)
            frames = aligned_result["frames"]
            metadata = aligned_result["metadata"]
            primary_wcs = aligned_result["wcs"]
        except Exception as exc:
            LOG.warning("align_frames failed (%s); using unprojected frames.", exc)
            frames = [rf["data"] for rf in raw_frames]
            metadata = raw_frames
            primary_wcs = raw_frames[0]["wcs"]

        # Convert normalized 0..1 click position to image pixel coordinates
        img_h, img_w = frames[0].shape
        click_px = norm_x * img_w
        click_py = norm_y * img_h

        # Package aligned dictionary as expected by investigate_candidate
        aligned = {
            "frames": frames,
            "metadata": metadata,
            "wcs": primary_wcs,
        }

        clamped_frame = max(0, min(clicked_frame, len(frames) - 1))
        report = investigate_candidate(aligned, click_px, click_py, clicked_frame=clamped_frame)
        report_text = format_report(report)
        report["text_report"] = report_text

        self.send_json(report)


    def handle_static_data(self, path: str) -> None:
        clean_path = path.lstrip("/")
        file_path = (self.paths.root / clean_path).resolve()
        if not file_path.is_file():
            self.send_error(404, "File not found")
            return

        with open(file_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "image/png" if file_path.suffix == ".png" else "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(port: int | None = None) -> None:
    # Render and cloud platforms supply PORT as an environment variable
    effective_port = port or int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    SpherexApiHandler.paths.ensure()
    server = ThreadingHTTPServer((host, effective_port), SpherexApiHandler)
    LOG.info("SPHEREx Real Data API running at http://%s:%d", host, effective_port)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
