# SPHEREx Phase 1 ingestion

This project discovers NASA SPHEREx spectral images at IRSA for a moving target, downloads only small FITS cutouts, validates them, and inventories their HDUs. The initial target is the nearby high-proper-motion brown dwarf WISE J085510.83−071442.5 (WISE 0855).

The pipeline deliberately does **not** download full Level 2 images. A download is accepted only when its SIA2 `access_url` is an HTTPS, on-premises IRSA IBE SPHEREx FITS URL to which the documented `center` and `size` cutout parameters can be added. A run defaults to 10 cutouts and a 300 MB total byte budget; an explicit `--max-files` may raise the count up to the 100-file safety ceiling.

## Install

Python 3.11 or newer is required.

```powershell
cd C:\Projects\SphereX\spherex-proj
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Commands

Discovery is read-only and writes the returned metadata to `data/metadata/`:

```powershell
python -m spherex.pipeline discover --target wise0855
```

Preview exactly which cutouts would be requested without downloading them:

```powershell
python -m spherex.pipeline download --target wise0855 --max-files 10 --max-mb 300 --dry-run
```

Download and validate a bounded sample:

```powershell
python -m spherex.pipeline download --target wise0855 --max-files 10 --max-mb 300
```

Inspect downloaded FITS files and make image diagnostics:

```powershell
python -m spherex.pipeline inspect --target wise0855
```

Generate WCS-aware scientific dashboards and a batch summary from only the
FITS cutouts already present on disk (this command performs no downloads):

```powershell
python -m spherex.pipeline science --target wise0855
```

## Reusable archive client

`SPHERExClient` is a metadata-only interface for coordinate searches that does
not depend on the moving-target pipeline:

```python
from spherex import SPHERExClient

client = SPHERExClient(timeout=60)
observations = client.search(
    ra=133.795,
    dec=-7.245,
    radius=0.05,
    max_results=50,
)
```

The client sends SIA2 `POS=CIRCLE` queries to IRSA's documented `/SIA`
endpoint for the DR1 and QR3 wide/deep collections. IRSA performs spherical
footprint intersection, including circles that cross RA 0/360 degrees. The
VOTable responses are normalized into dictionaries containing observation
times, spatial coverage, detector/band and wavelength metadata, exposure time,
file size, and the FITS `access_url`; the FITS files themselves are never read.
Optional `start_date` and `end_date` values are sent as the SIA2 `TIME`
constraint. See `examples/search_spherex.py` for an executable live query.

Launch the interactive FITS image viewer from the repository root with:

```powershell
python -m frontend.tkinter_viewer
```

The search and FITS processing run on background threads. Check the **Image**
box for up to six observations whose FITS files have already been downloaded,
then choose **Show Selected Images**. The viewer resolves files below
`data/raw`, creates WCS-centred 200 x 200 previews under
`data/processed/previews`, and displays a comparison gallery. Selecting a card
shows its source filename, dimensions, units, target pixel and processing
metadata. The viewer never changes the original FITS measurements.

DR1 and QR3 are both queried and their results are combined. To select one release explicitly, add `--collection spherex_dr1` or `--collection spherex_qr3`.

Useful bounded overrides are `--search-radius-deg` (default `0.05`), `--cutout-size-deg` (default `0.02`, or 1.2 arcmin), `--epoch`, `--max-files`, and `--max-mb`. The default remains 10 files. Explicit requests can raise it to a safety ceiling of 100 files, while `--max-mb` independently limits transferred bytes.

## Target astrometry

`backend/src/spherex/targets.py` defines reusable target records. For WISE 0855 it uses the measured ICRS position `08:55:08.433 -07:14:39.49` at MJD 56986.817 (2014-11-25), reported with 0.05 arcsec precision in Table 1 of [Beamín et al. (2016)](https://doi.org/10.1051/0004-6361/201628662). It uses the Luhman & Esplin astrometric solution quoted in the literature: `pm_ra_cosdec = -8118 ± 8 mas/yr`, `pm_dec = -680 ± 7 mas/yr`, and parallax `449 ± 8 mas`.

The code propagates the coordinate to the initial discovery epoch and independently to each image's observation midpoint. It does not confuse the J2000/ICRS coordinate frame with the measurement epoch. The scientific dashboard includes annual parallax using Astropy's built-in Earth ephemeris and the FITS geocentric spacecraft position when supplied. Radial velocity is set to zero; its perspective effect over this interval is negligible compared with the positional uncertainty and 6.15-arcsec SPHEREx pixels.

## Outputs and safety behavior

- `data/metadata/wise0855_observations.csv`: SIA2 records plus observation-epoch target coordinates, wavelength bounds, release, access URL, and the unmodified row serialized as JSON.
- `data/metadata/wise0855_manifest.csv`: source and cutout URLs, filenames, byte counts, SHA-256 hashes, status, and errors.
- `data/raw/wise0855/*.fits`: validated original cutout bytes, never rewritten.
- `data/metadata/wise0855_fits_inventory.csv`: HDU names, shapes, types, units, finite-pixel fraction, and a conservative semantic classification.
- `outputs/plots/wise0855/*_science_dashboard.png`: target-position, zodiacal-background, variance, flag, wavelength, and conservative forced-aperture diagnostics.
- `outputs/wise0855_observation_summary.csv`: one row per existing detector-observation FITS product, including target footprint, wavelength, quality, and detection status.

Downloads retry transient HTTP failures with exponential backoff, stream to `.part` files, enforce the byte budget while streaming, validate with Astropy before an atomic rename, and skip an existing file only after re-validating it. Unsupported URLs are reported and never fetched as full images.

Official SPHEREx cutouts retain `IMAGE`, `FLAGS`, `VARIANCE`, `ZODI`, `PSF`, and `WCS-WAVE`. The inspector uses names and actual array structure rather than assuming every release or extension has an identical layout. `WCS-WAVE` is useful for visualization; the archive guide recommends the separate `CWAVE` and `CBAND` calibration layers for science analysis.

## Testing

```powershell
pytest -q
```

Tests cover coordinate propagation, archive-row parsing, empty results, the full-image safety guard, duplicate handling, download-budget enforcement, FITS validation, and inspection of a synthetic multi-extension FITS file. Network access is not needed for unit tests.

## Scientific limitations

An image footprint containing the propagated coordinate is not evidence that WISE 0855 was detected. This phase only produces a reproducible image-cutout dataset. It does not infer a source spectrum from image pixels.

IRSA now provides a [SPHEREx Spectrophotometry Tool](https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/sp.html) that performs Tractor-based forced photometry and returns wavelength, flux, uncertainty, bandpass width, flags, and fit diagnostics. The reviewed public documentation describes the supported interactive Data Explorer/job workflow, but does not document a stable programmatic spectrophotometry submission API. For that reason this pipeline does not automate or reverse-engineer the service; using its documented UI with accurately propagated source positions is a separate follow-up phase.

## Official references

- [SPHEREx mission and releases at IRSA](https://irsa.ipac.caltech.edu/Missions/spherex.html)
- [SPHEREx data access and SIA2/cutout examples](https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-access/)
- [SPHEREx data products and release differences](https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products/)
- [IRSA program interfaces](https://irsa.ipac.caltech.edu/voapi.html)
- [SPHEREx Data Explorer FAQ](https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/faq.html)

As of November 2026, the archive documentation recommends DR1 and QR3 for new work. The pipeline searches the `spherex_dr1` and `spherex_qr3` wide-survey collections by default; the reusable client also supports their separate `_deep` collections. QR2 is no longer searchable through SIA2.
