# How to Run SphereX-Ares

SphereX-Ares searches NASA/IPAC's IRSA archive for SPHEREx observations and
provides a desktop viewer for browsing the returned metadata. It does not need
any bundled FITS data to start.

## Requirements

- Python 3.11 or newer
- An internet connection for live NASA IRSA searches
- Git, if cloning the repository
- Tkinter for the desktop viewer (included with most Windows and macOS Python
  installations)

## 1. Download the project

Clone it with Git:

```bash
git clone https://github.com/chitra-ramalingam/SphereX-Ares.git
cd SphereX-Ares
```

Alternatively, download the repository ZIP from GitHub, extract it, and open a
terminal in the extracted `SphereX-Ares` directory.

## 2. Create a virtual environment

Using a virtual environment keeps this project's packages separate from other
Python projects.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, the virtual environment can still be used
without activating it by replacing `python` in later commands with:

```powershell
.\.venv\Scripts\python.exe
```

### Windows Command Prompt

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Some Linux distributions require the virtual-environment and Tkinter packages:

```bash
# Debian or Ubuntu
sudo apt install python3-venv python3-tk
```

Use the equivalent packages for your Linux distribution when necessary.

## 3. Install the requirements and SphereX-Ares

The dependency file is named `requirements.txt` (not `requirements.bat`).
Upgrade the package installer, install the listed requirements, and then
register the local `spherex` package:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The first installation command installs packages such as Astropy, Astroquery,
PyVO, Requests, and Pytest. The editable installation makes the `spherex`
package importable while keeping its source files in the cloned repository.

## 4. Launch the desktop viewer

From the repository root, run:

```bash
python -m spherex.viewer
```

In the viewer:

1. Enter right ascension and declination in decimal degrees.
2. Enter a search radius in degrees.
3. Optionally enter ISO dates such as `2025-07-01` and `2026-01-31`.
4. Select **Search IRSA**.
5. Select an observation row to inspect all returned metadata.
6. Use **Open selected FITS URL** only when you intentionally want to open the
   archive file URL.

The default example position is:

- RA: `133.795`
- Dec: `-7.245`
- Radius: `0.05` degrees

## 5. Run a metadata search from Python

```python
from spherex import SPHERExClient

client = SPHERExClient(timeout=60)
observations = client.search(
    ra=133.795,
    dec=-7.245,
    radius=0.05,
    max_results=50,
)

for observation in observations:
    print(
        observation["observation_timestamp"],
        observation["observation_id"],
        observation["fits_url"],
    )
```

The search returns metadata dictionaries. It does not download FITS contents.

An executable version of this example is also included:

```bash
python examples/search_spherex.py
```

## 6. Run the tests

The unit tests use mocked archive responses and do not require IRSA to be
online:

```bash
python -m pytest -q
```

To run only the standalone archive client and viewer tests:

```bash
python -m pytest tests/test_spherex_client.py -q
```

## Troubleshooting

### `No module named spherex`

Make sure the command is being run from the repository root and install the
project again:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### Viewer reports HTTP 503

HTTP 503 means NASA/IPAC IRSA is temporarily unavailable or busy. It is not a
coordinate-validation error. Wait a few minutes and search again. The mocked
unit tests can still run while IRSA is unavailable.

### `No module named tkinter`

Windows and macOS users should install a standard Python build from
[python.org](https://www.python.org/downloads/). Linux users should install
their distribution's Tkinter package, commonly `python3-tk`.

### PowerShell script execution is disabled

Activation is optional. Run the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m spherex.viewer
```

### The viewer opens but searches return no observations

- Confirm RA is in the range `0 <= RA < 360`.
- Confirm declination is in the range `-90 <= Dec <= 90`.
- Try a slightly larger radius.
- Remove date filters to check all available observation dates.
- Remember that newly ingested SPHEREx data can appear in SIA later than other
  IRSA interfaces.

## Data and generated files

The repository intentionally excludes virtual environments, downloaded data,
FITS files, generated plots, caches, and local run outputs. Each user can
generate those artifacts locally without adding them to source control.
