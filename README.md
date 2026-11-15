# SPHEREx Observation Portal & Science Toolkit
## NASA Space Apps Challenge 2026 Submission

A dual-domain scientific platform designed for bounded discovery, FITS pixel ingestion, multi-epoch temporal astrometry, and 102-band hyperspectral slicing of **NASA/IPAC SPHEREx** observations.

---

### Project Architecture

- **Web Application (`frontend/web-app/`)**: Production React + TypeScript + Vite portal with HTML5 Canvas GPU rendering, on-the-fly contrast stretching, and click-to-investigate candidate astrometry. Ready for Vercel deployment.
- **Python Scientific Backend (`backend/src/spherex/`)**:
  - `spherex_client.py`: VO SIA2 cone query client connecting directly to NASA/IPAC IRSA.
  - `downloader.py`: Caltech IBE bounded FITS cutout downloader.
  - `observation_selector.py`: Astrometric multi-epoch and multi-detector selection service.
  - `image_cleaner.py`: SPHEREx bitmask `FLAGS` and `VARIANCE` HDU quality cleaner.
  - `image_renderer.py`: Astronomical contrast normalizer (`asinh`, `linear`, `log`).
  - `blink.py`: WCS reprojection onto a shared celestial sky grid (`reproject_interp`).
  - `candidate.py`: Flux-weighted 2D centroid fitting, proper motion velocity ($\mu$ in "/yr), and statistical significance ($\sigma$).
  - `api_server.py`: Multithreaded REST HTTP server bridging the scientific modules to the web app.
- **Desktop Tkinter Viewer (`frontend/tkinter_viewer/`)**: Standalone desktop Tkinter GUI for researchers working in local Python environments.

---

### Quick Start (Local Development)

#### 1. Start Python Scientific Backend
```bash
# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\activate
pip install -e .

# Run the API server
python backend/src/spherex/api_server.py
```
*API server will listen on `http://127.0.0.1:8000`.*

#### 2. Start the Web App
```bash
cd frontend/web-app
npm install
npm run dev
```
*Open `http://localhost:5173` in any modern web browser.*

#### 3. Run the Desktop Tkinter Viewer (Optional)
```bash
python -m frontend.tkinter_viewer
```

---

### Running Automated Verification Tests
```bash
pytest
```
*All 80 scientific unit tests cover client queries, WCS reprojection, image cleaning, and candidate astrometry.*

---

### Deployment Guide

#### Frontend Deployment on Vercel
1. In your terminal, navigate to the web app:
   ```bash
   cd frontend/web-app
   vercel
   ```
2. Follow the interactive prompts to link and deploy your project.
3. In your Vercel Project Settings, add the environment variable:
   - `VITE_API_BASE_URL`: `https://your-backend-api-url.com` (or leave unset for local default).

#### Backend Deployment
Deploy the Python backend (`backend/src/spherex/api_server.py` or wrapped in Gunicorn) to **Render**, **Railway**, or **Fly.io** with standard Python 3.11+.
