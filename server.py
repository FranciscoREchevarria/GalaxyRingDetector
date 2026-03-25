import asyncio
import io
import json
import logging
import os
import time
import uuid
from urllib.parse import urlparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import uvicorn
import numpy as np
import nest_asyncio
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ring_detection_model import RingDetectionZoobot
from downloader import download_legacy_fits, download_legacy_fits_async

from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

from torchvision import transforms
from galaxy_transforms import LuptonRgbTransform, ScaleToUnitIntervalTransform

from astropy.io import fits
from PIL import Image
from scipy.ndimage import zoom
from cachetools import TTLCache

from visualizations import Transformations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# Paths
path = Path(os.getcwd()).resolve()
images_path = path / 'images'
model_path = path / 'model'
static_path = path / 'static'
templates_path = path / 'templates'

os.makedirs(images_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
os.makedirs(static_path, exist_ok=True)
os.makedirs(templates_path, exist_ok=True)

IMAGES_DIR = images_path
MODEL_DIR = model_path
PREDICTIONS_DIR = path / "predictions"
PNG_CACHE_DIR = IMAGES_DIR / "png_cache"
PNG_CACHE_MAX_BYTES = 500 * 1024 * 1024  # 500 MB

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(PNG_CACHE_DIR, exist_ok=True)


def _read_prediction_cache(obj_id: str) -> dict | None:
    """Read cached prediction for obj_id. Returns dict with inner_ring_prob, outer_ring_prob, etc. or None."""
    cache_path = PREDICTIONS_DIR / f"{obj_id}.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_prediction_cache(
    obj_id: str,
    inner_ring_prob: float,
    outer_ring_prob: float,
    inner_ring_detected: bool,
    outer_ring_detected: bool,
) -> None:
    """Write prediction result to cache."""
    cache_path = PREDICTIONS_DIR / f"{obj_id}.json"
    try:
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "inner_ring_prob": inner_ring_prob,
                    "outer_ring_prob": outer_ring_prob,
                    "inner_ring_detected": inner_ring_detected,
                    "outer_ring_detected": outer_ring_detected,
                },
                f,
            )
    except OSError as e:
        logger.warning("Failed to write prediction cache for %s: %s", obj_id, e)

INNER_THRESHOLD = 0.580
OUTER_THRESHOLD = 0.480

BEST_CHECKPOINT = MODEL_DIR / "version_37" / "checkpoints" / "stage2-best-epoch=20.ckpt"

# Fast prediction: when True, skip TTA (single forward pass instead of 8). Set via env FAST_PREDICTION=true or request param fast=true
def _is_fast_prediction_env() -> bool:
    return os.environ.get("FAST_PREDICTION", "").lower() in ("true", "1", "yes")


def _use_tta(fast_param: bool | str | None = False) -> bool:
    """Return True if TTA should be used. Env FAST_PREDICTION or param fast=true disables TTA."""
    fast = fast_param is True or (
        isinstance(fast_param, str) and fast_param.lower() in ("true", "1", "yes")
    )
    return not (_is_fast_prediction_env() or fast)


def fits_to_rgb_png(fits_path, stretch=0.5, Q=6):
    """
    Convert FITS file to RGB PNG bytes.
    Opens FITS, applies Transformations.channels_to_rgb, converts to PNG via PIL.
    Returns io.BytesIO containing PNG bytes.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    data = np.asarray(data, dtype=np.float32)
    rgb = Transformations.channels_to_rgb(data, stretch=stretch, Q=Q)
    # rgb is HWC uint8 from make_lupton_rgb
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def _png_cache_path(fits_path: Path, stretch: float, q: int) -> Path:
    """Return cache file path for a FITS + Lupton params. Safe for filesystem."""
    safe_stem = fits_path.stem.replace(".", "p")
    return PNG_CACHE_DIR / f"{safe_stem}_s{stretch}_q{q}.png"


def _evict_png_cache_if_needed():
    """
    If PNG cache exceeds PNG_CACHE_MAX_BYTES, evict oldest files (by mtime) until under limit.
    Called before writing a new cache file.
    """
    png_files = list(PNG_CACHE_DIR.glob("*.png"))
    if not png_files:
        return

    total_bytes = sum(f.stat().st_size for f in png_files)
    if total_bytes <= PNG_CACHE_MAX_BYTES:
        return

    # Sort by mtime ascending (oldest first) for LRU eviction
    sorted_files = sorted(png_files, key=lambda f: f.stat().st_mtime)
    for f in sorted_files:
        if total_bytes <= PNG_CACHE_MAX_BYTES:
            break
        try:
            size = f.stat().st_size
            f.unlink()
            total_bytes -= size
        except OSError:
            pass


def get_or_create_rgb_png_cache(fits_path: Path, stretch: float, q: int) -> Path:
    """
    Return path to cached PNG. Generate and save if not cached.
    Used by galaxy_image and galaxy_image_fits endpoints.
    Evicts oldest files (LRU by mtime) when cache exceeds PNG_CACHE_MAX_BYTES.
    """
    cache_path = _png_cache_path(fits_path, stretch, q)
    if cache_path.exists():
        return cache_path
    _evict_png_cache_if_needed()
    buf = fits_to_rgb_png(fits_path, stretch=stretch, Q=q)
    cache_path.write_bytes(buf.read())
    return cache_path


def _load_model_and_pipeline():
    """Load model and baseline pipeline once. Used at startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model = FinetuneableZoobotClassifier(
        name='hf_hub:mwalmsley/zoobot-encoder-convnext_tiny',
        num_classes=2,
        label_col='ring_class'
    )
    encoder = temp_model.encoder
    temp_model = None

    model = RingDetectionZoobot.load_from_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        encoder=encoder,
        encoder_dim=768,
        hidden_dim=256,
        dropout_rate=0.4,
    )
    model = model.to(device)
    model.eval()

    baseline_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        LuptonRgbTransform(stretch=0.5, Q=10),
        ScaleToUnitIntervalTransform(),
    ])

    return model, device, baseline_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, store in app.state."""
    model, device, pipeline = _load_model_and_pipeline()
    app.state.model = model
    app.state.device = device
    app.state.baseline_pipeline = pipeline
    # Capped caches with TTL to prevent unbounded memory growth
    app.state.results = TTLCache(maxsize=10_000, ttl=3600)  # objId -> prediction result, 1h TTL
    app.state.uploaded_fits = TTLCache(maxsize=100, ttl=1800)  # fits_id -> Path, 30min TTL
    yield
    # Cleanup if needed
    app.state.results.clear()
    app.state.uploaded_fits.clear()


app = FastAPI(title='Deploying an ML Model with FastAPI', lifespan=lifespan)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))


# ----- Page routes -----

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render index.html with empty grid."""
    return templates.TemplateResponse("index.html", {"request": request, "results": []})


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Render predict.html with URL form."""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/api/debug_legacy_survey")
async def debug_legacy_survey():
    """Test if Legacy Survey is reachable from this environment."""
    import urllib.request
    url = "https://www.legacysurvey.org/viewer/cutout.fits?ra=168.532451&dec=47.80430912&layer=ls-dr9&pixscale=0.262&size=64&bands=grz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GalaxyRingDetector-test"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status": "ok", "content_length": resp.headers.get("Content-Length")}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/view/{obj_id}", response_class=HTMLResponse)
async def view_page(
    request: Request, obj_id: str, ra: str | None = None, dec: str | None = None,
    fast: bool = False,
):
    """Render view.html. ra and dec from query params, or from stored batch results.
    fast=true: skip TTA for faster prediction (single pass instead of 8)."""
    if ra is None or dec is None:
        stored = getattr(request.app.state, "results", {}).get(obj_id)
        if stored:
            if ra is None:
                ra = stored.get("ra")
            if dec is None:
                dec = stored.get("dec")
        if not ra or not dec:
            raise HTTPException(
                status_code=400,
                detail="ra and dec are required. Open this page from the grid View link.",
            )

    # Use stored prediction if available, then prediction cache, otherwise run prediction
    stored = getattr(request.app.state, "results", {}).get(obj_id)
    if stored and "inner_ring_prob" in stored:
        inner_ring_prob = stored["inner_ring_prob"]
        outer_ring_prob = stored["outer_ring_prob"]
        inner_ring_detected = stored["inner_ring_detected"]
        outer_ring_detected = stored["outer_ring_detected"]
    else:
        cached = _read_prediction_cache(obj_id)
        if cached:
            inner_ring_prob = cached["inner_ring_prob"]
            outer_ring_prob = cached["outer_ring_prob"]
            inner_ring_detected = cached["inner_ring_detected"]
            outer_ring_detected = cached["outer_ring_detected"]
        else:
            # Prefer obj_id.fits when obj_id available (batch/CSV mode)
            obj_path = IMAGES_DIR / f"{obj_id}.fits"
            ra_dec_path = IMAGES_DIR / f"ra{ra}_dec{dec}.fits"
            image_save_path = obj_path if obj_path.exists() else ra_dec_path
            if not image_save_path.exists():
                try:
                    image_save_path = await asyncio.to_thread(
                        download_legacy_fits, ra, dec, obj_path
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to download FITS: {e}")
            try:
                use_tta = _use_tta(fast)
                inner_ring_prob, outer_ring_prob = await asyncio.to_thread(
                    _run_prediction, request.app, image_save_path, use_tta
                )
                inner_ring_detected = inner_ring_prob >= INNER_THRESHOLD
                outer_ring_detected = outer_ring_prob >= OUTER_THRESHOLD
            except Exception as e:
                inner_ring_prob = 0.0
                outer_ring_prob = 0.0
                inner_ring_detected = False
                outer_ring_detected = False

    return templates.TemplateResponse(
        "view.html",
        {
            "request": request,
            "obj_id": obj_id,
            "ra": ra,
            "dec": dec,
            "inner_ring_prob": inner_ring_prob,
            "outer_ring_prob": outer_ring_prob,
            "inner_ring_detected": inner_ring_detected,
            "outer_ring_detected": outer_ring_detected,
        },
    )


# ----- API endpoints -----

def _run_prediction(app: FastAPI, fits_image_path: Path, use_tta: bool = True):
    """Run model prediction on a FITS file. Returns [inner_prob, outer_prob].
    use_tta: if True, use test-time augmentation (8 forward passes); if False, single pass (faster, less accurate)."""
    t0 = time.perf_counter()
    with fits.open(fits_image_path) as hdul:
        fits_image = hdul[0].data
    data_array = np.asarray(fits_image, dtype=np.float32)
    image_tensor = torch.from_numpy(data_array)
    t_fits = time.perf_counter() - t0

    device = app.state.device
    pipeline = app.state.baseline_pipeline
    t1 = time.perf_counter()
    image_tensor = pipeline(image_tensor).to(device)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    t_pipeline = time.perf_counter() - t1

    model = app.state.model
    t2 = time.perf_counter()
    with torch.inference_mode():
        if use_tta:
            prediction = model.predict_proba_tta(image_tensor)
        else:
            prediction = model.predict_proba(image_tensor)
    t_inference = time.perf_counter() - t2

    logger.info(
        "prediction_timing fits_load=%.3fs pipeline=%.3fs inference=%.3fs total=%.3fs tta=%s file=%s",
        t_fits, t_pipeline, t_inference, t_fits + t_pipeline + t_inference, use_tta, fits_image_path.name,
    )

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().tolist()
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.squeeze().tolist()

    if isinstance(prediction, list) and len(prediction) == 2:
        return float(prediction[0]), float(prediction[1])
    return 0.0, 0.0


def _format_sse(event: str, data: str) -> str:
    """Format a Server-Sent Event."""
    payload = f"event: {event}\n"
    for line in data.split("\n"):
        payload += f"data: {line}\n"
    payload += "\n"
    return payload


async def _stream_retry_row(app: FastAPI, obj_id: str, ra: str, dec: str, fast: bool):
    """Stream retry_row as SSE: download_progress, then row or error."""
    link = f"https://www.legacysurvey.org/viewer?ra={ra}&dec={dec}&layer=ls-dr9&zoom=14"
    filename = f"{obj_id}.fits"
    image_save_path = IMAGES_DIR / filename
    row_template = templates.env.get_template("partials/galaxy_row.html")

    try:
        image_save_path.unlink(missing_ok=True)
        image_path = None
        last_progress = {"downloaded": 0, "total": None}
        async for progress in download_legacy_fits_async(ra, dec, image_save_path):
            if "path" in progress:
                image_path = Path(progress["path"])
                if last_progress.get("total"):
                    yield _format_sse("download_progress", json.dumps({
                        "downloaded": last_progress["total"],
                        "total": last_progress["total"],
                    }))
                break
            last_progress = progress
            yield _format_sse("download_progress", json.dumps(progress))

        if not image_path:
            image_path = image_save_path

        use_tta = _use_tta(fast)
        inner_prob, outer_prob = await asyncio.to_thread(
            _run_prediction, app, image_path, use_tta
        )
        inner_detected = inner_prob >= INNER_THRESHOLD
        outer_detected = outer_prob >= OUTER_THRESHOLD
        result = {
            "objId": obj_id,
            "ra": ra,
            "dec": dec,
            "inner_ring_prob": inner_prob,
            "outer_ring_prob": outer_prob,
            "inner_ring_detected": inner_detected,
            "outer_ring_detected": outer_detected,
            "link": link,
            "fast": fast,
        }
        app.state.results[obj_id] = result
        _write_prediction_cache(obj_id, inner_prob, outer_prob, inner_detected, outer_detected)
        html = row_template.render(row=result)
        yield _format_sse("row", json.dumps({"success": True, "html": html}))
    except Exception as e:
        result = {
            "objId": obj_id,
            "ra": ra,
            "dec": dec,
            "inner_ring_prob": None,
            "outer_ring_prob": None,
            "inner_ring_detected": None,
            "outer_ring_detected": None,
            "link": link,
            "error": str(e),
            "fast": fast,
        }
        html = row_template.render(row=result)
        yield _format_sse("row", json.dumps({"success": False, "html": html, "error": str(e)}))


@app.post("/api/load_csv")
async def load_csv(
    request: Request, file: UploadFile = File(...),
    fast: str = Form("false"),
):
    """Load CSV without downloading or predicting. Returns rows as HTML.
    For rows where FITS exists: use prediction cache or run prediction once and cache."""
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV file required")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), dtype={'objID': str})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or malformed CSV: {e}")

    required = ['objID', 'ra', 'dec']
    for col in required:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required}")

    use_tta = _use_tta(fast)
    row_template = templates.env.get_template("partials/galaxy_row.html")
    rows_html = []

    for _, r in df.iterrows():
        obj_id = str(r['objID']).strip()
        ra = str(r['ra'])
        dec = str(r['dec'])
        link = f"https://www.legacysurvey.org/viewer?ra={ra}&dec={dec}&layer=ls-dr9&zoom=14"

        fits_path = IMAGES_DIR / f"{obj_id}.fits"
        if fits_path.exists():
            cached = _read_prediction_cache(obj_id)
            if cached:
                row = {
                    "objId": obj_id,
                    "ra": ra,
                    "dec": dec,
                    "inner_ring_prob": cached["inner_ring_prob"],
                    "outer_ring_prob": cached["outer_ring_prob"],
                    "inner_ring_detected": cached["inner_ring_detected"],
                    "outer_ring_detected": cached["outer_ring_detected"],
                    "link": link,
                    "fast": not use_tta,
                }
                request.app.state.results[obj_id] = row
            else:
                try:
                    inner_prob, outer_prob = await asyncio.to_thread(
                        _run_prediction, request.app, fits_path, use_tta
                    )
                    inner_detected = inner_prob >= INNER_THRESHOLD
                    outer_detected = outer_prob >= OUTER_THRESHOLD
                    row = {
                        "objId": obj_id,
                        "ra": ra,
                        "dec": dec,
                        "inner_ring_prob": inner_prob,
                        "outer_ring_prob": outer_prob,
                        "inner_ring_detected": inner_detected,
                        "outer_ring_detected": outer_detected,
                        "link": link,
                        "fast": not use_tta,
                    }
                    request.app.state.results[obj_id] = row
                    _write_prediction_cache(obj_id, inner_prob, outer_prob, inner_detected, outer_detected)
                except Exception as e:
                    row = {
                        "objId": obj_id,
                        "ra": ra,
                        "dec": dec,
                        "inner_ring_prob": None,
                        "outer_ring_prob": None,
                        "inner_ring_detected": None,
                        "outer_ring_detected": None,
                        "link": link,
                        "error": str(e),
                        "fast": not use_tta,
                    }
        else:
            row = {
                "objId": obj_id,
                "ra": ra,
                "dec": dec,
                "inner_ring_prob": None,
                "outer_ring_prob": None,
                "inner_ring_detected": None,
                "outer_ring_detected": None,
                "link": link,
                "fast": not use_tta,
            }

        rows_html.append(row_template.render(row=row))

    return {"rows_html": "".join(rows_html)}


@app.post("/api/retry_row_stream")
async def retry_row_stream(request: Request):
    """Stream retry_row as SSE with download progress. Returns download_progress events, then row."""
    body = await request.json()
    obj_id = str(body.get("obj_id", ""))
    ra = str(body.get("ra", ""))
    dec = str(body.get("dec", ""))
    fast = body.get("fast", False)
    if not obj_id or not ra or not dec:
        raise HTTPException(status_code=400, detail="obj_id, ra, and dec are required")
    return StreamingResponse(
        _stream_retry_row(request.app, obj_id, ra, dec, fast),
        media_type="text/event-stream",
    )


@app.post("/api/retry_row")
async def retry_row(request: Request):
    """Retry download and prediction for a single row. Returns row HTML on success.
    Body may include fast: true to skip TTA for faster prediction."""
    body = await request.json()
    obj_id = str(body.get("obj_id", ""))
    ra = str(body.get("ra", ""))
    dec = str(body.get("dec", ""))
    fast = body.get("fast", False)
    if not obj_id or not ra or not dec:
        raise HTTPException(status_code=400, detail="obj_id, ra, and dec are required")
    link = f"https://www.legacysurvey.org/viewer?ra={ra}&dec={dec}&layer=ls-dr9&zoom=14"
    filename = f"{obj_id}.fits"
    image_save_path = IMAGES_DIR / filename
    row_template = templates.env.get_template("partials/galaxy_row.html")

    try:
        image_save_path.unlink(missing_ok=True)
        image_save_path = await asyncio.to_thread(download_legacy_fits, ra, dec, image_save_path)
        use_tta = _use_tta(fast)
        inner_prob, outer_prob = await asyncio.to_thread(
            _run_prediction, request.app, image_save_path, use_tta
        )
    except Exception as e:
        result = {
            "objId": obj_id,
            "ra": ra,
            "dec": dec,
            "inner_ring_prob": None,
            "outer_ring_prob": None,
            "inner_ring_detected": None,
            "outer_ring_detected": None,
            "link": link,
            "error": str(e),
            "fast": fast,
        }
        html = row_template.render(row=result)
        return {"success": False, "html": html, "error": str(e)}

    inner_detected = inner_prob >= INNER_THRESHOLD
    outer_detected = outer_prob >= OUTER_THRESHOLD
    result = {
        "objId": obj_id,
        "ra": ra,
        "dec": dec,
        "inner_ring_prob": inner_prob,
        "outer_ring_prob": outer_prob,
        "inner_ring_detected": inner_detected,
        "outer_ring_detected": outer_detected,
        "link": link,
        "fast": fast,
    }
    request.app.state.results[obj_id] = result
    _write_prediction_cache(obj_id, inner_prob, outer_prob, inner_detected, outer_detected)
    html = row_template.render(row=result)
    return {"success": True, "html": html}


@app.get("/api/galaxy_image")
async def galaxy_image(
    ra: str, dec: str, obj_id: str | None = None,
    stretch: float = 0.5, q: float = 6,
):
    """Return PNG image from FITS. Prefers local files (obj_id.fits or ra_dec.fits), downloads from Legacy Survey only when not found.
    Lupton RGB params: stretch (0.1-2), q (1-20)."""
    image_save_path = None
    if obj_id:
        obj_path = IMAGES_DIR / f"{obj_id}.fits"
        if obj_path.exists():
            image_save_path = obj_path
    if image_save_path is None:
        ra_dec_path = IMAGES_DIR / f"ra{ra}_dec{dec}.fits"
        if ra_dec_path.exists():
            image_save_path = ra_dec_path
    if image_save_path is None:
        try:
            image_save_path = await asyncio.to_thread(
                download_legacy_fits, ra, dec, IMAGES_DIR / f"ra{ra}_dec{dec}.fits"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download FITS: {e}")

    stretch = max(0.1, min(2.0, stretch))
    q = max(1, min(20, int(q)))
    cache_path = await asyncio.to_thread(
        get_or_create_rgb_png_cache, image_save_path, stretch=stretch, q=q
    )
    return FileResponse(cache_path, media_type="image/png")


def _make_3d_pipeline(stretch: float, q: float):
    """Build pipeline with LuptonRgbTransform using given stretch and Q."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        LuptonRgbTransform(stretch=stretch, Q=int(q)),
        ScaleToUnitIntervalTransform(),
    ])


def _compute_3d_data(path: Path, stretch: float, q: float) -> dict:
    """
    Load FITS, apply pipeline (Lupton RGB + scale), downsample to 64x64.
    Returns dict for Plotly surface: {channels, x, y}.
    CPU-bound - run via asyncio.to_thread().
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
    data = np.asarray(data, dtype=np.float32)

    stretch = max(0.1, min(2.0, stretch))
    q = max(1, min(20, int(q)))
    pipeline = _make_3d_pipeline(stretch, q)
    tensor = torch.from_numpy(data)
    transformed = pipeline(tensor)
    if isinstance(transformed, torch.Tensor):
        arr = transformed.detach().cpu().numpy()
    else:
        arr = np.asarray(transformed)

    target_size = 64
    h, w = arr.shape[1], arr.shape[2]
    if h == 0 or w == 0:
        raise ValueError("Invalid FITS image: zero height or width")
    if h != target_size or w != target_size:
        factors = (1, target_size / h, target_size / w)
        arr = zoom(arr, factors, order=1)

    channel_names = ['g', 'r', 'z']
    channels = [
        {"name": name, "z_data": arr[i].tolist()}
        for i, name in enumerate(channel_names)
    ]
    x = list(range(arr.shape[2]))
    y = list(range(arr.shape[1]))

    return {"channels": channels, "x": x, "y": y}


@app.get("/api/galaxy_3d_data")
async def galaxy_3d_data(
    request: Request, ra: str, dec: str,
    stretch: float = 0.5, q: float = 6,
):
    """Load FITS, apply pipeline (Lupton RGB + scale), downsample to 64x64, return JSON for Plotly surface.
    Lupton params: stretch (0.1-2), q (1-20)."""
    filename = f"ra{ra}_dec{dec}.fits"
    image_save_path = IMAGES_DIR / filename
    if not image_save_path.exists():
        try:
            image_save_path = await asyncio.to_thread(
                download_legacy_fits, ra, dec, image_save_path
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download FITS: {e}")

    try:
        return await asyncio.to_thread(
            _compute_3d_data, image_save_path, stretch, q
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _parse_ra_dec_from_url(url: str) -> tuple[str | None, str | None]:
    """Extract ra and dec from Legacy Survey viewer/cutout URLs."""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host not in ("www.legacysurvey.org", "legacysurvey.org"):
            return None, None
        path = (parsed.path or "").lower()
        if "viewer" not in path and "cutout" not in path:
            return None, None
    except Exception:
        return None, None
    if "?" not in url:
        return None, None
    url_params = url.split("?")[1].split("&")
    params_dict = {}
    for p in url_params:
        if "=" in p:
            k, v = p.split("=", 1)
            params_dict[k] = v
    return params_dict.get("ra"), params_dict.get("dec")


@app.post("/api/predict_from_url", response_class=HTMLResponse)
async def predict_from_url(
    request: Request, legacy_survey_url: str = Form(...),
    fast: str = Form("false"),
):
    """Predict from Legacy Survey URL. Returns HTML partial (predict_result.html) for HTMX swap.
    fast=true: skip TTA for faster prediction (single pass instead of 8)."""
    ra, dec = _parse_ra_dec_from_url(legacy_survey_url)
    if not ra or not dec:
        raise HTTPException(
            status_code=415,
            detail="Unsupported URL. Provide a Legacy Survey viewer or cutout URL with ra and dec parameters.",
        )

    filename = f"ra{ra}_dec{dec}.fits"
    image_save_path = IMAGES_DIR / filename
    if not image_save_path.exists():
        try:
            image_save_path = await asyncio.to_thread(download_legacy_fits, ra, dec, image_save_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download FITS: {e}")

    use_tta = _use_tta(fast)
    inner_prob, outer_prob = await asyncio.to_thread(
        _run_prediction, request.app, image_save_path, use_tta
    )

    result = {
        "inner_ring_prob": inner_prob,
        "outer_ring_prob": outer_prob,
        "inner_ring_detected": inner_prob >= INNER_THRESHOLD,
        "outer_ring_detected": outer_prob >= OUTER_THRESHOLD,
        "ra": ra,
        "dec": dec,
        "fits_id": None,
    }

    return templates.TemplateResponse(
        "partials/predict_result.html",
        {"request": request, **result}
    )


@app.post("/api/predict_from_fits", response_class=HTMLResponse)
async def predict_from_fits(
    request: Request, fits_file: UploadFile = File(...),
    fast: str = Form("false"),
):
    """Predict from uploaded FITS file. Returns HTML partial (predict_result.html) for HTMX swap.
    fast=true: skip TTA for faster prediction (single pass instead of 8)."""
    if not fits_file.filename or not fits_file.filename.lower().endswith(('.fits', '.fit')):
        raise HTTPException(
            status_code=415,
            detail="FITS file required (.fits or .fit extension).",
        )

    fits_id = str(uuid.uuid4())
    contents = await fits_file.read()
    image_save_path = IMAGES_DIR / f"upload_{fits_id}.fits"
    try:
        image_save_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save FITS file: {e}")

    request.app.state.uploaded_fits[fits_id] = image_save_path

    try:
        use_tta = _use_tta(fast)
        inner_prob, outer_prob = await asyncio.to_thread(
            _run_prediction, request.app, image_save_path, use_tta
        )
    except Exception as e:
        if fits_id in request.app.state.uploaded_fits:
            del request.app.state.uploaded_fits[fits_id]
        try:
            image_save_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    result = {
        "inner_ring_prob": inner_prob,
        "outer_ring_prob": outer_prob,
        "inner_ring_detected": inner_prob >= INNER_THRESHOLD,
        "outer_ring_detected": outer_prob >= OUTER_THRESHOLD,
        "ra": None,
        "dec": None,
        "fits_id": fits_id,
    }

    return templates.TemplateResponse(
        "partials/predict_result.html",
        {"request": request, **result}
    )


async def _get_fits_path_with_retry(request: Request, fits_id: str, max_retries: int = 3, retry_delay: float = 0.3) -> Path | None:
    """Get FITS path from uploaded_fits cache, retrying silently if not found (handles race with upload)."""
    cache = getattr(request.app.state, "uploaded_fits", None) or {}
    for attempt in range(max_retries):
        path = cache.get(fits_id)
        if path and path.exists():
            return path
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
    return None


@app.get("/api/galaxy_image_fits")
async def galaxy_image_fits(
    request: Request, fits_id: str,
    stretch: float = 0.5, q: float = 6,
):
    """Return PNG image from uploaded FITS file by fits_id.
    Lupton RGB params: stretch (0.1-2), q (1-20)."""
    path = await _get_fits_path_with_retry(request, fits_id)
    if not path:
        raise HTTPException(status_code=404, detail="FITS file not found or expired")
    stretch = max(0.1, min(2.0, stretch))
    q = max(1, min(20, int(q)))
    cache_path = await asyncio.to_thread(get_or_create_rgb_png_cache, path, stretch=stretch, q=q)
    return FileResponse(cache_path, media_type="image/png")


@app.get("/api/galaxy_3d_data_fits")
async def galaxy_3d_data_fits(
    request: Request, fits_id: str,
    stretch: float = 0.5, q: float = 6,
):
    """Load uploaded FITS, apply pipeline (Lupton RGB + scale), downsample to 64x64, return JSON for Plotly surface.
    Lupton params: stretch (0.1-2), q (1-20)."""
    path = await _get_fits_path_with_retry(request, fits_id)
    if not path:
        raise HTTPException(status_code=404, detail="FITS file not found or expired")

    try:
        return await asyncio.to_thread(_compute_3d_data, path, stretch, q)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    nest_asyncio.apply()
    host = "127.0.0.1"
    uvicorn.run(app, host=host, port=8000)
