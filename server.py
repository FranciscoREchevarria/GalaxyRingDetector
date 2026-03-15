import asyncio
import io
import json
import os
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
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ring_detection_model import RingDetectionZoobot
from downloader import download_legacy_fits

from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

from torchvision import transforms
from galaxy_transforms import LuptonRgbTransform, ScaleToUnitIntervalTransform

from astropy.io import fits
from PIL import Image
from scipy.ndimage import zoom

from visualizations import Transformations

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

INNER_THRESHOLD = 0.580
OUTER_THRESHOLD = 0.480

BEST_CHECKPOINT = MODEL_DIR / "version_37" / "checkpoints" / "stage2-best-epoch=20.ckpt"


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
    app.state.results = {}  # objId -> prediction result for /view/{objId}
    app.state.uploaded_fits = {}  # fits_id -> Path for uploaded FITS files
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


@app.get("/view/{obj_id}", response_class=HTMLResponse)
async def view_page(request: Request, obj_id: str, ra: str | None = None, dec: str | None = None):
    """Render view.html. ra and dec from query params, or from stored batch results."""
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
    return templates.TemplateResponse(
        "view.html",
        {"request": request, "obj_id": obj_id, "ra": ra, "dec": dec}
    )


# ----- API endpoints -----

def _run_prediction(app: FastAPI, fits_image_path: Path):
    """Run model prediction on a FITS file. Returns [inner_prob, outer_prob]."""
    with fits.open(fits_image_path) as hdul:
        fits_image = hdul[0].data

    data_array = np.asarray(fits_image, dtype=np.float32)
    image_tensor = torch.from_numpy(data_array)
    device = app.state.device
    pipeline = app.state.baseline_pipeline

    image_tensor = pipeline(image_tensor).to(device)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    model = app.state.model
    with torch.no_grad():
        prediction = model.predict_proba_tta(image_tensor)

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().tolist()
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.squeeze().tolist()

    if isinstance(prediction, list) and len(prediction) == 2:
        return float(prediction[0]), float(prediction[1])
    return 0.0, 0.0


def _format_sse(event: str, data: str) -> str:
    """Format a Server-Sent Event. For multi-line data, each line is sent as data: line."""
    payload = f"event: {event}\n"
    for line in data.split("\n"):
        payload += f"data: {line}\n"
    payload += "\n"
    return payload


async def _stream_predictions(app: FastAPI, df: pd.DataFrame, templates_env):
    """Async generator yielding SSE events: meta, row (per result), done."""
    total = len(df)
    yield _format_sse("meta", json.dumps({"total": total}))

    row_template = templates_env.get_template("partials/galaxy_row.html")

    for _, row in df.iterrows():
        obj_id = str(row['objId'])
        ra = str(row['ra'])
        dec = str(row['dec'])
        link = f"https://www.legacysurvey.org/viewer?ra={ra}&dec={dec}&layer=ls-dr9&zoom=14"

        filename = f"ra{ra}_dec{dec}.fits"
        image_save_path = IMAGES_DIR / filename

        if not image_save_path.exists():
            try:
                image_path = await asyncio.to_thread(
                    download_legacy_fits, ra, dec, image_save_path
                )
            except Exception as e:
                result = {
                    "objId": obj_id,
                    "ra": ra,
                    "dec": dec,
                    "inner_ring_prob": 0.0,
                    "outer_ring_prob": 0.0,
                    "inner_ring_detected": False,
                    "outer_ring_detected": False,
                    "link": link,
                    "error": str(e),
                }
                html = row_template.render(row=result)
                yield _format_sse("row", html)
                continue
        else:
            image_path = image_save_path

        try:
            inner_prob, outer_prob = await asyncio.to_thread(
                _run_prediction, app, image_path
            )
        except Exception as e:
            result = {
                "objId": obj_id,
                "ra": ra,
                "dec": dec,
                "inner_ring_prob": 0.0,
                "outer_ring_prob": 0.0,
                "inner_ring_detected": False,
                "outer_ring_detected": False,
                "link": link,
                "error": str(e),
            }
            html = row_template.render(row=result)
            yield _format_sse("row", html)
            continue

        result = {
            "objId": obj_id,
            "ra": ra,
            "dec": dec,
            "inner_ring_prob": inner_prob,
            "outer_ring_prob": outer_prob,
            "inner_ring_detected": inner_prob >= INNER_THRESHOLD,
            "outer_ring_detected": outer_prob >= OUTER_THRESHOLD,
            "link": link,
        }
        app.state.results[obj_id] = result
        html = row_template.render(row=result)
        yield _format_sse("row", html)

    yield _format_sse("done", json.dumps({}))


@app.post("/api/batch_predict")
async def batch_predict(request: Request, file: UploadFile = File(...)):
    """Stream predictions as SSE: meta (total), row (per result), done."""
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV file required")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or malformed CSV: {e}")

    required = ['objId', 'ra', 'dec']
    for col in required:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required}")

    return StreamingResponse(
        _stream_predictions(request.app, df, templates.env),
        media_type="text/event-stream",
    )


@app.get("/api/galaxy_image")
async def galaxy_image(ra: str, dec: str):
    """Return PNG image from FITS at ra, dec. Uses cached images/ if available."""
    filename = f"ra{ra}_dec{dec}.fits"
    image_save_path = IMAGES_DIR / filename
    if not image_save_path.exists():
        try:
            image_save_path = download_legacy_fits(ra, dec, image_save_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download FITS: {e}")

    png_buf = fits_to_rgb_png(image_save_path, stretch=0.5, Q=6)
    return StreamingResponse(png_buf, media_type="image/png")


@app.get("/api/galaxy_3d_data")
async def galaxy_3d_data(request: Request, ra: str, dec: str):
    """Load FITS, apply baseline pipeline, downsample to 64x64, return JSON for Plotly surface."""
    filename = f"ra{ra}_dec{dec}.fits"
    image_save_path = IMAGES_DIR / filename
    if not image_save_path.exists():
        try:
            image_save_path = download_legacy_fits(ra, dec, image_save_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download FITS: {e}")

    with fits.open(image_save_path) as hdul:
        data = hdul[0].data
    data = np.asarray(data, dtype=np.float32)

    pipeline = request.app.state.baseline_pipeline
    tensor = torch.from_numpy(data)
    transformed = pipeline(tensor)
    if isinstance(transformed, torch.Tensor):
        arr = transformed.detach().cpu().numpy()
    else:
        arr = np.asarray(transformed)

    # Downsample to 64x64
    target_size = 64
    h, w = arr.shape[1], arr.shape[2]
    if h == 0 or w == 0:
        raise HTTPException(
            status_code=422,
            detail="Invalid FITS image: zero height or width",
        )
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
async def predict_from_url(request: Request, legacy_survey_url: str = Form(...)):
    """Predict from Legacy Survey URL. Returns HTML partial (predict_result.html) for HTMX swap."""
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
            image_save_path = download_legacy_fits(ra, dec, image_save_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download FITS: {e}")

    inner_prob, outer_prob = _run_prediction(request.app, image_save_path)

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
async def predict_from_fits(request: Request, fits_file: UploadFile = File(...)):
    """Predict from uploaded FITS file. Returns HTML partial (predict_result.html) for HTMX swap."""
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
        inner_prob, outer_prob = await asyncio.to_thread(
            _run_prediction, request.app, image_save_path
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


@app.get("/api/galaxy_image_fits")
async def galaxy_image_fits(request: Request, fits_id: str):
    """Return PNG image from uploaded FITS file by fits_id."""
    path = getattr(request.app.state, "uploaded_fits", {}).get(fits_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="FITS file not found or expired")
    png_buf = fits_to_rgb_png(path, stretch=0.5, Q=6)
    return StreamingResponse(png_buf, media_type="image/png")


@app.get("/api/galaxy_3d_data_fits")
async def galaxy_3d_data_fits(request: Request, fits_id: str):
    """Load uploaded FITS, apply baseline pipeline, downsample to 64x64, return JSON for Plotly surface."""
    path = getattr(request.app.state, "uploaded_fits", {}).get(fits_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="FITS file not found or expired")

    with fits.open(path) as hdul:
        data = hdul[0].data
    data = np.asarray(data, dtype=np.float32)

    pipeline = request.app.state.baseline_pipeline
    tensor = torch.from_numpy(data)
    transformed = pipeline(tensor)
    if isinstance(transformed, torch.Tensor):
        arr = transformed.detach().cpu().numpy()
    else:
        arr = np.asarray(transformed)

    target_size = 64
    h, w = arr.shape[1], arr.shape[2]
    if h == 0 or w == 0:
        raise HTTPException(
            status_code=422,
            detail="Invalid FITS image: zero height or width",
        )
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


if __name__ == "__main__":
    nest_asyncio.apply()
    host = "127.0.0.1"
    uvicorn.run(app, host=host, port=8000)
