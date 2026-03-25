import asyncio
import threading
import time
import urllib.request
from pathlib import Path
import logging
from collections.abc import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)



async def download_legacy_fits_async(
    ra, dec, out_path: Path, size=256, pixscale=0.262, layer="ls-dr9", bands="grz"
) -> AsyncGenerator[dict, None]:
    """
    Stream FITS download to disk, yielding progress {downloaded, total} per chunk.
    Final yield is {"path": out_path}. Use for SSE progress updates.
    """
    url = (
        "https://www.legacysurvey.org/viewer/cutout.fits?"
        f"ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size}&bands={bands}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading FITS for ra={ra} dec={dec}")
    headers = {"User-Agent": "GalaxyRingDetector/1.0 (https://huggingface.co/spaces/...)"}
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            total = None
            cl = resp.headers.get("content-length")
            if cl:
                try:
                    total = int(cl)
                except ValueError:
                    pass
            downloaded = 0
            yield {"downloaded": 0, "total": total}
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    yield {"downloaded": downloaded, "total": total}
    logger.info(f"Downloaded to {out_path}")
    yield {"path": str(out_path)}


def download_legacy_fits(ra, dec, out_path: Path, size=256, pixscale=0.262, layer="ls-dr9", bands="grz"):
    """Descarga un FITS 'cutout' de Legacy Survey."""
    url = (
        "https://www.legacysurvey.org/viewer/cutout.fits?"
        f"ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size}&bands={bands}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading FITS for ra={ra} dec={dec}")
    req = urllib.request.Request(url, headers={"User-Agent": "GalaxyRingDetector/1.0 (https://huggingface.co/spaces/...)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        with open(out_path, "wb") as f:
            chunk_size = 8192
            while chunk := resp.read(chunk_size):
                f.write(chunk)
    logger.info(f"Downloaded to {out_path}")
    return out_path