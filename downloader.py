import urllib.request
from pathlib import Path

def download_legacy_fits(ra, dec, out_path: Path, size=256, pixscale=0.262, layer="ls-dr9", bands="grz"):
    """Descarga un FITS 'cutout' de Legacy Survey."""
    url = (
        "https://www.legacysurvey.org/viewer/cutout.fits?"
        f"ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size}&bands={bands}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)
    return out_path