"""Download DIV2K HR archives and prepare disjoint image-level splits."""
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
GROUPS = {
    "train": ("DIV2K_train_HR", range(1, 801)),
    "valid": ("DIV2K_valid_HR", range(801, 901)),
}


def split_manifest(seed=42):
    """Use a private validation/test split; official challenge test HR is not used."""
    train = [f"DIV2K_train_HR/{i:04d}.png" for i in range(1, 801)]
    valid = [f"DIV2K_valid_HR/{i:04d}.png" for i in range(801, 901)]
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(valid)
    return {
        "dataset": "DIV2K",
        "seed": seed,
        "protocol": "Custom 50/50 split of official validation HR; not official challenge test.",
        "train": train,
        "train_smoke": train[:100],
        "validation": valid[:50],
        "test": valid[50:],
    }


def save_manifest(path, manifest):
    """Refuse to silently replace a different persisted split."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Existing manifest differs: {path}. Choose a new experiment folder.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract_hr_archive(archive, destination, folder, image_ids):
    """Extract only expected PNGs; ZIP reads validate their CRC before committing files."""
    destination = Path(destination)
    expected = [f"{folder}/{i:04d}.png" for i in image_ids]
    with zipfile.ZipFile(archive) as source:
        members = set(source.namelist())
        missing = set(expected) - members
        if missing:
            raise ValueError(f"Archive is missing {len(missing)} expected images.")
        for name in expected:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".png.part")
            try:
                with source.open(name) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                with Image.open(temporary) as image:
                    image.verify()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def download_div2k(data_root, cache_root):
    """Download full train/validation HR ZIPs; cache on Drive, extract locally.

    A 100-image smoke subset reduces processing, not archive download size.
    Interrupted downloads restart on rerun; completed archives are reused.
    """
    from tqdm.auto import tqdm

    data_root, cache_root = Path(data_root), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    for folder, image_ids in GROUPS.values():
        expected = [data_root / folder / f"{i:04d}.png" for i in image_ids]
        marker = data_root / folder / ".extraction_complete"
        if marker.exists() and all(p.is_file() and p.stat().st_size > 0 for p in expected):
            print(f"{folder}: already extracted ({len(expected)} images)")
            continue
        archive = cache_root / f"{folder}.zip"
        if not archive.exists():
            partial = archive.with_suffix(".zip.part")
            try:
                request = urllib.request.Request(BASE_URL + archive.name)
                with urllib.request.urlopen(request, timeout=120) as response:
                    total = int(response.headers.get("Content-Length", 0)) or None
                    with partial.open("wb") as handle, tqdm(
                        total=total, unit="B", unit_scale=True, desc=archive.name
                    ) as progress:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                            progress.update(len(chunk))
                    if total is not None and partial.stat().st_size != total:
                        raise IOError("Download length does not match Content-Length.")
                if not zipfile.is_zipfile(partial):
                    raise ValueError("Downloaded file is not a ZIP archive.")
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        try:
            extract_hr_archive(archive, data_root, folder, image_ids)
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Archive extraction failed for {archive}. "
                "If the cached archive is corrupt, remove that ZIP and rerun."
            ) from exc
        marker.write_text("complete\n", encoding="utf-8")
        print(f"{folder}: extracted {len(expected)} images")


def paired_crop(image, hr_size=96, scale=4, training=False, rng=None):
    """Crop HR first, then downsample with Pillow bicubic (not MATLAB imresize)."""
    if type(scale) is not int or type(hr_size) is not int or scale <= 0 or hr_size <= 0:
        raise ValueError("Crop size and scale must be positive integers.")
    if hr_size % scale:
        raise ValueError("HR size must be divisible by scale.")
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < hr_size:
        raise ValueError(f"Image {image.size} is smaller than the HR crop {hr_size}.")
    if training:
        rng = rng if rng is not None else random
        left, top = rng.randint(0, width - hr_size), rng.randint(0, height - hr_size)
    else:
        left, top = (width - hr_size) // 2, (height - hr_size) // 2
    hr = image.crop((left, top, left + hr_size, top + hr_size))
    lr = hr.resize((hr_size // scale, hr_size // scale), Image.Resampling.BICUBIC)
    return lr, hr
