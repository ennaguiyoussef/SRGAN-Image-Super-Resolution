"""Streamlit workspace for the project's three 4x upscaling methods."""
import hashlib
import io
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.evaluation import SCALE, resolve_image
from src.models import Generator


ROOT = Path(__file__).resolve().parent
MAX_INPUT_PIXELS = 1_000_000
METHODS = {
    "pretrained": "Générateur préentraîné",
    "bicubic": "Bicubique",
    "gan": "SRGAN",
}
CHECKPOINTS = {
    "pretrained": ROOT / "checkpoints/pretrain_full_continue_v1/best_generator.pt",
    "gan": ROOT / "checkpoints/adversarial_full_v2/best_gan_generator.pt",
}

HERO = """
<style>
.block-container { max-width: 78rem; padding-top: 3.25rem; }
.sr-lab-hero {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(10rem, 22rem);
    align-items: center; gap: 2rem; padding: 2.1rem clamp(1.15rem, 4vw, 3rem);
    background: #fff; border: 1px solid #ccd7eb; border-top: 4px solid #284bc4;
    border-radius: 8px; margin-bottom: 1.4rem;
}
.sr-lab-hero h1 {
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: clamp(2rem, 5vw, 3.65rem); line-height: 1.06;
    letter-spacing: -.045em; font-weight: 680; margin: .3rem 0 .55rem;
    padding: 0; color: #15233d;
}
.sr-lab-hero p { margin: 0; max-width: 42ch; color: #4a5d7a; font-size: 1.02rem; }
.sr-lab-hero .sr-lab-id { color: #284bc4; font-weight: 650; font-size: .91rem; }
.sr-lab-hero svg { display: block; width: 100%; max-height: 155px; }
.sr-lab-empty {
    border: 1px dashed #aebddb; border-radius: 8px; padding: 2.2rem 1.4rem;
    background: #f8faff; color: #4a5d7a;
}
.sr-lab-empty strong { display: block; color: #15233d; margin-bottom: .35rem; }
@media (max-width: 640px) {
    .sr-lab-hero { grid-template-columns: 1fr; gap: .85rem; }
    .sr-lab-hero svg { max-width: 15rem; max-height: 95px; }
}
</style>
<section class="sr-lab-hero">
  <div>
    <p class="sr-lab-id">Atelier SRGAN</p>
    <h1>Agrandir une image.</h1>
    <p>Multiplie sa largeur et sa hauteur par quatre. Compare les rendus côte à côte, puis télécharge ton choix.</p>
  </div>
  <svg viewBox="0 0 300 132" aria-hidden="true">
    <defs>
      <pattern id="sr-lr-grid" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M20 0H0V20" fill="none" stroke="#284bc4" stroke-width="1.4"/>
      </pattern>
      <pattern id="sr-hr-grid" width="5" height="5" patternUnits="userSpaceOnUse">
        <path d="M5 0H0V5" fill="none" stroke="#284bc4" stroke-width=".65"/>
      </pattern>
    </defs>
    <rect x="20" y="20" width="80" height="80" rx="2" fill="#dce7ff"/>
    <rect x="20" y="20" width="80" height="80" fill="url(#sr-lr-grid)" stroke="#284bc4"/>
    <path d="M120 60H176m-9-8 9 8-9 8" fill="none" stroke="#284bc4" stroke-width="2"/>
    <text x="147" y="48" text-anchor="middle" fill="#284bc4" font-size="16" font-weight="700">×4</text>
    <rect x="200" y="20" width="80" height="80" rx="2" fill="#eef3ff"/>
    <rect x="200" y="20" width="80" height="80" fill="url(#sr-hr-grid)" stroke="#284bc4"/>
    <text x="60" y="122" text-anchor="middle" fill="#4a5d7a" font-size="12">Image source</text>
    <text x="240" y="122" text-anchor="middle" fill="#4a5d7a" font-size="12">Image agrandie</text>
  </svg>
</section>
"""


@st.cache_resource(max_entries=4, show_spinner=False)
def load_generator(payload, device):
    """Cache a read-only inference model by exact checkpoint bytes and device."""
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except (EOFError, IndexError, pickle.UnpicklingError) as exc:
        raise ValueError("Invalid generator checkpoint.") from exc
    model = Generator()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def open_source(payload):
    with Image.open(io.BytesIO(payload), formats=("PNG", "JPEG", "WEBP")) as image:
        if image.width * image.height > MAX_INPUT_PIXELS:
            raise ValueError("Choisis une image de 1 mégapixel maximum.")
        return ImageOps.exif_transpose(image).convert("RGB")


def encode_result(image, method, weights, device):
    if method == "bicubic":
        size = (image.width * SCALE, image.height * SCALE)
        result = image.resize(size, Image.Resampling.BICUBIC)
    else:
        payload = weights.read_bytes() if isinstance(weights, Path) else weights
        model = load_generator(payload, device)
        pixels = resolve_image(model, image, tile_size=192 if device == "cuda" else 128)
        result = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


def weight_revision(source):
    if isinstance(source, Path):
        stat = source.stat()
        return (str(source), stat.st_mtime_ns, stat.st_size)
    return hashlib.sha256(source).hexdigest()


def main():
    st.set_page_config(page_title="Atelier SRGAN | Super-résolution ×4", page_icon="🔍", layout="wide")
    st.markdown(HERO, unsafe_allow_html=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.get_num_threads() > 4:
        torch.set_num_threads(4)

    controls, canvas = st.columns([1, 2.65], gap="large")
    with controls:
        st.subheader("Image et méthodes")
        upload = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg", "webp"], key="source_image")
        st.caption("PNG, JPEG ou WebP · 1 mégapixel maximum")
        default = ["pretrained"] if CHECKPOINTS["pretrained"].is_file() else ["bicubic"]
        selected = st.multiselect(
            "Rendus à générer", list(METHODS), default=default,
            format_func=METHODS.get, key="methods",
        )
        weights = {key: path for key, path in CHECKPOINTS.items() if path.is_file()}
        with st.expander("Poids des modèles"):
            for key, path in CHECKPOINTS.items():
                uploaded_weights = st.file_uploader(
                    METHODS[key], type=["pt", "pth"], key=f"weights_{key}"
                )
                if uploaded_weights is not None:
                    weights[key] = uploaded_weights.getvalue()
                    st.caption(f"Fichier importé · {uploaded_weights.name}")
                elif key in weights:
                    st.caption(f"Fichier local · {path.name}")
                else:
                    st.caption(f"Importe `{path.name}` depuis ton dossier Drive.")
        missing = [METHODS[key] for key in selected if key != "bicubic" and key not in weights]
        if missing:
            st.info(f"Ajoute les poids pour {', '.join(missing)} ou sélectionne uniquement Bicubique.")
        generate = st.button(
            "Générer les rendus ×4", type="primary", width="stretch",
            disabled=upload is None or not selected or bool(missing),
        )
        st.caption(f"Calcul sur {'GPU CUDA' if device == 'cuda' else 'CPU'}")

    with canvas:
        st.subheader("Comparaison")
        if upload is None:
            st.session_state.pop("render_request", None)
            st.session_state.pop("render_outputs", None)
            st.markdown(
                '<div class="sr-lab-empty"><strong>Dépose une image pour commencer.</strong>'
                'Sélectionne les méthodes que tu souhaites comparer.</div>',
                unsafe_allow_html=True,
            )
            return

        payload = upload.getvalue()
        try:
            image = open_source(payload)
        except ValueError as exc:
            st.error(str(exc))
            return
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error("Impossible de lire cette image. Essaie un fichier PNG, JPEG ou WebP valide.")
            return

        output_size = f"{image.width * SCALE} × {image.height * SCALE} px"
        st.caption(f"{image.width} × {image.height} px en entrée → {output_size} en sortie")
        request = (
            hashlib.sha256(payload).hexdigest(),
            tuple((key, weight_revision(weights[key]) if key in weights else None) for key in selected),
        )
        if st.session_state.get("render_request") != request:
            st.session_state["render_request"] = request
            st.session_state["render_outputs"] = {}

        if generate:
            outputs = {}
            for key in selected:
                try:
                    with st.spinner(f"Agrandissement avec {METHODS[key]}…"):
                        outputs[key] = encode_result(image, key, weights.get(key), device)
                except torch.cuda.OutOfMemoryError:
                    st.error("La mémoire GPU est insuffisante. Essaie une image plus petite.")
                except (OSError, RuntimeError, ValueError, TypeError, EOFError, FloatingPointError):
                    st.error(f"Échec pour {METHODS[key]}. Vérifie le fichier de poids et essaie une image plus petite.")
            st.session_state["render_outputs"] = outputs

        outputs = st.session_state.get("render_outputs", {})
        if not outputs:
            st.image(image, caption="Image source", width=min(420, max(180, image.width)))
            return

        panels = st.columns(len(outputs) + 1)
        with panels[0]:
            st.markdown("**Image source**")
            st.image(image, width="stretch")
        for panel, (key, png) in zip(panels[1:], outputs.items()):
            with panel:
                st.markdown(f"**{METHODS[key]} ×4**")
                st.image(png, width="stretch")
                st.download_button(
                    "Télécharger le PNG", png,
                    file_name=f"{Path(upload.name).stem}_{key}_x4.png",
                    mime="image/png", key=f"download_{key}", on_click="ignore", width="stretch",
                )


if __name__ == "__main__":
    main()
