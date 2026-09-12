"""
app.py
Dental Canal & Caries Detection System — Streamlit Web App

Features:
  - Upload X-ray image
  - Animated loading during processing
  - Side-by-side original vs result display
  - Download processed image
  - Clean medical UI — no confusing metrics shown to user

Run:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import tempfile
import os
import base64 as _b64


def _get_img_b64(path):
    """Load an image file and return base64 string. Falls back to empty if not found."""
    try:
        with open(path, 'rb') as f:
            return _b64.b64encode(f.read()).decode()
    except Exception:
        return ''


# Load logo image at startup (minion.png must be in same folder as app.py)
_LOGO_B64 = _get_img_b64('minion.png')
_LOGO_SRC  = f"data:image/png;base64,{_LOGO_B64}" if _LOGO_B64 else ""


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "DentalScan AI",
    page_icon   = "🦷",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — clean medical aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide streamlit default elements */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }

    /* Keep header and toolbar visible so sidebar toggle remains accessible */
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: visible !important; }

    /* Main background */
    .stApp {
        background: #f0f4f8;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(100,220,200,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .findings-text {
        color: #0c4a6e !important;
        font-size: 0.9rem;
        line-height: 1.5;
        font-weight: 400;
        margin: 0;
    }
    .hero-sub {
        font-size: 1rem;
        color: rgba(255,255,255,0.6);
        margin: 0;
        font-weight: 300;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(100,220,200,0.15);
        border: 1px solid rgba(100,220,200,0.3);
        color: #64dcc8;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 16px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 16px;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 48px 24px;
        text-align: center;
        background: #f8fafc;
        transition: all 0.2s;
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 12px;
    }
    .upload-text {
        color: #64748b;
        font-size: 0.95rem;
    }

    /* Detection result tags */
    .tag-canal {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.2);
        color: #16a34a;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .tag-caries {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.2);
        color: #dc2626;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .tag-none {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(148,163,184,0.1);
        border: 1px solid rgba(148,163,184,0.2);
        color: #64748b;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1e293b !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
    }

    /* Simple arrow color states: white when sidebar open, sidebar color when closed */
    /* Primary selector Streamlit uses for the toggle */
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] path {
        transition: fill 0.12s ease, color 0.12s ease;
    }
    [data-testid="collapsedControl"] button[aria-expanded="true"] svg,
    [data-testid="collapsedControl"] button[aria-expanded="true"] path {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="collapsedControl"] button[aria-expanded="false"] svg,
    [data-testid="collapsedControl"] button[aria-expanded="false"] path {
        fill: #1e293b !important;
        color: #1e293b !important;
    }

    /* Fallback: target any svg inside the collapsed control wrapper */
    [data-testid="collapsedControl"] [role="button"] svg {
        transition: fill 0.12s ease;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0f2027, #2c5364) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #ffffff !important;
        color: #1e293b !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        width: 100% !important;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 20px 0;
    }

    /* Step indicator */
    .step {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
        font-size: 0.9rem;
        color: #475569;
    }
    .step-num {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #0f2027;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 24px 0;
    }

    /* Image captions */
    .img-label {
        text-align: center;
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 8px;
    }

    /* Alert box */
    .notice-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #92400e;
        margin-bottom: 16px;
    }

    /* Target Streamlit's expand sidebar button specifically */
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
        transition: fill 0.12s ease, color 0.12s ease;
        fill: #1e293b !important;
        color: #1e293b !important;
    }
    [data-testid="stExpandSidebarButton"][aria-expanded="true"] [data-testid="stIconMaterial"] {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Make text selection clearly readable across the app */
    ::selection {
        background: #1e40af; /* strong indigo blue */
        color: #ffffff;
    }
    ::-moz-selection {
        background: #1e40af;
        color: #ffffff;
    }
    /* Ensure selection inside Streamlit components also follows the rule */
    .stApp *::selection { background: #1e40af; color: #ffffff; }
    .stApp *::-moz-selection { background: #1e40af; color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — cached so it loads only once
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """
    Load all models once and cache them.
    Subsequent calls return the same loaded objects.
    """
    try:
        from unet_inference_v2            import build_nerve_mask_unet
        from nerve_postprocessing_updated import post_process_nerve_mask
        from yolo_inference_v2            import detect_caries_yolo
        return {
            'build_nerve_mask' : build_nerve_mask_unet,
            'postprocess'      : post_process_nerve_mask,
            'detect_caries'    : detect_caries_yolo,
            'status'           : 'ok',
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(img_gray, pipeline, conf_threshold, iou_threshold,
                  use_tta, unet_checkpoint, yolo_weights):
    """Run full detection pipeline and return annotated image + findings."""

    # Step 1 — Canal detection
    raw_nerve  = pipeline['build_nerve_mask'](
        img_gray,
        checkpoint_path = unet_checkpoint,
        threshold       = 0.35,
        use_tta         = use_tta,
    )
    nerve_mask = pipeline['postprocess'](raw_nerve.copy(), img=img_gray)
    nerve_px   = int(np.sum(nerve_mask > 0))

    # Step 2 — Caries detection
    caries = pipeline['detect_caries'](
        img_gray,
        weights        = yolo_weights,
        conf_threshold = conf_threshold,
        iou_threshold  = iou_threshold,
        use_tta        = use_tta,
        nerve_mask     = nerve_mask,
    )

    # Step 3 — Annotate
    annotated = annotate(img_gray, nerve_mask, caries)

    return annotated, nerve_px, caries


def annotate(img, nerve_mask, caries):
    """Draw canal overlay and caries boxes — same as your existing pipeline."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Canal — green overlay
    if int(np.sum(nerve_mask > 0)) > 0:
        overlay = np.zeros_like(out)
        overlay[nerve_mask > 0] = [0, 220, 100]
        mask_bool = cv2.cvtColor(nerve_mask, cv2.COLOR_GRAY2BGR).astype(bool)
        out[mask_bool] = (
            0.65 * out[mask_bool] +
            0.35 * overlay[mask_bool]
        ).astype(np.uint8)

    # Caries — red boxes
    for cx, cy, conf, x1, y1, x2, y2 in caries:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label       = f"Caries {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1-th-6)),
                      (x1+tw+4, y1), (0, 0, 200), -1)
        cv2.putText(out, label, (x1+2, max(th, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

    return out


def img_to_bytes(img_bgr):
    """Convert BGR numpy array to PNG bytes for download."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf     = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        st.markdown("---")

        st.markdown("**Confidence Threshold**")
        conf = st.slider("", 0.10, 0.90, 0.25, 0.05,
                         key="conf",
                         help="Higher = fewer but more confident detections")

        st.markdown("**IoU Overlap Threshold**")
        iou = st.slider("", 0.10, 0.90, 0.30, 0.05,
                        key="iou",
                        help="Post-processing overlap filter. Less critical with YOLO26m's native end-to-end NMS-free head — adjust only if overlapping boxes appear")

        st.markdown("**Test-Time Augmentation**")
        tta = st.toggle("Enable TTA", value=True,
                        help="Improves accuracy slightly, takes longer")

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.8'>
        🟢 Green overlay — Nerve canal<br>
        🔴 Red box — Caries detection
        </div>
        """, unsafe_allow_html=True)

    # Model paths hardcoded — not exposed to user
    unet_ckpt = "attention_unet_best.pth"
    yolo_w    = "best_caries_yolo26.pt"

    return conf, iou, tta, unet_ckpt, yolo_w


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Sidebar
    conf, iou, tta, unet_ckpt, yolo_w = render_sidebar()

    # Hero header — logo loaded from minion.png at runtime
    _logo_tag = (
        f'<img src="{_LOGO_SRC}" ' +
        'style="width:60px;height:60px;border-radius:50%;object-fit:cover;' +
        'box-shadow:0 2px 8px rgba(0,0,0,0.3);flex-shrink:0;margin-right:4px;"/>' 
        if _LOGO_SRC else ""
    )
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">AI-Powered Analysis</div>
        <h1 class="hero-title" style="display:flex;align-items:center;gap:14px;">
            {_logo_tag}DentalScan AI
        </h1>
        <p class="hero-sub">
            Automated detection of mandibular nerve canals and dental caries
            from panoramic X-ray images
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How to use
    st.markdown("""
    <div class="card">
        <div class="card-title">How to use</div>
        <div class="step"><div class="step-num">1</div> Upload your panoramic X-ray image (JPG, PNG)</div>
        <div class="step"><div class="step-num">2</div> Adjust detection settings in the sidebar if needed</div>
        <div class="step"><div class="step-num">3</div> Click Analyse and wait for results</div>
        <div class="step"><div class="step-num">4</div> Download the annotated result image</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="card"><div class="card-title">Upload X-Ray Image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type   = ['jpg', 'jpeg', 'png', 'bmp'],
        help   = "Upload a panoramic dental X-ray image",
        label_visibility = "collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div style='text-align:center; padding:40px; color:#94a3b8'>
            <div style='font-size:3rem'>📤</div>
            <div style='font-size:0.95rem; margin-top:8px'>
                Upload an X-ray image above to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Convert uploaded file to grayscale numpy array
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Show preview + image info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 caption="Uploaded X-ray", width="stretch")
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Image Info</div>
            <div style='font-size:0.85rem; color:#475569; line-height:2'>
                <b>File:</b> {uploaded.name}<br>
                <b>Size:</b> {uploaded.size // 1024} KB<br>
                <b>Dimensions:</b> {img_gray.shape[1]} × {img_gray.shape[0]} px<br>
                <b>TTA:</b> {'Enabled' if tta else 'Disabled'}<br>
                <b>Confidence:</b> {conf}<br>
                <b>IoU Filter:</b> {iou}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="notice-box">⚠️ For best results, use high-quality panoramic X-ray images.</div>',
                    unsafe_allow_html=True)

    # Analyse button
    st.markdown("<br>", unsafe_allow_html=True)
    analyse_btn = st.button("🔍  Analyse X-Ray", use_container_width=True)

    if not analyse_btn:
        return

    # ── Load pipeline ──────────────────────────────────────────────────────
    pipeline = load_pipeline()
    if pipeline['status'] == 'error':
        st.error(f"Failed to load models: {pipeline['message']}")
        return
    
    # ── Synchronized loading + detection ───────────────────────────────────
    progress_bar = st.progress(0)
    status_text  = st.empty()

    def update_status(pct, msg):
        status_text.markdown(
            f"<div style='text-align:center; color:#475569; "
            f"font-size:0.95rem; padding:8px'>{msg}</div>",
            unsafe_allow_html=True)
        progress_bar.progress(pct)

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(
                suffix=f".{uploaded.name.split('.')[-1]}",
                delete=False) as tmp:
            tmp.write(file_bytes.tobytes())
            tmp_path = tmp.name

        update_status(0.10, "Loading AI models...")
        time.sleep(1.2)

        # Step 1 — Canal detection
        update_status(0.30, "Detecting nerve canal...")
        raw_nerve  = pipeline['build_nerve_mask'](
            img_gray,
            checkpoint_path = unet_ckpt,
            threshold       = 0.35,
            use_tta         = tta,
        )
        nerve_mask = pipeline['postprocess'](raw_nerve.copy(), img=img_gray)
        nerve_px   = int(np.sum(nerve_mask > 0))

        # Step 2 — Caries detection
        update_status(0.65, "Scanning for caries...")
        caries = pipeline['detect_caries'](
            img_gray,
            weights        = yolo_w,
            conf_threshold = conf,
            iou_threshold  = iou,
            use_tta        = tta,
            nerve_mask     = nerve_mask,
        )

        # Step 3 — Annotate
        update_status(0.85, "Generating annotated result...")
        time.sleep(1.0)
        annotated = annotate(img_gray, nerve_mask, caries)

        os.unlink(tmp_path)
        update_status(1.00, "Analysis complete!")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Detection failed: {str(e)}")
        st.info("Make sure model checkpoint paths are correct in the sidebar.")
        return


    # ── Clear loading UI ────────────────────────────────────────────────────
    progress_bar.empty()
    status_text.empty()

    # ── Results ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-title" style="margin-bottom:16px">Analysis Results</div>',
                unsafe_allow_html=True)

    # Side-by-side images
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 width="stretch")
        st.markdown('<div class="img-label">Original</div>',
                    unsafe_allow_html=True)
    with res_col2:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 width="stretch")
        st.markdown('<div class="img-label">Detection Result</div>',
                    unsafe_allow_html=True)

    # Findings summary — simple, no confusing metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">Findings</div>', unsafe_allow_html=True)

    canal_detected  = nerve_px > 100
    caries_detected = len(caries) > 0

    tags_html = ""
    if canal_detected:
        tags_html += '<span class="tag-canal">🟢 Nerve Canal Detected</span>'
    else:
        tags_html += '<span class="tag-none">⚪ No Canal Detected</span>'

    if caries_detected:
        tags_html += f'<span class="tag-caries">🔴 {len(caries)} Caries Region{"s" if len(caries)>1 else ""} Found</span>'
    else:
        tags_html += '<span class="tag-canal">🟢 No Caries Detected</span>'

    st.markdown(tags_html, unsafe_allow_html=True)

    # Simple findings text — clinical language, no numbers
    st.markdown("<br>", unsafe_allow_html=True)
    findings = []
    if canal_detected:
        findings.append(("✔", "The mandibular nerve canal has been identified and highlighted in green.", "#16a34a"))
    else:
        findings.append(("⚠", "The nerve canal could not be clearly identified in this image.", "#d97706"))
    if caries_detected:
        findings.append(("✔", f"{len(caries)} potential caries region{'s' if len(caries)>1 else ''} detected and marked in red. Physical examination and clinical correlation are recommended for confirmation.", "#dc2626"))
    else:
        findings.append(("✔", "No caries regions were detected in this image.", "#16a34a"))

    findings_html = '<div style="margin-top:8px">'
    for icon, text, color in findings:
        findings_html += (
            f'<div style="display:flex;align-items:flex-start;gap:10px;'
            f'padding:12px 16px;margin-bottom:8px;background:#ffffff;'
            f'border-radius:8px;border-left:4px solid {color};'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.05)">'
            f'<span style="color:{color};font-size:1rem;font-weight:600;flex-shrink:0">{icon}</span>'
            f'<p class="findings-text">{text}</p>'
            f'</div>'
        )
    findings_html += '</div>'
    st.markdown(findings_html, unsafe_allow_html=True)

    # Disclaimer — professional tone for clinical users
    st.markdown("""
    <div style='
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 16px;
        font-size: 0.85rem;
        color: #0c4a6e;
        line-height: 1.6;
    '>
        📋 <b>Clinical Note:</b> This AI-generated result is intended to assist
        in preliminary screening only. Clinical correlation and physical
        examination are strongly recommended before any diagnostic conclusion
        is made.
    </div>
    """, unsafe_allow_html=True)

    # Download button
    st.markdown("<br>", unsafe_allow_html=True)
    img_bytes = img_to_bytes(annotated)
    st.download_button(
        label    = "⬇️  Download Annotated Result",
        data     = img_bytes,
        file_name= f"dentalscan_{uploaded.name.split('.')[0]}_result.png",
        mime     = "image/png",
        use_container_width = True,
    )

    # Footer
    st.markdown("""
    <div class="footer">
        DentalScan AI — Powered by Attention U-Net &amp; YOLO26m &nbsp;|&nbsp;
        For research and educational purposes only
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()