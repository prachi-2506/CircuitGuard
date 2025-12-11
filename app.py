# app_updated.py
# CircuitGuard Streamlit app - updated with:
# - chunked processing for large batches
# - per-image PDF generation (reportlab)
# - CSV (full + summary) and ZIP export (PDFs + annotated PNGs)
# - selection & pagination for thumbnails
# - preserved original styling and charts

import os
import io
import time
import math
import zipfile
import tempfile
from collections import Counter
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# ------------------ CONFIG ------------------
LOCAL_MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\yolo deploy\best.pt"
CLOUD_MODEL_PATH = "best.pt"
MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else CLOUD_MODEL_PATH

CONFIDENCE = 0.25
IOU = 0.45

# UI / performance tuning
THUMBNAILS_PER_PAGE = 24
CHUNK_PROCESS_SIZE = 12  # reduce to fit memory / GPU
PDF_DPI = 150

st.set_page_config(
    page_title="CircuitGuard – PCB Defect Detection",
    page_icon="🛡️",
    layout="wide",
)

# ------------------ CUSTOM STYLING ------------------
# (Preserve the user's long CSS block from previous app.py)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bitcount+Prop+Single:wght@400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #f8fbff;
        font-family: 'Poppins', sans-serif;
        color: #102a43;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #e8f5ff 0%, #e7fff7 100%); border-right: 1px solid #d0e2ff; }
    [data-testid="stSidebar"] * { color: #102a43 !important; }
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code { background: #e5e7eb !important; color: #111827 !important; }
    [data-testid="stToolbar"] * { color: #e5e7eb !important; }
    h2, h3 { font-weight: 600; color: #13406b; font-family: 'Space Grotesk', 'Poppins', system-ui, -apple-system, sans-serif; }
    .stButton>button { border-radius: 999px; padding: 0.5rem 1.25rem; border: none; font-weight: 500; background: #85c5ff; color: #0f172a; box-shadow: 0 8px 14px rgba(148, 163, 184, 0.28); transition: transform 0.18s ease-out, box-shadow 0.18s ease-out, background 0.18s ease-out; animation: pulse-soft 2.4s ease-in-out infinite; }
    .stButton>button:hover { background: #63b1ff; transform: translateY(-1px) scale(1.01); box-shadow: 0 12px 22px rgba(148, 163, 184, 0.38); }
    @keyframes pulse-soft { 0% { transform: translateY(0); box-shadow: 0 8px 14px rgba(148, 163, 184, 0.25);} 50% { transform: translateY(-1px); box-shadow: 0 12px 22px rgba(148, 163, 184, 0.4);} 100% { transform: translateY(0); box-shadow: 0 8px 14px rgba(148, 163, 184, 0.25);} }
    [data-testid="stDownloadButton"] > button { background: #e5e7eb !important; color: #111827 !important; border-radius: 999px !important; border: 1px solid #cbd5f5 !important; font-weight: 500; }
    .upload-box { border-radius: 18px; border: 1px dashed #a3c9ff; padding: 1.5rem; background: #ffffff; }
    [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] label { color: #f9fafb !important; }
    [data-testid="stFileUploader"] button { background: #111827 !important; color: #f9fafb !important; border-radius: 999px !important; border: none !important; }
    .metric-card { border-radius: 18px; padding: 0.75rem 1rem; background: #ffffff; border: 1px solid #dbeafe; }
    .metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.1rem; }
    .metric-value { font-size: 1.15rem; font-weight: 600; color: #111827; font-family: 'Space Grotesk', 'Poppins', system-ui, sans-serif; }
    .logo-circle { width: 60px; height: 60px; border-radius: 50%; background: #e0f2fe; display: flex; align-items: center; justify-content: center; font-size: 32px; margin-bottom: 0.4rem; }
    .header-container { display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 0.5rem; margin-bottom: 0.75rem; }
    .main-title { font-family: 'Space Grotesk', system-ui, -apple-system, BlinkMacSystemFont, 'Poppins', sans-serif; font-weight: 600; font-size: 2.8rem; text-align: center; color: #13406b; letter-spacing: 0.03em; }
    .subtitle-text { font-size: 0.95rem; color: #334e68; text-align: center; }
    .instruction-card { border-radius: 18px; background: #ffffff; border: 1px solid #dbeafe; padding: 1rem 1.25rem; margin: 1rem 0; font-size: 0.9rem; }
    .instruction-card ol { margin-left: 1.1rem; padding-left: 0.5rem; }
    .instruction-card li { margin-bottom: 0.25rem; }
    .defect-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
    .defect-badge { padding: 0.2rem 0.6rem; border-radius: 999px; background: #e0f2fe; font-size: 0.8rem; color: #13406b; }
    .robot-success { margin: 1rem 0 0.4rem 0; padding: 0.8rem 1.2rem; border-radius: 12px; background: linear-gradient(90deg, #0f172a 0%, #1f2937 55%, #16a34a 100%); color: #e5f9ff; font-family: 'JetBrains Mono', SFMono-Regular, Menlo, monospace; font-size: 0.9rem; letter-spacing: 0.09em; position: relative; overflow: hidden; text-transform: uppercase; }
    .robot-success::after { content: ""; position: absolute; inset: 0; background: repeating-linear-gradient(0deg, rgba(148, 163, 184, 0.0), rgba(148, 163, 184, 0.0) 2px, rgba(148, 163, 184, 0.25) 3px); mix-blend-mode: soft-light; opacity: 0.4; pointer-events: none; animation: scanlines 6s linear infinite; }
    @keyframes scanlines { 0% { transform: translateY(-3px); } 100% { transform: translateY(3px); } }
    .robot-label { color: #a7f3d0; margin-right: 0.75rem; }
    .status-strip { margin: 0.1rem 0 1.2rem 0; padding: 0.65rem 1.1rem; border-radius: 999px; background: #d1fae5; color: #064e3b; font-size: 0.9rem; font-weight: 500; }
    .vega-embed, .vega-embed canvas { max-width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ SESSION STATE ------------------
if "full_results_df" not in st.session_state:
    st.session_state["full_results_df"] = None
if "annotated_images" not in st.session_state:
    st.session_state["annotated_images"] = []
if "show_download" not in st.session_state:
    st.session_state["show_download"] = False
if "images" not in st.session_state:
    st.session_state["images"] = []  # list of dicts for queued images
if "page" not in st.session_state:
    st.session_state["page"] = 1

# ------------------ MODEL LOADING & INFERENCE ------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def run_inference(model, image):
    """Run detection and return plotted image (PIL) + raw result."""
    results = model.predict(image, conf=CONFIDENCE, iou=IOU)
    r = results[0]
    plotted = r.plot()  # BGR numpy array
    plotted = plotted[:, :, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(plotted)
    return pil_img, r


# Annotation helper used for PDF table (convert result.boxes -> defects list)
def pack_defects_from_result(result) -> List[Dict]:
    boxes = result.boxes
    if len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.tolist()
    cls_indices = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    defects = []
    for i, (coords, c, cf) in enumerate(zip(xyxy, cls_indices, confs)):
        x1, y1, x2, y2 = coords
        defects.append({
            "defect_id": i + 1,
            "defect_type": str(c),
            "x": round(float(x1), 1),
            "y": round(float(y1), 1),
            "width": round(float(x2 - x1), 1),
            "height": round(float(y2 - y1), 1),
            "center_x": round(float((x1 + x2) / 2), 1),
            "center_y": round(float((y1 + y2) / 2), 1),
            "confidence": float(cf),
        })
    return defects


def generate_pdf_for_image(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    """Generate a single-image PDF containing before/after and defect table."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    page_w, page_h = landscape(A4)

    # Header
    header_text = f"{meta.get('project_name', 'CircuitGuard')} — {meta.get('batch_id','')} — {meta.get('filename','') }"
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, page_h - 30, header_text)
    c.setFont("Helvetica", 8)
    c.drawString(30, page_h - 45, f"Processed: {meta.get('processed_at', '')}    Model: {meta.get('model_version','')}")

    # Images placement
    max_img_h = page_h * 0.55
    max_img_w = (page_w - 100) / 2

    def place_pil(pil_img, x_offset):
        iw, ih = pil_img.size
        scale = min(max_img_w / iw, max_img_h / ih, 1.0)
        iw2, ih2 = int(iw * scale), int(ih * scale)
        reader = ImageReader(pil_img.resize((iw2, ih2)))
        y = page_h - 80 - ih2
        c.drawImage(reader, x_offset, y, width=iw2, height=ih2, preserveAspectRatio=True)

    place_pil(original_pil, 30)
    place_pil(annotated_pil, 50 + max_img_w)

    # Defect table
    c.setFont("Helvetica-Bold", 10)
    table_y = page_h - 80 - max_img_h - 20
    c.drawString(30, table_y + 12, "Detected defects (id, type, x, y, w, h, center_x, center_y, confidence)")
    c.setFont("Helvetica", 9)
    y = table_y
    row_h = 12
    headers = ["id", "type", "x", "y", "w", "h", "cx", "cy", "conf"]
    x_positions = [30, 65, 140, 180, 215, 250, 285, 320, 355]
    for htext, xpos in zip(headers, x_positions):
        c.drawString(xpos, y, htext)
    y -= row_h
    if not defects:
        c.drawString(30, y, "No defects detected.")
    else:
        for d in defects:
            if y < 40:
                c.showPage()
                y = page_h - 40
            row = [str(d.get("defect_id", "")), str(d.get("defect_type", "")), str(d.get("x", "")), str(d.get("y", "")), str(d.get("width", "")), str(d.get("height", "")), str(d.get("center_x", "")), str(d.get("center_y", "")), f"{d.get('confidence', 0):.3f}"]
            for item, xpos in zip(row, x_positions):
                c.drawString(xpos, y, item)
            y -= row_h

    c.setFont("Helvetica", 7)
    c.drawString(30, 20, f"Generated by CircuitGuard • {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()
    buf.seek(0)
    return buf.read()


def generate_csvs(images_list: List[Dict]) -> Tuple[bytes, bytes]:
    full_rows = []
    summary_rows = []
    for img in images_list:
        filename = img["filename"]
        batch_id = img.get("batch_id", "")
        model_version = img.get("model_version", "")
        defects = img.get("defects", [])
        for d in defects:
            full_rows.append({
                "batch_id": batch_id,
                "image_filename": filename,
                "defect_id": d.get("defect_id"),
                "defect_type": d.get("defect_type"),
                "x": d.get("x"),
                "y": d.get("y"),
                "width": d.get("width"),
                "height": d.get("height"),
                "center_x": d.get("center_x"),
                "center_y": d.get("center_y"),
                "confidence": d.get("confidence"),
                "model_version": model_version,
            })
        summary_rows.append({
            "batch_id": batch_id,
            "image_filename": filename,
            "defect_count": len(defects),
            "max_confidence": max([d["confidence"] for d in defects], default=0),
            "model_version": model_version,
        })
    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    b1 = full_df.to_csv(index=False).encode("utf-8") if not full_df.empty else b""
    b2 = summary_df.to_csv(index=False).encode("utf-8") if not summary_df.empty else b""
    return b1, b2


def make_zip_for_selection(images_list: List[Dict], include_pngs=True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        full_csv, summary_csv = generate_csvs(images_list)
        if full_csv:
            zf.writestr("results_full.csv", full_csv)
        if summary_csv:
            zf.writestr("results_summary.csv", summary_csv)
        for img in images_list:
            safe = os.path.splitext(img["filename"])[0]
            pdf = generate_pdf_for_image(img["original"], img.get("annotated" , img["original"]), img.get("defects", []), {
                "project_name": "CircuitGuard",
                "batch_id": img.get("batch_id", ""),
                "filename": img["filename"],
                "processed_at": img.get("processed_at", ""),
                "model_version": img.get("model_version", ""),
            })
            zf.writestr(f"{safe}_result.pdf", pdf)
            if include_pngs and img.get("annotated") is not None:
                b = io.BytesIO()
                img["annotated"].save(b, format="PNG")
                zf.writestr(f"annotated_{safe}.png", b.getvalue())
    buf.seek(0)
    return buf.getvalue()


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("Model configuration")
    st.write("**Active model path:**")
    st.code(MODEL_PATH, language="text")
    st.markdown("----")
    st.subheader("Model performance")
    st.markdown("""
        **mAP@50:** 0.9823  
        **mAP@50–95:** 0.5598  
        **Precision:** 0.9714  
        **Recall:** 0.9765
    """)
    st.markdown("----")
    st.write("Batch ID (used for exports)")
    batch_id = st.text_input("Batch ID", value=f"BATCH_{time.strftime('%Y%m%d_%H%M%S')}")
    st.write("Processing chunk size")
    pchunk = st.number_input("Chunk size", min_value=1, max_value=64, value=CHUNK_PROCESS_SIZE)
    st.write("Confidence threshold")
    conf_in = st.slider("Confidence", 0.0, 1.0, float(CONFIDENCE), 0.01)

# ------------------ MAIN LAYOUT ------------------
st.markdown(
    """
    <div class="header-container">
        <div class="logo-circle">🛡️</div>
        <div class="main-title">CircuitGuard – PCB Defect Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metrics row (kept similar)
metric_cols = st.columns(4)
metric_info = [
    ("mAP@50", "0.9823"),
    ("mAP@50–95", "0.5598"),
    ("Precision", "0.9714"),
    ("Recall", "0.9765"),
]
for col, (label, value) in zip(metric_cols, metric_info):
    with col:
        st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown(
    """
    <p class="subtitle-text">
    Detect and highlight <strong>PCB defects</strong> such as missing hole, mouse bite,
    open circuit, short, spur and spurious copper using a YOLO-based deep learning model.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="instruction-card">
      <strong>🧭 How to use CircuitGuard:</strong>
      <ol>
        <li>Prepare clear PCB images (top view, good lighting).</li>
        <li>Upload one or more images using the box below.</li>
        <li>Wait for the model to run – we’ll generate annotated results.</li>
        <li>Review the overview grid, then scroll to see before/after views for each image.</li>
        <li>Download individual annotated images or a ZIP with CSV + all annotated outputs.</li>
      </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    **Defect types detected by this model:**
    <div class="defect-badges">
      <span class="defect-badge">Missing hole</span>
      <span class="defect-badge">Mouse bite</span>
      <span class="defect-badge">Open circuit</span>
      <span class="defect-badge">Short</span>
      <span class="defect-badge">Spur</span>
      <span class="defect-badge">Spurious copper</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Upload PCB Images")

with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more PCB images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ DETECTION & DISPLAY (QUEUED, CHUNKED) ------------------
if uploaded_files:
    # append to session queue
    added = 0
    for f in uploaded_files:
        try:
            pil = Image.open(f).convert("RGB")
            st.session_state["images"].append({
                "filename": f.name,
                "original": pil,
                "annotated": None,
                "defects": [],
                "processed": False,
                "processing_error": None,
                "batch_id": batch_id,
                "model_version": os.path.basename(MODEL_PATH),
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            added += 1
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    st.success(f"Queued {added} images for processing. Go to 'Process' to start.")

# Process controls
st.markdown("### Processing controls")
proc_col1, proc_col2, proc_col3 = st.columns([1, 1, 1])
with proc_col1:
    if st.button("Process unprocessed images"):
        model = load_model(MODEL_PATH)
        names = getattr(model, "names", {}) or {}
        to_process = [img for img in st.session_state["images"] if not img["processed"]]
        total = len(to_process)
        if total == 0:
            st.info("No unprocessed images in queue.")
        else:
            st.info(f"Processing {total} images in chunks of {pchunk}...")
            for i in range(0, total, int(pchunk)):
                chunk = to_process[i:i+int(pchunk)]
                for img in chunk:
                    try:
                        plotted, result = run_inference(model, img["original"])  # plotted PIL
                        # annotate defects (pack metadata as dicts)
                        defects = pack_defects_from_result(result)
                        # map class ids to names if available
                        if names:
                            for d, box_cls in zip(defects, result.boxes.cls.tolist()):
                                d["defect_type"] = names.get(int(box_cls), str(box_cls))
                        img["annotated"] = plotted
                        img["defects"] = defects
                        img["processed"] = True
                        img["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        img["processing_error"] = str(e)
                        img["processed"] = True
                # small yield to allow UI update
                st.experimental_rerun()

with proc_col2:
    if st.button("Generate CSVs for queue"):
        processed = [img for img in st.session_state["images"] if img.get("processed")]
        if not processed:
            st.warning("No processed images yet.")
        else:
            b1, b2 = generate_csvs(processed)
            if b1:
                st.download_button("Download full_results.csv", data=b1, file_name=f"{batch_id}_full_results.csv", mime="text/csv")
            if b2:
                st.download_button("Download summary_results.csv", data=b2, file_name=f"{batch_id}_summary_results.csv", mime="text/csv")

with proc_col3:
    if st.button("Download ZIP (PDFs + CSV + PNGs) for processed"):
        processed = [img for img in st.session_state["images"] if img.get("processed")]
        if not processed:
            st.warning("No processed images yet.")
        else:
            z = make_zip_for_selection(processed, include_pngs=True)
            st.download_button("Download processed_bundle.zip", data=z, file_name=f"{batch_id}_bundle.zip", mime="application/zip")

# ------------------ Pagination + Thumbnail grid ------------------
n_images = len(st.session_state["images"])
st.markdown(f"**Images in queue:** {n_images}")
total_pages = max(1, math.ceil(n_images / THUMBNAILS_PER_PAGE))
page = st.session_state.get("page", 1)
pg_col1, pg_col2, pg_col3 = st.columns([1, 1, 1])
with pg_col1:
    if st.button("<< Prev Page"):
        st.session_state["page"] = max(1, st.session_state.get("page", 1) - 1)
with pg_col2:
    st.write(f"Page {st.session_state.get('page',1)} / {total_pages}")
with pg_col3:
    if st.button("Next Page >>"):
        st.session_state["page"] = min(total_pages, st.session_state.get("page", 1) + 1)

start = (page - 1) * THUMBNAILS_PER_PAGE
end = start + THUMBNAILS_PER_PAGE
visible = st.session_state["images"][start:end]
cols = st.columns(4)
for i, img in enumerate(visible):
    col = cols[i % 4]
    with col:
        thumb = img["original"].copy()
        thumb.thumbnail((360, 240))
        b = io.BytesIO()
        thumb.save(b, format="PNG")
        st.image(b.getvalue(), caption=img["filename"], use_column_width=True)
        st.caption((f"Defects: {len(img.get('defects', []))} • Processed: {img.get('processed')}"))
        with st.expander("Details & actions"):
            c1, c2 = st.columns(2)
            with c1:
                st.image(img["original"], caption="Original", use_column_width=True)
            with c2:
                if img.get("annotated") is not None:
                    st.image(img["annotated"], caption="Annotated", use_column_width=True)
                    # annotated download
                    b2 = io.BytesIO()
                    img["annotated"].save(b2, format="PNG")
                    b2.seek(0)
                    st.download_button("Download annotated PNG", data=b2, file_name=f"annotated_{os.path.splitext(img['filename'])[0]}.png", mime="image/png", key=f"dlpng_{start+i}")
                else:
                    st.info("Not annotated yet.")
            if img.get("defects"):
                df = pd.DataFrame(img["defects"])
                st.dataframe(df)
            # actions
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("Download PDF", key=f"pdf_{start+i}"):
                    pdf = generate_pdf_for_image(img["original"], img.get("annotated", img["original"]), img.get("defects", []), {
                        "project_name": "CircuitGuard",
                        "batch_id": img.get("batch_id", ""),
                        "filename": img.get("filename", ""),
                        "processed_at": img.get("processed_at", ""),
                        "model_version": img.get("model_version", ""),
                    })
                    st.download_button("Click to download PDF", data=pdf, file_name=f"{os.path.splitext(img['filename'])[0]}_result.pdf", mime="application/pdf", key=f"pdfbtn_{start+i}")
            with a2:
                if st.button("Select for ZIP", key=f"select_{start+i}"):
                    img["selected_for_zip"] = not img.get("selected_for_zip", False)
                    st.experimental_rerun()
            with a3:
                if st.button("Re-run detection", key=f"rerun_{start+i}"):
                    model = load_model(MODEL_PATH)
                    try:
                        plotted, result = run_inference(model, img["original"])
                        defects = pack_defects_from_result(result)
                        names = getattr(model, "names", {}) or {}
                        if names:
                            for d, box_cls in zip(defects, result.boxes.cls.tolist()):
                                d["defect_type"] = names.get(int(box_cls), str(box_cls))
                        img["annotated"] = plotted
                        img["defects"] = defects
                        img["processed"] = True
                        img["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        st.success("Re-run complete")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Re-run failed: {e}")

# Selected ZIP
selected = [img for img in st.session_state["images"] if img.get("selected_for_zip")]
if selected:
    st.markdown(f"**Selected for ZIP:** {len(selected)} images")
    if st.button("Download ZIP for selection"):
        z = make_zip_for_selection(selected, include_pngs=True)
        st.download_button("Download selected_bundle.zip", data=z, file_name=f"{batch_id}_selected_bundle.zip", mime="application/zip")

# Overall charts (aggregate counts)
all_processed = [img for img in st.session_state["images"] if img.get("processed")]
if all_processed:
    global_counts = Counter()
    for img in all_processed:
        for d in img.get("defects", []):
            global_counts.update([d.get("defect_type", "unknown")])
    if sum(global_counts.values()) > 0:
        st.subheader("Overall defect distribution across processed images")
        global_df = pd.DataFrame({"Defect Type": list(global_counts.keys()), "Count": list(global_counts.values())})
        bar_chart = (
            alt.Chart(global_df)
            .mark_bar(size=45)
            .encode(x=alt.X("Defect Type:N", sort="-y", axis=alt.Axis(labelAngle=0)), y=alt.Y("Count:Q"), tooltip=["Defect Type", "Count"]) 
            .properties(height=260)
        )
        st.altair_chart(bar_chart, use_container_width=True)

# Export area for all processed
if all_processed:
    if st.button("Generate CSVs & ZIP for all processed"):
        b1, b2 = generate_csvs(all_processed)
        z = make_zip_for_selection(all_processed, include_pngs=True)
        if b1:
            st.download_button("Download full CSV", data=b1, file_name=f"{batch_id}_full_results.csv", mime="text/csv")
        if b2:
            st.download_button("Download summary CSV", data=b2, file_name=f"{batch_id}_summary.csv", mime="text/csv")
        st.download_button("Download ZIP bundle", data=z, file_name=f"{batch_id}_bundle.zip", mime="application/zip")

st.markdown("---")
st.write("Notes: For large-scale production, move heavy PDF/image composition to background workers and persist outputs to object storage (S3). Use WebSockets for live progress updates and pagination/virtualization on the frontend for better UX.")

# End of file
