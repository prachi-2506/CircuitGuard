# app.py
# CircuitGuard — Updated UI: restored typography, lighter buttons, table-row toggles,
# removed per-row action buttons, added single "Finish defect detection" ZIP export
# containing per-image PDFs (original + annotated + defect table) and the CSV.

import os
import io
import zipfile
import time
from collections import Counter
from typing import List, Dict

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import altair as alt

# Try importing reportlab for PDF generation
HAS_REPORTLAB = True
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except Exception:
    HAS_REPORTLAB = False

# ------------------ CONFIG ------------------
LOCAL_MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\yolo deploy\best.pt"
CLOUD_MODEL_PATH = "best.pt"
MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else CLOUD_MODEL_PATH

CONFIDENCE = 0.25
IOU = 0.45

st.set_page_config(
    page_title="CircuitGuard – PCB Defect Detection",
    page_icon="🛡️",
    layout="wide"
)

# ------------------ RESTORED CSS (typography + lighter buttons) ------------------
st.markdown(
    """
    <style>
    /* Restored fonts & original look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bitcount+Prop+Single:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f8fbff;
        font-family: 'Poppins', sans-serif;
        color: #102a43;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5ff 0%, #e7fff7 100%);
        border-right: 1px solid #d0e2ff;
    }
    [data-testid="stSidebar"] * { color: #102a43 !important; }
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code { background: #e5e7eb !important; color: #111827 !important; }

    h1, h2, h3, h4, h5 {
        font-family: 'Space Grotesk', 'Poppins', system-ui, -apple-system, sans-serif;
        color: #13406b;
        font-weight: 600;
    }

    /* Make primary buttons lighter so text is readable */
    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        border: none;
        font-weight: 600;
        background: #e6f0ff; /* light base */
        color: #04293a;      /* dark readable text */
        box-shadow: 0 8px 14px rgba(148, 163, 184, 0.18);
        transition: transform 0.12s ease-out, box-shadow 0.12s ease-out;
    }
    .stButton>button:hover {
        background: #c9e1ff;
        transform: translateY(-2px);
    }

    /* Download buttons (explicitly light) */
    [data-testid="stDownloadButton"] > button {
        background: #f1f5fb !important;
        color: #04293a !important;
        border-radius: 999px !important;
        border: 1px solid #dbeafe !important;
        font-weight: 600;
    }

    .upload-box {
        border-radius: 18px;
        border: 1px dashed #a3c9ff;
        padding: 1.5rem;
        background: #ffffff;
    }

    .metric-card {
        border-radius: 18px;
        padding: 0.75rem 1rem;
        background: #ffffff;
        border: 1px solid #dbeafe;
    }

    .metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.1rem; }
    .metric-value { font-size: 1.15rem; font-weight: 600; color: #111827; }

    /* Result row card look (lighter table style) */
    .result-row {
        background: #ffffff;
        border: 1px solid #eef6ff;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 6px 16px rgba(14, 30, 37, 0.02);
    }
    .image-name-btn {
        background: transparent;
        border: none;
        color: #0b3a57;
        font-weight: 600;
        font-size: 1rem;
        text-align: left;
        cursor: pointer;
    }
    .image-name-btn:hover { text-decoration: underline; }

    .cell-small { color:#334e68; font-size:0.95rem; }

    .status-strip {
        margin: 0.5rem 0 1.2rem 0;
        padding: 0.65rem 1.1rem;
        border-radius: 999px;
        background: #dff6ea;
        color: #064e3b;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .robot-success { margin: 1rem 0 0.4rem 0; padding: 0.8rem 1.2rem; border-radius: 12px; background: linear-gradient(90deg, #0f172a 0%, #1f2937 55%, #16a34a 100%); color: #e5f9ff; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; letter-spacing: 0.09em; text-transform: uppercase; }
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
if "open_row_idx" not in st.session_state:
    st.session_state["open_row_idx"] = None

# ------------------ MODEL LOADING & INFERENCE ------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def run_inference(model, image):
    results = model.predict(image, conf=CONFIDENCE, iou=IOU)
    r = results[0]
    plotted = r.plot()
    plotted = plotted[:, :, ::-1]
    pil_img = Image.fromarray(plotted)
    return pil_img, r


def get_class_counts(result, class_names):
    if len(getattr(result, "boxes", [])) == 0:
        return {}
    cls_indices = result.boxes.cls.tolist()
    labels = [class_names[int(i)] for i in cls_indices]
    counts = Counter(labels)
    return dict(counts)


def get_defect_locations(result, class_names, image_name):
    if len(getattr(result, "boxes", [])) == 0:
        return []
    boxes = result.boxes
    xyxy = boxes.xyxy.tolist()
    cls_indices = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    rows = []
    for coords, c, cf in zip(xyxy, cls_indices, confs):
        x1, y1, x2, y2 = coords
        rows.append({
            "Image": image_name,
            "Defect type": class_names[int(c)] if isinstance(class_names, dict) and int(c) in class_names else str(c),
            "Confidence": round(float(cf), 2),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
        })
    return rows

# Full PDF generator (per-image). Uses reportlab.
def generate_pdf_for_image(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed; cannot generate PDF.")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    page_w, page_h = landscape(A4)

    header_text = f"{meta.get('project_name', 'CircuitGuard')} — {meta.get('batch_id','')} — {meta.get('filename','')}"
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, page_h - 30, header_text)
    c.setFont("Helvetica", 8)
    c.drawString(30, page_h - 45, f"Processed: {meta.get('processed_at', '')}    Model: {meta.get('model_version','')}")

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
        for i, d in enumerate(defects):
            if y < 40:
                c.showPage()
                y = page_h - 40
            row = [str(i+1), str(d.get("defect_type","")), str(d.get("x","")), str(d.get("y","")), str(d.get("width","")), str(d.get("height","")), str(d.get("center_x","")), str(d.get("center_y","")), f"{d.get('confidence',0):.3f}"]
            for item, xpos in zip(row, x_positions):
                c.drawString(xpos, y, item)
            y -= row_h

    c.setFont("Helvetica", 7)
    c.drawString(30, 20, f"Generated by CircuitGuard • {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()
    buf.seek(0)
    return buf.read()

# CSV generator for full results and summary (same structure as earlier)
def generate_csvs(images_list: List[Dict]) -> (bytes, bytes):
    full_rows = []
    summary_rows = []
    for img in images_list:
        filename = img["name"]
        batch_id = img.get("batch_id", "")
        model_version = img.get("model_version", "")
        defects = img.get("loc_rows", [])
        for d in defects:
            full_rows.append({
                "batch_id": batch_id,
                "image_filename": filename,
                "defect_type": d.get("Defect type"),
                "confidence": d.get("Confidence"),
                "x1": d.get("x1"),
                "y1": d.get("y1"),
                "x2": d.get("x2"),
                "y2": d.get("y2"),
                "model_version": model_version,
            })
        summary_rows.append({
            "batch_id": batch_id,
            "image_filename": filename,
            "defect_count": len(defects),
            "max_confidence": max([d.get("Confidence", 0) for d in defects], default=0),
            "model_version": model_version,
        })
    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    b1 = full_df.to_csv(index=False).encode("utf-8") if not full_df.empty else b""
    b2 = summary_df.to_csv(index=False).encode("utf-8") if not summary_df.empty else b""
    return b1, b2

def make_zip_with_pdfs_and_csv(images_list: List[Dict], batch_id: str = "BATCH") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        full_csv, summary_csv = generate_csvs(images_list)
        if full_csv:
            zf.writestr("circuitguard_detection_results.csv", full_csv)
        if summary_csv:
            zf.writestr("circuitguard_detection_summary.csv", summary_csv)
        for img in images_list:
            safe = os.path.splitext(img["name"])[0]
            # generate PDF for each image (full report)
            try:
                pdf = generate_pdf_for_image(img["original"], img["annotated"], [
                    {
                        "defect_type": r.get("Defect type"),
                        "x": r.get("x1"),
                        "y": r.get("y1"),
                        "width": float(r.get("x2")) - float(r.get("x1")) if r.get("x2") and r.get("x1") else "",
                        "height": float(r.get("y2")) - float(r.get("y1")) if r.get("y2") and r.get("y1") else "",
                        "center_x": (float(r.get("x1")) + float(r.get("x2"))) / 2 if r.get("x1") and r.get("x2") else "",
                        "center_y": (float(r.get("y1")) + float(r.get("y2"))) / 2 if r.get("y1") and r.get("y2") else "",
                        "confidence": r.get("Confidence")
                    } for r in img.get("loc_rows", [])
                ], {
                    "project_name": "CircuitGuard",
                    "batch_id": batch_id,
                    "filename": img["name"],
                    "processed_at": img.get("processed_at", ""),
                    "model_version": img.get("model_version", "")
                })
                zf.writestr(f"{safe}_report.pdf", pdf)
            except Exception as e:
                # If PDF can't be made (reportlab missing), skip PDFs but continue
                pass
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
    st.markdown(
        """
        **mAP@50:** 0.9823  
        **mAP@50–95:** 0.5598  
        **Precision:** 0.9714  
        **Recall:** 0.9765
        """
    )

# ------------------ MAIN UI ------------------
st.markdown(
    """
    <div class="header-container">
        <div class="logo-circle">🛡️</div>
        <div class="main-title">CircuitGuard – PCB Defect Detection</div>
    </div>
    """, unsafe_allow_html=True
)

# Metrics
metric_cols = st.columns(4)
metric_info = [
    ("mAP@50", "0.9823"),
    ("mAP@50–95", "0.5598"),
    ("Precision", "0.9714"),
    ("Recall", "0.9765"),
]
for col, (label, value) in zip(metric_cols, metric_info):
    with col:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>""",
            unsafe_allow_html=True,
        )

st.markdown("""
<p class="subtitle-text">
Detect and highlight <strong>PCB defects</strong> such as missing hole, mouse bite,
open circuit, short, spur and spurious copper using a YOLO-based deep learning model.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="instruction-card">
  <strong>🧭 How to use CircuitGuard:</strong>
  <ol>
    <li>Prepare clear PCB images (top view, good lighting).</li>
    <li>Upload one or more images using the box below.</li>
    <li>Wait for the model to run – we’ll generate annotated results.</li>
    <li>In the results table below click the image name to toggle details inline.</li>
    <li>When done, click <strong>Finish defect detection</strong> to download PDF reports + CSV as a ZIP.</li>
  </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("**Defect types detected by this model:**")
st.markdown("<div class='defect-badges'><span class='defect-badge'>Missing hole</span> <span class='defect-badge'>Mouse bite</span> <span class='defect-badge'>Open circuit</span> <span class='defect-badge'>Short</span> <span class='defect-badge'>Spur</span> <span class='defect-badge'>Spurious copper</span></div>", unsafe_allow_html=True)

st.markdown("### Upload PCB Images")
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload one or more PCB images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ DETECTION & RESULTS (table with inline details) ------------------
if uploaded_files:
    try:
        model = load_model(MODEL_PATH)
        class_names = model.names
    except Exception as e:
        st.error(f"Error loading model from `{MODEL_PATH}`: {e}")
        model = None
        class_names = {}
    else:
        global_counts = Counter()
        all_rows = []
        image_results: List[Dict] = []

        # Run detections
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            with st.spinner(f"Running detection on {file.name}..."):
                plotted_img, result = run_inference(model, img)

            counts = get_class_counts(result, class_names)
            global_counts.update(counts)

            loc_rows = get_defect_locations(result, class_names, file.name)
            all_rows.extend(loc_rows)

            image_results.append({
                "name": file.name,
                "original": img,
                "annotated": plotted_img,
                "result": result,
                "loc_rows": loc_rows,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_version": os.path.basename(MODEL_PATH),
                "batch_id": f"BATCH_{time.strftime('%Y%m%d')}"
            })

        # Build full results DF for export (all images)
        if all_rows:
            full_results_df = pd.DataFrame(all_rows)
            st.session_state["full_results_df"] = full_results_df
            st.session_state["annotated_images"] = [(res["name"], res["annotated"]) for res in image_results]
        else:
            st.session_state["full_results_df"] = None
            st.session_state["annotated_images"] = []

        # Success banner and status
        st.markdown('<div class="robot-success"><span class="robot-label">[SYSTEM]</span> DEFECT SCAN COMPLETE — ANALYSIS DASHBOARD ONLINE.</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-strip">Detection complete. Click any image name in the results table to view details inline.</div>', unsafe_allow_html=True)

        # Results header (table-like)
        st.markdown("### Results — summary table (click image name to toggle details)")
        header_cols = st.columns([4, 1, 1, 2])
        header_cols[0].markdown("**Image**")
        header_cols[1].markdown("**Defects**")
        header_cols[2].markdown("**Max confidence**")
        header_cols[3].markdown("**Processed at**")

        # Render rows
        for idx, res in enumerate(image_results):
            defect_count = len(res["loc_rows"])
            max_conf = 0.0
            if res["loc_rows"]:
                max_conf = max([r["Confidence"] for r in res["loc_rows"]])

            st.markdown('<div class="result-row">', unsafe_allow_html=True)
            row_cols = st.columns([4, 1, 1, 2])
            # image name button toggles the row index
            btn_key = f"img_row_btn_{idx}_{res['name']}"
            if row_cols[0].button(res["name"], key=btn_key):
                if st.session_state.get("open_row_idx") == idx:
                    st.session_state["open_row_idx"] = None
                else:
                    st.session_state["open_row_idx"] = idx
            row_cols[1].markdown(f"<div class='cell-small'>{defect_count}</div>", unsafe_allow_html=True)
            row_cols[2].markdown(f"<div class='cell-small'>{round(max_conf,2)}</div>", unsafe_allow_html=True)
            row_cols[3].markdown(f"<div class='cell-small'>{res.get('processed_at','')}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Inline details if open
            if st.session_state.get("open_row_idx") == idx:
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Original image")
                    buf_o = io.BytesIO(); res["original"].save(buf_o, format="PNG"); buf_o.seek(0)
                    st.image(buf_o.getvalue(), use_column_width=True)
                with c2:
                    st.subheader("Annotated image")
                    buf_a = io.BytesIO(); res["annotated"].save(buf_a, format="PNG"); buf_a.seek(0)
                    st.image(buf_a.getvalue(), use_column_width=True)

                if res["loc_rows"]:
                    loc_df = pd.DataFrame(res["loc_rows"])
                    st.markdown("**Defect locations (bounding boxes in pixels):**")
                    st.dataframe(loc_df.drop(columns=["Image"]), use_container_width=True)
                else:
                    st.info("No defects detected in this image.")

                # Small helper: indicate that per-row buttons were removed and exports are consolidated
                st.info("Per-image actions removed. Use 'Finish defect detection' below to download per-image PDFs + CSV in a ZIP.")
                st.markdown("---")

        # Overall charts
        if sum(global_counts.values()) > 0:
            st.subheader("Overall defect distribution across all uploaded images")
            global_df = pd.DataFrame({"Defect Type": list(global_counts.keys()), "Count": list(global_counts.values())})
            bar_chart = (alt.Chart(global_df).mark_bar(size=45)
                         .encode(x=alt.X("Defect Type:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                                 y=alt.Y("Count:Q"),
                                 tooltip=["Defect Type", "Count"])
                         .properties(height=260))
            st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("#### Defect type share")
            donut_chart = (alt.Chart(global_df).mark_arc(innerRadius=55, outerRadius=100)
                           .encode(theta=alt.Theta("Count:Q", stack=True),
                                   color=alt.Color("Defect Type:N", legend=alt.Legend(title="Defect type")),
                                   tooltip=["Defect Type", "Count"])
                           .properties(height=260))
            st.altair_chart(donut_chart, use_container_width=True)
        else:
            st.info("No defects detected in any of the uploaded images.")

        # -------- Export: consolidated Finish button (creates ZIP with PDFs+CSV) --------
        st.markdown("### Export results")
        if st.button("Finish defect detection"):
            # Build list to pass to zip maker: include original PIL, annotated PIL, loc_rows and metadata
            images_for_export = []
            for img in image_results:
                images_for_export.append({
                    "name": img["name"],
                    "original": img["original"],
                    "annotated": img["annotated"],
                    "loc_rows": img["loc_rows"],
                    "processed_at": img.get("processed_at",""),
                    "model_version": img.get("model_version",""),
                    "batch_id": img.get("batch_id","BATCH"),
                })

            if not HAS_REPORTLAB:
                st.warning("reportlab not installed — ZIP will include the CSV but PDFs cannot be generated. Install reportlab to enable PDFs.")
            zip_bytes = make_zip_with_pdfs_and_csv(images_for_export, batch_id=f"BATCH_{time.strftime('%Y%m%d')}")
            st.download_button("Download results (PDFs + CSV, ZIP)", data=zip_bytes, file_name="circuitguard_results.zip", mime="application/zip")

else:
    st.info("Upload one or more PCB images to start detection.")
