# app.py
# CircuitGuard — UI tweaks: remove per-image message, ensure PDFs in ZIP (reportlab OR PIL fallback),
# lighter browse button text, set main heading font to Bitcount Prop Double Ink (fallbacks included).

import os
import io
import zipfile
import time
from collections import Counter
from typing import List, Dict

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import altair as alt

# Optional: ReportLab (preferred for nicely formatted tables in PDFs)
HAS_REPORTLAB = True
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    HAS_REPORTLAB = False

# ------------------ CONFIG ------------------
LOCAL_MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\yolo deploy\best.pt"
CLOUD_MODEL_PATH = "best.pt"
MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else CLOUD_MODEL_PATH

CONFIDENCE = 0.25
IOU = 0.45

st.set_page_config(page_title="CircuitGuard – PCB Defect Detection", page_icon="🛡️", layout="wide")

# ------------------ CSS (restored look + lighter browse button + heading font) ------------------
st.markdown(
    """
    <style>
    /* Fonts: prefer Bitcount Prop Double Ink, fallback to Space Grotesk/Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');
    /* If a custom Bitcount Prop Double Ink is available via your hosting, it will be used here.
       If not available, browser falls back gracefully. */
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

    /* Main heading now prefers Bitcount Prop Double Ink (if available) */
    .main-title {
        font-family: 'Bitcount Prop Double Ink', 'Bitcount Prop Single', 'Space Grotesk', 'Poppins', system-ui, -apple-system, sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        color: #13406b;
        letter-spacing: 0.02em;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', 'Poppins', system-ui, -apple-system, sans-serif;
        color: #13406b;
        font-weight: 600;
    }

    /* Buttons: lighter background so text is readable */
    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        border: none;
        font-weight: 600;
        background: linear-gradient(90deg, #e6f0ff 0%, #d2eaff 100%);
        color: #04293a;
        box-shadow: 0 8px 14px rgba(148, 163, 184, 0.18);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }

    /* Make download buttons readable */
    [data-testid="stDownloadButton"] > button {
        background: #f1f5fb !important;
        color: #04293a !important;
        border-radius: 999px !important;
        border: 1px solid #dbeafe !important;
        font-weight: 600;
    }

    /* File uploader: make the Browse files button lighter so label is readable */
    [data-testid="stFileUploader"] button {
        background: #e6f0ff !important;
        color: #04293a !important;
        border-radius: 10px !important;
        padding: 6px 12px !important;
        border: 1px solid #cfe1ff !important;
    }
    [data-testid="stFileUploader"] label { color: #0b3a57 !important; }

    .upload-box { border-radius: 18px; border: 1px dashed #a3c9ff; padding: 1.5rem; background: #ffffff; }

    .metric-card { border-radius: 18px; padding: 0.75rem 1rem; background: #ffffff; border: 1px solid #dbeafe; }
    .metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.1rem; }
    .metric-value { font-size: 1.15rem; font-weight: 600; color: #111827; }

    .logo-circle { width: 60px; height: 60px; border-radius: 50%; background: #e0f2fe; display:flex; align-items:center; justify-content:center; font-size:32px; margin-bottom:0.4rem; }

    .instruction-card { border-radius: 18px; background: #ffffff; border: 1px solid #dbeafe; padding:1rem 1.25rem; margin:1rem 0; font-size:0.9rem; }
    .defect-badges { display:flex; gap:0.4rem; margin-top:0.4rem; }
    .defect-badge { padding:0.2rem 0.6rem; border-radius:999px; background:#e0f2fe; color:#13406b; font-size:0.8rem; }

    .robot-success { margin:1rem 0 0.4rem 0; padding:0.8rem 1.2rem; border-radius:12px; background: linear-gradient(90deg,#0f172a 0%,#1f2937 55%,#16a34a 100%); color:#e5f9ff; font-family:'JetBrains Mono', monospace; font-size:0.9rem; letter-spacing:0.09em; text-transform:uppercase; }
    .status-strip { margin:0.5rem 0 1.2rem 0; padding:0.65rem 1.1rem; border-radius:999px; background:#dff6ea; color:#064e3b; font-size:0.95rem; font-weight:600; }

    .result-row { background:#ffffff; border:1px solid #eef6ff; border-radius:12px; padding:12px 16px; margin-bottom:12px; box-shadow: 0 6px 16px rgba(14,30,37,0.02); }
    .image-name-btn { background:transparent; border:none; color:#0b3a57; font-weight:600; font-size:1rem; text-align:left; cursor:pointer; }
    .image-name-btn:hover { text-decoration: underline; }
    .cell-small { color:#334e68; font-size:0.95rem; }

    .vega-embed, .vega-embed canvas { max-width:100% !important; }
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

# === PDF generator: prefer ReportLab (clean table), fallback to PIL if not available ===
def generate_pdf_with_reportlab(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()
    header = Paragraph(f"<b>{meta.get('project_name','CircuitGuard')}</b> — {meta.get('filename','')}", styles["Heading2"])
    sub = Paragraph(f"Processed: {meta.get('processed_at','')}  |  Model: {meta.get('model_version','')}", styles["Normal"])
    story.extend([header, Spacer(1, 6), sub, Spacer(1, 12)])

    # images side-by-side
    page_w, page_h = landscape(A4)
    max_img_h = page_h * 0.45
    max_img_w = (page_w - 80) / 2

    def prepare_rl_image(pil_img, max_w, max_h):
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        iw2, ih2 = int(iw * scale), int(ih * scale)
        bio = io.BytesIO()
        pil_img.resize((iw2, ih2)).save(bio, format="PNG")
        bio.seek(0)
        return RLImage(ImageReader(bio), width=iw2, height=ih2)

    rl_orig = prepare_rl_image(original_pil, max_img_w, max_img_h)
    rl_ann = prepare_rl_image(annotated_pil, max_img_w, max_img_h)
    img_table = Table([[rl_orig, rl_ann]], colWidths=[max_img_w, max_img_w])
    img_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 0)]))
    story.extend([img_table, Spacer(1, 12)])

    # defect table
    headers = ["id", "type", "x", "y", "w", "h", "center_x", "center_y", "confidence"]
    rows = [headers]
    for i, d in enumerate(defects):
        w = d.get("width") if "width" in d else (float(d.get("x2", 0)) - float(d.get("x1", 0)) if d.get("x1") is not None and d.get("x2") is not None else "")
        h = d.get("height") if "height" in d else (float(d.get("y2", 0)) - float(d.get("y1", 0)) if d.get("y1") is not None and d.get("y2") is not None else "")
        cx = d.get("center_x") if "center_x" in d else ((float(d.get("x1", 0)) + float(d.get("x2", 0))) / 2 if d.get("x1") is not None and d.get("x2") is not None else "")
        cy = d.get("center_y") if "center_y" in d else ((float(d.get("y1", 0)) + float(d.get("y2", 0))) / 2 if d.get("y1") is not None and d.get("y2") is not None else "")
        rows.append([
            str(i+1),
            str(d.get("defect_type", d.get("Defect type", ""))),
            f"{d.get('x', d.get('x1', ''))}",
            f"{d.get('y', d.get('y1', ''))}",
            f"{w}",
            f"{h}",
            f"{cx}",
            f"{cy}",
            f"{d.get('confidence', d.get('Confidence', ''))}"
        ])

    colWidths = [30, 120, 60, 60, 50, 50, 60, 60, 60]
    defect_table = Table(rows, colWidths=colWidths, repeatRows=1)
    defect_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("Detected defects (id, type, x, y, w, h, center_x, center_y, confidence)", getSampleStyleSheet()["Heading4"]))
    story.append(Spacer(1, 6))
    story.append(defect_table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated by CircuitGuard • {time.strftime('%Y-%m-%d %H:%M:%S')}", getSampleStyleSheet()["Normal"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def generate_pdf_with_pil(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    # Create a single-page PDF using PIL: place images side-by-side and draw a textual table below.
    # This is a robust fallback when reportlab is not installed.
    # Canvas sizing
    margin = 40
    spacing = 20
    max_width = 2480  # approx A4 landscape at 300dpi ~ 3508x2480; keep below to be safe
    max_height = 1754
    # scale images to fit half of width
    target_img_w = (max_width - 3 * margin) // 2
    # scale originals
    def scale_to(pil_img, max_w, max_h):
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        return pil_img.resize((int(iw * scale), int(ih * scale)))
    # choose max image height ~40% of canvas
    max_img_h = int(max_height * 0.45)
    orig_s = scale_to(original_pil, target_img_w, max_img_h)
    ann_s = scale_to(annotated_pil, target_img_w, max_img_h)
    content_h = max(orig_s.height, ann_s.height)
    # compute height for table area: a line per defect
    line_h = 26
    table_h = max(120, line_h * (len(defects) + 2))
    canvas_h = margin + content_h + spacing + table_h + margin
    canvas_w = max_width
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    # paste images
    x1 = margin
    y1 = margin
    canvas.paste(orig_s, (x1, y1))
    x2 = margin + target_img_w + margin
    y2 = margin
    canvas.paste(ann_s, (x2, y2))
    # Draw header text
    try:
        # try to use a system font for nicer look, fallback to default
        font_h = ImageFont.truetype("arial.ttf", 28)
        font_m = ImageFont.truetype("arial.ttf", 14)
        font_t = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_h = ImageFont.load_default()
        font_m = ImageFont.load_default()
        font_t = ImageFont.load_default()
    header_text = f"{meta.get('project_name','CircuitGuard')}  —  {meta.get('filename','')}"
    draw.text((margin, canvas_h - table_h - spacing - 40), header_text, fill=(16, 42, 67), font=font_h)
    sub_text = f"Processed: {meta.get('processed_at','')}   Model: {meta.get('model_version','')}"
    draw.text((margin, canvas_h - table_h - spacing - 14), sub_text, fill=(64, 102, 130), font=font_m)
    # Draw table header
    table_x = margin
    table_y = canvas_h - table_h + 10
    cols = ["id", "type", "x", "y", "w", "h", "cx", "cy", "conf"]
    col_w = [50, 220, 90, 90, 70, 70, 90, 90, 80]
    # Draw column titles
    x = table_x
    y = table_y
    draw.rectangle([table_x - 6, table_y - 6, table_x + sum(col_w) + 6, table_y + line_h + 6], outline=(225, 232, 240), fill=(245, 248, 250))
    for i, c in enumerate(cols):
        draw.text((x + 4, y + 4), c, fill=(10, 20, 30), font=font_t)
        x += col_w[i]
    # rows
    y += line_h
    for i, d in enumerate(defects):
        x = table_x
        try:
            w = float(d.get("x2", 0)) - float(d.get("x1", 0))
            h = float(d.get("y2", 0)) - float(d.get("y1", 0))
            cx = (float(d.get("x1", 0)) + float(d.get("x2", 0))) / 2
            cy = (float(d.get("y1", 0)) + float(d.get("y2", 0))) / 2
        except Exception:
            w = h = cx = cy = ""
        row_values = [
            str(i + 1),
            str(d.get("Defect type", d.get("defect_type", ""))),
            str(d.get("x1", d.get("x", ""))),
            str(d.get("y1", d.get("y", ""))),
            f"{w}",
            f"{h}",
            f"{cx}",
            f"{cy}",
            f"{d.get('Confidence', d.get('confidence', ''))}"
        ]
        for j, v in enumerate(row_values):
            draw.text((x + 4, y + 4), v, fill=(10, 20, 30), font=font_t)
            x += col_w[j]
        y += line_h
        # avoid overflow — stop if we exceed area
        if y > table_y + table_h - line_h:
            break
    # Footer
    draw.text((margin, canvas_h - 18), f"Generated by CircuitGuard • {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(90, 110, 125), font=font_t)
    # Save as PDF (single page)
    out_buf = io.BytesIO()
    try:
        canvas.save(out_buf, "PDF", resolution=150.0)
    except Exception:
        # fallback to saving PNG (rare), but still return bytes as PDF-like
        tmp = io.BytesIO()
        canvas.save(tmp, format="PNG")
        tmp.seek(0)
        # create a minimal PDF wrapper by writing everything as an image — but PIL save to PDF above should work in most envs
        out_buf = tmp
    out_buf.seek(0)
    return out_buf.read()

def generate_pdf_for_image(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    # prefer ReportLab; fallback to PIL-based PDF
    if HAS_REPORTLAB:
        try:
            return generate_pdf_with_reportlab(original_pil, annotated_pil, defects, meta)
        except Exception:
            # fallback if reportlab failed for any reason
            return generate_pdf_with_pil(original_pil, annotated_pil, defects, meta)
    else:
        return generate_pdf_with_pil(original_pil, annotated_pil, defects, meta)

# CSV helpers
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
                "defect_type": d.get("Defect type") or d.get("defect_type"),
                "confidence": d.get("Confidence") or d.get("confidence"),
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
            # Build a defects list for PDF
            defects_for_pdf = []
            for r in img.get("loc_rows", []):
                try:
                    w = float(r.get("x2")) - float(r.get("x1"))
                    h = float(r.get("y2")) - float(r.get("y1"))
                    cx = (float(r.get("x1")) + float(r.get("x2"))) / 2
                    cy = (float(r.get("y1")) + float(r.get("y2"))) / 2
                except Exception:
                    w = h = cx = cy = ""
                defects_for_pdf.append({
                    "defect_type": r.get("Defect type"),
                    "x": r.get("x1"),
                    "y": r.get("y1"),
                    "width": w,
                    "height": h,
                    "center_x": cx,
                    "center_y": cy,
                    "confidence": r.get("Confidence"),
                    "x1": r.get("x1"),
                    "x2": r.get("x2"),
                    "y1": r.get("y1"),
                    "y2": r.get("y2"),
                })
            try:
                pdf_bytes = generate_pdf_for_image(img["original"], img["annotated"], defects_for_pdf, {
                    "project_name": "CircuitGuard",
                    "batch_id": batch_id,
                    "filename": img["name"],
                    "processed_at": img.get("processed_at", ""),
                    "model_version": img.get("model_version", "")
                })
                zf.writestr(f"{safe}_report.pdf", pdf_bytes)
            except Exception:
                # if PDF generation fails for a particular file, skip it but continue building the rest of the ZIP
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
    st.markdown("**mAP@50:** 0.9823  \n**mAP@50–95:** 0.5598  \n**Precision:** 0.9714  \n**Recall:** 0.9765")

# ------------------ MAIN UI ------------------
st.markdown("""
<div class="header-container">
    <div class="logo-circle">🛡️</div>
    <div class="main-title">CircuitGuard – PCB Defect Detection</div>
</div>
""", unsafe_allow_html=True)

metric_cols = st.columns(4)
metric_info = [("mAP@50", "0.9823"), ("mAP@50–95", "0.5598"), ("Precision", "0.9714"), ("Recall", "0.9765")]
for col, (label, value) in zip(metric_cols, metric_info):
    with col:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

st.markdown("""<p class="subtitle-text">Detect and highlight <strong>PCB defects</strong> such as missing hole, mouse bite, open circuit, short, spur and spurious copper using a YOLO-based deep learning model.</p>""", unsafe_allow_html=True)

st.markdown("""<div class="instruction-card"><strong>🧭 How to use CircuitGuard:</strong><ol><li>Prepare clear PCB images (top view, good lighting).</li><li>Upload one or more images using the box below.</li><li>Wait for the model to run – we’ll generate annotated results.</li><li>In the results table below click the image name to toggle details inline.</li><li>When done, click <strong>Finish defect detection</strong> to download PDF reports + CSV as a ZIP.</li></ol></div>""", unsafe_allow_html=True)

st.markdown("<div class='defect-badges'><span class='defect-badge'>Missing hole</span> <span class='defect-badge'>Mouse bite</span> <span class='defect-badge'>Open circuit</span> <span class='defect-badge'>Short</span> <span class='defect-badge'>Spur</span> <span class='defect-badge'>Spurious copper</span></div>", unsafe_allow_html=True)

st.markdown("### Upload PCB Images")
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload one or more PCB images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PROCESSING & RESULTS ------------------
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

        # Build export DF
        if all_rows:
            st.session_state["full_results_df"] = pd.DataFrame(all_rows)
            st.session_state["annotated_images"] = [(res["name"], res["annotated"]) for res in image_results]
        else:
            st.session_state["full_results_df"] = None
            st.session_state["annotated_images"] = []

        st.markdown('<div class="robot-success"><span class="robot-label">[SYSTEM]</span> DEFECT SCAN COMPLETE — ANALYSIS DASHBOARD ONLINE.</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-strip">Detection complete. Click any image name in the results table to view details inline.</div>', unsafe_allow_html=True)

        # Results header (table-like)
        st.markdown("### Results — summary table (click image name to toggle details)")
        header_cols = st.columns([4, 1, 1, 2])
        header_cols[0].markdown("**Image**")
        header_cols[1].markdown("**Defects**")
        header_cols[2].markdown("**Max confidence**")
        header_cols[3].markdown("**Processed at**")

        # Rows
        for idx, res in enumerate(image_results):
            defect_count = len(res["loc_rows"])
            max_conf = max([r["Confidence"] for r in res["loc_rows"]], default=0.0)
            st.markdown('<div class="result-row">', unsafe_allow_html=True)
            row_cols = st.columns([4, 1, 1, 2])
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

            if st.session_state.get("open_row_idx") == idx:
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Original image")
                    b_o = io.BytesIO(); res["original"].save(b_o, format="PNG"); b_o.seek(0)
                    st.image(b_o.getvalue(), use_column_width=True)
                with c2:
                    st.subheader("Annotated image")
                    b_a = io.BytesIO(); res["annotated"].save(b_a, format="PNG"); b_a.seek(0)
                    st.image(b_a.getvalue(), use_column_width=True)
                if res["loc_rows"]:
                    st.markdown("**Defect locations (bounding boxes in pixels):**")
                    st.dataframe(pd.DataFrame(res["loc_rows"]).drop(columns=["Image"]), use_container_width=True)
                else:
                    st.info("No defects detected in this image.")
                # per-image info message removed as requested
                st.markdown("---")

        # Charts
        if sum(global_counts.values()) > 0:
            st.subheader("Overall defect distribution across all uploaded images")
            global_df = pd.DataFrame({"Defect Type": list(global_counts.keys()), "Count": list(global_counts.values())})
            bar_chart = (alt.Chart(global_df).mark_bar(size=45).encode(x=alt.X("Defect Type:N", sort="-y", axis=alt.Axis(labelAngle=0)), y=alt.Y("Count:Q"), tooltip=["Defect Type", "Count"]).properties(height=260))
            st.altair_chart(bar_chart, use_container_width=True)
            st.markdown("#### Defect type share")
            donut_chart = (alt.Chart(global_df).mark_arc(innerRadius=55, outerRadius=100).encode(theta=alt.Theta("Count:Q", stack=True), color=alt.Color("Defect Type:N", legend=alt.Legend(title="Defect type")), tooltip=["Defect Type", "Count"]).properties(height=260))
            st.altair_chart(donut_chart, use_container_width=True)
        else:
            st.info("No defects detected in any of the uploaded images.")

        # Export: Finish detection -> ZIP with PDFs + CSV
        st.markdown("### Export results")
        if st.button("Finish defect detection"):
            images_for_export = []
            for img in image_results:
                images_for_export.append({
                    "name": img["name"],
                    "original": img["original"],
                    "annotated": img["annotated"],
                    "loc_rows": img["loc_rows"],
                    "processed_at": img.get("processed_at", ""),
                    "model_version": img.get("model_version", ""),
                    "batch_id": img.get("batch_id", f"BATCH_{time.strftime('%Y%m%d')}")
                })
            if not HAS_REPORTLAB:
                st.info("reportlab not installed — using a PIL fallback to produce PDFs. ReportLab will give cleaner table layout if installed.")
            with st.spinner("Generating ZIP (PDFs + CSV). This may take a moment for many images..."):
                zip_bytes = make_zip_with_pdfs_and_csv(images_for_export, batch_id=f"BATCH_{time.strftime('%Y%m%d')}")
            st.download_button("Download results (PDFs + CSV, ZIP)", data=zip_bytes, file_name=f"circuitguard_results_{time.strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")
else:
    st.info("Upload one or more PCB images to start detection.")
