# app.py
# CircuitGuard — UI with scrollable results and search/filter panel

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

# ReportLab (preferred). If missing we fallback to PIL generator.
HAS_REPORTLAB = True
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
except Exception:
    HAS_REPORTLAB = False

# ------------------ CONFIG ------------------
LOCAL_MODEL_PATH = r"C:\Users\asus\OneDrive\Desktop\yolo deploy\best.pt"
CLOUD_MODEL_PATH = "best.pt"
MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else CLOUD_MODEL_PATH

CONFIDENCE = 0.25
IOU = 0.45

st.set_page_config(page_title="CircuitGuard – PCB Defect Detection", page_icon="🛡️", layout="wide")

# ------------------ CSS: center header, lighter browse + download buttons (restore uploader preview) ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');
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

    .header-container {
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        margin-top:0.5rem;
        margin-bottom:0.75rem;
        text-align:center;
    }
    .logo-circle { width: 64px; height:64px; border-radius:50%; background:#e0f2fe; display:flex; align-items:center; justify-content:center; font-size:32px; margin-bottom:0.25rem; }
    .main-title { font-family:'Bitcount Prop Single', 'Space Grotesk', 'Poppins', system-ui, -apple-system, sans-serif; font-weight:700; font-size:2.6rem; color:#13406b; }

    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        border: none;
        font-weight: 600;
        background: linear-gradient(90deg, #e6f0ff 0%, #d2eaff 100%);
        color: #04293a;
        box-shadow: 0 8px 14px rgba(148,163,184,0.18);
    }
    .stButton>button:hover { transform: translateY(-2px); }

    [data-testid="stDownloadButton"] > button {
        background: #e6f3ff !important;
        color: #04293a !important;
        border-radius: 999px !important;
        border: 1px solid #d0eaff !important;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] button {
        background: #e6f0ff !important;
        color: #04293a !important;
        border-radius: 10px !important;
        padding: 6px 12px !important;
        border: 1px solid #cfe1ff !important;
    }
    [data-testid="stFileUploader"] label { color: #0b3a57 !important; }

    .metric-card { border-radius: 18px; padding: 0.75rem 1rem; background: #ffffff; border: 1px solid #dbeafe; }
    .metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.1rem; }
    .metric-value { font-size: 1.15rem; font-weight: 600; color: #111827; }

    .instruction-card { border-radius: 18px; background: #ffffff; border: 1px solid #dbeafe; padding:1rem 1.25rem; margin:1rem 0; font-size:0.9rem; }
    .defect-badge { padding:0.2rem 0.6rem; border-radius:999px; background:#e0f2fe; color:#13406b; font-size:0.8rem; margin-right:0.4rem; }

    .result-row { background:#ffffff; border:1px solid #eef6ff; border-radius:12px; padding:12px 16px; margin-bottom:12px; box-shadow: 0 6px 16px rgba(14,30,37,0.02); }
    .image-name-btn { background:transparent; border:none; color:#0b3a57; font-weight:600; font-size:1rem; text-align:left; cursor:pointer; }
    .image-name-btn:hover { text-decoration: underline; }
.results-block {
    max-height: 340px;              /* shows ~5–6 rows */
    overflow-y: scroll;             /* force scrollbar */
    overflow-x: hidden;
    padding: 14px;
    background: #f8fbff;
    border-radius: 14px;
    border: 1px solid #dbeafe;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
    margin-top: 12px;

    /* Firefox */
    scrollbar-width: thin;
    scrollbar-color: #93c5fd #f1f5f9;
}

/* Chrome / Edge / Brave */
.results-block::-webkit-scrollbar {
    width: 10px;
}

.results-block::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 10px;
}

.results-block::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #c7e5ff, #93c5fd);
    border-radius: 10px;
}



/* visible scrollbar */
.results-block::-webkit-scrollbar {
    width: 10px;
}
.results-block::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #dbeafe, #bfdbfe);
    border-radius: 8px;
}


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

# ------------------ PDF generator using ReportLab (improved table), PIL fallback if needed ------------------
def generate_pdf_with_reportlab(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()
    ps_left = ParagraphStyle("left", parent=styles["Normal"], alignment=TA_LEFT, fontSize=8)
    ps_right = ParagraphStyle("right", parent=styles["Normal"], alignment=TA_RIGHT, fontSize=8)
    ps_header = ParagraphStyle("hdr", parent=styles["Heading2"], alignment=TA_CENTER, fontSize=12)

    story.append(Paragraph(f"{meta.get('project_name','CircuitGuard')}", ps_header))
    story.append(Spacer(1,6))
    story.append(Paragraph(f"{meta.get('filename','')} — Processed: {meta.get('processed_at','')}  |  Model: {meta.get('model_version','')}", styles["Normal"]))
    story.append(Spacer(1,12))

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
    story.append(img_table)
    story.append(Spacer(1,12))

    headers = ["id", "type", "x", "y", "w", "h", "center_x", "center_y", "confidence"]
    table_rows = []
    header_row = []
    for h in headers:
        header_row.append(Paragraph(str(h), ParagraphStyle("hdrcell", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, textColor=colors.white, backColor=colors.HexColor("#0f172a"))))
    table_rows.append(header_row)

    for i, d in enumerate(defects):
        try:
            x1 = float(d.get("x1", d.get("x", 0) or 0))
            y1 = float(d.get("y1", d.get("y", 0) or 0))
            x2 = float(d.get("x2", 0) or 0)
            y2 = float(d.get("y2", 0) or 0)
            w = round(x2 - x1, 1) if x2 and x1 else ""
            h = round(y2 - y1, 1) if y2 and y1 else ""
            cx = round((x1 + x2) / 2, 1) if x1 and x2 else ""
            cy = round((y1 + y2) / 2, 1) if y1 and y2 else ""
        except Exception:
            x1 = d.get("x1", d.get("x", ""))
            y1 = d.get("y1", d.get("y", ""))
            w = d.get("width", "")
            h = d.get("height", "")
            cx = d.get("center_x", "")
            cy = d.get("center_y", "")

        row = [
            Paragraph(str(i+1), ps_right),
            Paragraph(str(d.get("defect_type", d.get("Defect type", ""))), ps_left),
            Paragraph(f"{round(float(x1),1) if isinstance(x1,(int,float)) else x1}", ps_right),
            Paragraph(f"{round(float(y1),1) if isinstance(y1,(int,float)) else y1}", ps_right),
            Paragraph(f"{w}", ps_right),
            Paragraph(f"{h}", ps_right),
            Paragraph(f"{cx}", ps_right),
            Paragraph(f"{cy}", ps_right),
            Paragraph(f"{round(float(d.get('confidence', d.get('Confidence', 0))), 3) if d.get('confidence', d.get('Confidence', None)) is not None else ''}", ps_right),
        ]
        table_rows.append(row)

    colWidths = [30, 130, 60, 60, 50, 50, 65, 65, 60]
    defect_table = Table(table_rows, colWidths=colWidths, repeatRows=1)
    defect_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (0,-1), "RIGHT"),
        ("ALIGN", (2,0), (-1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("Detected defects (id, type, x, y, w, h, center_x, center_y, confidence)", styles["Heading4"]))
    story.append(Spacer(1,6))
    story.append(defect_table)
    story.append(Spacer(1,12))
    story.append(Paragraph(f"Generated by CircuitGuard • {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def generate_pdf_with_pil(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    margin = 40
    spacing = 20
    canvas_w = 2480
    canvas_h = 1754
    target_img_w = (canvas_w - margin * 3) // 2
    def scale_to(pil_img, max_w, max_h):
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        return pil_img.resize((int(iw * scale), int(ih * scale)))
    max_img_h = int(canvas_h * 0.45)
    orig_s = scale_to(original_pil, target_img_w, max_img_h)
    ann_s = scale_to(annotated_pil, target_img_w, max_img_h)
    content_h = max(orig_s.height, ann_s.height)
    line_h = 28
    table_h = max(120, line_h * (len(defects) + 2))
    total_h = margin + content_h + spacing + table_h + margin
    if total_h > canvas_h:
        canvas_h = total_h + 20
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    x1 = margin
    y1 = margin
    canvas.paste(orig_s, (x1, y1))
    canvas.paste(ann_s, (x1 + target_img_w + margin, y1))
    try:
        font_h = ImageFont.truetype("arial.ttf", 28)
        font_m = ImageFont.truetype("arial.ttf", 14)
        font_t = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_h = ImageFont.load_default()
        font_m = ImageFont.load_default()
        font_t = ImageFont.load_default()
    draw.text((margin, margin + content_h + 6), f"{meta.get('project_name','CircuitGuard')} — {meta.get('filename','')}", fill=(16,42,67), font=font_m)
    draw.text((margin, margin + content_h + 26), f"Processed: {meta.get('processed_at','')}  |  Model: {meta.get('model_version','')}", fill=(80,100,120), font=font_t)
    table_x = margin
    table_y = margin + content_h + 56
    cols = ["id","type","x","y","w","h","cx","cy","conf"]
    col_w = [40, 220, 90, 90, 70, 70, 90, 90, 80]
    draw.rectangle([table_x-6, table_y-6, table_x + sum(col_w) + 6, table_y + line_h + 6], fill=(16,26,38))
    x = table_x
    for i,c in enumerate(cols):
        draw.text((x+4, table_y+4), c, fill=(255,255,255), font=font_t)
        x += col_w[i]
    y = table_y + line_h
    for i,d in enumerate(defects):
        x = table_x
        try:
            x1v = d.get("x1", "")
            y1v = d.get("y1", "")
            x2v = d.get("x2", "")
            y2v = d.get("y2", "")
            wv = (float(x2v) - float(x1v)) if x2v and x1v else ""
            hv = (float(y2v) - float(y1v)) if y2v and y1v else ""
            cx = (float(x1v)+float(x2v))/2 if x1v and x2v else ""
            cy = (float(y1v)+float(y2v))/2 if y1v and y2v else ""
        except Exception:
            wv=hv=cx=cy=""
        row = [str(i+1), str(d.get("Defect type", d.get("defect_type",""))),
               f"{x1v}", f"{y1v}", f"{round(wv,1) if isinstance(wv,(int,float)) else wv}",
               f"{round(hv,1) if isinstance(hv,(int,float)) else hv}",
               f"{round(cx,1) if isinstance(cx,(int,float)) else cx}", f"{round(cy,1) if isinstance(cy,(int,float)) else cy}",
               f"{round(float(d.get('Confidence', d.get('confidence',0))),3) if d.get('Confidence', d.get('confidence',None)) is not None else ''}"]
        for j,val in enumerate(row):
            draw.text((x+4, y+6), str(val), fill=(10,20,30), font=font_t)
            x += col_w[j]
        y += line_h
        if y > table_y + table_h - line_h:
            break
    draw.text((margin, canvas_h - 20), f"Generated by CircuitGuard • {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(90,110,125), font=font_t)
    out_buf = io.BytesIO()
    canvas.save(out_buf, "PDF", resolution=150.0)
    out_buf.seek(0)
    return out_buf.read()

def generate_pdf_for_image(original_pil: Image.Image, annotated_pil: Image.Image, defects: List[Dict], meta: Dict) -> bytes:
    if HAS_REPORTLAB:
        try:
            return generate_pdf_with_reportlab(original_pil, annotated_pil, defects, meta)
        except Exception:
            return generate_pdf_with_pil(original_pil, annotated_pil, defects, meta)
    else:
        return generate_pdf_with_pil(original_pil, annotated_pil, defects, meta)

# ------------------ CSV helpers / ZIP builder ------------------
def generate_csvs(images_list: List[Dict]) -> (bytes, bytes):
    full_rows = []
    summary_rows = []
    for img in images_list:
        filename = img["name"]
        batch_id = img.get("batch_id","")
        model_version = img.get("model_version","")
        defects = img.get("loc_rows",[])
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
            "max_confidence": max([d.get("Confidence",0) for d in defects], default=0),
            "model_version": model_version
        })
    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    b1 = full_df.to_csv(index=False).encode("utf-8") if not full_df.empty else b""
    b2 = summary_df.to_csv(index=False).encode("utf-8") if not summary_df.empty else b""
    return b1, b2

def make_zip_with_pdfs_and_csv(images_list: List[Dict], batch_id: str="BATCH") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        full_csv, summary_csv = generate_csvs(images_list)
        if full_csv:
            zf.writestr("circuitguard_detection_results.csv", full_csv)
        if summary_csv:
            zf.writestr("circuitguard_detection_summary.csv", summary_csv)

        for img in images_list:
            safe = os.path.splitext(img["name"])[0]
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
                    "x1": r.get("x1"),
                    "y1": r.get("y1"),
                    "x2": r.get("x2"),
                    "y2": r.get("y2"),
                    "width": w,
                    "height": h,
                    "center_x": cx,
                    "center_y": cy,
                    "Confidence": r.get("Confidence")
                })
            try:
                pdf_bytes = generate_pdf_for_image(img["original"], img["annotated"], defects_for_pdf, {
                    "project_name": "CircuitGuard",
                    "batch_id": batch_id,
                    "filename": img["name"],
                    "processed_at": img.get("processed_at",""),
                    "model_version": img.get("model_version","")
                })
                zf.writestr(f"{safe}_report.pdf", pdf_bytes)
            except Exception:
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

st.markdown("<p class='subtitle-text'>Detect and highlight <strong>PCB defects</strong> such as missing hole, mouse bite, open circuit, short, spur and spurious copper using a YOLO-based deep learning model.</p>", unsafe_allow_html=True)

st.markdown("""<div class="instruction-card"><strong>🧭 How to use CircuitGuard:</strong>
<ol><li>Prepare clear PCB images (top view, good lighting).</li><li>Upload images using the box below.</li><li>Click image name in the results table to view details inline.</li><li>When done click <strong>Finish defect detection</strong> to download PDFs + CSV as a ZIP.</li></ol></div>""", unsafe_allow_html=True)

st.markdown("<div class='defect-badges'><span class='defect-badge'>Missing hole</span> <span class='defect-badge'>Mouse bite</span> <span class='defect-badge'>Open circuit</span> <span class='defect-badge'>Short</span> <span class='defect-badge'>Spur</span> <span class='defect-badge'>Spurious copper</span></div>", unsafe_allow_html=True)

st.markdown("### Upload PCB Images")
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload images (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True, label_visibility="collapsed")
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
        total = len(uploaded_files)
        status_text = st.empty()
        progress = st.progress(0)
        status_text.info(f"Processing 0/{total} images...")
        global_counts = Counter()
        all_rows = []
        image_results: List[Dict] = []

        for i, file in enumerate(uploaded_files, start=1):
            status_text.info(f"Processing {i}/{total} images — {file.name}")
            progress.progress(int((i-1)/total * 100))
            img = Image.open(file).convert("RGB")
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
        progress.progress(100)
        status_text.success(f"Processing complete — {total} images processed.")
        time.sleep(0.2)
        status_text.empty()

        if all_rows:
            st.session_state["full_results_df"] = pd.DataFrame(all_rows)
            st.session_state["annotated_images"] = [(r["name"], r["annotated"]) for r in image_results]
        else:
            st.session_state["full_results_df"] = None
            st.session_state["annotated_images"] = []

        st.markdown('<div class="robot-success"><span class="robot-label">[SYSTEM]</span> DEFECT SCAN COMPLETE — ANALYSIS DASHBOARD ONLINE.</div>', unsafe_allow_html=True)

        # Build summary list for filtering
        summary_list = []
        for i, res in enumerate(image_results):
            defect_count = len(res["loc_rows"])
            max_conf = max([r["Confidence"] for r in res["loc_rows"]], default=0.0)
            defect_types = sorted({r.get("Defect type") for r in res["loc_rows"] if r.get("Defect type")})
            summary_list.append({
                "idx": i,
                "name": res["name"],
                "defect_count": defect_count,
                "max_confidence": round(max_conf, 3),
                "processed_at": res.get("processed_at",""),
                "defect_types": defect_types,
                "res_obj": res
            })

        # Search & filters UI
        st.markdown("### Results — summary table (click image name to toggle details)")
        with st.container():
            f1, f2, f3 = st.columns([2,2,2])
            query = f1.text_input("Search (image, defect name, number...)", value="", placeholder="e.g. missing_hole or 3 or 0.7")
            field = f2.selectbox("Search field", options=["All", "Image", "Defect type", "Defect count", "Max confidence", "Processed at"], index=0)
            all_defect_types = sorted(list({dt for s in summary_list for dt in s["defect_types"] if dt}))
            selected_defect_types = f3.multiselect("Filter defect types", options=all_defect_types, default=[])

        def matches_entry(entry, query_text, field_choice, selected_types):
            q = (query_text or "").strip().lower()
            if selected_types:
                if not any(dt in selected_types for dt in entry["defect_types"]):
                    return False
            if not q:
                return True
            try:
                q_num_int = int(q)
            except Exception:
                q_num_int = None
            try:
                q_num = float(q)
            except Exception:
                q_num = None

            if field_choice == "All":
                if q in entry["name"].lower():
                    return True
                if any(q in (dt or "").lower() for dt in entry["defect_types"]):
                    return True
                if q_num_int is not None and entry["defect_count"] == q_num_int:
                    return True
                if q_num is not None and abs(entry["max_confidence"] - q_num) < 1e-6:
                    return True
                if q in (entry["processed_at"] or "").lower():
                    return True
                return False
            if field_choice == "Image":
                return q in entry["name"].lower()
            if field_choice == "Defect type":
                return any(q in (dt or "").lower() for dt in entry["defect_types"])
            if field_choice == "Defect count":
                if q_num_int is not None:
                    return entry["defect_count"] == q_num_int
                return str(entry["defect_count"]) == q
            if field_choice == "Max confidence":
                if q_num is not None:
                    return abs(entry["max_confidence"] - q_num) < 1e-6
                return q in str(entry["max_confidence"])
            if field_choice == "Processed at":
                return q in (entry["processed_at"] or "").lower()
            return False

        filtered = [s for s in summary_list if matches_entry(s, query, field, selected_defect_types)]
        st.markdown(f"**Showing {len(filtered)} of {len(summary_list)} images**")

        # Scrollable results block
        st.markdown('<div class="results-block">', unsafe_allow_html=True)
        for s in filtered:
            idx = s["idx"]
            res = s["res_obj"]
            defect_count = s["defect_count"]
            max_conf = s["max_confidence"]
            st.markdown('<div class="result-row">', unsafe_allow_html=True)
            row_cols = st.columns([4,1,1,2])
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
                st.markdown("---")

        st.markdown('</div>', unsafe_allow_html=True)

        # Charts + export (unchanged)
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

        st.markdown("### Export results")
        if st.button("Finish defect detection"):
            images_for_export = []
            for img in image_results:
                images_for_export.append({
                    "name": img["name"],
                    "original": img["original"],
                    "annotated": img["annotated"],
                    "loc_rows": img["loc_rows"],
                    "processed_at": img.get("processed_at",""),
                    "model_version": img.get("model_version",""),
                    "batch_id": img.get("batch_id", f"BATCH_{time.strftime('%Y%m%d')}")
                })
            if not HAS_REPORTLAB:
                st.info("ReportLab not installed — using PIL fallback to produce PDFs. Install reportlab for the cleanest PDFs.")
            with st.spinner("Generating ZIP (PDFs + CSV). This may take a moment for many images..."):
                zip_bytes = make_zip_with_pdfs_and_csv(images_for_export, batch_id=f"BATCH_{time.strftime('%Y%m%d')}")
            st.download_button("Download results (PDFs + CSV, ZIP)", data=zip_bytes, file_name=f"circuitguard_results_{time.strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")
else:
    st.info("Upload one or more PCB images to start detection.")
