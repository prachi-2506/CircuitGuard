# app.py
# CircuitGuard — UI updated to show results as a list/table with collapsible details
import os
import io
import zipfile
import time
from collections import Counter

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import altair as alt

# Try optional reportlab imports (PDF generation). If missing, show message but do not crash UI.
HAS_REPORTLAB = True
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except Exception as _:
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

# ------------------ CUSTOM STYLING ------------------
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

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5ff 0%, #e7fff7 100%);
        border-right: 1px solid #d0e2ff;
    }

    [data-testid="stSidebar"] * {
        color: #102a43 !important;
    }

    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code {
        background: #e5e7eb !important;
        color: #111827 !important;
    }

    [data-testid="stToolbar"] * {
        color: #e5e7eb !important;
    }

    h2, h3 {
        font-weight: 600;
        color: #13406b;
        font-family: 'Space Grotesk', 'Poppins', system-ui, -apple-system, sans-serif;
    }

    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.25rem;
        border: none;
        font-weight: 500;
        background: #85c5ff;
        color: #0f172a;
        box-shadow: 0 8px 14px rgba(148, 163, 184, 0.28);
        transition: transform 0.18s ease-out, box-shadow 0.18s ease-out, background 0.18s ease-out;
        animation: pulse-soft 2.4s ease-in-out infinite;
    }

    .stButton>button:hover {
        background: #63b1ff;
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 12px 22px rgba(148, 163, 184, 0.38);
    }

    @keyframes pulse-soft {
        0% {
            transform: translateY(0);
            box-shadow: 0 8px 14px rgba(148, 163, 184, 0.25);
        }
        50% {
            transform: translateY(-1px);
            box-shadow: 0 12px 22px rgba(148, 163, 184, 0.4);
        }
        100% {
            transform: translateY(0);
            box-shadow: 0 8px 14px rgba(148, 163, 184, 0.25);
        }
    }

    [data-testid="stDownloadButton"] > button {
        background: #e5e7eb !important;
        color: #111827 !important;
        border-radius: 999px !important;
        border: 1px solid #cbd5f5 !important;
        font-weight: 500;
    }

    .upload-box {
        border-radius: 18px;
        border: 1px dashed #a3c9ff;
        padding: 1.5rem;
        background: #ffffff;
    }

    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label {
        color: #f9fafb !important;
    }
    [data-testid="stFileUploader"] button {
        background: #111827 !important;
        color: #f9fafb !important;
        border-radius: 999px !important;
        border: none !important;
    }

    .metric-card {
        border-radius: 18px;
        padding: 0.75rem 1rem;
        background: #ffffff;
        border: 1px solid #dbeafe;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 0.1rem;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        font-family: 'Space Grotesk', 'Poppins', system-ui, sans-serif;
    }

    .logo-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: #e0f2fe;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        margin-bottom: 0.4rem;
    }

    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .main-title {
        font-family: 'Space Grotesk', system-ui, -apple-system,
                     BlinkMacSystemFont, 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 2.8rem;
        text-align: center;
        color: #13406b;
        letter-spacing: 0.03em;
    }

    .subtitle-text {
        font-size: 0.95rem;
        color: #334e68;
        text-align: center;
    }

    .instruction-card {
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #dbeafe;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .instruction-card ol {
        margin-left: 1.1rem;
        padding-left: 0.5rem;
    }
    .instruction-card li {
        margin-bottom: 0.25rem;
    }

    .defect-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.4rem;
    }
    .defect-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: #e0f2fe;
        font-size: 0.8rem;
        color: #13406b;
    }

    .robot-success {
        margin: 1rem 0 0.4rem 0;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #0f172a 0%, #1f2937 55%, #16a34a 100%);
        color: #e5f9ff;
        font-family: 'JetBrains Mono', SFMono-Regular, Menlo, monospace;
        font-size: 0.9rem;
        letter-spacing: 0.09em;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
    }
    .robot-success::after {
        content: "";
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(
            0deg,
            rgba(148, 163, 184, 0.0),
            rgba(148, 163, 184, 0.0) 2px,
            rgba(148, 163, 184, 0.25) 3px
        );
        mix-blend-mode: soft-light;
        opacity: 0.4;
        pointer-events: none;
        animation: scanlines 6s linear infinite;
    }
    @keyframes scanlines {
        0% { transform: translateY(-3px); }
        100% { transform: translateY(3px); }
    }
    .robot-label {
        color: #a7f3d0;
        margin-right: 0.75rem;
    }

    .status-strip {
        margin: 0.1rem 0 1.2rem 0;
        padding: 0.65rem 1.1rem;
        border-radius: 999px;
        background: #d1fae5;
        color: #064e3b;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .vega-embed, .vega-embed canvas {
        max-width: 100% !important;
    }
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

# ------------------ MODEL LOADING & INFERENCE ------------------
@st.cache_resource
def load_model(path: str):
    """Load YOLO model once and cache it."""
    return YOLO(path)


def run_inference(model, image):
    """Run detection and return plotted image + raw result."""
    results = model.predict(image, conf=CONFIDENCE, iou=IOU)
    r = results[0]
    plotted = r.plot()  # BGR numpy array
    plotted = plotted[:, :, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(plotted)
    return pil_img, r


def get_class_counts(result, class_names):
    """Return a dict: {class_name: count} for one result."""
    if len(getattr(result, "boxes", [])) == 0:
        return {}
    cls_indices = result.boxes.cls.tolist()
    labels = [class_names[int(i)] for i in cls_indices]
    counts = Counter(labels)
    return dict(counts)


def get_defect_locations(result, class_names, image_name):
    """Return rows with defect type, confidence and bounding box coords + image name."""
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

# Top metrics row with custom cards
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
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <p class="subtitle-text">
    Detect and highlight <strong>PCB defects</strong> such as missing hole, mouse bite,
    open circuit, short, spur and spurious copper using a YOLO-based deep learning model.
    </p>
    """,
    unsafe_allow_html=True,
)

# Instructions card
st.markdown(
    """
    <div class="instruction-card">
      <strong>🧭 How to use CircuitGuard:</strong>
      <ol>
        <li>Prepare clear PCB images (top view, good lighting).</li>
        <li>Upload one or more images using the box below.</li>
        <li>Wait for the model to run – we’ll generate annotated results.</li>
        <li>Review the results list below; click an image name to open its details.</li>
        <li>Download individual annotated images or a ZIP with CSV + all annotated outputs.</li>
      </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# Highlight defect types
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

# ------------------ DETECTION & RESULTS LIST ------------------
if uploaded_files:
    try:
        model = load_model(MODEL_PATH)
        class_names = model.names  # dict: {id: name}
    except Exception as e:
        st.error(f"Error loading model from `{MODEL_PATH}`: {e}")
        model = None
        class_names = {}
    else:
        global_counts = Counter()
        all_rows = []
        image_results = []  # list of dicts: name, original, annotated, result, loc_rows

        # Run detection for all images
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")

            with st.spinner(f"Running detection on {file.name}..."):
                plotted_img, result = run_inference(model, img)

            counts = get_class_counts(result, class_names)
            global_counts.update(counts)

            loc_rows = get_defect_locations(result, class_names, file.name)
            all_rows.extend(loc_rows)

            image_results.append(
                {
                    "name": file.name,
                    "original": img,
                    "annotated": plotted_img,
                    "result": result,
                    "loc_rows": loc_rows,
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        # Build full results DF for export (all images)
        if all_rows:
            full_results_df = pd.DataFrame(all_rows)
            st.session_state["full_results_df"] = full_results_df
            st.session_state["annotated_images"] = [
                (res["name"], res["annotated"]) for res in image_results
            ]
        else:
            st.session_state["full_results_df"] = None
            st.session_state["annotated_images"] = []

        # Robotic animated success banner + clear info strip
        st.markdown(
            """
            <div class="robot-success">
              <span class="robot-label">[SYSTEM]</span>
              DEFECT SCAN COMPLETE — ANALYSIS DASHBOARD ONLINE.
            </div>
            <div class="status-strip">
              Detection complete. Click any image in the results list to view details.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- NEW: Results list (table) ----------
        st.markdown("### Results — summary list")
        # Prepare summary rows: one row per image
        summary_rows = []
        for res in image_results:
            defect_count = len(res["loc_rows"])
            max_conf = 0.0
            if res["loc_rows"]:
                max_conf = max([r["Confidence"] for r in res["loc_rows"]])
            summary_rows.append({
                "Image": res["name"],
                "Defects": defect_count,
                "Max confidence": round(max_conf, 2),
                "Processed at": res.get("processed_at", ""),
            })

        summary_df = pd.DataFrame(summary_rows)
        # show a compact dataframe with row selection (user can click expanders below)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("### Click an image name to expand details")
        # For each image create a collapsible expander keyed by filename
        for idx, res in enumerate(image_results):
            key = f"expander_{idx}_{res['name']}"
            with st.expander(f"{res['name']} — {len(res['loc_rows'])} defects", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original image")
                    buf_o = io.BytesIO()
                    res["original"].save(buf_o, format="PNG")
                    buf_o.seek(0)
                    st.image(buf_o.getvalue(), use_column_width=True)
                with col2:
                    st.subheader("Annotated image")
                    buf_a = io.BytesIO()
                    res["annotated"].save(buf_a, format="PNG")
                    buf_a.seek(0)
                    st.image(buf_a.getvalue(), use_column_width=True)

                    # downloads for single image
                    base = os.path.splitext(res["name"])[0]
                    st.download_button(
                        f"Download annotated image — {res['name']}",
                        data=buf_a.getvalue(),
                        file_name=f"annotated_{base}.png",
                        mime="image/png",
                        key=f"dl_img_{idx}"
                    )

                # defect table
                if res["loc_rows"]:
                    loc_df = pd.DataFrame(res["loc_rows"])
                    st.markdown("**Defect locations (bounding boxes in pixels):**")
                    st.dataframe(loc_df.drop(columns=["Image"]), use_container_width=True)
                else:
                    st.info("No defects detected in this image.")

                # Action buttons
                a1, a2, a3 = st.columns(3)
                with a1:
                    if st.button(f"Download PDF — {res['name']}", key=f"pdf_{idx}"):
                        if not HAS_REPORTLAB:
                            st.error("reportlab not installed — PDF generation not available in this environment.")
                        else:
                            # build a single-image PDF (keeps same format as before)
                            meta = {
                                "project_name": "CircuitGuard",
                                "batch_id": "",
                                "filename": res["name"],
                                "processed_at": res.get("processed_at", ""),
                                "model_version": "",
                            }
                            # simple PDF with annotated + original and table; re-use your previous generation if you have it
                            # For minimal change, create a tiny PDF here:
                            from reportlab.lib.pagesizes import A4, landscape
                            from reportlab.lib.utils import ImageReader
                            from reportlab.pdfgen import canvas

                            pdf_buf = io.BytesIO()
                            c = canvas.Canvas(pdf_buf, pagesize=landscape(A4))
                            page_w, page_h = landscape(A4)
                            # place images side-by-side
                            left_img = ImageReader(res["original"])
                            right_img = ImageReader(res["annotated"])
                            # scale both to same height
                            ih = res["original"].height
                            iw = res["original"].width
                            scale = min((page_h - 200) / ih, (page_w/2 - 60) / iw, 1.0)
                            w2 = iw * scale
                            h2 = ih * scale
                            c.drawImage(left_img, 30, page_h - 60 - h2, width=w2, height=h2)
                            c.drawImage(right_img, 50 + page_w/2, page_h - 60 - h2, width=w2, height=h2)
                            # small table header
                            c.setFont("Helvetica-Bold", 10)
                            c.drawString(30, 40, f"Defects for {res['name']}")
                            c.showPage()
                            c.save()
                            pdf_buf.seek(0)
                            st.download_button(
                                f"Download PDF file — {res['name']}",
                                data=pdf_buf.getvalue(),
                                file_name=f"{base}_result.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{idx}"
                            )
                with a2:
                    if st.button(f"Re-run detection — {res['name']}", key=f"rerun_{idx}"):
                        # rerun inference on this image
                        try:
                            model = load_model(MODEL_PATH)
                            plotted_img, result = run_inference(model, res["original"])
                            loc_rows = get_defect_locations(result, model.names, res["name"])
                            # update res in place
                            res["annotated"] = plotted_img
                            res["result"] = result
                            res["loc_rows"] = loc_rows
                            res["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            # refresh UI
                            st.success("Re-run complete for " + res["name"])
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Re-run failed: {e}")
                with a3:
                    # include this image in annotated_images for bundle if user wants
                    if st.button(f"Add to ZIP selection — {res['name']}", key=f"selectzip_{idx}"):
                        # append annotated to session list if not present
                        existing = [n for n, _ in st.session_state.get("annotated_images", [])]
                        if res["name"] not in existing:
                            st.session_state["annotated_images"].append((res["name"], res["annotated"]))
                            st.success(f"Added {res['name']} to bundle list.")
                        else:
                            st.info(f"{res['name']} already in bundle list.")

        # Overall charts (same as before)
        if sum(global_counts.values()) > 0:
            st.subheader("Overall defect distribution across all uploaded images")
            global_df = pd.DataFrame(
                {
                    "Defect Type": list(global_counts.keys()),
                    "Count": list(global_counts.values()),
                }
            )

            # Bar chart
            bar_chart = (
                alt.Chart(global_df)
                .mark_bar(size=45)
                .encode(
                    x=alt.X(
                        "Defect Type:N",
                        sort="-y",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("Count:Q"),
                    tooltip=["Defect Type", "Count"],
                )
                .properties(
                    height=260,
                    padding={"left": 5, "right": 5, "top": 10, "bottom": 10},
                )
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("#### Defect type share")
            donut_chart = (
                alt.Chart(global_df)
                .mark_arc(innerRadius=55, outerRadius=100)
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    color=alt.Color(
                        "Defect Type:N",
                        legend=alt.Legend(title="Defect type"),
                    ),
                    tooltip=["Defect Type", "Count"],
                )
                .properties(
                    height=260,
                    padding={"left": 0, "right": 0, "top": 10, "bottom": 10},
                )
            )
            st.altair_chart(donut_chart, use_container_width=True)

        else:
            st.info("No defects detected in any of the uploaded images.")

        # -------- Export flow: Finish + Download (CSV + annotated images) --------
        if st.session_state["full_results_df"] is not None:
            st.markdown("### Export results")
            if st.button("Finish defect detection"):
                st.session_state["show_download"] = True

            if st.session_state["show_download"]:
                full_results_df = st.session_state["full_results_df"]
                annotated_images = st.session_state["annotated_images"]

                csv_bytes = full_results_df.to_csv(index=False).encode("utf-8")

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # CSV
                    zf.writestr("circuitguard_detection_results.csv", csv_bytes)
                    # Annotated images
                    for name, pil_img in annotated_images:
                        img_bytes_io = io.BytesIO()
                        pil_img.save(img_bytes_io, format="PNG")
                        img_bytes_io.seek(0)
                        base = os.path.splitext(name)[0]
                        zf.writestr(f"annotated_{base}.png", img_bytes_io.getvalue())

                zip_buffer.seek(0)

                st.download_button(
                    "Download results (CSV + annotated images, ZIP)",
                    data=zip_buffer,
                    file_name="circuitguard_results.zip",
                    mime="application/zip",
                )
else:
    st.info("Upload one or more PCB images to start detection.")
