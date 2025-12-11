# app.py
# CircuitGuard — Table-style results with clickable image-name rows that toggle inline details.
# NOTE: This is your original app with UI fixes:
#  - removed st.experimental_rerun() to avoid runtime error
#  - table-style rows (lighter look) and inline toggling on image name click

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

# Optional PDF libs
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

# ------------------ CSS (kept + table styling) ------------------
st.markdown(
    """
    <style>
    /* keep your existing styling and add table row / card look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f8fbff;
        font-family: 'Poppins', sans-serif;
        color: #102a43;
    }

    /* Sidebar (unchanged) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f5ff 0%, #e7fff7 100%);
        border-right: 1px solid #d0e2ff;
    }
    [data-testid="stSidebar"] * { color: #102a43 !important; }
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code { background: #e5e7eb !important; color: #111827 !important; }

    /* Button style (unchanged) */
    .stButton>button { border-radius: 999px; padding: 0.5rem 1.25rem; border: none; font-weight: 500; background: #85c5ff; color: #0f172a; box-shadow: 0 8px 14px rgba(148,163,184,0.28); }
    .stButton>button:hover { background: #63b1ff; transform: translateY(-1px) scale(1.01); }

    /* Table-like row cards */
    .result-row {
        background: #ffffff;
        border: 1px solid #e6eefb;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 6px 16px rgba(14, 30, 37, 0.03);
    }
    .result-row .image-name-btn {
        background: #e6f0ff;
        border-radius: 999px;
        padding: 6px 14px;
        color: #08395b;
        border: none;
        font-weight: 600;
        cursor: pointer;
    }
    .result-header {
        display:flex; gap:12px; align-items:center; font-weight:600; color:#334e68; margin-bottom:8px;
    }
    .result-cells { color:#334e68; }

    /* small helper for metadata */
    .cell-small { color:#64748b; font-size:0.95rem; }

    /* responsive image inside details */
    .detail-img { max-width:100%; height:auto; border-radius:8px; border:1px solid #eef5ff; }

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
# store which row is open (index in the current run's image_results)
if "open_row_idx" not in st.session_state:
    st.session_state["open_row_idx"] = None

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

# top metrics (kept)
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

# instructions and defect badges (kept)
st.markdown(
    """
    <div class="instruction-card">
      <strong>🧭 How to use CircuitGuard:</strong>
      <ol>
        <li>Prepare clear PCB images (top view, good lighting).</li>
        <li>Upload one or more images using the box below.</li>
        <li>Wait for the model to run – we’ll generate annotated results.</li>
        <li>In the results table below click the image name to toggle details inline.</li>
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

# ------------------ DETECTION & RESULTS TABLE ------------------
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
        image_results: List[Dict] = []

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
            st.session_state["annotated_images"] = [(res["name"], res["annotated"]) for res in image_results]
        else:
            st.session_state["full_results_df"] = None
            st.session_state["annotated_images"] = []

        # success banners
        st.markdown(
            """
            <div class="robot-success">
              <span class="robot-label">[SYSTEM]</span>
              DEFECT SCAN COMPLETE — ANALYSIS DASHBOARD ONLINE.
            </div>
            <div class="status-strip">
              Detection complete. Click any image name in the results table to view details inline.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- Results header (table-like) ----------
        st.markdown("### Results — summary table (click image name to toggle details)")
        header_cols = st.columns([4, 1, 1, 2])
        header_cols[0].markdown("**Image**")
        header_cols[1].markdown("**Defects**")
        header_cols[2].markdown("**Max confidence**")
        header_cols[3].markdown("**Processed at**")

        # Render rows – each row is a "card" look; clicking the name toggles inline details
        for idx, res in enumerate(image_results):
            defect_count = len(res["loc_rows"])
            max_conf = 0.0
            if res["loc_rows"]:
                max_conf = max([r["Confidence"] for r in res["loc_rows"]])

            # container for the row card
            st.markdown(f'<div class="result-row">', unsafe_allow_html=True)
            row_cols = st.columns([4, 1, 1, 2])
            # using streamlit button to toggle the open index
            btn_key = f"img_row_btn_{idx}_{res['name']}"
            # render a "button-like" clickable name using st.button
            if row_cols[0].button(res["name"], key=btn_key):
                # toggle open/close
                if st.session_state.get("open_row_idx") == idx:
                    st.session_state["open_row_idx"] = None
                else:
                    st.session_state["open_row_idx"] = idx
                # no experimental rerun; Streamlit automatically reruns on button clicks

            # numeric cells
            row_cols[1].markdown(f"<div class='cell-small'>{defect_count}</div>", unsafe_allow_html=True)
            row_cols[2].markdown(f"<div class='cell-small'>{round(max_conf,2)}</div>", unsafe_allow_html=True)
            row_cols[3].markdown(f"<div class='cell-small'>{res.get('processed_at','')}</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # If open, render inline details directly under this row
            if st.session_state.get("open_row_idx") == idx:
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Original image")
                    buf_o = io.BytesIO()
                    res["original"].save(buf_o, format="PNG")
                    buf_o.seek(0)
                    st.image(buf_o.getvalue(), use_column_width=True)
                with c2:
                    st.subheader("Annotated image")
                    buf_a = io.BytesIO()
                    res["annotated"].save(buf_a, format="PNG")
                    buf_a.seek(0)
                    st.image(buf_a.getvalue(), use_column_width=True)

                    # download annotated image
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

                # action buttons
                a1, a2, a3 = st.columns(3)
                with a1:
                    if a1.button(f"Download PDF — {res['name']}", key=f"pdf_{idx}"):
                        if not HAS_REPORTLAB:
                            st.error("reportlab not installed — PDF generation not available in this environment.")
                        else:
                            # minimal safe PDF (you can replace with your full generator)
                            pdf_buf = io.BytesIO()
                            c = canvas.Canvas(pdf_buf, pagesize=landscape(A4))
                            page_w, page_h = landscape(A4)
                            left_img = ImageReader(res["original"])
                            right_img = ImageReader(res["annotated"])
                            iw, ih = res["original"].size
                            scale = min((page_h - 200) / ih, (page_w/2 - 60) / iw, 1.0)
                            w2 = iw * scale
                            h2 = ih * scale
                            c.drawImage(left_img, 30, page_h - 60 - h2, width=w2, height=h2)
                            c.drawImage(right_img, 50 + page_w/2, page_h - 60 - h2, width=w2, height=h2)
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
                    if a2.button(f"Re-run detection — {res['name']}", key=f"rerun_{idx}"):
                        try:
                            model = load_model(MODEL_PATH)
                            plotted_img, result = run_inference(model, res["original"])
                            loc_rows = get_defect_locations(result, model.names, res["name"])
                            res["annotated"] = plotted_img
                            res["result"] = result
                            res["loc_rows"] = loc_rows
                            res["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            st.success("Re-run complete for " + res["name"])
                            # no experimental rerun; UI will refresh on next interaction
                        except Exception as e:
                            st.error(f"Re-run failed: {e}")
                with a3:
                    if a3.button(f"Add to ZIP selection — {res['name']}", key=f"selectzip_{idx}"):
                        existing = [n for n, _ in st.session_state.get("annotated_images", [])]
                        if res["name"] not in existing:
                            st.session_state["annotated_images"].append((res["name"], res["annotated"]))
                            st.success(f"Added {res['name']} to bundle list.")
                        else:
                            st.info(f"{res['name']} already in bundle list.")

                st.markdown("---")

        # Overall charts (unchanged)
        if sum(global_counts.values()) > 0:
            st.subheader("Overall defect distribution across all uploaded images")
            global_df = pd.DataFrame({"Defect Type": list(global_counts.keys()), "Count": list(global_counts.values())})

            bar_chart = (
                alt.Chart(global_df)
                .mark_bar(size=45)
                .encode(x=alt.X("Defect Type:N", sort="-y", axis=alt.Axis(labelAngle=0)), y=alt.Y("Count:Q"), tooltip=["Defect Type", "Count"])
                .properties(height=260)
            )
            st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("#### Defect type share")
            donut_chart = (
                alt.Chart(global_df)
                .mark_arc(innerRadius=55, outerRadius=100)
                .encode(theta=alt.Theta("Count:Q", stack=True), color=alt.Color("Defect Type:N", legend=alt.Legend(title="Defect type")), tooltip=["Defect Type", "Count"])
                .properties(height=260)
            )
            st.altair_chart(donut_chart, use_container_width=True)
        else:
            st.info("No defects detected in any of the uploaded images.")

        # Export flow (unchanged)
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
