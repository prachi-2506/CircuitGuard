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
from PIL import Image
import pandas as pd
import altair as alt

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

# ------------------ CSS ------------------
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #f8fbff;
        font-family: 'Poppins', sans-serif;
        color: #102a43;
    }

    .header-container {
        display:flex;
        flex-direction:column;
        align-items:center;
        margin-top:0.5rem;
        margin-bottom:1rem;
    }

    .logo-circle {
        width:64px; height:64px;
        border-radius:50%;
        background:#e0f2fe;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:32px;
        margin-bottom:6px;
    }

    .main-title {
        font-weight:700;
        font-size:2.4rem;
        color:#13406b;
    }

    .metric-card {
        border-radius:16px;
        padding:0.75rem 1rem;
        background:#fff;
        border:1px solid #dbeafe;
    }

    .metric-label {
        font-size:0.75rem;
        text-transform:uppercase;
        color:#6b7280;
    }

    .metric-value {
        font-size:1.1rem;
        font-weight:600;
    }

    .result-row {
        background:#ffffff;
        border:1px solid #eef6ff;
        border-radius:12px;
        padding:12px 16px;
        margin-bottom:10px;
        box-shadow:0 4px 10px rgba(0,0,0,0.03);
    }

    .stButton>button {
        border-radius:999px;
        font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ SESSION STATE ------------------
if "open_row_idx" not in st.session_state:
    st.session_state["open_row_idx"] = None

# ------------------ MODEL ------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

def run_inference(model, image):
    result = model.predict(image, conf=CONFIDENCE, iou=IOU)[0]
    plotted = result.plot()[:, :, ::-1]
    return Image.fromarray(plotted), result

def get_defect_locations(result, class_names, image_name):
    if result.boxes is None:
        return []

    rows = []
    for box, cls, conf in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist()
    ):
        x1, y1, x2, y2 = box
        rows.append({
            "Image": image_name,
            "Defect type": class_names[int(cls)],
            "Confidence": round(float(conf), 2),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
        })
    return rows

# ------------------ HEADER ------------------
st.markdown("""
<div class="header-container">
  <div class="logo-circle">🛡️</div>
  <div class="main-title">CircuitGuard – PCB Defect Detection</div>
</div>
""", unsafe_allow_html=True)

# ------------------ METRICS ------------------
cols = st.columns(4)
metrics = [
    ("mAP@50", "0.9823"),
    ("mAP@50–95", "0.5598"),
    ("Precision", "0.9714"),
    ("Recall", "0.9765"),
]
for c, (k, v) in zip(cols, metrics):
    with c:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{k}</div>"
            f"<div class='metric-value'>{v}</div></div>",
            unsafe_allow_html=True
        )

# ------------------ UPLOAD ------------------
st.markdown("### Upload PCB Images")
uploaded_files = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ------------------ PROCESS ------------------
if uploaded_files:
    model = load_model(MODEL_PATH)
    class_names = model.names

    image_results = []
    all_rows = []
    global_counts = Counter()

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        annotated, result = run_inference(model, img)
        locs = get_defect_locations(result, class_names, file.name)

        global_counts.update([r["Defect type"] for r in locs])
        all_rows.extend(locs)

        image_results.append({
            "name": file.name,
            "original": img,
            "annotated": annotated,
            "loc_rows": locs,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    # ------------------ SUMMARY ------------------
    summary = []
    for i, r in enumerate(image_results):
        summary.append({
            "idx": i,
            "name": r["name"],
            "count": len(r["loc_rows"]),
            "max_conf": max([x["Confidence"] for x in r["loc_rows"]], default=0),
            "res": r
        })

    st.markdown("### Results — summary table")

    # ------------------ SEARCH ------------------
    q = st.text_input("Search", placeholder="image name / defect / number")

    filtered = [
        s for s in summary
        if q.lower() in s["name"].lower() or q == ""
    ]

    st.markdown(f"**Showing {len(filtered)} of {len(summary)} images**")

    # ------------------ SCROLLABLE RESULTS (CORRECT WAY) ------------------
    results_container = st.container(height=360)  # ✅ ONLY VALID SCROLL METHOD

    with results_container:
        for s in filtered:
            res = s["res"]
            idx = s["idx"]

            st.markdown("<div class='result-row'>", unsafe_allow_html=True)
            cols = st.columns([4, 1, 1, 2])

            if cols[0].button(res["name"], key=f"btn_{idx}"):
                st.session_state["open_row_idx"] = (
                    None if st.session_state["open_row_idx"] == idx else idx
                )

            cols[1].write(s["count"])
            cols[2].write(round(s["max_conf"], 2))
            cols[3].write(res["processed_at"])
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state["open_row_idx"] == idx:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(res["original"], caption="Original", use_column_width=True)
                with c2:
                    st.image(res["annotated"], caption="Annotated", use_column_width=True)

                if res["loc_rows"]:
                    st.dataframe(
                        pd.DataFrame(res["loc_rows"]).drop(columns=["Image"]),
                        use_container_width=True
                    )
                else:
                    st.info("No defects detected.")

                st.markdown("---")

    # ------------------ CHARTS ------------------
    if global_counts:
        st.subheader("Overall defect distribution")
        df = pd.DataFrame({
            "Defect": list(global_counts.keys()),
            "Count": list(global_counts.values())
        })
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x="Defect:N", y="Count:Q", tooltip=["Defect", "Count"]
            ),
            use_container_width=True
        )
else:
    st.info("Upload PCB images to start detection.")
