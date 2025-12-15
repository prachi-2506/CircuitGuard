# app.py
# CircuitGuard — Scrollable results + search panel (FINAL STABLE VERSION)

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
        font-family: Poppins, sans-serif;
        color: #102a43;
    }

    .header-container {
        display:flex;
        flex-direction:column;
        align-items:center;
        margin-bottom:1rem;
    }

    .logo-circle {
        width:64px;height:64px;border-radius:50%;
        background:#e0f2fe;display:flex;
        align-items:center;justify-content:center;
        font-size:32px;
    }

    .main-title {
        font-size:2.5rem;
        font-weight:700;
        color:#13406b;
        margin-top:0.4rem;
    }

    .metric-card {
        background:white;
        border-radius:16px;
        padding:1rem;
        border:1px solid #dbeafe;
    }

    .metric-label {
        font-size:0.75rem;
        color:#6b7280;
        text-transform:uppercase;
    }

    .metric-value {
        font-size:1.2rem;
        font-weight:600;
    }

    .instruction-card {
        background:white;
        border-radius:16px;
        padding:1rem;
        border:1px solid #dbeafe;
        margin:1rem 0;
    }

    .defect-badge {
        background:#e0f2fe;
        padding:0.3rem 0.6rem;
        border-radius:999px;
        margin-right:0.3rem;
        font-size:0.8rem;
    }

    .search-heading {
        font-weight:600;
        margin-bottom:0.4rem;
        margin-top:0.5rem;
        font-size:1rem;
    }

    /* SCROLLABLE RESULTS BLOCK */
    .results-block {
        max-height:520px;
        overflow-y:auto;
        padding:12px;
        border-radius:14px;
        border:1px solid #dbeafe;
        background:#f8fbff;
        margin-top:10px;
    }

    .result-row {
        background:white;
        border-radius:12px;
        padding:12px 16px;
        margin-bottom:10px;
        border:1px solid #eef6ff;
    }

    .results-block::-webkit-scrollbar {
        width:10px;
    }
    .results-block::-webkit-scrollbar-thumb {
        background:#c7e5ff;
        border-radius:8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ SESSION STATE ------------------
if "open_row_idx" not in st.session_state:
    st.session_state.open_row_idx = None

# ------------------ MODEL ------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

def run_inference(model, image):
    r = model.predict(image, conf=CONFIDENCE, iou=IOU)[0]
    img = Image.fromarray(r.plot()[:, :, ::-1])
    return img, r

def get_defect_locations(result, class_names, image_name):
    rows = []
    if result.boxes is None:
        return rows
    for box, cls, conf in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist()
    ):
        x1,y1,x2,y2 = box
        rows.append({
            "Image": image_name,
            "Defect type": class_names[int(cls)],
            "Confidence": round(float(conf),2),
            "x1": round(x1,1),
            "y1": round(y1,1),
            "x2": round(x2,1),
            "y2": round(y2,1)
        })
    return rows

# ------------------ HEADER ------------------
st.markdown("""
<div class="header-container">
    <div class="logo-circle">🛡️</div>
    <div class="main-title">CircuitGuard – PCB Defect Detection</div>
</div>
""", unsafe_allow_html=True)

metrics = [("mAP@50","0.9823"),("mAP@50–95","0.5598"),("Precision","0.9714"),("Recall","0.9765")]
cols = st.columns(4)
for col,(k,v) in zip(cols,metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{k}</div>
            <div class="metric-value">{v}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="instruction-card">
<strong>🧭 How to use:</strong>
<ol>
<li>Upload PCB images</li>
<li>View results in the table</li>
<li>Search & filter defects</li>
<li>Click image name to expand details</li>
</ol>
</div>
""", unsafe_allow_html=True)

# ------------------ UPLOAD ------------------
st.markdown("### Upload PCB Images")
uploaded_files = st.file_uploader(
    "Upload",
    type=["png","jpg","jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ------------------ PROCESS ------------------
if uploaded_files:
    model = load_model(MODEL_PATH)
    class_names = model.names

    image_results = []
    global_counts = Counter()

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        annotated, result = run_inference(model, img)
        locs = get_defect_locations(result, class_names, file.name)
        for d in locs:
            global_counts[d["Defect type"]] += 1
        image_results.append({
            "name": file.name,
            "original": img,
            "annotated": annotated,
            "loc_rows": locs,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    # ------------------ SUMMARY LIST ------------------
    summary = []
    for i,res in enumerate(image_results):
        summary.append({
            "idx": i,
            "name": res["name"],
            "defect_count": len(res["loc_rows"]),
            "max_conf": max([d["Confidence"] for d in res["loc_rows"]], default=0),
            "defect_types": list({d["Defect type"] for d in res["loc_rows"]}),
            "res": res
        })

    # ------------------ SEARCH ------------------
    st.markdown("### Results — summary table")
    st.markdown('<div class="search-heading">Search</div>', unsafe_allow_html=True)

    f1,f2,f3 = st.columns(3)
    q = f1.text_input("Search", placeholder="missing_hole / 3 / 0.8")
    field = f2.selectbox("Field",["All","Image","Defect type","Defect count","Max confidence"])
    all_types = sorted({t for s in summary for t in s["defect_types"]})
    sel_types = f3.multiselect("Defect type", all_types)

    def match(s):
        text = q.lower().strip()
        if sel_types and not any(t in sel_types for t in s["defect_types"]):
            return False
        if not text:
            return True
        if field in ("All","Image") and text in s["name"].lower():
            return True
        if field in ("All","Defect type") and any(text in t.lower() for t in s["defect_types"]):
            return True
        if field in ("All","Defect count") and text.isdigit() and int(text)==s["defect_count"]:
            return True
        if field in ("All","Max confidence"):
            try:
                return float(text)==round(s["max_conf"],2)
            except:
                pass
        return False

    filtered = [s for s in summary if match(s)]
    st.markdown(f"**Showing {len(filtered)} of {len(summary)} images**")

    # ------------------ SCROLLABLE RESULTS ------------------
    st.markdown('<div class="results-block">', unsafe_allow_html=True)
    for s in filtered:
        res = s["res"]
        st.markdown('<div class="result-row">', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns([4,1,1,2])
        if c1.button(res["name"], key=res["name"]):
            st.session_state.open_row_idx = s["idx"] if st.session_state.open_row_idx!=s["idx"] else None
        c2.write(s["defect_count"])
        c3.write(round(s["max_conf"],2))
        c4.write(res["processed_at"])
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.open_row_idx==s["idx"]:
            col1,col2 = st.columns(2)
            col1.image(res["original"], caption="Original", use_column_width=True)
            col2.image(res["annotated"], caption="Annotated", use_column_width=True)
            if res["loc_rows"]:
                st.dataframe(pd.DataFrame(res["loc_rows"]).drop(columns=["Image"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ CHARTS ------------------
    if global_counts:
        df = pd.DataFrame({"Defect":global_counts.keys(),"Count":global_counts.values()})
        st.subheader("Overall defect distribution")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(x="Defect",y="Count"),
            use_container_width=True
        )

else:
    st.info("Upload images to start detection.")
