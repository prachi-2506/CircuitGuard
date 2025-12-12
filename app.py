# (top of file unchanged) ...
# ------------------ CSS: center header, lighter browse + download buttons (restore uploader preview) ------------------
st.markdown(
    """
    <style>
    /* (existing CSS omitted for brevity — keep your original styles) */

    /* Results scroll block */
    .results-block {
        max-height: 520px;
        overflow: auto;
        padding: 12px;
        background: transparent;
    }
    /* keep row look inside scroll area */
    .result-row { background:#ffffff; border:1px solid #eef6ff; border-radius:12px; padding:12px 16px; margin-bottom:12px; box-shadow: 0 6px 16px rgba(14,30,37,0.02); }
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

# (all other functions unchanged) ...

# ------------------ MAIN UI (unchanged header + uploader) ------------------
# ... your existing header, metrics, uploader.

# ------------------ PROCESSING & RESULTS ------------------
if uploaded_files:
    # (processing code unchanged up to creating image_results and global_counts)
    # ... (process images into image_results, all_rows etc.)

    # create a small summary list with computed fields for filtering
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

    # build UI: search + filters
    st.markdown("### Results — summary table (click image name to toggle details)")
    with st.container():
        f1, f2, f3 = st.columns([2,2,2])
        query = f1.text_input("Search (image, defect name, number...)", value="", placeholder="e.g. missing_hole or 3 or 0.7")
        field = f2.selectbox("Search field", options=["All", "Image", "Defect type", "Defect count", "Max confidence", "Processed at"], index=0)
        # multi-select for defect types (useful)
        all_defect_types = sorted(list({dt for s in summary_list for dt in s["defect_types"] if dt}))
        selected_defect_types = f3.multiselect("Filter defect types", options=all_defect_types, default=[])

    # helper: does a summary entry match query+field+selected_defect_types
    def matches_entry(entry, query_text, field_choice, selected_types):
        q = (query_text or "").strip().lower()
        # defect-type multi-select: if present, require entry to have at least one chosen type
        if selected_types:
            if not any(dt in selected_types for dt in entry["defect_types"]):
                return False
        if not q:
            return True
        # field-specific checks
        try:
            # numeric parsing
            q_num_int = int(q)
        except Exception:
            q_num_int = None
        try:
            q_num = float(q)
        except Exception:
            q_num = None

        if field_choice == "All":
            # image name
            if q in entry["name"].lower():
                return True
            # defect types
            if any(q in (dt or "").lower() for dt in entry["defect_types"]):
                return True
            # numeric checks
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

    # filter summaries
    filtered = [s for s in summary_list if matches_entry(s, query, field, selected_defect_types)]
    # show a small status
    st.markdown(f"**Showing {len(filtered)} of {len(summary_list)} images**")

    # put the rows inside a scrollable block
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
            # toggle open by storing original index
            if st.session_state.get("open_row_idx") == idx:
                st.session_state["open_row_idx"] = None
            else:
                st.session_state["open_row_idx"] = idx
        row_cols[1].markdown(f"<div class='cell-small'>{defect_count}</div>", unsafe_allow_html=True)
        row_cols[2].markdown(f"<div class='cell-small'>{round(max_conf,2)}</div>", unsafe_allow_html=True)
        row_cols[3].markdown(f"<div class='cell-small'>{res.get('processed_at','')}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # if this item is open, show details
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

    # (rest unchanged: charts + export)
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
