import asyncio
import io
import os

import streamlit as st
from PIL import Image

from core_engine import fetch_photos, enhance


# --- CLOUD DEPLOYMENT HACK ---
# This ensures Playwright downloads the Chromium browser only once when the server boots.
@st.cache_resource
def install_playwright():
    os.system("playwright install chromium")
    return True

install_playwright()
# -----------------------------


st.set_page_config(page_title="Map Scrapper", page_icon="", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#e7f7f3,#eff4ff); margin-bottom: 1.4rem;}
      .hero h1 {margin: 0 0 .35rem 0; color: #143b3b;}
      .hero p {margin: 0; color: #486164;}
      div[data-testid="stImage"] img {
          border-radius: 10px;
          box-shadow: 0 4px 14px rgba(0,0,0,0.08);
      }
    </style>
    <div class="hero"><h1>MapScrapper AI</h1>
    <p>Find public Google Maps photos and give them a restrained local polish.</p></div>
    """,
    unsafe_allow_html=True,
)

if "raw_photos" not in st.session_state:
    st.session_state.raw_photos = []
if "visible_count" not in st.session_state:
    st.session_state.visible_count = 6
if "polished_photos" not in st.session_state:
    st.session_state.polished_photos = {}
if "search_message" not in st.session_state:
    st.session_state.search_message = ""

with st.form("photo_search"):
    query = st.text_input(
        "Resort or hotel name and location",
        placeholder="Taj Exotica Resort, Goa",
        help="Use the place name and city or region for a more precise Maps search.",
    )
    submitted = st.form_submit_button("Search Photos", type="primary", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Enter a resort or hotel name and location.")
    else:
        st.session_state.raw_photos = []
        st.session_state.visible_count = 6
        st.session_state.polished_photos = {}
        st.session_state.search_message = ""

        status = st.status("Starting photo search…", expanded=True)

        def report(message: str) -> None:
            status.write(message)

        try:
            result = asyncio.run(fetch_photos(query.strip(), count=24, report=report))
            status.update(label="Photo search completed", state="complete", expanded=False)
            st.session_state.raw_photos = result.get("photos", [])
            st.session_state.search_message = result.get("message", "")
        except Exception as exc:
            status.update(label="Photo search failed", state="error", expanded=True)
            st.error(f"The request could not be completed: {exc}")
            st.session_state.raw_photos = []

if st.session_state.search_message:
    st.info(st.session_state.search_message)

if st.session_state.raw_photos:
    photos = st.session_state.raw_photos
    visible_count = min(st.session_state.visible_count, len(photos))

    st.subheader(f"Discovered Photos ({len(photos)} found)")
    st.caption("Select the pictures you want to polish, then click 'Convert & Polish Selected'.")

    cols_per_row = 3
    selected_indices = []

    for row_start in range(0, visible_count, cols_per_row):
        row_photos = photos[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="large")
        for col_idx, photo_bytes in enumerate(row_photos):
            global_idx = row_start + col_idx
            with cols[col_idx]:
                st.image(Image.open(io.BytesIO(photo_bytes)), use_container_width=True)
                if st.checkbox(f"Select Photo {global_idx + 1}", key=f"select_photo_{global_idx}"):
                    selected_indices.append(global_idx)
        st.write("")

    # Controls: Load more and Polish buttons
    action_col1, action_col2 = st.columns(2, gap="medium")
    with action_col1:
        if visible_count < len(photos):
            if st.button("➕ Load more", use_container_width=True):
                st.session_state.visible_count = min(visible_count + 6, len(photos))
                st.rerun()
        else:
            st.button("All photos loaded", disabled=True, use_container_width=True)

    with action_col2:
        btn_label = f"Convert & Polish Selected ({len(selected_indices)})" if selected_indices else "✨ Convert & Polish Selected"
        if st.button(btn_label, type="primary", use_container_width=True, disabled=(len(selected_indices) == 0)):
            polish_status = st.status(f"Polishing {len(selected_indices)} selected photo(s)…", expanded=True)
            for i, idx in enumerate(selected_indices, start=1):
                polish_status.write(f"Enhancing photo {idx + 1} ({i}/{len(selected_indices)})…")
                try:
                    enh = enhance(photos[idx])
                    st.session_state.polished_photos[idx] = enh
                except Exception as err:
                    polish_status.write(f"Failed to enhance photo {idx + 1}: {err}")
            polish_status.update(label="Polishing complete!", state="complete", expanded=False)
            st.rerun()

if st.session_state.polished_photos:
    st.divider()
    polished_keys = sorted(st.session_state.polished_photos.keys())
    st.subheader(f"{len(polished_keys)} Polished Photo{'s' if len(polished_keys) != 1 else ''}")
    for idx in polished_keys:
        st.markdown(f"#### Photo {idx + 1}")
        original_col, enhanced_col = st.columns(2, gap="large")
        orig_data = st.session_state.raw_photos[idx]
        enh_data = st.session_state.polished_photos[idx]
        with original_col:
            st.caption("Original Raw Image")
            st.image(Image.open(io.BytesIO(orig_data)), use_container_width=True)
        with enhanced_col:
            st.caption("Polished Architectural Image")
            st.image(Image.open(io.BytesIO(enh_data)), use_container_width=True)
            st.download_button(
                "Download Enhanced Image",
                data=enh_data,
                file_name=f"stay-photo-{idx + 1}-enhanced.jpg",
                mime="image/jpeg",
                key=f"download-{idx + 1}",
                use_container_width=True,
            )
        st.divider()

st.caption("Photos are processed locally in memory. Availability depends on Google Maps page behavior and the public images exposed for the listing.")
