import glob
import os
import random
import time
from typing import List, Optional
from io import BytesIO

import cv2
import kagglehub
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Travel Setu AI Lab", page_icon="🌌", layout="wide")

LOCAL_DATASET_PATH = "architecture/Indian-monuments/images/test"
KAGGLE_DATASETS = [
    "wwymak/architecture-dataset",
    "tompaulat/modernarchitecture",
    "danushkumarv/indian-monuments-image-dataset",
]

ARCHITECTURE_DB = {
    "ajanta caves": {"summary": "Ajanta Caves are UNESCO-listed Buddhist cave monuments known for murals, sculpture, and rock-cut monastic architecture.", "style": "Rock-Cut Buddhist", "era": "2nd century BCE to 480 CE", "influence": "Buddhist monastic art and mural traditions", "region": "Maharashtra"},
    "ellora caves": {"summary": "Ellora combines Buddhist, Hindu, and Jain cave complexes, including the monumental Kailasa temple carved from a single rock mass.", "style": "Rock-Cut Multi-Faith", "era": "6th to 10th century CE", "influence": "Buddhist, Hindu, and Jain craftsmanship", "region": "Maharashtra"},
    "charar e sharif": {"summary": "Charar-e-Sharif is a deeply revered Sufi shrine with major spiritual significance in Kashmir.", "style": "Kashmiri Islamic", "era": "15th century origin", "influence": "Sufi devotional architecture", "region": "Jammu and Kashmir"},
    "chhota imambara": {"summary": "Chhota Imambara is a ceremonial monument in Lucknow, often called the Palace of Lights for its opulent interiors.", "style": "Indo-Persian", "era": "19th century", "influence": "Persian and Awadhi court design", "region": "Uttar Pradesh"},
    "fatehpur sikri": {"summary": "Fatehpur Sikri was Akbar's planned Mughal capital, known for red sandstone courtyards and syncretic design.", "style": "Mughal Imperial", "era": "16th century", "influence": "Persian, Mughal, and Hindu motifs", "region": "Uttar Pradesh"},
    "gateway of india": {"summary": "Gateway of India is Mumbai's iconic ceremonial arch built on the waterfront in the Indo-Saracenic idiom.", "style": "Indo-Saracenic", "era": "20th century", "influence": "Colonial ceremonial design with Indian motifs", "region": "Maharashtra"},
    "golden temple": {"summary": "The Golden Temple is Sikhism's holiest shrine, celebrated for its sacred sarovar, gilded sanctuary, and devotional atmosphere.", "style": "Sikh Sacred Architecture", "era": "16th century foundation", "influence": "Sikh, Mughal, and Rajput decorative forms", "region": "Punjab"},
    "hawa mahal": {"summary": "Hawa Mahal is Jaipur's famed facade of latticed windows designed for airflow, privacy, and ceremonial viewing.", "style": "Rajput", "era": "18th century", "influence": "Rajput court design and urban palace planning", "region": "Rajasthan"},
    "humayun's tomb": {"summary": "Humayun's Tomb introduced the grand Mughal garden-tomb format that later influenced the Taj Mahal.", "style": "Mughal Garden-Tomb", "era": "16th century", "influence": "Persian charbagh planning and Mughal symmetry", "region": "Delhi"},
    "khajuraho": {"summary": "Khajuraho's temple group is renowned for Nagara spires, sculptural richness, and refined Chandela craftsmanship.", "style": "Nagara Temple", "era": "10th to 12th century", "influence": "Chandela temple architecture", "region": "Madhya Pradesh"},
    "lotus temple": {"summary": "The Lotus Temple is a modern Bahai House of Worship celebrated for its petal-like geometry and serene material palette.", "style": "Contemporary Sacred", "era": "20th century", "influence": "Modernist expression and symbolic floral geometry", "region": "Delhi"},
    "mysore palace": {"summary": "Mysore Palace is a ceremonial royal residence with domes, arches, stained glass, and dramatic Indo-Saracenic detailing.", "style": "Indo-Saracenic Palace", "era": "20th century", "influence": "Hindu, Muslim, Rajput, and Gothic revival elements", "region": "Karnataka"},
    "qutub minar": {"summary": "Qutub Minar is a towering minaret complex marking the early Delhi Sultanate and the adaptation of Islamic forms in India.", "style": "Delhi Sultanate", "era": "12th to 13th century", "influence": "Early Indo-Islamic masonry and inscriptions", "region": "Delhi"},
    "sun temple konark": {"summary": "Konark Sun Temple is a masterwork of Kalinga architecture, conceived as Surya's stone chariot.", "style": "Kalinga Temple", "era": "13th century", "influence": "Solar symbolism and Odishan temple design", "region": "Odisha"},
    "taj mahal": {"summary": "The Taj Mahal is a globally recognized Mughal mausoleum prized for its symmetry, marble inlay, and refined proportions.", "style": "Mughal Mausoleum", "era": "17th century", "influence": "Persian, Timurid, and Mughal funerary architecture", "region": "Uttar Pradesh"},
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Manrope:wght@400;500;600;700&display=swap');
        :root { --bg-1:#0b0f1a; --bg-2:#020617; --panel:rgba(255,255,255,0.06); --text:#f8fafc; --muted:#b8c1d9; --blue:#00d4ff; --violet:#7c3aed; --cyan:#22d3ee; --line:rgba(255,255,255,0.12); --shadow:0 24px 60px rgba(0,0,0,0.35); }
        html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
        .stApp { background:radial-gradient(circle at top left, rgba(34,211,238,0.18), transparent 28%), radial-gradient(circle at top right, rgba(124,58,237,0.14), transparent 24%), linear-gradient(135deg, var(--bg-1) 0%, #071225 40%, var(--bg-2) 100%); color:var(--text); }
        .stApp::before { content:""; position:fixed; inset:0; background:linear-gradient(120deg, rgba(0,212,255,0.04), rgba(124,58,237,0.02), rgba(34,211,238,0.03)); background-size:200% 200%; animation:drift 18s ease infinite; pointer-events:none; z-index:0; }
        @keyframes drift { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }
        .block-container { padding-top:0.6rem; padding-bottom:2rem; max-width:1220px; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg, rgba(7,18,37,0.96), rgba(2,6,23,0.92)); border-right:1px solid rgba(255,255,255,0.08); }
        [data-testid="stSidebar"] > div:first-child { backdrop-filter:blur(18px); }
        h1,h2,h3 { font-family:'Orbitron',sans-serif; letter-spacing:0.03em; color:var(--text); }
        p,label,span,div { color:var(--text); }
        .hero-shell,.glass-card,.output-shell,.detail-shell,.sidebar-shell { position:relative; z-index:1; background:var(--panel); border:1px solid var(--line); border-radius:20px; box-shadow:var(--shadow); backdrop-filter:blur(18px); }
        .hero-shell { padding:2rem 2rem 1.5rem 2rem; margin-bottom:1.2rem; overflow:hidden; }
        .hero-shell::after { content:""; position:absolute; inset:auto -10% -45% 50%; width:320px; height:320px; background:radial-gradient(circle, rgba(0,212,255,0.20), transparent 70%); transform:translateX(-50%); }
        .hero-title { text-align:center; font-size:3rem; font-weight:800; background:linear-gradient(90deg, #ffffff 0%, #8be9ff 40%, #a78bfa 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-shadow:0 0 28px rgba(0,212,255,0.16); margin-bottom:0.25rem; }
        .hero-subtitle { text-align:center; color:var(--muted); font-size:1rem; margin-bottom:1.15rem; }
        .hero-divider { height:1px; width:100%; background:linear-gradient(90deg, transparent, rgba(0,212,255,0.9), rgba(124,58,237,0.8), transparent); box-shadow:0 0 14px rgba(34,211,238,0.3); margin-bottom:1rem; }
        .hero-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:0.9rem; }
        .hero-stat,.metric-card,.upload-shell { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09); border-radius:18px; padding:1rem 1.1rem; }
        .hero-stat-label,.metric-label,.small-kicker { color:var(--muted); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; }
        .hero-stat-value { font-family:'Orbitron',sans-serif; font-size:1.3rem; margin-top:0.35rem; }
        .section-title { font-family:'Orbitron',sans-serif; font-size:1.2rem; margin:0.2rem 0 1rem 0; }
        .section-divider { height:1px; background:linear-gradient(90deg, rgba(34,211,238,0.6), rgba(124,58,237,0.2), transparent); margin:0.25rem 0 1.2rem 0; }
        .upload-shell { border-style:dashed; border-color:rgba(34,211,238,0.45); min-height:150px; box-shadow:inset 0 0 0 1px rgba(255,255,255,0.02); transition:border-color 0.2s ease, box-shadow 0.2s ease; }
        .upload-shell:hover { border-color:rgba(0,212,255,0.85); box-shadow:0 0 0 1px rgba(0,212,255,0.18), 0 0 24px rgba(0,212,255,0.12); }
        .upload-title { font-family:'Orbitron',sans-serif; font-size:0.95rem; margin-bottom:0.35rem; }
        .upload-copy,.metric-subvalue,.sidebar-copy,.preview-label { color:var(--muted); font-size:0.92rem; line-height:1.55; }
        .metric-card { min-height:120px; box-shadow:0 0 0 1px rgba(255,255,255,0.02), 0 10px 24px rgba(0,0,0,0.18); }
        .metric-value,.sidebar-title { font-size:1.1rem; font-weight:700; color:var(--text); font-family:'Orbitron',sans-serif; }
        .metric-subvalue { margin-top:0.45rem; color:#d7def3; }
        .output-shell,.detail-shell { padding:1rem; margin-top:1rem; }
        .loader-shell { text-align:center; padding:1rem 0 1.2rem 0; color:var(--muted); font-size:0.98rem; }
        .stButton > button,.stDownloadButton > button { width:100%; min-height:3rem; border:none; border-radius:999px; color:white; font-weight:700; letter-spacing:0.02em; background:linear-gradient(90deg, var(--blue), var(--violet)); box-shadow:0 12px 28px rgba(0,212,255,0.22); transition:box-shadow 0.2s ease, filter 0.2s ease; }
        .stButton > button:hover,.stDownloadButton > button:hover { filter:brightness(1.05); box-shadow:0 0 0 1px rgba(255,255,255,0.12), 0 0 28px rgba(0,212,255,0.28); }
        .stFileUploader section { background:rgba(255,255,255,0.03); border:1px dashed rgba(255,255,255,0.18); border-radius:16px; }
        .stSelectbox > div > div,.stSlider > div > div,.stRadio > div,.stExpander { background:rgba(255,255,255,0.04); border-radius:16px; }
        [data-testid="stMetric"] { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:1rem; }
        @media (max-width:900px) { .hero-title { font-size:2.25rem; } .hero-grid { grid-template-columns:1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet")


@st.cache_resource(show_spinner=False)
def resolve_dataset_path() -> Optional[str]:
    candidate_paths = [LOCAL_DATASET_PATH]
    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            return candidate

    for dataset_ref in KAGGLE_DATASETS:
        try:
            dataset_root = kagglehub.dataset_download(dataset_ref)
        except Exception:
            continue

        search_roots = [
            dataset_root,
            os.path.join(dataset_root, "images"),
            os.path.join(dataset_root, "Indian-monuments"),
            os.path.join(dataset_root, "Indian-monuments", "images"),
        ]

        for search_root in search_roots:
            if not os.path.isdir(search_root):
                continue

            direct_test_dir = os.path.join(search_root, "test")
            if os.path.isdir(direct_test_dir):
                return direct_test_dir

            for root, dirs, _ in os.walk(search_root):
                if os.path.basename(root).lower() == "test" and dirs:
                    return root

    return None


def get_monuments():
    dataset_path = resolve_dataset_path()
    monuments = []
    if dataset_path and os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                monuments.append(item)
    return sorted(monuments) if monuments else ["ajanta caves", "ellora caves"]


def format_monument_name(name: str) -> str:
    return name.replace("_", " ").title()


def open_uploaded_image(uploaded_file) -> Image.Image:
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return Image.open(uploaded_file)


def get_background_image(monument_name: str, uploaded_bg) -> Optional[Image.Image]:
    if uploaded_bg is not None:
        return open_uploaded_image(uploaded_bg).convert("RGB")

    dataset_path = resolve_dataset_path()
    if not dataset_path:
        return None

    pattern = os.path.join(dataset_path, monument_name, "*.jpg")
    bg_files = glob.glob(pattern)
    if not bg_files:
        bg_files = glob.glob(os.path.join(dataset_path, monument_name, "*.png"))
    if not bg_files:
        return None
    return Image.open(random.choice(bg_files)).convert("RGB")


def infer_architecture_details(class_name: str, confidence: float):
    normalized = class_name.lower().replace("_", " ").strip()
    lookup = ARCHITECTURE_DB.get(normalized)
    if lookup:
        return {
            "label": format_monument_name(normalized),
            "confidence": f"{confidence:.1%}",
            "style": lookup["style"],
            "era": lookup["era"],
            "influence": lookup["influence"],
            "region": lookup["region"],
            "summary": lookup["summary"],
        }

    style = "Historic Landmark Form"
    era = "Heritage period not precisely mapped"
    category_style = {
        "palace": ("Royal Heritage Architecture", "Late medieval to modern royal eras"),
        "mosque": ("Islamic Sacred Architecture", "Medieval to early modern"),
        "temple": ("Sacred Temple Architecture", "Ancient to medieval"),
        "monastery": ("Monastic Architecture", "Ancient to medieval"),
        "dome": ("Monumental Masonry", "Medieval to colonial"),
        "fort": ("Military Heritage Architecture", "Medieval to early modern"),
    }
    for keyword, values in category_style.items():
        if keyword in normalized:
            style, era = values
            break

    return {
        "label": class_name.replace("_", " ").title(),
        "confidence": f"{confidence:.1%}",
        "style": style,
        "era": era,
        "influence": "Model inferred from visual composition and form language",
        "region": "Cross-region heritage reference",
        "summary": (
            f"The model's top visual match is {class_name.replace('_', ' ').title()}. "
            "This is a visual similarity signal from a general image model, not a monument-specific classifier."
        ),
    }


def analyze_architecture(uploaded_file):
    img = open_uploaded_image(uploaded_file).convert("RGB")
    img_array = image.img_to_array(img.resize((224, 224)))
    img_array = np.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)
    predictions = load_model().predict(img_array, verbose=0)
    top_labels = decode_predictions(predictions, top=5)[0]
    details = infer_architecture_details(top_labels[0][1], top_labels[0][2])
    return img, details, top_labels


def enhance_realism(subject: Image.Image, scene: Image.Image, strength: float = 0.7) -> Image.Image:
    subject_rgba = subject.convert("RGBA")
    subject_rgb = np.array(subject_rgba.convert("RGB"), dtype=np.uint8)
    scene_rgb = np.array(scene.convert("RGB"), dtype=np.uint8)

    scene_lab = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    subject_lab = cv2.cvtColor(subject_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    scene_l_mean = max(np.mean(scene_lab[:, :, 0]), 1.0)
    subject_l_mean = max(np.mean(subject_lab[:, :, 0]), 1.0)
    l_adjust = 1 + ((scene_l_mean / subject_l_mean) - 1) * strength
    subject_lab[:, :, 0] = np.clip(subject_lab[:, :, 0] * l_adjust, 0, 255)

    enhanced_rgb = cv2.cvtColor(subject_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    alpha = np.array(subject_rgba.getchannel("A"), dtype=np.uint8)
    return Image.merge("RGBA", [Image.fromarray(enhanced_rgb[:, :, i]) for i in range(3)] + [Image.fromarray(alpha)])


def photorealistic_darshan(fg_file, bg_img: Image.Image, scale=1.0, shadow=0.4, lighting=0.7) -> Image.Image:
    fg_img = open_uploaded_image(fg_file).convert("RGBA")
    fg_no_bg = remove(fg_img)
    if not isinstance(fg_no_bg, Image.Image):
        fg_no_bg = Image.open(BytesIO(fg_no_bg)).convert("RGBA")

    human_scale = (bg_img.height * 0.14) / max(fg_img.height, 1) * scale
    new_size = (max(1, int(fg_no_bg.width * human_scale)), max(1, int(fg_no_bg.height * human_scale)))
    fg_resized = fg_no_bg.resize(new_size, Image.Resampling.LANCZOS)

    y_placement = bg_img.height - new_size[1] - int(bg_img.height * 0.03)
    x_placement = (bg_img.width - new_size[0]) // 2

    shadow_base = fg_resized.copy().convert("L").filter(ImageFilter.GaussianBlur(radius=25))
    shadow_base = ImageEnhance.Contrast(shadow_base).enhance(4)
    shadow_base = ImageEnhance.Brightness(shadow_base).enhance(shadow * 0.25)
    shadow_contact = shadow_base.filter(ImageFilter.GaussianBlur(radius=15))
    shadow_float = shadow_base.filter(ImageFilter.GaussianBlur(radius=35))

    result = bg_img.copy().convert("RGBA")
    result.paste(shadow_contact, (x_placement + 25, y_placement + 20), shadow_contact)
    result.paste(shadow_float, (x_placement + 10, y_placement + 10), shadow_float)

    fg_enhanced = enhance_realism(fg_resized, bg_img, lighting)
    fg_mask = fg_enhanced.getchannel("A").filter(ImageFilter.GaussianBlur(radius=5))
    result.paste(fg_enhanced, (x_placement, y_placement), fg_mask)
    return result.convert("RGB")


def render_hero(monuments_count: int) -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-title">TRAVEL SETU AI LAB</div>
            <div class="hero-subtitle">Virtual Travel &amp; Architecture Intelligence</div>
            <div class="hero-divider"></div>
            <div class="hero-grid">
                <div class="hero-stat">
                    <div class="hero-stat-label">Vision Engine</div>
                    <div class="hero-stat-value">ResNet50 + Scene Blend</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Monument Library</div>
                    <div class="hero-stat-value">{monuments_count} Destinations</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Experience Mode</div>
                    <div class="hero-stat-value">Premium Control Panel</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="small-kicker">{icon} Module</div>
        <div class="section-title">{title}</div>
        <div style="color:#b8c1d9; margin-bottom:0.35rem;">{subtitle}</div>
        <div class="section-divider"></div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subvalue: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subvalue">{subvalue}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preview_card(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="upload-shell">
            <div class="upload-title">{title}</div>
            <div class="upload-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    dataset_path = resolve_dataset_path()
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-shell" style="padding:1rem; margin-bottom:1rem;">
                <div class="sidebar-title">AI Control Panel</div>
                <div class="sidebar-copy">Navigate between travel generation and architecture analysis with a dashboard-style workflow.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["🖼 Virtual Travel Photo Generator", "🏛 Architecture Analyzer"],
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div class="sidebar-shell" style="padding:1rem; margin-top:1rem;">
                <div class="sidebar-title">System Notes</div>
                <div class="sidebar-copy">Dark gradient backdrop, glass cards, premium controls, and AI-guided visual output panels.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if dataset_path:
            source_label = "Local dataset" if os.path.isdir(LOCAL_DATASET_PATH) else "Kaggle cache"
            st.caption(f"Dataset source: {source_label}")
        else:
            st.caption("Dataset source unavailable. Configure Kaggle credentials or add a local dataset folder.")
    return page


def render_virtual_travel(monuments: List[str]) -> None:
    render_section_header(
        "🖼",
        "Virtual Travel Photo Generator",
        "Blend a subject into a destination scene with smart scaling, softer shadows, and lighting alignment.",
    )

    upload_col, scene_col = st.columns(2)
    with upload_col:
        render_preview_card("👤 User Image Input", "Upload a full-body portrait for background removal and perspective-aware placement.")
        fg_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"], key="fg_file", label_visibility="collapsed")
    with scene_col:
        render_preview_card("📍 Location Image Input", "Upload a custom destination shot or leave it empty and use the monument library below.")
        custom_bg = st.file_uploader("Upload a location image", type=["jpg", "jpeg", "png"], key="custom_bg", label_visibility="collapsed")

    control_col1, control_col2 = st.columns([1.2, 1.8])
    with control_col1:
        selected_monument = st.selectbox("Select Monument", monuments, format_func=format_monument_name)
    with control_col2:
        size_col, shadow_col, light_col = st.columns(3)
        size_scale = size_col.slider("Size", 0.5, 1.5, 1.0, 0.1)
        shadow_strength = shadow_col.slider("Shadow", 0.0, 1.0, 0.4, 0.1)
        lighting_blend = light_col.slider("Lighting Match", 0.0, 1.0, 0.7, 0.1)

    background_image = get_background_image(selected_monument, custom_bg)

    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.markdown('<div class="preview-label">Subject Preview</div>', unsafe_allow_html=True)
        if fg_file:
            st.image(open_uploaded_image(fg_file), use_container_width=True)
        else:
            render_preview_card("Awaiting Subject", "Your uploaded portrait will appear here.")
    with preview_right:
        st.markdown('<div class="preview-label">Scene Preview</div>', unsafe_allow_html=True)
        if background_image is not None:
            st.image(background_image, use_container_width=True)
        else:
            render_preview_card("Scene Unavailable", "No location image was found for the selected monument yet.")

    generate = st.button("Generate Realistic Photo", key="generate_photo", type="primary")
    regenerate = st.button("Regenerate Scene", key="regenerate_photo")

    if regenerate and selected_monument:
        st.session_state["darshan_result"] = None
        st.rerun()

    if generate:
        if fg_file is None:
            st.warning("Upload a subject image to generate a virtual travel photo.")
        elif background_image is None:
            st.warning("Provide a custom location image or choose a monument with dataset images.")
        else:
            with st.spinner("Processing AI Model... Generating realistic output..."):
                st.markdown('<div class="loader-shell">Processing AI Model...<br>Generating realistic output...</div>', unsafe_allow_html=True)
                time.sleep(0.2)
                st.session_state["darshan_result"] = photorealistic_darshan(fg_file, background_image, size_scale, shadow_strength, lighting_blend)
                st.session_state["darshan_monument"] = selected_monument

    result = st.session_state.get("darshan_result")
    if result is not None:
        st.markdown('<div class="output-shell">', unsafe_allow_html=True)
        st.image(result, caption="Hyper-Realistic Virtual Darshan", use_container_width=True)
        buf = BytesIO()
        result.save(buf, format="PNG")
        st.download_button(
            "Download Composite",
            data=buf.getvalue(),
            file_name=f"darshan_{st.session_state.get('darshan_monument', 'travel_setu')}.png",
            mime="image/png",
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_architecture_analyzer() -> None:
    render_section_header(
        "🏛",
        "Architecture Analyzer",
        "Inspect a landmark image and surface structured visual intelligence in a cleaner dashboard report.",
    )

    left_col, right_col = st.columns([1.1, 1.4])
    with left_col:
        render_preview_card("📸 Landmark Upload", "Drop a building, temple, palace, or monument image to run the recognition pipeline.")
        uploaded_file = st.file_uploader("Upload architecture image", type=["jpg", "jpeg", "png"], key="architecture_file", label_visibility="collapsed")
        analyze = st.button("Analyze Architecture", key="analyze_architecture", type="primary")
    with right_col:
        render_preview_card("🧠 Analysis Feed", "Structured outputs will appear as premium cards for style, confidence, era, influence, and region.")
        if uploaded_file:
            st.image(open_uploaded_image(uploaded_file), use_container_width=True)

    if analyze:
        if uploaded_file is None:
            st.warning("Upload an image before running the architecture analyzer.")
        else:
            with st.spinner("Analyzing... Processing AI Model..."):
                st.markdown('<div class="loader-shell">Analyzing...<br>Processing AI Model...</div>', unsafe_allow_html=True)
                time.sleep(0.2)
                image_preview, details, top_labels = analyze_architecture(uploaded_file)
                st.session_state["analysis_preview"] = image_preview
                st.session_state["analysis_details"] = details
                st.session_state["analysis_top_labels"] = top_labels

    details = st.session_state.get("analysis_details")
    top_labels = st.session_state.get("analysis_top_labels", [])
    if details:
        metric_cols = st.columns(5)
        cards = [
            ("Architecture Style", details["style"], ""),
            ("Confidence Score", details["confidence"], "Top model match"),
            ("Era", details["era"], ""),
            ("Cultural Influence", details["influence"], ""),
            ("Region", details["region"], ""),
        ]
        for col, (label, value, subvalue) in zip(metric_cols, cards):
            with col:
                render_metric_card(label, value, subvalue)

        st.markdown('<div class="detail-shell">', unsafe_allow_html=True)
        with st.expander("Detailed Report", expanded=True):
            st.markdown(f"### {details['label']}")
            st.write(details["summary"])
            if top_labels:
                st.markdown("**Top Visual Matches**")
                for idx, (_, class_name, score) in enumerate(top_labels[:5], start=1):
                    st.write(f"{idx}. {class_name.replace('_', ' ').title()} - {score:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)


inject_css()

monuments = get_monuments()

if "darshan_result" not in st.session_state:
    st.session_state["darshan_result"] = None

render_hero(len(monuments))
current_page = render_sidebar()

if current_page == "🖼 Virtual Travel Photo Generator":
    render_virtual_travel(monuments)
else:
    render_architecture_analyzer()

footer_left, footer_right = st.columns(2)
with footer_left:
    st.metric("Monuments Indexed", len(monuments))
with footer_right:
    st.metric("Interface Mode", "Futuristic Glass UI")
