import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import io

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UAV Aerial Image Analysis",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 650;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    model_status = True

except Exception as e:

    model_status = False

    st.error("❌ Unable to load YOLO model.")
    st.code(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚁 UAV Aerial Image Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered aerial imagery analysis and object detection using YOLO'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ AI Detection System")

st.sidebar.success("🟢 YOLO Model Loaded")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a UAV or aerial image. "
    "The trained YOLO model will automatically "
    "detect objects in the image."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "UAV Aerial Image Analysis\n\n"
    "Computer Vision • YOLO • AI"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload UAV Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Select an aerial image for AI analysis",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a UAV/aerial image above to start AI detection."
    )

    st.markdown("### 🔬 Analysis Pipeline")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📷")
        st.write("Image Input")

    with col2:
        st.markdown("### 🤖")
        st.write("YOLO Detection")

    with col3:
        st.markdown("### 🎯")
        st.write("Object Analysis")

    with col4:
        st.markdown("### 📊")
        st.write("Detection Report")

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(uploaded_file).convert("RGB")

except Exception:

    st.error("❌ Unable to read the uploaded image.")
    st.stop()


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📋 Image Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Image Width",
        f"{image.width}px"
    )

with info2:
    st.metric(
        "Image Height",
        f"{image.height}px"
    )

with info3:
    st.metric(
        "Confidence",
        f"{confidence:.0%}"
    )


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("📷 Original Image")

    st.image(
        image,
        use_container_width=True
    )


# ============================================================
# AI DETECTION
# ============================================================

with st.spinner("🤖 AI is analyzing the UAV image..."):

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )


result = results[0]


# ============================================================
# ANNOTATED IMAGE
# ============================================================

annotated_image = result.plot()

# YOLO plot returns BGR image.
# Convert it to RGB for Streamlit.
annotated_rgb = annotated_image[:, :, ::-1]


with col2:

    st.subheader("🎯 AI Detection Result")

    st.image(
        annotated_rgb,
        use_container_width=True
    )


# ============================================================
# DETECTION DATA
# ============================================================

boxes = result.boxes


st.divider()

st.markdown(
    '<div class="section-title">📊 Detection Summary</div>',
    unsafe_allow_html=True
)


# ============================================================
# OBJECT DETECTION RESULTS
# ============================================================

if boxes is not None and len(boxes) > 0:

    class_names = result.names

    detected_classes = []

    detection_data = []

    confidences = boxes.conf.tolist()

    for i in range(len(boxes)):

        class_id = int(boxes.cls[i].item())

        class_name = class_names[class_id]

        confidence_score = float(
            boxes.conf[i].item()
        )

        detected_classes.append(class_name)

        detection_data.append(
            {
                "No.": i + 1,
                "Object": class_name,
                "Confidence": f"{confidence_score:.2%}"
            }
        )


    # ========================================================
    # CLASS COUNTS
    # ========================================================

    class_counts = {}

    for class_name in detected_classes:

        class_counts[class_name] = (
            class_counts.get(class_name, 0) + 1
        )


    total_objects = len(detected_classes)

    total_classes = len(class_counts)

    average_confidence = (
        sum(confidences) / len(confidences)
    )

    highest_confidence = max(confidences)


    # ========================================================
    # TOP METRICS
    # ========================================================

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "🎯 Total Objects",
            total_objects
        )

    with metric2:

        st.metric(
            "🏷️ Object Classes",
            total_classes
        )

    with metric3:

        st.metric(
            "📈 Avg Confidence",
            f"{average_confidence:.1%}"
        )

    with metric4:

        st.metric(
            "⭐ Highest Confidence",
            f"{highest_confidence:.1%}"
        )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    st.subheader("🔎 Detected Objects")

    for class_name, count in class_counts.items():

        st.write(
            f"**{class_name}** — {count}"
        )


    # ========================================================
    # DETAILED DETECTION TABLE
    # ========================================================

    st.subheader("📋 Detailed Detection Results")

    st.dataframe(
        detection_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CONFIDENCE SCORES
    # ========================================================

    st.subheader("🎯 Confidence Scores")

    for i, confidence_score in enumerate(confidences):

        class_id = int(
            boxes.cls[i].item()
        )

        class_name = class_names[class_id]

        st.write(
            f"{i + 1}. **{class_name}** — "
            f"{confidence_score:.2%}"
        )


    # ========================================================
    # DOWNLOAD RESULT
    # ========================================================

    st.divider()

    st.subheader("⬇️ Export Detection Result")

    result_image = Image.fromarray(annotated_rgb)

    image_buffer = io.BytesIO()

    result_image.save(
        image_buffer,
        format="JPEG"
    )

    image_buffer.seek(0)


    st.download_button(
        label="⬇️ Download Annotated Image",
        data=image_buffer,
        file_name="uav_detection_result.jpg",
        mime="image/jpeg"
    )


else:

    st.warning(
        "⚠️ No objects detected at the current confidence threshold."
    )

    st.info(
        "Try lowering the Confidence Threshold from the sidebar."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UAV Aerial Image Analysis | "
    "YOLO-based Computer Vision | "
    "KRYOMAI Defence AI"
)