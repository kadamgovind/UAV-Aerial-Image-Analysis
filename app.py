# ============================================================
# KRYOMAI DEFENCE AI
# UAV AERIAL IMAGE ANALYSIS
# app.py
# ============================================================

import io
import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


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
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best.pt"

HISTORY_FILE = "reports/detection_history.csv"

HIGH_CONFIDENCE_THRESHOLD = 0.80

HIGH_OBJECT_COUNT_THRESHOLD = 20


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
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

try:

    model = load_model()

    model_status = True

except Exception as e:

    model_status = False

    st.error(
        "❌ Unable to load YOLO model."
    )

    st.markdown(
        """
        Make sure your trained model exists at:

        `models/best.pt`
        """
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚁 UAV Aerial Image Analysis'
    '</div>',
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

st.sidebar.title(
    "🛡️ AI Detection System"
)


if model_status:

    st.sidebar.success(
        "🟢 YOLO Model Loaded"
    )

else:

    st.sidebar.error(
        "🔴 YOLO Model Error"
    )


st.sidebar.markdown("---")

st.sidebar.subheader(
    "⚙️ Detection Settings"
)


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
    '<div class="section-title">'
    '📤 Upload UAV Image'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Select an aerial image for AI analysis",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a UAV/aerial image above "
        "to start AI detection."
    )

    st.markdown(
        "### 🔬 Analysis Pipeline"
    )

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

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as e:

    st.error(
        "❌ Unable to read the uploaded image."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# CREATE / PRESERVE MISSION ID
# ============================================================

if (
    "uploaded_file_name"
    not in st.session_state
    or st.session_state.uploaded_file_name
    != uploaded_file.name
):

    st.session_state.uploaded_file_name = (
        uploaded_file.name
    )

    st.session_state.mission_id = (
        "UAV-"
        + uuid.uuid4().hex[:8].upper()
    )


mission_id = st.session_state.mission_id


# ============================================================
# ANALYSIS TIME
# ============================================================

analysis_time = datetime.now().strftime(
    "%d %b %Y, %H:%M:%S"
)


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📋 Image Information'
    '</div>',
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
# IMAGE + AI DETECTION
# ============================================================

st.divider()

image_col1, image_col2 = st.columns(2)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

with image_col1:

    st.subheader(
        "📷 Original Image"
    )

    st.image(
        image,
        use_container_width=True
    )


# ============================================================
# YOLO AI DETECTION
# ============================================================

with st.spinner(
    "🤖 AI is analyzing the UAV image..."
):

    try:

        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False
        )

    except Exception as e:

        st.error(
            "❌ YOLO detection failed."
        )

        st.code(str(e))

        st.stop()


# ============================================================
# GET RESULT
# ============================================================

result = results[0]

boxes = result.boxes


# ============================================================
# ANNOTATED IMAGE
# ============================================================

annotated_image = result.plot()


# YOLO returns BGR.
# Convert BGR → RGB.

annotated_rgb = annotated_image[:, :, ::-1]


# ============================================================
# DISPLAY AI RESULT
# ============================================================

with image_col2:

    st.subheader(
        "🎯 AI Detection Result"
    )

    st.image(
        annotated_rgb,
        use_container_width=True
    )


# ============================================================
# MISSION INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🛰️ Mission Analysis"
)


# ============================================================
# MISSION VARIABLES
# ============================================================

image_resolution = (
    f"{image.width} × {image.height}"
)

model_name = "YOLO"

analysis_status = (
    "ANALYSIS COMPLETE"
)


# ============================================================
# MISSION METRICS
# ============================================================

mission_col1, mission_col2, mission_col3 = (
    st.columns(3)
)


with mission_col1:

    st.metric(
        "🛰️ Mission ID",
        mission_id
    )

    st.metric(
        "📷 Image Resolution",
        image_resolution
    )


with mission_col2:

    st.metric(
        "🕐 Analysis Time",
        analysis_time
    )

    st.metric(
        "🤖 AI Model",
        model_name
    )


with mission_col3:

    st.metric(
        "🎚️ Confidence Threshold",
        f"{confidence:.0%}"
    )

    st.metric(
        "🟢 Status",
        analysis_status
    )


# ============================================================
# INITIALIZE DETECTION VARIABLES
# ============================================================

class_counts = {}

detected_classes = []

detection_data = []

confidences = []

total_objects = 0

total_classes = 0

average_confidence = 0.0

highest_confidence = 0.0

lowest_confidence = 0.0

alerts = []


# ============================================================
# PROCESS DETECTIONS
# ============================================================

if boxes is not None and len(boxes) > 0:

    class_names = result.names

    confidences = (
        boxes.conf.tolist()
    )

    for i in range(len(boxes)):

        class_id = int(
            boxes.cls[i].item()
        )

        class_name = class_names[
            class_id
        ]

        confidence_score = float(
            boxes.conf[i].item()
        )

        detected_classes.append(
            class_name
        )

        detection_data.append(
            {
                "No.": i + 1,
                "Object": class_name,
                "Confidence": (
                    f"{confidence_score:.2%}"
                )
            }
        )

        class_counts[class_name] = (
            class_counts.get(
                class_name,
                0
            ) + 1
        )


    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    total_objects = len(
        detected_classes
    )

    total_classes = len(
        class_counts
    )

    average_confidence = (
        sum(confidences)
        / len(confidences)
    )

    highest_confidence = max(
        confidences
    )

    lowest_confidence = min(
        confidences
    )


# ============================================================
# DETECTION SUMMARY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📊 Detection Summary'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DETECTIONS FOUND
# ============================================================

if total_objects > 0:

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


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
    # OBJECT DISTRIBUTION
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Object Distribution"
    )


    chart_data = pd.DataFrame(
        {
            "Object": list(
                class_counts.keys()
            ),
            "Count": list(
                class_counts.values()
            )
        }
    )


    st.bar_chart(
        chart_data.set_index(
            "Object"
        )
    )


    # ========================================================
    # DETECTED OBJECT LIST
    # ========================================================

    st.subheader(
        "🔎 Detected Objects"
    )


    for class_name, count in (
        class_counts.items()
    ):

        st.write(
            f"**{class_name}** — {count}"
        )


else:

    st.warning(
        "⚠️ No objects detected at the "
        "current confidence threshold."
    )

    st.info(
        "Try lowering the Confidence Threshold "
        "from the sidebar."
    )


# ============================================================
# DETECTION ALERT SYSTEM
# ============================================================

st.divider()

st.subheader(
    "🚨 Detection Alert System"
)


# ============================================================
# HIGH CONFIDENCE COUNT
# ============================================================

high_confidence_count = sum(
    1
    for score in confidences
    if score >= HIGH_CONFIDENCE_THRESHOLD
)


# ============================================================
# HIGH CONFIDENCE ALERT
# ============================================================

if high_confidence_count > 0:

    alerts.append(
        {
            "level": "HIGH",
            "message": (
                f"{high_confidence_count} detection(s) "
                f"have confidence ≥ "
                f"{HIGH_CONFIDENCE_THRESHOLD:.0%}."
            )
        }
    )


# ============================================================
# HIGH OBJECT COUNT ALERT
# ============================================================

if total_objects >= HIGH_OBJECT_COUNT_THRESHOLD:

    alerts.append(
        {
            "level": "MEDIUM",
            "message": (
                f"High object density detected: "
                f"{total_objects} objects identified."
            )
        }
    )


# ============================================================
# MULTIPLE OBJECT CLASSES ALERT
# ============================================================

if total_classes >= 4:

    alerts.append(
        {
            "level": "INFO",
            "message": (
                f"Multiple object categories detected: "
                f"{total_classes} classes."
            )
        }
    )


# ============================================================
# ALERT COUNTS
# ============================================================

high_alert_count = sum(
    1
    for alert in alerts
    if alert["level"] == "HIGH"
)

medium_alert_count = sum(
    1
    for alert in alerts
    if alert["level"] == "MEDIUM"
)

info_alert_count = sum(
    1
    for alert in alerts
    if alert["level"] == "INFO"
)

total_alert_count = len(
    alerts
)


# ============================================================
# DISPLAY ALERTS
# ============================================================

if alerts:

    for alert in alerts:

        if alert["level"] == "HIGH":

            st.error(
                f"🔴 HIGH ALERT — "
                f"{alert['message']}"
            )

        elif alert["level"] == "MEDIUM":

            st.warning(
                f"🟠 MEDIUM ALERT — "
                f"{alert['message']}"
            )

        else:

            st.info(
                f"🔵 INFORMATION — "
                f"{alert['message']}"
            )

else:

    st.success(
        "🟢 No detection alerts triggered."
    )


# ============================================================
# CONFIDENCE ANALYTICS
# ============================================================

if total_objects > 0:

    st.divider()

    st.subheader(
        "📈 Confidence Analytics"
    )


    confidence_data = pd.DataFrame(
        {
            "Detection": range(
                1,
                len(confidences) + 1
            ),
            "Confidence": confidences
        }
    )


    st.line_chart(
        confidence_data.set_index(
            "Detection"
        )
    )


    # ========================================================
    # CONFIDENCE STATISTICS
    # ========================================================

    confidence_col1, confidence_col2, confidence_col3 = (
        st.columns(3)
    )


    with confidence_col1:

        st.metric(
            "Average Confidence",
            f"{average_confidence:.2%}"
        )


    with confidence_col2:

        st.metric(
            "Highest Confidence",
            f"{highest_confidence:.2%}"
        )


    with confidence_col3:

        st.metric(
            "Lowest Confidence",
            f"{lowest_confidence:.2%}"
        )


# ============================================================
# DETAILED DETECTION TABLE
# ============================================================

if total_objects > 0:

    st.divider()

    st.subheader(
        "📋 Detailed Detection Results"
    )


    st.dataframe(
        detection_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INDIVIDUAL CONFIDENCE SCORES
# ============================================================

if total_objects > 0:

    st.subheader(
        "🎯 Individual Confidence Scores"
    )


    class_names = result.names


    for i, confidence_score in enumerate(
        confidences
    ):

        class_id = int(
            boxes.cls[i].item()
        )

        class_name = class_names[
            class_id
        ]

        st.write(
            f"{i + 1}. **{class_name}** — "
            f"{confidence_score:.2%}"
        )


# ============================================================
# AUTOMATIC MISSION REPORT
# ============================================================

st.divider()

st.subheader(
    "📄 Automatic Mission Report"
)


# ============================================================
# OBJECT SUMMARY
# ============================================================

if class_counts:

    object_summary = []

    for class_name, count in (
        class_counts.items()
    ):

        object_summary.append(
            f"- {class_name}: {count}"
        )

    object_summary_text = "\n".join(
        object_summary
    )

else:

    object_summary_text = (
        "- No objects detected."
    )


# ============================================================
# ALERT SUMMARY
# ============================================================

if alerts:

    alert_summary = []

    for alert in alerts:

        alert_summary.append(
            f"- {alert['level']}: "
            f"{alert['message']}"
        )

    alert_text = "\n".join(
        alert_summary
    )

else:

    alert_text = (
        "- No detection alerts triggered."
    )


# ============================================================
# MISSION REPORT
# ============================================================

mission_report = f"""
============================================================
KRYOMAI DEFENCE AI
UAV AERIAL IMAGE ANALYSIS
AUTOMATIC MISSION REPORT
============================================================

MISSION INFORMATION
------------------------------------------------------------
Mission ID           : {mission_id}
Analysis Time        : {analysis_time}
Analysis Status      : {analysis_status}
AI Model             : {model_name}
Confidence Threshold : {confidence:.0%}

IMAGE INFORMATION
------------------------------------------------------------
Image Resolution     : {image.width} × {image.height}
Image Width          : {image.width}px
Image Height         : {image.height}px

DETECTION SUMMARY
------------------------------------------------------------
Total Objects        : {total_objects}
Object Classes       : {total_classes}
Average Confidence   : {average_confidence:.2%}
Highest Confidence   : {highest_confidence:.2%}
Lowest Confidence    : {lowest_confidence:.2%}

OBJECT DISTRIBUTION
------------------------------------------------------------
{object_summary_text}

DETECTION ALERTS
------------------------------------------------------------
Total Alerts         : {total_alert_count}
High Alerts          : {high_alert_count}
Medium Alerts        : {medium_alert_count}
Information Alerts   : {info_alert_count}

{alert_text}

SYSTEM STATUS
------------------------------------------------------------
Status               : ANALYSIS COMPLETE
Detection Engine     : YOLO
Computer Vision      : ENABLED
UAV Image Analysis   : ENABLED

============================================================
END OF MISSION REPORT
============================================================

KRYOMAI Defence AI
UAV Aerial Image Analysis
"""


# ============================================================
# REPORT PREVIEW
# ============================================================

with st.expander(
    "👁️ Preview Mission Report"
):

    st.text(
        mission_report
    )


# ============================================================
# DOWNLOAD MISSION REPORT
# ============================================================

st.download_button(
    label="📄 Download Mission Report",
    data=mission_report,
    file_name=(
        f"{mission_id}_mission_report.txt"
    ),
    mime="text/plain"
)


# ============================================================
# EXPORT DETECTION RESULT
# ============================================================

st.divider()

st.subheader(
    "⬇️ Export Detection Result"
)


# ============================================================
# CONVERT ANNOTATED IMAGE TO JPEG
# ============================================================

result_image = Image.fromarray(
    annotated_rgb
)


image_buffer = io.BytesIO()


result_image.save(
    image_buffer,
    format="JPEG",
    quality=95
)


image_buffer.seek(0)


# ============================================================
# DOWNLOAD ANNOTATED IMAGE
# ============================================================

st.download_button(
    label="⬇️ Download Annotated Image",
    data=image_buffer,
    file_name=(
        f"{mission_id}_uav_detection.jpg"
    ),
    mime="image/jpeg"
)


# ============================================================
# DETECTION HISTORY
# ============================================================

st.divider()

st.subheader(
    "🗂️ Detection History"
)


st.write(
    "Save this completed analysis to Detection History."
)


# ============================================================
# SAVE ANALYSIS
# ============================================================

if st.button(
    "💾 Save Analysis to History",
    type="primary"
):

    try:

        # ----------------------------------------------------
        # CREATE REPORTS DIRECTORY
        # ----------------------------------------------------

        os.makedirs(
            "reports",
            exist_ok=True
        )


        # ----------------------------------------------------
        # OBJECT CLASS LIST
        # ----------------------------------------------------

        object_classes = ", ".join(
            class_counts.keys()
        )


        # ----------------------------------------------------
        # NEW HISTORY RECORD
        # ----------------------------------------------------

        new_record = pd.DataFrame(
            [
                {
                    "Mission ID": mission_id,

                    "Analysis Time": (
                        analysis_time
                    ),

                    "Image Resolution": (
                        image_resolution
                    ),

                    "Confidence Threshold": (
                        f"{confidence:.0%}"
                    ),

                    "Total Objects": (
                        total_objects
                    ),

                    "Object Classes": (
                        object_classes
                    ),

                    "Average Confidence": (
                        f"{average_confidence:.2%}"
                    ),

                    "Highest Confidence": (
                        f"{highest_confidence:.2%}"
                    ),

                    "Lowest Confidence": (
                        f"{lowest_confidence:.2%}"
                    ),

                    "Alert Count": (
                        total_alert_count
                    ),

                    "High Alerts": (
                        high_alert_count
                    ),

                    "Medium Alerts": (
                        medium_alert_count
                    ),

                    "Info Alerts": (
                        info_alert_count
                    ),

                    "Status": (
                        analysis_status
                    )
                }
            ]
        )


        # ----------------------------------------------------
        # EXISTING HISTORY
        # ----------------------------------------------------

        if os.path.exists(
            HISTORY_FILE
        ):

            history_df = pd.read_csv(
                HISTORY_FILE
            )


            # ------------------------------------------------
            # HANDLE OLD CSV FILES
            # ------------------------------------------------

            required_columns = (
                new_record.columns.tolist()
            )


            for column in required_columns:

                if column not in history_df.columns:

                    history_df[column] = ""


            # Keep same column order.

            history_df = history_df[
                required_columns
            ]


            # ------------------------------------------------
            # DUPLICATE MISSION CHECK
            # ------------------------------------------------

            existing_ids = (
                history_df[
                    "Mission ID"
                ]
                .astype(str)
                .tolist()
            )


            if mission_id in existing_ids:

                st.info(
                    "ℹ️ This mission is already "
                    "saved in Detection History."
                )

                history_df = None


            else:

                history_df = pd.concat(
                    [
                        history_df,
                        new_record
                    ],
                    ignore_index=True
                )


        else:

            history_df = new_record


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        if history_df is not None:

            history_df.to_csv(
                HISTORY_FILE,
                index=False
            )

            st.success(
                "✅ Analysis successfully saved "
                "to Detection History."
            )


    except Exception as e:

        st.error(
            "❌ Unable to save detection history."
        )

        st.code(
            str(e)
        )


# ============================================================
# DISPLAY SAVED DETECTION HISTORY
# ============================================================

if os.path.exists(
    HISTORY_FILE
):

    try:

        history_df = pd.read_csv(
            HISTORY_FILE
        )


        if not history_df.empty:

            st.subheader(
                "📚 Saved Analyses"
            )


            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # DOWNLOAD HISTORY
            # =================================================

            history_csv = (
                history_df
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )


            st.download_button(
                label=(
                    "⬇️ Download Detection History"
                ),
                data=history_csv,
                file_name=(
                    "detection_history.csv"
                ),
                mime="text/csv"
            )


        else:

            st.info(
                "No saved detection history yet."
            )


    except Exception as e:

        st.error(
            "❌ Unable to read detection history."
        )

        st.code(
            str(e)
        )


else:

    st.info(
        "No saved detection history yet. "
        "Run an analysis and click "
        "'Save Analysis to History'."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "KRYOMAI Defence AI | "
    "UAV Aerial Image Analysis | "
    "YOLO-based Computer Vision | "
    "Mission Control"
)