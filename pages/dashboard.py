# ============================================================
# KRYOMAI DEFENCE AI
# MISSION CONTROL DASHBOARD
# dashboard.py
# ============================================================

import os
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mission Control | KRYOMAI Defence AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = "reports/detection_history.csv"
REPORTS_DIR = "reports/mission_reports"


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

    .status-box {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛰️ KRYOMAI Defence AI — Mission Control'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'UAV Aerial Image Analysis & Mission Monitoring Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ Mission Control")

st.sidebar.success(
    "🟢 System Online"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📡 Dashboard"
)

st.sidebar.info(
    "Monitor UAV image-analysis missions, "
    "detection statistics, alerts and mission reports."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "KRYOMAI Defence AI\n\n"
    "UAV • Computer Vision • YOLO"
)


# ============================================================
# LOAD HISTORY
# ============================================================

if not os.path.exists(HISTORY_FILE):

    st.warning(
        "⚠️ No detection history found."
    )

    st.info(
        "Run an analysis from app.py first."
    )

    st.stop()


try:

    history_df = pd.read_csv(
        HISTORY_FILE
    )

except Exception as e:

    st.error(
        "❌ Unable to read detection history."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# EMPTY HISTORY
# ============================================================

if history_df.empty:

    st.warning(
        "⚠️ Detection history is empty."
    )

    st.stop()


# ============================================================
# DATA NORMALIZATION
# ============================================================

required_columns = [
    "Mission ID",
    "Analysis Time",
    "Image Resolution",
    "Confidence Threshold",
    "Total Objects",
    "Object Classes",
    "Average Confidence",
    "Highest Confidence",
    "Lowest Confidence",
    "Status"
]


for column in required_columns:

    if column not in history_df.columns:

        history_df[column] = ""


# ============================================================
# NUMERIC COLUMNS
# ============================================================

history_df["Total Objects"] = pd.to_numeric(
    history_df["Total Objects"],
    errors="coerce"
).fillna(0)


# ============================================================
# CONFIDENCE CLEANING
# ============================================================

def clean_percentage(value):

    try:

        value = str(value).replace(
            "%",
            ""
        )

        return float(value)

    except Exception:

        return 0.0


history_df["Average Confidence Numeric"] = (
    history_df["Average Confidence"]
    .apply(clean_percentage)
)


history_df["Highest Confidence Numeric"] = (
    history_df["Highest Confidence"]
    .apply(clean_percentage)
)


history_df["Lowest Confidence Numeric"] = (
    history_df["Lowest Confidence"]
    .apply(clean_percentage)
)


# ============================================================
# ALERT COLUMNS
# ============================================================

# New versions of app.py may contain these columns.
# Older CSV files will automatically receive default values.

if "High Alerts" not in history_df.columns:

    history_df["High Alerts"] = 0


if "Medium Alerts" not in history_df.columns:

    history_df["Medium Alerts"] = 0


if "Info Alerts" not in history_df.columns:

    history_df["Info Alerts"] = 0


if "Total Alerts" not in history_df.columns:

    history_df["Total Alerts"] = (
        history_df["High Alerts"]
        + history_df["Medium Alerts"]
        + history_df["Info Alerts"]
    )


# Convert alert values to numeric.

for column in [
    "High Alerts",
    "Medium Alerts",
    "Info Alerts",
    "Total Alerts"
]:

    history_df[column] = pd.to_numeric(
        history_df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# SEARCH & FILTER
# ============================================================

st.subheader(
    "🔎 Mission Search & Filters"
)


filter_col1, filter_col2, filter_col3 = (
    st.columns(3)
)


# ------------------------------------------------------------
# Mission Search
# ------------------------------------------------------------

with filter_col1:

    search_text = st.text_input(
        "🔎 Search Mission",
        placeholder="Enter Mission ID..."
    )


# ------------------------------------------------------------
# Status Filter
# ------------------------------------------------------------

with filter_col2:

    statuses = sorted(
        history_df["Status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_status = st.selectbox(
        "📌 Status",
        ["ALL"] + statuses
    )


# ------------------------------------------------------------
# Object Count Filter
# ------------------------------------------------------------

with filter_col3:

    max_objects = int(
        history_df["Total Objects"].max()
    )

    object_limit = st.slider(
        "🎯 Maximum Objects",
        min_value=0,
        max_value=max(
            max_objects,
            1
        ),
        value=max(
            max_objects,
            1
        )
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = history_df.copy()


# Mission search

if search_text:

    filtered_df = filtered_df[
        filtered_df["Mission ID"]
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False
        )
    ]


# Status filter

if selected_status != "ALL":

    filtered_df = filtered_df[
        filtered_df["Status"]
        .astype(str)
        == selected_status
    ]


# Object count

filtered_df = filtered_df[
    filtered_df["Total Objects"]
    <= object_limit
]


st.caption(
    f"Showing {len(filtered_df)} of "
    f"{len(history_df)} missions"
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

st.divider()

st.subheader(
    "📊 Mission Overview"
)


total_missions = len(
    filtered_df
)


total_objects = int(
    filtered_df["Total Objects"].sum()
)


if total_missions > 0:

    average_objects = (
        filtered_df["Total Objects"].mean()
    )

    average_confidence = (
        filtered_df[
            "Average Confidence Numeric"
        ].mean()
    )

else:

    average_objects = 0

    average_confidence = 0


total_alerts = int(
    filtered_df["Total Alerts"].sum()
)


col1, col2, col3, col4, col5 = (
    st.columns(5)
)


with col1:

    st.metric(
        "🛰️ Missions",
        total_missions
    )


with col2:

    st.metric(
        "🎯 Objects Detected",
        total_objects
    )


with col3:

    st.metric(
        "📈 Avg Objects/Mission",
        f"{average_objects:.1f}"
    )


with col4:

    st.metric(
        "🎯 Avg Confidence",
        f"{average_confidence:.1f}%"
    )


with col5:

    st.metric(
        "🚨 Total Alerts",
        total_alerts
    )


# ============================================================
# ALERT STATISTICS
# ============================================================

st.divider()

st.subheader(
    "🚨 Alert Statistics"
)


high_alerts = int(
    filtered_df["High Alerts"].sum()
)

medium_alerts = int(
    filtered_df["Medium Alerts"].sum()
)

info_alerts = int(
    filtered_df["Info Alerts"].sum()
)


alert_col1, alert_col2, alert_col3, alert_col4 = (
    st.columns(4)
)


with alert_col1:

    st.metric(
        "🔴 High Alerts",
        high_alerts
    )


with alert_col2:

    st.metric(
        "🟠 Medium Alerts",
        medium_alerts
    )


with alert_col3:

    st.metric(
        "🔵 Information",
        info_alerts
    )


with alert_col4:

    st.metric(
        "🚨 Total Alerts",
        high_alerts
        + medium_alerts
        + info_alerts
    )


# ============================================================
# ALERT CHART
# ============================================================

alert_chart = pd.DataFrame(
    {
        "Alert Type": [
            "HIGH",
            "MEDIUM",
            "INFO"
        ],
        "Count": [
            high_alerts,
            medium_alerts,
            info_alerts
        ]
    }
)


st.bar_chart(
    alert_chart.set_index(
        "Alert Type"
    )
)


# ============================================================
# MISSION STATISTICS
# ============================================================

st.divider()

st.subheader(
    "📈 Mission Statistics"
)


if not filtered_df.empty:

    mission_chart = filtered_df[
        [
            "Mission ID",
            "Total Objects"
        ]
    ].copy()


    mission_chart = mission_chart.set_index(
        "Mission ID"
    )


    st.bar_chart(
        mission_chart
    )

else:

    st.info(
        "No missions match the current filters."
    )


# ============================================================
# OBJECT CLASS STATISTICS
# ============================================================

st.divider()

st.subheader(
    "🏷️ Object Class Statistics"
)


class_totals = {}


for value in (
    filtered_df["Object Classes"]
    .astype(str)
):

    if not value or value == "nan":

        continue


    for class_name in value.split(","):

        class_name = class_name.strip()


        if class_name:

            class_totals[class_name] = (
                class_totals.get(
                    class_name,
                    0
                ) + 1
            )


if class_totals:

    class_df = pd.DataFrame(
        {
            "Object Class": list(
                class_totals.keys()
            ),
            "Missions": list(
                class_totals.values()
            )
        }
    )


    st.bar_chart(
        class_df.set_index(
            "Object Class"
        )
    )

else:

    st.info(
        "No object class information available."
    )


# ============================================================
# MISSION DETAILS
# ============================================================

st.divider()

st.subheader(
    "🔍 Mission Details"
)


if not filtered_df.empty:

    mission_options = (
        filtered_df["Mission ID"]
        .astype(str)
        .tolist()
    )


    selected_mission = st.selectbox(
        "Select Mission",
        mission_options
    )


    selected_rows = filtered_df[
        filtered_df["Mission ID"]
        .astype(str)
        == selected_mission
    ]


    if not selected_rows.empty:

        mission = selected_rows.iloc[0]


        detail_col1, detail_col2 = (
            st.columns(2)
        )


        with detail_col1:

            st.markdown(
                f"""
                **🛰️ Mission ID:**  
                {mission["Mission ID"]}

                **🕐 Analysis Time:**  
                {mission["Analysis Time"]}

                **📷 Image Resolution:**  
                {mission["Image Resolution"]}

                **🤖 AI Model:**  
                YOLO

                **🎚️ Confidence Threshold:**  
                {mission["Confidence Threshold"]}
                """
            )


        with detail_col2:

            st.markdown(
                f"""
                **🎯 Total Objects:**  
                {int(mission["Total Objects"])}

                **🏷️ Object Classes:**  
                {mission["Object Classes"]}

                **📈 Average Confidence:**  
                {mission["Average Confidence"]}

                **⭐ Highest Confidence:**  
                {mission["Highest Confidence"]}

                **📉 Lowest Confidence:**  
                {mission["Lowest Confidence"]}

                **📌 Status:**  
                {mission["Status"]}
                """
            )


        # ----------------------------------------------------
        # Alert details
        # ----------------------------------------------------

        st.markdown("### 🚨 Mission Alerts")


        mission_high = int(
            mission["High Alerts"]
        )

        mission_medium = int(
            mission["Medium Alerts"]
        )

        mission_info = int(
            mission["Info Alerts"]
        )


        alert_detail_col1, alert_detail_col2, alert_detail_col3 = (
            st.columns(3)
        )


        with alert_detail_col1:

            st.metric(
                "🔴 High",
                mission_high
            )


        with alert_detail_col2:

            st.metric(
                "🟠 Medium",
                mission_medium
            )


        with alert_detail_col3:

            st.metric(
                "🔵 Info",
                mission_info
            )


# ============================================================
# MISSION REPORT ACCESS
# ============================================================

st.divider()

st.subheader(
    "📄 Mission Report Access"
)


if not os.path.exists(REPORTS_DIR):

    st.info(
        "No mission reports available yet."
    )

else:

    report_files = [
        file
        for file in os.listdir(REPORTS_DIR)
        if file.lower().endswith(".txt")
    ]


    if not report_files:

        st.info(
            "No mission reports available yet."
        )

    else:

        report_files = sorted(
            report_files,
            reverse=True
        )


        report_mission_ids = [
            os.path.splitext(file)[0]
            .replace(
                "_mission_report",
                ""
            )
            for file in report_files
        ]


        report_col1, report_col2 = (
            st.columns(2)
        )


        with report_col1:

            selected_report_id = st.selectbox(
                "📄 Select Mission Report",
                report_mission_ids
            )


        selected_report_file = (
            selected_report_id
            + "_mission_report.txt"
        )


        selected_report_path = os.path.join(
            REPORTS_DIR,
            selected_report_file
        )


        if os.path.exists(
            selected_report_path
        ):

            with open(
                selected_report_path,
                "r",
                encoding="utf-8"
            ) as report_file:

                report_content = (
                    report_file.read()
                )


            with st.expander(
                "👁️ Preview Selected Mission Report"
            ):

                st.text(
                    report_content
                )


            st.download_button(
                label="⬇️ Download Selected Mission Report",
                data=report_content,
                file_name=selected_report_file,
                mime="text/plain"
            )


# ============================================================
# RECENT MISSIONS
# ============================================================

st.divider()

st.subheader(
    "🗂️ Recent Missions"
)


if not filtered_df.empty:

    recent_df = (
        filtered_df
        .tail(10)
        .iloc[::-1]
        .copy()
    )


    display_columns = [
        "Mission ID",
        "Analysis Time",
        "Image Resolution",
        "Total Objects",
        "Object Classes",
        "Average Confidence",
        "Total Alerts",
        "Status"
    ]


    st.dataframe(
        recent_df[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No recent missions available."
    )


# ============================================================
# EXPORT FILTERED DATA
# ============================================================

st.divider()

st.subheader(
    "⬇️ Export Dashboard Data"
)


filtered_csv = (
    filtered_df
    .drop(
        columns=[
            "Average Confidence Numeric",
            "Highest Confidence Numeric",
            "Lowest Confidence Numeric"
        ],
        errors="ignore"
    )
    .to_csv(
        index=False
    )
    .encode("utf-8")
)


st.download_button(
    label="⬇️ Download Filtered Mission History",
    data=filtered_csv,
    file_name="kryomai_filtered_mission_history.csv",
    mime="text/csv"
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.subheader(
    "🛡️ System Status"
)


status1, status2, status3, status4 = (
    st.columns(4)
)


with status1:

    st.success(
        "🟢 Detection Engine Online"
    )


with status2:

    st.success(
        "🟢 Mission History Online"
    )


with status3:

    if os.path.exists(REPORTS_DIR):

        st.success(
            "🟢 Mission Reports Online"
        )

    else:

        st.warning(
            "🟠 Mission Reports Empty"
        )


with status4:

    st.success(
        "🟢 Dashboard Online"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "KRYOMAI Defence AI | "
    "UAV Aerial Image Analysis | "
    "Mission Control Dashboard"
)