
import streamlit as st

# ==========================================
# SENTINEL — PROFESSIONAL DASHBOARD THEME
# ==========================================

st.set_page_config(
    page_title="Sentinel | Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101713 0%, #18221b 100%);
    color: #ecf5ee;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

/* Headings */
h1, h2, h3 {
    color: #effcf2 !important;
    font-weight: 700 !important;
}

/* Body text */
p, label, [data-testid="stCaptionContainer"] {
    color: #b9cbbd;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #202e24, #17221a);
    border: 1px solid #344b39;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.20);
}

[data-testid="stMetricLabel"] {
    color: #b9cbbd !important;
}

[data-testid="stMetricValue"] {
    color: #50e890 !important;
    font-weight: 700;
}

/* CSV upload */
[data-testid="stFileUploader"] {
    background: #1b2820;
    border: 1px dashed #4c9b66;
    border-radius: 12px;
    padding: 12px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #16834a, #26b866);
    color: #ffffff;
    border: 1px solid #38ce78;
    border-radius: 9px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #20a65d, #40df85);
    color: #ffffff;
    border: 1px solid #65f3a0;
    box-shadow: 0 0 14px rgba(56, 206, 120, 0.25);
}

/* Tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #344b39;
    border-radius: 10px;
    overflow: hidden;
}

/* Dividers */
hr {
    border-color: #344b39;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111a14;
    border-right: 1px solid #344b39;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Inputs */
input, textarea {
    color: #ecf5ee !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background-color: #202e24;
    border-color: #405b46;
    color: #ecf5ee;
}
</style>
""", unsafe_allow_html=True)


import pandas as pd
import json
import plotly.express as px

from main import (
    detect_anomalies,
    detect_rolling_anomalies,
    explain_anomalies,
)

st.set_page_config(
    page_title="Sentinel | Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

# Dashboard heading
st.title("🛡️ Sentinel")
st.subheader("Intelligent Time-Series Anomaly Detection")

st.write(
    "Monitor incoming data, identify unusual patterns, "
    "and investigate suspicious events."
)

# Dataset upload
st.subheader("📂 Upload Your Dataset")


uploaded_file = st.file_uploader(
    "Upload a CSV file (optional)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        if len(df.columns) < 2:
            st.error("Please upload a CSV with at least two columns.")
            st.stop()

        st.success("Dataset uploaded successfully!")

    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")
        st.stop()

else:
    try:
        df = pd.read_csv("art_daily_jumpsup.csv")
        st.info("Using the NAB dataset: art_daily_jumpsup.csv")

    except Exception as error:
        st.error(f"Could not load the NAB dataset: {error}")
        st.stop()

st.write("Preview of your data:")
st.dataframe(df.head())

st.write("Select the columns to analyze:")


time_col = st.selectbox(
        "Time column",
        options=df.columns,
        index=0
    )

numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

if not numeric_cols:
        st.error("Your CSV needs at least one numeric column.")
        st.stop()

reading_col = st.selectbox(
        "Readings column",
        options=numeric_cols
    )

df = df[[time_col, reading_col]].copy()
df.columns = ["Time", "Readings"]

df["Readings"] = pd.to_numeric(
        df["Readings"],
        errors="coerce"
    )

df = df.dropna(subset=["Readings"]).reset_index(drop=True)

if df.empty:
        st.error("No valid numeric readings were found.")
        st.stop()



# Run both anomaly detectors
df["Is_Anomaly"] = detect_anomalies(df["Readings"]).astype(bool)

df["Rolling_Anomaly"] = (
    detect_rolling_anomalies(df["Readings"]).astype(bool)
)

# Calculate distance from the average
mean_value = df["Readings"].mean()
std_value = df["Readings"].std()

if pd.notna(std_value) and std_value > 0:
    df["Z_Score"] = (
        (df["Readings"] - mean_value) / std_value
    ).abs()
else:
    df["Z_Score"] = 0.0

def get_explanation(row):
    value = row["Readings"]

    if row["Is_Anomaly"] and row["Rolling_Anomaly"]:
        return (
            f"High alert: {value} is unusual compared with the overall data "
            "and recent readings."
        )
    elif row["Is_Anomaly"]:
        return (
            f"High alert: {value} is far from the overall average. "
            "This may indicate an unusual event."
        )
    elif row["Rolling_Anomaly"]:
        return (
            f"Rolling alert: {value} changed significantly compared with "
            "recent readings, even though it may not be unusual overall."
        )
    else:
        return "Reading is within the expected range."

df["Explanation"] = df.apply(get_explanation, axis=1)

df["Severity"] = df.apply(
lambda row: (
"High" if row["Z_Score"] > 3
else "Medium" if (
row["Z_Score"] > 2 or row["Rolling_Anomaly"]
)
else "Normal"
),
axis=1
)

# Mark readings detected by either detector
df["Status"] = (
    df["Is_Anomaly"] | df["Rolling_Anomaly"]
).map({
    True: "Anomaly",
    False: "Normal"
})

# Mark a reading as anomalous if either detector flags it
df["Combined_Anomaly"] = (
    df["Is_Anomaly"] | df["Rolling_Anomaly"]
)

df["Status"] = df["Combined_Anomaly"].map({
    True: "Anomaly",
    False: "Normal"
})

# Calculate all metrics BEFORE displaying them
total = len(df)
anomalies = int(df["Combined_Anomaly"].sum())
rolling_anomalies = int(df["Rolling_Anomaly"].sum())
normal = total - anomalies

high_count = int((df["Severity"] == "High").sum())
medium_count = int((df["Severity"] == "Medium").sum())

# Severity breakdown
st.subheader("🎯 Severity Breakdown")

col1, col2, col3 = st.columns(3)

col1.metric("🔴 High Severity", high_count)
col2.metric("🟠 Medium Severity", medium_count)
col3.metric("📈 Rolling Detector Alerts", rolling_anomalies)

# Overall metrics
st.subheader("📊 Detection Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Readings", total)
col2.metric("Anomalies Detected", anomalies)
col3.metric("Normal Readings", normal)

# Alert message
if anomalies > 0:
    st.warning(
        f"🚨 Sentinel detected {anomalies} anomalie(s) "
        "with the baseline detector!"
    )
else:
    st.success("✅ No baseline anomalies detected.")

if rolling_anomalies > 0:
    st.info(
        f"📈 The rolling detector flagged "
        f"{rolling_anomalies} reading(s)."
    )

# Time-series chart
st.divider()
st.subheader("📈 Time-Series Monitoring")


fig = px.line(
    df,
    x="Time",
    y="Readings",
    title="Server Activity",
    markers=True
)

# Emerald & Graphite styling
fig.update_traces(
    line=dict(color="#26b866", width=3),
    marker=dict(
        color="#50e890",
        size=7,
        line=dict(color="#101713", width=1)
    ),
    hovertemplate=(
        "Time: %{x}<br>"
        "Reading: %{y}<extra></extra>"
    )
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#101713",
    plot_bgcolor="#18221b",
    font=dict(
        color="#ecf5ee",
        family="Arial",
        size=13
    ),
    title=dict(
        text="🌿 Server Activity",
        font=dict(size=22, color="#50e890")
    ),
    xaxis=dict(
        title="Time",
        gridcolor="#344b39",
        zerolinecolor="#344b39"
    ),
    yaxis=dict(
        title="Readings",
        gridcolor="#344b39",
        zerolinecolor="#344b39"
    ),
    margin=dict(l=20, r=20, t=60, b=20),
    hovermode="x unified"
)


rolling_anomaly_rows = df[df["Rolling_Anomaly"]]
fig.add_scatter(
    x = rolling_anomaly_rows["Time"],
    y = rolling_anomaly_rows["Readings"],
    mode="markers",
    name="Rolling Detector Alert",
    marker=dict(
        color="#ffb020",
        size=11,
        symbol="diamond"
    ),
    hovertemplate=(
        "Time: %{x}<br>"
        "Reading: %{y}<br>"
        "Detector: Rolling<extra></extra>"
    )
)


# Highlight baseline anomalies
anomaly_points = df[df["Is_Anomaly"]]

fig.add_scatter(
    x=anomaly_points["Time"],
    y=anomaly_points["Readings"],
    mode="markers",
    name="Baseline Anomaly",
    marker=dict(color="red", size=12, symbol="x")
)

# Highlight rolling detector alerts
rolling_points = df[df["Rolling_Anomaly"]]

fig.add_scatter(
    x=rolling_points["Time"],
    y=rolling_points["Readings"],
    mode="markers",
    name="Rolling Detector Alert",
    marker=dict(color="orange", size=10, symbol="diamond")
)

st.plotly_chart(fig, use_container_width=True)

# Anomaly investigation table
st.subheader("🚨 Detected Anomalies")

# Combine baseline and rolling detector alerts
all_anomalies = df[
    df["Is_Anomaly"] | df["Rolling_Anomaly"]
].copy()

# Label which detector(s) flagged each reading
all_anomalies["Detected By"] = all_anomalies.apply(
    lambda row: (
        "Both Detectors"
        if row["Is_Anomaly"] and row["Rolling_Anomaly"]
        else "Baseline Detector"
        if row["Is_Anomaly"]
        else "Rolling Detector"
    ),
    axis=1
)

# Display combined anomaly table
if not all_anomalies.empty:
    st.dataframe(
        all_anomalies[
            [
                "Time",
                "Readings",
                "Status",
                "Severity",
                "Z_Score",
                "Detected By",
                "Explanation"
            ]
        ],
        use_container_width=True
    )
else:
    st.success("Neither detector found any anomalies.")

# Download baseline anomaly report
csv_data = anomaly_points[
    [
        "Time",
        "Readings",
        "Status",
        "Severity",
        "Z_Score",
        "Explanation"
    ]
].to_csv(index=False)

st.download_button(
    label="📥 Download Anomaly Report",
    data=csv_data,
    file_name="sentinel_anomaly_report.csv",
    mime="text/csv"
)

# Download combined detector results
st.download_button(
    label="📥 Download Full Detection Results",
    data=df.to_csv(index=False),
    file_name="sentinel_full_detection_report.csv",
    mime="text/csv"
)


# Evaluation metrics using labelled test data
st.divider()
st.subheader("📊 Detector Evaluation")

# Evaluate against the known NAB anomaly timestamp
labels_path = "combined_labels.json"

try:
    with open(labels_path, "r", encoding="utf-8") as file:
        nab_labels = json.load(file)

    known_timestamps = nab_labels.get(
        "artificialWithAnomaly/art_daily_jumpsup.csv", []
    )

    timestamps = pd.to_datetime(df["Time"], errors="coerce")
    actual_nab = timestamps.isin(
        pd.to_datetime(known_timestamps, errors="coerce")
    )
    predicted_nab = df["Combined_Anomaly"].astype(bool)

    tp = int((actual_nab & predicted_nab).sum())
    fp = int((~actual_nab & predicted_nab).sum())
    fn = int((actual_nab & ~predicted_nab).sum())

    st.subheader("NAB Known-Point Check")
    st.write(f"Known labeled anomaly points: {int(actual_nab.sum())}")
    st.write(f"Detected labeled anomaly points: {tp}")
    st.write(f"Other rows flagged by SentinelAI: {fp}")

    if tp:
        st.success("SentinelAI detected the known NAB anomaly point!")
    else:
        st.warning("SentinelAI did not detect the known NAB anomaly point.")

    st.caption(
        "Limited evaluation: the JSON contains only known anomaly timestamps, "
        "not complete ground-truth labels for every row. These counts are not "
        "valid full-dataset precision, recall, or F1 scores."
    )

except (FileNotFoundError, json.JSONDecodeError) as error:
    st.warning(f"Could not load combined_labels.json: {error}")

st.caption("Sentinel AI | Utkrishta 2026")
