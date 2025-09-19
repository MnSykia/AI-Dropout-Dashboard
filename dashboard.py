# dashboard.py
# Streamlit dashboard for Drop-out Prediction and Counseling
# Run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Dropout Risk Dashboard", layout="wide")

# -----------------------
# Utility: generate sample data if CSVs are missing
# -----------------------
def generate_sample_data(n=150, seed=42):
    np.random.seed(seed)
    student_ids = [f"S{1000+i}" for i in range(n)]
    names = [f"Student_{i}" for i in range(n)]
    mentors = [f"mentor_{i%6}" for i in range(n)]
    courses = [f"CE-{(i%4)+1}" for i in range(n)]

    attendance = np.clip(np.random.normal(80, 12, n).round(1), 30, 100)
    failed_attempts = np.random.choice([0,0,0,1,1,2], n, p=[0.5,0.2,0.1,0.1,0.06,0.04])

    base_date = datetime.now()
    last_payment_days_ago = np.random.choice(range(0,120), n)
    last_payment_date = [(base_date - timedelta(days=int(d))).date() for d in last_payment_days_ago]

    attendance_df = pd.DataFrame({
        "student_id": student_ids,
        "name": names,
        "course": courses,
        "mentor": mentors,
        "attendance_percent": attendance
    })

    scores_df = pd.DataFrame({
        "student_id": student_ids,
        "assessment_1": np.clip(np.random.normal(60, 22, n).round(1), 0, 100),
        "assessment_2": np.clip(np.random.normal(62, 20, n).round(1), 0, 100),
        "assessment_3": np.clip(np.random.normal(58, 25, n).round(1), 0, 100),
        "failed_attempts": failed_attempts
    })
    scores_df["avg_score"] = scores_df[["assessment_1","assessment_2","assessment_3"]].mean(axis=1).round(1)

    fees_df = pd.DataFrame({
        "student_id": student_ids,
        "total_fee_due": np.random.choice([0,5000,10000], n, p=[0.6,0.25,0.15]),
        "last_payment_date": last_payment_date
    })

    return attendance_df, scores_df, fees_df

# -----------------------
# Rule engine
# -----------------------
def evaluate_risk(df, thresholds):
    labels = []
    flags = []
    score_map = {"green":0,"amber":1,"red":2}
    risk_scores = []

    for _, r in df.iterrows():
        f = {}

        # attendance
        if r["attendance_percent"] < thresholds["attendance_red"]:
            f["attendance"] = "red"
        elif r["attendance_percent"] < thresholds["attendance_amber"]:
            f["attendance"] = "amber"
        else:
            f["attendance"] = "green"

        # score
        if r["avg_score"] < thresholds["score_red"]:
            f["score"] = "red"
        elif r["avg_score"] < thresholds["score_amber"]:
            f["score"] = "amber"
        else:
            f["score"] = "green"

        # attempts
        if r["failed_attempts"] >= thresholds["failed_attempts_red"]:
            f["attempts"] = "red"
        elif r["failed_attempts"] >= thresholds["failed_attempts_amber"]:
            f["attempts"] = "amber"
        else:
            f["attempts"] = "green"

        # fees
        fod = r.get("fees_overdue_days", 999)
        if fod >= thresholds["fees_overdue_days_red"]:
            f["fees"] = "red"
        elif fod >= thresholds["fees_overdue_days_amber"]:
            f["fees"] = "amber"
        else:
            f["fees"] = "green"

        # combined label
        if "red" in f.values():
            label = "Red"
        elif "amber" in f.values():
            label = "Amber"
        else:
            label = "Green"

        rs = sum(score_map[v] for v in f.values())
        labels.append(label)
        flags.append(f)
        risk_scores.append(rs)

    df = df.copy()
    df["rule_label"] = labels
    df["rule_flags"] = flags
    df["rule_risk_score"] = risk_scores
    return df

# -----------------------
# Data loading / merging
# -----------------------
@st.cache_data
def load_and_merge(att_path=None, scores_path=None, fees_path=None):
    if att_path and scores_path and fees_path:
        try:
            att = pd.read_csv(att_path)
            sc = pd.read_csv(scores_path)
            fees = pd.read_csv(fees_path)

            if "last_payment_date" in fees.columns:
                fees["last_payment_date"] = pd.to_datetime(fees["last_payment_date"], errors="coerce")

        except Exception as e:
            st.warning("Could not read provided CSVs. Falling back to sample data. Error: " + str(e))
            att, sc, fees = generate_sample_data()
    else:
        att, sc, fees = generate_sample_data()

    df = att.merge(sc, on="student_id", how="outer")
    df = df.merge(fees, on="student_id", how="outer")

    today = pd.to_datetime(datetime.now().date())
    df["attendance_percent"] = pd.to_numeric(df.get("attendance_percent", 0), errors="coerce").fillna(0)
    df["avg_score"] = pd.to_numeric(df.get("avg_score", 0), errors="coerce").fillna(0)
    df["failed_attempts"] = pd.to_numeric(df.get("failed_attempts", 0), errors="coerce").fillna(0).astype(int)
    df["last_payment_date"] = pd.to_datetime(df.get("last_payment_date"), errors="coerce")
    df["fees_overdue_days"] = (today - df["last_payment_date"]).dt.days.fillna(999)

    cols = ["student_id","name","course","mentor","attendance_percent","avg_score","failed_attempts","fees_overdue_days"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# -----------------------
# Sidebar: file upload + thresholds
# -----------------------
st.sidebar.header("Data Input")
uploaded_att = st.sidebar.file_uploader("Upload attendance CSV", type=["csv"])
uploaded_scores = st.sidebar.file_uploader("Upload scores CSV", type=["csv"])
uploaded_fees = st.sidebar.file_uploader("Upload fees CSV", type=["csv"])

use_uploaded = uploaded_att and uploaded_scores and uploaded_fees
process_now = st.sidebar.button("Process Uploaded Files")

st.sidebar.header("Risk Thresholds")
attendance_red = st.sidebar.number_input("Attendance red threshold (below)", min_value=0, max_value=100, value=65)
attendance_amber = st.sidebar.number_input("Attendance amber threshold (below)", min_value=0, max_value=100, value=75)
score_red = st.sidebar.number_input("Score red threshold (below)", min_value=0, max_value=100, value=35)
score_amber = st.sidebar.number_input("Score amber threshold (below)", min_value=0, max_value=100, value=50)
failed_red = st.sidebar.number_input("Failed attempts red (>=)", min_value=0, value=2)
failed_amber = st.sidebar.number_input("Failed attempts amber (>=)", min_value=0, value=1)
fees_red = st.sidebar.number_input("Fees overdue days red (>=)", min_value=0, value=30)
fees_amber = st.sidebar.number_input("Fees overdue days amber (>=)", min_value=0, value=7)

thresholds = {
    "attendance_red": attendance_red,
    "attendance_amber": attendance_amber,
    "score_red": score_red,
    "score_amber": score_amber,
    "failed_attempts_red": failed_red,
    "failed_attempts_amber": failed_amber,
    "fees_overdue_days_red": fees_red,
    "fees_overdue_days_amber": fees_amber
}

# -----------------------
# Load and process data
# -----------------------
if use_uploaded and process_now:
    df = load_and_merge(uploaded_att, uploaded_scores, uploaded_fees)
else:
    df = load_and_merge()

df = evaluate_risk(df, thresholds)

# -----------------------
# Dashboard display
# -----------------------
st.title("Dropout Risk Dashboard")
st.markdown("A simple, transparent dashboard to spot students at risk early.")

# top metrics
total = len(df)
counts = df["rule_label"].value_counts().reindex(["Red","Amber","Green"]).fillna(0).astype(int)
c1, c2, c3, c4 = st.columns([1,1,1,2])
c1.metric("Total students", total)
c2.metric("Red", counts["Red"])
c3.metric("Amber", counts["Amber"])
c4.metric("Green", counts["Green"])

# -----------------------
# Filters
# -----------------------
with st.expander("Filters"):
    mentors = sorted(df["mentor"].dropna().unique().tolist())
    mentors_sel = st.multiselect("Mentor", ["All"] + mentors, default=["All"])
    courses = sorted(df["course"].dropna().unique().tolist())
    course_sel = st.multiselect("Course", ["All"] + courses, default=["All"])
    label_sel = st.multiselect("Risk label", ["Red","Amber","Green"], default=["Red","Amber","Green"])

df_view = df.copy()
if "All" not in mentors_sel:
    df_view = df_view[df_view["mentor"].isin(mentors_sel)]
if "All" not in course_sel:
    df_view = df_view[df_view["course"].isin(course_sel)]
df_view = df_view[df_view["rule_label"].isin(label_sel)]

# -----------------------
# Show table with colored label column
# -----------------------
display_cols = ["student_id","name","mentor","course","attendance_percent","avg_score","failed_attempts","fees_overdue_days","rule_label"]
disp = df_view[display_cols].copy()

def label_badge(label):
    if label == "Red":
        color = "#ff4b4b"
    elif label == "Amber":
        color = "#ffb84d"
    else:
        color = "#66c266"
    return f'<div style="background:{color};padding:6px;border-radius:6px;text-align:center;color:#000;font-weight:600">{label}</div>'

disp["label_html"] = disp["rule_label"].apply(label_badge)
disp_for_table = disp.drop(columns=["rule_label"])

st.subheader("Students table")
st.markdown("You can sort the table columns. Click a row to view details below.")
st.dataframe(disp_for_table.style.format({"attendance_percent": "{:.1f}", "avg_score": "{:.1f}"}), height=420)

# -----------------------
# Student details
# -----------------------
st.subheader("Student details")
selected = st.text_input("Enter student_id to see flags and notes (e.g. S1000)", value="")
if selected:
    sel = df[df["student_id"] == selected]
    if sel.empty:
        st.info("Student id not found in current dataset.")
    else:
        r = sel.iloc[0]
        st.write(r[["student_id","name","mentor","course","attendance_percent","avg_score","failed_attempts","fees_overdue_days"]])
        st.write("Risk label:", r["rule_label"])
        st.write("Rule flags:")
        for k,v in r["rule_flags"].items():
            st.write("-", k, ":", v)

# -----------------------
# Charts
# -----------------------
st.subheader("Distributions and insights")
col_a, col_b = st.columns(2)
with col_a:
    st.caption("Attendance distribution")
    st.bar_chart(df["attendance_percent"].dropna().astype(float).value_counts(bins=20).sort_index())
with col_b:
    st.caption("Average score distribution")
    st.bar_chart(df["avg_score"].dropna().astype(float).value_counts(bins=20).sort_index())

# quick pivot
flag_rows = []
for _, row in df.iterrows():
    f = row["rule_flags"]
    for k,v in f.items():
        flag_rows.append({"student_id": row["student_id"], "reason": k, "level": v})
flag_df = pd.DataFrame(flag_rows)
pivot = flag_df.groupby(["reason","level"]).size().unstack(fill_value=0)
st.subheader("Flag counts (by reason)")
st.table(pivot)

# -----------------------
# Actions
# -----------------------
st.subheader("Actions")
red_df = df[df["rule_label"] == "Red"]
csv = red_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Red list as CSV", csv, file_name="red_list.csv", mime="text/csv")

if st.button("Generate fresh sample data (in memory)"):
    att, sc, fees = generate_sample_data()
    st.success("Generated new sample data. Reload the app to see updated dataset.")

st.info("Thresholds are editable in the left sidebar. Change them to tune alerts. For real data, upload the three CSVs at the top of the sidebar.")
