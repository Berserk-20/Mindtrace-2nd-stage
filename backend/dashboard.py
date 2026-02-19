import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MindTrace",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"   # login | signup


# ----------------------------------------------------------
# API HELPERS
# ----------------------------------------------------------
def api_get(endpoint, params=None):
    return requests.get(f"{API}{endpoint}", params=params or {}).json()

def api_post(endpoint, payload=None):
    return requests.post(f"{API}{endpoint}", json=payload or {}).json()


# ----------------------------------------------------------
# AUTH PAGES
# ----------------------------------------------------------
def signup_page():
    st.title("🧠 MindTrace – Create Account")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if not username or not password:
            st.error("All fields are required")
            return

        if password != confirm:
            st.error("Passwords do not match")
            return

        res = api_post("/admin/create-user", {
            "username": username,
            "password": password,
            "role": "user"
        })

        if res.get("success"):
            st.success("Account created successfully. Please login.")
            st.session_state.auth_page = "login"
            st.rerun()
        else:
            st.error("Username already exists")

    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.rerun()


def login_page():
    st.title("🧠 MindTrace – Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            res = api_post("/login", {
                "username": username,
                "password": password
            })

            if res.get("success"):
                st.session_state.user = res["user"]
                st.rerun()
            else:
                st.error("Invalid credentials")

    with col2:
        if st.button("Create new account"):
            st.session_state.auth_page = "signup"
            st.rerun()


# ----------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------
def dashboard():
    user = st.session_state.user
    role = user["role"]

    # ================= SIDEBAR =================
    st.sidebar.title("🧠 MindTrace")

    pages = ["🎥 Record Session", "📊 Analytics Dashboard"]
    if role == "admin":
        pages.append("👑 Admin Panel")

    page = st.sidebar.radio("Navigation", pages)

    if st.sidebar.button("Logout"):
        api_post("/stop")
        st.session_state.user = None
        st.rerun()

    # ======================================================
    # RECORD SESSION
    # ======================================================
    if page == "🎥 Record Session":
        st.header("🎥 Live Session Control")

        c1, c2, c3, c4 = st.columns(4)
        c1.button("▶ Start", on_click=lambda: api_post("/start", {"user_id": user["user_id"]}))
        c2.button("⏸ Pause", on_click=lambda: api_post("/pause"))
        c3.button("▶ Resume", on_click=lambda: api_post("/resume"))
        c4.button("⏹ Stop", on_click=lambda: api_post("/stop"))

        st.divider()
        st.image(f"{API}/video")

    # ======================================================
    #  ANALYTICS DASHBOARD
    # ======================================================
    elif page == "📊 Analytics Dashboard":
        st.markdown("## 🧠 **MindTrace – Emotion, Focus & Fatigue Monitoring**")

        sessions = api_get("/sessions", {
            "user_id": user["user_id"],
            "role": role
        })

        if not sessions:
            st.info("No sessions available.")
            return

        df_sessions = pd.DataFrame(sessions)

        if "session_id" not in df_sessions.columns:
            st.error("Session data is invalid.")
            return

        session_id = st.selectbox(
            "Select Session",
            df_sessions["session_id"]
        )

        data = api_get("/emotions", {"session_id": session_id})
        if not data:
            st.warning("No emotion data for this session.")
            return

        df = pd.DataFrame(data)

        if "timestamp" not in df.columns or "emotion" not in df.columns:
            st.error("Invalid emotion data format.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 🔧 NORMALIZE focus_level
        if "focus_level" not in df.columns:
            if "focus" in df.columns:
                df["focus_level"] = df["focus"]
            elif "confidence" in df.columns:
                df["focus_level"] = df["confidence"]
            else:
                df["focus_level"] = 0.0

        # ---------------- KPIs ----------------
        focus_pct = round(df["focus_level"].mean() * 100, 2)
        fatigue_pct = round(100 - focus_pct, 2)

        k1, k2, k3 = st.columns(3)
        k1.metric("🎯 Focus %", f"{focus_pct}%")
        k2.metric("😴 Fatigue %", f"{fatigue_pct}%")
        k3.metric("📊 Total Records", len(df))

        st.divider()

        # ---------------- EMOTION DISTRIBUTION ----------------
        st.subheader("Emotion Distribution")

        emo_counts = df["emotion"].value_counts().reset_index()
        emo_counts.columns = ["emotion", "count"]

        fig_pie = px.pie(
            emo_counts,
            names="emotion",
            values="count",
            hole=0.55
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # ---------------- EMOTION TIMELINE ----------------
        st.subheader("Emotion Timeline")

        emo_map = {
            "Neutral": 0,
            "Happy": 1,
            "Angry": 2,
            "Sad": 3,
            "Fear": 4,
            "Surprise": 5,
            "Disgust": 6
        }

        df["emotion_code"] = df["emotion"].map(emo_map)

        fig_timeline = px.scatter(
            df,
            x="timestamp",
            y="emotion_code",
            color="emotion",
            height=400
        )

        fig_timeline.update_yaxes(
            tickvals=list(emo_map.values()),
            ticktext=list(emo_map.keys()),
            title="Emotion"
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

        # ---------------- INSIGHTS ----------------
        st.subheader("⚠️ Insights")

        if focus_pct >= 70:
            st.success("✅ Good focus maintained")
        elif focus_pct >= 40:
            st.warning("⚠️ Moderate focus detected")
        else:
            st.error("🚨 Low focus detected")

        # ---------------- RAW DATA ----------------
        with st.expander("📄 Raw Data"):
            st.dataframe(
                df[["timestamp", "emotion", "focus_level"]],
                use_container_width=True
            )

    # ======================================================
    #  ADMIN PANEL
    # ======================================================
    elif page == "👑 Admin Panel":
        st.markdown("## 👑 **Admin Panel – System Overview**")

        st.subheader("👥 Registered Users")
        users = api_get("/admin/user-summary")
        if users:
            st.dataframe(pd.DataFrame(users), use_container_width=True)
        else:
            st.info("No users found.")

        st.divider()

        st.subheader("📊 All Sessions")
        sessions = api_get("/sessions", {"role": "admin"})
        if sessions:
            st.dataframe(pd.DataFrame(sessions), use_container_width=True)
        else:
            st.info("No sessions available.")

        st.divider()

        st.subheader("🚨 System Notifications")
        alerts = api_get("/notifications")
        if alerts:
            df_alerts = pd.DataFrame(alerts)
            df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"])
            st.dataframe(
                df_alerts.sort_values("timestamp", ascending=False),
                use_container_width=True
            )
        else:
            st.success("No alerts generated.")


# ----------------------------------------------------------
# ROUTER
# ----------------------------------------------------------
if st.session_state.user is None:
    if st.session_state.auth_page == "login":
        login_page()
    else:
        signup_page()
else:
    dashboard()
