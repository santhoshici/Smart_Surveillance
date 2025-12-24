import streamlit as st
import requests
from PIL import Image
import io
import time


st.set_page_config(page_title="Suspicious Activity Dashboard", layout="wide")
st.title(" Suspicious Activity Detection Dashboard")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value="http://localhost:5000/api", key="api_url")
    refresh_rate = st.slider("Refresh Rate (ms)", 100, 2000, 500, step=100)
    st.divider()
    st.info("Make sure the backend is running before using this dashboard.")

col1, col2 = st.columns([3, 1])

with col1:
    stframe = st.empty()

with col2:
    st.subheader("Statistics")
    stats_container = st.empty()
    alert_container = st.empty()

status_text = st.empty()


def get_status(url):
    """Fetch status from backend"""
    try:
        response = requests.get(f"{url}/status", timeout=2)
        return response.json()
    except Exception as e:
        return None


def get_frame(url):
    """Fetch current frame from backend"""
    try:
        response = requests.get(f"{url}/frame", timeout=2)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except Exception as e:
        pass
    return None


print("Starting Streamlit dashboard...")
status_text.success("✓ Dashboard ready. Connecting to backend...")

while True:
    try:
        data = get_status(api_url)
        frame = get_frame(api_url)

        if frame is not None:
            stframe.image(
                frame, channels="RGB", use_column_width=True, caption="Live Feed"
            )
        else:
            stframe.warning("Cannot connect to camera feed")

        if data:
            with stats_container.container():
                st.metric("Activity", data["label"].upper())
                st.metric("Confidence", f"{data['confidence'] * 100:.1f}%")
                st.metric("FPS", f"{data['fps']:.1f}")

            if data.get("alert"):
                alert_container.error("🚨 SUSPICIOUS ACTIVITY DETECTED!", icon="⚠️")
            else:
                alert_container.empty()

            status_text.success(
                f"✓ Connected | Last update: {time.strftime('%H:%M:%S')}"
            )
        else:
            status_text.warning("Waiting for backend or camera connection...")

    except Exception as e:
        status_text.error(f"Error: {str(e)}")

    # Refresh rate
    time.sleep(refresh_rate / 1000.0)
