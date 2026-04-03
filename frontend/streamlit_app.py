from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ------------------------------
# Session state
# ------------------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "base_url" not in st.session_state:
    st.session_state.base_url = "http://127.0.0.1:8000"

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Product Recommender AI",
    page_icon=f"{st.session_state.base_url}/favicon",
    layout="wide",
)


# ------------------------------
# Helper functions
# ------------------------------
def get_headers() -> Dict[str, str]:
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return headers


def api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
) -> Dict[str, Any]:
    url = f"{st.session_state.base_url}{endpoint}"
    headers = get_headers()

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            if files:
                # Let requests set multipart boundary; do not set Content-Type manually
                response = requests.post(url, headers=headers, data=data, files=files)
            else:
                response = requests.post(url, headers=headers, data=data)
        else:
            raise ValueError("Unsupported method")

        response.raise_for_status()
        return response.json() if response.content else {}
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        if e.response is not None:
            try:
                st.error(e.response.json())
            except Exception as exp:
                st.error(
                    f"Request Exception:\t{e.response.text}\nResponse Exception:\t{exp}"
                )
        return {}


def login(username: str, password: str) -> bool:
    data = {"username": username, "password": password, "grant_type": "password"}
    result = api_request("POST", "/token", data=data)
    if "access_token" in result:
        st.session_state.token = result["access_token"]
        st.session_state.username = username
        return True
    return False


def register(username: str, password: str, email: str, full_name: str) -> bool:
    data = {
        "username": username,
        "password": password,
        "email": email,
        "full_name": full_name,
    }
    result = api_request("POST", "/users/register", data=data)
    return "username" in result


def classify_image(image_file) -> Optional[Dict]:
    """Send image for classification with correct MIME type."""
    files = {
        "uploaded_image": (
            image_file.name,
            image_file.getvalue(),  # read bytes
            image_file.type if image_file.type else "image/jpeg",
        )
    }
    return api_request("POST", "/model/image_classification", files=files)


def analyze_sentiment(review: str) -> Optional[Dict]:
    data = {"review": review}
    return api_request("POST", "/model/sentiment/form", data=data)


def get_model_info() -> Optional[Dict]:
    return api_request("GET", "/model/info")


# ------------------------------
# Sidebar: Authentication
# ------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.title("Product Recommender")

    base_url = st.text_input("Backend URL", value=st.session_state.base_url)
    if base_url != st.session_state.base_url:
        st.session_state.base_url = base_url
        st.rerun()

    st.divider()

    if not st.session_state.token:
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", use_container_width=True):
            if login_user and login_pass and login(login_user, login_pass):
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.divider()
        st.subheader("Register")
        reg_user = st.text_input("Username*", key="reg_user")
        reg_pass = st.text_input("Password*", type="password", key="reg_pass")
        reg_email = st.text_input("Email", key="reg_email")
        reg_full = st.text_input("Full name", key="reg_full")
        if st.button("Register", use_container_width=True):
            if reg_user and reg_pass:
                if register(reg_user, reg_pass, reg_email, reg_full):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Registration failed (username may exist).")
            else:
                st.warning("Username and password required")
    else:
        st.success(f"Logged in as: {st.session_state.username}")
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()

# ------------------------------
# Main area
# ------------------------------
st.title("🛍️ AI Product Recommender")
st.markdown(
    "Upload a product image and provide a review. The AI will predict the product category and analyze sentiment, then give a recommendation."
)

if not st.session_state.token:
    st.warning("Please log in to use the recommender.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Product Image")
    uploaded_image = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear product image",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Product", use_container_width=True)

with col2:
    st.subheader("📝 Product Review")
    review_text = st.text_area(
        "Write your review",
        height=200,
        placeholder="Example: This product is amazing! High quality and fast shipping.",
    )

if st.button("🔍 Analyze & Recommend", type="primary", use_container_width=True):
    if not uploaded_image:
        st.error("Please upload a product image.")
    elif not review_text.strip():
        st.error("Please write a product review.")
    else:
        with st.spinner("Calling AI models..."):
            image_result = classify_image(uploaded_image)
            sentiment_result = analyze_sentiment(review_text)

        if not image_result or "prediction_result" not in image_result:
            st.error("Image classification failed. Check API logs.")
        elif not sentiment_result or "predictions" not in sentiment_result:
            st.error("Sentiment analysis failed.")
        else:
            # --- Correctly extract category name and confidence from new API ---
            pred_category = image_result["prediction_result"][
                "category"
            ]  # string, e.g., "Footwear"
            img_conf = image_result["prediction_result"]["confidence"]  # float
            categories = image_result.get("all_possible_categories", [])

            sentiment = sentiment_result["predictions"].get("sentiment", "unknown")
            sentiment_conf = sentiment_result["predictions"].get("confidence", 0.0)

            # Recommendation logic (adjust threshold as needed)
            is_recommended = (sentiment == "positive") and (img_conf > 0.5)

            st.divider()
            st.subheader("📊 Prediction Results")

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric(
                    "Product Category",
                    pred_category,
                    delta=f"Confidence: {img_conf:.1%}",
                )
            with col_r2:
                st.metric(
                    "Sentiment",
                    sentiment.capitalize(),
                    delta=f"Confidence: {sentiment_conf:.1%}",
                )
            with col_r3:
                if is_recommended:
                    st.success("✅ **RECOMMENDED**")
                    st.markdown("This product meets our criteria!")
                else:
                    st.error("❌ **NOT RECOMMENDED**")
                    st.markdown("Low confidence or negative sentiment.")

            # Confidence bar chart
            st.subheader("📈 Confidence Scores")
            conf_data = pd.DataFrame(
                {
                    "Metric": ["Product Category", "Sentiment"],
                    "Confidence": [img_conf, sentiment_conf],
                }
            )
            fig = px.bar(
                conf_data,
                x="Metric",
                y="Confidence",
                color="Metric",
                range_y=[0, 1],  # fixed: use range_y instead of range
                text=conf_data["Confidence"].apply(lambda x: f"{x:.1%}"),
                title="Prediction Confidence",
            )
            fig.add_hline(
                y=0.6,
                line_dash="dash",
                line_color="red",
                annotation_text="Recommendation threshold",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Optional: alternative predictions (image)
            with st.expander("🔍 Alternative predictions (image)"):
                for alt in image_result.get("alternative_predictions", []):
                    alt_name = alt.get(
                        "category_name", f"Class {alt.get('category_index')}"
                    )
                    st.write(f"- **{alt_name}**: {alt['confidence_score']:.1%}")

            with st.expander("📄 Raw API Responses"):
                st.json(image_result)
                st.json(sentiment_result)
