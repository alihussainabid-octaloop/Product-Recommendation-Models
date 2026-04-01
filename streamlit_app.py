import io

import requests
import streamlit as st
from PIL import Image

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Change this to your API URL

# Initialize session state
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user" not in st.session_state:
    st.session_state.user = None


def login(username, password):
    """Login and get access token"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/token",
            data={"username": username, "password": password, "grant_type": "password"},
        )
        if response.status_code == 200:
            token_data = response.json()
            st.session_state.access_token = token_data["access_token"]
            return True, "Login successful!"
        else:
            return False, f"Login failed: {response.text}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def register(username, password, email, full_name):
    """Register new user"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/users/register",
            data={
                "username": username,
                "password": password,
                "email": email,
                "full_name": full_name,
            },
        )
        if response.status_code == 201:
            return True, "Registration successful! Please login."
        else:
            return False, f"Registration failed: {response.text}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_current_user():
    """Get current user info"""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.get(f"{API_BASE_URL}/users/me/", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None


def get_model_info():
    """Get model status/info from backend"""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.get(f"{API_BASE_URL}/model/info", headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def classify_image(image_bytes):
    """Classify image"""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        files = {"uploaded_image": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(
            f"{API_BASE_URL}/model/image_classification", headers=headers, files=files
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Classification failed: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def logout():
    """Logout user"""
    st.session_state.access_token = None
    st.session_state.user = None
    st.rerun()


def main():
    st.set_page_config(
        page_title="Image Classification App", page_icon="🖼️", layout="wide"
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .stButton > button {
        width: 100%;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        "<h1 class='main-header'>🖼️ Image Classification App</h1>",
        unsafe_allow_html=True,
    )

    # Sidebar for authentication
    with st.sidebar:
        st.header("🔐 Authentication")

        if st.session_state.access_token is None:
            # Login/Register Tabs
            tab1, tab2 = st.tabs(["Login", "Register"])

            with tab1:
                st.subheader("Login")
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input(
                    "Password", type="password", key="login_password"
                )

                if st.button("Login", type="primary"):
                    if login_username and login_password:
                        success, message = login(login_username, login_password)
                        if success:
                            st.success(message)
                            user_data = get_current_user()
                            if user_data:
                                st.session_state.user = user_data
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter both username and password")

            with tab2:
                st.subheader("Register")
                reg_username = st.text_input("Username", key="reg_username")
                reg_email = st.text_input("Email", key="reg_email")
                reg_full_name = st.text_input("Full Name", key="reg_full_name")
                reg_password = st.text_input(
                    "Password", type="password", key="reg_password"
                )
                reg_confirm_password = st.text_input(
                    "Confirm Password", type="password", key="reg_confirm_password"
                )

                if st.button("Register"):
                    if all([reg_username, reg_email, reg_full_name, reg_password]):
                        if reg_password == reg_confirm_password:
                            success, message = register(
                                reg_username, reg_password, reg_email, reg_full_name
                            )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        else:
                            st.warning("Passwords do not match")
                    else:
                        st.warning("Please fill all fields")

        else:
            # User info when logged in
            if st.session_state.user:
                st.success(
                    f"✅ Logged in as: **{st.session_state.user.get('full_name', st.session_state.user.get('username'))}**"
                )
                st.info(
                    f"📧 Email: {st.session_state.user.get('email', 'Not provided')}"
                )

            if st.button("🚪 Logout"):
                logout()

    # Main content area
    if st.session_state.access_token is None:
        # Show welcome message for non-logged in users
        st.markdown("""
        ### Welcome to the Image Classification App! 🎉

        This application allows you to:
        - **Upload images** for AI-powered classification
        - **Get instant predictions** about the content of your images
        - **Track your classification history**

        **Please login or register to get started!**

        #### Features:
        - 🔐 Secure authentication
        - 🖼️ Support for various image formats (JPG, PNG, etc.)
        - 🤖 AI-powered image classification
        - 📊 Detailed classification results
        """)
    else:
        # Show classification interface for logged-in users
        st.header("📤 Upload Image for Classification")

        # Show model info under auth state
        model_info = get_model_info()
        if model_info:
            st.success("Model is loaded and ready")

            # Render predictable categories table
            class_names = model_info.get("class_names", [])
            if class_names:
                st.markdown("### Possible Predictable categories")
                class_df = [{"Category": name} for name in class_names]
                st.table(class_df)

            # Display input shape requirement
            input_shape = model_info.get("input_shape")
            if input_shape:
                st.info(
                    f"⚙️ Required model input size: {input_shape}. "
                    "Uploads are resized by backend to this shape automatically."
                )
            else:
                st.info(
                    "⚙️ Input size info not available from model info. "
                    "Backend does resize images to expected dimensions."
                )

            # Keep raw model info for debugging
            # st.markdown("---")
            # st.markdown("**Raw model info**")
            # st.json(model_info)

        else:
            st.warning("Model info unavailable: backend may be starting or token invalid")

        # Create two columns for better layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Upload an image file to classify",
            )

            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Prepare for classification
                if st.button(
                    "🔍 Classify Image", type="primary", use_container_width=True
                ):
                    with st.spinner("Classifying image..."):
                        # Convert image to bytes
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format="JPEG")
                        img_bytes = img_bytes.getvalue()

                        # Call classification API
                        success, result = classify_image(img_bytes)

                        if success:
                            st.session_state.last_result = result
                            st.success("Classification completed!")
                        else:
                            st.error(result)

        with col2:
            # Display classification results
            if "last_result" in st.session_state and st.session_state.last_result:
                st.subheader("📊 Classification Results")

                result = st.session_state.last_result

                if isinstance(result, dict):
                    def resolve_category_name(category, idx, all_categories):
                        if category and isinstance(category, str):
                            if category.startswith("Class_") and all_categories:
                                try:
                                    i = int(category.split("Class_")[-1])
                                    return all_categories[i]
                                except Exception:
                                    pass
                            return category

                        if isinstance(idx, (int, float)) and all_categories:
                            try:
                                return all_categories[int(idx)]
                            except Exception:
                                pass

                        return None

                    pred = result.get("prediction_result") or {}
                    alt = result.get("alternative_predictions", [])
                    all_cats = result.get("all_possible_categories", [])

                    st.markdown("**Top prediction:**")
                    if pred:
                        predicted_name = resolve_category_name(
                            pred.get("category"),
                            pred.get("category_index"),
                            all_cats,
                        )
                        confidence_value = pred.get("confidence", 0)

                        if not predicted_name:
                            predicted_name = "Unknown"

                        st.markdown(
                            f"- Category: **{predicted_name}**\n- Confidence: **{confidence_value:.2%}**"
                        )
                    else:
                        st.info("No main prediction found, showing raw response")

                    if alt:
                        st.markdown("**Other candidate labels:**")
                        alt_df = [
                            {
                                "Category": resolve_category_name(
                                    x.get("category_name") or x.get("category"),
                                    x.get("category_index"),
                                    all_cats,
                                )
                                or "-",
                                "Confidence": f"{x.get('confidence_score', 0):.2%}",
                            }
                            for x in alt
                        ]
                        st.table(alt_df)

                else:
                    st.write(result)

                # Add a button to clear results
                if st.button("Clear Results", use_container_width=True):
                    del st.session_state.last_result
                    st.rerun()
            else:
                st.info(
                    "No classification results yet. Upload an image and click 'Classify Image' to get started."
                )

        # Add a section for classification history (if you want to implement it)
        st.markdown("---")
        st.caption(
            "💡 Tip: The classification model will analyze your image and provide predictions based on its training."
        )


if __name__ == "__main__":
    main()
