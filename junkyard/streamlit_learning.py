import io

import cv2 as cv
import requests
import streamlit as st
from fastapi import HTTPException

API_BASE_URL = "http://127.0.0.1:8000"

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user" not in st.session_state:
    st.session_state.user = None


def login(username: str, password: str):
    """Login and Get access token."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/token",
            data={"username": username, "password": password, "grant_type": "password"},
        )

        if response.status_code == 200:
            token_data = response.json()
            st.session_state.access_token = token_data["access_token"]
            return True, "Login Successful"
        else:
            return False, f"Login Failed: {response.text}"
    except HTTPException as e:
        return False, f"Error: {str(e)}"
