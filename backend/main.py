import io
import json
import os
import pickle
import re
import string
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
import nltk
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
from pwdlib import PasswordHash
from pydantic import BaseModel
from starlette.responses import FileResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==============================
# Configuration
# ==============================
SECRET_KEY = "7c5902732404d7970efa708bafc0f8ce5d5b391c57e02c1000d9ca61bf561cdf"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# ==============================
# Pydantic Models
# ==============================
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: str | None = None
    full_name: str | None = None

    @classmethod
    def as_form(
        cls,
        username: str = Form(...),
        password: str = Form(...),
        email: str = Form(None),
        full_name: str = Form(None),
    ):
        return cls(
            username=username,
            password=password,
            email=email,
            full_name=full_name,
        )


class SentimentRequest(BaseModel):
    review: str


# ==============================
# Fake User DB & Password Hashing
# ==============================
try:
    with open("fake_users_db.json", "r") as opened_file:
        fake_users_db = json.load(opened_file)
except FileNotFoundError:
    fake_users_db = {}

password_hash = PasswordHash.recommended()
DUMMY_HASH = password_hash.hash("dummypassword")


def verify_password(plain_password: str, hashed_password: str):
    return password_hash.verify(plain_password, hashed_password)


def get_hashed_password(password: str):
    return password_hash.hash(password)


def get_user(db: dict, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        verify_password(plain_password=password, hashed_password=DUMMY_HASH)
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# ==============================
# Authentication Dependencies
# ==============================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credential_exception = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail="Could not Validate Credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, key=SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credential_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credential_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credential_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive User",
        )
    return current_user


# ==============================
# Image Model Manager (Weights‑only)
# ==============================
class ModelManager:
    _instance = None
    _model = None
    _class_names = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, weights_path: str, class_names_path: str):
        """
        Rebuild the MobileNetV2 architecture and load trained weights.
        This avoids any 'training' flag issues from saved .keras files.
        """
        if self._model is not None:
            return

        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2

        # 1. Load class names
        with open(class_names_path, "rb") as f:
            self._class_names = pickle.load(f)
        num_classes = len(self._class_names)
        print(
            f"📦 Loading image classifier for {num_classes} classes: {self._class_names}"
        )

        # 2. Recreate the inference model (same as training but without augmentation)
        base_model = MobileNetV2(
            input_shape=(128, 128, 3), include_top=False, weights="imagenet"
        )
        base_model.trainable = False

        model = models.Sequential(
            [
                layers.Input(shape=(80, 60, 3)),
                layers.Resizing(128, 128, interpolation="bilinear"),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # 3. Load the trained weights
        model.load_weights(weights_path)
        print(f"✅ Weights loaded from {weights_path}")

        # 4. Compile (not strictly needed for inference)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self._model = model
        print("✅ Image classification model ready for inference.")

    def predict(self, image_array: np.ndarray):
        if self._model is None:
            raise Exception("Model not loaded")
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        predictions = self._model.predict(image_array, verbose=0)
        return predictions[0]  # 1D probability array

    @property
    def class_names(self):
        return self._class_names if self._class_names is not None else []

    @property
    def num_classes(self):
        return len(self._class_names) if self._class_names else 0


# ==============================
# Sentiment Model Manager (unchanged)
# ==============================
class SentimentModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    _id2label = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str = "models/sentiment_model"):
        if self._model is not None:
            return

        # NLTK resources (for preprocessing)
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass

        # Load from directory (Transformers format)
        if os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        ):
            print(f"Loading sentiment model from directory: {model_path}")
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._id2label = {
                int(k): v
                for k, v in getattr(self._model.config, "id2label", {}).items()
            }
            if not self._id2label:
                self._id2label = {0: "negative", 1: "neutral", 2: "positive"}
            print("✅ Sentiment model loaded from directory")
            return

        raise FileNotFoundError(f"Sentiment model not found at {model_path}")

    @staticmethod
    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}
        lemmatizer = WordNetLemmatizer()
        words = [
            lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in stop_words
        ]
        return " ".join(words)

    def predict(self, raw_text: str):
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Sentiment model is not loaded")
        clean_text = self.preprocess_text(raw_text)
        inputs = self._tokenizer(
            clean_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        predicted_idx = int(np.argmax(probs))
        predicted_label = self._id2label.get(predicted_idx, str(predicted_idx))
        return {
            "review": raw_text,
            "processed_review": clean_text,
            "predicted_label": predicted_label,
            "predicted_index": predicted_idx,
            "confidence": float(probs[predicted_idx]),
            "probabilities": probs.tolist(),
            "id2label": self._id2label,
        }


# ==============================
# FastAPI App & Startup
# ==============================
app = FastAPI()
model_manager = ModelManager()
sentiment_model_manager = SentimentModelManager()


@app.on_event("startup")
async def startup_event():
    # Load sentiment model
    try:
        sentiment_model_manager.load_model("./models/sentiment_model")
        print("✅ Sentiment model loaded successfully on startup")
    except Exception as e:
        print(f"❌ Failed to load sentiment model: {e}")

    # Load image classification model (weights + architecture)
    try:
        model_manager.load_model(
            weights_path="./models/model_weights.weights.h5",
            class_names_path="./models/class_names_mobilenet.pkl",
        )
        print("✅ Image model loaded successfully on startup")
    except Exception as e:
        print(f"❌ Failed to load Image Model: {e}")


# ==============================
# Helper: Preprocess image for TF
# ==============================
def preprocess_image_for_tensorflow(image_bytes: bytes, target_size=(80, 60)):
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(pil_image, dtype=np.float32) / 255.0
    return image_array


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


# ==============================
# Public Endpoints
# ==============================
@app.get("/favicon", include_in_schema=True)
async def favicon():
    return FileResponse("./static/favicon.png", media_type="image/png")


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.post("/users/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_new_user(
    new_user: Annotated[UserCreate, Depends(UserCreate.as_form)],
):
    if new_user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    hashed_password = get_hashed_password(new_user.password)
    user_dict = {
        "username": new_user.username,
        "full_name": new_user.full_name,
        "email": new_user.email,
        "hashed_password": hashed_password,
        "disabled": False,
    }
    fake_users_db[new_user.username] = user_dict
    with open("fake_users_db.json", "w") as json_file:
        json.dump(fake_users_db, json_file, indent=2)
    return User(
        username=user_dict["username"],
        email=user_dict["email"],
        full_name=user_dict["full_name"],
        disabled=user_dict["disabled"],
    )


@app.get("/users/me/", response_model=User, status_code=status.HTTP_200_OK)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return current_user


@app.get("/users/me/items/")
async def read_own_item(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return [{"item_id": "Foo", "owner": current_user.username}]


# ==============================
# ML Model Endpoints
# ==============================
@app.post("/model/image_classification", status_code=status.HTTP_200_OK)
async def image_classification(
    uploaded_image: Annotated[UploadFile, File(...)],
    authenticated_user: Annotated[User, Depends(get_current_user)],
):
    if model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image model not loaded",
        )
    if not uploaded_image.content_type or not uploaded_image.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )
    try:
        raw_image_bytes = await uploaded_image.read()
        if len(raw_image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image too large. Max 10MB",
            )
        preprocessed = preprocess_image_for_tensorflow(
            raw_image_bytes, target_size=(80, 60)
        )
        probabilities = model_manager.predict(preprocessed)
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        predicted_category = (
            model_manager.class_names[predicted_idx]
            if model_manager.class_names
            else f"Class_{predicted_idx}"
        )

        # Top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3 = [
            {
                "category_name": model_manager.class_names[int(idx)]
                if model_manager.class_names
                else f"Class_{int(idx)}",
                "confidence_score": float(probabilities[idx]),
                "category_index": int(idx),
            }
            for idx in top3_indices
        ]

        response = {
            "classification_status": "success",
            "uploaded_filename": uploaded_image.filename,
            "file_size_bytes": len(raw_image_bytes),
            "file_type": uploaded_image.content_type,
            "processed_by_user": authenticated_user.username,
            "prediction_result": {
                "category": predicted_category,
                "confidence": confidence,
                "category_index": predicted_idx,
            },
            "alternative_predictions": top3,
            "all_possible_categories": model_manager.class_names,
            "total_classes": model_manager.num_classes,
        }
        return convert_numpy_types(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        await uploaded_image.close()


@app.post("/model/sentiment/form", status_code=status.HTTP_200_OK)
async def sentiment_analysis_form(
    review: Annotated[str, Form(...)],
    authenticated_user: Annotated[User, Depends(get_current_user)],
):
    if sentiment_model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment model not loaded",
        )
    results = sentiment_model_manager.predict(review)
    # Map numeric index to human‑readable label
    idx = results.get("predicted_index")
    if idx == 0:
        human_label = "negative"
    elif idx == 1:
        human_label = "neutral"
    elif idx == 2:
        human_label = "positive"
    else:
        human_label = results.get("predicted_label", "unknown")
    results["sentiment"] = human_label
    return {
        "status": "success",
        "predictions": results,
        "processed_by_user": authenticated_user.username,
    }


@app.get("/model/info")
async def model_info(authenticated_user: Annotated[User, Depends(get_current_user)]):
    # Image model info
    image_info = {
        "loaded": model_manager._model is not None,
        "num_classes": model_manager.num_classes,
        "class_names": model_manager.class_names,
        "input_shape": (80, 60, 3),
    }
    # Sentiment model info
    sentiment_info = {"loaded": sentiment_model_manager._model is not None}
    if sentiment_model_manager._model is not None:
        config = sentiment_model_manager._model.config
        sentiment_info.update(
            {
                "model_name": getattr(config, "_name_or_path", "unknown"),
                "num_labels": getattr(config, "num_labels", None),
                "id2label": getattr(config, "id2label", {}),
                "max_length": 512,
                "preprocessing": {
                    "lowercase": True,
                    "remove_punctuation": True,
                    "remove_stopwords": True,
                    "lemmatization": True,
                },
            }
        )
    return convert_numpy_types(
        {
            "image_model": image_info,
            "sentiment_model": sentiment_info,
        }
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "image_model": {"loaded": model_manager._model is not None},
        "sentiment_model": {"loaded": sentiment_model_manager._model is not None},
    }
