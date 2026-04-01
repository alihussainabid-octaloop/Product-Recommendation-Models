import io
import json
import os
import pickle
import re
import string
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
import numpy as np
import tensorflow as tf
import torch
import nltk
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
from pwdlib import PasswordHash
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SECRET_KEY = "7c5902732404d7970efa708bafc0f8ce5d5b391c57e02c1000d9ca61bf561cdf"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


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


# Load users database
try:
    with open("fake_users_db.json", "r") as opened_file:
        fake_users_db = json.load(opened_file)
except FileNotFoundError:
    fake_users_db = {}

password_hash = PasswordHash.recommended()
DUMMY_HASH = password_hash.hash("dummypassword")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password: str, hashed_password: str):
    return password_hash.verify(plain_password, hashed_password)


def get_hashed_password(password: str):
    return password_hash.hash(password)


def get_user(db: dict, username):
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


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """Returns current authenticated active user."""
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


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(
        db=fake_users_db, username=form_data.username, password=form_data.password
    )
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
    return Token(access_token=access_token, token_type="Bearer")


@app.post("/users/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_new_user(
    new_user: Annotated[UserCreate, Depends(UserCreate.as_form)],
):
    # Check if username already exists
    if new_user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Hash the password
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


# ============== MODEL LOADING AND INFERENCE ==============


class ModelManager:
    """Singleton class to manage model loading and inference"""

    _instance = None
    _model = None
    _class_names = None
    _label_mapping = None
    _num_classes = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str):
        """Load the TensorFlow/Keras model from pickle file"""
        if self._model is not None:
            return  # Already loaded

        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            # Rebuild the model from config
            if "config" in model_data and "weights" in model_data:
                # Recreate the model architecture
                self._model = tf.keras.Sequential.from_config(model_data["config"])
                # Set the weights
                self._model.set_weights(model_data["weights"])

                # Convert class_names to list if it's numpy array
                class_names = model_data.get("class_names", [])
                if isinstance(class_names, np.ndarray):
                    class_names = class_names.tolist()
                self._class_names = class_names

                # Convert label_mapping to use Python ints
                label_mapping = model_data.get("label_mapping", {})
                if isinstance(label_mapping, dict):
                    self._label_mapping = {k: int(v) for k, v in label_mapping.items()}
                else:
                    self._label_mapping = {}

                self._num_classes = int(
                    model_data.get("num_classes", len(self._class_names))
                )

                # Compile the model (not necessary for inference, but good practice)
                self._model.compile(
                    optimizer="adam",
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"],
                )

                print(f"✅ Model loaded successfully. Classes: {self._num_classes}")
            else:
                raise ValueError("Invalid model format: missing config or weights")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise Exception(f"Failed to load model: {str(e)}")

    def predict(self, image_array: np.ndarray):
        """Run inference on preprocessed image array"""
        if self._model is None:
            raise Exception("Model not loaded")

        # Add batch dimension if needed
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

        # Run inference
        predictions = self._model.predict(image_array, verbose=0)

        # Apply softmax to convert logits to probabilities
        probabilities = tf.nn.softmax(predictions[0]).numpy()

        return probabilities

    @property
    def class_names(self):
        if self._class_names is None:
            return []
        if isinstance(self._class_names, np.ndarray):
            return self._class_names.tolist()
        return list(self._class_names)

    @property
    def label_mapping(self):
        return self._label_mapping or {}

    @property
    def num_classes(self):
        if self._num_classes is None:
            return None
        return int(self._num_classes)


class SentimentModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    _id2label = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_dir: str = "models/sentiment_model"):
        if self._model is not None:
            return

        # NLTK assets used for text preprocessing
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass

        if not os.path.exists(model_dir):
            # fallback to load from final_model_state.pkl for state_dict
            pkl_path = "models/final_model_state.pkl"
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f"No sentiment model directory at '{model_dir}' and no {pkl_path}"
                )

            checkpoint = "google-bert/bert-base-uncased"
            self._model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, num_labels=3
            )
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            if "model_state_dict" in data:
                state_dict = data["model_state_dict"]
                self._model.load_state_dict(state_dict)

            self._id2label = {0: "negative", 1: "neutral", 2: "positive"}
            return

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self._id2label = {
            int(k): v
            for k, v in getattr(self._model.config, "id2label", {}).items()
        }
        if not self._id2label:
            self._id2label = {0: "negative", 1: "neutral", 2: "positive"}

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
        id2label: dict[int, str] = (
            self._id2label if isinstance(self._id2label, dict) else {}
        )
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        return {
            "review": raw_text,
            "processed_review": clean_text,
            "predicted_label": predicted_label,
            "predicted_index": predicted_idx,
            "confidence": float(probs[predicted_idx]),
            "probabilities": probs.tolist(),
            "id2label": self._id2label,
        }


model_manager = ModelManager()
sentiment_model_manager = SentimentModelManager()


def preprocess_image_for_tensorflow(image_bytes: bytes, target_size=(80, 60)):
    """Preprocess image for TensorFlow model"""
    pil_image = Image.open(io.BytesIO(image_bytes))

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)

    image_array = np.array(pil_image, dtype=np.float32) / 255.0

    return image_array


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        model_manager.load_model("./models/image_classifier_ecommerce.pkl")
        print("✅ Image model loaded successfully on startup")
    except Exception as e:
        print(f"❌ Failed to load image model on startup: {e}")

    try:
        sentiment_model_manager.load_model("./models/sentiment_model")
        print("✅ Sentiment model loaded successfully on startup")
    except Exception as e:
        print(f"❌ Failed to load sentiment model on startup: {e}")


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            normalized_key = convert_numpy_types(key)
            if not isinstance(normalized_key, str):
                normalized_key = str(normalized_key)
            normalized[normalized_key] = convert_numpy_types(value)
        return normalized
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


@app.post("/model/image_classification", status_code=status.HTTP_200_OK)
async def image_classification(
    uploaded_image: Annotated[UploadFile, File(...)],
    authenticated_user: Annotated[User, Depends(get_current_user)],
):
    if model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    if (
        uploaded_image.content_type is None
        or not uploaded_image.content_type.startswith("image/")
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
                detail="Image too large. Maximum size is 10MB",
            )

        preprocessed_image = preprocess_image_for_tensorflow(
            raw_image_bytes, target_size=(80, 60)
        )

        probabilities = model_manager.predict(preprocessed_image)

        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        reverse_mapping = {int(v): k for k, v in (model_manager.label_mapping or {}).items()}
        predicted_category = reverse_mapping.get(predicted_idx, f"Class_{predicted_idx}")

        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                "category_name": reverse_mapping.get(int(idx), f"Class_{int(idx)}"),
                "confidence_score": float(probabilities[idx]),
                "category_index": int(idx),
            }
            for idx in top_3_indices
        ]

        response_data = {
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
            "alternative_predictions": top_3_predictions,
            "all_possible_categories": model_manager.class_names,
            "total_classes": model_manager.num_classes,
        }

        response_data = convert_numpy_types(response_data)

        return response_data

    except HTTPException:
        raise
    except Exception as processing_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(processing_error)}",
        )
    finally:
        await uploaded_image.close()


class SentimentRequest(BaseModel):
    review: str


@app.post("/model/sentiment", status_code=status.HTTP_200_OK)
async def sentiment_analysis(
    payload: SentimentRequest,
    authenticated_user: Annotated[User, Depends(get_current_user)],
):
    if sentiment_model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment model not loaded. Please check server logs.",
        )

    results = sentiment_model_manager.predict(payload.review)

    return {
        "status": "success",
        "predictions": results,
        "processed_by_user": authenticated_user.username,
    }


@app.post("/model/sentiment/form", status_code=status.HTTP_200_OK)
async def sentiment_analysis_form(
    review: Annotated[str, Form(...)],
    authenticated_user: Annotated[User, Depends(get_current_user)],
):
    if sentiment_model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment model not loaded. Please check server logs.",
        )

    results = sentiment_model_manager.predict(review)

    return {
        "status": "success",
        "predictions": results,
        "processed_by_user": authenticated_user.username,
    }


@app.post("/model/recommend", status_code=status.HTTP_200_OK)
async def recommend_product(
    review: Annotated[str, Form(...)],
    uploaded_image: Annotated[UploadFile, File(...)],
    product_id: Annotated[str | None, Form(None)],
    category: Annotated[str | None, Form(None)],
    price: Annotated[float | None, Form(None)],
    authenticated_user: Annotated[User, Depends(get_current_user)],
):
    if model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image model not loaded. Please check server logs.",
        )
    if sentiment_model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment model not loaded. Please check server logs.",
        )

    if (
        uploaded_image.content_type is None
        or not uploaded_image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    raw_image_bytes = await uploaded_image.read()
    if len(raw_image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image too large. Maximum size is 10MB",
        )

    image_array = preprocess_image_for_tensorflow(raw_image_bytes, target_size=(80, 60))
    image_probas = model_manager.predict(image_array)

    image_pred_idx = int(np.argmax(image_probas))
    image_pred_confidence = float(image_probas[image_pred_idx])
    reverse_mapping = {int(v): k for k, v in (model_manager.label_mapping or {}).items()}
    image_pred_category = reverse_mapping.get(image_pred_idx, f"Class_{image_pred_idx}")

    sentiment_results = sentiment_model_manager.predict(review)
    sentiment_label = str(sentiment_results.get("predicted_label", ""))
    sentiment_confidence = float(sentiment_results.get("confidence", 0.0))

    sentiment_score_map = {"negative": 0.0, "neutral": 0.5, "positive": 1.0}
    sentiment_score = sentiment_score_map[sentiment_label] if sentiment_label in sentiment_score_map else 0.5

    combined_score = (sentiment_score * 0.6) + (image_pred_confidence * 0.4)

    if sentiment_label == "positive" and image_pred_confidence >= 0.7:
        decision = "highly_recommended"
        reason = "Positive review sentiment and strong image category match."
    elif sentiment_label in ["positive", "neutral"] and image_pred_confidence >= 0.5:
        decision = "recommended"
        reason = "Overall sentiment/image confidence supports recommendation."
    elif sentiment_label == "negative":
        decision = "not_recommended"
        reason = "Negative sentiment dominates recommendation."
    else:
        decision = "consider"
        reason = "Moderate confidence. User review may require more context."

    top_3_image_indices = np.argsort(image_probas)[-3:][::-1]
    top_3_image_predictions = [
        {
            "category_name": reverse_mapping.get(int(idx), f"Class_{int(idx)}"),
            "confidence_score": float(image_probas[idx]),
            "category_index": int(idx),
        }
        for idx in top_3_image_indices
    ]

    response = {
        "status": "success",
        "processed_by_user": authenticated_user.username,
        "product_id": product_id,
        "product_category": category,
        "product_price": price,
        "sentiment": sentiment_results,
        "image_classification": {
            "category": image_pred_category,
            "confidence": image_pred_confidence,
            "category_index": image_pred_idx,
            "top_3": top_3_image_predictions,
            "all_classes": model_manager.class_names,
        },
        "recommendation": {
            "decision": decision,
            "reason": reason,
            "combined_score": float(combined_score),
            "sentiment_score": float(sentiment_score),
            "sentiment_confidence": float(sentiment_confidence),
            "image_score": float(image_pred_confidence),
        },
    }

    return convert_numpy_types(response)


@app.get("/model/info")
async def model_info(authenticated_user: Annotated[User, Depends(get_current_user)]):
    if model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    label_mapping_converted = {
        k: int(v) for k, v in (model_manager.label_mapping or {}).items()
    }

    response_data = {
        "num_classes": int(model_manager.num_classes)
        if model_manager.num_classes is not None
        else None,
        "class_names": model_manager.class_names,
        "input_shape": (80, 60, 3),
        "label_mapping": label_mapping_converted,
    }

    return convert_numpy_types(response_data)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager._model is not None,
        "model_classes": int(model_manager.num_classes)
        if model_manager._model and model_manager.num_classes is not None
        else None,
    }
