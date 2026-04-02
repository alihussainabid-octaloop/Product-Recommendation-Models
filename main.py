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
import tensorflow as tf
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


@app.get("/favicon", include_in_schema=True)
async def favicon():
    return FileResponse("./static/favicon.png", media_type="image/png")


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
        print(f"Attempting to load sentiment model from {model_path}")
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
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
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
        # Ensure it is serializable list and not a generator-like object
        return list(self._class_names)

    @property
    def label_mapping(self):
        return self._label_mapping or {}

    @property
    def num_classes(self):
        if self._num_classes is None:
            return None
        return int(self._num_classes)


# -----------------------------
# Sentiment transformer model manager
# -----------------------------
class SentimentModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    _id2label = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str = "models/final_model_state.pkl"):
        """Load sentiment model from a Transformers directory or legacy pickle file."""
        if self._model is not None:
            return

        # Download NLTK data (used for preprocessing)
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # ----- 1. Load from a full Transformers directory (new format) -----
        if os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        ):
            print(f"Loading model from directory: {model_path}")
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

        # ----- 2. Legacy: single pickle file (keep for backward compatibility) -----
        pkl_path = model_path
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model file not found: {pkl_path}")

        # Try full model load (torch.save of entire model)
        try:
            self._model = torch.load(pkl_path, map_location="cpu", weights_only=False)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "google-bert/bert-base-uncased"
                )
            except Exception:
                self._tokenizer = None
            if hasattr(self._model, "config") and hasattr(
                self._model.config, "id2label"
            ):
                self._id2label = self._model.config.id2label
            else:
                self._id2label = {0: "negative", 1: "neutral", 2: "positive"}
            print("✅ Sentiment model loaded from legacy pickle (full model)")
            return
        except Exception:
            pass

        # Fallback: load base model + state dict
        checkpoint = "google-bert/bert-base-uncased"
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, num_labels=3, local_files_only=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, local_files_only=True
            )
        except Exception:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, num_labels=3
            )
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        data = torch.load(pkl_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in data:
            self._model.load_state_dict(data["model_state_dict"])
        else:
            self._model.load_state_dict(data)

        self._id2label = {
            int(k): v for k, v in getattr(self._model.config, "id2label", {}).items()
        }
        if not self._id2label:
            self._id2label = {0: "negative", 1: "neutral", 2: "positive"}
        print("✅ Sentiment model loaded from legacy pickle (state dict)")

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
        if self._id2label is not None and isinstance(self._id2label, dict):
            predicted_label = self._id2label.get(predicted_idx, str(predicted_idx))
        else:
            predicted_label = str(predicted_idx)

        return {
            "review": raw_text,
            "processed_review": clean_text,
            "predicted_label": predicted_label,
            "predicted_index": predicted_idx,
            "confidence": float(probs[predicted_idx]),
            "probabilities": probs.tolist(),
            "id2label": self._id2label,
        }


# Initialize model managers
model_manager = ModelManager()
sentiment_model_manager = SentimentModelManager()


def preprocess_image_for_tensorflow(image_bytes: bytes, target_size=(80, 60)):
    """Preprocess image for TensorFlow model"""
    # Open image
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize
    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(pil_image, dtype=np.float32) / 255.0

    return image_array


@app.on_event("startup")
async def startup_event():
    # ... image model loading ...
    try:
        sentiment_model_manager.load_model(
            "./models/sentiment_model"
        )  # directory with config.json
        print("✅ Sentiment model loaded successfully on startup")
    except Exception as e:
        print(f"❌ Failed to load sentiment model on startup: {e}")


# Add this helper function to convert numpy types to Python native types
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.generic):
        # Covers np.int64, np.float32, np.bool_, np.str_, etc.
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
    """Classify uploaded image using the trained model"""

    # Check if model is loaded
    if model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    # Validate file type
    if (
        uploaded_image.content_type is None
        or not uploaded_image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    # Process image and run inference
    try:
        # Read image bytes
        raw_image_bytes = await uploaded_image.read()

        # Validate file size (limit to 10MB)
        if len(raw_image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image too large. Maximum size is 10MB",
            )

        # Preprocess image for TensorFlow
        preprocessed_image = preprocess_image_for_tensorflow(
            raw_image_bytes, target_size=(80, 60)
        )

        # Get predictions
        probabilities = model_manager.predict(preprocessed_image)

        # Get predicted class (convert to Python int)
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])

        # Map to category name
        reverse_mapping = {
            int(v): k for k, v in (model_manager.label_mapping or {}).items()
        }
        predicted_category = reverse_mapping.get(
            predicted_idx, f"Class_{predicted_idx}"
        )

        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                "category_name": reverse_mapping.get(int(idx), f"Class_{int(idx)}"),
                "confidence_score": float(probabilities[idx]),
                "category_index": int(idx),
            }
            for idx in top_3_indices
        ]

        # Prepare response with all numpy types converted
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

        # Convert any remaining numpy types to Python native types
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


@app.post("/model/sentiment/form", status_code=status.HTTP_200_OK)
async def sentiment_analysis_form(
    review: Annotated[str, Form(...)],
    authenticated_user: Annotated[User, Depends(get_current_user)],
):
    """Form-based sentiment review endpoint returning human-readable labels."""
    print(f"Model is loaded: {sentiment_model_manager._model is not None}")
    if sentiment_model_manager._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment model not loaded. Please check server logs.",
        )

    results = sentiment_model_manager.predict(review)

    # --- BEGIN: Ensure human-readable sentiment label ---
    predicted_idx = results.get("predicted_index")
    if predicted_idx is not None:
        # Map numeric index to standard human-readable labels
        if predicted_idx == 0:
            human_label = "negative"
        elif predicted_idx == 1:
            human_label = "neutral"
        elif predicted_idx == 2:
            human_label = "positive"
        else:
            # Fallback: use whatever label the model provided (if any)
            human_label = results.get("predicted_label", "unknown")
        results["sentiment"] = human_label
    else:
        # If no index is available, try to use the existing predicted_label
        results["sentiment"] = results.get("predicted_label", "unknown")
    # --- END ---

    return {
        "status": "success",
        "predictions": results,
        "processed_by_user": authenticated_user.username,
    }


# Also update the model info endpoint to handle numpy types
@app.get("/model/info")
async def model_info(authenticated_user: Annotated[User, Depends(get_current_user)]):
    """Get information about both the image classification and sentiment analysis models"""

    # Image model info
    image_info = {
        "loaded": model_manager._model is not None,
        "num_classes": int(model_manager.num_classes)
        if model_manager.num_classes is not None
        else None,
        "class_names": model_manager.class_names,
        "input_shape": (80, 60, 3),
        "label_mapping": {
            k: int(v) for k, v in (model_manager.label_mapping or {}).items()
        },
    }

    # Sentiment model info
    sentiment_info = {
        "loaded": sentiment_model_manager._model is not None,
    }
    if sentiment_model_manager._model is not None:
        # Extract information from the loaded model
        config = sentiment_model_manager._model.config
        sentiment_info["model_name"] = getattr(config, "_name_or_path", "unknown")
        sentiment_info["num_labels"] = getattr(config, "num_labels", None)
        sentiment_info["id2label"] = getattr(config, "id2label", {})
        sentiment_info["label2id"] = getattr(config, "label2id", {})
        sentiment_info["max_length"] = 512  # Default used in tokenizer
        # Add tokenizer info if available
        if sentiment_model_manager._tokenizer:
            tokenizer_name = getattr(
                sentiment_model_manager._tokenizer, "name_or_path", "unknown"
            )
            sentiment_info["tokenizer"] = tokenizer_name
        # Add preprocess steps description
        sentiment_info["preprocessing"] = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True,
            "lemmatization": True,
        }
    else:
        sentiment_info["error"] = "Model not loaded"

    # Convert any numpy types to Python natives
    response_data = {
        "image_model": convert_numpy_types(image_info),
        "sentiment_model": convert_numpy_types(sentiment_info),
    }

    return response_data


# Update health check endpoint
@app.get("/health")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "image_model": {
            "loaded": model_manager._model is not None,
            "num_classes": int(model_manager.num_classes)
            if model_manager._model and model_manager.num_classes is not None
            else None,
        },
        "sentiment_model": {
            "loaded": sentiment_model_manager._model is not None,
        },
    }
