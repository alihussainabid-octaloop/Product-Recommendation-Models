import io
import pickle
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image


class ModelManager:
    """Singleton class to manage image classification model."""

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
        if self._model is not None:
            return

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        if not isinstance(model_data, dict) or "config" not in model_data or "weights" not in model_data:
            raise ValueError("Invalid image model format: missing config or weights")

        self._model = tf.keras.Sequential.from_config(model_data["config"])
        self._model.set_weights(model_data["weights"])

        class_names = model_data.get("class_names", [])
        if isinstance(class_names, np.ndarray):
            class_names = class_names.tolist()
        self._class_names = class_names

        label_mapping = model_data.get("label_mapping", {})
        if isinstance(label_mapping, dict):
            self._label_mapping = {k: int(v) for k, v in label_mapping.items()}
        else:
            self._label_mapping = {}

        self._num_classes = int(model_data.get("num_classes", len(self._class_names)))

        self._model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def predict(self, image_array: np.ndarray):
        if self._model is None:
            raise Exception("Image model not loaded")

        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

        predictions = self._model.predict(image_array, verbose=0)
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
    """Singleton manager for sentiment model."""

    _instance = None
    _model: Any = None
    _vectorizer: Any = None
    _tokenizer: Any = None
    _model_type: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        if hasattr(data, "predict"):
            self._model = data
            self._model_type = "sklearn"
            return

        if isinstance(data, dict):
            if "pipeline" in data and hasattr(data["pipeline"], "predict"):
                self._model = data["pipeline"]
                self._model_type = "sklearn"
                return

            if "model" in data and hasattr(data["model"], "predict"):
                self._model = data["model"]
                self._model_type = "sklearn"
                self._vectorizer = data.get("vectorizer")
                self._tokenizer = data.get("tokenizer")
                return

            if "state_dict" in data:
                try:
                    import importlib
                    torch = importlib.import_module("torch")
                    if "model_class" in data:
                        model_cls = data["model_class"]
                        self._model = model_cls()
                        self._model.load_state_dict(data["state_dict"])
                        if hasattr(self._model, "to"):
                            self._model = self._model.to(torch.device("cpu"))
                        self._model.eval()
                        self._model_type = "torch"
                        self._tokenizer = data.get("tokenizer")
                        return
                except Exception:
                    pass

            if "vectorizer" in data and "estimator" in data:
                self._vectorizer = data["vectorizer"]
                self._model = data["estimator"]
                self._model_type = "sklearn"
                return

            if "weights" in data and "bias" in data and "vectorizer" in data:
                self._vectorizer = data["vectorizer"]
                self._model = data
                self._model_type = "manual"
                return

        raise ValueError("Unsupported sentiment model format")

    def predict(self, text: str):
        if self._model is None:
            raise Exception("Sentiment model is not loaded")

        if self._model_type == "sklearn":
            if not hasattr(self._model, "predict"):
                raise Exception("Sentiment model object is not a predictor")
            model_obj = self._model
            if self._vectorizer is not None:
                vectorizer_obj = self._vectorizer
                x = vectorizer_obj.transform([text])
                pred = model_obj.predict(x)
            else:
                pred = model_obj.predict([text])
            return pred[0]

        if self._model_type == "torch":
            import importlib
            torch = importlib.import_module("torch")
            if self._tokenizer is None:
                raise Exception("Torch sentiment model requires tokenizer")

            tokens = self._tokenizer(text)
            if isinstance(tokens, dict):
                inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in tokens.items()}
            else:
                inputs = torch.tensor(tokens).unsqueeze(0)

            with torch.no_grad():
                outputs = self._model(**inputs) if isinstance(inputs, dict) else self._model(inputs)

            if hasattr(outputs, "logits"):
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                conf, idx = torch.max(probs, dim=-1)
                return {"label": int(idx.item()), "confidence": float(conf.item())}

            return outputs

        if self._model_type == "manual":
            vector = self._vectorizer.transform([text])
            weights = np.array(self._model["weights"])
            bias = float(self._model["bias"])
            score = vector.dot(weights) + bias
            prob = 1 / (1 + np.exp(-score))
            label = "positive" if prob >= 0.5 else "negative"
            return {"label": label, "score": float(prob)}

        raise Exception("Unknown sentiment model type")


def preprocess_image_for_tensorflow(image_bytes: bytes, target_size=(80, 60)):
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(pil_image, dtype=np.float32) / 255.0
    return image_array


def convert_numpy_types(obj: Any):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(x) for x in obj)
    else:
        return obj


image_model_manager = ModelManager()
sentiment_model_manager = SentimentModelManager()
