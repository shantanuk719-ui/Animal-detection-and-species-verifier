"""
Automated Animal Detection and Species Identification
Backend Server — Flask + TensorFlow
Author: shantanu kumar (25SCS1003000480)
Supervisor: Prof. Shantanu bhindewari | IILM University, Greater Noida
"""

import os
import json
import time
import threading
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
import io
import base64

# ─── Global State ────────────────────────────────────────────────────────────
model = None
training_status = {
    "state": "idle",          # idle | training | done | error
    "epoch": 0,
    "total_epochs": 10,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "test_accuracy": None,
    "message": "Model not trained yet. Click 'Train Model' to begin.",
    "log": []
}
training_lock = threading.Lock()

ANIMAL_CLASSES = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
ANIMAL_ICONS   = ['🐦', '🐱', '🦌', '🐶', '🐸', '🐴']

app = Flask(__name__)
CORS(app)

# ─── Helper: push a log line ──────────────────────────────────────────────────
def log(msg):
    with training_lock:
        training_status["log"].append(msg)
        if len(training_status["log"]) > 200:
            training_status["log"] = training_status["log"][-200:]

# ─── Training Thread ──────────────────────────────────────────────────────────
def train_model_thread():
    global model

    try:
        # ── imports (deferred so server starts fast) ──
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.datasets import cifar10
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

        with training_lock:
            training_status["state"]   = "training"
            training_status["message"] = "Loading CIFAR-10 dataset…"
            training_status["log"]     = []

        log("✅ TensorFlow " + tf.__version__ + " loaded")

        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                log("✅ GPU available and configured")
            except RuntimeError as e:
                log(f"GPU config error: {e}")
        else:
            log("⚠️  No GPU detected, using CPU (training may be slow)")

        log("📂 Downloading CIFAR-10 dataset (first run may take ~1 min)…")

        # ── Load & filter CIFAR-10 ──────────────────────────────────────────
        animal_indices = [2, 3, 4, 5, 6, 7]
        (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()

        train_mask = np.isin(y_train_full.flatten(), animal_indices)
        test_mask  = np.isin(y_test_full.flatten(),  animal_indices)

        X_train = X_train_full[train_mask]
        y_train = y_train_full[train_mask]
        X_test  = X_test_full[test_mask]
        y_test  = y_test_full[test_mask]

        label_mapping = {2:0, 3:1, 4:2, 5:3, 6:4, 7:5}
        y_train = np.array([label_mapping[l[0]] for l in y_train]).reshape(-1, 1)
        y_test  = np.array([label_mapping[l[0]] for l in y_test]).reshape(-1, 1)

        X_train = X_train.astype('float32') / 255.0
        X_test  = X_test.astype('float32')  / 255.0

        y_train_cat = to_categorical(y_train, 6)
        y_test_cat  = to_categorical(y_test,  6)

        log(f"✅ Dataset ready — {X_train.shape[0]} train / {X_test.shape[0]} test samples")
        log("🏗️  Building CNN model…")

        with training_lock:
            training_status["message"] = "Building CNN model…"

        # ── Build model ─────────────────────────────────────────────────────
        m = Sequential([
            Conv2D(32,  (5, 5), activation='relu', input_shape=(32, 32, 3), padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64,  (5, 5), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (5, 5), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(6, activation='softmax')
        ], name='AnimalDetectionCNN')

        m.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        log("✅ CNN model built  —  " + str(m.count_params()) + " parameters")

        # ── Data augmentation ────────────────────────────────────────────────
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        datagen.fit(X_train)

        # ── Live progress callback ───────────────────────────────────────────
        class LiveCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                ta = logs.get('accuracy',     0)
                va = logs.get('val_accuracy',  0)
                tl = logs.get('loss',          0)
                vl = logs.get('val_loss',      0)
                with training_lock:
                    training_status["epoch"]      = epoch + 1
                    training_status["train_acc"]  = round(ta * 100, 2)
                    training_status["val_acc"]    = round(va * 100, 2)
                    training_status["train_loss"] = round(tl,       4)
                    training_status["val_loss"]   = round(vl,       4)
                    training_status["message"]    = f"Epoch {epoch+1}/10 — val_acc: {va*100:.1f}%"
                log(f"  Epoch {epoch+1:02d}/10 | acc {ta*100:.1f}% | val_acc {va*100:.1f}% | loss {tl:.4f}")

        log("🚀 Training started (up to 10 epochs, early-stopping enabled)…")
        with training_lock:
            training_status["message"] = "Training in progress…"

        history = m.fit(
            datagen.flow(X_train, y_train_cat, batch_size=128),
            epochs=10,
            validation_data=(X_test, y_test_cat),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0),
                LiveCallback()
            ],
            verbose=0
        )

        # ── Evaluate ─────────────────────────────────────────────────────────
        _, test_acc = m.evaluate(X_test, y_test_cat, verbose=0)
        model = m

        log(f"✅ Training complete!  Test accuracy: {test_acc*100:.2f}%")
        with training_lock:
            training_status["state"]         = "done"
            training_status["test_accuracy"] = round(test_acc * 100, 2)
            training_status["message"]       = f"Model ready! Test accuracy: {test_acc*100:.2f}%"

        # Optionally save model
        try:
            m.save("animal_detection_model.h5")
            log("💾 Model saved to animal_detection_model.h5")
        except Exception:
            pass

    except Exception as e:
        with training_lock:
            training_status["state"]   = "error"
            training_status["message"] = f"Error: {str(e)}"
        log(f"❌ Training failed: {str(e)}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def start_training():
    with training_lock:
        if training_status["state"] == "training":
            return jsonify({"ok": False, "msg": "Training already in progress."})
        # reset
        training_status.update({
            "state": "idle", "epoch": 0, "total_epochs": 30,
            "train_acc": 0, "val_acc": 0, "train_loss": 0, "val_loss": 0,
            "test_accuracy": None, "log": [],
            "message": "Starting…"
        })

    t = threading.Thread(target=train_model_thread, daemon=True)
    t.start()
    return jsonify({"ok": True, "msg": "Training started."})


@app.route("/api/status")
def get_status():
    with training_lock:
        return jsonify(dict(training_status))


@app.route("/api/status/stream")
def status_stream():
    """Server-Sent Events stream so the browser updates in real-time."""
    def generate():
        last_epoch = -1
        last_state = ""
        while True:
            with training_lock:
                s = dict(training_status)
            # only push when something changes
            if s["epoch"] != last_epoch or s["state"] != last_state:
                last_epoch = s["epoch"]
                last_state = s["state"]
                yield f"data: {json.dumps(s)}\n\n"
            if s["state"] in ("done", "error", "idle"):
                # send one final update then close
                yield f"data: {json.dumps(s)}\n\n"
                break
            time.sleep(0.8)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        # Try to load saved model
        if os.path.exists("animal_detection_model.h5"):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model("animal_detection_model.h5")
            except Exception as e:
                return jsonify({"ok": False, "msg": f"Model not loaded: {str(e)}"}), 400
        else:
            return jsonify({"ok": False, "msg": "Model not trained yet. Please train the model first."}), 400

    # ── Read image ──────────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"ok": False, "msg": "No image file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "msg": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Build base64 thumbnail for display
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Preprocess
        img_resized = img.resize((32, 32), Image.LANCZOS)
        arr = np.array(img_resized, dtype='float32') / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        preds = model.predict(arr, verbose=0)[0]
        idx   = int(np.argmax(preds))
        conf  = float(preds[idx]) * 100
        is_animal = conf >= 40.0

        all_preds = [
            {"label": ANIMAL_CLASSES[i], "icon": ANIMAL_ICONS[i],
             "confidence": round(float(preds[i]) * 100, 2)}
            for i in range(6)
        ]
        all_preds.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "ok":             True,
            "predicted":      ANIMAL_CLASSES[idx],
            "icon":           ANIMAL_ICONS[idx],
            "confidence":     round(conf, 2),
            "is_animal":      is_animal,
            "all_predictions": all_preds,
            "thumbnail":      f"data:image/jpeg;base64,{b64}"
        })

    except Exception as e:
        return jsonify({"ok": False, "msg": f"Prediction error: {str(e)}"}), 500


@app.route("/api/logs")
def get_logs():
    with training_lock:
        return jsonify({"logs": training_status["log"]})


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  🐾 Animal Detection Web Hub")
    print("  Project by: shantanu kumar (25SCS1003000480)")
    print("  Supervisor: shantanu bhindewari | IILM University")
    print("=" * 65)
    print("  🌐 Open http://localhost:5000 in your browser")
    print("=" * 65)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
