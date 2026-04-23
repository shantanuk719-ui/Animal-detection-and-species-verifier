# 🐾 Automated Animal Detection & Species Identification
### Web Hub — Kushagra Agrawal (25SCS1003000659)
**Supervisor:** Prof. Shobhit Agrawal | IILM University, Greater Noida | CSE-AIML

---

## What This Does
A full-stack web application that:
- Trains a CNN on CIFAR-10 animal images (Bird, Cat, Deer, Dog, Frog, Horse)
- Shows live training progress (epoch, accuracy, loss) in the browser
- Accepts any image upload and classifies the animal species
- Warns if the uploaded image doesn't appear to be an animal

---

## Quick Start (Run Locally)

### 1. Make sure Python 3.9+ is installed
```bash
python --version
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> ⏱ TensorFlow installation takes 2–5 minutes on first run.

### 3. Start the server
```bash
python app.py
```

### 4. Open your browser
```
http://localhost:5000
```

---

## How to Use the Web Hub

1. **Train the Model** — Click the green "Train Model" button on the left panel.
   - The CIFAR-10 dataset will download automatically (~170 MB, first time only).
   - Training runs up to 30 epochs with early-stopping.
   - Live progress (epoch, accuracy, loss) streams to the browser in real time.
   - Estimated time: ~5–15 minutes depending on CPU/GPU.

2. **Identify an Animal** — Once training is done, use the right panel:
   - Click the upload zone or drag & drop any image.
   - Click "Identify Animal".
   - View the predicted species, confidence score, and probability bars for all 6 classes.

---

## Deployment (Show Online)

### Option A — Railway (free, easiest)
1. Push this folder to a GitHub repo
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Done! Railway auto-detects Flask and deploys.

### Option B — Render
1. Push to GitHub
2. Go to https://render.com → New Web Service
3. Set start command: `python app.py`

### Option C — Heroku
Add a `Procfile` with:
```
web: python app.py
```

---

## Project Structure
```
animal_detection/
├── app.py                  ← Flask backend + all API routes
├── templates/
│   └── index.html          ← Frontend (single-file, no framework)
├── requirements.txt
└── README.md
```

---

## Architecture
- **Frontend:** Vanilla HTML/CSS/JS (no framework, works everywhere)
- **Backend:** Python Flask with Server-Sent Events for live training updates
- **Model:** 3-layer CNN (32→64→128 filters) + Dense(128) + Softmax(6)
- **Dataset:** CIFAR-10 filtered to 6 animal classes (~30,000 images)
- **Training:** Adam optimizer, categorical cross-entropy, early stopping

---

*Made for academic demonstration — IILM University CSE-AIML Department*
