# 🔥 AI Wildfire Detection from Drone Feed

A complete computer-vision pipeline: **CNN** (fire/no-fire classifier) +
**YOLOv8** (real-time object detection) + **Grad-CAM** (explainable AI heatmaps)
+ **FastAPI** backend + **React** frontend.

---

## Project Structure

```
wildfire-cv/
├── requirements.txt
├── fire_dataset.yaml          ← YOLO dataset config
├── 1_data/
│   ├── raw/
│   │   ├── fire/              ← Put fire images here
│   │   └── no_fire/           ← Put no-fire images here
│   ├── processed/             ← Auto-created by prepare_data.py
│   └── prepare_data.py
├── 2_model/
│   ├── train_cnn.py
│   ├── train_yolo.py
│   ├── gradcam.py
│   └── saved/                 ← Saved model weights go here
├── 3_backend/
│   ├── main.py                ← FastAPI server
│   └── detect.py              ← YOLO + CNN inference
└── 4_frontend/
    └── src/
        └── App.jsx            ← React dashboard
```

---

## Step 0 — Install Python Dependencies

Open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Get the Dataset

### Option A — Kaggle Fire Dataset (quickest)
1. Visit https://www.kaggle.com/datasets/phylake1337/fire-dataset
2. Download and unzip
3. Copy images into:
   - `1_data/raw/fire/`
   - `1_data/raw/no_fire/`

### Option B — FLAME Drone Dataset (drone-specific)
- https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs

---

## Step 2 — Preprocess Data

```bash
python 1_data/prepare_data.py
```

Creates `1_data/processed/train/` and `1_data/processed/val/`.

---

## Step 3 — Train the CNN

```bash
python 2_model/train_cnn.py
```

Trains ResNet-18 for 15 epochs. Best model saved to `2_model/saved/cnn_wildfire.pth`.

---

## Step 4 — Test Grad-CAM

```bash
python 2_model/gradcam.py --image 1_data/raw/fire/your_image.jpg
```

Saves a `*_gradcam.jpg` heatmap overlay next to the input image.

---

## Step 5 — Train YOLO (optional but recommended for real-time)

1. Download YOLO-format fire/smoke labels from Roboflow:
   https://universe.roboflow.com/school-tvtyg/fire-and-smoke-detection
2. Extract dataset to `1_data/yolo_dataset/`
3. Run:

```bash
python 2_model/train_yolo.py
```

Best weights saved to `2_model/saved/wildfire_yolo/weights/best.pt`.

---

## Step 6 — Start the FastAPI Backend

```bash
cd 3_backend
uvicorn main:app --reload --port 8000
```

API will be available at http://localhost:8000
Interactive docs at http://localhost:8000/docs

---

## Step 7 — Set Up and Start the React Frontend

If you haven't created the React app yet:

```bash
cd 4_frontend
npx create-react-app wildfire-ui
cd wildfire-ui
npm install axios
```

Copy `4_frontend/src/App.jsx` into `4_frontend/wildfire-ui/src/App.jsx`, then:

```bash
npm start
```

Frontend runs at http://localhost:3000

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict/image` | Upload image → get prediction |
| GET | `/heatmap/{filename}` | Retrieve Grad-CAM overlay |
| WS | `/ws/stream` | Live drone WebSocket stream |

---

## VS Code Tips

- Install the **Python** extension (Microsoft)
- Install the **ES7+ React** extension for JSX support
- Open integrated terminal: `` Ctrl+` ``
- Run each step in a separate terminal tab

---

## Phase 7 — Generative AI Layer (NEW)

Three Claude-powered features added on top of the existing CV pipeline:

| Feature | Endpoint | Description |
|---|---|---|
| Incident Report | `POST /genai/report` | Structured JSON report after detection |
| Ranger SMS Alert | `POST /genai/alert` | AI-written SMS replacing Twilio template |
| NL Dashboard Chat | `POST /genai/chat` | Tool-use chat for rangers to query the system |

### Setup

1. Get an Anthropic API key from https://console.anthropic.com
2. Set it before starting the backend:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

3. Install the new dependency:
```bash
pip install anthropic
```

### GenAI Data Flow

```
Detection event → POST /genai/report → Claude API → Structured JSON report
                → POST /genai/alert  → Claude API → SMS text via Twilio
Ranger question → POST /genai/chat  → Claude tool-use → FastAPI fetch → Plain-English reply
```

### Frontend

After uploading and analysing a drone image, a **Phase 7 — Generative AI Layer** panel
appears below the detection result with three interactive sections:
- Generate incident report (shows formatted JSON)
- Generate ranger SMS alert (with editable location field)
- Natural language chat box (ask anything about the dashboard)
