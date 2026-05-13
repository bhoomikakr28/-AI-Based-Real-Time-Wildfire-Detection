from dotenv import load_dotenv
load_dotenv()

import os, shutil, uuid, requests, cv2, base64, threading, math, random
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from detect import predict_image
from groq import Groq
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("=" * 50)
print("GROQ KEY:", "✅ Loaded" if os.environ.get("GROQ_API_KEY") else "❌ NOT FOUND — GenAI will fail!")
print("=" * 50)

FOREST_ZONES = [
    {"name": "Bandipur National Park",     "lat": 11.6750, "lon": 76.6347, "state": "Karnataka"},
    {"name": "Nagarhole National Park",    "lat": 12.0500, "lon": 76.1333, "state": "Karnataka"},
    {"name": "Wayanad Wildlife Sanctuary", "lat": 11.6854, "lon": 76.1320, "state": "Kerala"},
    {"name": "Mudumalai Tiger Reserve",    "lat": 11.5833, "lon": 76.6333, "state": "Tamil Nadu"},
    {"name": "Sathyamangalam Forest",      "lat": 11.5038, "lon": 77.2384, "state": "Tamil Nadu"},
    {"name": "BRT Wildlife Sanctuary",     "lat": 11.9667, "lon": 77.0500, "state": "Karnataka"},
    {"name": "Annamalai Tiger Reserve",    "lat": 10.3167, "lon": 77.0333, "state": "Tamil Nadu"},
    {"name": "Periyar Tiger Reserve",      "lat":  9.4581, "lon": 77.1676, "state": "Kerala"},
    {"name": "Dandeli Wildlife Sanctuary", "lat": 15.2500, "lon": 74.6167, "state": "Karnataka"},
    {"name": "Kanha Tiger Reserve",        "lat": 22.3333, "lon": 80.6167, "state": "Madhya Pradesh"},
    {"name": "Jim Corbett National Park",  "lat": 29.5300, "lon": 78.7747, "state": "Uttarakhand"},
    {"name": "Sundarbans National Park",   "lat": 21.9497, "lon": 88.9420, "state": "West Bengal"},
    {"name": "Kaziranga National Park",    "lat": 26.5775, "lon": 93.1711, "state": "Assam"},
    {"name": "Gir Forest National Park",   "lat": 21.1239, "lon": 70.8242, "state": "Gujarat"},
    {"name": "Ranthambore National Park",  "lat": 26.0173, "lon": 76.5026, "state": "Rajasthan"},
]

rtsp_state = {
    "running": False,
    "url": "",
    "latest_result": None,
    "latest_frame": None,
    "thread": None
}

# ── HELPER FUNCTIONS ──────────────────────────────────────────

def ask(prompt, max_tokens=1000):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        return f"Error generating response: {str(e)}"

def get_weather(city="Bengaluru"):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        r = requests.get(url, timeout=5)
        d = r.json()
        current = d["current_condition"][0]
        return {
            "temperature": int(current["temp_C"]),
            "feels_like":  int(current["FeelsLikeC"]),
            "humidity":    int(current["humidity"]),
            "wind_speed":  int(current["windspeedKmph"]),
            "visibility":  int(current["visibility"]),
            "description": current["weatherDesc"][0]["value"],
            "uv_index":    int(current["uvIndex"]),
            "source":      "live"
        }
    except:
        return {
            "temperature": 32, "feels_like": 36,
            "humidity": 45,    "wind_speed": 18,
            "visibility": 10,  "description": "Partly cloudy",
            "uv_index": 6,     "source": "fallback"
        }

def get_fire_weather(detection_result: dict):
    base = get_weather()
    if detection_result.get("label") != "fire":
        return base
    conf          = detection_result.get("confidence", 0.5)
    fire_temp     = max(base["temperature"], int(35 + conf * 20))
    fire_humidity = min(base["humidity"],    int(40 - conf * 30))
    fire_wind     = max(base["wind_speed"],  int(20 + conf * 30))
    fire_uv       = max(base["uv_index"],    8)
    fire_humidity = max(fire_humidity, 5)
    fire_wind     = min(fire_wind, 80)
    return {
        "temperature": fire_temp,
        "feels_like":  fire_temp + 4,
        "humidity":    fire_humidity,
        "wind_speed":  fire_wind,
        "visibility":  max(base["visibility"] - 3, 1),
        "description": "Dry & Hot — High Fire Risk 🔥",
        "uv_index":    fire_uv,
        "source":      "fire-adjusted"
    }

def get_fire_location(confidence: float):
    if confidence > 0.8:
        high_risk = FOREST_ZONES[:6]
    elif confidence > 0.5:
        high_risk = FOREST_ZONES[:10]
    else:
        high_risk = FOREST_ZONES
    zone = random.choice(high_risk)
    return {
        "forest_name": zone["name"],
        "state":       zone["state"],
        "lat":         zone["lat"],
        "lon":         zone["lon"],
        "alert_level": "CRITICAL" if confidence > 0.8 else "HIGH" if confidence > 0.5 else "MODERATE"
    }

def rtsp_worker(url: str):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        rtsp_state["running"] = False
        return
    frame_count = 0
    while rtsp_state["running"]:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result  = predict_image(pil_img)
            rtsp_state["latest_result"] = result
            if result.get("boxes"):
                for box in result["boxes"]:
                    cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{box['label']} {box['confidence']:.0%}",
                        (box["x1"], box["y1"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            label = result["label"]
            color = (0, 0, 255) if label == "fire" else (0, 255, 0)
            cv2.putText(frame, f"{label.upper()} {result['confidence']:.0%}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            _, buffer = cv2.imencode(".jpg", frame)
            rtsp_state["latest_frame"] = buffer.tobytes()
    cap.release()
    rtsp_state["running"] = False

def generate_frames():
    while rtsp_state["running"]:
        if rtsp_state["latest_frame"] is not None:
            frame = rtsp_state["latest_frame"]
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ── ROUTES ────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/weather")
def weather(city: str = "Bengaluru"):
    return get_weather(city)

@app.get("/forests")
def get_forests():
    return FOREST_ZONES

@app.post("/predict/image")
async def predict(file: UploadFile = File(...)):
    fname = f"{uuid.uuid4()}_{file.filename}"
    fpath = UPLOAD_DIR / fname
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result   = predict_image(Image.open(fpath))
    w        = get_fire_weather(result)
    location = get_fire_location(result["confidence"]) if result["label"] == "fire" else None
    return {**result, "filename": fname, "weather": w, "location": location}

@app.post("/predict/frame")
async def predict_frame(data: dict):
    try:
        img_data  = base64.b64decode(data["frame"].split(",")[1])
        img_array = np.frombuffer(img_data, np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(rgb)
        result    = predict_image(pil_img)
        location  = get_fire_location(result["confidence"]) if result["label"] == "fire" else None
        return {**result, "weather": get_fire_weather(result), "location": location}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    fname    = f"{uuid.uuid4()}_{file.filename}"
    fpath    = UPLOAD_DIR / fname
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    cap = cv2.VideoCapture(str(fpath))
    if not cap.isOpened():
        return {"error": "Could not open video"}
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_name = f"out_{fname}.mp4"
    out_path = UPLOAD_DIR / out_name
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    out      = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    fire_frames  = 0
    total_frames = 0
    max_conf     = 0.0
    all_boxes    = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        if total_frames % 5 == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result  = predict_image(pil_img)
            if result["label"] == "fire":
                fire_frames += 1
                max_conf = max(max_conf, result["confidence"])
            if result.get("boxes"):
                for box in result["boxes"]:
                    all_boxes.append(box)
                    cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 3)
                    cv2.rectangle(frame, (box["x1"], box["y1"] - 25), (box["x1"] + 120, box["y1"]), (0, 255, 0), -1)
                    cv2.putText(frame, f"{box['label']} {box['confidence']:.0%}",
                        (box["x1"] + 4, box["y1"] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            label = result["label"]
            conf  = result["confidence"]
            color = (0, 0, 255) if label == "fire" else (0, 200, 0)
            cv2.rectangle(frame, (5, 5), (320, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"{label.upper()}  {conf:.0%}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        out.write(frame)
    cap.release()
    out.release()
    fire_pct     = (fire_frames / max(total_frames // 5, 1)) * 100
    final_label  = "fire" if fire_pct > 20 else "no_fire"
    fire_weather = get_fire_weather({"label": final_label, "confidence": max_conf})
    location     = get_fire_location(max_conf) if final_label == "fire" else None
    return {
        "label":        final_label,
        "confidence":   round(max_conf, 3),
        "fire_percent": round(fire_pct, 1),
        "total_frames": total_frames,
        "fire_frames":  fire_frames,
        "output_video": out_name,
        "weather":      fire_weather,
        "location":     location,
        "boxes":        all_boxes[:10]
    }

@app.get("/video/{filename}")
def get_video(filename: str):
    p = UPLOAD_DIR / filename
    if not p.exists():
        return {"error": "not found"}
    def iter_file():
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
    return StreamingResponse(iter_file(), media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Disposition": f"inline; filename={filename}"})

@app.post("/rtsp/start")
async def rtsp_start(data: dict):
    if rtsp_state["running"]:
        return {"status": "already running"}
    url = data.get("url", "")
    if not url:
        return {"error": "No URL provided"}
    rtsp_state["running"] = True
    rtsp_state["url"]     = url
    rtsp_state["latest_result"] = None
    rtsp_state["latest_frame"]  = None
    t = threading.Thread(target=rtsp_worker, args=(url,), daemon=True)
    rtsp_state["thread"] = t
    t.start()
    return {"status": "started", "url": url}

@app.post("/rtsp/stop")
async def rtsp_stop():
    rtsp_state["running"] = False
    return {"status": "stopped"}

@app.get("/rtsp/result")
async def rtsp_result():
    return rtsp_state["latest_result"] or {"label": "waiting", "confidence": 0, "boxes": []}

@app.get("/rtsp/feed")
def rtsp_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/heatmap/{filename}")
def heatmap(filename: str):
    p = UPLOAD_DIR / filename
    return FileResponse(p) if p.exists() else {"error": "not found"}

# ── GENAI ROUTES ─────────────────────────────────────────────

@app.post("/genai/report")
async def genai_report(data: dict):
    # Safely get nested dicts — handles None values
    weather  = data.get("weather")  or {}
    location = data.get("location") or {}
    reply = ask(
        f"Generate a detailed wildfire incident report for this detection: {data}. "
        f"Location: {location.get('forest_name', 'Unknown')}, {location.get('state', 'Unknown')}. "
        f"Alert Level: {location.get('alert_level', 'UNKNOWN')}. "
        f"Weather: Temp {weather.get('temperature', 'N/A')}°C, "
        f"Humidity {weather.get('humidity', 'N/A')}%, "
        f"Wind {weather.get('wind_speed', 'N/A')} km/h. "
        "Structure with these sections: "
        "1. INCIDENT SUMMARY "
        "2. LOCATION ANALYSIS "
        "3. WEATHER RISK ANALYSIS "
        "4. FIRE SPREAD PREDICTION "
        "5. RECOMMENDED ACTIONS "
        "6. RESOURCE DEPLOYMENT. "
        "Be specific and actionable."
    )
    return {"report": reply}

@app.post("/genai/alert")
async def genai_alert(data: dict):
    # Safely get location — handles None
    location = data.get("location") or {}
    reply = ask(
        f"Write a concise emergency SMS alert for a forest ranger about this wildfire: {data}. "
        f"Location: {location.get('forest_name', 'Unknown forest')}, "
        f"{location.get('state', 'Unknown state')}. "
        f"Alert level: {location.get('alert_level', 'HIGH')}. "
        "Max 160 characters. Be direct and actionable. Return only the SMS text.",
        max_tokens=100
    )
    return {"sms": reply}

@app.post("/genai/chat")
async def genai_chat(data: dict):
    # Safely get context — handles None
    context  = data.get("context")  or {}
    location = context.get("location") or {}
    reply = ask(
        f"You are a wildfire detection AI assistant helping forest rangers. "
        f"Answer this question: {data.get('question', '')}. "
        f"Current detection context: Label={context.get('label', 'unknown')}, "
        f"Confidence={context.get('confidence', 0)}, "
        f"Location={location.get('forest_name', 'unknown')}. "
        "Give a clear, helpful, actionable answer.",
        max_tokens=500
    )
    return {"reply": reply}