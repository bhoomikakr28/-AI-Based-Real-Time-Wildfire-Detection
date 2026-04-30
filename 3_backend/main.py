import os, shutil, uuid, requests, cv2, base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from detect import predict_image
from groq import Groq
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask(prompt, max_tokens=1000):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_weather(city="Bengaluru"):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        r = requests.get(url, timeout=5)
        d = r.json()
        current = d["current_condition"][0]
        return {
            "temperature": int(current["temp_C"]),
            "feels_like": int(current["FeelsLikeC"]),
            "humidity": int(current["humidity"]),
            "wind_speed": int(current["windspeedKmph"]),
            "visibility": int(current["visibility"]),
            "description": current["weatherDesc"][0]["value"],
            "uv_index": int(current["uvIndex"]),
        }
    except:
        return {
            "temperature": 32,
            "feels_like": 36,
            "humidity": 45,
            "wind_speed": 18,
            "visibility": 10,
            "description": "Partly cloudy",
            "uv_index": 6,
        }

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/weather")
def weather(city: str = "Bengaluru"):
    return get_weather(city)

@app.post("/predict/image")
async def predict(file: UploadFile = File(...)):
    fname = f"{uuid.uuid4()}_{file.filename}"
    fpath = UPLOAD_DIR / fname
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = predict_image(Image.open(fpath))
    weather = get_weather()
    return {**result, "filename": fname, "weather": weather}

@app.post("/predict/frame")
async def predict_frame(data: dict):
    try:
        img_data = base64.b64decode(data["frame"].split(",")[1])
        img_array = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        result = predict_image(pil_img)
        return {**result, "weather": get_weather()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/heatmap/{filename}")
def heatmap(filename: str):
    p = UPLOAD_DIR / filename
    return FileResponse(p) if p.exists() else {"error": "not found"}

@app.post("/genai/report")
async def genai_report(data: dict):
    weather = data.get("weather", {})
    reply = ask(
        f"Generate a detailed wildfire incident report for this detection: {data}. "
        f"Weather conditions: Temperature {weather.get('temperature')}°C, "
        f"Humidity {weather.get('humidity')}%, Wind {weather.get('wind_speed')} km/h. "
        "Structure the report with these sections: "
        "1. INCIDENT SUMMARY, 2. WEATHER RISK ANALYSIS, "
        "3. FIRE SPREAD PREDICTION, 4. RECOMMENDED ACTIONS, 5. RESOURCE DEPLOYMENT. "
        "Be specific and actionable. Use clear headings."
    )
    return {"report": reply}

@app.post("/genai/alert")
async def genai_alert(data: dict):
    reply = ask(
        f"Write a concise emergency SMS alert for a forest ranger about this wildfire detection: {data}. "
        "Max 160 characters. Be direct and actionable. Return only the SMS text.",
        max_tokens=100
    )
    return {"sms": reply}

@app.post("/genai/chat")
async def genai_chat(data: dict):
    reply = ask(
        f"You are a wildfire detection AI assistant. Answer this ranger question: {data.get('question', '')}. "
        f"Current detection data: {data.get('context', {})}. Give a clear, helpful answer.",
        max_tokens=500
    )
    return {"reply": reply}