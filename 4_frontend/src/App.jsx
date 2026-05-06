import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function App() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [report, setReport]   = useState(null);
  const [sms, setSms]         = useState(null);
  const [chat, setChat]       = useState("");
  const [reply, setReply]     = useState(null);
  const [loc, setLoc]         = useState("Forest Zone A");
  const [loading, setLoading] = useState("");
  const [error, setError]     = useState(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null); setReport(null); setSms(null); setReply(null); setError(null);
  };

  const analyse = async () => {
    if (!file) return;
    setLoading("Analysing image...");
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const { data } = await axios.post(`${API}/predict/image`, fd);
      setResult(data);
    } catch (e) {
      setError("❌ Could not connect to backend. Make sure uvicorn is running on port 8000.");
    }
    setLoading("");
  };

  const genReport = async () => {
    setLoading("Generating report...");
    setError(null);
    try {
      const { data } = await axios.post(`${API}/genai/report`, result);
      setReport(data.report);
    } catch (e) {
      setError("❌ Report generation failed. Check your GROQ_API_KEY.");
    }
    setLoading("");
  };

  const genAlert = async () => {
    setLoading("Generating SMS...");
    setError(null);
    try {
      const { data } = await axios.post(`${API}/genai/alert`, { ...result, location: loc });
      setSms(data.sms);
    } catch (e) {
      setError("❌ SMS generation failed. Check your GROQ_API_KEY.");
    }
    setLoading("");
  };

  const askChat = async () => {
    if (!chat.trim()) return;
    setLoading("Thinking...");
    setError(null);
    try {
      const { data } = await axios.post(`${API}/genai/chat`, { question: chat, context: result });
      setReply(data.reply);
    } catch (e) {
      setError("❌ Chat failed. Check your GROQ_API_KEY.");
    }
    setLoading("");
  };

  const isFire = result?.label === "fire";

  return (
    <div style={{ maxWidth: 750, margin: "40px auto", fontFamily: "sans-serif", padding: 20 }}>
      <h1 style={{ color: "#e74c3c" }}>🔥 Wildfire Detection Dashboard</h1>

      {error && (
        <div style={{ background: "#fde8e8", border: "1px solid #e74c3c", borderRadius: 8, padding: 12, marginBottom: 16, color: "#c0392b" }}>
          {error}
        </div>
      )}

      {/* Upload Section */}
      <div style={card}>
        <h2>📷 Upload Drone Image</h2>
        <input type="file" accept="image/*" onChange={handleFile} />
        {preview && (
          <img src={preview} alt="preview"
            style={{ display: "block", marginTop: 12, maxHeight: 200, borderRadius: 8, objectFit: "cover", width: "100%" }} />
        )}
        <br />
        <button onClick={analyse} disabled={!file || !!loading} style={btn("#e74c3c")}>
          {loading === "Analysing image..." ? "⏳ Analysing..." : "🔍 Analyse Image"}
        </button>

        {loading && <p style={{ color: "#888" }}>⏳ {loading}</p>}

        {result && (
          <div style={{ marginTop: 12, padding: 16, background: isFire ? "#fde8e8" : "#e8fde8", borderRadius: 8, border: `1px solid ${isFire ? "#e74c3c" : "#27ae60"}` }}>
            <strong style={{ fontSize: 18, color: isFire ? "#c0392b" : "#1e8449" }}>
              {isFire ? "🔥 FIRE DETECTED" : "✅ No Fire Detected"}
            </strong>
            <p style={{ margin: "8px 0 0" }}>Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong></p>
            {result.boxes && result.boxes.length > 0 && (
              <div style={{ marginTop: 8, fontSize: 13, color: "#555" }}>
                <strong>YOLO Detections:</strong>
                {result.boxes.map((box, i) => (
                  <div key={i}>• {box.label} — {(box.confidence * 100).toFixed(1)}%</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* GenAI Section */}
      {result && (
        <div style={card}>
          <h2>🤖 Generative AI Layer</h2>

          {/* Incident Report */}
          <div style={section}>
            <h3>📋 Incident Report</h3>
            <button onClick={genReport} disabled={!!loading} style={btn("#2980b9")}>
              {loading === "Generating report..." ? "⏳ Generating..." : "Generate Report"}
            </button>
            {report && (
              <pre style={{ background: "#f4f4f4", padding: 14, borderRadius: 8, overflow: "auto", fontSize: 13, marginTop: 10, whiteSpace: "pre-wrap" }}>
                {report}
              </pre>
            )}
          </div>

          {/* SMS Alert */}
          <div style={section}>
            <h3>📲 Ranger SMS Alert</h3>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input value={loc} onChange={e => setLoc(e.target.value)}
                placeholder="Location name..."
                style={input} />
              <button onClick={genAlert} disabled={!!loading} style={btn("#27ae60")}>
                {loading === "Generating SMS..." ? "⏳ Generating..." : "Generate SMS"}
              </button>
            </div>
            {sms && (
              <div style={{ marginTop: 10, padding: 12, background: "#f0fff4", border: "1px solid #27ae60", borderRadius: 8, fontSize: 14 }}>
                📱 <strong>SMS Preview:</strong><br />{sms}
              </div>
            )}
          </div>

          {/* Ask Claude */}
          <div style={section}>
            <h3>💬 Ask the Dashboard</h3>
            <div style={{ display: "flex", gap: 8 }}>
              <input value={chat} onChange={e => setChat(e.target.value)}
                placeholder="e.g. Is this dangerous? What should rangers do?"
                style={{ ...input, flex: 1 }}
                onKeyDown={e => e.key === "Enter" && askChat()} />
              <button onClick={askChat} disabled={!!loading} style={btn("#8e44ad")}>
                {loading === "Thinking..." ? "⏳" : "Ask"}
              </button>
            </div>
            {reply && (
              <div style={{ marginTop: 10, padding: 12, background: "#f8f0ff", border: "1px solid #8e44ad", borderRadius: 8, fontSize: 14, lineHeight: 1.6 }}>
                ✨ <strong>AI Response:</strong><br />{reply}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const card = { border: "1px solid #ddd", borderRadius: 10, padding: 24, marginBottom: 20, boxShadow: "0 2px 8px rgba(0,0,0,0.06)" };
const section = { marginBottom: 20, paddingBottom: 20, borderBottom: "1px solid #eee" };
const input = { padding: "8px 12px", borderRadius: 6, border: "1px solid #ccc", fontSize: 14, width: 200 };
const btn = (bg) => ({ background: bg, color: "#fff", border: "none", padding: "9px 18px", borderRadius: 6, cursor: "pointer", fontWeight: 600, fontSize: 14 });