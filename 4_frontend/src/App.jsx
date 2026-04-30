import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function App() {
  const [file, setFile]       = useState(null);
  const [result, setResult]   = useState(null);
  const [report, setReport]   = useState(null);
  const [sms, setSms]         = useState(null);
  const [chat, setChat]       = useState("");
  const [reply, setReply]     = useState(null);
  const [loc, setLoc]         = useState("Forest Zone A");
  const [loading, setLoading] = useState("");

  const analyse = async () => {
    if (!file) return;
    setLoading("Analysing image...");
    const fd = new FormData();
    fd.append("file", file);
    const { data } = await axios.post(`${API}/predict/image`, fd);
    setResult(data);
    setLoading("");
  };

  const genReport = async () => {
    setLoading("Generating report...");
    const { data } = await axios.post(`${API}/genai/report`, result);
    setReport(data.report);
    setLoading("");
  };

  const genAlert = async () => {
    setLoading("Generating SMS...");
    const { data } = await axios.post(`${API}/genai/alert`, { ...result, location: loc });
    setSms(data.sms);
    setLoading("");
  };

  const askChat = async () => {
    setLoading("Thinking...");
    const { data } = await axios.post(`${API}/genai/chat`, { question: chat, context: result });
    setReply(data.reply);
    setLoading("");
  };

  return (
    <div style={{ maxWidth: 700, margin: "40px auto", fontFamily: "sans-serif", padding: 20 }}>
      <h1>🔥 Wildfire Detection Dashboard</h1>

      <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 20, marginBottom: 20 }}>
        <h2>📷 Upload Drone Image</h2>
        <input type="file" accept="image/*" onChange={e => setFile(e.target.files[0])} />
        <br /><br />
        <button onClick={analyse} style={btn("#e74c3c")}>Analyse</button>
        {loading && <p>⏳ {loading}</p>}
        {result && (
          <div style={{ marginTop: 12, padding: 12, background: result.label === "fire" ? "#fde8e8" : "#e8fde8", borderRadius: 6 }}>
            <strong>{result.label === "fire" ? "🔥 FIRE DETECTED" : "✅ No Fire"}</strong>
            <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          </div>
        )}
      </div>

      {result && (
        <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 20 }}>
          <h2>🤖 Phase 7 — Generative AI Layer</h2>

          <div style={{ marginBottom: 20 }}>
            <h3>📋 Incident Report</h3>
            <button onClick={genReport} style={btn("#2980b9")}>Generate Report</button>
            {report && <pre style={{ background: "#f4f4f4", padding: 12, borderRadius: 6, overflow: "auto", fontSize: 13 }}>{report}</pre>}
          </div>

          <div style={{ marginBottom: 20 }}>
            <h3>📲 Ranger SMS Alert</h3>
            <input value={loc} onChange={e => setLoc(e.target.value)} placeholder="Location"
              style={{ padding: 8, marginRight: 8, borderRadius: 4, border: "1px solid #ccc", width: 200 }} />
            <button onClick={genAlert} style={btn("#27ae60")}>Generate SMS</button>
            {sms && <div style={{ marginTop: 10, padding: 12, background: "#f0fff4", borderRadius: 6 }}>{sms}</div>}
          </div>

          <div>
            <h3>💬 Ask the Dashboard</h3>
            <input value={chat} onChange={e => setChat(e.target.value)} placeholder="Ask anything about the detection..."
              style={{ padding: 8, width: "65%", marginRight: 8, borderRadius: 4, border: "1px solid #ccc" }} />
            <button onClick={askChat} style={btn("#8e44ad")}>Ask</button>
            {reply && <div style={{ marginTop: 10, padding: 12, background: "#f8f0ff", borderRadius: 6 }}>{reply}</div>}
          </div>
        </div>
      )}
    </div>
  );
}

const btn = (bg) => ({
  background: bg, color: "#fff", border: "none", padding: "8px 16px",
  borderRadius: 6, cursor: "pointer"
});