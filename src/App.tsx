import { useState, useRef, useEffect } from "react";
import "./App.css";

// API base URL — empty string means relative (works in dev via vite proxy).
// In production (Vercel etc.), set VITE_API_BASE_URL to your backend host,
// e.g. "https://your-backend.onrender.com" (no trailing slash).
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

function api(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path;
}

interface ModelInfo {
  total_parameters: number;
  config: {
    block_size: number;
    vocab_size: number;
    n_layer: number;
    n_head: number;
    n_embd: number;
  };
  device: string;
}

interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
  model_exists: boolean;
}

function App() {
  const [prompt, setPrompt] = useState("Once upon a time, in a land far, far away,");
  const [maxTokens, setMaxTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(200);
  const [generations, setGenerations] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth();
    fetchModelInfo();
  }, []);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [streamingText, generations]);

  async function fetchHealth() {
    try {
      const res = await fetch(api("/api/health"));
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }

  async function fetchModelInfo() {
    try {
      const res = await fetch(api("/api/model/info"));
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch {
      setModelInfo(null);
    }
  }

  async function handleGenerate() {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setStreamingText("");

    try {
      const res = await fetch(api("/api/generate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
          top_k: topK,
          num_sequences: 1,
        }),
      });
      const data = await res.json();
      setGenerations((prev) => [...data.generations, ...prev]);
    } catch (err) {
      console.error("Generation failed:", err);
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleStreamGenerate() {
    if (!prompt.trim() || isStreaming) return;
    setIsStreaming(true);
    setStreamingText("");

    try {
      const res = await fetch(api("/api/generate/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
          top_k: topK,
          num_sequences: 1,
        }),
      });

      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n").filter((l) => l.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.done) {
              setGenerations((prev) => [fullText, ...prev]);
              setStreamingText("");
            } else if (data.token) {
              fullText += data.token;
              setStreamingText(fullText);
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } catch (err) {
      console.error("Stream generation failed:", err);
    } finally {
      setIsStreaming(false);
    }
  }

  function formatParams(n: number): string {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toString();
  }

  return (
    <div className="app">
      <header className="header">
        <h1>GPT-2</h1>
        <p className="subtitle">A minimal GPT-2 implementation from scratch</p>
        {health && (
          <div className={`status-badge ${health.model_loaded ? "online" : "offline"}`}>
            {health.model_loaded ? "Model loaded" : "Model not loaded"}
            {health.device && <span className="device"> ({health.device})</span>}
          </div>
        )}
      </header>

      <main className="main">
        {modelInfo && (
          <div className="model-info-bar">
            <span>{formatParams(modelInfo.total_parameters)} params</span>
            <span>{modelInfo.config.n_layer} layers</span>
            <span>{modelInfo.config.n_head} heads</span>
            <span>{modelInfo.config.n_embd} embd</span>
            <span>ctx {modelInfo.config.block_size}</span>
          </div>
        )}

        <section className="panel generate-panel">
          <h2>Generate</h2>
          <div className="input-group">
            <label>Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="Enter your prompt..."
            />
          </div>

          <div className="controls-row">
            <div className="input-group small">
              <label>Max tokens</label>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                min={1}
                max={1024}
              />
            </div>
            <div className="input-group small">
              <label>Temperature</label>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                min={0.1}
                max={2.0}
                step={0.1}
              />
            </div>
            <div className="input-group small">
              <label>Top-k</label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                min={1}
                max={50257}
              />
            </div>
          </div>

          <div className="button-row">
            <button
              className="btn primary"
              onClick={handleGenerate}
              disabled={isGenerating || isStreaming}
            >
              {isGenerating ? "Generating..." : "Generate"}
            </button>
            <button
              className="btn secondary"
              onClick={handleStreamGenerate}
              disabled={isGenerating || isStreaming}
            >
              {isStreaming ? "Streaming..." : "Stream Generate"}
            </button>
          </div>

          <div className="output" ref={outputRef}>
            {streamingText && (
              <div className="generation streaming">
                <span className="generation-text">{streamingText}</span>
                <span className="cursor">|</span>
              </div>
            )}
            {generations.map((text, i) => (
              <div key={i} className="generation">
                <span className="generation-label">#{generations.length - i}</span>
                <span className="generation-text">{text}</span>
              </div>
            ))}
            {!streamingText && generations.length === 0 && (
              <div className="placeholder">
                Generated text will appear here. Enter a prompt and press Generate
                to see the model continue the text.
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>GPT-2 API — built with FastAPI and React</p>
      </footer>
    </div>
  );
}

export default App;