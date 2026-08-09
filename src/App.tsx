import { useState, useRef, useEffect } from "react";
import "./App.css";

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

interface TrainStatus {
  is_training: boolean;
  current_iteration: number;
  total_iterations: number;
  current_loss: number;
  losses: number[];
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
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);

  const [trainIterations, setTrainIterations] = useState(50);
  const [isTraining, setIsTraining] = useState(false);
  const [trainLosses, setTrainLosses] = useState<number[]>([]);

  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth();
    fetchModelInfo();
    fetchTrainStatus();
    const interval = setInterval(fetchTrainStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [streamingText, generations]);

  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }

  async function fetchModelInfo() {
    try {
      const res = await fetch("/api/model/info");
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch {
      setModelInfo(null);
    }
  }

  async function fetchTrainStatus() {
    try {
      const res = await fetch("/api/train/status");
      if (res.ok) {
        const data = await res.json();
        setTrainStatus(data);
        setIsTraining(data.is_training);
        if (data.losses?.length > 0) {
          setTrainLosses(data.losses);
        }
      }
    } catch {
      // ignore
    }
  }

  async function handleGenerate() {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setStreamingText("");

    try {
      const res = await fetch("/api/generate", {
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
      const res = await fetch("/api/generate/stream", {
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

  async function handleTrain() {
    if (isTraining) return;
    setIsTraining(true);
    setTrainLosses([]);

    try {
      await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ iterations: trainIterations }),
      });
      fetchTrainStatus();
    } catch (err) {
      console.error("Training failed:", err);
    } finally {
      setIsTraining(false);
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
        <p className="subtitle">From scratch, following Andrej Karpathy</p>
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

        <div className="panels">
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
                  Generated text will appear here. This is a randomly initialized GPT-2
                  model trained on Shakespeare's Coriolanus — output will be rough until
                  more training is done.
                </div>
              )}
            </div>
          </section>

          <section className="panel train-panel">
            <h2>Train on Shakespeare</h2>
            <p className="train-desc">
              Fine-tune the model on Coriolanus (Act 1, Scene 1). Training from
              scratch on this tiny dataset produces Shakespeare-ish gibberish —
              that's the point.
            </p>

            <div className="input-group small">
              <label>Iterations</label>
              <input
                type="number"
                value={trainIterations}
                onChange={(e) => setTrainIterations(Number(e.target.value))}
                min={1}
                max={10000}
              />
            </div>

            <button
              className="btn primary"
              onClick={handleTrain}
              disabled={isTraining}
            >
              {isTraining
                ? `Training... ${trainStatus?.current_iteration || 0}/${trainStatus?.total_iterations || 0}`
                : "Start Training"}
            </button>

            {trainStatus?.is_training && (
              <div className="train-progress">
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${((trainStatus.current_iteration / trainStatus.total_iterations) * 100)}%`,
                    }}
                  />
                </div>
                <span className="loss">
                  Loss: {trainStatus.current_loss.toFixed(4)}
                </span>
              </div>
            )}

            {trainLosses.length > 0 && (
              <div className="loss-chart">
                <h3>Training Loss</h3>
                <div className="chart">
                  {trainLosses.map((loss, i) => {
                    const maxLoss = Math.max(...trainLosses);
                    const height = maxLoss > 0 ? (loss / maxLoss) * 100 : 0;
                    return (
                      <div
                        key={i}
                        className="bar"
                        style={{ height: `${height}%` }}
                        title={`Iter ${i + 1}: ${loss.toFixed(4)}`}
                      />
                    );
                  })}
                </div>
                <div className="chart-labels">
                  <span>1</span>
                  <span>{trainLosses.length}</span>
                </div>
              </div>
            )}

            <div className="dataset-info">
              <h3>Dataset</h3>
              <p>Shakespeare — Coriolanus, Act 1, Scene 1</p>
              <p>The classic Karpathy "Let's build GPT" training text.</p>
            </div>
          </section>
        </div>
      </main>

      <footer className="footer">
        <p>
          Built following{" "}
          <a
            href="https://www.youtube.com/watch?v=kCc8FmEb1nY"
            target="_blank"
            rel="noreferrer"
          >
            Andrej Karpathy's "Let's build GPT from scratch"
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
