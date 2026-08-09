# GPT-2

A minimal GPT-2 implementation from scratch, featuring a FastAPI backend for text generation and training, paired with a modern React + TypeScript frontend for an interactive experience.

<p align="center">

  <!-- Core -->
  ![GitHub License](https://img.shields.io/github/license/H0NEYP0T-466/GPT-2?style=for-the-badge&color=brightgreen)
  ![GitHub Stars](https://img.shields.io/github/stars/H0NEYP0T-466/GPT-2?style=for-the-badge&color=yellow)
  ![GitHub Forks](https://img.shields.io/github/forks/H0NEYP0T-466/GPT-2?style=for-the-badge&color=blue)
  ![GitHub Issues](https://img.shields.io/github/issues/H0NEYP0T-466/GPT-2?style=for-the-badge&color=red)
  ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/H0NEYP0T-466/GPT-2?style=for-the-badge&color=orange)
  ![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)

  <!-- Activity -->
  ![Last Commit](https://img.shields.io/github/last-commit/H0NEYP0T-466/GPT-2?style=for-the-badge&color=purple)
  ![Commit Activity](https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/GPT-2?style=for-the-badge&color=teal)
  ![Repo Size](https://img.shields.io/github/repo-size/H0NEYP0T-466/GPT-2?style=for-the-badge&color=blueviolet)
  ![Code Size](https://img.shields.io/github/languages/code-size/H0NEYP0T-466/GPT-2?style=for-the-badge&color=indigo)

  <!-- Languages -->
  ![Top Language](https://img.shields.io/github/languages/top/H0NEYP0T-466/GPT-2?style=for-the-badge&color=critical)
  ![Languages Count](https://img.shields.io/github/languages/count/H0NEYP0T-466/GPT-2?style=for-the-badge&color=success)

  <!-- Community -->
  ![Discussions](https://img.shields.io/github/discussions/H0NEYP0T-466/GPT-2?style=for-the-badge&color=blue)
  ![Documentation](https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white)
  ![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)

</p>

A lightweight, from-scratch reimplementation of OpenAI's GPT-2 architecture (124M parameters) with a FastAPI-powered REST API and a React-based UI. Train the model on your own dataset, stream generated text in real time, and inspect model internals — all from a single command.

---

## 🔗 Links

- [📖 Documentation](README.md)
- [🚀 Live Demo](#-usage-examples)
- [🐛 Issues](https://github.com/H0NEYP0T-466/GPT-2/issues)
- [🤝 Contributing](CONTRIBUTING.md)

---

## 📑 Table of Contents

- [🚀 Installation](#-installation)
- [⚡ Usage Examples](#-usage-examples)
- [✨ Features](#-features)
- [📂 Folder Structure](#-folder-structure)
- [📦 Submodules](#-submodules)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🛡 Security](#-security)
- [📏 Code of Conduct](#-code-of-conduct)
- [🛠 Tech Stack](#-tech-stack)
- [📦 Dependencies & Packages](#-dependencies--packages)

---

## 🚀 Installation

### Prerequisites

- **Node.js** 18+ (for the frontend)
- **Python** 3.10+ (for the backend)
- **pip** (Python package manager)
- A GPU (optional, but recommended for training)

### Step-by-Step Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/H0NEYP0T-466/GPT-2.git
   cd GPT-2
   ```

2. **Install backend dependencies**

   The backend uses a Python virtual environment. A `venv` is automatically created by `start.sh`, or you can set it up manually:

   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**

   ```bash
   cd ..
   npm install
   ```

4. **(Optional) Download pre-trained weights**

   The backend automatically downloads OpenAI's official GPT-2 weights on first run if no trained model is found in `backend/model/`. To skip this, place your own `model.pt` in that directory.

5. **Start both services**

   ```bash
   ./start.sh
   ```

   Or start them manually:

   ```bash
   # Backend (in one terminal)
   cd backend && source venv/bin/activate && python app.py

   # Frontend (in another terminal)
   npm run dev
   ```

---

## ⚡ Usage Examples

### API Endpoints

The FastAPI backend exposes the following REST endpoints (docs available at `http://localhost:8011/docs`):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Check model health and device |
| `GET` | `/api/model/info` | Get model architecture info |
| `POST` | `/api/generate` | Generate text from a prompt |
| `POST` | `/api/generate/stream` | Stream generated tokens in real time |
| `POST` | `/api/train` | Fine-tune the model on your dataset |
| `GET` | `/api/train/status` | Get training progress |

#### Generate Text (REST)

```bash
curl -X POST http://localhost:8011/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_k": 200
  }'
```

#### Stream Generation (SSE-like JSON stream)

```bash
curl -X POST http://localhost:8011/api/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "In a distant galaxy", "max_new_tokens": 50}'
```

Each line is a JSON object with `token` and `token_id`, ending with `{"done": true}`.

#### Train the Model

```bash
curl -X POST http://localhost:8011/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "iterations": 50,
    "batch_size": 4,
    "block_size": 32,
    "learning_rate": 0.0003
  }'
```

### Frontend UI

Visit `http://localhost:5173` after starting the project. The UI lets you:

- Enter a prompt and adjust generation parameters (max tokens, temperature, top-k).
- Generate text in one shot or stream it token-by-token.
- View model info (parameter count, layers, heads, embedding dim, context length).
- See the model's health status and device (CPU/GPU).

### Direct Python Usage

You can also run the training script directly:

```bash
cd backend
source venv/bin/activate
python GPT2.py
```

This trains the model from scratch on `data.txt` using the hyperparameters defined in `main()`.

---

## ✨ Features

- **From-scratch GPT-2** — Clean PyTorch implementation of the transformer decoder architecture (causal self-attention, MLP, layer norm, positional embeddings).
- **Pre-trained weight loading** — Load official OpenAI GPT-2 weights via `transformers` with weight-transposition handling.
- **FastAPI REST API** — Full-featured API for generation, streaming generation, and training with CORS support.
- **Streaming responses** — Token-by-token SSE-style streaming via `StreamingResponse`.
- **Training pipeline** — Built-in data loader, mixed-precision training (AMP), gradient accumulation, cosine LR schedule, and evaluation loop.
- **React + TypeScript frontend** — Modern UI with parameter controls, live streaming output, and model status dashboard.
- **Vite dev server** — Fast HMR, proxy to backend API, and production-ready build.
- **Oxlint linting** — Fast Rust-based linter for TypeScript/React code.
- **One-command startup** — `start.sh` spins up both backend and frontend with auto venv creation.
- **Kaggle-compatible** — Dataset path logic works on Kaggle notebooks out of the box.

---

## 📂 Folder Structure

```
GPT-2/
├── backend/
│   ├── app.py                    # FastAPI server + REST endpoints
│   ├── GPT2.py                   # GPT-2 model, DataLoader, training loop
│   ├── requirements.txt          # Python dependencies
│   ├── run_commands.txt          # Helper commands
│   ├── dataset/
│   │   ├── data.txt              # Training dataset
│   │   └── input.txt             # Sample input
│   └── model/
│       └── model.pt              # Trained model checkpoint (gitignored)
├── public/
│   ├── favicon.svg
│   └── icons.svg
├── src/
│   ├── App.tsx                   # Main React component (UI + API calls)
│   ├── App.css                   # Component styles
│   ├── main.tsx                  # React entry point
│   ├── index.css                 # Global design tokens & reset
│   └── assets/
│       ├── hero.png
│       ├── react.svg
│       └── vite.svg
├── .github/                      # GitHub automation (issue templates, PR template)
├── dist/                         # Production build output (gitignored)
├── node_modules/                 # npm dependencies (gitignored)
├── .gitignore
├── .oxlintrc.json                # Oxlint configuration
├── index.html                    # HTML entry point
├── package.json
├── package-lock.json
├── README.md
├── start.sh                      # One-command startup script
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
└── vite.config.ts                # Vite config with API proxy
```

---

## 📦 Submodules

No Git submodules are currently configured. The repository is self-contained. If you'd like to split the backend or frontend into separate sub-repos, open an issue to discuss.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to fork, submit PRs, report bugs, and propose features.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🛡 Security

If you believe you've found a security vulnerability, please report it responsibly. See [SECURITY.md](SECURITY.md) for our disclosure policy.

---

## 📏 Code of Conduct

We expect all participants to follow our Code of Conduct. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the full text.

---

## 🛠 Tech Stack

### Languages

![TypeScript](https://img.shields.io/badge/TypeScript-%233178C6.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)

### Frameworks & Libraries

![FastAPI](https://img.shields.io/badge/FastAPI-%230055FF.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-%2361DAFB.svg?style=for-the-badge&logo=react&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-%23646EFF.svg?style=for-the-badge&logo=vite&logoColor=white)

### Databases

None — the model and data are stored as files on disk.

### DevOps / CI / Tools

![Uvicorn](https://img.shields.io/badge/Uvicorn-%23177245.svg?style=for-the-badge&logo=uvicorn&logoColor=white)
![Oxlint](https://img.shields.io/badge/Oxlint-%23FF6B35.svg?style=for-the-badge&logo=oxc&logoColor=white)
![npm](https://img.shields.io/badge/npm-%23CB3837.svg?style=for-the-badge&logo=npm&logoColor=white)
![pip](https://img.shields.io/badge/pip-%23306998.svg?style=for-the-badge&logo=python&logoColor=white)
![Git](https://img.shields.io/badge/Git-%23F05032.svg?style=for-the-badge&logo=git&logoColor=white)

### Cloud / Hosting

Self-hosted / local deployment. Compatible with any environment that supports Python 3.10+ and Node.js 18+.

---

## 📦 Dependencies & Packages

### Runtime Dependencies

#### Python (Backend)

![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi) — Web framework for the API.
![uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn) — ASGI server.
![torch](https://img.shields.io/pypi/v/torch?style=for-the-badge&label=torch) — Deep learning framework.
![tiktoken](https://img.shields.io/pypi/v/tiktoken?style=for-the-badge&label=tiktoken) — BPE tokenizer for GPT-2.
![transformers](https://img.shields.io/pypi/v/transformers?style=for-the-badge&label=transformers) — Hugging Face library for loading pre-trained weights.
![pydantic](https://img.shields.io/pypi/v/pydantic?style=for-the-badge&label=pydantic) — Data validation and settings management.

#### JavaScript / TypeScript (Frontend)

![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react) — UI library.
![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom) — React DOM renderer.

### Dev Dependencies

#### Python (Backend)

_No dedicated dev dependencies in requirements.txt. The backend venv is used for runtime only._

#### JavaScript / TypeScript (Frontend)

![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite) — Build tool and dev server.
[@vitejs/plugin-react](https://img.shields.io/npm/v/@vitejs/plugin-react?style=for-the-badge&label=@vitejs/plugin-react) — Vite plugin for React Fast Refresh.
![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript) — TypeScript compiler.
![oxlint](https://img.shields.io/npm/v/oxlint?style=for-the-badge&label=oxlint) — Fast Rust-based linter.
[@types/node](https://img.shields.io/npm/v/@types/node?style=for-the-badge&label=@types/node) — Node.js type definitions.
[@types/react](https://img.shields.io/npm/v/@types/react?style=for-the-badge&label=@types/react) — React type definitions.
[@types/react-dom](https://img.shields.io/npm/v/@types/react-dom?style=for-the-badge&label=@types/react-dom) — React DOM type definitions.

### Peer / Optional Dependencies

None. All dependencies are resolved from the manifest files.

<details>
<summary>Click to expand: exact versions from lockfiles</summary>

**package.json** (frontend):

| Package | Version Range |
|---------|---------------|
| react | ^19.2.7 |
| react-dom | ^19.2.7 |
| @types/node | ^24.13.2 |
| @types/react | ^19.2.17 |
| @types/react-dom | ^19.2.3 |
| @vitejs/plugin-react | ^6.0.3 |
| oxlint | ^1.71.0 |
| typescript | ~6.0.2 |
| vite | ^8.1.1 |

**backend/requirements.txt** (backend):

| Package | Note |
|---------|------|
| fastapi | Latest from PyPI |
| uvicorn[standard] | Latest from PyPI |
| torch | Latest from PyPI (CUDA support included) |
| tiktoken | Latest from PyPI |
| transformers | Latest from PyPI |
| pydantic | Latest from PyPI |

Exact pinned versions are recorded in `package-lock.json` and the pip resolver at install time.

</details>

---

<p align="center">Made with ❤ by H0NEYP0T-466</p>