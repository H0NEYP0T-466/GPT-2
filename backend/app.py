import os
import sys
import json
import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Add backend dir to path so we can import GPT2
sys.path.insert(0, os.path.dirname(__file__))
from GPT2 import GPT2, GPT2Config, get_device, DATASET_PATH

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model.pt")

# Global state
model = None
device = None
encoding = None
training_state = {
    "is_training": False,
    "current_iteration": 0,
    "total_iterations": 0,
    "current_loss": 0.0,
    "losses": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, encoding
    device = get_device()
    encoding = tiktoken.get_encoding("gpt2")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading trained model from {MODEL_SAVE_PATH}...")
        model = GPT2(GPT2Config())
        model.load(MODEL_SAVE_PATH)
    else:
        print("No trained model found. Loading pretrained GPT-2 weights as fallback...")
        model = GPT2.from_pretrained("gpt2")
        model.save(MODEL_SAVE_PATH)

    model.to(device)
    model.eval()
    print(f"Model ready on {device}")
    yield


app = FastAPI(title="GPT-2 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 200
    num_sequences: int = 1


class TrainRequest(BaseModel):
    iterations: int = 50
    batch_size: int = 4
    block_size: int = 32
    learning_rate: float = 3e-4


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "model_exists": os.path.exists(MODEL_SAVE_PATH),
    }


@app.get("/api/model/info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "total_parameters": total_params,
        "config": {
            "block_size": model.config.block_size,
            "vocab_size": model.config.vocab_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
        },
        "device": str(device),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model.eval()
    results = []

    for _ in range(req.num_sequences):
        tokens = encoding.encode(req.prompt)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        x = x.repeat(1, 1)

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        generated = model.generate(
            x,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        text = encoding.decode(generated[0].tolist())
        results.append(text)

    return {"generations": results, "prompt": req.prompt}


@app.post("/api/generate/stream")
def generate_stream(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model.eval()
    tokens = encoding.encode(req.prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    def token_generator():
        current = x
        for _ in range(req.max_new_tokens):
            with torch.no_grad():
                logits, _ = model(current)
                logits = logits[:, -1, :] / req.temperature
                if req.top_k is not None:
                    v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token[0, 0].item()
                token_text = encoding.decode([token_id])
                current = torch.cat((current, next_token), dim=1)
                yield json.dumps({"token": token_text, "token_id": token_id}) + "\n"
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(token_generator(), media_type="application/json")


@app.post("/api/train")
def train(req: TrainRequest):
    global model, training_state

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if training_state["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    training_state["is_training"] = True
    training_state["current_iteration"] = 0
    training_state["total_iterations"] = req.iterations
    training_state["losses"] = []

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=req.learning_rate)
    loader = __import__("GPT2").DataLoader(req.batch_size, req.block_size, DATASET_PATH)

    for i in range(req.iterations):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        training_state["current_iteration"] = i + 1
        training_state["current_loss"] = loss_val
        training_state["losses"].append(loss_val)

    model.eval()
    model.save()
    training_state["is_training"] = False

    return {
        "status": "complete",
        "iterations": req.iterations,
        "final_loss": training_state["current_loss"],
        "losses": training_state["losses"],
    }


@app.get("/api/train/status")
def train_status():
    return training_state


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
