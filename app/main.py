import json, os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from . import store
from .models import Assessment, Claim
from .synth import sample_claims
from .tools import investigate

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.getenv("PORT", "8001"))
app = FastAPI(title="ResolveAI", description="Evidence-linked insurance claim investigation (human-in-the-loop)")

@app.get("/api/samples")
def samples(): return sample_claims()

@app.post("/api/investigate", response_model=Assessment)
async def run(claim: Claim, mode: str = "deterministic"):
    if mode == "adk":
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI")):
            raise HTTPException(400, "ADK mode needs GOOGLE_API_KEY or Vertex AI credentials")
        from ..agentsents import investigate_adk
        a = await investigate_adk(claim)
    else:
        a = investigate(claim)
    store.save(a)
    return a

@app.get("/api/eval")
def eval_results():
    p = ROOT / "eval" / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}

app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
