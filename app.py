from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd
import shutil

app = FastAPI(title="InduxAI Motor IA")

MODEL_DIR = Path("/app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {}


class CaoLivreInput(BaseModel):
    rotacao_forno: float
    torque_forno: float
    vazao_combustivel_calcinador: float
    vazao_combustivel_forno: float
    vazao_af: float
    rotacao_id_fan: float
    temperatura_calcinacao: float
    temperatura_zona_queima: float
    temperatura_ar_secundario: float
    temperatura_ar_terciario: float
    rotacao_grelha: float


@app.get("/")
def health():
    return {
        "status": "online",
        "service": "InduxAI Motor IA"
    }


@app.post("/models/upload")
def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Envie apenas arquivos .pkl")

    destination = MODEL_DIR / file.filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "ok",
        "message": "Modelo enviado com sucesso",
        "file": file.filename
    }


@app.get("/models")
def list_models():
    files = [f.name for f in MODEL_DIR.glob("*.pkl")]
    return {
        "models": files
    }


@app.post("/models/load/{model_name}")
def load_model(model_name: str):
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    MODELS["cao_livre"] = joblib.load(model_path)

    return {
        "status": "ok",
        "message": "Modelo carregado em memória",
        "model": model_name
    }


@app.post("/predict/cao-livre")
def predict_cao_livre(data: CaoLivreInput):
    if "cao_livre" not in MODELS:
        raise HTTPException(
            status_code=400,
            detail="Nenhum modelo de CaO Livre carregado. Use /models/load primeiro."
        )

    df = pd.DataFrame([data.dict()])

    prediction = MODELS["cao_livre"].predict(df)[0]

    return {
        "variavel": "CaO Livre",
        "predicao": round(float(prediction), 3),
        "unidade": "%",
        "status": "ok"
    }