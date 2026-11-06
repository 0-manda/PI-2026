import io
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(title="Trabalho PI - Regressão Linear")

# Onde o modelo fica salvo (vamos ligar essa pasta a um volume do Docker)
PASTA_MODELOS = Path("/code/modelos")
PASTA_MODELOS.mkdir(exist_ok=True)
ARQ_MODELO = PASTA_MODELOS / "modelo.joblib"
ARQ_METRICAS = PASTA_MODELOS / "metricas.json"

FEATURES = ["time-5", "time-4", "time-3", "time-2", "time-1"]
ALVO = "time"


def ler_csv(conteudo: bytes, colunas_obrigatorias: list[str]) -> pd.DataFrame:
    """Lê o CSV, confere as colunas e remove linhas inválidas."""
    try:
        df = pd.read_csv(io.BytesIO(conteudo))
    except Exception:
        raise HTTPException(400, "Não consegui ler o arquivo como CSV.")

    faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltando:
        raise HTTPException(400, f"Colunas faltando no CSV: {faltando}")

    # Converte para número (texto vira NaN) e descarta linhas com problema
    df = df[colunas_obrigatorias].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise HTTPException(400, "O CSV não tem linhas válidas.")
    return df


@app.get("/status")
def status():
    treinado = ARQ_MODELO.exists()
    resposta = {"modelo_treinado": treinado}
    if treinado:
        resposta["desempenho_esperado"] = json.loads(ARQ_METRICAS.read_text())
    return resposta


@app.post("/train")
async def train(arquivo: UploadFile = File(...)):
    df = ler_csv(await arquivo.read(), FEATURES + [ALVO])
    if len(df) < 20:
        raise HTTPException(400, "Poucas linhas para treinar (mínimo 20).")

    X, y = df[FEATURES], df[ALVO]

    # Pipeline = normalização min-max + regressão, sempre juntas
    modelo = Pipeline([
        ("minmax", MinMaxScaler()),
        ("regressao", LinearRegression()),
    ])

    # Desempenho esperado: validação cruzada 10-fold
    cv = cross_validate(
        modelo, X, y,
        cv=KFold(n_splits=10, shuffle=True, random_state=42),
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
    )
    metricas = {
        "rmse": round(float(-cv["test_rmse"].mean()), 4),
        "mae": round(float(-cv["test_mae"].mean()), 4),
        "r2": round(float(cv["test_r2"].mean()), 4),
        "linhas_treino": len(df),
    }

    # Treino final com TODOS os dados, depois salva em disco
    modelo.fit(X, y)
    joblib.dump(modelo, ARQ_MODELO)
    ARQ_METRICAS.write_text(json.dumps(metricas))

    return {"mensagem": "Modelo treinado!", "desempenho_esperado": metricas}


@app.post("/reset")
def reset():
    ARQ_MODELO.unlink(missing_ok=True)
    ARQ_METRICAS.unlink(missing_ok=True)
    return {"mensagem": "Modelo apagado. Pronto para treinar em outra base."}