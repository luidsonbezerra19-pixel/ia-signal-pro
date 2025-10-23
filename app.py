
import os
import math
import time
import io
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image, ImageOps

# =========================
# CONFIG (sem .env necessário)
# =========================
APP_NAME = "IA PRO - SINAL ÚNICO (Kraken)"
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]  # BNB pode não existir na Kraken/USDT
DEFAULT_INTERVAL_MIN = 1  # minutos
CANDLES = 800  # usa ~800 velas para indicadores estáveis
PORT = int(os.environ.get("PORT", "8080"))

# Mapeia símbolos app -> par Kraken
# Observação: Kraken usa "XBT" para BTC. USDT existe para vários pares.
KRAKEN_PAIR_MAP = {
    "BTC/USDT": "XBTUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USDT": "XRPUSDT",
    "ADA/USDT": "ADAUSDT",
    # Fallbacks USD se USDT não existir no futuro (exemplo):
    "BTC/USD": "XXBTZUSD",
    "ETH/USD": "XETHZUSD",
    "SOL/USD": "SOLUSD",
    "XRP/USD": "XXRPZUSD",
    "ADA/USD": "ADAUSD",
}

# =========================
# App
# =========================
app = Flask(__name__)

# =========================
# Utils
# =========================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def to_kraken_pair(symbol: str) -> str:
    if symbol in KRAKEN_PAIR_MAP:
        return KRAKEN_PAIR_MAP[symbol]
    # tenta variações
    s = symbol.upper().replace("-", "/")
    if s in KRAKEN_PAIR_MAP:
        return KRAKEN_PAIR_MAP[s]
    # tenta troca de XBT/BTC
    if s.startswith("BTC/"):
        s_try = s.replace("BTC", "XBT")
        return KRAKEN_PAIR_MAP.get(s_try, s_try.replace("/", ""))
    return s.replace("/", "")

def fetch_ohlc_kraken(symbol: str, interval_min: int = DEFAULT_INTERVAL_MIN, candles: int = CANDLES) -> pd.DataFrame:
    """Busca OHLC da Kraken via REST pública."""
    pair = to_kraken_pair(symbol)
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval_min}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")

    # A resposta contém { 'result': { '<PAIR>': [[time, o, h, l, c, v, vwap, count], ...], 'last': <ts> } }
    # Pega a primeira key de par no result
    res = data["result"]
    # remove a key "last"
    if "last" in res:
        res.pop("last", None)
    # pega a primeira lista dentro do result
    if not res:
        raise RuntimeError("Sem dados da Kraken.")

    pair_key = list(res.keys())[0]
    rows = res[pair_key]
    # Converte para DataFrame
    cols = ["time", "open", "high", "low", "close", "volume", "vwap", "count"]
    df = pd.DataFrame(rows, columns=cols)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume", "vwap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # Limita a N candles recentes
    if len(df) > candles:
        df = df.iloc[-candles:].reset_index(drop=True)
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def trend_flags(close: pd.Series) -> Dict[str, float]:
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    slope200 = ema200.diff().iloc[-1]
    bull = float((ema9.iloc[-1] > ema21.iloc[-1]) and (ema21.iloc[-1] > ema50.iloc[-1]) and (ema50.iloc[-1] > ema200.iloc[-1]) and (slope200 > 0))
    bear = float((ema9.iloc[-1] < ema21.iloc[-1]) and (ema21.iloc[-1] < ema50.iloc[-1]) and (ema50.iloc[-1] < ema200.iloc[-1]) and (slope200 < 0))
    flat = 1.0 - max(bull, bear)
    return {"ema9": float(ema9.iloc[-1]), "ema21": float(ema21.iloc[-1]), "ema50": float(ema50.iloc[-1]), "ema200": float(ema200.iloc[-1]), "bull": bull, "bear": bear, "flat": float(flat)}

def smart_ai_decision(df: pd.DataFrame) -> Dict[str, float]:
    """
    IA 'única' e direta: consolida MACD, RSI e Tendência em uma pontuação.
    Retorna direção 'BUY' ou 'SELL' com confiança [0..1]. Nunca fica mudo: se os filtros rígidos não passarem,
    devolve o que chegar mais perto do ideal.
    """
    close = df["close"]
    macd_line, signal_line, hist = macd(close)
    rsi14 = rsi(close, 14)
    tf = trend_flags(close)

    # Pontos por indicador
    score = 0.0
    reasons = []

    # Tendência de fundo
    if tf["bull"] > 0.5:
        score += 0.9
        reasons.append("Tendência: altista (EMA9>21>50>200 e EMA200 subindo)")
    elif tf["bear"] > 0.5:
        score -= 0.9
        reasons.append("Tendência: baixista (EMA9<21<50<200 e EMA200 caindo)")
    else:
        reasons.append("Tendência: neutra/mista")

    # MACD
    if macd_line.iloc[-1] > signal_line.iloc[-1] and hist.iloc[-1] > 0:
        score += 0.5
        reasons.append("MACD: cruzado para cima (line > signal)")
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and hist.iloc[-1] < 0:
        score -= 0.5
        reasons.append("MACD: cruzado para baixo (line < signal)")
    else:
        reasons.append("MACD: indefinido/achatado")

    # RSI zonas
    last_rsi = float(rsi14.iloc[-1])
    if last_rsi < 30:
        # sobrevendido → viés de compra (reversão leve)
        score += 0.3
        reasons.append(f"RSI: sobrevendido ({last_rsi:.1f})")
    elif last_rsi > 70:
        # sobrecomprado → viés de venda (reversão leve)
        score -= 0.3
        reasons.append(f"RSI: sobrecomprado ({last_rsi:.1f})")
    else:
        reasons.append(f"RSI: neutro ({last_rsi:.1f})")

    # Normaliza confiança
    conf = min(1.0, max(0.0, abs(score) / 1.7))  # 1.7 = soma dos pesos máximos absolutos

    direction = "BUY" if score >= 0 else "SELL"
    return {
        "direction": direction,
        "confidence": round(conf, 3),
        "rsi": round(last_rsi, 2),
        "macd_line": round(float(macd_line.iloc[-1]), 6),
        "macd_signal": round(float(signal_line.iloc[-1]), 6),
        "macd_hist": round(float(hist.iloc[-1]), 6),
        "trend": "bull" if tf["bull"] > 0.5 else ("bear" if tf["bear"] > 0.5 else "flat"),
        "ema9": round(tf["ema9"], 6),
        "ema21": round(tf["ema21"], 6),
        "ema50": round(tf["ema50"], 6),
        "ema200": round(tf["ema200"], 6),
        "reasons": reasons,
    }

def analyze_image_basic(file_stream: io.BytesIO) -> Dict[str, float]:
    """
    Análise simples do gráfico enviado:
    - Converte para grayscale, equaliza contraste e mede inclinação dominante via gradientes.
    - Devolve 'up/down/flat' e uma confiança aproximada.
    OBS.: É heurístico, não substitui os indicadores calculados nos preços reais.
    """
    img = Image.open(file_stream).convert("L")
    img = ImageOps.autocontrast(img)
    arr = np.array(img, dtype=np.float32)
    # Gradientes simples (Sobel-like)
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    # Direção média do gradiente
    angle = np.arctan2(gy, gx)  # [-pi, pi]
    # Usa apenas bordas fortes
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag, 85)
    mask = mag >= thresh
    if not np.any(mask):
        return {"trend_from_image": "flat", "image_confidence": 0.0, "note": "Poucas bordas fortes detectadas."}
    mean_angle = float(np.mean(angle[mask]))
    # Converte ângulo para rótulo
    # Ângulos próximos de 0 → horizontal/flat; positivos → up; negativos → down
    if mean_angle > 0.15:
        t = "up"
        conf = min(1.0, (mean_angle / (math.pi/2)))
    elif mean_angle < -0.15:
        t = "down"
        conf = min(1.0, (-mean_angle / (math.pi/2)))
    else:
        t = "flat"
        conf = 0.2
    return {"trend_from_image": t, "image_confidence": round(float(conf), 3), "note": "Heurística por gradiente."}

# =========================
# API
# =========================
@app.route("/")
def index():
    # UI minimalista mantendo ordem de seções pedida: rótulo > Night Mode > Moeda/Expiração > Classe > Entrada > Ação > métricas; Plano T+1
    return render_template("index.html", app_name=APP_NAME, default_symbols=",".join(DEFAULT_SYMBOLS), default_interval=DEFAULT_INTERVAL_MIN)

@app.route("/api/signal", methods=["GET"])
def api_signal():
    symbol = request.args.get("symbol", DEFAULT_SYMBOLS[0]).upper().replace("-", "/")
    interval = int(request.args.get("interval", DEFAULT_INTERVAL_MIN))
    try:
        df = fetch_ohlc_kraken(symbol, interval_min=interval, candles=CANDLES)
        if len(df) < 60:
            return jsonify({"ok": False, "error": "Dados insuficientes da Kraken."}), 400
        decision = smart_ai_decision(df)
        last_close = float(df["close"].iloc[-1])
        ts = df["time"].iloc[-1].isoformat()
        return jsonify({
            "ok": True,
            "symbol": symbol,
            "interval": interval,
            "last_close": last_close,
            "last_candle_time": ts,
            "decision": decision,
            "server_time": now_utc_iso()
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/signal_best", methods=["GET"])
def api_signal_best():
    symbols = request.args.get("symbols")
    interval = int(request.args.get("interval", DEFAULT_INTERVAL_MIN))
    if not symbols:
        symbols_list = DEFAULT_SYMBOLS
    else:
        symbols_list = [s.strip().upper().replace("-", "/") for s in symbols.split(",") if s.strip()]
    results = []
    for s in symbols_list:
        try:
            df = fetch_ohlc_kraken(s, interval_min=interval, candles=CANDLES)
            if len(df) < 60:
                continue
            d = smart_ai_decision(df)
            d["symbol"] = s
            d["last_close"] = float(df["close"].iloc[-1])
            d["time"] = df["time"].iloc[-1].isoformat()
            results.append(d)
        except Exception as e:
            results.append({"symbol": s, "error": str(e), "confidence": 0.0, "direction": "HOLD"})
    # escolhe melhor por maior confiança
    results_sorted = sorted(results, key=lambda x: x.get("confidence", 0.0), reverse=True)
    best = results_sorted[0] if results_sorted else {"error": "Sem resultados."}
    return jsonify({"ok": True, "best": best, "all": results_sorted, "server_time": now_utc_iso()})

@app.route("/api/upload-image", methods=["POST"])
def api_upload_image():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "Nenhum arquivo enviado (campo 'file')."}), 400
    f = request.files["file"]
    try:
        analysis = analyze_image_basic(f.stream)
        return jsonify({"ok": True, "image_analysis": analysis, "server_time": now_utc_iso()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"ok": True, "app": APP_NAME, "time": now_utc_iso()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
