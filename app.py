# ==============================================================
#  IA SIGNAL PRO - ANÁLISE DE 200 CANDLES EM TEMPO REAL
#  Flask + TradingView + Binance + GPT-5.1
# ==============================================================

import os
import json
import datetime
import requests
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

# ==============================================================
#  CONFIGURAÇÃO OPENAI GPT-5.1
# ==============================================================

client = OpenAI(api_key="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")   # <--- COLOQUE A SUA KEY AQUI


# ==============================================================
#  ANALISADOR SUPER INTELIGENTE COM GPT-5.1
# ==============================================================

class OptimizedSuperIntelligentAnalyzer:

    def analyze_raw_data(self, symbol, timeframe, candles):
        """
        Análise completa usando os últimos 200 candles do ativo.
        """

        prompt = f"""
Você é uma IA profissional de trading. 
Analise os últimos 200 candles do ativo {symbol} em {timeframe}.

Dados OHLCV:
{json.dumps(candles, indent=2)}

Forneça como resposta:

1. Tendência (forte, moderada ou fraca)
2. Suportes e resistências
3. Padrões técnicos (triângulo, bandeira, M, W, pullback etc)
4. Análise do volume e volatilidade
5. Direção provável do próximo movimento
6. Sinal final (BUY, SELL ou HOLD)
7. Preço sugerido de entrada
8. Stop Loss ideal
9. Take Profit ideal
10. Explicação detalhada e clara
11. Confiabilidade do sinal (0% a 100%)
"""

        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message["content"]


# ==============================================================
#  SERVIÇO FLASK
# ==============================================================

def create_app():
    app = Flask(__name__)
    app.analyzer = OptimizedSuperIntelligentAnalyzer()
    return app


app = create_app()


# ==============================================================
#  HTML COMPLETO COM GRÁFICOS + PAINEL + ANÁLISE
# ==============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8" />
<title>IA Signal Pro - REAL TIME 🧠⚡</title>

<script src="https://s3.tradingview.com/tv.js"></script>

<style>
    body {
        background: #0b1220;
        color: #e9eef2;
        font-family: Arial, sans-serif;
        padding: 20px;
    }
    .grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
    }
    .box {
        background: #111827;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #00eaff55;
    }
    .title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #00eaff;
    }
    .btn {
        background: #00ffaa;
        border: none;
        padding: 12px;
        font-size: 16px;
        margin-top: 10px;
        cursor: pointer;
        border-radius: 10px;
        width: 100%;
        font-weight: bold;
    }
    pre {
        white-space: pre-wrap;
        background: #0e172a;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #00eaff66;
    }
</style>
</head>
<body>

<h1 style="text-align:center;">📈 IA SIGNAL PRO – Análise em Tempo Real 🧠⚡</h1>

<!-- ======================== GRÁFICOS TRADINGVIEW =========================== -->

<div class="grid">
    <div class="box"><div class="title">BTC/USDT</div><div id="tv-btc" style="height:300px;"></div></div>
    <div class="box"><div class="title">ETH/USDT</div><div id="tv-eth" style="height:300px;"></div></div>
    <div class="box"><div class="title">BNB/USDT</div><div id="tv-bnb" style="height:300px;"></div></div>
    <div class="box"><div class="title">ADA/USDT</div><div id="tv-ada" style="height:300px;"></div></div>
    <div class="box"><div class="title">XRP/USDT</div><div id="tv-xrp" style="height:300px;"></div></div>
    <div class="box"><div class="title">SOL/USDT</div><div id="tv-sol" style="height:300px;"></div></div>
</div>

<script>
function loadChart(container, symbol) {
    new TradingView.widget({
        "width": "100%",
        "height": 300,
        "symbol": symbol,
        "interval": "1",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "br",
        "container_id": container
    });
}

loadChart("tv-btc", "BINANCE:BTCUSDT");
loadChart("tv-eth", "BINANCE:ETHUSDT");
loadChart("tv-bnb", "BINANCE:BNBUSDT");
loadChart("tv-ada", "BINANCE:ADAUSDT");
loadChart("tv-xrp", "BINANCE:XRPUSDT");
loadChart("tv-sol", "BINANCE:SOLUSDT");
</script>

<!-- ====================== PAINEL DE ANÁLISE =============================== -->

<div class="box" style="margin-top:30px;">
    <div class="title">⚙️ Configurações da IA</div>

    <label>Ativo:</label>
    <select id="symbol" style="width:100%; padding:10px;">
        <option value="BTCUSDT">BTC/USDT</option>
        <option value="ETHUSDT">ETH/USDT</option>
        <option value="BNBUSDT">BNB/USDT</option>
        <option value="ADAUSDT">ADA/USDT</option>
        <option value="XRPUSDT">XRP/USDT</option>
        <option value="SOLUSDT">SOL/USDT</option>
    </select>

    <label>Timeframe:</label>
    <select id="timeframe" style="width:100%; padding:10px;">
        <option value="1m">1 minuto</option>
        <option value="5m">5 minutos</option>
    </select>

    <button onclick="analisar()" class="btn">🔍 Executar Análise da IA</button>
</div>

<div class="box" style="margin-top:20px;">
    <div class="title">📊 Resultado da IA</div>
    <pre id="resultadoIA">Clique para gerar análise…</pre>
</div>


<script>
async function analisar() {

    const symbol = document.getElementById("symbol").value;
    const timeframe = document.getElementById("timeframe").value;

    document.getElementById("resultadoIA").innerText = "⏳ Analisando os últimos 200 candles...";

    const response = await fetch("/analisar_ativo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, timeframe })
    });

    const data = await response.json();
    document.getElementById("resultadoIA").innerText = data.resultado;
}
</script>

</body>
</html>
"""


# ==============================================================
#  BACKEND — ROTA PARA ANÁLISE (200 CANDLES)
# ==============================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/analisar_ativo", methods=["POST"])
def analisar_ativo():
    data = request.json
    symbol = data["symbol"]
    timeframe = data["timeframe"]

    # Buscar últimos 200 candles da Binance
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit=200"
    candles_raw = requests.get(url).json()

    candles = []
    for c in candles_raw:
        candles.append({
            "open_time": c[0],
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "close_time": c[6]
        })

    resultado = app.analyzer.analyze_raw_data(symbol, timeframe, candles)

    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe,
        "resultado": resultado
    })


# ==============================================================
#  START SERVER (RAILWAY)
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
