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

client = OpenAI(api_key="SUA_API_KEY_AQUI")   # <-- COLOQUE SUA API KEY AQUI


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

Forneça como resposta, em português:

1. Tendência (forte, moderada ou fraca)
2. Suportes e resistências principais
3. Padrões técnicos identificados (triângulos, bandeiras, M, W, pullback etc.)
4. Análise do volume e da volatilidade
5. Direção provável do próximo movimento (alta ou baixa)
6. Sinal final (BUY, SELL ou HOLD)
7. Preço sugerido de entrada
8. Stop Loss ideal
9. Um ou dois Take Profits ideais
10. Risco x retorno aproximado
11. Confiabilidade do sinal (0% a 100%)
12. Explicação detalhada, mas objetiva, do raciocínio.
"""

        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content


# ==============================================================
#  SERVIÇO FLASK
# ==============================================================

def create_app():
    app = Flask(__name__)
    app.analyzer = OptimizedSuperIntelligentAnalyzer()
    return app


app = create_app()


# ==============================================================
#  HTML COMPLETO COM UM GRÁFICO + PAINEL + RELÓGIO
# ==============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8" />
<title>IA Signal Pro - REAL TIME 🧠⚡</title>

<script src="https://s3.tradingview.com/tv.js"></script>

<style>
    * { box-sizing: border-box; }

    body {
        background: #020617;
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        padding: 20px;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .title {
        font-size: 24px;
        font-weight: 800;
        color: #38bdf8;
    }

    .clock {
        font-size: 18px;
        font-weight: 600;
        color: #a5b4fc;
    }

    .layout {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 20px;
    }

    .box {
        background: #020617;
        border-radius: 14px;
        border: 1px solid #1e293b;
        padding: 15px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.7);
    }

    .box-title {
        font-size: 16px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 10px;
    }

    .controls label {
        font-size: 13px;
        color: #9ca3af;
        display: block;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    .select {
        width: 100%;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #1f2937;
        background: #020617;
        color: #e5e7eb;
        outline: none;
    }

    .btn {
        background: linear-gradient(90deg, #22c55e, #06b6d4);
        border: none;
        padding: 12px;
        font-size: 15px;
        margin-top: 16px;
        cursor: pointer;
        border-radius: 10px;
        width: 100%;
        font-weight: 700;
        color: #0b1120;
    }

    .btn:hover {
        opacity: 0.9;
    }

    pre {
        white-space: pre-wrap;
        background: #020617;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #1e293b;
        max-height: 500px;
        overflow-y: auto;
        font-size: 13px;
    }

    #tv-chart {
        height: 550px;
    }
</style>
</head>
<body>

<div class="topbar">
    <div class="title">📈 IA SIGNAL PRO – Análise em Tempo Real</div>
    <div class="clock" id="clock">--:--:--</div>
</div>

<div class="layout">
    <!-- Lado ESQUERDO: gráfico único -->
    <div class="box">
        <div class="box-title" id="chart-title">BTC/USDT – 1m</div>
        <div id="tv-chart"></div>
    </div>

    <!-- Lado DIREITO: controles + IA -->
    <div class="box">
        <div class="box-title">⚙️ Configurações da IA</div>

        <div class="controls">
            <label for="symbol">Ativo</label>
            <select id="symbol" class="select" onchange="updateChart()">
                <option value="BTCUSDT">BTC/USDT</option>
                <option value="ETHUSDT">ETH/USDT</option>
                <option value="BNBUSDT">BNB/USDT</option>
                <option value="ADAUSDT">ADA/USDT</option>
                <option value="XRPUSDT">XRP/USDT</option>
                <option value="SOLUSDT">SOL/USDT</option>
            </select>

            <label for="timeframe">Timeframe</label>
            <select id="timeframe" class="select" onchange="updateChart()">
                <option value="1m">1 minuto</option>
                <option value="5m">5 minutos</option>
            </select>

            <button class="btn" onclick="analisar()">
                🔍 Executar Análise da IA (200 candles)
            </button>
        </div>
    </div>
</div>

<div class="box" style="margin-top:20px;">
    <div class="box-title">📊 Resultado da IA</div>
    <pre id="resultadoIA">Clique em "Executar Análise da IA" para gerar o relatório…</pre>
</div>

<script>
let currentWidget = null;

// --------------------- RELÓGIO ----------------------
function updateClock() {
    const now = new Date();
    const formatted = now.toLocaleTimeString('pt-BR', { hour12: false });
    document.getElementById("clock").innerText = formatted;
}
setInterval(updateClock, 1000);
updateClock();

// -------------------- GRÁFICO TV --------------------
function loadChart(symbol, interval) {
    const containerId = "tv-chart";

    // limpa container (pra não empilhar iframes)
    document.getElementById(containerId).innerHTML = "";

    const tvSymbol = "BINANCE:" + symbol;

    new TradingView.widget({
        "width": "100%",
        "height": 550,
        "symbol": tvSymbol,
        "interval": interval,
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "br",
        "container_id": containerId
    });

    const titleEl = document.getElementById("chart-title");
    const tfLabel = interval === "1" ? "1m" : interval + "m";
    titleEl.innerText = symbol.replace("USDT", "/USDT") + " – " + tfLabel;
}

function updateChart() {
    const symbol = document.getElementById("symbol").value;
    const timeframe = document.getElementById("timeframe").value;

    // TradingView interval precisa ser "1", "5", etc.
    const interval = timeframe === "1m" ? "1" : "5";
    loadChart(symbol, interval);
}

// inicializa com BTC 1m
updateChart();

// ------------------- CHAMAR BACKEND IA ------------------

async function analisar() {
    const symbol = document.getElementById("symbol").value;
    const timeframe = document.getElementById("timeframe").value;

    document.getElementById("resultadoIA").innerText =
        "⏳ Buscando últimos 200 candles de " + symbol + " em " + timeframe + " e analisando com a IA…";

    try {
        const resp = await fetch("/analisar_ativo", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ symbol, timeframe })
        });

        const data = await resp.json();

        if (data.error) {
            document.getElementById("resultadoIA").innerText = "Erro: " + data.error;
        } else {
            document.getElementById("resultadoIA").innerText = data.resultado;
        }
    } catch (e) {
        document.getElementById("resultadoIA").innerText =
            "Erro ao chamar a IA: " + e;
    }
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
    try:
        data = request.json
        symbol = data["symbol"]
        timeframe = data["timeframe"]

        # Mapear timeframe para o formato da Binance
        tf = timeframe  # já vem como "1m" ou "5m"

        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit=200"
        candles_raw = requests.get(url, timeout=10).json()

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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================================================
#  START SERVER (RAILWAY)
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
