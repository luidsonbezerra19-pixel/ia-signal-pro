from __future__ import annotations

"""
IA Signal Pro — Análise Inteligente por Foto
--------------------------------------------
Versão super assertiva para análise de gráficos via foto
Retorna entrada direta para o próximo candle
"""

import base64
import io
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

try:
    import cv2 as cv
except Exception:
    cv = None


# =========================
#  IA SUPER INTELIGENTE
# =========================
class SuperIntelligentAnalyzer:
    def _deps(self) -> None:
        if cv is None:
            raise RuntimeError("Instale: pillow numpy opencv-python-headless")

    def _read(self, blob: bytes) -> "np.ndarray":
        self._deps()
        arr = np.frombuffer(blob, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            pil = Image.open(io.BytesIO(blob)).convert("RGB")
            img = cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)
        return img

    def _prep(self, img: "np.ndarray") -> "np.ndarray":
        h, w = img.shape[:2]
        c = 0.05
        return img[int(h*c):int(h*(1-c)), int(w*c):int(w*(1-c))]

    def _edges(self, gray: "np.ndarray") -> "np.ndarray":
        edges = cv.Canny(gray, 60, 150)
        k = np.ones((3,3), np.uint8)
        edges = cv.dilate(edges, k, 1)
        edges = cv.erode(edges, k, 1)
        return edges

    def _trend_vol(self, edges: "np.ndarray") -> tuple[float, float]:
        ys, xs = np.nonzero(edges)
        if len(xs) < 100: 
            return 0.0, 0.01
        
        X = np.vstack([xs, np.ones_like(xs)]).T
        m, b = np.linalg.lstsq(X, ys, rcond=None)[0]
        slope = -m / max(1.0, edges.shape[1]*0.75)
        vol = float(np.std(ys - (m*xs + b)) / max(1.0, edges.shape[0]))
        return float(slope), float(vol)

    def _rsi_proxy(self, gray: "np.ndarray") -> float:
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, 3)
        gpos = float(np.clip(gy[gy>0].mean() if (gy>0).any() else 0.0, 0, 255))
        gneg = float(np.clip((-gy[gy<0]).mean() if (gy<0).any() else 0.0, 0, 255))
        base = 100.0 * (gpos / (gpos + gneg + 1e-9)) if (gpos+gneg)>0 else 50.0
        return float(max(0.0, min(100.0, base)))

    def _macd_proxy(self, s: "np.ndarray") -> dict:
        if len(s) < 50: 
            return {"signal":"neutral"}
        
        def ema(x, n):
            k = 2/(n+1)
            e = [float(x[0])]
            for v in x[1:]: 
                e.append(e[-1] + k*(float(v)-e[-1]))
            return np.array(e, dtype=float)
        
        e12 = ema(s, 12)
        e26 = ema(s, 26)
        macd = e12[-len(e26):] - e26
        sig = ema(macd, 9)
        hist = float(macd[-1] - sig[-1])
        
        return {"signal": "bullish" if hist > 0 else ("bearish" if hist < 0 else "neutral")}

    def analyze(self, blob: bytes) -> Dict[str, Any]:
        img = self._prep(self._read(blob))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = self._edges(gray)
        slope, vol = self._trend_vol(edges)
        
        h, w = gray.shape[:2]
        row = gray[int(h*0.5), :].astype(np.float32) + 1.0
        macd = self._macd_proxy(row)

        # Análise Bollinger (proxy)
        p = min(20, len(row))
        if p >= 12:
            win = row[-p:]
            ma = float(win.mean())
            sd = float(win.std())
            last = float(win[-1])
            
            if last > ma + 2*sd: 
                boll = "overbought"
            elif last < ma - 2*sd: 
                boll = "oversold"
            elif last > ma: 
                boll = "bullish"
            elif last < ma: 
                boll = "bearish"
            else: 
                boll = "neutral"
        else:
            boll = "neutral"

        # DECISÃO SUPER INTELIGENTE
        rsi = self._rsi_proxy(gray)
        
        # Fatores de decisão
        trend_factor = 1.0 if slope > 0.002 else (-1.0 if slope < -0.002 else 0)
        macd_factor = 1.0 if macd["signal"] == "bullish" else (-1.0 if macd["signal"] == "bearish" else 0)
        rsi_factor = 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0)
        boll_factor = 1.0 if boll == "oversold" else (-1.0 if boll == "overbought" else 0)
        
        # Score final
        total_score = trend_factor + macd_factor + rsi_factor + boll_factor
        
        # Decisão assertiva
        if total_score > 0:
            direction = "buy"
            confidence = min(0.95, 0.6 + (total_score * 0.1))
        elif total_score < 0:
            direction = "sell" 
            confidence = min(0.95, 0.6 + (abs(total_score) * 0.1))
        else:
            direction = "neutral"
            confidence = 0.5

        return {
            "direction": direction,
            "final_confidence": float(confidence),
            "entry_signal": f"{direction.upper()} para próximo candle",
            "metrics": {
                "rsi": rsi,
                "macd": macd["signal"],
                "bollinger": boll,
                "trend_strength": float(abs(slope) * 1000),
                "volatility": float(vol)
            }
        }


# ===============
#  APLICAÇÃO FLASK
# ===============
app = Flask(__name__)
ANALYZER = SuperIntelligentAnalyzer()

PAGE = r"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>IA Signal Pro — Análise Inteligente</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {
    background: #0b1220;
    color: #e9eef2;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif;
    margin: 0;
    padding: 20px;
  }
  
  .container {
    max-width: 600px;
    margin: 0 auto;
    background: #0f1627;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #2a3552;
  }
  
  .title {
    font-size: 24px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 20px;
    color: #3a86ff;
  }
  
  .upload-area {
    border: 2px dashed #3a86ff;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    background: #0e1524;
    margin-bottom: 20px;
  }
  
  .file-input {
    margin: 15px 0;
  }
  
  .analyze-btn {
    background: #3a86ff;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 800;
    cursor: pointer;
    width: 100%;
    transition: background 0.3s;
  }
  
  .analyze-btn:hover {
    background: #2a76ef;
  }
  
  .analyze-btn:disabled {
    background: #2a3552;
    cursor: not-allowed;
  }
  
  .result {
    background: #0e1524;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    border: 1px solid #223152;
    display: none;
  }
  
  .signal-buy {
    color: #00ff88;
    font-weight: 800;
    font-size: 20px;
  }
  
  .signal-sell {
    color: #ff4444;
    font-weight: 800;
    font-size: 20px;
  }
  
  .signal-neutral {
    color: #ffaa00;
    font-weight: 800;
    font-size: 20px;
  }
  
  .confidence {
    font-size: 16px;
    margin: 10px 0;
  }
  
  .metrics {
    margin-top: 15px;
    font-size: 14px;
    color: #9db0d1;
  }
</style>
</head>
<body>
  <div class="container">
    <div class="title">📈 IA SIGNAL PRO - ANÁLISE INTELIGENTE</div>
    
    <div class="upload-area">
      <div>📷 COLE O PRINT DO GRÁFICO AQUI</div>
      <input type="file" id="fileInput" class="file-input" accept="image/*">
      <button class="analyze-btn" id="analyzeBtn">🎯 ANALISAR E OBTER SINAL</button>
    </div>
    
    <div class="result" id="result">
      <div id="signalText"></div>
      <div class="confidence" id="confidenceText"></div>
      <div class="metrics" id="metricsText"></div>
    </div>
  </div>

<script>
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const result = document.getElementById('result');
const signalText = document.getElementById('signalText');
const confidenceText = document.getElementById('confidenceText');
const metricsText = document.getElementById('metricsText');

let selectedFile = null;

fileInput.addEventListener('change', (e) => {
  selectedFile = e.target.files[0] || null;
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('Selecione uma imagem do gráfico primeiro!');
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'ANALISANDO...';
  result.style.display = 'block';
  signalText.textContent = '🔍 Analisando gráfico...';
  confidenceText.textContent = '';
  metricsText.textContent = '';

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.ok) {
      const direction = data.direction;
      const confidence = (data.final_confidence * 100).toFixed(1);
      
      if (direction === 'buy') {
        signalText.className = 'signal-buy';
        signalText.textContent = '🎯 COMPRAR - Entrada para próximo candle';
      } else if (direction === 'sell') {
        signalText.className = 'signal-sell';
        signalText.textContent = '🎯 VENDER - Entrada para próximo candle';
      } else {
        signalText.className = 'signal-neutral';
        signalText.textContent = '⚡ AGUARDAR - Sem sinal claro';
      }
      
      confidenceText.textContent = `Confiança: ${confidence}%`;
      
      // Métricas
      const metrics = data.metrics || {};
      let metricsHtml = '<strong>Análise Técnica:</strong><br>';
      metricsHtml += `RSI: ${metrics.rsi?.toFixed(1) || 'N/A'}<br>`;
      metricsHtml += `MACD: ${metrics.macd || 'N/A'}<br>`;
      metricsHtml += `Bollinger: ${metrics.bollinger || 'N/A'}<br>`;
      metricsHtml += `Força da Tendência: ${metrics.trend_strength?.toFixed(1) || 'N/A'}`;
      
      metricsText.innerHTML = metricsHtml;
    } else {
      signalText.className = 'signal-neutral';
      signalText.textContent = '❌ Erro na análise';
      confidenceText.textContent = data.error || 'Tente novamente';
    }
  } catch (error) {
    signalText.className = 'signal-neutral';
    signalText.textContent = '❌ Erro de conexão';
    confidenceText.textContent = 'Verifique sua internet e tente novamente';
  }
  
  analyzeBtn.disabled = false;
  analyzeBtn.textContent = '🎯 ANALISAR E OBTER SINAL';
});
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.post("/analyze")
def analyze_photo():
    if not request.files or "image" not in request.files:
        return jsonify({"ok": False, "error": "Nenhuma imagem enviada"}), 400
    
    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"ok": False, "error": "Arquivo inválido"}), 400
    
    try:
        image_bytes = image_file.read()
        analysis = ANALYZER.analyze(image_bytes)
        
        return jsonify({
            "ok": True,
            "direction": analysis["direction"],
            "final_confidence": analysis["final_confidence"],
            "entry_signal": analysis["entry_signal"],
            "metrics": analysis["metrics"]
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": f"Erro na análise: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
