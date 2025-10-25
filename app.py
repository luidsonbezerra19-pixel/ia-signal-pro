from __future__ import annotations

"""
IA Signal Pro — Análise Inteligente por Foto
Versão otimizada para produção no Railway
"""

import base64
import io
import os
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
            raise RuntimeError("OpenCV não instalado")

    def _read(self, blob: bytes) -> "np.ndarray":
        self._deps()
        try:
            arr = np.frombuffer(blob, np.uint8)
            img = cv.imdecode(arr, cv.IMREAD_COLOR)
            if img is not None:
                return img
        except:
            pass
        
        # Fallback para PIL
        pil = Image.open(io.BytesIO(blob)).convert("RGB")
        return cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)

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
        try:
            gy = cv.Sobel(gray, cv.CV_32F, 0, 1, 3)
            gpos = float(np.clip(gy[gy>0].mean() if (gy>0).any() else 0.0, 0, 255))
            gneg = float(np.clip((-gy[gy<0]).mean() if (gy<0).any() else 0.0, 0, 255))
            base = 100.0 * (gpos / (gpos + gneg + 1e-9)) if (gpos+gneg)>0 else 50.0
            return float(max(0.0, min(100.0, base)))
        except:
            return 50.0

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
        try:
            img = self._prep(self._read(blob))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            edges = self._edges(gray)
            slope, vol = self._trend_vol(edges)
            
            h, w = gray.shape[:2]
            row = gray[int(h*0.5), :].astype(np.float32) + 1.0
            macd = self._macd_proxy(row)

            # Análise Bollinger
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
        except Exception as e:
            return {
                "direction": "neutral",
                "final_confidence": 0.5,
                "entry_signal": "Erro na análise",
                "metrics": {
                    "rsi": 50.0,
                    "macd": "neutral",
                    "bollinger": "neutral",
                    "trend_strength": 0.0,
                    "volatility": 0.0
                }
            }


# ===============
#  APLICAÇÃO FLASK
# ===============
app = Flask(__name__)
ANALYZER = SuperIntelligentAnalyzer()

# Template HTML simplificado e otimizado
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Signal Pro - Análise Inteligente</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #0b1220 0%, #1a1f38 100%);
            color: #e9eef2;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 500px;
            margin: 0 auto;
            background: rgba(15, 22, 39, 0.95);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid #2a3552;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        
        .title {
            text-align: center;
            font-size: 28px;
            font-weight: 800;
            margin-bottom: 25px;
            background: linear-gradient(90deg, #3a86ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .upload-area {
            border: 3px dashed #3a86ff;
            border-radius: 15px;
            padding: 40px 20px;
            text-align: center;
            background: rgba(14, 21, 36, 0.6);
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #00ff88;
            background: rgba(14, 21, 36, 0.8);
        }
        
        .file-input {
            margin: 20px 0;
            padding: 10px;
            background: rgba(42, 53, 82, 0.3);
            border: 1px solid #3a86ff;
            border-radius: 8px;
            color: white;
            width: 100%;
        }
        
        .analyze-btn {
            background: linear-gradient(135deg, #3a86ff 0%, #2a76ef 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 16px 32px;
            font-size: 18px;
            font-weight: 800;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .analyze-btn:hover {
            background: linear-gradient(135deg, #2a76ef 0%, #1a66df 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(58, 134, 255, 0.4);
        }
        
        .analyze-btn:disabled {
            background: #2a3552;
            transform: none;
            box-shadow: none;
            cursor: not-allowed;
        }
        
        .result {
            background: rgba(14, 21, 36, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin-top: 25px;
            border: 1px solid #223152;
            display: none;
        }
        
        .signal-buy {
            color: #00ff88;
            font-weight: 800;
            font-size: 24px;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .signal-sell {
            color: #ff4444;
            font-weight: 800;
            font-size: 24px;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .signal-neutral {
            color: #ffaa00;
            font-weight: 800;
            font-size: 24px;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .confidence {
            font-size: 18px;
            text-align: center;
            margin: 15px 0;
            color: #9db0d1;
        }
        
        .metrics {
            margin-top: 20px;
            font-size: 14px;
            color: #9db0d1;
            background: rgba(42, 53, 82, 0.3);
            padding: 15px;
            border-radius: 10px;
        }
        
        .metric-item {
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
        }
        
        .loading {
            text-align: center;
            color: #3a86ff;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="title">📈 IA SIGNAL PRO</div>
        
        <div class="upload-area">
            <div style="font-size: 20px; margin-bottom: 10px;">📷 COLE O PRINT DO GRÁFICO</div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
            <button class="analyze-btn" id="analyzeBtn">🎯 ANALISAR E OBTER SINAL</button>
        </div>
        
        <div class="result" id="result">
            <div id="signalText" class="signal-neutral"></div>
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
            if (selectedFile) {
                analyzeBtn.style.background = 'linear-gradient(135deg, #00ff88 0%, #00cc66 100%)';
            }
        });

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('📸 Selecione uma imagem do gráfico primeiro!');
                return;
            }

            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '🔄 ANALISANDO...';
            result.style.display = 'block';
            signalText.className = 'signal-neutral';
            signalText.textContent = '🔍 Analisando gráfico...';
            confidenceText.textContent = '';
            metricsText.innerHTML = '<div class="loading">Processando imagem e calculando indicadores...</div>';

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
                    let metricsHtml = '<div style="margin-bottom: 10px; font-weight: bold;">📊 Análise Técnica:</div>';
                    metricsHtml += `<div class="metric-item"><span>RSI:</span> <span>${metrics.rsi?.toFixed(1) || 'N/A'}</span></div>`;
                    metricsHtml += `<div class="metric-item"><span>MACD:</span> <span>${metrics.macd || 'N/A'}</span></div>`;
                    metricsHtml += `<div class="metric-item"><span>Bollinger:</span> <span>${metrics.bollinger || 'N/A'}</span></div>`;
                    metricsHtml += `<div class="metric-item"><span>Força da Tendência:</span> <span>${metrics.trend_strength?.toFixed(1) || 'N/A'}</span></div>`;
                    metricsHtml += `<div class="metric-item"><span>Volatilidade:</span> <span>${metrics.volatility?.toFixed(3) || 'N/A'}</span></div>`;
                    
                    metricsText.innerHTML = metricsHtml;
                } else {
                    signalText.className = 'signal-neutral';
                    signalText.textContent = '❌ Erro na análise';
                    confidenceText.textContent = data.error || 'Tente novamente com outra imagem';
                }
            } catch (error) {
                signalText.className = 'signal-neutral';
                signalText.textContent = '❌ Erro de conexão';
                confidenceText.textContent = 'Verifique sua internet e tente novamente';
            }
            
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🎯 ANALISAR E OBTER SINAL';
            analyzeBtn.style.background = 'linear-gradient(135deg, #3a86ff 0%, #2a76ef 100%)';
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_photo():
    try:
        if not request.files or 'image' not in request.files:
            return jsonify({'ok': False, 'error': 'Nenhuma imagem enviada'}), 400
        
        image_file = request.files['image']
        if not image_file or image_file.filename == '':
            return jsonify({'ok': False, 'error': 'Arquivo inválido'}), 400
        
        # Verificar tamanho do arquivo (máximo 10MB)
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'ok': False, 'error': 'Imagem muito grande (máximo 10MB)'}), 400
        
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return jsonify({'ok': False, 'error': 'Arquivo vazio'}), 400
        
        analysis = ANALYZER.analyze(image_bytes)
        
        return jsonify({
            'ok': True,
            'direction': analysis['direction'],
            'final_confidence': analysis['final_confidence'],
            'entry_signal': analysis['entry_signal'],
            'metrics': analysis['metrics']
        })
        
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Erro interno: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'IA Signal Pro está funcionando!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
