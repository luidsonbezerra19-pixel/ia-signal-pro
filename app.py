from __future__ import annotations

"""
IA Signal Pro — Análise Inteligente por Foto
Versão SUPER ASSERTIVA para detectar padrões
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
#  IA SUPER ASSERTIVA
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
        
        pil = Image.open(io.BytesIO(blob)).convert("RGB")
        return cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)

    def _prep(self, img: "np.ndarray") -> "np.ndarray":
        h, w = img.shape[:2]
        # Recorta apenas a área central (remove bordas)
        c = 0.08
        return img[int(h*c):int(h*(1-c)), int(w*c):int(w*(1-c))]

    def _enhance_contrast(self, gray: "np.ndarray") -> "np.ndarray":
        # Melhora o contraste para detectar melhor os padrões
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(gray)

    def _detect_candles(self, img: "np.ndarray") -> Dict[str, float]:
        """Detecta padrões de candle mais agressivamente"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        enhanced = self._enhance_contrast(gray)
        
        # Detecta bordas com parâmetros mais sensíveis
        edges = cv.Canny(enhanced, 30, 100)
        
        # Encontra linhas com HoughLines mais sensível
        lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                              minLineLength=20, maxLineGap=10)
        
        bullish_count = 0
        bearish_count = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Linhas ascendentes (bullish)
                if -80 < angle < -10:
                    bullish_count += 1
                # Linhas descendentes (bearish)  
                elif 10 < angle < 80:
                    bearish_count += 1
        
        total_lines = bullish_count + bearish_count + 1e-9
        bullish_ratio = bullish_count / total_lines
        bearish_ratio = bearish_count / total_lines
        
        return {
            "bullish_ratio": bullish_ratio,
            "bearish_ratio": bearish_ratio,
            "total_lines": total_lines
        }

    def _analyze_trend_strength(self, img: "np.ndarray") -> Dict[str, float]:
        """Análise de tendência mais agressiva"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Analisa múltiplas linhas horizontais
        trend_scores = []
        for i in range(5):
            y_pos = int(h * (0.2 + i * 0.15))
            row = gray[y_pos, :].astype(np.float32)
            
            # Remove ruído
            row = cv.GaussianBlur(row, (5, 5), 0)
            
            # Calcula tendência
            if len(row) > 10:
                x = np.arange(len(row))
                slope, intercept = np.polyfit(x, row, 1)
                trend_scores.append(slope)
        
        avg_slope = np.mean(trend_scores) if trend_scores else 0
        trend_strength = abs(avg_slope) * 1000
        
        return {
            "trend_direction": -1 if avg_slope < 0 else 1,
            "trend_strength": float(trend_strength),
            "avg_slope": float(avg_slope)
        }

    def _calculate_rsi_aggressive(self, gray: "np.ndarray") -> float:
        """RSI mais sensível"""
        try:
            # Analisa gradientes verticais de forma mais agressiva
            gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
            
            gains = gy[gy > 2].mean() if np.any(gy > 2) else 0
            losses = -gy[gy < -2].mean() if np.any(gy < -2) else 0
            
            if gains + losses > 0:
                rsi = 100 * gains / (gains + losses)
            else:
                rsi = 50.0
                
            return float(np.clip(rsi, 0, 100))
        except:
            return 50.0

    def _detect_support_resistance(self, gray: "np.ndarray") -> Dict[str, float]:
        """Detecta níveis de suporte e resistência"""
        edges = cv.Canny(gray, 50, 150)
        
        # Encontra linhas horizontais (suporte/resistência)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=25,
                              minLineLength=30, maxLineGap=5)
        
        horizontal_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if -10 < angle < 10:  # Linha horizontal
                    horizontal_count += 1
        
        return {
            "support_resistance_levels": float(horizontal_count),
            "market_structure": min(1.0, horizontal_count / 10.0)
        }

    def analyze(self, blob: bytes) -> Dict[str, Any]:
        try:
            img = self._read(blob)
            img_prep = self._prep(img)
            gray = cv.cvtColor(img_prep, cv.COLOR_BGR2GRAY)
            
            # Análises múltiplas
            candle_patterns = self._detect_candles(img_prep)
            trend_analysis = self._analyze_trend_strength(img_prep)
            support_resistance = self._detect_support_resistance(gray)
            
            rsi = self._calculate_rsi_aggressive(gray)
            
            # ========== DECISÃO SUPER ASSERTIVA ==========
            
            # Fatores com pesos mais agressivos
            factors = []
            
            # 1. Tendência (peso alto)
            trend_score = trend_analysis["trend_direction"] * min(2.0, trend_analysis["trend_strength"] / 50.0)
            factors.append(("trend", trend_score))
            
            # 2. Padrões de candle (peso alto)
            candle_bias = (candle_patterns["bullish_ratio"] - candle_patterns["bearish_ratio"]) * 3.0
            factors.append(("candles", candle_bias))
            
            # 3. RSI (peso médio)
            rsi_bias = 0
            if rsi < 35:  # Oversold
                rsi_bias = 1.5
            elif rsi > 65:  # Overbought
                rsi_bias = -1.5
            elif rsi > 55:  # Levemente sobrecomprado
                rsi_bias = -0.5
            elif rsi < 45:  # Levemente sobrevendido
                rsi_bias = 0.5
            factors.append(("rsi", rsi_bias))
            
            # 4. Estrutura de mercado (peso baixo)
            structure_bias = support_resistance["market_structure"] * trend_analysis["trend_direction"]
            factors.append(("structure", structure_bias))
            
            # Calcula score total
            total_score = sum(score for _, score in factors)
            
            # Decisão assertiva
            if total_score > 1.0:
                direction = "buy"
                confidence = min(0.95, 0.7 + (total_score * 0.08))
                reasoning = "Tendência de alta detectada com força"
            elif total_score < -1.0:
                direction = "sell"
                confidence = min(0.95, 0.7 + (abs(total_score) * 0.08))
                reasoning = "Tendência de baixa detectada com força"
            else:
                direction = "neutral"
                confidence = 0.5
                reasoning = "Mercado em equilíbrio - aguardar confirmação"
            
            # Métricas detalhadas
            metrics = {
                "rsi": rsi,
                "trend_strength": trend_analysis["trend_strength"],
                "bullish_candles_ratio": candle_patterns["bullish_ratio"],
                "bearish_candles_ratio": candle_patterns["bearish_ratio"],
                "support_resistance_levels": support_resistance["support_resistance_levels"],
                "market_structure_score": support_resistance["market_structure"],
                "analysis_score": float(total_score)
            }

            return {
                "direction": direction,
                "final_confidence": float(confidence),
                "entry_signal": f"{direction.upper()} - {reasoning}",
                "metrics": metrics,
                "reasoning": reasoning
            }
            
        except Exception as e:
            # Fallback em caso de erro
            return {
                "direction": "neutral",
                "final_confidence": 0.5,
                "entry_signal": "AGUARDAR - Análise em ajuste",
                "metrics": {
                    "rsi": 50.0,
                    "trend_strength": 0.0,
                    "bullish_candles_ratio": 0.5,
                    "bearish_candles_ratio": 0.5,
                    "support_resistance_levels": 0.0,
                    "market_structure_score": 0.0,
                    "analysis_score": 0.0
                },
                "reasoning": f"Análise em processo de ajuste: {str(e)}"
            }


# ===============
#  APLICAÇÃO FLASK
# ===============
app = Flask(__name__)
ANALYZER = SuperIntelligentAnalyzer()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Signal Pro - Análise Super Assertiva</title>
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
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #3a86ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            text-align: center;
            color: #9db0d1;
            margin-bottom: 25px;
            font-size: 14px;
        }
        
        .upload-area {
            border: 3px dashed #3a86ff;
            border-radius: 15px;
            padding: 30px 20px;
            text-align: center;
            background: rgba(14, 21, 36, 0.6);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .upload-area.drag-over {
            border-color: #00ff88;
            background: rgba(14, 21, 36, 0.8);
        }
        
        .file-input {
            margin: 15px 0;
            padding: 10px;
            background: rgba(42, 53, 82, 0.3);
            border: 1px solid #3a86ff;
            border-radius: 8px;
            color: white;
            width: 100%;
            cursor: pointer;
        }
        
        .analyze-btn {
            background: linear-gradient(135deg, #3a86ff 0%, #2a76ef 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 16px;
            font-size: 16px;
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
            background: rgba(14, 21, 36, 0.9);
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            border: 1px solid #223152;
            display: none;
        }
        
        .signal-buy {
            color: #00ff88;
            font-weight: 800;
            font-size: 22px;
            text-align: center;
            margin-bottom: 15px;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
        }
        
        .signal-sell {
            color: #ff4444;
            font-weight: 800;
            font-size: 22px;
            text-align: center;
            margin-bottom: 15px;
            text-shadow: 0 0 10px rgba(255, 68, 68, 0.3);
        }
        
        .signal-neutral {
            color: #ffaa00;
            font-weight: 800;
            font-size: 22px;
            text-align: center;
            margin-bottom: 15px;
            text-shadow: 0 0 10px rgba(255, 170, 0, 0.3);
        }
        
        .confidence {
            font-size: 16px;
            text-align: center;
            margin: 15px 0;
            color: #9db0d1;
        }
        
        .reasoning {
            text-align: center;
            margin: 10px 0;
            color: #9db0d1;
            font-style: italic;
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
            align-items: center;
        }
        
        .metric-value {
            font-weight: 600;
            color: #e9eef2;
        }
        
        .loading {
            text-align: center;
            color: #3a86ff;
            font-size: 16px;
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background: #2a3552;
            border-radius: 2px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3a86ff, #00ff88);
            width: 0%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="title">🎯 IA SIGNAL PRO</div>
        <div class="subtitle">Análise Super Assertiva - Padrões de Gráfico</div>
        
        <div class="upload-area" id="uploadArea">
            <div style="font-size: 18px; margin-bottom: 10px;">📸 ARRASTE OU CLIQUE PARA ENVIAR O GRÁFICO</div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
            <button class="analyze-btn" id="analyzeBtn">🚀 ANALISAR E OBTER SINAL</button>
        </div>
        
        <div class="result" id="result">
            <div id="signalText" class="signal-neutral"></div>
            <div class="reasoning" id="reasoningText"></div>
            <div class="confidence" id="confidenceText"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="metrics" id="metricsText"></div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const uploadArea = document.getElementById('uploadArea');
        const result = document.getElementById('result');
        const signalText = document.getElementById('signalText');
        const reasoningText = document.getElementById('reasoningText');
        const confidenceText = document.getElementById('confidenceText');
        const progressFill = document.getElementById('progressFill');
        const metricsText = document.getElementById('metricsText');

        let selectedFile = null;

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                selectedFile = e.dataTransfer.files[0];
                updateButtonState();
            }
        });

        fileInput.addEventListener('change', (e) => {
            selectedFile = e.target.files[0] || null;
            updateButtonState();
        });

        function updateButtonState() {
            if (selectedFile) {
                analyzeBtn.style.background = 'linear-gradient(135deg, #00ff88 0%, #00cc66 100%)';
                analyzeBtn.textContent = '🚀 PRONTO PARA ANALISAR!';
            }
        }

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('📸 Selecione uma imagem do gráfico primeiro!');
                return;
            }

            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '🔄 ANALISANDO PADRÕES...';
            result.style.display = 'block';
            signalText.className = 'signal-neutral';
            signalText.textContent = '🔍 Analisando padrões do gráfico...';
            reasoningText.textContent = 'Detectando tendências e formações...';
            confidenceText.textContent = '';
            progressFill.style.width = '30%';
            
            metricsText.innerHTML = `
                <div class="loading">
                    <div>📊 Calculando indicadores...</div>
                    <div style="font-size: 12px; margin-top: 5px;">IA Super Assertiva em ação</div>
                </div>
            `;

            try {
                const formData = new FormData();
                formData.append('image', selectedFile);
                
                progressFill.style.width = '60%';
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                progressFill.style.width = '90%';
                
                const data = await response.json();
                
                if (data.ok) {
                    const direction = data.direction;
                    const confidence = (data.final_confidence * 100).toFixed(1);
                    
                    progressFill.style.width = '100%';
                    
                    if (direction === 'buy') {
                        signalText.className = 'signal-buy';
                        signalText.textContent = '🎯 COMPRAR - Entrada Imediata!';
                    } else if (direction === 'sell') {
                        signalText.className = 'signal-sell';
                        signalText.textContent = '🎯 VENDER - Entrada Imediata!';
                    } else {
                        signalText.className = 'signal-neutral';
                        signalText.textContent = '⚡ AGUARDAR - Sem sinal claro';
                    }
                    
                    reasoningText.textContent = data.reasoning || 'Análise concluída';
                    confidenceText.textContent = `Confiança: ${confidence}%`;
                    
                    // Métricas detalhadas
                    const metrics = data.metrics || {};
                    let metricsHtml = '<div style="margin-bottom: 10px; font-weight: bold; text-align: center;">📊 ANÁLISE DETALHADA</div>';
                    
                    metricsHtml += `<div class="metric-item">
                        <span>RSI:</span>
                        <span class="metric-value">${metrics.rsi?.toFixed(1) || 'N/A'}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Força da Tendência:</span>
                        <span class="metric-value">${metrics.trend_strength?.toFixed(1) || 'N/A'}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Padrões Bullish:</span>
                        <span class="metric-value">${(metrics.bullish_candles_ratio * 100)?.toFixed(1) || 'N/A'}%</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Padrões Bearish:</span>
                        <span class="metric-value">${(metrics.bearish_candles_ratio * 100)?.toFixed(1) || 'N/A'}%</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Níveis S/R:</span>
                        <span class="metric-value">${metrics.support_resistance_levels?.toFixed(0) || 'N/A'}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Score da Análise:</span>
                        <span class="metric-value">${metrics.analysis_score?.toFixed(2) || 'N/A'}</span>
                    </div>`;
                    
                    metricsText.innerHTML = metricsHtml;
                    
                } else {
                    signalText.className = 'signal-neutral';
                    signalText.textContent = '❌ Erro na análise';
                    reasoningText.textContent = data.error || 'Tente novamente com outra imagem';
                    confidenceText.textContent = '';
                    progressFill.style.width = '0%';
                }
            } catch (error) {
                signalText.className = 'signal-neutral';
                signalText.textContent = '❌ Erro de conexão';
                reasoningText.textContent = 'Verifique sua internet e tente novamente';
                confidenceText.textContent = '';
                progressFill.style.width = '0%';
            }
            
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🚀 ANALISAR NOVAMENTE';
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
        
        # Verificar tamanho do arquivo
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
            'metrics': analysis['metrics'],
            'reasoning': analysis.get('reasoning', 'Análise concluída')
        })
        
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Erro interno: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'IA Signal Pro Super Assertiva está funcionando!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
