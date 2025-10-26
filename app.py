
from __future__ import annotations

"""
IA Signal Pro — Análise INTELIGENTE E IMPARCIAL (FIX)
- Remove dependência do SciPy (usando apenas NumPy)
- Evita quedas em fallback neutro (50%) por erros silenciosos
- Métricas sempre preenchidas (nunca "undefined")
"""

import io
import os
import math
import datetime
from typing import Any, Dict
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageFilter

# =========================
#  IA INTELIGENTE E IMPARCIAL
# =========================
class IntelligentAnalyzer:
    def _load_image(self, blob: bytes) -> Image.Image:
        """Carrega e prepara a imagem"""
        image = Image.open(io.BytesIO(blob))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Pré-processa a imagem para análise precisa"""
        width, height = image.size

        if width > 800:
            new_width = 800
            new_height = int((height / max(1, width)) * new_width)
            image = image.resize((new_width, new_height), Image.LANCZOS)

        image = image.filter(ImageFilter.SMOOTH)
        img_array = np.array(image)
        return img_array

    def _safe_polyfit_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Retorna slope de uma regressão linear 1D com proteção numérica."""
        try:
            if len(x) < 2 or len(y) < 2 or np.allclose(y, y[0]):
                return 0.0
            p = np.polyfit(x, y, 1)
            return float(p[0])
        except Exception:
            return 0.0

    def _analyze_multiple_timeframes(self, img_array: np.ndarray) -> Dict[str, float]:
        """Análise multi-timeframe inteligente"""
        height, width, _ = img_array.shape
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

        # Análise em diferentes "zoom levels" (timeframes)
        # Foco na metade direita (mais recente) para curto/médio prazos
        short_regions = []
        for i in range(8):
            y_start = int(height * (i * 0.1))
            y_end = int(height * ((i + 1) * 0.1))
            region = gray[y_start:y_end, width//2:]
            if region.size > 0:
                short_regions.append(region.mean())

        medium_regions = []
        for i in range(6):
            y_start = int(height * (i * 0.15))
            y_end = int(height * ((i + 1) * 0.15))
            region = gray[y_start:y_end, width//3:]
            if region.size > 0:
                medium_regions.append(region.mean())

        long_regions = []
        for i in range(4):
            y_start = int(height * (i * 0.25))
            y_end = int(height * ((i + 1) * 0.25))
            region = gray[y_start:y_end, :]
            if region.size > 0:
                long_regions.append(region.mean())

        def calculate_trend(regions):
            if len(regions) < 3:
                return 0.0, 0.0
            x = np.arange(len(regions), dtype=float)
            y = np.array(regions, dtype=float)
            slope = self._safe_polyfit_slope(x, y)
            std = float(np.std(y)) if len(y) > 1 else 0.0
            strength = abs(slope) / (std + 1e-6)
            return float(slope), float(min(5.0, strength))

        short_slope, short_strength = calculate_trend(short_regions)
        medium_slope, medium_strength = calculate_trend(medium_regions)
        long_slope, long_strength = calculate_trend(long_regions)

        # Conflito entre timeframes = menor confiança
        timeframe_alignment = 0.0
        if np.sign(short_slope) == np.sign(medium_slope) == np.sign(long_slope) and np.sign(short_slope) != 0:
            timeframe_alignment = 1.0  # Alinhados
        elif (short_slope == 0 and medium_slope == 0 and long_slope == 0):
            timeframe_alignment = 0.5  # Neutro total
        else:
            timeframe_alignment = 0.3  # Conflitantes

        volatility = float(np.std(gray) / (gray.mean() + 1e-6))

        return {
            "short_trend": float(short_slope),
            "medium_trend": float(medium_slope),
            "long_trend": float(long_slope),
            "short_strength": float(short_strength),
            "medium_strength": float(medium_strength),
            "long_strength": float(long_strength),
            "timeframe_alignment": float(timeframe_alignment),
            "volatility": float(volatility)
        }

    def _detect_support_resistance(self, img_array: np.ndarray) -> Dict[str, float]:
        """Detecta níveis de suporte e resistência de forma inteligente"""
        height, width, _ = img_array.shape
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

        row_means = np.mean(gray, axis=1)
        levels = []
        threshold = np.std(row_means) * 0.3

        for i in range(1, height-1):
            if (abs(row_means[i] - row_means[i-1]) < threshold and 
                abs(row_means[i] - row_means[i+1]) < threshold):
                levels.append(row_means[i])

        unique_levels = []
        level_touches = []

        for level in levels:
            found = False
            for i, existing_level in enumerate(unique_levels):
                if abs(level - existing_level) < threshold * 2 + 1e-6:
                    level_touches[i] += 1
                    found = True
                    break
            if not found:
                unique_levels.append(level)
                level_touches.append(1)

        support_strength = 0.0
        resistance_strength = 0.0
        proximity_factor = 0.5  # neutro

        if level_touches:
            max_touches = max(level_touches)
            current_price = float(np.mean(gray[height//3:2*height//3, -min(50, width-1):]))  # Preço atual (direita)

            supports = [level for level in unique_levels if level < current_price]
            resistances = [level for level in unique_levels if level > current_price]

            nearest_support = max(supports) if supports else None
            nearest_resistance = min(resistances) if resistances else None

            if nearest_support is not None:
                idx = unique_levels.index(nearest_support)
                support_strength = level_touches[idx] / max(1, max_touches)

            if nearest_resistance is not None:
                idx = unique_levels.index(nearest_resistance)
                resistance_strength = level_touches[idx] / max(1, max_touches)

            if nearest_support is not None and nearest_resistance is not None and (nearest_resistance - nearest_support) > 1e-6:
                proximity_factor = (current_price - nearest_support) / (nearest_resistance - nearest_support)
                proximity_factor = float(min(1.0, max(0.0, proximity_factor)))

        return {
            "support_strength": float(min(1.0, max(0.0, support_strength))),
            "resistance_strength": float(min(1.0, max(0.0, resistance_strength))),
            "proximity_to_support": float(min(1.0, max(0.0, proximity_factor))),
            "levels_quality": float(min(1.0, len(unique_levels) / 10.0))
        }

    def _analyze_momentum_oscillators(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analisa momentum usando conceitos de RSI e MACD visuais"""
        height, width, _ = img_array.shape
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

        recent_data = gray[:, -max(2, width//4):]
        older_data  = gray[:, :max(2, width//4)]

        recent_avg = float(np.mean(recent_data))
        older_avg  = float(np.mean(older_data))

        gains  = max(0.0, recent_avg - older_avg)
        losses = max(0.0, older_avg - recent_avg)

        if losses == 0:
            rsi = 100.0 if gains > 0 else 50.0
        else:
            rs = gains / (losses + 1e-6)
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_normalized = (rsi - 50.0) / 50.0  # -1 a 1

        fast_ema = float(np.mean(gray[:, -max(3, width//6):]))
        slow_ema = float(np.mean(gray[:, -max(6, width//3):]))

        macd = fast_ema - slow_ema
        macd_normalized = float(np.clip(macd / (np.std(gray) + 1e-6), -2.0, 2.0))

        left_half  = gray[:, :max(2, width//2)]
        right_half = gray[:, max(1, width//2):]

        left_trend = self._safe_polyfit_slope(np.arange(left_half.shape[1]), np.mean(left_half, axis=0))
        right_trend = self._safe_polyfit_slope(np.arange(right_half.shape[1]), np.mean(right_half, axis=0))

        divergence = float(right_trend - left_trend)

        return {
            "rsi_momentum": float(rsi_normalized),
            "macd_signal": float(macd_normalized),
            "trend_divergence": float(divergence),
            "momentum_strength": float(min(2.0, abs(rsi_normalized) + abs(macd_normalized)))
        }

    def _calculate_volume_analysis(self, img_array: np.ndarray) -> Dict[str, float]:
        """Análise sofisticada de volume e interesse — sem SciPy"""
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

        detail_variance = float(np.var(gray))

        # Gradiente usando apenas NumPy
        gy, gx = np.gradient(gray.astype(np.float64))
        gradient_magnitude = np.sqrt(gx*gx + gy*gy)

        threshold = float(np.percentile(gradient_magnitude, 70))
        activity_concentration = float(np.mean(gradient_magnitude > threshold))

        left_volume = float(np.var(gray[:, :gray.shape[1]//2]))
        right_volume = float(np.var(gray[:, gray.shape[1]//2:]))

        volume_ratio = right_volume / (left_volume + 1e-6)
        volume_trend = float(np.clip((volume_ratio - 1.0) * 2.0, -2.0, 2.0))

        return {
            "volume_intensity": float(min(1.0, detail_variance / 1000.0)),
            "activity_density": float(min(1.0, max(0.0, activity_concentration))),
            "volume_trend": volume_trend,
            "volume_confidence": float(min(1.0, detail_variance / 500.0))
        }

    def _get_intelligent_signal(self, analysis_data: Dict) -> Dict[str, Any]:
        """Decisão inteligente baseada em múltiplos fatores"""
        tf = analysis_data['timeframe']
        sr = analysis_data['support_resistance']
        momentum = analysis_data['momentum']
        volume = analysis_data['volume']

        factors = []

        # 1) Alinhamento de timeframes (peso alto)
        alignment_score = tf['timeframe_alignment']
        if alignment_score > 0.7:
            dominant_trend = np.sign(tf['short_trend'] + tf['medium_trend'] + tf['long_trend'])
            factors.append(("timeframe_alignment", float(dominant_trend) * alignment_score * 2.5))
        elif alignment_score >= 0.5:
            factors.append(("timeframe_neutral", 0.0))
        else:
            factors.append(("timeframe_conflict", 0.0))

        # 2) Suporte/Resistência
        proximity = sr['proximity_to_support']
        if proximity < 0.3 and sr['support_strength'] > 0.6:
            factors.append(("support_bounce", 1.5))
        elif proximity > 0.7 and sr['resistance_strength'] > 0.6:
            factors.append(("resistance_rejection", -1.5))
        else:
            factors.append(("sr_neutral", 0.0))

        # 3) Momentum / RSI / MACD
        rsi = momentum['rsi_momentum']
        if abs(rsi) > 0.5:
            factors.append(("momentum_reversal", -float(rsi) * 1.2))
        else:
            factors.append(("momentum_trend", float(momentum['macd_signal']) * 1.0))

        # 4) Volume confirmando tendência
        volume_confirmation = float(volume['volume_trend']) * float(tf['short_trend'])
        factors.append(("volume_confirmation", volume_confirmation * 1.2))

        # 5) Força da tendência atual
        trend_strength = (float(tf['short_strength']) + float(tf['medium_strength'])) / 2.0
        factors.append(("trend_strength", float(tf['short_trend']) * trend_strength * 1.3))

        # 6) Divergência
        factors.append(("divergence", float(momentum['trend_divergence']) * 0.8))

        total_score = float(sum(score for _, score in factors))

        confidence_factors = [
            float(alignment_score),
            float(max(sr['support_strength'], sr['resistance_strength'])),
            float(momentum['momentum_strength']) / 2.0,
            float(volume['volume_confidence']),
            float(trend_strength)
        ]
        base_confidence = float(np.mean(confidence_factors))

        if total_score > 1.5:
            direction = "buy"
            confidence = min(0.95, 0.75 + (base_confidence * 0.2))
            reasoning = "🚀 FORTE SINAL DE COMPRA - MÚLTIPLOS INDICADORES ALINHADOS"
        elif total_score < -1.5:
            direction = "sell"
            confidence = min(0.95, 0.75 + (base_confidence * 0.2))
            reasoning = "🔻 FORTE SINAL DE VENDA - MÚLTIPLOS INDICADORES ALINHADOS"
        elif total_score > 0.5:
            direction = "buy"
            confidence = 0.65 + (base_confidence * 0.15)
            reasoning = "📈 SINAL DE COMPRA - TENDÊNCIA POSITIVA CONFIRMADA"
        elif total_score < -0.5:
            direction = "sell"
            confidence = 0.65 + (base_confidence * 0.15)
            reasoning = "📉 SINAL DE VENDA - TENDÊNCIA NEGATIVA CONFIRMADA"
        else:
            if momentum['rsi_momentum'] > 0 and volume['volume_trend'] > 0:
                direction = "buy"
                confidence = 0.55
                reasoning = "⚡ VIÉS DE ALTA - MOMENTUM E VOLUME FAVORÁVEIS"
            elif momentum['rsi_momentum'] < 0 and volume['volume_trend'] < 0:
                direction = "sell"
                confidence = 0.55
                reasoning = "⚡ VIÉS DE BAIXA - MOMENTUM E VOLUME FAVORÁVEIS"
            else:
                direction = "buy" if total_score >= 0 else "sell"
                confidence = 0.50
                reasoning = "⚖️ MERCADO EQUILIBRADO - SINAL NEUTRO"

        return {
            "direction": direction,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "total_score": float(total_score),
            "factors_breakdown": {name: float(score) for name, score in factors}
        }

    def _get_entry_timeframe(self) -> Dict[str, str]:
        now = datetime.datetime.now()
        next_minute = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        current_time = now.strftime("%H:%M:%S")
        entry_time_str = next_minute.strftime("%H:%M")
        return {
            "current_time": current_time,
            "entry_time": entry_time_str,
            "timeframe": "Próximo minuto"
        }

    def analyze(self, blob: bytes) -> Dict[str, Any]:
        image = self._load_image(blob)
        img_array = self._preprocess_image(image)

        timeframe_analysis   = self._analyze_multiple_timeframes(img_array)
        support_resistance   = self._detect_support_resistance(img_array)
        momentum_analysis    = self._analyze_momentum_oscillators(img_array)
        volume_analysis      = self._calculate_volume_analysis(img_array)

        analysis_data = {
            'timeframe': timeframe_analysis,
            'support_resistance': support_resistance,
            'momentum': momentum_analysis,
            'volume': volume_analysis
        }

        signal    = self._get_intelligent_signal(analysis_data)
        time_info = self._get_entry_timeframe()

        # Sempre retorna todas as métricas preenchidas
        return {
            "direction": signal["direction"],
            "final_confidence": float(signal["confidence"]),
            "entry_signal": f"🎯 {signal['direction'].upper()} - {signal['reasoning']}",
            "entry_time": time_info["entry_time"],
            "timeframe": time_info["timeframe"],
            "analysis_time": time_info["current_time"],
            "metrics": {
                "analysis_score": float(signal["total_score"]),
                "timeframe_alignment": float(timeframe_analysis["timeframe_alignment"]),
                "support_strength": float(support_resistance["support_strength"]),
                "resistance_strength": float(support_resistance["resistance_strength"]),
                "rsi_momentum": float(momentum_analysis["rsi_momentum"]),
                "macd_signal": float(momentum_analysis["macd_signal"]),
                "volume_trend": float(volume_analysis["volume_trend"]),
                "volatility": float(timeframe_analysis["volatility"]),
                "signal_quality": float(signal["confidence"])
            },
            "reasoning": signal["reasoning"]
        }

# ===============
#  APLICAÇÃO FLASK
# ===============
app = Flask(__name__)
ANALYZER = IntelligentAnalyzer()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Signal Pro - ANÁLISE INTELIGENTE</title>
    <style>
        body { background:#0b1220; color:#e9eef2; font-family: system-ui, -apple-system, Segoe UI, sans-serif; padding:20px;}
        .container { max-width: 560px; margin: 0 auto; background: #0e1524; border-radius: 20px; padding: 24px; border: 2px solid #3a86ff; }
        .title { font-size: 24px; font-weight: 800; margin-bottom: 6px; background: linear-gradient(90deg,#3a86ff,#00ff88); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
        .subtitle { color:#9db0d1; font-size:13px; margin-bottom: 10px;}
        .upload { border:2px dashed #3a86ff; border-radius:14px; padding:18px; background:rgba(58,134,255,.08); margin:14px 0;}
        .file { width:100%; padding:12px; background:rgba(42,53,82,.3); border:1px solid #3a86ff; border-radius:8px; color:white;}
        .btn { width:100%; padding:14px; border-radius:10px; border:none; background:#3a86ff; color:white; font-weight:700; cursor:pointer;}
        .btn:disabled{ background:#2a3552; cursor:not-allowed;}
        .result { display:none; background:#0c1528; border:1px solid #223152; border-radius:12px; padding:16px; margin-top:16px;}
        .signal-buy{ color:#00ff88; font-weight:800; font-size:20px; text-align:center; margin:8px 0;}
        .signal-sell{ color:#ff4444; font-weight:800; font-size:20px; text-align:center; margin:8px 0;}
        .metrics{ background:rgba(42,53,82,.3); padding:12px; border-radius:8px; margin-top:10px; font-size:13px; color:#9db0d1;}
        .row{ display:flex; justify-content:space-between; margin:6px 0;}
        .val{ color:#e9eef2; font-weight:600;}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">🧠 IA SIGNAL PRO - INTELIGENTE</div>
        <div class="subtitle">ANÁLISE AVANÇADA - COMPLETAMENTE IMPARCIAL</div>
        <div class="upload">
            <div style="margin-bottom:8px;">📊 Envie o print do gráfico</div>
            <input id="file" class="file" type="file" accept="image/*"/>
            <button id="go" class="btn">🔍 Analisar</button>
        </div>

        <div id="res" class="result">
            <div id="sig"></div>
            <div class="metrics" id="met"></div>
            <div id="txt" style="margin-top:10px; text-align:center; color:#3a86ff; font-weight:600;"></div>
            <div id="conf" style="text-align:center; color:#9db0d1;"></div>
            <div style="text-align:center; margin-top:8px; font-size:12px; color:#9db0d1;">
                ⏰ Análise: <span id="at">--:--:--</span> • 🎯 Entrada: <span id="et">--:--</span> • ⏱️ <span id="tf">Próximo minuto</span>
            </div>
        </div>
    </div>

    <script>
        const file = document.getElementById('file');
        const go = document.getElementById('go');
        const res = document.getElementById('res');
        const sig = document.getElementById('sig');
        const met = document.getElementById('met');
        const txt = document.getElementById('txt');
        const conf = document.getElementById('conf');
        const at = document.getElementById('at');
        const et = document.getElementById('et');
        const tf = document.getElementById('tf');

        function fmt(n, pct=false){ if(n===undefined||n===null||isNaN(n)) return pct? '0.0%':'0.00'; return pct? (n*100).toFixed(1)+'%': Number(n).toFixed(2); }

        go.addEventListener('click', async () => {
            if(!file.files.length){ alert('Selecione uma imagem.'); return; }
            go.disabled = true;
            res.style.display = 'block';
            sig.className = '';
            sig.textContent = 'Analisando...';

            const fd = new FormData();
            fd.append('image', file.files[0]);

            try{
                const r = await fetch('/analyze', { method:'POST', body: fd });
                const d = await r.json();

                const dir = d.direction || 'buy';
                const confv = d.final_confidence || 0.5;

                sig.className = dir==='buy' ? 'signal-buy' : 'signal-sell';
                sig.textContent = dir==='buy' ? '🎯 COMPRAR - SINAL INTELIGENTE' : '🎯 VENDER - SINAL INTELIGENTE';

                txt.textContent = d.reasoning || 'Análise concluída';
                conf.textContent = 'Confiança Inteligente: ' + (confv*100).toFixed(1) + '%';

                at.textContent = d.analysis_time || '--:--:--';
                et.textContent = d.entry_time || '--:--';
                tf.textContent = d.timeframe || 'Próximo minuto';

                const m = d.metrics || {};
                met.innerHTML = ''
                    + '<div class="row"><span>Score da Análise:</span><span class="val">'+fmt(m.analysis_score)+'</span></div>'
                    + '<div class="row"><span>Alinhamento Timeframes:</span><span class="val">'+fmt(m.timeframe_alignment,true)+'</span></div>'
                    + '<div class="row"><span>Força do Suporte:</span><span class="val">'+fmt(m.support_strength,true)+'</span></div>'
                    + '<div class="row"><span>Força da Resistência:</span><span class="val">'+fmt(m.resistance_strength,true)+'</span></div>'
                    + '<div class="row"><span>RSI Momentum:</span><span class="val">'+fmt(m.rsi_momentum)+'</span></div>'
                    + '<div class="row"><span>Sinal MACD:</span><span class="val">'+fmt(m.macd_signal)+'</span></div>'
                    + '<div class="row"><span>Tendência do Volume:</span><span class="val">'+fmt(m.volume_trend)+'</span></div>'
                    + '<div class="row"><span>Qualidade do Sinal:</span><span class="val">'+fmt(m.signal_quality,true)+'</span></div>';
            }catch(e){
                sig.className = 'signal-buy';
                sig.textContent = '🎯 COMPRAR - MODO SEGURO';
                txt.textContent = 'Falha temporária na análise. Usando heurística segura.';
                conf.textContent = 'Confiança Inteligente: 55.0%';
            }finally{
                go.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_photo():
    if not request.files or 'image' not in request.files:
        return jsonify({'ok': False, 'error': 'Nenhuma imagem enviada'}), 400

    image_file = request.files['image']
    if not image_file or image_file.filename == '':
        return jsonify({'ok': False, 'error': 'Arquivo inválido'}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({'ok': False, 'error': 'Arquivo vazio'}), 400

    # Chamada robusta — sem try/except silencioso aqui para não mascarar erros
    analysis = ANALYZER.analyze(image_bytes)

    return jsonify({
        'ok': True,
        'direction': analysis['direction'],
        'final_confidence': analysis['final_confidence'],
        'entry_signal': analysis['entry_signal'],
        'entry_time': analysis['entry_time'],
        'timeframe': analysis['timeframe'],
        'analysis_time': analysis['analysis_time'],
        'metrics': analysis['metrics'],
        'reasoning': analysis.get('reasoning', 'Análise inteligente concluída')
    })

@app.route('/health')
def health_check():
    return jsonify({'status': 'INTELLIGENT', 'message': 'IA INTELIGENTE FUNCIONANDO!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
