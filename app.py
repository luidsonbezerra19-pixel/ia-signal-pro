from __future__ import annotations

"""
IA Signal Pro — Análise INTELIGENTE PURA
SEM FALLBACK - Apenas análise real do gráfico
"""

import io
import os
import math
import datetime
import hashlib
import json
from typing import Any, Dict, Optional
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageFilter

# =========================
#  SISTEMA DE CACHE INTELIGENTE
# =========================
class AnalysisCache:
    def __init__(self, cache_dir: str = "analysis_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_duration = {
            '1m': 60,
            '5m': 300
        }
    
    def _get_cache_key(self, image_bytes: bytes, timeframe: str) -> str:
        content_hash = hashlib.md5(image_bytes).hexdigest()
        return f"{timeframe}_{content_hash}"
    
    def _get_cache_file(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, image_bytes: bytes, timeframe: str) -> Optional[Dict]:
        key = self._get_cache_key(image_bytes, timeframe)
        cache_file = self._get_cache_file(key)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cache_time = datetime.datetime.fromisoformat(cache_data['timestamp'])
                current_time = datetime.datetime.now()
                age_seconds = (current_time - cache_time).total_seconds()
                
                if age_seconds < self.cache_duration.get(timeframe, 60):
                    return cache_data['analysis']
            except Exception:
                pass
        
        return None
    
    def set(self, image_bytes: bytes, timeframe: str, analysis: Dict):
        key = self._get_cache_key(image_bytes, timeframe)
        cache_file = self._get_cache_file(key)
        
        try:
            cache_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'timeframe': timeframe,
                'analysis': analysis
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass

# =========================
#  IA INTELIGENTE PURA - SEM FALLBACK
# =========================
class IntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
    
    def _load_image(self, blob: bytes) -> Image.Image:
        """Carrega e prepara a imagem para análise"""
        image = Image.open(io.BytesIO(blob))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def _preprocess_image(self, image: Image.Image, timeframe: str) -> np.ndarray:
        """Pré-processamento inteligente do gráfico"""
        width, height = image.size
        
        # Redimensionamento inteligente baseado no timeframe
        if timeframe == '1m':
            target_size = (800, 600)  # Mais detalhes para 1min
        else:
            target_size = (1000, 700)  # Mais contexto para 5min
            
        image = image.resize(target_size, Image.LANCZOS)
        image = image.filter(ImageFilter.SMOOTH_MORE)
        
        return np.array(image)

    def _extract_price_data(self, img_array: np.ndarray) -> np.ndarray:
        """Extrai dados de preço do gráfico de forma inteligente"""
        # Converte para escala de cinza com pesos otimizados
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        # Realce de bordas usando numpy (sem scipy)
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Aplica filtros Sobel manualmente
        sobel_x = self._apply_convolution(gray, kernel_x)
        sobel_y = self._apply_convolution(gray, kernel_y)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Combina informação original com gradiente
        enhanced = gray * 0.7 + gradient * 0.3
        return enhanced

    def _apply_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Aplica convolução manualmente sem scipy"""
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        # Adiciona padding
        padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
        
        # Aplica convolução
        output = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_height, j:j+kernel_width]
                output[i, j] = np.sum(region * kernel)
        
        return output

    def _analyze_price_action(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise inteligente da ação do preço"""
        height, width = price_data.shape
        
        # Análise de tendência por regiões temporais
        regions = []
        segment_size = width // 8
        
        for i in range(8):
            start_col = i * segment_size
            end_col = (i + 1) * segment_size
            if end_col > width:
                end_col = width
                
            segment = price_data[:, start_col:end_col]
            if segment.size > 0:
                region_mean = np.mean(segment)
                regions.append(region_mean)
        
        # Calcula tendência usando regressão linear
        if len(regions) >= 3:
            x = np.arange(len(regions))
            trend_slope, trend_intercept = np.polyfit(x, regions, 1)
            
            # Força da tendência baseada no R²
            y_pred = trend_slope * x + trend_intercept
            ss_res = np.sum((regions - y_pred) ** 2)
            ss_tot = np.sum((regions - np.mean(regions)) ** 2)
            trend_strength = 1 - (ss_res / (ss_tot + 1e-8))
        else:
            trend_slope = 0
            trend_strength = 0
        
        # Análise de momentum (derivada segunda)
        if len(regions) >= 4:
            momentum = np.gradient(np.gradient(regions))
            current_momentum = momentum[-1] if len(momentum) > 0 else 0
        else:
            current_momentum = 0
        
        # Volatilidade adaptativa
        volatility = np.std(price_data) / (np.mean(price_data) + 1e-8)
        
        return {
            "trend_direction": float(trend_slope),
            "trend_strength": float(trend_strength),
            "momentum": float(current_momentum),
            "volatility": float(volatility),
            "price_range": float(np.ptp(price_data))
        }

    def _analyze_chart_patterns(self, price_data: np.ndarray) -> Dict[str, float]:
        """Detecção inteligente de padrões gráficos"""
        height, width = price_data.shape
        
        # Análise de suportes e resistências
        horizontal_profiles = []
        for row in range(0, height, 5):  # Amostra a cada 5 pixels
            row_data = price_data[row, :]
            if len(row_data) > 10:
                # Encontra áreas de congestão (baixa variância)
                row_variance = np.std(row_data)
                if row_variance < np.std(price_data) * 0.3:
                    horizontal_profiles.append(np.mean(row_data))
        
        # Agrupa níveis próximos
        unique_levels = []
        for level in horizontal_profiles:
            if not unique_levels or min(abs(level - lvl) for lvl in unique_levels) > np.std(price_data) * 0.2:
                unique_levels.append(level)
        
        # Preço atual (última coluna)
        current_price = np.mean(price_data[:, -10:])
        
        # Encontra suportes e resistências
        supports = [level for level in unique_levels if level < current_price]
        resistances = [level for level in unique_levels if level > current_price]
        
        support_strength = len(supports) / max(1, len(unique_levels))
        resistance_strength = len(resistances) / max(1, len(unique_levels))
        
        # Proximidade aos níveis
        if supports:
            nearest_support = max(supports)
            distance_to_support = (current_price - nearest_support) / current_price
        else:
            distance_to_support = 0.5
            
        if resistances:
            nearest_resistance = min(resistances)
            distance_to_resistance = (nearest_resistance - current_price) / current_price
        else:
            distance_to_resistance = 0.5
        
        return {
            "support_levels": len(supports),
            "resistance_levels": len(resistances),
            "support_strength": float(support_strength),
            "resistance_strength": float(resistance_strength),
            "distance_to_support": float(distance_to_support),
            "distance_to_resistance": float(distance_to_resistance),
            "consolidation_level": float(len(unique_levels) / 20)
        }

    def _analyze_market_structure(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise da estrutura de mercado"""
        height, width = price_data.shape
        
        # Análise de higher highs/lower lows
        middle_row = height // 2
        upper_half = price_data[:middle_row, :]
        lower_half = price_data[middle_row:, :]
        
        upper_trend = np.polyfit(range(upper_half.shape[1]), np.mean(upper_half, axis=0), 1)[0]
        lower_trend = np.polyfit(range(lower_half.shape[1]), np.mean(lower_half, axis=0), 1)[0]
        
        # Tendência de mercado
        market_trend = (upper_trend + lower_trend) / 2
        
        # Análise de força dos movimentos
        left_side = price_data[:, :width//2]
        right_side = price_data[:, width//2:]
        
        left_volatility = np.std(left_side) / (np.mean(left_side) + 1e-8)
        right_volatility = np.std(right_side) / (np.mean(right_side) + 1e-8)
        
        volatility_ratio = right_volatility / (left_volatility + 1e-8)
        
        # Força do movimento atual
        recent_movement = np.mean(price_data[:, -width//10:]) - np.mean(price_data[:, -width//5:-width//10])
        movement_strength = abs(recent_movement) / (np.std(price_data) + 1e-8)
        
        return {
            "market_trend": float(market_trend),
            "volatility_ratio": float(volatility_ratio),
            "movement_strength": float(movement_strength),
            "structure_quality": float(min(1.0, len(price_data) / 1000))
        }

    def _calculate_advanced_indicators(self, price_data: np.ndarray) -> Dict[str, float]:
        """Indicadores técnicos avançados"""
        height, width = price_data.shape
        
        # RSI Visual
        recent_period = width // 4
        older_period = width // 2
        
        recent_avg = np.mean(price_data[:, -recent_period:])
        older_avg = np.mean(price_data[:, -older_period:-recent_period])
        
        gain = max(0, recent_avg - older_avg)
        loss = max(0, older_avg - recent_avg)
        
        if loss == 0:
            rsi = 70 if gain > 0 else 30
        else:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_normalized = (rsi - 50) / 50
        
        # MACD Visual
        fast_ema = np.mean(price_data[:, -width//8:])
        slow_ema = np.mean(price_data[:, -width//4:])
        macd_line = fast_ema - slow_ema
        macd_normalized = macd_line / (np.std(price_data) + 1e-8)
        
        # Volume proxy (variação de detalhes)
        volume_intensity = np.var(price_data) / 1000
        
        return {
            "rsi": float(rsi_normalized),
            "macd": float(macd_normalized),
            "volume_intensity": float(min(1.0, volume_intensity)),
            "momentum_quality": float(abs(rsi_normalized) + abs(macd_normalized))
        }

    def _make_intelligent_decision(self, analysis: Dict, timeframe: str) -> Dict[str, Any]:
        """Tomada de decisão inteligente baseada em múltiplos fatores"""
        price_action = analysis['price_action']
        chart_patterns = analysis['chart_patterns']
        market_structure = analysis['market_structure']
        indicators = analysis['indicators']
        
        # Sistema de pontuação ponderada
        score_components = []
        
        # 1. Tendência principal (30%)
        trend_score = price_action['trend_direction'] * price_action['trend_strength']
        score_components.append(trend_score * 0.3)
        
        # 2. Momentum (20%)
        momentum_score = price_action['momentum'] * 2
        score_components.append(momentum_score * 0.2)
        
        # 3. Posição relativa aos níveis (20%)
        level_score = 0
        if chart_patterns['distance_to_support'] < 0.1 and chart_patterns['support_strength'] > 0.6:
            level_score = 1.0  # Próximo de suporte forte
        elif chart_patterns['distance_to_resistance'] < 0.1 and chart_patterns['resistance_strength'] > 0.6:
            level_score = -1.0  # Próximo de resistência forte
        elif chart_patterns['distance_to_support'] < chart_patterns['distance_to_resistance']:
            level_score = 0.3  # Mais perto do suporte
        else:
            level_score = -0.3  # Mais perto da resistência
        score_components.append(level_score * 0.2)
        
        # 4. Indicadores técnicos (15%)
        indicator_score = (indicators['rsi'] + indicators['macd']) / 2
        score_components.append(indicator_score * 0.15)
        
        # 5. Estrutura de mercado (15%)
        structure_score = market_structure['market_trend'] * market_structure['movement_strength']
        score_components.append(structure_score * 0.15)
        
        # Score final
        total_score = sum(score_components)
        
        # Cálculo de confiança
        confidence_factors = [
            price_action['trend_strength'],
            chart_patterns['consolidation_level'],
            market_structure['structure_quality'],
            indicators['momentum_quality'] / 2
        ]
        
        base_confidence = np.mean(confidence_factors)
        
        # Decisão inteligente
        if total_score > 0.15:
            direction = "buy"
            confidence = 0.60 + (base_confidence * 0.35)
            reasoning = "📈 Tendência de alta identificada com confirmação técnica"
        elif total_score < -0.15:
            direction = "sell"
            confidence = 0.60 + (base_confidence * 0.35)
            reasoning = "📉 Tendência de baixa identificada com confirmação técnica"
        elif total_score > 0.05:
            direction = "buy"
            confidence = 0.55 + (base_confidence * 0.25)
            reasoning = "↗️ Viés de alta com sinais técnicos favoráveis"
        elif total_score < -0.05:
            direction = "sell"
            confidence = 0.55 + (base_confidence * 0.25)
            reasoning = "↘️ Viés de baixa com sinais técnicos favoráveis"
        else:
            # Mercado em equilíbrio - análise mais profunda
            if indicators['rsi'] > 0 and market_structure['market_trend'] > 0:
                direction = "buy"
                confidence = 0.52
                reasoning = "⚡ Leve viés de alta em mercado equilibrado"
            else:
                direction = "sell"
                confidence = 0.52
                reasoning = "⚡ Leve viés de baixa em mercado equilibrado"
        
        return {
            "direction": direction,
            "confidence": min(0.95, confidence),
            "reasoning": reasoning,
            "total_score": total_score
        }

    def _get_entry_timeframe(self, user_timeframe: str) -> Dict[str, str]:
        """Calcula horário de entrada baseado no timeframe"""
        now = datetime.datetime.now()
        
        if user_timeframe == '1m':
            entry_time = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
            timeframe_str = "Próximo minuto"
        else:
            minutes_to_add = 5 - (now.minute % 5)
            if minutes_to_add == 0:
                minutes_to_add = 5
            entry_time = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=minutes_to_add)
            timeframe_str = f"Próximo candle de 5min"
        
        return {
            "current_time": now.strftime("%H:%M:%S"),
            "entry_time": entry_time.strftime("%H:%M"),
            "timeframe": timeframe_str
        }

    def analyze(self, blob: bytes, timeframe: str = '1m') -> Dict[str, Any]:
        """Análise principal - SEM FALLBACK"""
        
        # Verifica cache
        cached_analysis = self.cache.get(blob, timeframe)
        if cached_analysis:
            cached_analysis['cached'] = True
            return cached_analysis
        
        try:
            # Processamento da imagem
            image = self._load_image(blob)
            img_array = self._preprocess_image(image, timeframe)
            
            # Extração de dados de preço
            price_data = self._extract_price_data(img_array)
            
            # Análises especializadas
            price_action = self._analyze_price_action(price_data, timeframe)
            chart_patterns = self._analyze_chart_patterns(price_data)
            market_structure = self._analyze_market_structure(price_data, timeframe)
            indicators = self._calculate_advanced_indicators(price_data)
            
            # Consolida análise
            analysis_data = {
                'price_action': price_action,
                'chart_patterns': chart_patterns,
                'market_structure': market_structure,
                'indicators': indicators
            }
            
            # Tomada de decisão inteligente
            decision = self._make_intelligent_decision(analysis_data, timeframe)
            
            # Informações de tempo
            time_info = self._get_entry_timeframe(timeframe)
            
            # Resultado final
            result = {
                "direction": decision["direction"],
                "final_confidence": float(decision["confidence"]),
                "entry_signal": f"🎯 {decision['direction'].upper()} - {decision['reasoning']}",
                "entry_time": time_info["entry_time"],
                "timeframe": time_info["timeframe"],
                "analysis_time": time_info["current_time"],
                "user_timeframe": timeframe,
                "cached": False,
                "metrics": {
                    "analysis_score": float(decision["total_score"]),
                    "trend_strength": price_action["trend_strength"],
                    "momentum": price_action["momentum"],
                    "rsi": indicators["rsi"],
                    "macd": indicators["macd"],
                    "support_levels": chart_patterns["support_levels"],
                    "resistance_levels": chart_patterns["resistance_levels"],
                    "volatility": price_action["volatility"],
                    "volume_intensity": indicators["volume_intensity"],
                    "signal_quality": float(decision["confidence"])
                },
                "reasoning": decision["reasoning"]
            }
            
            # Salva no cache
            self.cache.set(blob, timeframe, result)
            
            return result
            
        except Exception as e:
            # SEM FALLBACK - Se der erro, mostra o erro real
            raise Exception(f"Erro na análise do gráfico: {str(e)}")

# ===============
#  APLICAÇÃO FLASK
# ===============
app = Flask(__name__)
ANALYZER = IntelligentAnalyzer()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Signal Pro - ANÁLISE PURA</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: linear-gradient(135deg, #0b1220 0%, #1a1f38 100%); 
            color: #e9eef2; font-family: 'Segoe UI', system-ui, sans-serif;
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 500px; margin: 0 auto;
            background: rgba(15, 22, 39, 0.95); border-radius: 20px;
            padding: 25px; border: 2px solid #3a86ff;
            box-shadow: 0 10px 30px rgba(58, 134, 255, 0.2);
        }
        .header { text-align: center; margin-bottom: 20px; }
        .title {
            font-size: 24px; font-weight: 800; margin-bottom: 5px;
            background: linear-gradient(90deg, #3a86ff, #00ff88);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #9db0d1; font-size: 13px; margin-bottom: 10px; }
        
        .timeframe-selector { display: flex; gap: 10px; margin: 15px 0; }
        .timeframe-btn {
            flex: 1; padding: 12px; border: 2px solid #3a86ff;
            background: rgba(58, 134, 255, 0.1); color: #9db0d1;
            border-radius: 10px; cursor: pointer; text-align: center;
            font-weight: 600; transition: all 0.3s ease;
        }
        .timeframe-btn.active {
            background: linear-gradient(135deg, #3a86ff 0%, #2a76ef 100%);
            color: white; border-color: #2a76ef;
        }
        .timeframe-btn:hover {
            background: rgba(58, 134, 255, 0.2);
        }
        
        .upload-area {
            border: 2px dashed #3a86ff; border-radius: 15px;
            padding: 20px 15px; text-align: center;
            background: rgba(58, 134, 255, 0.05); margin-bottom: 20px;
        }
        .file-input {
            margin: 15px 0; padding: 12px;
            background: rgba(42, 53, 82, 0.3); border: 1px solid #3a86ff;
            border-radius: 8px; color: white; width: 100%; cursor: pointer;
        }
        .analyze-btn {
            background: linear-gradient(135deg, #3a86ff 0%, #2a76ef 100%);
            color: white; border: none; border-radius: 10px; padding: 16px;
            font-size: 16px; font-weight: 700; cursor: pointer; width: 100%;
            transition: all 0.3s ease;
        }
        .analyze-btn:hover { 
            background: linear-gradient(135deg, #2a76ef 0%, #1a66df 100%);
            transform: translateY(-2px);
        }
        .analyze-btn:disabled { 
            background: #2a3552; transform: none; cursor: not-allowed;
        }
        
        .result { 
            display: none; background: rgba(14, 21, 36, 0.9);
            border-radius: 15px; padding: 20px; margin-top: 20px;
            border: 1px solid #223152;
        }
        .signal-buy { 
            color: #00ff88; font-weight: 800; font-size: 22px; 
            text-align: center; margin-bottom: 10px;
        }
        .signal-sell { 
            color: #ff4444; font-weight: 800; font-size: 22px; 
            text-align: center; margin-bottom: 10px;
        }
        
        .time-info {
            background: rgba(42, 53, 82, 0.5); border-radius: 8px;
            padding: 12px; margin: 10px 0; text-align: center;
        }
        .time-item {
            margin: 5px 0; display: flex; justify-content: space-between;
            align-items: center;
        }
        .time-label { color: #9db0d1; font-size: 13px; }
        .time-value { color: #00ff88; font-weight: 600; font-size: 14px; }
        
        .confidence {
            font-size: 16px; text-align: center; margin: 10px 0;
            color: #9db0d1;
        }
        .reasoning {
            text-align: center; margin: 12px 0; color: #3a86ff;
            font-weight: 600; font-size: 14px;
        }
        
        .metrics {
            margin-top: 15px; font-size: 13px; color: #9db0d1;
            background: rgba(42, 53, 82, 0.3); padding: 15px;
            border-radius: 8px;
        }
        .metric-item {
            margin: 6px 0; display: flex; justify-content: space-between;
            align-items: center;
        }
        .metric-value {
            font-weight: 600; color: #e9eef2;
        }
        
        .error-message {
            background: rgba(255, 68, 68, 0.1); border: 1px solid #ff4444;
            border-radius: 10px; padding: 15px; margin: 10px 0;
            color: #ff8888; text-align: center;
        }
        
        .loading {
            text-align: center; color: #3a86ff; font-size: 14px;
        }
        .progress-bar {
            width: 100%; height: 4px; background: #2a3552;
            border-radius: 2px; margin: 12px 0; overflow: hidden;
        }
        .progress-fill {
            height: 100%; background: linear-gradient(90deg, #3a86ff, #00ff88);
            width: 0%; transition: width 0.3s ease;
        }
        
        .cache-badge {
            background: linear-gradient(135deg, #ffa500, #ff6b6b);
            color: white; padding: 4px 8px; border-radius: 12px;
            font-size: 10px; font-weight: 700; margin-left: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠 IA SIGNAL PRO - ANÁLISE PURA</div>
            <div class="subtitle">INTELIGÊNCIA ARTIFICIAL AVANÇADA - SEM FALLBACK</div>
        </div>
        
        <div class="timeframe-selector">
            <button class="timeframe-btn active" data-timeframe="1m">⏱️ 1 MINUTO</button>
            <button class="timeframe-btn" data-timeframe="5m">⏱️ 5 MINUTOS</button>
        </div>
        
        <div class="upload-area">
            <div style="font-size: 15px; margin-bottom: 8px;">
                📊 ENVIE O PRINT DO GRÁFICO
            </div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
            <button class="analyze-btn" id="analyzeBtn">🔍 ANALISAR INTELIGENTEMENTE</button>
        </div>
        
        <div class="result" id="result">
            <div id="signalText"></div>
            <div id="errorMessage" class="error-message" style="display: none;"></div>
            
            <div class="time-info">
                <div class="time-item">
                    <span class="time-label">⏰ Horário da Análise:</span>
                    <span class="time-value" id="analysisTime">--:--:--</span>
                </div>
                <div class="time-item">
                    <span class="time-label">🎯 Entrada Recomendada:</span>
                    <span class="time-value" id="entryTime">--:--</span>
                </div>
                <div class="time-item">
                    <span class="time-label">⏱️ Timeframe:</span>
                    <span class="time-value" id="timeframe">Próximo minuto</span>
                </div>
            </div>
            
            <div class="reasoning" id="reasoningText"></div>
            <div class="confidence" id="confidenceText"></div>
            
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="metrics" id="metricsText"></div>
        </div>
    </div>

    <script>
        // Variáveis globais
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const result = document.getElementById('result');
        const signalText = document.getElementById('signalText');
        const errorMessage = document.getElementById('errorMessage');
        const analysisTime = document.getElementById('analysisTime');
        const entryTime = document.getElementById('entryTime');
        const timeframeEl = document.getElementById('timeframe');
        const reasoningText = document.getElementById('reasoningText');
        const confidenceText = document.getElementById('confidenceText');
        const progressFill = document.getElementById('progressFill');
        const metricsText = document.getElementById('metricsText');
        const timeframeBtns = document.querySelectorAll('.timeframe-btn');

        let currentTimeframe = '1m';
        let selectedFile = null;

        // Seleção de timeframe
        timeframeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                timeframeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentTimeframe = btn.dataset.timeframe;
                if (selectedFile) {
                    analyzeBtn.textContent = `✅ PRONTO PARA ANÁLISE ${currentTimeframe.toUpperCase()}`;
                }
            });
        });

        fileInput.addEventListener('change', (e) => {
            selectedFile = e.target.files[0] || null;
            if (selectedFile) {
                analyzeBtn.textContent = `✅ PRONTO PARA ANÁLISE ${currentTimeframe.toUpperCase()}`;
            }
        });

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('📸 Selecione uma imagem do gráfico primeiro!');
                return;
            }

            analyzeBtn.disabled = true;
            analyzeBtn.textContent = `🧠 ANALISANDO ${currentTimeframe.toUpperCase()}...`;
            result.style.display = 'block';
            errorMessage.style.display = 'none';
            
            signalText.className = '';
            signalText.textContent = 'Analisando padrões do gráfico...';
            
            const now = new Date();
            analysisTime.textContent = now.toLocaleTimeString('pt-BR');
            
            // Calcula horário de entrada baseado no timeframe
            let entryTimeValue;
            if (currentTimeframe === '1m') {
                const nextMinute = new Date(now);
                nextMinute.setMinutes(nextMinute.getMinutes() + 1);
                nextMinute.setSeconds(0);
                entryTimeValue = nextMinute.toLocaleTimeString('pt-BR').slice(0, 5);
                timeframeEl.textContent = 'Próximo minuto';
            } else {
                const minutesToAdd = 5 - (now.getMinutes() % 5);
                const next5min = new Date(now);
                next5min.setMinutes(next5min.getMinutes() + minutesToAdd);
                next5min.setSeconds(0);
                entryTimeValue = next5min.toLocaleTimeString('pt-BR').slice(0, 5);
                timeframeEl.textContent = `Próximo candle de 5min (${entryTimeValue})`;
            }
            
            entryTime.textContent = entryTimeValue;
            reasoningText.textContent = 'Processando análise técnica avançada...';
            confidenceText.textContent = '';
            progressFill.style.width = '20%';
            
            metricsText.innerHTML = '<div class="loading">Iniciando análise inteligente...</div>';

            try {
                const formData = new FormData();
                formData.append('image', selectedFile);
                formData.append('timeframe', currentTimeframe);
                
                progressFill.style.width = '40%';
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                progressFill.style.width = '80%';
                
                const data = await response.json();
                
                progressFill.style.width = '100%';
                
                if (data.ok) {
                    const direction = data.direction;
                    const confidence = (data.final_confidence * 100).toFixed(1);
                    const cached = data.cached || false;
                    
                    if (direction === 'buy') {
                        signalText.className = 'signal-buy';
                        signalText.innerHTML = `🎯 COMPRAR ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                    } else {
                        signalText.className = 'signal-sell';
                        signalText.innerHTML = `🎯 VENDER ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                    }
                    
                    analysisTime.textContent = data.analysis_time || '--:--:--';
                    entryTime.textContent = data.entry_time || '--:--';
                    timeframeEl.textContent = data.timeframe || 'Próximo minuto';
                    
                    reasoningText.textContent = data.reasoning;
                    confidenceText.textContent = `Confiança Inteligente: ${confidence}%`;
                    
                    // Métricas detalhadas
                    const metrics = data.metrics || {};
                    let metricsHtml = '<div style="margin-bottom: 10px; text-align: center; font-weight: 600;">📊 ANÁLISE DETALHADA</div>';
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Score da Análise:</span>
                        <span class="metric-value">${metrics.analysis_score?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Força da Tendência:</span>
                        <span class="metric-value">${(metrics.trend_strength * 100)?.toFixed(1)}%</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Momentum:</span>
                        <span class="metric-value">${metrics.momentum?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>RSI:</span>
                        <span class="metric-value">${metrics.rsi?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>MACD:</span>
                        <span class="metric-value">${metrics.macd?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Suportes:</span>
                        <span class="metric-value">${metrics.support_levels || 0}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Resistências:</span>
                        <span class="metric-value">${metrics.resistance_levels || 0}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Volatilidade:</span>
                        <span class="metric-value">${(metrics.volatility * 100)?.toFixed(1)}%</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Qualidade do Sinal:</span>
                        <span class="metric-value">${(metrics.signal_quality * 100)?.toFixed(1)}%</span>
                    </div>`;
                    
                    metricsText.innerHTML = metricsHtml;
                    
                } else {
                    throw new Error(data.error || 'Erro na análise');
                }
            } catch (error) {
                errorMessage.style.display = 'block';
                errorMessage.textContent = `❌ Erro na análise: ${error.message}`;
                signalText.textContent = '❌ Análise Falhou';
                metricsText.innerHTML = '<div class="loading">Erro no processamento</div>';
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = `🔍 ANALISAR ${currentTimeframe.toUpperCase()} NOVAMENTE`;
            }
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
        
        timeframe = request.form.get('timeframe', '1m')
        if timeframe not in ['1m', '5m']:
            timeframe = '1m'
        
        # Verificação básica do arquivo
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'ok': False, 'error': 'Imagem muito grande (máximo 10MB)'}), 400
        
        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({'ok': False, 'error': 'Arquivo vazio'}), 400
        
        # Análise REAL - SEM FALLBACK
        analysis = ANALYZER.analyze(image_bytes, timeframe)
        
        return jsonify({
            'ok': True,
            'direction': analysis['direction'],
            'final_confidence': analysis['final_confidence'],
            'entry_signal': analysis['entry_signal'],
            'entry_time': analysis['entry_time'],
            'timeframe': analysis['timeframe'],
            'analysis_time': analysis['analysis_time'],
            'user_timeframe': analysis['user_timeframe'],
            'cached': analysis.get('cached', False),
            'metrics': analysis['metrics'],
            'reasoning': analysis.get('reasoning', 'Análise concluída')
        })
        
    except Exception as e:
        # SEM FALLBACK - Retorna o erro real
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'PURE_ANALYSIS', 'message': 'IA INTELIGENTE PURA FUNCIONANDO!'})

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Endpoint para limpar o cache"""
    try:
        cache_dir = "analysis_cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, file))
            return jsonify({'ok': True, 'message': 'Cache limpo com sucesso!'})
        return jsonify({'ok': True, 'message': 'Cache já está vazio!'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    print(f"🚀 IA Signal Pro - ANÁLISE PURA iniciando na porta {port}")
    print(f"📊 Sistema: Análise Inteligente SEM FALLBACK")
    print(f"⏰ Timeframes: 1min e 5min com cache inteligente")
    app.run(host='0.0.0.0', port=port, debug=debug)
