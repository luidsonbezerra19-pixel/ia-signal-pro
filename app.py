from __future__ import annotations

"""
IA Signal Pro — Análise INTELIGENTE PURA - VERSÃO ESTÁVEL
Sistema otimizado para deploy em produção
"""

import io
import os
import math
import datetime
import hashlib
import json
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
from flask import Flask, jsonify, render_template_string, request, Response
from PIL import Image, ImageFilter
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            key = self._get_cache_key(image_bytes, timeframe)
            cache_file = self._get_cache_file(key)
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cache_time = datetime.datetime.fromisoformat(cache_data['timestamp'])
                current_time = datetime.datetime.now()
                age_seconds = (current_time - cache_time).total_seconds()
                
                if age_seconds < self.cache_duration.get(timeframe, 60):
                    return cache_data['analysis']
        except Exception as e:
            logger.warning(f"Erro no cache: {e}")
        
        return None
    
    def set(self, image_bytes: bytes, timeframe: str, analysis: Dict):
        try:
            key = self._get_cache_key(image_bytes, timeframe)
            cache_file = self._get_cache_file(key)
            
            cache_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'timeframe': timeframe,
                'analysis': analysis
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")

# =========================
#  IA INTELIGENTE PURA - VERSÃO ESTÁVEL
# =========================
class IntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
        self.min_confidence_threshold = 0.58
    
    def _load_image(self, blob: bytes) -> Image.Image:
        """Carrega e prepara a imagem para análise"""
        try:
            image = Image.open(io.BytesIO(blob))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Erro ao carregar imagem: {str(e)}")
    
    def _validate_chart_image(self, image: Image.Image) -> bool:
        """Validação básica do gráfico"""
        width, height = image.size
        
        if width < 100 or height < 100:
            raise ValueError("Imagem muito pequena (mínimo 100x100 pixels)")
        
        try:
            img_array = np.array(image)
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            contrast = np.std(gray)
            
            if contrast < 10:
                raise ValueError("Contraste insuficiente para análise")
            
            return True
        except Exception as e:
            raise ValueError(f"Erro na validação: {str(e)}")

    def _preprocess_image(self, image: Image.Image, timeframe: str) -> np.ndarray:
        """Pré-processamento otimizado"""
        width, height = image.size
        
        # Redimensionamento adequado
        target_size = (400, 300)  # Tamanho fixo para estabilidade
        image = image.resize(target_size, Image.LANCZOS)
        
        return np.array(image)

    def _extract_price_data(self, img_array: np.ndarray) -> np.ndarray:
        """Extrai dados de preço de forma estável"""
        try:
            # Converte para escala de cinza
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Filtro simples para realce
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = self._apply_simple_convolution(gray, kernel)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Erro na extração: {e}")
            return np.dot(img_array[...,:3], [0.299, 0.587, 0.114])  # Fallback

    def _apply_simple_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Aplica convolução de forma simples e estável"""
        try:
            kernel_height, kernel_width = kernel.shape
            pad_height = kernel_height // 2
            pad_width = kernel_width // 2
            
            padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
            output = np.zeros_like(image)
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded[i:i+kernel_height, j:j+kernel_width]
                    output[i, j] = np.sum(region * kernel)
            
            return np.clip(output, 0, 255)  # Garante valores válidos
        except Exception:
            return image  # Fallback

    def _analyze_price_action(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise de price action estável"""
        try:
            height, width = price_data.shape
            
            # Análise simples por segmentos
            segments = 4
            segment_size = max(1, width // segments)
            region_means = []
            
            for i in range(segments):
                start_col = i * segment_size
                end_col = min((i + 1) * segment_size, width)
                segment = price_data[:, start_col:end_col]
                
                if segment.size > 0:
                    region_means.append(np.mean(segment))
            
            # Tendência básica
            if len(region_means) >= 2:
                x = np.arange(len(region_means))
                slope, _ = np.polyfit(x, region_means, 1)
                
                # Força da tendência (simplificado)
                y_pred = slope * x + np.mean(region_means)
                ss_res = np.sum((region_means - y_pred) ** 2)
                ss_tot = np.sum((region_means - np.mean(region_means)) ** 2)
                trend_strength = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0
            else:
                slope = 0
                trend_strength = 0
            
            # Métricas básicas
            volatility = np.std(price_data) / (np.mean(price_data) + 1e-8)
            price_range = np.ptp(price_data)
            
            return {
                "trend_direction": float(slope),
                "trend_strength": float(min(1.0, max(0.0, trend_strength))),
                "momentum": float(slope * 0.5),  # Simplificado
                "volatility": float(volatility),
                "price_range": float(price_range)
            }
        except Exception as e:
            logger.warning(f"Erro na análise de price action: {e}")
            return {
                "trend_direction": 0.0,
                "trend_strength": 0.0,
                "momentum": 0.0,
                "volatility": 0.0,
                "price_range": 0.0
            }

    def _analyze_chart_patterns(self, price_data: np.ndarray) -> Dict[str, float]:
        """Análise de padrões simplificada"""
        try:
            height, width = price_data.shape
            
            # Análise de níveis horizontais básica
            levels = []
            step = max(1, height // 20)
            
            for row in range(0, height, step):
                row_data = price_data[row, :]
                if len(row_data) > 5:
                    row_std = np.std(row_data)
                    if row_std < np.std(price_data) * 0.4:
                        levels.append(np.mean(row_data))
            
            # Agrupa níveis próximos
            unique_levels = []
            threshold = np.std(price_data) * 0.2 if np.std(price_data) > 0 else 1.0
            
            for level in sorted(levels):
                if not unique_levels or min(abs(level - lvl) for lvl in unique_levels) > threshold:
                    unique_levels.append(level)
            
            # Preço atual
            current_price = np.mean(price_data[:, -min(5, width):])
            
            # Classifica suportes e resistências
            supports = [lvl for lvl in unique_levels if lvl < current_price]
            resistances = [lvl for lvl in unique_levels if lvl > current_price]
            
            support_strength = len(supports) / 10.0
            resistance_strength = len(resistances) / 10.0
            
            return {
                "support_levels": len(supports),
                "resistance_levels": len(resistances),
                "support_strength": float(min(1.0, support_strength)),
                "resistance_strength": float(min(1.0, resistance_strength)),
                "distance_to_support": 0.5,  # Simplificado
                "distance_to_resistance": 0.5,
                "consolidation_level": float(min(1.0, len(unique_levels) / 15.0))
            }
        except Exception as e:
            logger.warning(f"Erro na análise de padrões: {e}")
            return {
                "support_levels": 0,
                "resistance_levels": 0,
                "support_strength": 0.0,
                "resistance_strength": 0.0,
                "distance_to_support": 0.5,
                "distance_to_resistance": 0.5,
                "consolidation_level": 0.0
            }

    def _analyze_market_structure(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise de estrutura de mercado simplificada"""
        try:
            height, width = price_data.shape
            
            if height < 2 or width < 2:
                return {
                    "market_trend": 0.0,
                    "volatility_ratio": 1.0,
                    "movement_strength": 0.0,
                    "structure_quality": 0.0
                }
            
            # Tendência geral
            overall_trend = np.polyfit(range(width), np.mean(price_data, axis=0), 1)[0] if width > 1 else 0
            
            # Força do movimento
            if width > 10:
                recent = np.mean(price_data[:, -5:])
                older = np.mean(price_data[:, -10:-5])
                movement = (recent - older) / (np.std(price_data) + 1e-8)
            else:
                movement = 0
            
            return {
                "market_trend": float(overall_trend),
                "volatility_ratio": 1.0,
                "movement_strength": float(min(2.0, abs(movement))),
                "structure_quality": float(min(1.0, (height * width) / 100000.0))
            }
        except Exception as e:
            logger.warning(f"Erro na análise de estrutura: {e}")
            return {
                "market_trend": 0.0,
                "volatility_ratio": 1.0,
                "movement_strength": 0.0,
                "structure_quality": 0.0
            }

    def _calculate_advanced_indicators(self, price_data: np.ndarray) -> Dict[str, float]:
        """Indicadores técnicos simplificados"""
        try:
            height, width = price_data.shape
            
            if width < 5:
                return {
                    "rsi": 0.0,
                    "macd": 0.0,
                    "volume_intensity": 0.0,
                    "momentum_quality": 0.0
                }
            
            # RSI simplificado
            if width > 10:
                recent = np.mean(price_data[:, -5:])
                older = np.mean(price_data[:, -10:-5])
                change = recent - older
                
                if change > 0:
                    rsi = 60
                elif change < 0:
                    rsi = 40
                else:
                    rsi = 50
                
                rsi_normalized = (rsi - 50) / 50
            else:
                rsi_normalized = 0.0
            
            # MACD simplificado
            if width > 8:
                fast = np.mean(price_data[:, -3:])
                slow = np.mean(price_data[:, -8:])
                macd_normalized = (fast - slow) / (np.std(price_data) + 1e-8)
            else:
                macd_normalized = 0.0
            
            return {
                "rsi": float(max(-1.0, min(1.0, rsi_normalized))),
                "macd": float(max(-1.0, min(1.0, macd_normalized))),
                "volume_intensity": float(min(1.0, np.var(price_data) / 1000.0)),
                "momentum_quality": float(min(1.0, (abs(rsi_normalized) + abs(macd_normalized)) / 2))
            }
        except Exception as e:
            logger.warning(f"Erro nos indicadores: {e}")
            return {
                "rsi": 0.0,
                "macd": 0.0,
                "volume_intensity": 0.0,
                "momentum_quality": 0.0
            }

    def _calculate_signal_quality(self, analysis: Dict) -> float:
        """Cálculo de qualidade simplificado"""
        try:
            factors = [
                analysis['price_action'].get('trend_strength', 0) * 0.4,
                analysis['chart_patterns'].get('consolidation_level', 0) * 0.3,
                analysis['market_structure'].get('structure_quality', 0) * 0.3
            ]
            
            return min(1.0, max(0.0, sum(factors)))
        except Exception:
            return 0.5

    def _make_intelligent_decision(self, analysis: Dict, timeframe: str) -> Dict[str, Any]:
        """Tomada de decisão estável e conservadora"""
        try:
            price_action = analysis['price_action']
            chart_patterns = analysis['chart_patterns']
            market_structure = analysis['market_structure']
            indicators = analysis['indicators']
            
            # Sistema de pontuação simples
            score_components = []
            
            # 1. Tendência (40%)
            trend_score = price_action['trend_direction'] * price_action['trend_strength']
            score_components.append(trend_score * 0.4)
            
            # 2. Momentum (30%)
            momentum_score = price_action['momentum'] + indicators['rsi'] * 0.5
            score_components.append(momentum_score * 0.3)
            
            # 3. Estrutura (30%)
            structure_score = market_structure['market_trend'] * market_structure['movement_strength']
            score_components.append(structure_score * 0.3)
            
            total_score = sum(score_components)
            
            # Confiança base
            base_confidence = (
                price_action['trend_strength'] * 0.4 +
                chart_patterns['consolidation_level'] * 0.3 +
                market_structure['structure_quality'] * 0.3
            )
            
            # Decisão conservadora
            if total_score > 0.2:
                direction = "buy"
                confidence = 0.65 + (base_confidence * 0.3)
                reasoning = "📈 Tendência de alta identificada"
            elif total_score < -0.2:
                direction = "sell"
                confidence = 0.65 + (base_confidence * 0.3)
                reasoning = "📉 Tendência de baixa identificada"
            elif total_score > 0.05:
                direction = "buy"
                confidence = 0.58 + (base_confidence * 0.2)
                reasoning = "↗️ Viés de alta detectado"
            elif total_score < -0.05:
                direction = "sell"
                confidence = 0.58 + (base_confidence * 0.2)
                reasoning = "↘️ Viés de baixa detectado"
            else:
                direction = "hold"
                confidence = 0.55
                reasoning = "⚡ Mercado em equilíbrio"
            
            return {
                "direction": direction,
                "confidence": min(0.90, max(self.min_confidence_threshold, confidence)),
                "reasoning": reasoning,
                "total_score": total_score
            }
        except Exception as e:
            logger.warning(f"Erro na decisão: {e}")
            return {
                "direction": "hold",
                "confidence": 0.55,
                "reasoning": "🔄 Análise em andamento",
                "total_score": 0.0
            }

    def _get_entry_timeframe(self, user_timeframe: str) -> Dict[str, str]:
        """Calcula horário de entrada"""
        now = datetime.datetime.now()
        
        if user_timeframe == '1m':
            entry_time = (now + datetime.timedelta(minutes=1)).strftime("%H:%M")
            timeframe_str = "Próximo minuto"
        else:
            minutes_to_add = 5 - (now.minute % 5)
            if minutes_to_add == 0:
                minutes_to_add = 5
            entry_time = (now + datetime.timedelta(minutes=minutes_to_add)).strftime("%H:%M")
            timeframe_str = "Próximo candle de 5min"
        
        return {
            "current_time": now.strftime("%H:%M:%S"),
            "entry_time": entry_time,
            "timeframe": timeframe_str
        }

    def analyze(self, blob: bytes, timeframe: str = '1m') -> Dict[str, Any]:
        """Análise principal - VERSÃO ESTÁVEL"""
        
        # Verifica cache primeiro
        cached_analysis = self.cache.get(blob, timeframe)
        if cached_analysis:
            cached_analysis['cached'] = True
            return cached_analysis
        
        try:
            # Processamento básico
            image = self._load_image(blob)
            self._validate_chart_image(image)
            
            img_array = self._preprocess_image(image, timeframe)
            price_data = self._extract_price_data(img_array)
            
            # Análises
            price_action = self._analyze_price_action(price_data, timeframe)
            chart_patterns = self._analyze_chart_patterns(price_data)
            market_structure = self._analyze_market_structure(price_data, timeframe)
            indicators = self._calculate_advanced_indicators(price_data)
            
            analysis_data = {
                'price_action': price_action,
                'chart_patterns': chart_patterns,
                'market_structure': market_structure,
                'indicators': indicators
            }
            
            signal_quality = self._calculate_signal_quality(analysis_data)
            decision = self._make_intelligent_decision(analysis_data, timeframe)
            time_info = self._get_entry_timeframe(timeframe)
            
            # Determina qualidade
            if signal_quality > 0.7:
                analysis_grade = "high"
            elif signal_quality > 0.5:
                analysis_grade = "medium" 
            else:
                analysis_grade = "low"
            
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
                "signal_quality": float(signal_quality),
                "analysis_grade": analysis_grade,
                "metrics": {
                    "analysis_score": float(decision["total_score"]),
                    "trend_strength": price_action["trend_strength"],
                    "momentum": price_action["momentum"],
                    "rsi": indicators["rsi"],
                    "macd": indicators["macd"],
                    "support_levels": chart_patterns["support_levels"],
                    "resistance_levels": chart_patterns["resistance_levels"],
                    "volatility": price_action["volatility"]
                },
                "reasoning": decision["reasoning"]
            }
            
            # Salva no cache
            self.cache.set(blob, timeframe, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            # Retorna resultado de fallback em vez de levantar exceção
            return {
                "direction": "hold",
                "final_confidence": 0.55,
                "entry_signal": f"⚠️ Análise Básica - {str(e)}",
                "entry_time": "Aguardando",
                "timeframe": "Indefinido",
                "analysis_time": datetime.datetime.now().strftime("%H:%M:%S"),
                "user_timeframe": timeframe,
                "cached": False,
                "signal_quality": 0.3,
                "analysis_grade": "low",
                "metrics": {
                    "analysis_score": 0.0,
                    "trend_strength": 0.0,
                    "momentum": 0.0,
                    "rsi": 0.0,
                    "macd": 0.0,
                    "support_levels": 0,
                    "resistance_levels": 0,
                    "volatility": 0.0
                },
                "reasoning": "Análise básica devido a erro no processamento"
            }

# =========================
#  APLICAÇÃO FLASK ESTÁVEL
# =========================
app = Flask(__name__)
analyzer = IntelligentAnalyzer()

# Configurações para produção
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['JSON_SORT_KEYS'] = False

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Signal Pro - ANÁLISE ESTÁVEL</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: linear-gradient(135deg, #0b1220 0%, #1a1f38 100%); 
            color: #e9eef2; 
            font-family: 'Segoe UI', system-ui, sans-serif;
            min-height: 100vh; 
            padding: 20px;
        }
        .container {
            max-width: 500px; 
            margin: 0 auto;
            background: rgba(15, 22, 39, 0.95); 
            border-radius: 20px;
            padding: 25px; 
            border: 2px solid #3a86ff;
            box-shadow: 0 10px 30px rgba(58, 134, 255, 0.2);
        }
        .header { 
            text-align: center; 
            margin-bottom: 20px; 
        }
        .title {
            font-size: 24px; 
            font-weight: 800; 
            margin-bottom: 5px;
            background: linear-gradient(90deg, #3a86ff, #00ff88);
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;
        }
        .subtitle { 
            color: #9db0d1; 
            font-size: 13px; 
            margin-bottom: 10px; 
        }
        
        .upload-area {
            border: 2px dashed #3a86ff; 
            border-radius: 15px;
            padding: 30px 15px; 
            text-align: center;
            background: rgba(58, 134, 255, 0.05); 
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #00ff88;
            background: rgba(0, 255, 136, 0.05);
        }
        .file-input {
            margin: 15px 0; 
            padding: 12px;
            background: rgba(42, 53, 82, 0.3); 
            border: 1px solid #3a86ff;
            border-radius: 8px; 
            color: white; 
            width: 100%; 
            cursor: pointer;
        }
        
        .timeframe-selector { 
            display: flex; 
            gap: 10px; 
            margin: 15px 0; 
        }
        .timeframe-btn {
            flex: 1; 
            padding: 12px; 
            border: 2px solid #3a86ff;
            background: rgba(58, 134, 255, 0.1); 
            color: #9db0d1;
            border-radius: 10px; 
            cursor: pointer; 
            text-align: center;
            font-weight: 600; 
            transition: all 0.3s ease;
        }
        .timeframe-btn.active {
            background: linear-gradient(135deg, #3a86ff 0%, #2a76ef 100%);
            color: white; 
            border-color: #2a76ef;
        }
        
        .analyze-btn {
            background: linear-gradient(135deg, #3a86ff 0%, #2a76ef 100%);
            color: white; 
            border: none; 
            border-radius: 10px; 
            padding: 16px;
            font-size: 16px; 
            font-weight: 700; 
            cursor: pointer; 
            width: 100%;
            transition: all 0.3s ease;
        }
        .analyze-btn:hover { 
            background: linear-gradient(135deg, #2a76ef 0%, #1a66df 100%);
            transform: translateY(-2px);
        }
        .analyze-btn:disabled { 
            background: #2a3552; 
            transform: none; 
            cursor: not-allowed;
        }
        
        .result { 
            display: none; 
            background: rgba(14, 21, 36, 0.9);
            border-radius: 15px; 
            padding: 20px; 
            margin-top: 20px;
            border: 1px solid #223152;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .signal-buy { color: #00ff88; }
        .signal-sell { color: #ff4444; }
        .signal-hold { color: #ffaa00; }
        
        .signal-text {
            font-weight: 800; 
            font-size: 22px; 
            text-align: center; 
            margin-bottom: 10px;
        }
        
        .time-info {
            background: rgba(42, 53, 82, 0.5); 
            border-radius: 8px;
            padding: 12px; 
            margin: 10px 0; 
            text-align: center;
        }
        .time-item {
            margin: 5px 0; 
            display: flex; 
            justify-content: space-between;
            align-items: center;
        }
        .time-label { color: #9db0d1; font-size: 13px; }
        .time-value { color: #00ff88; font-weight: 600; font-size: 14px; }
        
        .confidence {
            font-size: 16px; 
            text-align: center; 
            margin: 10px 0;
            color: #9db0d1;
        }
        .reasoning {
            text-align: center; 
            margin: 12px 0; 
            color: #3a86ff;
            font-weight: 600; 
            font-size: 14px;
        }
        
        .quality-indicator {
            text-align: center; 
            margin: 10px 0; 
            padding: 8px;
            border-radius: 8px; 
            font-weight: 700; 
            font-size: 13px;
        }
        .quality-high { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; }
        .quality-medium { background: rgba(255, 165, 0, 0.1); color: #ffa500; border: 1px solid #ffa500; }
        .quality-low { background: rgba(255, 68, 68, 0.1); color: #ff4444; border: 1px solid #ff4444; }
        
        .metrics {
            margin-top: 15px; 
            font-size: 13px; 
            color: #9db0d1;
            background: rgba(42, 53, 82, 0.3); 
            padding: 15px;
            border-radius: 8px;
        }
        .metric-item {
            margin: 6px 0; 
            display: flex; 
            justify-content: space-between;
            align-items: center;
        }
        .metric-value {
            font-weight: 600; 
            color: #e9eef2;
        }
        
        .error-message {
            background: rgba(255, 68, 68, 0.1); 
            border: 1px solid #ff4444;
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0;
            color: #ff8888; 
            text-align: center;
        }
        
        .loading {
            text-align: center; 
            color: #3a86ff; 
            font-size: 14px;
        }
        
        .cache-badge {
            background: linear-gradient(135deg, #ffa500, #ff6b6b);
            color: white; 
            padding: 4px 8px; 
            border-radius: 12px;
            font-size: 10px; 
            font-weight: 700; 
            margin-left: 8px;
        }
        
        .progress-bar {
            width: 100%; 
            height: 4px; 
            background: #2a3552;
            border-radius: 2px; 
            margin: 12px 0; 
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
        <div class="header">
            <div class="title">🧠 IA SIGNAL PRO - ESTÁVEL</div>
            <div class="subtitle">SISTEMA INTELIGENTE DE ANÁLISE TÉCNICA</div>
        </div>
        
        <div class="timeframe-selector">
            <button class="timeframe-btn active" data-timeframe="1m">⏱️ 1 MINUTO</button>
            <button class="timeframe-btn" data-timeframe="5m">⏱️ 5 MINUTOS</button>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div style="font-size: 15px; margin-bottom: 8px;">
                📊 CLIQUE OU ARRASTE A IMAGEM DO GRÁFICO
            </div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
        </div>
        
        <button class="analyze-btn" id="analyzeBtn">🔍 ANALISAR GRÁFICO</button>
        
        <div class="result" id="result">
            <div id="signalText" class="signal-text"></div>
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
            <div id="qualityIndicator" class="quality-indicator"></div>
            
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="metrics" id="metricsText"></div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('fileInput');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const uploadArea = document.getElementById('uploadArea');
            const result = document.getElementById('result');
            const signalText = document.getElementById('signalText');
            const errorMessage = document.getElementById('errorMessage');
            const analysisTime = document.getElementById('analysisTime');
            const entryTime = document.getElementById('entryTime');
            const timeframeEl = document.getElementById('timeframe');
            const reasoningText = document.getElementById('reasoningText');
            const confidenceText = document.getElementById('confidenceText');
            const qualityIndicator = document.getElementById('qualityIndicator');
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
                });
            });

            // Upload de arquivo
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#00ff88';
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = '#3a86ff';
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#3a86ff';
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    handleFileSelect();
                }
            });

            fileInput.addEventListener('change', handleFileSelect);

            function handleFileSelect() {
                selectedFile = fileInput.files[0];
                if (selectedFile) {
                    analyzeBtn.textContent = `✅ PRONTO PARA ANÁLISE ${currentTimeframe.toUpperCase()}`;
                    analyzeBtn.disabled = false;
                }
            }

            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) {
                    alert('📸 Selecione uma imagem do gráfico primeiro!');
                    return;
                }

                analyzeBtn.disabled = true;
                analyzeBtn.textContent = `🧠 ANALISANDO ${currentTimeframe.toUpperCase()}...`;
                result.style.display = 'block';
                errorMessage.style.display = 'none';
                
                signalText.className = 'signal-text';
                signalText.textContent = 'Analisando padrões do gráfico...';
                qualityIndicator.textContent = '';
                
                const now = new Date();
                analysisTime.textContent = now.toLocaleTimeString('pt-BR');
                
                // Calcula horário de entrada
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
                    timeframeEl.textContent = `Próximo candle de 5min`;
                }
                
                entryTime.textContent = entryTimeValue;
                reasoningText.textContent = 'Processando análise técnica...';
                confidenceText.textContent = '';
                progressFill.style.width = '20%';
                
                metricsText.innerHTML = '<div class="loading">Iniciando análise...</div>';

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
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    progressFill.style.width = '100%';
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    displayResults(data);
                    
                } catch (error) {
                    console.error('Erro:', error);
                    errorMessage.style.display = 'block';
                    errorMessage.textContent = `❌ Erro na análise: ${error.message}`;
                    signalText.textContent = '❌ Análise Falhou';
                    metricsText.innerHTML = '<div class="loading">Erro no processamento</div>';
                } finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = `🔍 ANALISAR ${currentTimeframe.toUpperCase()} NOVAMENTE`;
                }
            });

            function displayResults(data) {
                const direction = data.direction;
                const confidence = (data.final_confidence * 100).toFixed(1);
                const cached = data.cached || false;
                const quality = data.analysis_grade || 'medium';
                
                // Define classe e texto do sinal
                signalText.className = `signal-text signal-${direction}`;
                let directionText = '';
                switch(direction) {
                    case 'buy': directionText = '🎯 COMPRAR'; break;
                    case 'sell': directionText = '🎯 VENDER'; break;
                    default: directionText = '⏸️ AGUARDAR';
                }
                signalText.innerHTML = `${directionText} ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                
                // Atualiza informações
                analysisTime.textContent = data.analysis_time || '--:--:--';
                entryTime.textContent = data.entry_time || '--:--';
                timeframeEl.textContent = data.timeframe || 'Próximo minuto';
                
                reasoningText.textContent = data.reasoning;
                confidenceText.textContent = `Confiança: ${confidence}%`;
                
                // Indicador de qualidade
                qualityIndicator.className = `quality-indicator quality-${quality}`;
                if (quality === 'high') {
                    qualityIndicator.textContent = '✅ ALTA QUALIDADE - Sinal confiável';
                } else if (quality === 'medium') {
                    qualityIndicator.textContent = '⚠️ QUALIDADE MÉDIA - Use com atenção';
                } else {
                    qualityIndicator.textContent = '🔍 QUALIDADE BAIXA - Use com cautela';
                }
                
                // Métricas detalhadas
                const metrics = data.metrics || {};
                let metricsHtml = '<div style="margin-bottom: 10px; text-align: center; font-weight: 600;">📊 ANÁLISE DETALHADA</div>';
                
                const metricItems = [
                    ['Score da Análise', metrics.analysis_score?.toFixed(3)],
                    ['Força da Tendência', (metrics.trend_strength * 100)?.toFixed(1) + '%'],
                    ['Momentum', metrics.momentum?.toFixed(3)],
                    ['RSI', metrics.rsi?.toFixed(3)],
                    ['MACD', metrics.macd?.toFixed(3)],
                    ['Suportes', metrics.support_levels || 0],
                    ['Resistências', metrics.resistance_levels || 0],
                    ['Volatilidade', (metrics.volatility * 100)?.toFixed(1) + '%']
                ];
                
                metricItems.forEach(([label, value]) => {
                    metricsHtml += `
                        <div class="metric-item">
                            <span>${label}:</span>
                            <span class="metric-value">${value}</span>
                        </div>
                    `;
                });
                
                metricsText.innerHTML = metricsHtml;
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Página principal"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_photo():
    """Endpoint de análise de imagem"""
    try:
        logger.info("Recebendo requisição de análise")
        
        if 'image' not in request.files:
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
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'ok': False, 'error': 'Imagem muito grande (máximo 10MB)'}), 400
        
        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({'ok': False, 'error': 'Arquivo vazio'}), 400
        
        # Análise
        logger.info(f"Iniciando análise para timeframe: {timeframe}")
        analysis = analyzer.analyze(image_bytes, timeframe)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Erro no endpoint /analyze: {e}")
        return jsonify({
            'ok': False,
            'error': f'Erro interno: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check para monitoramento"""
    return jsonify({
        'status': 'healthy', 
        'service': 'IA Signal Pro',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '2.0.0-stable'
    })

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Limpa o cache de análises"""
    try:
        cache_dir = "analysis_cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, file))
            return jsonify({'ok': True, 'message': 'Cache limpo com sucesso!'})
        return jsonify({'ok': True, 'message': 'Cache já está vazio!'})
    except Exception as e:
        logger.error(f"Erro ao limpar cache: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500

# Handler de erro global
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Erro interno do servidor'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint não encontrado'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 IA Signal Pro - VERSÃO ESTÁVEL iniciando na porta {port}")
    logger.info(f"📊 Sistema: Análise Inteligente com Tratamento de Erros")
    logger.info(f"⏰ Timeframes: 1min e 5min com cache inteligente")
    logger.info(f"🛡️ Status: ESTÁVEL E FUNCIONAL")
    
    # Use waitress para produção em vez do servidor de desenvolvimento do Flask
    if os.environ.get('PRODUCTION', 'False').lower() == 'true':
        from waitress import serve
        serve(app, host='0.0.0.0', port=port)
    else:
        app.run(host='0.0.0.0', port=port, debug=debug)
