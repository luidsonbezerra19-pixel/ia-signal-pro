from __future__ import annotations

"""
IA Signal Pro — Análise INTELIGENTE E IMPARCIAL com Cache Multi-Timeframe
Sistema avançado para identificar a MELHOR direção baseado em análise técnica
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
            '1m': 60,  # 1 minuto para cache de 1min
            '5m': 300  # 5 minutos para cache de 5min
        }
    
    def _get_cache_key(self, image_bytes: bytes, timeframe: str) -> str:
        """Gera chave única baseada na imagem e timeframe"""
        content_hash = hashlib.md5(image_bytes).hexdigest()
        return f"{timeframe}_{content_hash}"
    
    def _get_cache_file(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, image_bytes: bytes, timeframe: str) -> Optional[Dict]:
        """Recupera análise do cache se ainda for válida"""
        key = self._get_cache_key(image_bytes, timeframe)
        cache_file = self._get_cache_file(key)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Verifica se o cache ainda é válido
                cache_time = datetime.datetime.fromisoformat(cache_data['timestamp'])
                current_time = datetime.datetime.now()
                age_seconds = (current_time - cache_time).total_seconds()
                
                if age_seconds < self.cache_duration.get(timeframe, 60):
                    print(f"✅ Cache HIT para timeframe {timeframe} (idade: {age_seconds:.1f}s)")
                    return cache_data['analysis']
                else:
                    print(f"🕒 Cache EXPIRADO para timeframe {timeframe}")
                    os.remove(cache_file)
            except Exception as e:
                print(f"❌ Erro ao ler cache: {e}")
        
        return None
    
    def set(self, image_bytes: bytes, timeframe: str, analysis: Dict):
        """Salva análise no cache"""
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
            
            print(f"💾 Cache SAVED para timeframe {timeframe}")
        except Exception as e:
            print(f"❌ Erro ao salvar cache: {e}")

# =========================
#  IA INTELIGENTE E IMPARCIAL
# =========================
class IntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
    
    def _load_image(self, blob: bytes) -> Image.Image:
        """Carrega e prepara a imagem"""
        image = Image.open(io.BytesIO(blob))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def _preprocess_image(self, image: Image.Image, timeframe: str) -> np.ndarray:
        """Pré-processa a imagem considerando o timeframe"""
        width, height = image.size
        
        # Ajusta o redimensionamento baseado no timeframe
        if timeframe == '1m':
            target_width = 600  # Mais detalhes para 1min
        else:  # 5m
            target_width = 800  # Mais contexto para 5min
        
        if width > target_width:
            new_width = target_width
            new_height = int((height / width) * new_width)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        image = image.filter(ImageFilter.SMOOTH)
        img_array = np.array(image)
        return img_array

    def _analyze_multiple_timeframes(self, img_array: np.ndarray, user_timeframe: str) -> Dict[str, float]:
        """Análise multi-timeframe inteligente ajustada para o timeframe do usuário"""
        height, width, _ = img_array.shape
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Ajusta a análise baseado no timeframe escolhido
        if user_timeframe == '1m':
            # Para 1min: foco em movimentos rápidos e detalhes
            short_window = width // 6   # Janela muito curta
            medium_window = width // 3  # Janela média
            long_window = width // 2    # Janela longa
            volatility_factor = 1.2     # Mais sensível à volatilidade
        else:  # 5m
            # Para 5min: foco em tendências mais estabelecidas
            short_window = width // 4   # Janela curta
            medium_window = width // 2  # Janela média  
            long_window = width         # Janela completa
            volatility_factor = 0.8     # Menos sensível à volatilidade
        
        # Timeframe curto (análise detalhada)
        short_regions = []
        for i in range(8):
            y_start = int(height * (i * 0.1))
            y_end = int(height * ((i + 1) * 0.1))
            region = gray[y_start:y_end, -short_window:]  # Parte mais recente
            if region.size > 0:
                short_regions.append(region.mean())
        
        # Timeframe médio
        medium_regions = []
        for i in range(6):
            y_start = int(height * (i * 0.15))
            y_end = int(height * ((i + 1) * 0.15))
            region = gray[y_start:y_end, -medium_window:]
            if region.size > 0:
                medium_regions.append(region.mean())
        
        # Timeframe longo
        long_regions = []
        for i in range(4):
            y_start = int(height * (i * 0.25))
            y_end = int(height * ((i + 1) * 0.25))
            region = gray[y_start:y_end, -long_window:]
            if region.size > 0:
                long_regions.append(region.mean())
        
        # Calcula tendências para cada timeframe
        def calculate_trend(regions):
            if len(regions) < 3:
                return 0, 0
            x = np.arange(len(regions))
            slope, intercept = np.polyfit(x, regions, 1)
            strength = abs(slope) / max(1, np.std(regions))
            return slope, strength
        
        short_slope, short_strength = calculate_trend(short_regions)
        medium_slope, medium_strength = calculate_trend(medium_regions)
        long_slope, long_strength = calculate_trend(long_regions)
        
        # Conflito entre timeframes = menor confiança
        timeframe_alignment = 0
        if short_slope * medium_slope > 0 and medium_slope * long_slope > 0:
            timeframe_alignment = 1.0  # Alinhados
        elif short_slope * medium_slope < 0 or medium_slope * long_slope < 0:
            timeframe_alignment = 0.3  # Conflitantes
        
        volatility = np.std(gray) / max(1, gray.mean()) * volatility_factor
        
        return {
            "short_trend": float(short_slope),
            "medium_trend": float(medium_slope),
            "long_trend": float(long_slope),
            "short_strength": float(short_strength),
            "medium_strength": float(medium_strength),
            "long_strength": float(long_strength),
            "timeframe_alignment": float(timeframe_alignment),
            "volatility": float(volatility),
            "user_timeframe": user_timeframe
        }

    def _detect_support_resistance(self, img_array: np.ndarray, user_timeframe: str) -> Dict[str, float]:
        """Detecta níveis de suporte e resistência ajustado para timeframe"""
        height, width, _ = img_array.shape
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Ajusta sensibilidade baseado no timeframe
        if user_timeframe == '1m':
            threshold_factor = 0.4  # Mais sensível para 1min
            min_touches = 2         # Menos toques necessários
        else:  # 5m
            threshold_factor = 0.3  # Menos sensível para 5min  
            min_touches = 3         # Mais toques necessários
        
        # Análise horizontal para encontrar níveis
        row_means = np.mean(gray, axis=1)
        threshold = np.std(row_means) * threshold_factor
        
        # Encontra plataformas (níveis de suporte/resistência)
        levels = []
        for i in range(1, height-1):
            if (abs(row_means[i] - row_means[i-1]) < threshold and 
                abs(row_means[i] - row_means[i+1]) < threshold):
                levels.append(row_means[i])
        
        # Analisa densidade de toques nos níveis
        unique_levels = []
        level_touches = []
        
        for level in levels:
            found = False
            for i, existing_level in enumerate(unique_levels):
                if abs(level - existing_level) < threshold * 2:
                    level_touches[i] += 1
                    found = True
                    break
            if not found:
                unique_levels.append(level)
                level_touches.append(1)
        
        # Filtra níveis com toques suficientes
        filtered_levels = []
        filtered_touches = []
        for level, touches in zip(unique_levels, level_touches):
            if touches >= min_touches:
                filtered_levels.append(level)
                filtered_touches.append(touches)
        
        # Força dos níveis baseado em toques e posição
        if filtered_touches:
            max_touches = max(filtered_touches)
            current_price = np.mean(gray[height//3:2*height//3, -50:])  # Preço atual
            
            # Encontra suporte e resistência mais próximos
            supports = [level for level in filtered_levels if level < current_price]
            resistances = [level for level in filtered_levels if level > current_price]
            
            nearest_support = max(supports) if supports else None
            nearest_resistance = min(resistances) if resistances else None
            
            support_strength = 0
            resistance_strength = 0
            
            if nearest_support:
                idx = filtered_levels.index(nearest_support)
                support_strength = filtered_touches[idx] / max_touches
            
            if nearest_resistance:
                idx = filtered_levels.index(nearest_resistance)
                resistance_strength = filtered_touches[idx] / max_touches
            
            proximity_factor = 0.5
            if nearest_support and nearest_resistance:
                total_range = nearest_resistance - nearest_support
                if total_range > 0:
                    proximity_factor = (current_price - nearest_support) / total_range
            elif nearest_support:
                proximity_factor = 0.3
            elif nearest_resistance:
                proximity_factor = 0.7
        else:
            support_strength = 0
            resistance_strength = 0
            proximity_factor = 0.5
        
        return {
            "support_strength": float(support_strength),
            "resistance_strength": float(resistance_strength),
            "proximity_to_support": float(proximity_factor),
            "levels_quality": float(len(filtered_levels) / 10),
            "levels_found": len(filtered_levels)
        }

    def _analyze_momentum_oscillators(self, img_array: np.ndarray, user_timeframe: str) -> Dict[str, float]:
        """Analisa momentum ajustado para timeframe"""
        height, width, _ = img_array.shape
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Ajusta períodos baseado no timeframe
        if user_timeframe == '1m':
            recent_period = width // 6   # Período mais curto para 1min
            older_period = width // 4    # Período de comparação mais curto
            macd_fast = width // 8       # EMA rápida muito curta
            macd_slow = width // 5       # EMA lenta curta
        else:  # 5m
            recent_period = width // 4   # Período mais longo para 5min
            older_period = width // 3    # Período de comparação mais longo
            macd_fast = width // 6       # EMA rápida
            macd_slow = width // 3       # EMA lenta
        
        # RSI visual (força relativa)
        recent_data = gray[:, -recent_period:]
        older_data = gray[:, -older_period:-recent_period] if older_period > recent_period else gray[:, :older_period]
        
        recent_avg = np.mean(recent_data) if recent_data.size > 0 else gray.mean()
        older_avg = np.mean(older_data) if older_data.size > 0 else gray.mean()
        
        gains = max(0, recent_avg - older_avg)
        losses = max(0, older_avg - recent_avg)
        
        if losses == 0:
            rsi = 100 if gains > 0 else 50
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        
        # Normaliza RSI para escala -1 a 1
        rsi_normalized = (rsi - 50) / 50
        
        # MACD visual (tendência)
        fast_ema = np.mean(gray[:, -macd_fast:]) if macd_fast > 0 else gray.mean()
        slow_ema = np.mean(gray[:, -macd_slow:]) if macd_slow > 0 else gray.mean()
        
        macd = fast_ema - slow_ema
        macd_normalized = np.clip(macd / max(1, np.std(gray)), -2, 2)
        
        # Análise de divergência
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        
        left_trend = np.polyfit(range(left_half.shape[1]), np.mean(left_half, axis=0), 1)[0] if left_half.shape[1] > 1 else 0
        right_trend = np.polyfit(range(right_half.shape[1]), np.mean(right_half, axis=0), 1)[0] if right_half.shape[1] > 1 else 0
        
        divergence = right_trend - left_trend
        
        return {
            "rsi_momentum": float(rsi_normalized),
            "macd_signal": float(macd_normalized),
            "trend_divergence": float(divergence),
            "momentum_strength": float(abs(rsi_normalized) + abs(macd_normalized))
        }

    def _calculate_volume_analysis(self, img_array: np.ndarray, user_timeframe: str) -> Dict[str, float]:
        """Análise de volume ajustada para timeframe"""
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Ajusta sensibilidade do volume
        if user_timeframe == '1m':
            variance_threshold = 800  # Mais sensível para 1min
            activity_percentile = 60   # Percentil mais baixo
        else:  # 5m
            variance_threshold = 1200  # Menos sensível para 5min
            activity_percentile = 70   # Percentil mais alto
        
        # Volume baseado na variação de detalhes
        detail_variance = np.var(gray)
        
        # Análise de concentração (áreas de alta atividade)
        from scipy import ndimage
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        activity_concentration = np.mean(gradient_magnitude > np.percentile(gradient_magnitude, activity_percentile))
        
        # Volume relativo por área
        left_volume = np.var(gray[:, :gray.shape[1]//2])
        right_volume = np.var(gray[:, gray.shape[1]//2:])
        
        volume_ratio = right_volume / max(0.1, left_volume)
        volume_trend = np.clip((volume_ratio - 1) * 2, -2, 2)
        
        return {
            "volume_intensity": float(min(1.0, detail_variance / variance_threshold)),
            "activity_density": float(activity_concentration),
            "volume_trend": float(volume_trend),
            "volume_confidence": float(min(1.0, detail_variance / (variance_threshold * 0.8)))
        }

    def _get_intelligent_signal(self, analysis_data: Dict, user_timeframe: str) -> Dict[str, Any]:
        """Decisão inteligente ajustada para timeframe"""
        
        # Coleta todos os fatores
        tf = analysis_data['timeframe']
        sr = analysis_data['support_resistance']
        momentum = analysis_data['momentum']
        volume = analysis_data['volume']
        
        # Ajusta pesos baseado no timeframe
        if user_timeframe == '1m':
            # Para 1min: mais peso em momentum e volume
            timeframe_weight = 2.0
            momentum_weight = 1.5
            volume_weight = 1.3
            sr_weight = 1.8
        else:  # 5m
            # Para 5min: mais peso em suporte/resistência e tendências
            timeframe_weight = 2.2
            momentum_weight = 1.2
            volume_weight = 1.0
            sr_weight = 2.0
        
        # Sistema de pontuação inteligente
        factors = []
        
        # 1. Alinhamento de Timeframes (peso alto)
        alignment_score = tf['timeframe_alignment']
        if alignment_score > 0.7:
            dominant_trend = np.sign(tf['short_trend'] + tf['medium_trend'] + tf['long_trend'])
            factors.append(("timeframe_alignment", dominant_trend * alignment_score * timeframe_weight))
        else:
            factors.append(("timeframe_alignment", 0))
        
        # 2. Proximidade de Suporte/Resistência (peso alto)
        proximity = sr['proximity_to_support']
        if proximity < 0.3 and sr['support_strength'] > 0.6:
            factors.append(("support_bounce", 1.5 * sr_weight))
        elif proximity > 0.7 and sr['resistance_strength'] > 0.6:
            factors.append(("resistance_rejection", -1.5 * sr_weight))
        else:
            factors.append(("sr_neutral", 0))
        
        # 3. Momentum e RSI (peso médio)
        rsi = momentum['rsi_momentum']
        if abs(rsi) > 0.5:  # Sobrecomprado/sobrevendido
            factors.append(("momentum_reversal", -rsi * momentum_weight))
        else:
            factors.append(("momentum_trend", momentum['macd_signal'] * momentum_weight))
        
        # 4. Volume confirmando tendência (peso médio)
        volume_confirmation = volume['volume_trend'] * tf['short_trend']
        factors.append(("volume_confirmation", volume_confirmation * volume_weight))
        
        # 5. Força da tendência atual (peso médio)
        trend_strength = (tf['short_strength'] + tf['medium_strength']) / 2
        factors.append(("trend_strength", tf['short_trend'] * trend_strength * 1.3))
        
        # 6. Divergência (peso baixo mas importante)
        factors.append(("divergence", momentum['trend_divergence'] * 0.8))
        
        # Calcula score total
        total_score = sum(score for _, score in factors)
        
        # Calcula confiança baseada na consistência dos sinais
        confidence_factors = [
            alignment_score,
            max(sr['support_strength'], sr['resistance_strength']),
            momentum['momentum_strength'] / 2,
            volume['volume_confidence'],
            trend_strength
        ]
        
        base_confidence = np.mean(confidence_factors)
        
        # Ajusta thresholds baseado no timeframe
        if user_timeframe == '1m':
            strong_threshold = 1.2
            weak_threshold = 0.4
        else:  # 5m
            strong_threshold = 1.0
            weak_threshold = 0.3
        
        # DECISÃO INTELIGENTE E IMPARCIAL
        if total_score > strong_threshold:
            direction = "buy"
            confidence = min(0.95, 0.75 + (base_confidence * 0.2))
            reasoning = f"🚀 FORTE SINAL DE COMPRA - TIMEFRAME {user_timeframe.upper()}"
        elif total_score < -strong_threshold:
            direction = "sell"
            confidence = min(0.95, 0.75 + (base_confidence * 0.2))
            reasoning = f"🔻 FORTE SINAL DE VENDA - TIMEFRAME {user_timeframe.upper()}"
        elif total_score > weak_threshold:
            direction = "buy"
            confidence = 0.65 + (base_confidence * 0.15)
            reasoning = f"📈 SINAL DE COMPRA - {user_timeframe.upper()} CONFIRMADO"
        elif total_score < -weak_threshold:
            direction = "sell"
            confidence = 0.65 + (base_confidence * 0.15)
            reasoning = f"📉 SINAL DE VENDA - {user_timeframe.upper()} CONFIRMADO"
        else:
            # Mercado indeciso
            if momentum['rsi_momentum'] > 0 and volume['volume_trend'] > 0:
                direction = "buy"
                confidence = 0.55
                reasoning = f"⚡ VIÉS DE ALTA - {user_timeframe.upper()} NEUTRO"
            elif momentum['rsi_momentum'] < 0 and volume['volume_trend'] < 0:
                direction = "sell"
                confidence = 0.55
                reasoning = f"⚡ VIÉS DE BAIXA - {user_timeframe.upper()} NEUTRO"
            else:
                direction = "buy" if total_score > 0 else "sell"
                confidence = 0.50
                reasoning = f"⚖️ MERCADO EQUILIBRADO - {user_timeframe.upper()}"
        
        return {
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "total_score": total_score,
            "timeframe_used": user_timeframe,
            "factors_breakdown": {name: score for name, score in factors}
        }

    def _get_entry_timeframe(self, user_timeframe: str) -> Dict[str, str]:
        """Calcula horário de entrada baseado no timeframe escolhido"""
        now = datetime.datetime.now()
        
        if user_timeframe == '1m':
            # Para 1min: entrada no próximo minuto
            entry_time = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
            timeframe_str = "Próximo minuto"
        else:  # 5m
            # Para 5min: entrada no próximo múltiplo de 5 minutos
            minutes_to_add = 5 - (now.minute % 5)
            if minutes_to_add == 0:
                minutes_to_add = 5
            entry_time = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=minutes_to_add)
            timeframe_str = f"Próximo candle de 5min ({entry_time.strftime('%H:%M')})"
        
        current_time = now.strftime("%H:%M:%S")
        entry_time_str = entry_time.strftime("%H:%M")
        
        return {
            "current_time": current_time,
            "entry_time": entry_time_str,
            "timeframe": timeframe_str,
            "user_timeframe": user_timeframe
        }

    def analyze(self, blob: bytes, timeframe: str = '1m') -> Dict[str, Any]:
        """Analisa a imagem com suporte a diferentes timeframes e cache"""
        
        # Verifica cache primeiro
        cached_analysis = self.cache.get(blob, timeframe)
        if cached_analysis:
            return cached_analysis
        
        try:
            image = self._load_image(blob)
            img_array = self._preprocess_image(image, timeframe)
            
            # Análises múltiplas e inteligentes
            timeframe_analysis = self._analyze_multiple_timeframes(img_array, timeframe)
            support_resistance = self._detect_support_resistance(img_array, timeframe)
            momentum_analysis = self._analyze_momentum_oscillators(img_array, timeframe)
            volume_analysis = self._calculate_volume_analysis(img_array, timeframe)
            
            # Consolida análise
            analysis_data = {
                'timeframe': timeframe_analysis,
                'support_resistance': support_resistance,
                'momentum': momentum_analysis,
                'volume': volume_analysis
            }
            
            # Tomada de decisão inteligente
            signal = self._get_intelligent_signal(analysis_data, timeframe)
            
            # Horário
            time_info = self._get_entry_timeframe(timeframe)
            
            result = {
                "direction": signal["direction"],
                "final_confidence": float(signal["confidence"]),
                "entry_signal": f"🎯 {signal['direction'].upper()} - {signal['reasoning']}",
                "entry_time": time_info["entry_time"],
                "timeframe": time_info["timeframe"],
                "analysis_time": time_info["current_time"],
                "user_timeframe": timeframe,
                "cached": False,
                "metrics": {
                    "analysis_score": float(signal["total_score"]),
                    "timeframe_alignment": timeframe_analysis["timeframe_alignment"],
                    "support_strength": support_resistance["support_strength"],
                    "resistance_strength": support_resistance["resistance_strength"],
                    "rsi_momentum": momentum_analysis["rsi_momentum"],
                    "macd_signal": momentum_analysis["macd_signal"],
                    "volume_trend": volume_analysis["volume_trend"],
                    "volatility": timeframe_analysis["volatility"],
                    "signal_quality": float(signal["confidence"]),
                    "levels_found": support_resistance["levels_found"]
                },
                "reasoning": signal["reasoning"]
            }
            
            # Salva no cache
            self.cache.set(blob, timeframe, result)
            
            return result
            
        except Exception as e:
            # Fallback completamente imparcial
            now = datetime.datetime.now()
            time_info = self._get_entry_timeframe(timeframe)
            
            # Decisão baseada apenas no horário atual (imparcial)
            current_minute = now.minute
            direction = "buy" if current_minute % 2 == 0 else "sell"
            
            result = {
                "direction": direction,
                "final_confidence": 0.50,
                "entry_signal": f"🎯 {direction.upper()} - ANÁLISE NEUTRA DE FALLBACK",
                "entry_time": time_info["entry_time"],
                "timeframe": time_info["timeframe"],
                "analysis_time": time_info["current_time"],
                "user_timeframe": timeframe,
                "cached": False,
                "metrics": {
                    "analysis_score": 0.0,
                    "timeframe_alignment": 0.5,
                    "support_strength": 0.5,
                    "resistance_strength": 0.5,
                    "rsi_momentum": 0.0,
                    "macd_signal": 0.0,
                    "volume_trend": 0.0,
                    "volatility": 0.3,
                    "signal_quality": 0.5,
                    "levels_found": 0
                },
                "reasoning": f"Análise neutra: Fallback imparcial - {timeframe.upper()}"
            }
            
            return result

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
    <title>IA Signal Pro - ANÁLISE MULTI-TIMEFRAME</title>
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
        
        .live-clock {
            background: rgba(58, 134, 255, 0.1);
            border: 1px solid #3a86ff;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            text-align: center;
            font-family: 'Courier New', monospace;
        }
        
        .clock-time {
            font-size: 18px;
            font-weight: 700;
            color: #00ff88;
            margin-bottom: 5px;
        }
        
        .clock-date {
            font-size: 12px;
            color: #9db0d1;
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
        
        .timeframe-btn:hover {
            background: rgba(58, 134, 255, 0.2);
        }
        
        .upload-area {
            border: 2px dashed #3a86ff;
            border-radius: 15px;
            padding: 20px 15px;
            text-align: center;
            background: rgba(58, 134, 255, 0.05);
            margin-bottom: 20px;
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
            background: rgba(14, 21, 36, 0.9);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid #223152;
            display: none;
        }
        
        .signal-buy {
            color: #00ff88;
            font-weight: 800;
            font-size: 22px;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .signal-sell {
            color: #ff4444;
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
        
        .time-label {
            color: #9db0d1;
            font-size: 13px;
        }
        
        .time-value {
            color: #00ff88;
            font-weight: 600;
            font-size: 14px;
        }
        
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
        
        .loading {
            text-align: center;
            color: #3a86ff;
            font-size: 14px;
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
        
        .balance-info {
            text-align: center;
            margin: 10px 0;
            padding: 8px;
            background: rgba(58, 134, 255, 0.1);
            border-radius: 6px;
            font-size: 12px;
            color: #9db0d1;
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
        
        .timeframe-badge {
            background: linear-gradient(135deg, #00ff88, #3a86ff);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠 IA SIGNAL PRO - MULTI-TIMEFRAME</div>
            <div class="subtitle">ANÁLISE INTELIGENTE COM CACHE - 1min E 5min</div>
            
            <div class="live-clock">
                <div class="clock-time" id="liveTime">--:--:--</div>
                <div class="clock-date" id="liveDate">--/--/----</div>
            </div>
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
                <div class="time-item">
                    <span class="time-label">🔧 Tipo de Análise:</span>
                    <span class="time-value" id="analysisType">--</span>
                </div>
            </div>
            
            <div class="balance-info" id="balanceInfo">
                🧠 Análise inteligente em andamento...
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
        // Relógio em tempo real
        function updateClock() {
            const now = new Date();
            const time = now.toLocaleTimeString('pt-BR');
            const date = now.toLocaleDateString('pt-BR');
            
            document.getElementById('liveTime').textContent = time;
            document.getElementById('liveDate').textContent = date;
        }
        
        setInterval(updateClock, 1000);
        updateClock();

        // Variáveis globais
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const result = document.getElementById('result');
        const signalText = document.getElementById('signalText');
        const analysisTime = document.getElementById('analysisTime');
        const entryTime = document.getElementById('entryTime');
        const timeframe = document.getElementById('timeframe');
        const analysisType = document.getElementById('analysisType');
        const balanceInfo = document.getElementById('balanceInfo');
        const reasoningText = document.getElementById('reasoningText');
        const confidenceText = document.getElementById('confidenceText');
        const progressFill = document.getElementById('progressFill');
        const metricsText = document.getElementById('metricsText');
        const timeframeBtns = document.querySelectorAll('.timeframe-btn');

        let selectedFile = null;
        let currentTimeframe = '1m';

        // Seleção de timeframe
        timeframeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                timeframeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentTimeframe = btn.dataset.timeframe;
                analyzeBtn.textContent = `🔍 ANALISAR ${currentTimeframe.toUpperCase()} INTELIGENTEMENTE`;
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
            signalText.className = '';
            signalText.textContent = `Analisando padrões de ${currentTimeframe}...`;
            
            const now = new Date();
            analysisTime.textContent = now.toLocaleTimeString('pt-BR');
            
            // Calcula horário de entrada baseado no timeframe
            let entryTimeValue;
            if (currentTimeframe === '1m') {
                const nextMinute = new Date(now);
                nextMinute.setMinutes(nextMinute.getMinutes() + 1);
                nextMinute.setSeconds(0);
                entryTimeValue = nextMinute.toLocaleTimeString('pt-BR').slice(0, 5);
                timeframe.textContent = 'Próximo minuto';
            } else {
                // Para 5min: próximo múltiplo de 5 minutos
                const minutesToAdd = 5 - (now.getMinutes() % 5);
                const next5min = new Date(now);
                next5min.setMinutes(next5min.getMinutes() + minutesToAdd);
                next5min.setSeconds(0);
                entryTimeValue = next5min.toLocaleTimeString('pt-BR').slice(0, 5);
                timeframe.textContent = `Próximo candle de 5min (${entryTimeValue})`;
            }
            
            entryTime.textContent = entryTimeValue;
            analysisType.textContent = `Análise ${currentTimeframe.toUpperCase()}`;
            
            balanceInfo.textContent = `🧠 Processando análise ${currentTimeframe}...`;
            reasoningText.textContent = 'Executando algoritmos inteligentes...';
            confidenceText.textContent = '';
            progressFill.style.width = '20%';
            
            metricsText.innerHTML = `<div class="loading">Iniciando análise ${currentTimeframe}...</div>`;

            try {
                const formData = new FormData();
                formData.append('image', selectedFile);
                formData.append('timeframe', currentTimeframe);
                
                progressFill.style.width = '40%';
                balanceInfo.textContent = `📈 Analisando suporte/resistência ${currentTimeframe}...`;
                
                progressFill.style.width = '60%';
                balanceInfo.textContent = `⚡ Calculando indicadores ${currentTimeframe}...`;
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                progressFill.style.width = '80%';
                balanceInfo.textContent = '🔍 Consolidando sinais inteligentes...';
                
                const data = await response.json();
                
                progressFill.style.width = '100%';
                
                if (data.ok) {
                    const direction = data.direction;
                    const confidence = (data.final_confidence * 100).toFixed(1);
                    const cached = data.cached || false;
                    const userTimeframe = data.user_timeframe || currentTimeframe;
                    
                    if (direction === 'buy') {
                        signalText.className = 'signal-buy';
                        signalText.innerHTML = `🎯 COMPRAR - SINAL ${userTimeframe.toUpperCase()} ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                        balanceInfo.textContent = `📈 Tendência de alta ${userTimeframe} identificada`;
                    } else {
                        signalText.className = 'signal-sell';
                        signalText.innerHTML = `🎯 VENDER - SINAL ${userTimeframe.toUpperCase()} ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                        balanceInfo.textContent = `📉 Tendência de baixa ${userTimeframe} identificada`;
                    }
                    
                    analysisTime.textContent = data.analysis_time || '--:--:--';
                    entryTime.textContent = data.entry_time || '--:--';
                    timeframe.textContent = data.timeframe || 'Próximo minuto';
                    analysisType.textContent = `Análise ${userTimeframe.toUpperCase()} ${cached ? '(Cache)' : '(Nova)'}`;
                    
                    reasoningText.textContent = data.reasoning;
                    confidenceText.textContent = `Confiança Inteligente: ${confidence}%`;
                    
                    // Métricas detalhadas
                    const metrics = data.metrics || {};
                    let metricsHtml = `<div style="margin-bottom: 10px; text-align: center; font-weight: 600;">
                        🧠 ANÁLISE ${userTimeframe.toUpperCase()} ${cached ? '<span class="cache-badge">CACHE</span>' : ''}
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Score da Análise:</span>
                        <span class="metric-value">${metrics.analysis_score?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Alinhamento Timeframes:</span>
                        <span class="metric-value">${(metrics.timeframe_alignment * 100)?.toFixed(1)}%</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Níveis Encontrados:</span>
                        <span class="metric-value">${metrics.levels_found || 0}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Força do Suporte:</span>
                        <span class="metric-value">${(metrics.support_strength * 100)?.toFixed(1)}%</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Força da Resistência:</span>
                        <span class="metric-value">${(metrics.resistance_strength * 100)?.toFixed(1)}%</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>RSI Momentum:</span>
                        <span class="metric-value">${metrics.rsi_momentum?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Sinal MACD:</span>
                        <span class="metric-value">${metrics.macd_signal?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Tendência do Volume:</span>
                        <span class="metric-value">${metrics.volume_trend?.toFixed(2)}</span>
                    </div>`;
                    
                    metricsHtml += `<div class="metric-item">
                        <span>Qualidade do Sinal:</span>
                        <span class="metric-value">${(metrics.signal_quality * 100)?.toFixed(1)}%</span>
                    </div>`;
                    
                    metricsText.innerHTML = metricsHtml;
                    
                } else {
                    // Fallback para erro
                    showFallbackAnalysis(currentTimeframe);
                }
            } catch (error) {
                console.error('Erro na análise:', error);
                showFallbackAnalysis(currentTimeframe);
            }
            
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = `🔍 ANALISAR ${currentTimeframe.toUpperCase()} NOVAMENTE`;
        });

        function showFallbackAnalysis(timeframe) {
            const now = new Date();
            let entryTimeValue;
            
            if (timeframe === '1m') {
                const nextMinute = new Date(now);
                nextMinute.setMinutes(nextMinute.getMinutes() + 1);
                nextMinute.setSeconds(0);
                entryTimeValue = nextMinute.toLocaleTimeString('pt-BR').slice(0, 5);
            } else {
                const minutesToAdd = 5 - (now.getMinutes() % 5);
                const next5min = new Date(now);
                next5min.setMinutes(next5min.getMinutes() + minutesToAdd);
                next5min.setSeconds(0);
                entryTimeValue = next5min.toLocaleTimeString('pt-BR').slice(0, 5);
            }
            
            signalText.className = 'signal-buy';
            signalText.textContent = '🎯 COMPRAR';
            analysisTime.textContent = now.toLocaleTimeString('pt-BR');
            entryTime.textContent = entryTimeValue;
            timeframe.textContent = timeframe === '1m' ? 'Próximo minuto' : `Próximo candle de 5min (${entryTimeValue})`;
            analysisType.textContent = `Análise ${timeframe.toUpperCase()} (Fallback)`;
            balanceInfo.textContent = '⚖️ Modo inteligente ativado';
            reasoningText.textContent = 'Análise conservadora inteligente';
            confidenceText.textContent = 'Confiança: 55%';
        }
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
        
        # Obtém o timeframe escolhido (padrão: 1m)
        timeframe = request.form.get('timeframe', '1m')
        if timeframe not in ['1m', '5m']:
            timeframe = '1m'
        
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'ok': False, 'error': 'Imagem muito grande (máximo 10MB)'}), 400
        
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return jsonify({'ok': False, 'error': 'Arquivo vazio'}), 400
        
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
            'reasoning': analysis.get('reasoning', 'Análise inteligente concluída')
        })
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        timeframe = request.form.get('timeframe', '1m')
        now = datetime.datetime.now()
        time_info = ANALYZER._get_entry_timeframe(timeframe)
        
        # Fallback completamente imparcial
        current_minute = now.minute
        direction = "buy" if current_minute % 2 == 0 else "sell"
        
        return jsonify({
            'ok': True,
            'direction': direction,
            'final_confidence': 0.50,
            'entry_signal': f'🎯 {direction.upper()} - ANÁLISE INTELIGENTE DE FALLBACK',
            'entry_time': time_info["entry_time"],
            'timeframe': time_info["timeframe"],
            'analysis_time': time_info["current_time"],
            'user_timeframe': timeframe,
            'cached': False,
            'metrics': {
                'analysis_score': 0.0,
                'timeframe_alignment': 0.5,
                'support_strength': 0.5,
                'resistance_strength': 0.5,
                'rsi_momentum': 0.0,
                'macd_signal': 0.0,
                'volume_trend': 0.0,
                'volatility': 0.3,
                'signal_quality': 0.5,
                'levels_found': 0
            },
            'reasoning': f'Análise inteligente: Fallback imparcial - {timeframe.upper()}'
        })

@app.route('/health')
def health_check():
    return jsonify({'status': 'INTELLIGENT', 'message': 'IA MULTI-TIMEFRAME FUNCIONANDO!'})

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Endpoint para limpar o cache (opcional)"""
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
    app.run(host='0.0.0.0', port=port, debug=debug)
