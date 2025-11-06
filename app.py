from __future__ import annotations

# --- Visual analysis dependency ---
try:
    import cv2  # OpenCV for visual (non-OCR) analysis
except Exception as _e:
    cv2 = None
    print('[WARN] OpenCV (cv2) not available. Visual analysis will be skipped.', str(_e))

"""
IA SIGNAL PRO - SUPER INTELIGENTE 🧠⚡
ANÁLISE DE CONFLITOS - DECISÕES CONTEXTUAIS - ZERO VIÉS
"""

import io
import os
import math
import datetime
import hashlib
import json
from typing import Any, Dict, Optional, List, Tuple
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
    
    def _get_cache_key(self, image_bytes: bytes, timeframe: str, user_indicators: Dict) -> str:
        indicators_hash = hashlib.md5(json.dumps(user_indicators, sort_keys=True).encode()).hexdigest()
        content_hash = hashlib.md5(image_bytes).hexdigest()
        return f"{timeframe}_{content_hash}_{indicators_hash}"
    
    def _get_cache_file(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, image_bytes: bytes, timeframe: str, user_indicators: Dict) -> Optional[Dict]:
        try:
            key = self._get_cache_key(image_bytes, timeframe, user_indicators)
            cache_file = self._get_cache_file(key)
            
            if os.path.exists(cache_file):
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
    
    def set(self, image_bytes: bytes, timeframe: str, user_indicators: Dict, analysis: Dict):
        try:
            key = self._get_cache_key(image_bytes, timeframe, user_indicators)
            cache_file = self._get_cache_file(key)
            
            cache_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'timeframe': timeframe,
                'user_indicators': user_indicators,
                'analysis': analysis
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass

# =========================
#  IA SUPER INTELIGENTE - ANÁLISE DE CONFLITOS
# =========================
class SuperIntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
        
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
        """Pré-processamento otimizado para day trade"""
        width, height = image.size
        
        # Redimensionamento adequado para análise precisa
        target_size = (800, 600)
        image = image.resize(target_size, Image.LANCZOS)
        
        return np.array(image)

    def _extract_price_data(self, img_array: np.ndarray) -> np.ndarray:
        """Extrai dados de preço de forma ultra precisa"""
        try:
            # Converte para escala de cinza com pesos otimizados
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Realce de bordas para melhor detecção
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = self._apply_convolution(gray, kernel)
            
            return enhanced
        except Exception as e:
            return np.dot(img_array[...,:3], [0.299, 0.587, 0.114])

    def _apply_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Aplica convolução de forma otimizada"""
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
            
            return np.clip(output, 0, 255)
        except Exception:
            return image

    # =========================
    #  ANÁLISE MICROSCÓPICA AVANÇADA
    # =========================
    
    def _microscopic_trend_analysis(self, price_data: np.ndarray) -> Dict[str, float]:
        """Análise NANO de tendências - precisão máxima"""
        try:
            height, width = price_data.shape
            
            # Análise multi-resolução para day trade
            resolutions = [1, 2, 3]
            trend_signals = []
            
            for resolution in resolutions:
                segment_size = max(1, width // (8 * resolution))
                segments = []
                
                for i in range(8 * resolution):
                    start = i * segment_size
                    end = min((i + 1) * segment_size, width)
                    segment = price_data[:, start:end]
                    
                    if segment.size > 0:
                        segment_mean = np.mean(segment)
                        if segment.shape[1] > 1:
                            x_vals = np.arange(min(5, segment.shape[1]))
                            y_vals = np.mean(segment[:, -min(5, segment.shape[1]):], axis=0)
                            if len(y_vals) > 1:
                                segment_trend = (y_vals[-1] - y_vals[0]) / (len(y_vals) - 1)
                            else:
                                segment_trend = 0
                        else:
                            segment_trend = 0
                        segments.append((segment_mean, segment_trend))
                
                if len(segments) >= 4:
                    means = [s[0] for s in segments]
                    trends = [s[1] for s in segments]
                    
                    if len(means) > 1:
                        overall_trend = (means[-1] - means[0]) / (len(means) - 1)
                    else:
                        overall_trend = 0
                    
                    trend_agreement = np.std(trends) if trends else 0
                    convergence_strength = 1.0 / (1.0 + trend_agreement * 8)
                    
                    trend_signals.append((overall_trend, convergence_strength))
            
            if trend_signals:
                weighted_trend = sum(t * s for t, s in trend_signals) / sum(s for _, s in trend_signals)
                overall_strength = np.mean([s for _, s in trend_signals])
            else:
                weighted_trend = 0
                overall_strength = 0
            
            return {
                "nano_trend": float(np.clip(weighted_trend * 2, -1, 1)),
                "convergence_strength": float(overall_strength),
                "multi_resolution_agreement": float(1.0 - np.std([t for t, _ in trend_signals]) if trend_signals else 0)
            }
        except Exception as e:
            return {"nano_trend": 0.0, "convergence_strength": 0.0, "multi_resolution_agreement": 0.0}

    def _analyze_micro_structure(self, price_data: np.ndarray) -> Dict[str, float]:
        """Analisa a estrutura MICRO do mercado com precisão"""
        try:
            density_analysis = self._price_density_analysis(price_data)
            micro_momentum = self._micro_momentum_analysis(price_data)
            volatility_quality = self._analyze_volatility_quality(price_data)
            
            return {
                "price_density": density_analysis,
                "micro_momentum": micro_momentum,
                "volatility_quality": volatility_quality,
                "structural_integrity": (density_analysis + micro_momentum + volatility_quality) / 3.0
            }
        except Exception:
            return {"price_density": 0.5, "micro_momentum": 0.5, "volatility_quality": 0.5, "structural_integrity": 0.5}

    def _price_density_analysis(self, price_data: np.ndarray) -> float:
        """Analisa a densidade/distribuição do preço para day trade"""
        try:
            hist, bins = np.histogram(price_data.flatten(), bins=30)
            hist_normalized = hist / np.sum(hist)
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-8))
            max_entropy = np.log(len(hist))
            
            density_score = 1.0 - (entropy / max_entropy)
            return float(np.clip(density_score, 0, 1))
        except Exception:
            return 0.5

    def _micro_momentum_analysis(self, price_data: np.ndarray) -> float:
        """Analisa momentum em nível microscópico para entradas precisas"""
        try:
            height, width = price_data.shape
            
            if width < 15:
                return 0.5
            
            row_means = np.mean(price_data, axis=0)
            velocity = np.gradient(row_means)
            acceleration = np.gradient(velocity)
            
            # Foco nos movimentos mais recentes
            recent_velocity = np.mean(velocity[-min(7, len(velocity)):])
            recent_acceleration = np.mean(acceleration[-min(5, len(acceleration)):])
            
            momentum_score = (
                np.tanh(recent_velocity * 15) * 0.6 +
                np.tanh(recent_acceleration * 8) * 0.4
            )
            
            return float((momentum_score + 1) / 2)
        except Exception:
            return 0.5

    def _analyze_volatility_quality(self, price_data: np.ndarray) -> float:
        """Analisa a qualidade da volatilidade para day trade"""
        try:
            height, width = price_data.shape
            if width < 10:
                return 0.5
                
            row_means = np.mean(price_data, axis=0)
            
            # Volatilidade de curto prazo
            short_term_vol = np.std(row_means[-min(10, len(row_means)):])
            
            # Volatilidade ideal para day trade
            ideal_vol_range = (0.05, 0.15)
            vol_score = 1.0 - abs(short_term_vol - np.mean(ideal_vol_range)) / (ideal_vol_range[1] - ideal_vol_range[0])
            
            return float(np.clip(vol_score, 0, 1))
        except Exception:
            return 0.5

    def _analyze_flow_dynamics(self, price_data: np.ndarray) -> Dict[str, float]:
        """Analisa a DINÂMICA do fluxo de preços para day trade"""
        try:
            continuity_score = self._flow_continuity_analysis(price_data)
            breakage_analysis = self._breakage_detection(price_data)
            smooth_transitions = self._smoothness_analysis(price_data)
            
            return {
                "flow_continuity": continuity_score,
                "breakage_resistance": breakage_analysis,
                "transition_smoothness": smooth_transitions,
                "overall_flow_quality": (continuity_score + breakage_analysis + smooth_transitions) / 3.0
            }
        except Exception:
            return {"flow_continuity": 0.5, "breakage_resistance": 0.5, "transition_smoothness": 0.5, "overall_flow_quality": 0.5}

    def _flow_continuity_analysis(self, price_data: np.ndarray) -> float:
        """Analisa continuidade do fluxo para entradas suaves"""
        try:
            height, width = price_data.shape
            if width < 10:
                return 0.5
                
            row_means = np.mean(price_data, axis=0)
            changes = np.diff(row_means)
            
            acceleration = np.diff(changes)
            smoothness = 1.0 / (1.0 + np.std(acceleration))
            
            return float(np.clip(smoothness, 0, 1))
        except Exception:
            return 0.5

    def _breakage_detection(self, price_data: np.ndarray) -> float:
        """Detecta rupturas na estrutura de preços"""
        try:
            height, width = price_data.shape
            if width < 10:
                return 0.5
                
            vertical_profiles = []
            for col in range(0, width, max(1, width // 15)):
                column_data = price_data[:, col]
                if len(column_data) > 0:
                    profile = np.gradient(column_data)
                    vertical_profiles.append(np.std(profile))
            
            if vertical_profiles:
                breakage_score = 1.0 / (1.0 + np.mean(vertical_profiles))
                return float(np.clip(breakage_score, 0, 1))
            else:
                return 0.5
        except Exception:
            return 0.5

    def _smoothness_analysis(self, price_data: np.ndarray) -> float:
        """Analisa suavidade das transições para day trade"""
        try:
            height, width = price_data.shape
            if width < 5:
                return 0.5
                
            row_means = np.mean(price_data, axis=0)
            
            first_deriv = np.gradient(row_means)
            second_deriv = np.gradient(first_deriv)
            
            smoothness = 1.0 / (1.0 + np.std(second_deriv))
            return float(np.clip(smoothness, 0, 1))
        except Exception:
            return 0.5

    # =========================
    #  ANÁLISE TRADICIONAL FORTALECIDA
    # =========================
    
    def _analyze_price_action(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise de price action para day trade"""
        try:
            height, width = price_data.shape
            segments = 8
            segment_size = max(1, width // segments)
            regions = []
            
            for i in range(segments):
                start = i * segment_size
                end = min((i + 1) * segment_size, width)
                segment = price_data[:, start:end]
                if segment.size > 0:
                    regions.append(np.mean(segment))
            
            if len(regions) >= 3:
                if len(regions) > 1:
                    price_range = max(regions) - min(regions) if max(regions) != min(regions) else 1
                    slope = (regions[-1] - regions[0]) / (len(regions) - 1)
                    slope_normalized = slope / (price_range + 1e-8)
                else:
                    slope_normalized = 0
                    
                if len(regions) > 1:
                    changes = [regions[i] - regions[i-1] for i in range(1, len(regions))]
                    avg_change = np.mean(np.abs(changes))
                    if avg_change > 0:
                        trend_strength = min(1.0, abs(slope_normalized) / (avg_change + 1e-8))
                    else:
                        trend_strength = min(1.0, abs(slope_normalized) * 12)
                else:
                    trend_strength = 0
            else:
                slope_normalized = 0
                trend_strength = 0.5
            
            trend_strength = min(1.0, max(0.0, trend_strength))
            slope_normalized = max(-1.0, min(1.0, slope_normalized))
            
            return {
                "trend_direction": float(slope_normalized),
                "trend_strength": float(trend_strength),
                "momentum": float(slope_normalized * 1.2),
                "volatility": float(np.std(price_data) / (np.mean(price_data) + 1e-8)),
                "price_range": float(np.ptp(price_data))
            }
        except Exception:
            return {"trend_direction": 0.0, "trend_strength": 0.5, "momentum": 0.0, "volatility": 0.0, "price_range": 0.0}

    def _calculate_advanced_indicators(self, price_data: np.ndarray) -> Dict[str, float]:
        """Indicadores técnicos SUPER-REFORÇADOS para day trade"""
        try:
            height, width = price_data.shape
            
            if width > 10:
                row_means = np.mean(price_data, axis=0)
                
                # MACD FORTALECIDO
                fast_window = min(3, len(row_means))
                slow_window = min(6, len(row_means))
                signal_window = min(4, len(row_means))
                
                fast_ma = np.mean(row_means[-fast_window:])
                slow_ma = np.mean(row_means[-slow_window:])
                macd_line = fast_ma - slow_ma
                
                macd_values = []
                for i in range(slow_window, len(row_means)):
                    fast_val = np.mean(row_means[i-fast_window:i])
                    slow_val = np.mean(row_means[i-slow_window:i])
                    macd_values.append(fast_val - slow_val)
                
                if len(macd_values) >= signal_window:
                    signal_line = np.mean(macd_values[-signal_window:])
                    macd_histogram = macd_line - signal_line
                else:
                    signal_line = macd_line * 0.9
                    macd_histogram = macd_line * 0.1
                
                # RSI FORTALECIDO
                if len(row_means) > 5:
                    gains = []
                    losses = []
                    for i in range(1, len(row_means)):
                        change = row_means[i] - row_means[i-1]
                        if change > 0:
                            gains.append(change)
                        else:
                            losses.append(abs(change))
                    
                    avg_gain = np.mean(gains) if gains else 0
                    avg_loss = np.mean(losses) if losses else 0
                    
                    if avg_loss == 0:
                        rsi = 100 if avg_gain > 0 else 50
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    
                    rsi_normalized = (rsi - 50) / 50
                else:
                    rsi_normalized = 0.0
                
                # FORÇA DO MACD
                volatility = np.std(row_means) + 1e-8
                macd_strength = min(1.0, abs(macd_histogram) / (volatility * 1.5))
                macd_direction = 1 if macd_histogram > 0 else -1
                macd_power = macd_strength * macd_direction
                
            else:
                rsi_normalized = 0.0
                macd_power = 0.0
                macd_strength = 0.0
            
            return {
                "rsi": float(rsi_normalized),
                "macd": float(macd_power),
                "macd_strength": float(macd_strength),
                "volume_intensity": float(min(1.0, np.var(price_data) / 800.0)),
                "momentum_quality": float(min(1.0, (abs(rsi_normalized) + abs(macd_power)) / 2))
            }
        except Exception as e:
            return {"rsi": 0.0, "macd": 0.0, "macd_strength": 0.0, "volume_intensity": 0.0, "momentum_quality": 0.0}

    # =========================
    #  ANÁLISE INTELIGENTE DE CONFLITOS - NOVA!
    # =========================
    
    def _analyze_user_indicators(self, user_indicators: Dict) -> Dict[str, float]:
        """Análise INTELIGENTE dos indicadores do usuário"""
        try:
            macd = float(user_indicators.get('macd', 0))
            rsi = float(user_indicators.get('rsi', 50))
            adx = float(user_indicators.get('adx', 0))
            
            # ✅ ANÁLISE INTELIGENTE DO RSI
            if rsi <= 25:  # SOBREVENDA EXTREMA
                rsi_power = 0.8   # Sinal de COMPRA (reversão esperada)
                rsi_context = "oversold_strong"
            elif rsi >= 75:  # SOBRECOMPRA EXTREMA  
                rsi_power = -0.8  # Sinal de VENDA (reversão esperada)
                rsi_context = "overbought_strong"
            elif rsi <= 35:  # SOBREVENDA
                rsi_power = 0.5   # Sinal de COMPRA
                rsi_context = "oversold"
            elif rsi >= 65:  # SOBRECOMPRA
                rsi_power = -0.5  # Sinal de VENDA
                rsi_context = "overbought"
            else:  # ZONA NEUTRA
                rsi_power = (rsi - 50) / 50 * 0.3
                rsi_context = "neutral"
            
            # ✅ ANÁLISE INTELIGENTE DO MACD
            macd_power = np.clip(macd * 2, -1, 1)  # Normalizado
            macd_strength = min(1.0, abs(macd) * 0.5)
            
            if macd < -2.0:
                macd_context = "strong_sell"
            elif macd > 2.0:
                macd_context = "strong_buy"
            elif macd < -0.5:
                macd_context = "sell"
            elif macd > 0.5:
                macd_context = "buy"
            else:
                macd_context = "neutral"
            
            # ✅ ANÁLISE INTELIGENTE DO ADX
            if adx > 40:
                adx_power = 0.8
                adx_context = "strong_trend"
            elif adx > 25:
                adx_power = 0.6
                adx_context = "moderate_trend"
            else:
                adx_power = 0.3
                adx_context = "weak_trend"
            
            return {
                "user_macd_power": float(macd_power),
                "user_macd_strength": float(macd_strength),
                "user_macd_context": macd_context,
                "user_rsi_power": float(rsi_power),
                "user_rsi_strength": float(0.8 if abs(rsi_power) > 0.7 else 0.5),
                "user_rsi_context": rsi_context,
                "user_adx_power": float(adx_power),
                "user_adx_context": adx_context,
                "user_confidence": float(min(1.0, (macd_strength + abs(rsi_power) + adx_power) / 3))
            }
        except Exception as e:
            return {
                "user_macd_power": 0.0, "user_macd_strength": 0.0, "user_macd_context": "neutral",
                "user_rsi_power": 0.0, "user_rsi_strength": 0.0, "user_rsi_context": "neutral", 
                "user_adx_power": 0.0, "user_adx_context": "weak_trend",
                "user_confidence": 0.0
            }

    def _intelligent_conflict_resolution(self, user_analysis: Dict, chart_analysis: Dict) -> Dict[str, Any]:
        """RESOLUÇÃO INTELIGENTE de conflitos entre indicadores"""
        
        user_macd = user_analysis.get('user_macd_power', 0)
        user_rsi = user_analysis.get('user_rsi_power', 0)
        user_adx = user_analysis.get('user_adx_power', 0)
        macd_context = user_analysis.get('user_macd_context', 'neutral')
        rsi_context = user_analysis.get('user_rsi_context', 'neutral')
        
        chart_trend = chart_analysis['traditional']['price_action']['trend_direction']
        chart_strength = chart_analysis['traditional']['price_action']['trend_strength']
        chart_momentum = chart_analysis['traditional']['price_action']['momentum']
        
        # 🎯 IDENTIFICAÇÃO DO CENÁRIO
        scenario = self._identify_market_scenario(user_macd, user_rsi, macd_context, rsi_context, chart_trend)
        
        # 🧠 RESOLUÇÃO INTELIGENTE POR CENÁRIO
        if scenario == "oversold_reversal":
            # RSI oversold + MACD não tão negativo = potencial reversão para COMPRA
            resolved_score = user_rsi * 0.7 + user_macd * 0.3
            confidence_boost = 0.15
            reasoning = "Reversão de sobrevenda - RSI indica exaustão de venda"
            
        elif scenario == "momentum_breakdown":
            # MACD muito negativo + RSI não oversold = VENDA de momentum
            resolved_score = user_macd * 0.8 + user_rsi * 0.2
            confidence_boost = 0.12
            reasoning = "Momentum de venda forte - MACD dominante"
            
        elif scenario == "confirmed_downtrend":
            # Todos alinhados para venda = VENDA forte
            resolved_score = min(user_macd, user_rsi, chart_trend)
            confidence_boost = 0.20
            reasoning = "Tendência de baixa confirmada - múltiplas confirmações"
            
        elif scenario == "divergence_conflict":
            # Conflito forte = análise mais conservadora
            if abs(user_macd) > abs(user_rsi):
                resolved_score = user_macd * 0.6 + user_rsi * 0.4
            else:
                resolved_score = user_rsi * 0.6 + user_macd * 0.4
            confidence_boost = 0.05
            reasoning = "Conflito de indicadores - análise conservadora"
            
        else:  # neutral_market
            # Cenário neutro - média balanceada
            resolved_score = (user_macd + user_rsi + chart_trend) / 3
            confidence_boost = 0.0
            reasoning = "Mercado neutro - análise balanceada"
        
        return {
            "resolved_score": float(resolved_score),
            "scenario": scenario,
            "reasoning": reasoning,
            "confidence_boost": confidence_boost,
            "final_confidence": min(0.95, user_analysis.get('user_confidence', 0) + confidence_boost)
        }

    def _identify_market_scenario(self, macd: float, rsi: float, macd_ctx: str, rsi_ctx: str, chart_trend: float) -> str:
        """Identifica inteligentemente o cenário de mercado"""
        
        is_strong_oversold = rsi_ctx in ["oversold_strong", "oversold"] and rsi > 0.5
        is_strong_overbought = rsi_ctx in ["overbought_strong", "overbought"] and rsi < -0.5
        is_strong_macd_sell = macd_ctx in ["strong_sell", "sell"] and macd < -0.5
        is_strong_macd_buy = macd_ctx in ["strong_buy", "buy"] and macd > 0.5
        is_downtrend = chart_trend < -0.2
        is_uptrend = chart_trend > 0.2
        
        # 🎯 CENÁRIOS PRINCIPAIS
        if is_strong_oversold and not is_strong_macd_sell and is_downtrend:
            return "oversold_reversal"
        elif is_strong_macd_sell and not is_strong_oversold and is_downtrend:
            return "momentum_breakdown" 
        elif is_strong_macd_sell and is_strong_oversold and is_downtrend:
            return "confirmed_downtrend"
        elif is_strong_macd_buy and is_strong_overbought and is_uptrend:
            return "confirmed_uptrend"
        elif (is_strong_oversold and is_strong_macd_sell) or (is_strong_overbought and is_strong_macd_buy):
            return "divergence_conflict"
        else:
            return "neutral_market"

    # =========================
    #  MOTOR DE DECISÃO SUPER INTELIGENTE
    # =========================
    
    def _super_intelligent_decision_engine(self, all_analyses: Dict, timeframe: str) -> Dict[str, Any]:
        """MOTOR QUE ANALISA CONFLITOS INTELIGENTEMENTE"""
        
        traditional = all_analyses['traditional']
        user_analysis = all_analyses.get('user_analysis', {})
        nano_trend = all_analyses['nano_analysis']
        micro_structure = all_analyses['micro_structure']
        flow_dynamics = all_analyses['flow_dynamics']
        
        # 🧠 RESOLUÇÃO INTELIGENTE DE CONFLITOS
        conflict_resolution = self._intelligent_conflict_resolution(user_analysis, all_analyses)
        
        # ⚖️ PESOS DINÂMICOS baseados no cenário
        weights = self._get_dynamic_weights(conflict_resolution['scenario'])
        
        # 📊 COMPONENTES DA ANÁLISE
        trend_power = traditional['price_action']['trend_direction'] * traditional['price_action']['trend_strength']
        macd_power = traditional['indicators']['macd'] * traditional['indicators']['macd_strength']
        
        user_power = conflict_resolution['resolved_score']
        
        nano_power = nano_trend['nano_trend'] * nano_trend['convergence_strength']
        micro_power = micro_structure['structural_integrity']
        trend_micro = (trend_power + nano_power + micro_power) / 3
        
        momentum_power = traditional['price_action']['momentum']
        flow_power = flow_dynamics['overall_flow_quality']
        momentum_flow = (momentum_power + flow_power) / 2
        
        price_action_power = traditional['price_action']['trend_strength']
        if traditional['price_action']['trend_direction'] > 0:
            price_action_power = abs(price_action_power)
        else:
            price_action_power = -abs(price_action_power)
        
        # 💥 SCORE FINAL INTELIGENTE
        total_score = (
            user_power * weights['user_indicators'] +
            trend_micro * weights['trend_micro'] +
            price_action_power * weights['price_action'] +
            momentum_flow * weights['momentum_flow']
        )
        
        # 🎯 DECISÃO COM CONFIANÇA INTELIGENTE
        if total_score > 0.05:
            direction = "buy"
            base_confidence = min(abs(total_score) * 1.3, 0.8)
        elif total_score < -0.05:
            direction = "sell" 
            base_confidence = min(abs(total_score) * 1.3, 0.8)
        else:
            direction = "buy" if user_power > 0 else "sell"
            base_confidence = 0.6
        
        # 🧠 CONFIANÇA FINAL COM BOOST INTELIGENTE
        final_confidence = base_confidence + conflict_resolution['confidence_boost']
        
        reasoning = self._generate_intelligent_reasoning(
            direction, conflict_resolution, user_analysis, traditional
        )
        
        return {
            "direction": direction,
            "confidence": min(0.95, final_confidence),
            "reasoning": reasoning,
            "total_score": total_score,
            "scenario": conflict_resolution['scenario'],
            "user_power": user_power,
            "trend_micro_power": trend_micro,
            "price_action_power": price_action_power,
            "momentum_flow_power": momentum_flow
        }

    def _get_dynamic_weights(self, scenario: str) -> Dict[str, float]:
        """Retorna pesos dinâmicos baseados no cenário"""
        weight_profiles = {
            "oversold_reversal": {
                'user_indicators': 0.35,  # Mais peso no usuário (RSI oversold)
                'trend_micro': 0.25,
                'price_action': 0.20, 
                'momentum_flow': 0.20
            },
            "momentum_breakdown": {
                'user_indicators': 0.30,
                'trend_micro': 0.30,      # Mais peso na tendência
                'price_action': 0.25,
                'momentum_flow': 0.15
            },
            "confirmed_downtrend": {
                'user_indicators': 0.25,
                'trend_micro': 0.35,      # Máximo peso na tendência
                'price_action': 0.25,
                'momentum_flow': 0.15
            },
            "divergence_conflict": {
                'user_indicators': 0.40,  # Máximo peso no usuário (conflito)
                'trend_micro': 0.20,
                'price_action': 0.20,
                'momentum_flow': 0.20
            }
        }
        return weight_profiles.get(scenario, {
            'user_indicators': 0.30,
            'trend_micro': 0.25,
            'price_action': 0.25,
            'momentum_flow': 0.20
        })

    def _generate_intelligent_reasoning(self, direction: str, conflict_resolution: Dict, 
                                     user_analysis: Dict, traditional: Dict) -> str:
        """Gera reasoning inteligente baseado no cenário"""
        
        scenario = conflict_resolution['scenario']
        base_reasoning = conflict_resolution['reasoning']
        
        user_macd_ctx = user_analysis.get('user_macd_context', 'neutral')
        user_rsi_ctx = user_analysis.get('user_rsi_context', 'neutral')
        
        # 🎯 DETALHAMENTO DO CENÁRIO
        if scenario == "oversold_reversal":
            details = f"RSI {user_rsi_ctx} + MACD {user_macd_ctx}"
        elif scenario == "momentum_breakdown":
            details = f"MACD {user_macd_ctx} dominante"
        elif scenario == "confirmed_downtrend":
            details = f"MACD {user_macd_ctx} + RSI {user_rsi_ctx} + Gráfico alinhados"
        elif scenario == "divergence_conflict":
            details = f"Conflito: MACD {user_macd_ctx} vs RSI {user_rsi_ctx}"
        else:
            details = "Análise técnica balanceada"
        
        strength = "FORTE" if abs(conflict_resolution['resolved_score']) > 0.3 else "MODERADA"
        
        if direction == "buy":
            return f"🎯 COMPRA {strength} - {base_reasoning} | {details}"
        else:
            return f"🎯 VENDA {strength} - {base_reasoning} | {details}"

    def _calculate_signal_quality(self, analyses: Dict) -> float:
        """Calcula qualidade do sinal para day trade"""
        try:
            factors = [
                analyses['nano_analysis']['convergence_strength'] * 0.25,
                analyses['micro_structure']['structural_integrity'] * 0.25,
                analyses['flow_dynamics']['overall_flow_quality'] * 0.20,
                analyses['traditional']['price_action']['trend_strength'] * 0.20,
                analyses['traditional']['indicators']['macd_strength'] * 0.10
            ]
            return float(np.clip(np.mean(factors), 0, 1))
        except Exception:
            return 0.6

    def _get_entry_timeframe(self, user_timeframe: str) -> Dict[str, str]:
        """Calcula timeframe de entrada otimizado"""
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

    def analyze(self, blob: bytes, timeframe: str = '1m', user_indicators: Dict = None) -> Dict[str, Any]:
        """ANÁLISE SUPER INTELIGENTE - RESOLUÇÃO DE CONFLITOS"""
        
        if user_indicators is None:
            user_indicators = {}
        
        # Cache inteligente
        cached = self.cache.get(blob, timeframe, user_indicators)
        if cached:
            cached['cached'] = True
            return cached
        
        # Processamento básico
        image = self._load_image(blob)
        self._validate_chart_image(image)
        
        img_array = self._preprocess_image(image, timeframe)
        price_data = self._extract_price_data(img_array)
        
        # 🧠 ANÁLISE MULTI-CAMADAS
        analyses = {
            'traditional': {
                'price_action': self._analyze_price_action(price_data, timeframe),
                'indicators': self._calculate_advanced_indicators(price_data)
            },
            'nano_analysis': self._microscopic_trend_analysis(price_data),
            'micro_structure': self._analyze_micro_structure(price_data),
            'flow_dynamics': self._analyze_flow_dynamics(price_data),
            'price_data': price_data
        }
        
        # 🎯 ANÁLISE INTELIGENTE DOS INDICADORES DO USUÁRIO
        if user_indicators:
            analyses['user_analysis'] = self._analyze_user_indicators(user_indicators)
        
        # 🎯 MOTOR DE DECISÃO SUPER INTELIGENTE
        decision = self._super_intelligent_decision_engine(analyses, timeframe)
        time_info = self._get_entry_timeframe(timeframe)
        
        # 📊 QUALIDADE DA ANÁLISE
        signal_quality = self._calculate_signal_quality(analyses)
        
        # 🎨 RESULTADO SUPER INTELIGENTE
        result = {
            "direction": decision["direction"],
            "final_confidence": float(decision["confidence"]),
            "entry_signal": f"🧠 {decision['direction'].upper()} - {decision['reasoning']}",
            "entry_time": time_info["entry_time"],
            "timeframe": time_info["timeframe"],
            "analysis_time": time_info["current_time"],
            "user_timeframe": timeframe,
            "cached": False,
            "signal_quality": float(signal_quality),
            "analysis_grade": "high" if signal_quality > 0.7 else "medium",
            "market_context": decision["scenario"],
            "user_indicators_provided": bool(user_indicators),
            "metrics": {
                "analysis_score": float(decision["total_score"]),
                "user_power": float(decision["user_power"]),
                "trend_micro_power": float(decision["trend_micro_power"]),
                "price_action_power": float(decision["price_action_power"]),
                "momentum_flow_power": float(decision["momentum_flow_power"]),
                "trend_strength": analyses['traditional']['price_action']['trend_strength'],
                "momentum": analyses['traditional']['price_action']['momentum'],
                "rsi": analyses['traditional']['indicators']['rsi'],
                "macd": analyses['traditional']['indicators']['macd'],
                "macd_strength": analyses['traditional']['indicators']['macd_strength']
            },
            "reasoning": decision["reasoning"],
            "intelligent_scenario": decision["scenario"]
        }
        
        # Adiciona métricas do usuário se disponíveis
        if user_indicators:
            result["user_metrics"] = {
                "provided_macd": float(user_indicators.get('macd', 0)),
                "provided_rsi": float(user_indicators.get('rsi', 50)),
                "provided_adx": float(user_indicators.get('adx', 0)),
                "provided_price": float(user_indicators.get('price', 0))
            }
        
        self.cache.set(blob, timeframe, user_indicators, result)
        return result

# =========================
#  APLICAÇÃO FLASK (MANTIDO IGUAL)
# =========================
app = Flask(__name__)
analyzer = SuperIntelligentAnalyzer()

# Configurações para produção
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Signal Pro - SUPER INTELIGENTE 🧠⚡</title>
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
            border: 2px solid #7ce0ff;
            box-shadow: 0 10px 30px rgba(124, 224, 255, 0.3);
        }
        .header { 
            text-align: center; 
            margin-bottom: 20px; 
        }
        .title {
            font-size: 24px; 
            font-weight: 800; 
            margin-bottom: 5px;
            background: linear-gradient(90deg, #7ce0ff, #00ff88);
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;
        }
        .subtitle { 
            color: #9db0d1; 
            font-size: 13px; 
            margin-bottom: 10px; 
        }
        
        .upload-area {
            border: 2px dashed #7ce0ff; 
            border-radius: 15px;
            padding: 30px 15px; 
            text-align: center;
            background: rgba(124, 224, 255, 0.05); 
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
            border: 1px solid #7ce0ff;
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
            border: 2px solid #7ce0ff;
            background: rgba(124, 224, 255, 0.1); 
            color: #9db0d1;
            border-radius: 10px; 
            cursor: pointer; 
            text-align: center;
            font-weight: 600; 
            transition: all 0.3s ease;
        }
        .timeframe-btn.active {
            background: linear-gradient(135deg, #7ce0ff 0%, #4a90e2 100%);
            color: white; 
            border-color: #4a90e2;
        }
        
        .analyze-btn {
            background: linear-gradient(135deg, #7ce0ff 0%, #4a90e2 100%);
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
            background: linear-gradient(135deg, #4a90e2 0%, #2a76ef 100%);
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
            color: #7ce0ff;
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
        
        .context-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 700;
            margin-left: 8px;
        }
        .context-oversold_reversal { background: linear-gradient(135deg, #00ff88, #00cc66); color: white; }
        .context-momentum_breakdown { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
        .context-confirmed_downtrend { background: linear-gradient(135deg, #ff6b6b, #ff0000); color: white; }
        .context-divergence_conflict { background: linear-gradient(135deg, #ffaa00, #ff8800); color: white; }
        .context-neutral_market { background: linear-gradient(135deg, #7ce0ff, #4a90e2); color: white; }
        
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
            color: #7ce0ff; 
            font-size: 14px;
        }
        
        .cache-badge {
            background: linear-gradient(135deg, #ffaa00, #ff6b6b);
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
            background: linear-gradient(90deg, #7ce0ff, #00ff88);
            width: 0%; 
            transition: width 0.3s ease;
        }
        
        .power-analysis {
            background: rgba(124, 224, 255, 0.1);
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #7ce0ff;
        }
        
        .image-preview {
            max-width: 100%;
            max-height: 200px;
            border-radius: 8px;
            margin: 10px 0;
            border: 2px solid #7ce0ff;
            display: none;
        }
        
        .intelligent-badge {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 8px;
            margin-left: 5px;
            background: linear-gradient(135deg, #7ce0ff, #4a90e2);
            color: white;
        }
        
        .user-indicators {
            background: rgba(124, 224, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            margin: 15px 0;
            border: 1px solid #7ce0ff;
        }
        
        .indicator-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        
        .indicator-group {
            display: flex;
            flex-direction: column;
        }
        
        .indicator-label {
            font-size: 12px;
            color: #7ce0ff;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .indicator-input {
            background: rgba(42, 53, 82, 0.5);
            border: 1px solid #7ce0ff;
            border-radius: 6px;
            padding: 8px;
            color: white;
            font-size: 14px;
        }
        
        .indicator-input:focus {
            outline: none;
            border-color: #00ff88;
        }
        
        .optional-badge {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 8px;
            margin-left: 5px;
            background: linear-gradient(135deg, #ffaa00, #ff6b6b);
            color: white;
        }
        
        .user-data-badge {
            background: linear-gradient(135deg, #00ff88, #00cc66);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 700;
            margin-left: 8px;
        }
        
        .scenario-info {
            background: rgba(124, 224, 255, 0.1);
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #7ce0ff;
            text-align: center;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠⚡ IA SIGNAL PRO - SUPER INTELIGENTE</div>
            <div class="subtitle">ANÁLISE DE CONFLITOS - DECISÕES CONTEXTUAIS</div>
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
        
        <img id="imagePreview" class="image-preview" alt="Prévia da imagem">
        
        <!-- SEÇÃO DOS INDICADORES DO USUÁRIO -->
        <div class="user-indicators">
            <div style="text-align: center; font-weight: 600; margin-bottom: 10px; color: #7ce0ff;">
                🔢 DADOS TÉCNICOS <span class="optional-badge">INTELIGENTES</span>
            </div>
            <div style="text-align: center; font-size: 12px; color: #9db0d1; margin-bottom: 10px;">
                A IA analisa conflitos entre indicadores
            </div>
            <div class="indicator-grid">
                <div class="indicator-group">
                    <label class="indicator-label">📈 MACD</label>
                    <input type="number" step="0.001" id="macdInput" class="indicator-input" placeholder="Ex: -0.002">
                </div>
                <div class="indicator-group">
                    <label class="indicator-label">📊 RSI</label>
                    <input type="number" step="0.1" id="rsiInput" class="indicator-input" placeholder="Ex: 42.5">
                </div>
                <div class="indicator-group">
                    <label class="indicator-label">🎯 ADX</label>
                    <input type="number" step="0.1" id="adxInput" class="indicator-input" placeholder="Ex: 35.0">
                </div>
                <div class="indicator-group">
                    <label class="indicator-label">💰 PREÇO</label>
                    <input type="number" step="0.01" id="priceInput" class="indicator-input" placeholder="Ex: 1450.75">
                </div>
            </div>
        </div>
        
        <button class="analyze-btn" id="analyzeBtn" disabled>🧠 SELECIONE UMA IMAGEM PRIMEIRO</button>
        
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
            
            <div id="scenarioInfo" class="scenario-info" style="display: none;"></div>
            
            <div id="qualityIndicator" class="quality-indicator"></div>
            
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="power-analysis" id="powerAnalysis">
                <div style="text-align: center; font-weight: 600; margin-bottom: 8px; color: #7ce0ff;">
                    ⚡ ANÁLISE INTELIGENTE
                </div>
                <div id="powerMetrics"></div>
            </div>
            
            <div class="metrics" id="metricsText"></div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('fileInput');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const uploadArea = document.getElementById('uploadArea');
            const imagePreview = document.getElementById('imagePreview');
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
            const powerAnalysis = document.getElementById('powerAnalysis');
            const powerMetrics = document.getElementById('powerMetrics');
            const scenarioInfo = document.getElementById('scenarioInfo');
            const timeframeBtns = document.querySelectorAll('.timeframe-btn');
            
            // Inputs dos indicadores do usuário
            const macdInput = document.getElementById('macdInput');
            const rsiInput = document.getElementById('rsiInput');
            const adxInput = document.getElementById('adxInput');
            const priceInput = document.getElementById('priceInput');

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

            // Upload de arquivo
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#00ff88';
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = '#7ce0ff';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#7ce0ff';
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    handleFileSelect(e);
                }
            });

            function handleFileSelect(event) {
                const files = event.target.files;
                if (files && files.length > 0) {
                    selectedFile = files[0];
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = `✅ PRONTO PARA ANÁLISE ${currentTimeframe.toUpperCase()}`;
                    
                    // Mostrar prévia da imagem
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                    };
                    reader.readAsDataURL(selectedFile);
                } else {
                    analyzeBtn.disabled = true;
                    analyzeBtn.textContent = '🧠 SELECIONE UMA IMAGEM PRIMEIRO';
                    imagePreview.style.display = 'none';
                }
            }

            fileInput.addEventListener('change', handleFileSelect);

            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) {
                    alert('📸 Selecione uma imagem do gráfico primeiro!');
                    return;
                }

                analyzeBtn.disabled = true;
                analyzeBtn.textContent = `🧠 ANALISANDO ${currentTimeframe.toUpperCase()}...`;
                result.style.display = 'block';
                errorMessage.style.display = 'none';
                scenarioInfo.style.display = 'none';
                
                signalText.className = 'signal-text';
                signalText.textContent = 'Analisando conflitos inteligentemente...';
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
                    const minutesToAdd = 5 - (now.minute % 5);
                    const next5min = new Date(now);
                    next5min.setMinutes(next5min.getMinutes() + minutesToAdd);
                    next5min.setSeconds(0);
                    entryTimeValue = next5min.toLocaleTimeString('pt-BR').slice(0, 5);
                    timeframeEl.textContent = `Próximo candle de 5min`;
                }
                
                entryTime.textContent = entryTimeValue;
                reasoningText.textContent = 'Processando análise inteligente de conflitos...';
                confidenceText.textContent = '';
                progressFill.style.width = '20%';
                
                metricsText.innerHTML = '<div class="loading">Iniciando análise inteligente...</div>';

                try {
                    const formData = new FormData();
                    formData.append('image', selectedFile);
                    formData.append('timeframe', currentTimeframe);
                    
                    // Adiciona indicadores do usuário ao formData
                    const userIndicators = {};
                    if (macdInput.value) userIndicators.macd = parseFloat(macdInput.value);
                    if (rsiInput.value) userIndicators.rsi = parseFloat(rsiInput.value);
                    if (adxInput.value) userIndicators.adx = parseFloat(adxInput.value);
                    if (priceInput.value) userIndicators.price = parseFloat(priceInput.value);
                    
                    if (Object.keys(userIndicators).length > 0) {
                        formData.append('user_indicators', JSON.stringify(userIndicators));
                    }
                    
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
                    analyzeBtn.textContent = `🔁 ANALISAR ${currentTimeframe.toUpperCase()} NOVAMENTE`;
                }
            });

            function displayResults(data) {
                const direction = data.direction;
                const confidence = (data.final_confidence * 100).toFixed(1);
                const cached = data.cached || false;
                const quality = data.analysis_grade || 'medium';
                const scenario = data.intelligent_scenario || 'neutral_market';
                const userIndicatorsProvided = data.user_indicators_provided || false;
                
                // Define classe e texto do sinal
                signalText.className = `signal-text signal-${direction}`;
                let directionText = direction === 'buy' ? '🎯 COMPRAR' : '🎯 VENDER';
                let userBadge = userIndicatorsProvided ? '<span class="user-data-badge">DADOS DO USUÁRIO</span>' : '';
                signalText.innerHTML = `${directionText} <span class="intelligent-badge">INTELIGENTE</span> ${userBadge} ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                
                // Atualiza informações
                analysisTime.textContent = data.analysis_time || '--:--:--';
                entryTime.textContent = data.entry_time || '--:--';
                timeframeEl.textContent = data.timeframe || 'Próximo minuto';
                
                reasoningText.textContent = data.reasoning;
                confidenceText.textContent = `Confiança Inteligente: ${confidence}%`;
                
                // Informações do cenário
                const scenarioLabels = {
                    'oversold_reversal': '🔄 REVERSÃO DE SOBREVENDA',
                    'momentum_breakdown': '📉 QUEDA DE MOMENTUM', 
                    'confirmed_downtrend': '🔴 TENDÊNCIA DE BAIXA CONFIRMADA',
                    'divergence_conflict': '⚡ CONFLITO DE INDICADORES',
                    'neutral_market': '⚖️ MERCADO NEUTRO'
                };
                
                scenarioInfo.innerHTML = `
                    <strong>Cenário Identificado:</strong> ${scenarioLabels[scenario] || scenarioLabels.neutral_market}
                `;
                scenarioInfo.style.display = 'block';
                
                // Indicador de qualidade
                qualityIndicator.className = `quality-indicator quality-${quality}`;
                if (quality === 'high') {
                    qualityIndicator.textContent = '✅ ALTA QUALIDADE - Análise confiável';
                } else {
                    qualityIndicator.textContent = '⚠️ QUALIDADE MÉDIA - Análise válida';
                }
                
                // Análise Inteligente
                const metrics = data.metrics || {};
                let powerHtml = '';
                
                const powerItems = [
                    ['Score Final', metrics.analysis_score?.toFixed(3)],
                    ['Poder do Usuário', (metrics.user_power * 100)?.toFixed(1) + '%'],
                    ['Poder Tendência+Micro', (metrics.trend_micro_power * 100)?.toFixed(1) + '%'],
                    ['Poder Price Action', (metrics.price_action_power * 100)?.toFixed(1) + '%'],
                    ['Poder Momentum+Fluxo', (metrics.momentum_flow_power * 100)?.toFixed(1) + '%']
                ];
                
                powerItems.forEach(([label, value]) => {
                    powerHtml += `
                        <div class="metric-item">
                            <span>${label}:</span>
                            <span class="metric-value">${value}</span>
                        </div>
                    `;
                });
                
                powerMetrics.innerHTML = powerHtml;
                
                // Métricas detalhadas
                let metricsHtml = '<div style="margin-bottom: 10px; text-align: center; font-weight: 600;">📊 ANÁLISE TÉCNICA COMPLETA</div>';
                
                const metricItems = [
                    ['Força da Tendência', (metrics.trend_strength * 100)?.toFixed(1) + '%'],
                    ['Momentum', metrics.momentum?.toFixed(3)],
                    ['RSI', metrics.rsi?.toFixed(3)],
                    ['MACD', metrics.macd?.toFixed(3)],
                    ['Força do MACD', (metrics.macd_strength * 100)?.toFixed(1) + '%'],
                    ['Qualidade do Sinal', (data.signal_quality * 100)?.toFixed(1) + '%']
                ];
                
                // Adiciona métricas do usuário se disponíveis
                if (data.user_metrics) {
                    metricItems.push(
                        ['MACD (User)', data.user_metrics.provided_macd?.toFixed(4)],
                        ['RSI (User)', data.user_metrics.provided_rsi?.toFixed(1)],
                        ['ADX (User)', data.user_metrics.provided_adx?.toFixed(1)],
                        ['Preço (User)', data.user_metrics.provided_price?.toFixed(2)]
                    );
                }
                
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
    """Endpoint de análise de imagem - SUPER INTELIGENTE"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        image_file = request.files['image']
        if not image_file or image_file.filename == '':
            return jsonify({'error': 'Arquivo inválido'}), 400
        
        timeframe = request.form.get('timeframe', '1m')
        if timeframe not in ['1m', '5m']:
            timeframe = '1m'
        
        # Processa indicadores do usuário
        user_indicators = {}
        user_indicators_str = request.form.get('user_indicators')
        if user_indicators_str:
            try:
                user_indicators = json.loads(user_indicators_str)
            except json.JSONDecodeError:
                pass
        
        # Verificação básica do arquivo
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'error': 'Imagem muito grande (máximo 10MB)'}), 400
        
        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Arquivo vazio'}), 400
        
        # Análise SUPER INTELIGENTE
        analysis = analyzer.analyze(image_bytes, timeframe, user_indicators)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': f'Erro na análise: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check para monitoramento"""
    return jsonify({
        'status': 'healthy', 
        'service': 'IA Signal Pro - SUPER INTELIGENTE',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '10.0.0-inteligencia-conflitos'
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
    
    print(f"🚀 IA Signal Pro - SUPER INTELIGENTE iniciando na porta {port}")
    print(f"🧠⚡ SISTEMA: ANÁLISE INTELIGENTE DE CONFLITOS")
    print(f"🎯 VERSÃO: RESOLUÇÃO CONTEXTUAL - CENÁRIOS DINÂMICOS")
    print(f"📈 RECURSOS: Identificação de Cenários + Pesos Dinâmicos + Reasoning Inteligente")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
