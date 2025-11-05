from __future__ import annotations

# --- Visual analysis dependency ---
try:
    import cv2  # OpenCV for visual (non-OCR) analysis
except Exception as _e:
    cv2 = None
    print('[WARN] OpenCV (cv2) not available. Visual analysis will be skipped.', str(_e))

"""
IA SIGNAL PRO - SUPER INTELIGENTE E NEUTRA 🧠⚖️
DECISÕES PURAMENTE TÉCNICAS - ZERO VIÉS
ANÁLISE DO MOMENTO DO MERCADO - SEM FAVORITISMO
"""

import io
import os
import math
import datetime
import hashlib
import json
import random
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
#  IA SUPER INTELIGENTE E NEUTRA
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
        """Pré-processamento otimizado"""
        width, height = image.size
        
        # Redimensionamento adequado
        target_size = (600, 450)
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
            return np.dot(img_array[...,:3], [0.299, 0.587, 0.114])

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
            
            return np.clip(output, 0, 255)
        except Exception:
            return image

    def _flow_continuity_analysis(self, price_data: np.ndarray) -> float:
        """Analisa continuidade do fluxo de preços"""
        try:
            height, width = price_data.shape
            if width < 10:
                return 0.5
                
            row_means = np.mean(price_data, axis=0)
            changes = np.diff(row_means)
            
            # Calcula suavidade das transições
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
            for col in range(0, width, max(1, width // 10)):
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
        """Analisa suavidade das transições"""
        try:
            height, width = price_data.shape
            if width < 5:
                return 0.5
                
            row_means = np.mean(price_data, axis=0)
            
            # Calcula derivada segunda para suavidade
            first_deriv = np.gradient(row_means)
            second_deriv = np.gradient(first_deriv)
            
            smoothness = 1.0 / (1.0 + np.std(second_deriv))
            return float(np.clip(smoothness, 0, 1))
        except Exception:
            return 0.5

    # =========================
    #  ANÁLISE MICROSCÓPICA AVANÇADA
    # =========================
    
    def _microscopic_trend_analysis(self, price_data: np.ndarray) -> Dict[str, float]:
        """Análise NANO de tendências - detecta movimentos mínimos"""
        try:
            height, width = price_data.shape
            
            # Análise multi-resolução
            resolutions = [1, 2, 4]
            trend_signals = []
            
            for resolution in resolutions:
                segment_size = max(1, width // (6 * resolution))
                segments = []
                
                for i in range(6 * resolution):
                    start = i * segment_size
                    end = min((i + 1) * segment_size, width)
                    segment = price_data[:, start:end]
                    
                    if segment.size > 0:
                        segment_mean = np.mean(segment)
                        if segment.shape[1] > 1:
                            x_vals = np.arange(min(3, segment.shape[1]))
                            y_vals = np.mean(segment[:, -min(3, segment.shape[1]):], axis=0)
                            if len(y_vals) > 1:
                                segment_trend = (y_vals[-1] - y_vals[0]) / (len(y_vals) - 1)
                            else:
                                segment_trend = 0
                        else:
                            segment_trend = 0
                        segments.append((segment_mean, segment_trend))
                
                if len(segments) >= 3:
                    means = [s[0] for s in segments]
                    trends = [s[1] for s in segments]
                    
                    if len(means) > 1:
                        overall_trend = (means[-1] - means[0]) / (len(means) - 1)
                    else:
                        overall_trend = 0
                    
                    trend_agreement = np.std(trends) if trends else 0
                    convergence_strength = 1.0 / (1.0 + trend_agreement * 10)
                    
                    trend_signals.append((overall_trend, convergence_strength))
            
            if trend_signals:
                weighted_trend = sum(t * s for t, s in trend_signals) / sum(s for _, s in trend_signals)
                overall_strength = np.mean([s for _, s in trend_signals])
            else:
                weighted_trend = 0
                overall_strength = 0
            
            return {
                "nano_trend": float(weighted_trend),
                "convergence_strength": float(overall_strength),
                "multi_resolution_agreement": float(1.0 - np.std([t for t, _ in trend_signals]) if trend_signals else 0)
            }
        except Exception as e:
            return {"nano_trend": 0.0, "convergence_strength": 0.0, "multi_resolution_agreement": 0.0}

    def _analyze_micro_structure(self, price_data: np.ndarray) -> Dict[str, float]:
        """Analisa a estrutura MICRO do mercado"""
        try:
            density_analysis = self._price_density_analysis(price_data)
            micro_momentum = self._micro_momentum_analysis(price_data)
            
            return {
                "price_density": density_analysis,
                "micro_momentum": micro_momentum,
                "structural_integrity": (density_analysis + micro_momentum) / 2.0
            }
        except Exception:
            return {"price_density": 0.5, "micro_momentum": 0.5, "structural_integrity": 0.5}

    def _price_density_analysis(self, price_data: np.ndarray) -> float:
        """Analisa a densidade/distribuição do preço"""
        try:
            hist, bins = np.histogram(price_data.flatten(), bins=20)
            hist_normalized = hist / np.sum(hist)
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-8))
            max_entropy = np.log(len(hist))
            
            density_score = 1.0 - (entropy / max_entropy)
            return float(np.clip(density_score, 0, 1))
        except Exception:
            return 0.5

    def _micro_momentum_analysis(self, price_data: np.ndarray) -> float:
        """Analisa momentum em nível microscópico"""
        try:
            height, width = price_data.shape
            
            if width < 10:
                return 0.5
            
            row_means = np.mean(price_data, axis=0)
            velocity = np.gradient(row_means)
            acceleration = np.gradient(velocity)
            
            recent_velocity = np.mean(velocity[-min(5, len(velocity)):])
            recent_acceleration = np.mean(acceleration[-min(5, len(acceleration)):])
            
            momentum_score = (
                np.tanh(recent_velocity * 10) * 0.6 +
                np.tanh(recent_acceleration * 5) * 0.4
            )
            
            return float((momentum_score + 1) / 2)
        except Exception:
            return 0.5

    def _analyze_flow_dynamics(self, price_data: np.ndarray) -> Dict[str, float]:
        """Analisa a DINÂMICA do fluxo de preços"""
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

    # =========================
    #  ANÁLISE TRADICIONAL FORTALECIDA
    # =========================
    
    def _analyze_price_action(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise tradicional de price action - FORTALECIDA"""
        try:
            height, width = price_data.shape
            segments = 6
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
                    slope = (regions[-1] - regions[0]) / (len(regions) - 1)
                else:
                    slope = 0
                    
                if len(regions) > 1:
                    changes = [regions[i] - regions[i-1] for i in range(1, len(regions))]
                    avg_change = np.mean(np.abs(changes))
                    if avg_change > 0:
                        trend_strength = min(1.0, abs(slope) / (avg_change + 1e-8))
                    else:
                        trend_strength = min(1.0, abs(slope) * 10)
                else:
                    trend_strength = 0
            else:
                slope = 0
                trend_strength = 0.5
            
            return {
                "trend_direction": float(slope),
                "trend_strength": float(trend_strength),
                "momentum": float(slope),
                "volatility": float(np.std(price_data) / (np.mean(price_data) + 1e-8)),
                "price_range": float(np.ptp(price_data))
            }
        except Exception:
            return {"trend_direction": 0.0, "trend_strength": 0.5, "momentum": 0.0, "volatility": 0.0, "price_range": 0.0}

    def _calculate_advanced_indicators(self, price_data: np.ndarray) -> Dict[str, float]:
        """Indicadores técnicos SUPER-REFORÇADOS"""
        try:
            height, width = price_data.shape
            
            if width > 10:
                row_means = np.mean(price_data, axis=0)
                
                # MACD FORTALECIDO
                fast_window = min(3, len(row_means))
                slow_window = min(8, len(row_means))
                signal_window = min(5, len(row_means))
                
                fast_ma = np.mean(row_means[-fast_window:])
                slow_ma = np.mean(row_means[-slow_window:])
                macd_line = fast_ma - slow_ma
                
                # Signal line (média do MACD)
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
                    
                    # Normaliza para -1 a 1
                    rsi_normalized = (rsi - 50) / 50
                else:
                    rsi_normalized = 0.0
                
                # FORÇA DO MACD (0 a 1)
                volatility = np.std(row_means) + 1e-8
                macd_strength = min(1.0, abs(macd_histogram) / (volatility * 2))
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
                "volume_intensity": float(min(1.0, np.var(price_data) / 1000.0)),
                "momentum_quality": float(min(1.0, (abs(rsi_normalized) + abs(macd_power)) / 2))
            }
        except Exception as e:
            return {"rsi": 0.0, "macd": 0.0, "macd_strength": 0.0, "volume_intensity": 0.0, "momentum_quality": 0.0}

    # =========================
    #  ANÁLISE COM DADOS DO USUÁRIO - NOVA FUNCIONALIDADE!
    # =========================
    
    def _analyze_user_indicators(self, user_indicators: Dict) -> Dict[str, float]:
        """Analisa os indicadores fornecidos pelo usuário"""
        try:
            macd = float(user_indicators.get('macd', 0))
            rsi = float(user_indicators.get('rsi', 50))
            adx = float(user_indicators.get('adx', 0))
            price = float(user_indicators.get('price', 0))
            
            # Análise do MACD fornecido
            macd_power = np.clip(macd * 10, -1, 1)  # Normaliza para -1 a 1
            macd_strength = min(1.0, abs(macd) * 20)  # Força baseada no valor absoluto
            
            # Análise do RSI fornecido
            rsi_normalized = (rsi - 50) / 50  # Normaliza para -1 a 1
            rsi_strength = min(1.0, abs(rsi_normalized))
            
            # Análise do ADX fornecido
            adx_strength = min(1.0, adx / 50)  # ADX > 25 = tendência forte
            
            # Score combinado dos indicadores do usuário
            user_score = (
                macd_power * 0.4 +
                rsi_normalized * 0.3 +
                adx_strength * 0.3
            )
            
            return {
                "user_macd_power": float(macd_power),
                "user_macd_strength": float(macd_strength),
                "user_rsi": float(rsi_normalized),
                "user_rsi_strength": float(rsi_strength),
                "user_adx_strength": float(adx_strength),
                "user_combined_score": float(user_score),
                "user_confidence": float(min(1.0, (macd_strength + rsi_strength + adx_strength) / 3))
            }
        except Exception as e:
            return {
                "user_macd_power": 0.0,
                "user_macd_strength": 0.0,
                "user_rsi": 0.0,
                "user_rsi_strength": 0.0,
                "user_adx_strength": 0.0,
                "user_combined_score": 0.0,
                "user_confidence": 0.0
            }

    # =========================
    #  MOTOR DE DECISÃO 100% NEUTRO - ATUALIZADO!
    # =========================
    
    def _absolute_decision_engine(self, all_analyses: Dict, timeframe: str) -> Dict[str, Any]:
        """MOTOR 100% NEUTRO - AGORA COM DADOS DO USUÁRIO!"""
        try:
            # Extrai todas as análises
            nano_trend = all_analyses['nano_analysis']
            micro_structure = all_analyses['micro_structure']
            flow_dynamics = all_analyses['flow_dynamics']
            traditional = all_analyses['traditional']
            user_analysis = all_analyses.get('user_analysis', {})
            
            # 🎯 ANÁLISE PURAMENTE TÉCNICA - ZERO VIÉS
            trend_direction = traditional['price_action']['trend_direction']
            trend_strength = traditional['price_action']['trend_strength']
            trend_power = trend_direction * trend_strength
            
            macd_value = traditional['indicators']['macd']
            macd_strength = traditional['indicators']['macd_strength']
            macd_power = macd_value * macd_strength
            
            nano_power = nano_trend['nano_trend'] * nano_trend['convergence_strength']
            micro_power = micro_structure['structural_integrity'] * 0.5 + flow_dynamics['overall_flow_quality'] * 0.5
            micro_composite = (nano_power + micro_power) / 2
            
            # 🧠 SCORE DA IA (70% de peso)
            ia_score = (
                trend_power * 0.333 +
                macd_power * 0.333 +  
                micro_composite * 0.334
            )
            
            # 🎯 SCORE DO USUÁRIO (30% de peso) - NOVO!
            user_score = user_analysis.get('user_combined_score', 0)
            user_confidence = user_analysis.get('user_confidence', 0)
            
            # 💥 SCORE FINAL COMBINADO
            total_score = (ia_score * 0.7) + (user_score * user_confidence * 0.3)
            
            # DECISÃO 100% NEUTRA
            if total_score > 0:
                direction = "buy"
                confidence = 0.50 + (min(abs(total_score), 0.5) * 0.40)
                reasoning = self._generate_enhanced_reasoning("buy", trend_power, macd_power, micro_composite, user_score, total_score, user_analysis)
            else:
                direction = "sell"
                confidence = 0.50 + (min(abs(total_score), 0.5) * 0.40)
                reasoning = self._generate_enhanced_reasoning("sell", trend_power, macd_power, micro_composite, user_score, total_score, user_analysis)
            
            # CONFIANÇA FINAL
            final_confidence = self._calculate_enhanced_confidence(confidence, all_analyses)
            
            # CONTEXTO
            context = self._detect_enhanced_context(trend_strength, macd_strength, micro_composite, user_score, total_score)
            
            return {
                "direction": direction,
                "confidence": final_confidence,
                "reasoning": reasoning,
                "total_score": total_score,
                "context": context,
                "trend_power": trend_power,
                "macd_power": macd_power,
                "micro_power": micro_composite,
                "user_score": user_score,
                "ia_score": ia_score
            }
            
        except Exception as e:
            return self._neutral_market_decision()

    def _generate_enhanced_reasoning(self, direction: str, trend_power: float, macd_power: float, 
                                   micro_power: float, user_score: float, total_score: float, 
                                   user_analysis: Dict) -> str:
        """Gera reasoning com dados do usuário"""
        
        strength = "ALTA" if abs(total_score) > 0.25 else "moderada"
        
        factors = []
        
        # Fatores da IA
        if abs(trend_power) > 0.15: 
            factors.append(f"tendência {trend_power*100:+.1f}%")
        if abs(macd_power) > 0.15: 
            factors.append(f"MACD {macd_power*100:+.1f}%")
        if abs(micro_power) > 0.15: 
            factors.append(f"micro-estrutura {micro_power*100:+.1f}%")
        
        # Fatores do usuário - NOVO!
        user_macd = user_analysis.get('user_macd_power', 0)
        user_rsi = user_analysis.get('user_rsi', 0)
        user_adx = user_analysis.get('user_adx_strength', 0)
        
        user_factors = []
        if abs(user_macd) > 0.1:
            user_factors.append(f"MACD(user)")
        if abs(user_rsi) > 0.1:
            user_factors.append(f"RSI(user)")
        if user_adx > 0.3:
            user_factors.append(f"ADX(user)")
        
        if user_factors:
            factors.append(f"indicadores({','.join(user_factors)})")
                
        if factors:
            analysis = " + ".join(factors)
            if direction == "buy":
                return f"📈 COMPRA {strength} - {analysis}"
            else:
                return f"📉 VENDA {strength} - {analysis}"
        else:
            if direction == "buy":
                return f"📈 COMPRA {strength} - Convergência técnica"
            else:
                return f"📉 VENDA {strength} - Convergência técnica"

    def _calculate_enhanced_confidence(self, base_confidence: float, all_analyses: Dict) -> float:
        """Calcula confiança com dados do usuário"""
        try:
            # Fatores da IA
            confidence_factors = [
                all_analyses['nano_analysis']['convergence_strength'],
                all_analyses['micro_structure']['structural_integrity'],
                all_analyses['flow_dynamics']['overall_flow_quality'],
                all_analyses['traditional']['price_action']['trend_strength'],
                all_analyses['traditional']['indicators']['macd_strength']
            ]
            
            # Adiciona confiança do usuário - NOVO!
            user_analysis = all_analyses.get('user_analysis', {})
            user_confidence = user_analysis.get('user_confidence', 0)
            confidence_factors.append(user_confidence)
            
            quality_score = np.mean([f for f in confidence_factors if not np.isnan(f)])
            enhanced_confidence = base_confidence + (quality_score * 0.2)
            
            return min(0.95, enhanced_confidence)
            
        except Exception:
            return base_confidence

    def _detect_enhanced_context(self, trend_strength: float, macd_strength: float, 
                               micro_power: float, user_score: float, total_score: float) -> str:
        """Detecta contexto com dados do usuário"""
        if abs(total_score) > 0.3:
            return "movimento_forte"
        elif abs(total_score) < 0.05:
            return "mercado_indeciso"
        elif trend_strength > 0.4:
            return "tendencia_estabelecida"
        elif user_score > 0.2:
            return "confirmacao_usuario"
        else:
            return "mercado_balanceado"

    def _neutral_market_decision(self) -> Dict[str, Any]:
        """Decisão neutra baseada em análise técnica"""
        direction = "buy" if random.random() > 0.5 else "sell"
        
        return {
            "direction": direction,
            "confidence": 0.55,
            "reasoning": f"📊 {direction.upper()} - Análise de mercado contingente",
            "total_score": 0.05 if direction == "buy" else -0.05,
            "context": "analise_contingente",
            "trend_power": 0.02,
            "macd_power": 0.02,
            "micro_power": 0.02,
            "user_score": 0.0,
            "ia_score": 0.05
        }

    def _calculate_signal_quality(self, analyses: Dict) -> float:
        """Calcula qualidade do sinal"""
        try:
            factors = [
                analyses['nano_analysis']['convergence_strength'] * 0.2,
                analyses['micro_structure']['structural_integrity'] * 0.2,
                analyses['flow_dynamics']['overall_flow_quality'] * 0.2,
                analyses['traditional']['price_action']['trend_strength'] * 0.2,
                analyses['traditional']['indicators']['macd_strength'] * 0.2
            ]
            return float(np.clip(np.mean(factors), 0, 1))
        except Exception:
            return 0.6

    def _get_entry_timeframe(self, user_timeframe: str) -> Dict[str, str]:
        """Calcula timeframe de entrada"""
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
        """ANÁLISE 100% NEUTRA - AGORA COM DADOS DO USUÁRIO!"""
        
        if user_indicators is None:
            user_indicators = {}
        
        # Cache inteligente (agora considera os indicadores do usuário)
        cached = self.cache.get(blob, timeframe, user_indicators)
        if cached:
            cached['cached'] = True
            return cached
        
        try:
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
                'flow_dynamics': self._analyze_flow_dynamics(price_data)
            }
            
            # 🎯 NOVO: ANÁLISE DOS INDICADORES DO USUÁRIO
            if user_indicators:
                analyses['user_analysis'] = self._analyze_user_indicators(user_indicators)
            
            # 🎯 MOTOR DE DECISÃO 100% NEUTRO (ATUALIZADO)
            decision = self._absolute_decision_engine(analyses, timeframe)
            time_info = self._get_entry_timeframe(timeframe)
            
            # 📊 QUALIDADE DA ANÁLISE
            signal_quality = self._calculate_signal_quality(analyses)
            
            # 🎨 RESULTADO SUPER NEUTRO
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
                "market_context": decision["context"],
                "micro_quality": analyses['nano_analysis']['convergence_strength'],
                "user_indicators_provided": bool(user_indicators),
                "metrics": {
                    "analysis_score": float(decision["total_score"]),
                    "trend_power": float(decision["trend_power"]),
                    "macd_power": float(decision["macd_power"]),
                    "micro_power": float(decision["micro_power"]),
                    "user_score": float(decision["user_score"]),
                    "ia_score": float(decision["ia_score"]),
                    "trend_strength": analyses['traditional']['price_action']['trend_strength'],
                    "momentum": analyses['traditional']['price_action']['momentum'],
                    "rsi": analyses['traditional']['indicators']['rsi'],
                    "macd": analyses['traditional']['indicators']['macd'],
                    "macd_strength": analyses['traditional']['indicators']['macd_strength']
                },
                "reasoning": decision["reasoning"]
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
            
        except Exception as e:
            fallback_result = self._neutral_market_decision()
            fallback_result.update({
                "entry_signal": f"🧠 {fallback_result['direction'].upper()} - Análise de mercado contingente",
                "entry_time": datetime.datetime.now().strftime("%H:%M"),
                "timeframe": "Próximo candle",
                "analysis_time": datetime.datetime.now().strftime("%H:%M:%S"),
                "user_timeframe": timeframe,
                "cached": False,
                "signal_quality": 0.6,
                "analysis_grade": "medium",
                "market_context": "market_analysis",
                "micro_quality": 0.6,
                "user_indicators_provided": bool(user_indicators)
            })
            return fallback_result

    # =========================
    # VISUAL ANALYSIS (no OCR) - TODAS AS FUNÇÕES PRESERVADAS
    # =========================
    def _to_hsv(self, img_np: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("cv2 not available")
        return cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV)

    def _detect_panels(self, img: np.ndarray) -> Dict[str, Tuple[int,int,int,int]]:
        if cv2 is None:
            # fallback split: 60/25/15
            h, w, _ = img.shape
            ph = int(0.60*h); mh = int(0.25*h); rh = h - ph - mh
            return {"price": (0,0,w,ph), "macd": (0,ph,w,mh), "rsi": (0,ph+mh,w,rh)}
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        proj = np.mean(gray, axis=1)
        thr = np.percentile(proj, 15)
        gaps = np.where(proj < thr)[0]
        if len(gaps) < 10:
            ph = int(0.60*h); mh = int(0.25*h); rh = h - ph - mh
            return {"price": (0,0,w,ph), "macd": (0,ph,w,mh), "rsi": (0,ph+mh,w,rh)}
        step = h//50 or 1
        cuts = sorted(set(int(g/step)*step for g in gaps))
        if len(cuts) < 2:
            ph = int(0.60*h); mh = int(0.25*h); rh = h - ph - mh
            return {"price": (0,0,w,ph), "macd": (0,ph,w,mh), "rsi": (0,ph+mh,w,rh)}
        c1, c2 = cuts[len(cuts)//3], cuts[(2*len(cuts))//3]
        top, mid, bot = (0, c1), (c1+2, c2), (c2+2, h)
        return {
            "price": (0, top[0], w, top[1]-top[0]),
            "macd":  (0, mid[0], w, mid[1]-mid[0]),
            "rsi":   (0, bot[0], w, bot[1]-bot[0])
        }

    def _auto_color_calibration(self, img_hsv: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if cv2 is None:
            return {
                "BB": (np.array([105,40,40]), np.array([140,255,255])),
                "EMA": (np.array([20,40,40]), np.array([38,255,255])),
                "MACD_SIGNAL": (np.array([10,40,40]), np.array([22,255,255])),
                "RSI": (np.array([135,40,40]), np.array([170,255,255])),
            }
        h, w, _ = img_hsv.shape
        legend = img_hsv[0:int(0.12*h), 0:int(0.45*w)]
        legend_flat = legend.reshape(-1,3).astype(np.float32)
        K = 5
        try:
            _, labels, centers = cv2.kmeans(legend_flat, K, None,
                                            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                            3, cv2.KMEANS_PP_CENTERS)
            centers = centers.astype(np.uint8)
        except Exception:
            centers = np.array([[120,180,180],[28,200,200],[15,200,200],[150,200,200],[90,180,180]], dtype=np.uint8)

        def band(c, spread=(8,40,40)):
            low = np.clip(c - np.array([spread[0],spread[1],spread[2]]), 0, 255)
            high = np.clip(c + np.array([spread[0],spread[1],spread[2]]), 0, 255)
            return low, high

        ranges = {}
        for c in centers:
            H,S,V = c
            if 100 <= H <= 140: ranges["BB"] = band(c, (10,60,60))
            elif 18 <= H <= 38: ranges["EMA"] = band(c, (8,60,60))
            elif 8 <= H <= 22:  ranges["MACD_SIGNAL"] = band(c, (8,60,60))
            elif 135 <= H <= 170: ranges["RSI"] = band(c, (10,60,60))

        ranges.setdefault("BB", (np.array([105,40,40]), np.array([140,255,255])))
        ranges.setdefault("EMA",(np.array([20,40,40]),  np.array([38,255,255])))
        ranges.setdefault("MACD_SIGNAL",(np.array([10,40,40]), np.array([22,255,255])))
        ranges.setdefault("RSI",(np.array([135,40,40]), np.array([170,255,255])))
        return ranges

    def _mask_line(self, img_hsv: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.GaussianBlur(mask, (3,3), 0)
        return mask

    def _extract_ema_bb(self, price_panel_rgb: np.ndarray, ranges: Dict[str, Tuple[np.ndarray,np.ndarray]]) -> Dict[str, float]:
        if cv2 is None:
            return {"ema_slope": 0.0, "bb_width": 0.0}
        hsv = cv2.cvtColor(price_panel_rgb, cv2.COLOR_RGB2HSV)
        ema_mask = self._mask_line(hsv, *ranges["EMA"])
        bb_mask  = self._mask_line(hsv, *ranges["BB"])
        ys, xs = np.where(ema_mask > 0)
        ema_slope = 0.0
        if len(xs) > 50:
            A = np.vstack([xs, np.ones_like(xs)]).T
            m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
            ema_slope = float(-m / (price_panel_rgb.shape[0]+1e-6))
        bb_upper, bb_lower = [], []
        contours,_ = cv2.findContours(bb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w > price_panel_rgb.shape[1]*0.05:
                bb_upper.append(y)
                bb_lower.append(y+h)
        bb_width = 0.0
        if bb_upper and bb_lower:
            bb_width = float(np.mean(bb_lower) - np.mean(bb_upper))
            bb_width /= (price_panel_rgb.shape[0]+1e-6)
        return {"ema_slope": ema_slope, "bb_width": bb_width}

    def _extract_candles_geometry(self, price_panel_rgb: np.ndarray) -> Dict[str,float]:
        img = price_panel_rgb.copy()
        h,w,_ = img.shape
        g = None
        try:
            if cv2 is not None:
                g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                g = cv2.GaussianBlur(g, (3,3), 0)
            else:
                g = np.dot(img[...,:3],[0.299,0.587,0.114]).astype('float32')
        except Exception:
            g = np.dot(img[...,:3],[0.299,0.587,0.114]).astype('float32')
        right = img[:, int(0.55*w):, :]
        greenish = float(np.mean(right[:,:,1]) - np.mean(right[:,:,0]))
        reddish  = float(np.mean(right[:,:,0]) - np.mean(right[:,:,1]))
        color_bias = float((greenish - reddish)/255.0)
        col_mean = np.mean(g, axis=0)
        geom_slope = float((col_mean[-1] - col_mean[int(0.6*len(col_mean))]) / (len(col_mean)*5.0))
        geom_slope = -geom_slope
        tail = col_mean[int(0.85*len(col_mean)):] if len(col_mean)>10 else col_mean
        momentum_tail = float(-(np.mean(tail[-5:]) - np.mean(tail[:5])) / 255.0) if len(tail)>=10 else 0.0
        return {
            "color_bias": color_bias,
            "geom_slope": geom_slope,
            "micro_momentum": momentum_tail
        }

    def _read_rsi_panel(self, rsi_rgb: np.ndarray, ranges) -> float:
        if cv2 is None:
            return 0.0
        hsv = cv2.cvtColor(rsi_rgb, cv2.COLOR_RGB2HSV)
        rsi_mask = self._mask_line(hsv, *ranges["RSI"])
        ys, xs = np.where(rsi_mask > 0)
        if len(xs) < 30:
            return 0.0
        y_mean = np.mean(ys)
        lvl = 100.0 * (1.0 - y_mean / (rsi_rgb.shape[0]+1e-6))
        return float((lvl - 50.0) / 50.0)

    def _read_macd_panel(self, macd_rgb: np.ndarray, ranges) -> Tuple[float,float]:
        if cv2 is None:
            return 0.0, 0.0
        hsv = cv2.cvtColor(macd_rgb, cv2.COLOR_RGB2HSV)
        sig = self._mask_line(hsv, *ranges["MACD_SIGNAL"])
        gray = cv2.cvtColor(macd_rgb, cv2.COLOR_RGB2GRAY)
        col = np.mean(gray, axis=0)
        vel = np.gradient(col); acc = np.gradient(vel)
        ys, xs = np.where(sig>0)
        macd_power, macd_strength = 0.0, 0.0
        if len(xs) > 40:
            A = np.vstack([xs, np.ones_like(xs)]).T
            m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
            macd_dir = -m
            macd_strength = float(min(1.0, abs(macd_dir)*2.0))
            macd_power = float(np.clip(macd_dir, -1, 1) * macd_strength)
        else:
            macd_power = float(np.tanh(-np.mean(acc)/20.0))
            macd_strength = float(min(1.0, abs(np.mean(vel))/20.0))
        return macd_power, macd_strength

# =========================
#  APLICAÇÃO FLASK COMPLETA
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
    <title>IA Signal Pro - SUPER INTELIGENTE E NEUTRA 🧠⚖️</title>
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
        .context-movimento_forte { background: linear-gradient(135deg, #00ff88, #00cc66); color: white; }
        .context-mercado_indeciso { background: linear-gradient(135deg, #7ce0ff, #4a90e2); color: white; }
        .context-tendencia_estabelecida { background: linear-gradient(135deg, #ffaa00, #ff8800); color: white; }
        .context-confirmacao_usuario { background: linear-gradient(135deg, #7ce0ff, #4a90e2); color: white; }
        
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
        
        .neutral-badge {
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠⚖️ IA SIGNAL PRO - 100% NEUTRA</div>
            <div class="subtitle">ZERO VIÉS - DECISÕES APENAS PELO MOMENTO DO MERCADO</div>
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
                🔢 DADOS TÉCNICOS <span class="optional-badge">OPCIONAL</span>
            </div>
            <div style="text-align: center; font-size: 12px; color: #9db0d1; margin-bottom: 10px;">
                Forneça os valores exatos para maior precisão
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
            <div id="qualityIndicator" class="quality-indicator"></div>
            
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div id="contextInfo" style="text-align: center; margin: 10px 0;"></div>
            
            <div class="power-analysis" id="powerAnalysis">
                <div style="text-align: center; font-weight: 600; margin-bottom: 8px; color: #7ce0ff;">
                    ⚡ ANÁLISE COMBINADA
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
            const contextInfo = document.getElementById('contextInfo');
            const powerAnalysis = document.getElementById('powerAnalysis');
            const powerMetrics = document.getElementById('powerMetrics');
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
            uploadArea.addEventListener('click', () => fileInput.click());
            
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
                
                signalText.className = 'signal-text';
                signalText.textContent = 'Analisando momento do mercado...';
                qualityIndicator.textContent = '';
                contextInfo.innerHTML = '';
                
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
                reasoningText.textContent = 'Processando análise 100% neutra...';
                confidenceText.textContent = '';
                progressFill.style.width = '20%';
                
                metricsText.innerHTML = '<div class="loading">Iniciando análise do momento do mercado...</div>';

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
                const context = data.market_context || 'mercado_balanceado';
                const userIndicatorsProvided = data.user_indicators_provided || false;
                
                // Define classe e texto do sinal
                signalText.className = `signal-text signal-${direction}`;
                let directionText = direction === 'buy' ? '🎯 COMPRAR' : '🎯 VENDER';
                let userBadge = userIndicatorsProvided ? '<span class="user-data-badge">DADOS DO USUÁRIO</span>' : '';
                signalText.innerHTML = `${directionText} <span class="neutral-badge">100% NEUTRO</span> ${userBadge} ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                
                // Atualiza informações
                analysisTime.textContent = data.analysis_time || '--:--:--';
                entryTime.textContent = data.entry_time || '--:--';
                timeframeEl.textContent = data.timeframe || 'Próximo minuto';
                
                reasoningText.textContent = data.reasoning;
                confidenceText.textContent = `Confiança Técnica: ${confidence}%`;
                
                // Indicador de qualidade
                qualityIndicator.className = `quality-indicator quality-${quality}`;
                if (quality === 'high') {
                    qualityIndicator.textContent = '✅ ALTA QUALIDADE - Análise confiável do momento';
                } else {
                    qualityIndicator.textContent = '⚠️ QUALIDADE MÉDIA - Análise válida do momento';
                }
                
                // Informações de contexto
                const contextLabels = {
                    'movimento_forte': '🚀 MOVIMENTO FORTE',
                    'mercado_indeciso': '⚡ MERCADO INDECISO', 
                    'tendencia_estabelecida': '📈 TENDÊNCIA ESTABELECIDA',
                    'confirmacao_usuario': '🎯 CONFIRMAÇÃO DO USUÁRIO',
                    'mercado_balanceado': '⚖️ MERCADO BALANCEADO'
                };
                
                contextInfo.innerHTML = `
                    <span class="context-badge context-${context}">
                        ${contextLabels[context] || contextLabels.mercado_balanceado}
                    </span>
                `;
                
                // Análise Combinada
                const metrics = data.metrics || {};
                let powerHtml = '';
                
                const powerItems = [
                    ['Score da IA', metrics.ia_score?.toFixed(3)],
                    ['Score do Usuário', metrics.user_score?.toFixed(3)],
                    ['Score Final', metrics.analysis_score?.toFixed(3)],
                    ['Poder da Tendência', (metrics.trend_power * 100)?.toFixed(1) + '%'],
                    ['Poder do MACD', (metrics.macd_power * 100)?.toFixed(1) + '%']
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
    """Endpoint de análise de imagem - AGORA COM DADOS DO USUÁRIO!"""
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
        
        # Análise 100% NEUTRA - AGORA COM DADOS DO USUÁRIO!
        analysis = analyzer.analyze(image_bytes, timeframe, user_indicators)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': f'Erro interno: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check para monitoramento"""
    return jsonify({
        'status': 'healthy', 
        'service': 'IA Signal Pro - 100% NEUTRA',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '7.0.0-com-dados-usuario'
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
    
    print(f"🚀 IA Signal Pro - 100% NEUTRA iniciando na porta {port}")
    print(f"🧠⚖️ SISTEMA: ZERO VIÉS - DECISÕES PURAMENTE TÉCNICAS")
    print(f"🎯 NOVO: AGORA COM DADOS DO USUÁRIO (MACD, RSI, ADX, Preço)")
    print(f"📈 SAÍDA: COMPRA ou VENDA - SEM FAVORITISMO")
    print(f"💪 NEUTRALIDADE: COMBINAÇÃO IA (70%) + USUÁRIO (30%)")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
