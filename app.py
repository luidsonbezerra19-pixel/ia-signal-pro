from __future__ import annotations

"""
IA Signal Pro — Análise INTELIGENTE PURA - VERSÃO MAIS ASSERTIVA
Sistema otimizado para maior precisão sem perder sensibilidade
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
#  IA INTELIGENTE PURA - VERSÃO MAIS ASSERTIVA
# =========================
class IntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
        self.min_confidence_threshold = 0.58  # Aumentado para maior assertividade
    
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
        """Validação mais rigorosa do gráfico"""
        width, height = image.size
        
        # Verifica dimensões mínimas aumentadas
        if width < 250 or height < 180:
            raise ValueError("Imagem muito pequena para análise (mínimo 250x180 pixels)")
        
        # Verificação de contraste mais rigorosa
        try:
            img_array = np.array(image)
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            contrast = np.std(gray)
            if contrast < 20:  # Aumentado o limite mínimo de contraste
                raise ValueError("Contraste insuficiente - gráfico não legível")
            
            # Verifica se há variação suficiente nas cores
            color_variance = np.var(img_array)
            if color_variance < 500:
                raise ValueError("Pouca variação de cores - imagem pode não ser um gráfico")
            
            return True
        except Exception as e:
            raise ValueError(f"Erro na validação da imagem: {str(e)}")

    def _preprocess_image(self, image: Image.Image, timeframe: str) -> np.ndarray:
        """Pré-processamento otimizado para melhor análise"""
        width, height = image.size
        
        # Redimensionamento baseado na qualidade
        base_size = min(width, height)
        if base_size < 400:
            target_size = (600, 450)
        else:
            target_size = (800, 600)
            
        image = image.resize(target_size, Image.LANCZOS)
        
        # Aplica filtros para realce de características
        image = image.filter(ImageFilter.SMOOTH_MORE)
        image = image.filter(ImageFilter.SHARPEN)
        
        return np.array(image)

    def _extract_price_data(self, img_array: np.ndarray) -> np.ndarray:
        """Extrai dados de preço com melhor precisão"""
        try:
            # Converte para escala de cinza com pesos otimizados
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Realce de bordas melhorado
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            sobel_x = self._apply_convolution(gray, kernel_x)
            sobel_y = self._apply_convolution(gray, kernel_y)
            gradient = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Combinação otimizada para melhor detecção
            enhanced = gray * 0.6 + gradient * 0.4
            return enhanced
        except Exception as e:
            raise ValueError(f"Erro na extração de dados: {str(e)}")

    def _apply_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Aplica convolução manualmente sem scipy"""
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
            
            return output
        except Exception as e:
            raise ValueError(f"Erro na convolução: {str(e)}")

    def _analyze_price_action(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise de price action mais assertiva"""
        try:
            height, width = price_data.shape
            
            # Análise multi-temporal
            regions = []
            segment_size = max(2, width // 6)  # Menos regiões para mais consistência
            
            for i in range(6):
                start_col = i * segment_size
                end_col = min((i + 1) * segment_size, width)
                    
                segment = price_data[:, start_col:end_col]
                if segment.size > 0:
                    # Usa mediana para reduzir ruído
                    region_median = np.median(segment)
                    regions.append(region_median)
            
            # Análise de tendência com validação
            if len(regions) >= 3:
                x = np.arange(len(regions))
                trend_slope, trend_intercept = np.polyfit(x, regions, 1)
                
                # Calcula R² para validar tendência
                y_pred = trend_slope * x + trend_intercept
                ss_res = np.sum((regions - y_pred) ** 2)
                ss_tot = np.sum((regions - np.mean(regions)) ** 2)
                
                if ss_tot > 0:
                    trend_strength = 1 - (ss_res / ss_tot)
                else:
                    trend_strength = 0
                    
                # Filtro de tendência: só considera se R² > 0.3
                if abs(trend_slope) > 0.1 and trend_strength > 0.3:
                    validated_trend_slope = trend_slope
                    validated_trend_strength = trend_strength
                else:
                    validated_trend_slope = 0
                    validated_trend_strength = 0
            else:
                validated_trend_slope = 0
                validated_trend_strength = 0
            
            # Momentum com suavização
            if len(regions) >= 4:
                # Suaviza as regiões antes do cálculo
                smoothed_regions = np.convolve(regions, [0.25, 0.5, 0.25], mode='valid')
                if len(smoothed_regions) >= 2:
                    momentum = np.gradient(smoothed_regions)
                    current_momentum = momentum[-1] if len(momentum) > 0 else 0
                else:
                    current_momentum = 0
            else:
                current_momentum = 0
            
            # Volatilidade normalizada
            price_mean = np.mean(price_data)
            volatility = np.std(price_data) / (price_mean + 1e-8) if price_mean > 0 else 0
            
            return {
                "trend_direction": float(validated_trend_slope),
                "trend_strength": float(min(1.0, max(0.0, validated_trend_strength))),
                "momentum": float(current_momentum),
                "volatility": float(volatility),
                "price_range": float(np.ptp(price_data)),
                "price_stability": float(1.0 - min(1.0, volatility * 2))  # Nova métrica
            }
        except Exception as e:
            return {
                "trend_direction": 0.0,
                "trend_strength": 0.0,
                "momentum": 0.0,
                "volatility": 0.0,
                "price_range": 0.0,
                "price_stability": 0.0
            }

    def _analyze_chart_patterns(self, price_data: np.ndarray) -> Dict[str, float]:
        """Detecção de padrões mais confiável"""
        try:
            height, width = price_data.shape
            
            # Análise de níveis com tolerância adaptativa
            horizontal_profiles = []
            step = max(2, height // 40)  # Menos pontos, mais significativos
            
            price_std = np.std(price_data)
            congestion_threshold = price_std * 0.25  # Mais seletivo
            
            for row in range(0, height, step):
                row_data = price_data[row, :]
                if len(row_data) > 8:
                    row_variance = np.std(row_data)
                    if row_variance < congestion_threshold:
                        horizontal_profiles.append(np.median(row_data))
            
            # Agrupamento mais inteligente
            unique_levels = []
            threshold = price_std * 0.15  # Mais rigoroso
            
            for level in sorted(horizontal_profiles):
                if not unique_levels:
                    unique_levels.append(level)
                else:
                    # Só adiciona se for significativamente diferente
                    min_distance = min(abs(level - lvl) for lvl in unique_levels)
                    if min_distance > threshold:
                        unique_levels.append(level)
            
            # Preço atual (média das últimas colunas)
            current_window = min(15, width)
            current_price = np.median(price_data[:, -current_window:])
            
            # Classificação de níveis com validação
            supports = []
            resistances = []
            
            for level in unique_levels:
                distance_pct = abs(level - current_price) / (current_price + 1e-8)
                
                # Só considera níveis próximos (até 5%)
                if distance_pct <= 0.05:
                    if level < current_price:
                        supports.append(level)
                    else:
                        resistances.append(level)
            
            # Força dos níveis baseada na proximidade e quantidade
            support_strength = len(supports) / 10.0  # Normalizado
            resistance_strength = len(resistances) / 10.0
            
            # Distância aos níveis mais próximos
            if supports:
                nearest_support = max(supports)
                distance_to_support = abs(current_price - nearest_support) / (current_price + 1e-8)
            else:
                distance_to_support = 1.0
                
            if resistances:
                nearest_resistance = min(resistances)
                distance_to_resistance = abs(nearest_resistance - current_price) / (current_price + 1e-8)
            else:
                distance_to_resistance = 1.0
            
            # Nível de consolidação baseado na qualidade dos níveis
            consolidation_quality = min(1.0, len(unique_levels) / 8.0)
            
            return {
                "support_levels": len(supports),
                "resistance_levels": len(resistances),
                "support_strength": float(min(1.0, support_strength)),
                "resistance_strength": float(min(1.0, resistance_strength)),
                "distance_to_support": float(min(1.0, distance_to_support * 10)),  # Escalado
                "distance_to_resistance": float(min(1.0, distance_to_resistance * 10)),
                "consolidation_level": float(consolidation_quality),
                "levels_quality": float(min(1.0, len(unique_levels) / 12.0))  # Nova métrica
            }
        except Exception as e:
            return {
                "support_levels": 0,
                "resistance_levels": 0,
                "support_strength": 0.0,
                "resistance_strength": 0.0,
                "distance_to_support": 1.0,
                "distance_to_resistance": 1.0,
                "consolidation_level": 0.0,
                "levels_quality": 0.0
            }

    def _analyze_market_structure(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise de estrutura de mercado mais robusta"""
        try:
            height, width = price_data.shape
            
            if height < 3 or width < 3:
                return {
                    "market_trend": 0.0,
                    "volatility_ratio": 1.0,
                    "movement_strength": 0.0,
                    "structure_quality": 0.0,
                    "trend_consistency": 0.0
                }
            
            # Análise de múltiplas timeframe internas
            segments = 4
            segment_width = max(1, width // segments)
            segment_trends = []
            
            for i in range(segments):
                start_col = i * segment_width
                end_col = min((i + 1) * segment_width, width)
                segment = price_data[:, start_col:end_col]
                
                if segment.size > 0:
                    # Tendência do segmento
                    segment_flat = np.mean(segment, axis=0)
                    if len(segment_flat) > 1:
                        seg_trend = np.polyfit(range(len(segment_flat)), segment_flat, 1)[0]
                        segment_trends.append(seg_trend)
            
            # Consistência da tendência
            if segment_trends:
                trend_consistency = np.std(segment_trends)
                # Inverte: menor desvio = maior consistência
                trend_consistency = 1.0 / (1.0 + trend_consistency * 10)
                market_trend = np.mean(segment_trends)
            else:
                trend_consistency = 0.0
                market_trend = 0.0
            
            # Análise de força com validação temporal
            recent_segment = max(1, width // 6)
            older_segment = max(1, width // 3)
            
            if width > older_segment:
                recent_data = price_data[:, -recent_segment:]
                older_data = price_data[:, -older_segment:-recent_segment]
                
                if recent_data.size > 0 and older_data.size > 0:
                    recent_mean = np.mean(recent_data)
                    older_mean = np.mean(older_data)
                    movement = recent_mean - older_mean
                    movement_strength = abs(movement) / (np.std(price_data) + 1e-8)
                else:
                    movement_strength = 0.0
            else:
                movement_strength = 0.0
            
            # Qualidade da estrutura baseada em múltiplos fatores
            structure_quality = min(1.0, (
                trend_consistency * 0.4 +
                (height * width) / 200000.0 * 0.3 +
                min(1.0, movement_strength) * 0.3
            ))
            
            return {
                "market_trend": float(market_trend),
                "volatility_ratio": 1.0,  # Simplificado para focar no essencial
                "movement_strength": float(min(3.0, movement_strength)),
                "structure_quality": float(structure_quality),
                "trend_consistency": float(trend_consistency)
            }
        except Exception as e:
            return {
                "market_trend": 0.0,
                "volatility_ratio": 1.0,
                "movement_strength": 0.0,
                "structure_quality": 0.0,
                "trend_consistency": 0.0
            }

    def _calculate_advanced_indicators(self, price_data: np.ndarray) -> Dict[str, float]:
        """Indicadores técnicos mais confiáveis"""
        try:
            height, width = price_data.shape
            
            if width < 12:
                return {
                    "rsi": 0.0,
                    "macd": 0.0,
                    "volume_intensity": 0.0,
                    "momentum_quality": 0.0,
                    "trend_quality": 0.0
                }
            
            # RSI com períodos otimizados
            fast_period = max(3, width // 6)
            slow_period = max(6, width // 3)
            
            if width > slow_period:
                recent_avg = np.median(price_data[:, -fast_period:])
                older_avg = np.median(price_data[:, -slow_period:-fast_period])
                
                gain = max(0, recent_avg - older_avg)
                loss = max(0, older_avg - recent_avg)
                
                if loss == 0:
                    rsi = 70 if gain > 0 else 50
                else:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                
                # Normaliza para -1 a 1
                rsi_normalized = (rsi - 50) / 30  # Mais sensível
                rsi_normalized = max(-1.0, min(1.0, rsi_normalized))
            else:
                rsi_normalized = 0.0
            
            # MACD com suavização
            fast_len = max(2, width // 8)
            slow_len = max(4, width // 4)
            signal_len = max(2, width // 10)
            
            if width >= slow_len + signal_len:
                # EMAs simplificadas
                fast_ema = np.mean(price_data[:, -fast_len:])
                slow_ema = np.mean(price_data[:, -slow_len:])
                macd_line = fast_ema - slow_ema
                
                # Sinal (EMA do MACD)
                macd_values = []
                for i in range(signal_len):
                    start = -slow_len - i
                    end = -i if i > 0 else None
                    fast_val = np.mean(price_data[:, start:end][:, :fast_len])
                    slow_val = np.mean(price_data[:, start:end])
                    macd_values.append(fast_val - slow_val)
                
                signal_line = np.mean(macd_values) if macd_values else 0
                macd_histogram = macd_line - signal_line
                
                price_std = np.std(price_data) if np.std(price_data) > 0 else 1.0
                macd_normalized = macd_histogram / price_std
            else:
                macd_normalized = 0.0
            
            # Qualidade do trend baseada em múltiplos timeframes
            trend_quality = min(1.0, (abs(rsi_normalized) + min(1.0, abs(macd_normalized))) / 2)
            
            return {
                "rsi": float(rsi_normalized),
                "macd": float(max(-1.5, min(1.5, macd_normalized))),
                "volume_intensity": float(min(1.0, np.var(price_data) / 800.0)),
                "momentum_quality": float(min(1.0, abs(rsi_normalized) + abs(macd_normalized))),
                "trend_quality": float(trend_quality)
            }
        except Exception as e:
            return {
                "rsi": 0.0,
                "macd": 0.0,
                "volume_intensity": 0.0,
                "momentum_quality": 0.0,
                "trend_quality": 0.0
            }

    def _calculate_signal_quality(self, analysis: Dict) -> float:
        """Cálculo de qualidade mais rigoroso"""
        try:
            quality_factors = []
            
            # 1. Força e consistência da tendência (30%)
            trend_strength = analysis['price_action'].get('trend_strength', 0)
            trend_consistency = analysis['market_structure'].get('trend_consistency', 0)
            trend_quality = trend_strength * trend_consistency
            quality_factors.append(trend_quality * 0.30)
            
            # 2. Qualidade dos níveis de suporte/resistência (25%)
            support_strength = analysis['chart_patterns'].get('support_strength', 0)
            resistance_strength = analysis['chart_patterns'].get('resistance_strength', 0)
            levels_quality = analysis['chart_patterns'].get('levels_quality', 0)
            level_quality = (support_strength + resistance_strength + levels_quality) / 3
            quality_factors.append(level_quality * 0.25)
            
            # 3. Momentum e qualidade do trend (20%)
            momentum_quality = analysis['indicators'].get('momentum_quality', 0)
            trend_quality_ind = analysis['indicators'].get('trend_quality', 0)
            momentum_score = (momentum_quality + trend_quality_ind) / 2
            quality_factors.append(momentum_score * 0.20)
            
            # 4. Estabilidade do preço (15%)
            price_stability = analysis['price_action'].get('price_stability', 0)
            structure_quality = analysis['market_structure'].get('structure_quality', 0)
            stability_score = (price_stability + structure_quality) / 2
            quality_factors.append(stability_score * 0.15)
            
            # 5. Clareza do padrão (10%)
            pattern_clarity = analysis['chart_patterns'].get('consolidation_level', 0)
            quality_factors.append(pattern_clarity * 0.10)
            
            final_quality = sum(quality_factors)
            
            # Aplica filtro de qualidade mínima
            if final_quality < 0.3:
                final_quality *= 0.7  # Penaliza sinais de muito baixa qualidade
            
            return min(1.0, max(0.0, final_quality))
            
        except Exception:
            return 0.4  # Qualidade baixa em caso de erro

    def _make_intelligent_decision(self, analysis: Dict, timeframe: str) -> Dict[str, Any]:
        """Tomada de decisão mais assertiva e conservadora"""
        try:
            price_action = analysis['price_action']
            chart_patterns = analysis['chart_patterns']
            market_structure = analysis['market_structure']
            indicators = analysis['indicators']
            
            # Sistema de pontuação mais conservador
            score_components = []
            weight_explanations = []
            
            # 1. Tendência principal (25% - reduzido para dar mais peso a confirmações)
            trend_direction = price_action['trend_direction']
            trend_strength = price_action['trend_strength']
            trend_score = trend_direction * trend_strength
            score_components.append(trend_score * 0.25)
            weight_explanations.append(f"Trend: {trend_score:.2f}")
            
            # 2. Confirmação de momentum (20%)
            momentum = price_action['momentum']
            rsi = indicators['rsi']
            macd = indicators['macd']
            
            # Só considera momentum se confirmado por indicadores
            momentum_confirmation = 0
            if abs(momentum) > 0.05:  # Momentum significativo
                if (momentum > 0 and rsi > 0) or (momentum < 0 and rsi < 0):
                    momentum_confirmation = momentum * 2
                elif (momentum > 0 and macd > 0) or (momentum < 0 and macd < 0):
                    momentum_confirmation = momentum * 1.5
            
            score_components.append(momentum_confirmation * 0.20)
            weight_explanations.append(f"Momentum: {momentum_confirmation:.2f}")
            
            # 3. Análise de níveis críticos (25% - aumentado)
            distance_to_support = chart_patterns['distance_to_support']
            distance_to_resistance = chart_patterns['distance_to_resistance']
            support_strength = chart_patterns['support_strength']
            resistance_strength = chart_patterns['resistance_strength']
            
            level_score = 0
            if distance_to_support < 0.3 and support_strength > 0.4:  # Mais conservador
                level_score = 0.8  # Forte sinal de compra perto de suporte
                if distance_to_support < 0.15 and support_strength > 0.6:
                    level_score = 1.2  # Sinal muito forte
            elif distance_to_resistance < 0.3 and resistance_strength > 0.4:
                level_score = -0.8  # Forte sinal de venda perto de resistência
                if distance_to_resistance < 0.15 and resistance_strength > 0.6:
                    level_score = -1.2  # Sinal muito forte
            elif distance_to_support < distance_to_resistance:
                level_score = 0.3  # Viés de compra
            else:
                level_score = -0.3  # Viés de venda
                
            score_components.append(level_score * 0.25)
            weight_explanations.append(f"Levels: {level_score:.2f}")
            
            # 4. Estrutura de mercado (20%)
            market_trend = market_structure['market_trend']
            movement_strength = market_structure['movement_strength']
            structure_score = market_trend * min(1.0, movement_strength)
            score_components.append(structure_score * 0.20)
            weight_explanations.append(f"Structure: {structure_score:.2f}")
            
            # 5. Indicadores de sobrecompra/sobrevenda (10%)
            overbought_oversold = 0
            if rsi > 0.3:  # Sobrecompra
                overbought_oversold = -0.5
            elif rsi < -0.3:  # Sobrevendido
                overbought_oversold = 0.5
            score_components.append(overbought_oversold * 0.10)
            weight_explanations.append(f"RSI Extreme: {overbought_oversold:.2f}")
            
            # Score final
            total_score = sum(score_components)
            
            # Cálculo de confiança mais rigoroso
            confidence_factors = [
                price_action['trend_strength'],
                chart_patterns['levels_quality'],
                market_structure['structure_quality'],
                indicators['trend_quality'],
                price_action['price_stability']
            ]
            
            base_confidence = np.mean([cf for cf in confidence_factors if cf is not None])
            
            # Ajusta confiança base baseado no score
            if abs(total_score) > 0.3:
                confidence_boost = min(0.3, abs(total_score) * 0.5)
            else:
                confidence_boost = 0
            
            base_confidence = min(0.8, base_confidence + confidence_boost)
            
            # DECISÃO MAIS ASSERTIVA - Limiares aumentados
            if total_score > 0.25:  # Aumentado de 0.15
                direction = "buy"
                confidence = 0.65 + (base_confidence * 0.30)  # Confiança base maior
                reasoning = "📈 ALTA CONFIRMADA - Tendência forte com múltiplas confirmações"
                
            elif total_score < -0.25:  # Aumentado de -0.15
                direction = "sell" 
                confidence = 0.65 + (base_confidence * 0.30)
                reasoning = "📉 BAIXA CONFIRMADA - Tendência forte com múltiplas confirmações"
                
            elif total_score > 0.12:  # Aumentado de 0.05
                direction = "buy"
                confidence = 0.58 + (base_confidence * 0.25)
                reasoning = "↗️ VIES DE ALTA - Sinais técnicos favoráveis"
                
            elif total_score < -0.12:  # Aumentado de -0.05
                direction = "sell"
                confidence = 0.58 + (base_confidence * 0.25)
                reasoning = "↘️ VIES DE BAIXA - Sinais técnicos favoráveis"
                
            else:
                # Mercado em equilíbrio - análise mais conservadora
                if indicators['rsi'] > 0.1 and market_trend > 0:
                    direction = "buy"
                    confidence = 0.55
                    reasoning = "⚡ LEVE ALTA - Mercado equilibrado com viés positivo"
                elif indicators['rsi'] < -0.1 and market_trend < 0:
                    direction = "sell"
                    confidence = 0.55
                    reasoning = "⚡ LEVE BAIXA - Mercado equilibrado com viés negativo"
                else:
                    direction = "hold"
                    confidence = 0.52
                    reasoning = "⏸️ AGUARDAR - Mercado sem direção clara"

            # Garante confiança mínima
            confidence = max(self.min_confidence_threshold, confidence)
            
            return {
                "direction": direction,
                "confidence": min(0.92, confidence),  # Limite máximo conservador
                "reasoning": reasoning,
                "total_score": total_score,
                "score_breakdown": " | ".join(weight_explanations)
            }
            
        except Exception as e:
            return {
                "direction": "hold",
                "confidence": 0.51,
                "reasoning": "🔄 AGUARDANDO - Análise conservativa em andamento",
                "total_score": 0.0,
                "score_breakdown": "Erro na análise"
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
        """Análise principal - VERSÃO MAIS ASSERTIVA"""
        
        # Verifica cache
        cached_analysis = self.cache.get(blob, timeframe)
        if cached_analysis:
            cached_analysis['cached'] = True
            return cached_analysis
        
        try:
            # Processamento da imagem
            image = self._load_image(blob)
            
            # VALIDAÇÃO MAIS RIGOROSA
            self._validate_chart_image(image)
            
            img_array = self._preprocess_image(image, timeframe)
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
            
            # Qualidade do sinal
            signal_quality = self._calculate_signal_quality(analysis_data)
            
            # Tomada de decisão inteligente
            decision = self._make_intelligent_decision(analysis_data, timeframe)
            
            # Informações de tempo
            time_info = self._get_entry_timeframe(timeframe)
            
            # Determina grau de qualidade com critérios mais rigorosos
            if signal_quality > 0.75:
                analysis_grade = "high"
            elif signal_quality > 0.55:
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
                    "levels_quality": chart_patterns["levels_quality"],
                    "structure_quality": market_structure["structure_quality"],
                    "trend_quality": indicators["trend_quality"]
                },
                "score_breakdown": decision["score_breakdown"]
            }
            
            # Cache do resultado
            self.cache.set(blob, timeframe, result)
            
            return result
            
        except Exception as e:
            return {
                "direction": "hold",
                "final_confidence": 0.51,
                "entry_signal": f"⚠️ ANÁLISE EM AJUSTE - {str(e)}",
                "entry_time": "Aguardando",
                "timeframe": "Indefinido",
                "analysis_time": datetime.datetime.now().strftime("%H:%M:%S"),
                "user_timeframe": timeframe,
                "cached": False,
                "signal_quality": 0.1,
                "analysis_grade": "low",
                "metrics": {
                    "analysis_score": 0.0,
                    "trend_strength": 0.0,
                    "momentum": 0.0,
                    "levels_quality": 0.0,
                    "structure_quality": 0.0,
                    "trend_quality": 0.0
                },
                "score_breakdown": "Erro na análise"
            }

# =========================
#  SISTEMA WEB FLASK
# =========================
app = Flask(__name__)
analyzer = IntelligentAnalyzer()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>IA Signal Pro - Análise Inteligente</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            color: #e0e0ff;
            min-height: 100vh;
        }
        .container { 
            background: rgba(30, 30, 60, 0.95); 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid #404080;
            backdrop-filter: blur(10px);
        }
        h1 { 
            color: #7ce0ff; 
            text-align: center; 
            margin-bottom: 10px;
            font-size: 2.2em;
            text-shadow: 0 0 20px rgba(124, 224, 255, 0.5);
        }
        .subtitle {
            text-align: center; 
            color: #a0a0ff; 
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .upload-area {
            border: 2px dashed #404080; 
            padding: 40px; 
            text-align: center; 
            border-radius: 10px; 
            margin: 20px 0; 
            background: rgba(40, 40, 80, 0.3);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #7ce0ff;
            background: rgba(40, 40, 80, 0.5);
        }
        input[type="file"], select, button {
            padding: 12px 20px; 
            margin: 10px 5px; 
            border: none; 
            border-radius: 8px; 
            font-size: 16px;
            transition: all 0.3s ease;
        }
        input[type="file"] {
            background: rgba(60, 60, 100, 0.8);
            color: #e0e0ff;
            width: 100%;
            box-sizing: border-box;
        }
        select {
            background: rgba(60, 60, 100, 0.8);
            color: #e0e0ff;
            cursor: pointer;
        }
        button {
            background: linear-gradient(135deg, #7ce0ff 0%, #4a90e2 100%);
            color: #0f0f23;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(124, 224, 255, 0.3);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(124, 224, 255, 0.4);
        }
        .result { 
            margin-top: 30px; 
            padding: 25px; 
            border-radius: 12px; 
            background: rgba(40, 40, 80, 0.4);
            border-left: 5px solid #7ce0ff;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .signal-buy { border-left-color: #00ff88 !important; background: rgba(0, 255, 136, 0.1) !important; }
        .signal-sell { border-left-color: #ff4444 !important; background: rgba(255, 68, 68, 0.1) !important; }
        .signal-hold { border-left-color: #ffaa00 !important; background: rgba(255, 170, 0, 0.1) !important; }
        .confidence-high { color: #00ff88; font-weight: bold; }
        .confidence-medium { color: #ffaa00; font-weight: bold; }
        .confidence-low { color: #ff4444; font-weight: bold; }
        .metric-bar {
            height: 8px; 
            background: rgba(60, 60, 100, 0.8); 
            border-radius: 4px; 
            margin: 5px 0 15px 0;
            overflow: hidden;
        }
        .metric-fill {
            height: 100%;
            background: linear-gradient(90deg, #7ce0ff, #4a90e2);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .metric-card {
            background: rgba(50, 50, 90, 0.6);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #404080;
        }
        .loading {
            text-align: center;
            color: #7ce0ff;
            font-size: 1.1em;
        }
        .pulse {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .quality-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin-left: 10px;
        }
        .quality-high { background: #00ff88; color: #003322; }
        .quality-medium { background: #ffaa00; color: #332200; }
        .quality-low { background: #ff4444; color: #330000; }
        .time-info {
            background: rgba(30, 30, 60, 0.8);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #404080;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 IA SIGNAL PRO</h1>
        <div class="subtitle">Sistema Inteligente de Análise Técnica - VERSÃO MAIS ASSERTIVA</div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>📤 CLIQUE PARA ENVIAR O GRÁFICO</h3>
                <p>Arraste ou clique para selecionar a imagem do gráfico</p>
                <input type="file" id="fileInput" name="file" accept="image/*" required style="display: none;" onchange="previewImage(this)">
            </div>
            
            <div style="text-align: center; margin: 20px 0;">
                <select name="timeframe" required>
                    <option value="1m">⏱️ Timeframe 1 Minuto</option>
                    <option value="5m" selected>⏱️ Timeframe 5 Minutos</option>
                </select>
                
                <button type="submit">🎯 ANALISAR GRÁFICO INTELIGENTE</button>
            </div>
        </form>
        
        <div id="imagePreview" style="text-align: center; margin: 20px 0; display: none;">
            <img id="preview" style="max-width: 100%; max-height: 300px; border-radius: 8px; border: 2px solid #404080;">
        </div>
        
        <div id="result"></div>
        <div id="loading" class="loading" style="display: none;">
            <div class="pulse">🔍 ANALISANDO PADRÕES E SINAIS...</div>
        </div>
    </div>

    <script>
        function previewImage(input) {
            const preview = document.getElementById('preview');
            const previewContainer = document.getElementById('imagePreview');
            
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    previewContainer.style.display = 'block';
                }
                
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                let signalClass = 'signal-hold';
                if (data.direction === 'buy') signalClass = 'signal-buy';
                else if (data.direction === 'sell') signalClass = 'signal-sell';
                
                let confidenceClass = 'confidence-low';
                if (data.final_confidence > 0.7) confidenceClass = 'confidence-high';
                else if (data.final_confidence > 0.6) confidenceClass = 'confidence-medium';
                
                let qualityBadge = '';
                if (data.analysis_grade === 'high') {
                    qualityBadge = '<span class="quality-badge quality-high">ALTA QUALIDADE</span>';
                } else if (data.analysis_grade === 'medium') {
                    qualityBadge = '<span class="quality-badge quality-medium">QUALIDADE MÉDIA</span>';
                } else {
                    qualityBadge = '<span class="quality-badge quality-low">QUALIDADE BAIXA</span>';
                }
                
                resultDiv.innerHTML = `
                    <div class="result ${signalClass}">
                        <h2 style="margin-top: 0;">${data.entry_signal}</h2>
                        
                        <div class="time-info">
                            <strong>⏰ Horário de Entrada:</strong> ${data.entry_time}<br>
                            <strong>📊 Timeframe:</strong> ${data.timeframe}<br>
                            <strong>🕒 Análise Gerada:</strong> ${data.analysis_time}
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>Confiança do Sinal:</strong> 
                                <span class="${confidenceClass}">${(data.final_confidence * 100).toFixed(1)}%</span>
                            </div>
                            ${qualityBadge}
                        </div>
                        
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${data.final_confidence * 100}%"></div>
                        </div>
                        
                        <div><strong>🔍 Qualidade da Análise:</strong> ${(data.signal_quality * 100).toFixed(1)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${data.signal_quality * 100}%"></div>
                        </div>
                        
                        <div><strong>📈 Score Técnico:</strong> ${data.metrics.analysis_score.toFixed(3)}</div>
                        <div><strong>🧠 Composição do Score:</strong> ${data.score_breakdown}</div>
                        
                        ${data.cached ? '<div style="color: #ffaa00; margin-top: 10px;">⚡ Resultado em cache (análise recente)</div>' : ''}
                        
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <strong>📊 Força da Tendência</strong>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: ${data.metrics.trend_strength * 100}%"></div>
                                </div>
                                ${(data.metrics.trend_strength * 100).toFixed(1)}%
                            </div>
                            
                            <div class="metric-card">
                                <strong>⚡ Momentum</strong>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: ${(Math.abs(data.metrics.momentum) * 50 + 50)}%"></div>
                                </div>
                                ${data.metrics.momentum.toFixed(3)}
                            </div>
                            
                            <div class="metric-card">
                                <strong>🎯 Qualidade dos Níveis</strong>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: ${data.metrics.levels_quality * 100}%"></div>
                                </div>
                                ${(data.metrics.levels_quality * 100).toFixed(1)}%
                            </div>
                            
                            <div class="metric-card">
                                <strong>🏛️ Estrutura do Mercado</strong>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: ${data.metrics.structure_quality * 100}%"></div>
                                </div>
                                ${(data.metrics.structure_quality * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                `;
                
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="result signal-hold">
                        <h3 style="color: #ff4444;">❌ Erro na Análise</h3>
                        <p>Erro: ${error.message}</p>
                    </div>
                `;
            } finally {
                loadingDiv.style.display = 'none';
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        timeframe = request.form.get('timeframe', '5m')
        
        # Analisa a imagem
        blob = file.read()
        result = analyzer.analyze(blob, timeframe)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Erro no processamento: {str(e)}',
            'direction': 'hold',
            'final_confidence': 0.51,
            'entry_signal': f'⚠️ ERRO: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
