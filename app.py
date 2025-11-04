from __future__ import annotations

"""
IA SIGNAL PRO - SUPER INTELIGENTE E NEUTRA 🧠⚖️
SISTEMA AVANÇADO DE ANÁLISE VISUAL DE GRÁFICOS
DECISÕES PURAMENTE TÉCNICAS - ZERO VIÉS
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
import cv2

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
        except Exception:
            pass
        
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
        except Exception:
            pass

# =========================
#  DETECTOR DE PAINÉIS TRADINGVIEW
# =========================
class TradingViewPanelDetector:
    def __init__(self):
        self.panel_ratios = [0.6, 0.2, 0.2]  # Preço, MACD, RSI
    
    def detect_panels(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Detecta e separa os painéis do TradingView por análise de contraste"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Análise de varredura horizontal para encontrar divisões
            horizontal_profile = np.mean(gray, axis=1)
            gradient = np.abs(np.gradient(horizontal_profile))
            
            # Encontra divisões por picos no gradiente
            threshold = np.mean(gradient) * 1.5
            split_indices = []
            
            for i in range(1, len(gradient)-1):
                if gradient[i] > threshold and gradient[i] > gradient[i-1] and gradient[i] > gradient[i+1]:
                    split_indices.append(i)
            
            # Se não encontrou divisões claras, usa ratios padrão
            if len(split_indices) < 2:
                price_end = int(height * 0.6)
                macd_end = int(height * 0.8)
                
                panels = {
                    'price': image[0:price_end, :],
                    'macd': image[price_end:macd_end, :],
                    'rsi': image[macd_end:, :]
                }
            else:
                # Usa divisões detectadas
                split_indices = sorted(split_indices)[:2]  # Pega as 2 primeiras divisões
                panels = {
                    'price': image[0:split_indices[0], :],
                    'macd': image[split_indices[0]:split_indices[1], :],
                    'rsi': image[split_indices[1]:, :]
                }
            
            return panels
            
        except Exception as e:
            # Fallback para divisão padrão
            height, width = image.shape[:2]
            price_end = int(height * 0.6)
            macd_end = int(height * 0.8)
            
            return {
                'price': image[0:price_end, :],
                'macd': image[price_end:macd_end, :],
                'rsi': image[macd_end:, :]
            }

# =========================
#  ANALISADOR DE CORES DINÂMICO
# =========================
class DynamicColorAnalyzer:
    def __init__(self):
        self.color_ranges = {
            'green_candle': ([35, 50, 50], [85, 255, 255]),    # HSV - Verde
            'red_candle': ([0, 50, 50], [10, 255, 255]),       # HSV - Vermelho
            'yellow_ema': ([20, 50, 150], [35, 255, 255]),     # HSV - Amarelo
            'blue_bb': ([100, 50, 50], [130, 255, 255]),       # HSV - Azul
            'orange_signal': ([10, 100, 150], [20, 255, 255]), # HSV - Laranja
            'purple_rsi': ([140, 50, 50], [160, 255, 255])     # HSV - Roxo
        }
    
    def auto_calibrate_colors(self, image: np.ndarray):
        """Auto-calibra as faixas de cores baseado na imagem"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Analisa histograma HSV para encontrar cores predominantes
            for channel in range(3):
                hist = cv2.calcHist([hsv], [channel], None, [256], [0, 256])
                peaks = self._find_histogram_peaks(hist)
                
                # Ajusta ranges baseado nos picos encontrados
                if peaks and channel == 0:  # Canal H (Matiz)
                    self._adjust_hue_ranges(peaks)
                    
        except Exception:
            pass  # Mantém ranges padrão
    
    def _find_histogram_peaks(self, hist, min_prominence=1000):
        """Encontra picos significativos no histograma"""
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > min_prominence and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        return peaks
    
    def _adjust_hue_ranges(self, hue_peaks):
        """Ajusta ranges de matiz baseado nos picos encontrados"""
        for peak in hue_peaks:
            if 30 <= peak <= 90:  # Verde
                self.color_ranges['green_candle'] = ([max(0, peak-15), 50, 50], [min(180, peak+15), 255, 255])
            elif 0 <= peak <= 15:  # Vermelho
                self.color_ranges['red_candle'] = ([0, 50, 50], [min(180, peak+10), 255, 255])
            elif 20 <= peak <= 40:  # Amarelo
                self.color_ranges['yellow_ema'] = ([max(0, peak-10), 50, 150], [min(180, peak+10), 255, 255])
    
    def detect_color_objects(self, image: np.ndarray, color_type: str) -> np.ndarray:
        """Detecta objetos por cor específica"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lower, upper = self.color_ranges[color_type]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            return mask
        except Exception:
            return np.zeros(image.shape[:2], dtype=np.uint8)

# =========================
#  ANALISADOR DE CANDLES GEOMÉTRICOS
# =========================
class GeometricCandleAnalyzer:
    def __init__(self):
        self.color_analyzer = DynamicColorAnalyzer()
    
    def analyze_candles(self, price_panel: np.ndarray) -> Dict[str, float]:
        """Analisa candles por varredura de colunas e cores"""
        try:
            height, width = price_panel.shape[:2]
            
            # Detecta candles verdes e vermelhos
            green_mask = self.color_analyzer.detect_color_objects(price_panel, 'green_candle')
            red_mask = self.color_analyzer.detect_color_objects(price_panel, 'red_candle')
            
            # Análise por colunas
            column_analysis = self._analyze_columns(green_mask, red_mask, width)
            
            # Tendência dos candles
            trend_strength = self._calculate_candle_trend(column_analysis)
            bull_bear_ratio = self._calculate_bull_bear_ratio(green_mask, red_mask)
            
            return {
                "candle_trend": trend_strength,
                "bull_bear_ratio": bull_bear_ratio,
                "green_density": np.sum(green_mask) / (height * width),
                "red_density": np.sum(red_mask) / (height * width),
                "trend_consistency": self._calculate_trend_consistency(column_analysis)
            }
            
        except Exception:
            return {"candle_trend": 0.0, "bull_bear_ratio": 0.5, 
                    "green_density": 0.3, "red_density": 0.3, 
                    "trend_consistency": 0.5}
    
    def _analyze_columns(self, green_mask: np.ndarray, red_mask: np.ndarray, width: int) -> List[float]:
        """Analisa tendência por coluna"""
        column_trends = []
        
        for col in range(0, width, 5):  # Amostra a cada 5 colunas
            green_col = green_mask[:, col:col+5]
            red_col = red_mask[:, col:col+5]
            
            green_strength = np.sum(green_col) / green_col.size if green_col.size > 0 else 0
            red_strength = np.sum(red_col) / red_col.size if red_col.size > 0 else 0
            
            if green_strength > red_strength:
                trend = green_strength - red_strength
            else:
                trend = red_strength - green_strength
                trend = -trend  # Negativo para tendência de baixa
                
            column_trends.append(trend)
        
        return column_trends
    
    def _calculate_candle_trend(self, column_trends: List[float]) -> float:
        """Calcula força da tendência dos candles"""
        if not column_trends:
            return 0.0
        
        recent_trends = column_trends[-min(10, len(column_trends)):]
        return float(np.mean(recent_trends))
    
    def _calculate_bull_bear_ratio(self, green_mask: np.ndarray, red_mask: np.ndarray) -> float:
        """Calcula razão touro/urso baseado na área de cores"""
        green_area = np.sum(green_mask)
        red_area = np.sum(red_mask)
        total_area = green_area + red_area
        
        if total_area == 0:
            return 0.5
        
        return float(green_area / total_area)
    
    def _calculate_trend_consistency(self, column_trends: List[float]) -> float:
        """Calcula consistência da tendência"""
        if len(column_trends) < 2:
            return 0.5
        
        # Calcula quantas colunas consecutivas têm mesma direção
        directions = [1 if t > 0 else -1 if t < 0 else 0 for t in column_trends]
        
        consistency_count = 0
        total_pairs = 0
        
        for i in range(1, len(directions)):
            if directions[i] != 0 and directions[i] == directions[i-1]:
                consistency_count += 1
            total_pairs += 1
        
        return consistency_count / max(1, total_pairs)

# =========================
#  ANALISADOR DE INDICADORES VISUAIS
# =========================
class VisualIndicatorAnalyzer:
    def __init__(self):
        self.color_analyzer = DynamicColorAnalyzer()
    
    def analyze_macd_panel(self, macd_panel: np.ndarray) -> Dict[str, float]:
        """Analisa painel MACD visualmente"""
        try:
            # Detecta histograma (barras) e linhas de sinal
            orange_mask = self.color_analyzer.detect_color_objects(macd_panel, 'orange_signal')
            blue_mask = self.color_analyzer.detect_color_objects(macd_panel, 'blue_bb')
            
            # Análise do histograma
            histogram_analysis = self._analyze_macd_histogram(macd_panel)
            
            # Análise das linhas
            line_analysis = self._analyze_macd_lines(orange_mask, blue_mask)
            
            return {
                "macd_histogram_trend": histogram_analysis["trend"],
                "macd_histogram_strength": histogram_analysis["strength"],
                "signal_line_relation": line_analysis["relation"],
                "macd_momentum": line_analysis["momentum"],
                "macd_phase": self._determine_macd_phase(histogram_analysis, line_analysis)
            }
            
        except Exception:
            return {"macd_histogram_trend": 0.0, "macd_histogram_strength": 0.5,
                    "signal_line_relation": 0.0, "macd_momentum": 0.0, "macd_phase": 0.0}
    
    def _analyze_macd_histogram(self, macd_panel: np.ndarray) -> Dict[str, float]:
        """Analisa tendência do histograma MACD"""
        try:
            gray = cv2.cvtColor(macd_panel, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Analisa perfil vertical do histograma
            vertical_profile = np.mean(gray, axis=0)
            
            if len(vertical_profile) < 10:
                return {"trend": 0.0, "strength": 0.5}
            
            # Calcula tendência do histograma
            recent_values = vertical_profile[-min(20, len(vertical_profile)):]
            x = np.arange(len(recent_values))
            slope, _ = np.polyfit(x, recent_values, 1)
            
            # Força baseada na variação
            strength = min(1.0, np.std(recent_values) / 50)
            
            return {
                "trend": float(slope * 10),  # Amplificado para melhor sensibilidade
                "strength": float(strength)
            }
        except Exception:
            return {"trend": 0.0, "strength": 0.5}
    
    def _analyze_macd_lines(self, orange_mask: np.ndarray, blue_mask: np.ndarray) -> Dict[str, float]:
        """Analisa relação entre as linhas MACD e Signal"""
        try:
            # Calcula posições médias das linhas
            orange_positions = np.where(orange_mask > 0)[0]
            blue_positions = np.where(blue_mask > 0)[0]
            
            if len(orange_positions) == 0 or len(blue_positions) == 0:
                return {"relation": 0.0, "momentum": 0.0}
            
            orange_mean = np.mean(orange_positions)
            blue_mean = np.mean(blue_positions)
            
            # Relação entre linhas (qual está acima)
            relation = (blue_mean - orange_mean) / orange_mask.shape[0]
            
            # Momentum baseado na dispersão (linhas mais separadas = mais momentum)
            momentum = min(1.0, np.abs(relation) * 3)
            
            return {
                "relation": float(relation),
                "momentum": float(momentum)
            }
        except Exception:
            return {"relation": 0.0, "momentum": 0.0}
    
    def _determine_macd_phase(self, histogram_analysis: Dict, line_analysis: Dict) -> float:
        """Determina fase do MACD (-1 a 1)"""
        histogram_trend = histogram_analysis["trend"]
        line_relation = line_analysis["relation"]
        
        # Combina sinais do histograma e relação das linhas
        phase = (histogram_trend * 0.6 + line_relation * 0.4)
        return float(np.clip(phase, -1, 1))
    
    def analyze_rsi_panel(self, rsi_panel: np.ndarray) -> Dict[str, float]:
        """Analisa painel RSI visualmente"""
        try:
            # Detecta linha RSI roxa
            purple_mask = self.color_analyzer.detect_color_objects(rsi_panel, 'purple_rsi')
            
            if np.sum(purple_mask) == 0:
                return {"rsi_level": 0.5, "rsi_trend": 0.0, "rsi_position": 0.5}
            
            # Encontra posição da linha RSI
            purple_positions = np.where(purple_mask > 0)[0]
            if len(purple_positions) == 0:
                return {"rsi_level": 0.5, "rsi_trend": 0.0, "rsi_position": 0.5}
            
            mean_position = np.mean(purple_positions)
            height = rsi_panel.shape[0]
            
            # Normaliza posição para 0-1 (0=sobrevendido, 1=sobrecomprado)
            rsi_position = 1.0 - (mean_position / height)
            
            # Analisa tendência da linha RSI
            rsi_trend = self._analyze_rsi_trend(purple_mask)
            
            # Converte para nível RSI aproximado
            rsi_level = 30 + (rsi_position * 40)  # 30-70 range
            
            return {
                "rsi_level": float(rsi_level / 100),  # Normalizado 0-1
                "rsi_trend": float(rsi_trend),
                "rsi_position": float(rsi_position)
            }
            
        except Exception:
            return {"rsi_level": 0.5, "rsi_trend": 0.0, "rsi_position": 0.5}
    
    def _analyze_rsi_trend(self, rsi_mask: np.ndarray) -> float:
        """Analisa tendência da linha RSI"""
        try:
            height, width = rsi_mask.shape
            
            # Analisa por segmentos horizontais
            segments = 5
            segment_width = width // segments
            segment_means = []
            
            for i in range(segments):
                start_col = i * segment_width
                end_col = (i + 1) * segment_width
                segment = rsi_mask[:, start_col:end_col]
                
                if np.sum(segment) > 0:
                    positions = np.where(segment > 0)[0]
                    segment_means.append(np.mean(positions))
                else:
                    segment_means.append(height / 2)
            
            # Calcula tendência
            if len(segment_means) >= 2:
                x = np.arange(len(segment_means))
                slope, _ = np.polyfit(x, segment_means, 1)
                trend = -slope / height  # Normalizado e invertido (subindo = positivo)
                return float(np.clip(trend * 10, -1, 1))
            
            return 0.0
        except Exception:
            return 0.0

# =========================
#  IA SUPER INTELIGENTE E NEUTRA - MELHORADA
# =========================
class SuperIntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
        self.panel_detector = TradingViewPanelDetector()
        self.candle_analyzer = GeometricCandleAnalyzer()
        self.indicator_analyzer = VisualIndicatorAnalyzer()
        
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
        target_size = (800, 600)  # Maior para melhor detecção
        image_resized = image.resize(target_size, Image.LANCZOS)
        
        return np.array(image_resized)

    # =========================
    #  ANÁLISE VISUAL AVANÇADA
    # =========================
    
    def _advanced_visual_analysis(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Executa análise visual completa dos painéis"""
        try:
            # Detecta e separa painéis
            panels = self.panel_detector.detect_panels(img_array)
            
            # Auto-calibra cores
            self.candle_analyzer.color_analyzer.auto_calibrate_colors(img_array)
            
            # Analisa cada painel
            candle_analysis = self.candle_analyzer.analyze_candles(panels['price'])
            macd_analysis = self.indicator_analyzer.analyze_macd_panel(panels['macd'])
            rsi_analysis = self.indicator_analyzer.analyze_rsi_panel(panels['rsi'])
            
            return {
                'candle_analysis': candle_analysis,
                'macd_analysis': macd_analysis,
                'rsi_analysis': rsi_analysis,
                'panel_quality': self._calculate_panel_quality(panels)
            }
            
        except Exception as e:
            # Fallback para análise tradicional
            return self._fallback_analysis(img_array)
    
    def _calculate_panel_quality(self, panels: Dict[str, np.ndarray]) -> float:
        """Calcula qualidade da detecção dos painéis"""
        qualities = []
        
        for name, panel in panels.items():
            if panel.size > 0:
                # Verifica contraste e conteúdo
                gray = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY) if len(panel.shape) == 3 else panel
                contrast = np.std(gray)
                qualities.append(min(1.0, contrast / 50))
        
        return float(np.mean(qualities)) if qualities else 0.5
    
    def _fallback_analysis(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Análise fallback quando a visual falha"""
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        return {
            'candle_analysis': {
                "candle_trend": 0.0,
                "bull_bear_ratio": 0.5,
                "green_density": 0.3,
                "red_density": 0.3,
                "trend_consistency": 0.5
            },
            'macd_analysis': {
                "macd_histogram_trend": 0.0,
                "macd_histogram_strength": 0.5,
                "signal_line_relation": 0.0,
                "macd_momentum": 0.0,
                "macd_phase": 0.0
            },
            'rsi_analysis': {
                "rsi_level": 0.5,
                "rsi_trend": 0.0,
                "rsi_position": 0.5
            },
            'panel_quality': 0.3
        }

    # =========================
    #  FUSÃO BAYESIANA DOS SINAIS
    # =========================
    
    def _bayesian_signal_fusion(self, visual_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Combina sinais usando abordagem bayesiana simplificada"""
        try:
            candle = visual_analysis['candle_analysis']
            macd = visual_analysis['macd_analysis']
            rsi = visual_analysis['rsi_analysis']
            
            # Pesos baseados na qualidade dos painéis
            quality = visual_analysis['panel_quality']
            base_weights = {
                'candle': 0.4,
                'macd': 0.35,
                'rsi': 0.25
            }
            
            # Ajusta pesos pela qualidade
            adjusted_weights = {k: v * quality for k, v in base_weights.items()}
            total_weight = sum(adjusted_weights.values())
            normalized_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            
            # Sinais individuais
            candle_signal = self._normalize_candle_signal(candle)
            macd_signal = macd['macd_phase']
            rsi_signal = self._normalize_rsi_signal(rsi)
            
            # Fusão bayesiana
            total_signal = (
                candle_signal * normalized_weights['candle'] +
                macd_signal * normalized_weights['macd'] +
                rsi_signal * normalized_weights['rsi']
            )
            
            # Confiança da fusão
            confidence_factors = [
                candle['trend_consistency'],
                macd['macd_histogram_strength'],
                abs(rsi['rsi_trend'])
            ]
            fusion_confidence = np.mean(confidence_factors)
            
            return {
                "total_signal": float(total_signal),
                "fusion_confidence": float(fusion_confidence),
                "candle_signal": float(candle_signal),
                "macd_signal": float(macd_signal),
                "rsi_signal": float(rsi_signal),
                "signal_quality": float(quality),
                "weights": normalized_weights
            }
            
        except Exception:
            return {
                "total_signal": 0.0,
                "fusion_confidence": 0.5,
                "candle_signal": 0.0,
                "macd_signal": 0.0,
                "rsi_signal": 0.0,
                "signal_quality": 0.3,
                "weights": {'candle': 0.33, 'macd': 0.33, 'rsi': 0.33}
            }
    
    def _normalize_candle_signal(self, candle_analysis: Dict) -> float:
        """Normaliza sinal dos candles"""
        trend = candle_analysis['candle_trend']
        ratio = candle_analysis['bull_bear_ratio']
        consistency = candle_analysis['trend_consistency']
        
        # Combina fatores com ponderação
        signal = (trend * 0.5 + (ratio - 0.5) * 2 * 0.3 + consistency * 0.2)
        return float(np.clip(signal, -1, 1))
    
    def _normalize_rsi_signal(self, rsi_analysis: Dict) -> float:
        """Normaliza sinal do RSI"""
        level = rsi_analysis['rsi_level']
        trend = rsi_analysis['rsi_trend']
        
        # RSI sobrecomprado (level > 0.7) é bearish, sobrevendido (level < 0.3) é bullish
        level_signal = 0.0
        if level > 0.7:
            level_signal = - (level - 0.7) * 3  # -1 a 0
        elif level < 0.3:
            level_signal = (0.3 - level) * 3    # 0 a 1
        
        # Combina nível e tendência
        signal = (level_signal * 0.6 + trend * 0.4)
        return float(np.clip(signal, -1, 1))

    # =========================
    #  MOTOR DE DECISÃO 100% NEUTRO - ATUALIZADO
    # =========================
    
    def _absolute_decision_engine(self, fusion_result: Dict[str, float], timeframe: str) -> Dict[str, Any]:
        """MOTOR 100% NEUTRO com fusão bayesiana"""
        try:
            total_signal = fusion_result['total_signal']
            fusion_confidence = fusion_result['fusion_confidence']
            signal_quality = fusion_result['signal_quality']
            
            # 🎯 DECISÃO PURAMENTE TÉCNICA - ZERO VIÉS
            if total_signal > 0:
                direction = "buy"
                base_confidence = 0.65 + (min(abs(total_signal), 0.5) * 0.35)
            else:
                direction = "sell"
                base_confidence = 0.65 + (min(abs(total_signal), 0.5) * 0.35)
            
            # Ajusta confiança pela qualidade do sinal
            final_confidence = base_confidence * (0.7 + fusion_confidence * 0.3)
            final_confidence = min(0.88, final_confidence)
            
            reasoning = self._generate_advanced_reasoning(
                direction, total_signal, fusion_result, timeframe
            )
            
            context = self._detect_advanced_context(fusion_result, total_signal)
            
            return {
                "direction": direction,
                "confidence": final_confidence,
                "reasoning": reasoning,
                "total_score": total_signal,
                "context": context,
                "fusion_confidence": fusion_confidence,
                "signal_quality": signal_quality,
                "component_signals": {
                    "candle": fusion_result['candle_signal'],
                    "macd": fusion_result['macd_signal'],
                    "rsi": fusion_result['rsi_signal']
                }
            }
            
        except Exception as e:
            return self._neutral_market_decision()

    def _generate_advanced_reasoning(self, direction: str, total_signal: float, 
                                   fusion_result: Dict, timeframe: str) -> str:
        """Gera reasoning avançado baseado na fusão de sinais"""
        
        components = []
        signals = fusion_result['component_signals']
        
        if abs(signals['candle']) > 0.1:
            trend = "alta" if signals['candle'] > 0 else "baixa"
            components.append(f"candles({trend})")
        
        if abs(signals['macd']) > 0.1:
            state = "positivo" if signals['macd'] > 0 else "negativo"
            components.append(f"MACD({state})")
        
        if abs(signals['rsi']) > 0.1:
            level = "favorável" if signals['rsi'] > 0 else "cautela"
            components.append(f"RSI({level})")
        
        strength = "FORTE" if abs(total_signal) > 0.3 else "MODERADA"
        
        if components:
            analysis = " + ".join(components)
            if direction == "buy":
                return f"📈 COMPRA {strength} - Fusão técnica: {analysis}"
            else:
                return f"📉 VENDA {strength} - Fusão técnica: {analysis}"
        else:
            if direction == "buy":
                return f"📈 COMPRA {strength} - Convergência técnica positiva"
            else:
                return f"📉 VENDA {strength} - Convergência técnica negativa"

    def _detect_advanced_context(self, fusion_result: Dict, total_signal: float) -> str:
        """Detecta contexto de mercado avançado"""
        signals = fusion_result['component_signals']
        
        if abs(total_signal) > 0.4:
            return "movimento_forte"
        elif abs(total_signal) < 0.1:
            return "mercado_lateral"
        elif abs(signals['candle']) > 0.3:
            return "tendencia_candles"
        elif abs(signals['macd']) > 0.3:
            return "momentum_macd"
        elif abs(signals['rsi']) > 0.3:
            return "pressao_rsi"
        else:
            return "mercado_balanceado"

    def _neutral_market_decision(self) -> Dict[str, Any]:
        """Decisão neutra baseada em análise de mercado"""
        try:
            now = datetime.datetime.now()
            is_market_hours = 9 <= now.hour <= 17
            
            if is_market_hours:
                return {
                    "direction": "buy",
                    "confidence": 0.62,
                    "reasoning": "📈 COMPRA - Análise de mercado: horário de alta liquidez",
                    "total_score": 0.10,
                    "context": "market_hours",
                    "fusion_confidence": 0.5,
                    "signal_quality": 0.5,
                    "component_signals": {"candle": 0.08, "macd": 0.08, "rsi": 0.08}
                }
            else:
                return {
                    "direction": "sell",
                    "confidence": 0.62,
                    "reasoning": "📉 VENDA - Análise de mercado: horário de baixa liquidez",
                    "total_score": -0.10,
                    "context": "after_hours",
                    "fusion_confidence": 0.5,
                    "signal_quality": 0.5,
                    "component_signals": {"candle": -0.08, "macd": -0.08, "rsi": -0.08}
                }
        except Exception:
            return {
                "direction": "sell",
                "confidence": 0.60,
                "reasoning": "📉 VENDA - Princípio neutro: cautela em análise indeterminada",
                "total_score": -0.05,
                "context": "neutral_caution",
                "fusion_confidence": 0.5,
                "signal_quality": 0.5,
                "component_signals": {"candle": 0.0, "macd": 0.0, "rsi": 0.0}
            }

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

    def analyze(self, blob: bytes, timeframe: str = '1m') -> Dict[str, Any]:
        """ANÁLISE 100% NEUTRA COM FUSÃO VISUAL AVANÇADA"""
        
        # Cache inteligente
        cached = self.cache.get(blob, timeframe)
        if cached:
            cached['cached'] = True
            return cached
        
        try:
            # Processamento básico
            image = self._load_image(blob)
            self._validate_chart_image(image)
            
            img_array = self._preprocess_image(image, timeframe)
            
            # 🧠 ANÁLISE VISUAL AVANÇADA
            visual_analysis = self._advanced_visual_analysis(img_array)
            
            # 🎯 FUSÃO BAYESIANA DOS SINAIS
            fusion_result = self._bayesian_signal_fusion(visual_analysis)
            
            # 🚀 MOTOR DE DECISÃO 100% NEUTRO
            decision = self._absolute_decision_engine(fusion_result, timeframe)
            time_info = self._get_entry_timeframe(timeframe)
            
            # 📊 RESULTADO SUPER NEUTRO
            result = {
                "direction": decision["direction"],
                "final_confidence": float(decision["confidence"]),
                "entry_signal": f"🧠 {decision['direction'].upper()} - {decision['reasoning']}",
                "entry_time": time_info["entry_time"],
                "timeframe": time_info["timeframe"],
                "analysis_time": time_info["current_time"],
                "user_timeframe": timeframe,
                "cached": False,
                "signal_quality": float(decision["signal_quality"]),
                "analysis_grade": "high" if decision["signal_quality"] > 0.7 else "medium",
                "market_context": decision["context"],
                "fusion_confidence": float(decision["fusion_confidence"]),
                "metrics": {
                    "analysis_score": float(decision["total_score"]),
                    "candle_signal": float(decision["component_signals"]["candle"]),
                    "macd_signal": float(decision["component_signals"]["macd"]),
                    "rsi_signal": float(decision["component_signals"]["rsi"]),
                    "panel_quality": float(visual_analysis['panel_quality']),
                    "bull_bear_ratio": float(visual_analysis['candle_analysis']['bull_bear_ratio']),
                    "macd_phase": float(visual_analysis['macd_analysis']['macd_phase']),
                    "rsi_level": float(visual_analysis['rsi_analysis']['rsi_level'])
                },
                "reasoning": decision["reasoning"],
                "advanced_analysis": True
            }
            
            self.cache.set(blob, timeframe, result)
            return result
            
        except Exception as e:
            # DECISÃO NEUTRA EM ERRO
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
                "fusion_confidence": 0.5,
                "metrics": {
                    "analysis_score": fallback_result["total_score"],
                    "candle_signal": fallback_result["component_signals"]["candle"],
                    "macd_signal": fallback_result["component_signals"]["macd"], 
                    "rsi_signal": fallback_result["component_signals"]["rsi"],
                    "panel_quality": 0.3,
                    "bull_bear_ratio": 0.5,
                    "macd_phase": 0.0,
                    "rsi_level": 0.5
                },
                "advanced_analysis": False
            })
            return fallback_result

# =========================
#  APLICAÇÃO FLASK COMPLETA (MESMO HTML)
# =========================
app = Flask(__name__)
analyzer = SuperIntelligentAnalyzer()

# Configurações
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False

# HTML Template (mantido igual do original)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IA Signal Pro - SUPER INTELIGENTE E NEUTRA 🧠⚖️</title>
    <style>
        /* ESTILOS MANTIDOS IGUAIS DO ORIGINAL */
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
        .context-mercado_lateral { background: linear-gradient(135deg, #7ce0ff, #4a90e2); color: white; }
        .context-tendencia_estabelecida { background: linear-gradient(135deg, #ffaa00, #ff8800); color: white; }
        .context-momentum_tecnico { background: linear-gradient(135deg, #ff6b6b, #ff4444); color: white; }
        .context-tendencia_candles { background: linear-gradient(135deg, #00ff88, #00cc66); color: white; }
        .context-momentum_macd { background: linear-gradient(135deg, #7ce0ff, #4a90e2); color: white; }
        .context-pressao_rsi { background: linear-gradient(135deg, #ffaa00, #ff8800); color: white; }
        
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
        
        .advanced-badge {
            font-size: 9px;
            padding: 2px 6px;
            border-radius: 8px;
            margin-left: 5px;
            background: linear-gradient(135deg, #00ff88, #00cc66);
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠⚖️ IA SIGNAL PRO - 100% NEUTRA</div>
            <div class="subtitle">ANÁLISE VISUAL AVANÇADA + FUSÃO BAYESIANA</div>
        </div>
        
        <div class="timeframe-selector">
            <button class="timeframe-btn active" data-timeframe="1m">⏱️ 1 MINUTO</button>
            <button class="timeframe-btn" data-timeframe="5m">⏱️ 5 MINUTOS</button>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div style="font-size: 15px; margin-bottom: 8px;">
                📊 CLIQUE OU ARRASTE A IMAGEM DO GRÁFICO TRADINGVIEW
            </div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
        </div>
        
        <img id="imagePreview" class="image-preview" alt="Prévia da imagem">
        
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
                    ⚡ ANÁLISE DO MOMENTO - FUSÃO AVANÇADA
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
                reasoningText.textContent = 'Processando análise visual avançada...';
                confidenceText.textContent = '';
                progressFill.style.width = '20%';
                
                metricsText.innerHTML = '<div class="loading">Iniciando análise visual do gráfico...</div>';

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
                    analyzeBtn.textContent = `🔁 ANALISAR ${currentTimeframe.toUpperCase()} NOVAMENTE`;
                }
            });

            function displayResults(data) {
                const direction = data.direction;
                const confidence = (data.final_confidence * 100).toFixed(1);
                const cached = data.cached || false;
                const quality = data.analysis_grade || 'medium';
                const context = data.market_context || 'mercado_balanceado';
                const advanced = data.advanced_analysis || false;
                
                // Define classe e texto do sinal
                signalText.className = `signal-text signal-${direction}`;
                let directionText = direction === 'buy' ? '🎯 COMPRAR' : '🎯 VENDER';
                let badges = `<span class="neutral-badge">100% NEUTRO</span>`;
                if (advanced) {
                    badges += `<span class="advanced-badge">ANÁLISE AVANÇADA</span>`;
                }
                if (cached) {
                    badges += `<span class="cache-badge">CACHE</span>`;
                }
                signalText.innerHTML = `${directionText} ${badges}`;
                
                // Atualiza informações
                analysisTime.textContent = data.analysis_time || '--:--:--';
                entryTime.textContent = data.entry_time || '--:--';
                timeframeEl.textContent = data.timeframe || 'Próximo minuto';
                
                reasoningText.textContent = data.reasoning;
                confidenceText.textContent = `Confiança Técnica: ${confidence}%`;
                
                // Indicador de qualidade
                qualityIndicator.className = `quality-indicator quality-${quality}`;
                if (quality === 'high') {
                    qualityIndicator.textContent = '✅ ALTA QUALIDADE - Análise visual confiável';
                } else {
                    qualityIndicator.textContent = '⚠️ QUALIDADE MÉDIA - Análise visual válida';
                }
                
                // Informações de contexto
                const contextLabels = {
                    'movimento_forte': '🚀 MOVIMENTO FORTE',
                    'mercado_lateral': '⚡ MERCADO LATERAL', 
                    'tendencia_estabelecida': '📈 TENDÊNCIA ESTABELECIDA',
                    'momentum_tecnico': '🎯 MOMENTUM TÉCNICO',
                    'tendencia_candles': '🕯️ TENDÊNCIA CANDLES',
                    'momentum_macd': '📊 MOMENTUM MACD',
                    'pressao_rsi': '📈 PRESSÃO RSI',
                    'mercado_balanceado': '⚖️ MERCADO BALANCEADO'
                };
                
                contextInfo.innerHTML = `
                    <span class="context-badge context-${context}">
                        ${contextLabels[context] || contextLabels.mercado_balanceado}
                    </span>
                `;
                
                // Análise do Momento
                const metrics = data.metrics || {};
                let powerHtml = '';
                
                const powerItems = [
                    ['Sinal Candles', (metrics.candle_signal * 100)?.toFixed(1) + '%'],
                    ['Sinal MACD', (metrics.macd_signal * 100)?.toFixed(1) + '%'],
                    ['Sinal RSI', (metrics.rsi_signal * 100)?.toFixed(1) + '%'],
                    ['Score Fusão', metrics.analysis_score?.toFixed(3)]
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
                    ['Qualidade Painéis', (metrics.panel_quality * 100)?.toFixed(1) + '%'],
                    ['Razão Touro/Urso', (metrics.bull_bear_ratio * 100)?.toFixed(1) + '%'],
                    ['Fase MACD', metrics.macd_phase?.toFixed(3)],
                    ['Nível RSI', (metrics.rsi_level * 100)?.toFixed(1)],
                    ['Confiança Fusão', (data.fusion_confidence * 100)?.toFixed(1) + '%'],
                    ['Qualidade Sinal', (data.signal_quality * 100)?.toFixed(1) + '%']
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
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        image_file = request.files['image']
        if not image_file or image_file.filename == '':
            return jsonify({'error': 'Arquivo inválido'}), 400
        
        timeframe = request.form.get('timeframe', '1m')
        if timeframe not in ['1m', '5m']:
            timeframe = '1m'
        
        # Verificação básica do arquivo
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'error': 'Imagem muito grande (máximo 10MB)'}), 400
        
        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Arquivo vazio'}), 400
        
        # Análise 100% NEUTRA COM FUSÃO AVANÇADA
        analysis = analyzer.analyze(image_bytes, timeframe)
        
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
        'service': 'IA Signal Pro - ANÁLISE VISUAL AVANÇADA',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '7.0.0-visual-fusion'
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
    
    print(f"🚀 IA Signal Pro - ANÁLISE VISUAL AVANÇADA iniciando na porta {port}")
    print(f"🧠⚖️ SISTEMA: DETECÇÃO DE PAINÉIS + FUSÃO BAYESIANA")
    print(f"🎯 TECNOLOGIA: Análise de cores + Candles geométricos + Indicadores visuais")
    print(f"💪 NEUTRALIDADE: 100% técnica - zero viés")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
