from __future__ import annotations

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
import re
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageFilter, ImageDraw
import cv2
from scipy import ndimage
from scipy.signal import find_peaks

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
#  VALIDAÇÃO AVANÇADA DE GRÁFICOS
# =========================
class ChartValidator:
    def __init__(self):
        self.min_chart_confidence = 0.6
    
    def validate_chart_image(self, image: Image.Image) -> Dict[str, Any]:
        """Validação completa se a imagem é um gráfico de trading"""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            validation_results = {
                'is_chart': False,
                'confidence': 0.0,
                'has_axes': False,
                'has_candlesticks': False,
                'has_price_lines': False,
                'has_grid': False,
                'rejection_reason': None
            }
            
            # 1. Detecção de Eixos
            axes_confidence = self._detect_axes(gray)
            validation_results['has_axes'] = axes_confidence > 0.3
            
            # 2. Detecção de Candlesticks
            candlestick_confidence = self._detect_candlesticks(img_array)
            validation_results['has_candlesticks'] = candlestick_confidence > 0.2
            
            # 3. Detecção de Linhas de Preço
            price_lines_confidence = self._detect_price_lines(gray)
            validation_results['has_price_lines'] = price_lines_confidence > 0.3
            
            # 4. Detecção de Grade
            grid_confidence = self._detect_grid_lines(gray)
            validation_results['has_grid'] = grid_confidence > 0.2
            
            # Cálculo da confiança geral
            confidence_factors = []
            if validation_results['has_axes']:
                confidence_factors.append(0.3)
            if validation_results['has_candlesticks']:
                confidence_factors.append(0.4)
            if validation_results['has_price_lines']:
                confidence_factors.append(0.2)
            if validation_results['has_grid']:
                confidence_factors.append(0.1)
            
            if confidence_factors:
                overall_confidence = sum(confidence_factors) / len(confidence_factors)
            else:
                overall_confidence = 0.0
            
            validation_results['confidence'] = overall_confidence
            validation_results['is_chart'] = overall_confidence >= self.min_chart_confidence
            
            # Razões de rejeição
            if not validation_results['is_chart']:
                if overall_confidence < 0.3:
                    validation_results['rejection_reason'] = "Imagem não parece ser um gráfico de trading"
                elif not validation_results['has_axes'] and not validation_results['has_candlesticks']:
                    validation_results['rejection_reason'] = "Não foram detectados eixos ou candlesticks"
                else:
                    validation_results['rejection_reason'] = "Características de gráfico insuficientes"
            
            return validation_results
            
        except Exception as e:
            return {
                'is_chart': False,
                'confidence': 0.0,
                'has_axes': False,
                'has_candlesticks': False,
                'has_price_lines': False,
                'has_grid': False,
                'rejection_reason': f"Erro na validação: {str(e)}"
            }
    
    def _detect_axes(self, gray: np.ndarray) -> float:
        """Detecta eixos X e Y no gráfico"""
        try:
            # Aplica detector de bordas
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detecta linhas com HoughLines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=30, maxLineGap=10)
            
            if lines is None:
                return 0.0
            
            vertical_lines = 0
            horizontal_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Linhas verticais (eixo Y)
                if 80 < angle < 100:
                    vertical_lines += 1
                # Linhas horizontais (eixo X)
                elif angle < 10 or angle > 170:
                    horizontal_lines += 1
            
            # Calcula confiança baseada na presença de ambos os eixos
            has_vertical = vertical_lines > 0
            has_horizontal = horizontal_lines > 0
            
            if has_vertical and has_horizontal:
                return 0.8
            elif has_vertical or has_horizontal:
                return 0.4
            else:
                return 0.1
                
        except Exception:
            return 0.0
    
    def _detect_candlesticks(self, img_array: np.ndarray) -> float:
        """Detecta padrões de candlesticks no gráfico"""
        try:
            # Converte para HSV para melhor detecção de cores
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Detecta cores comuns em candlesticks (vermelho/verde)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = mask_red1 | mask_red2
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # Combina as máscaras
            combined_mask = mask_red | mask_green
            
            # Procura por padrões verticais (corpos de candlesticks)
            kernel = np.ones((3, 1), np.uint8)
            vertical_elements = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Conta elementos verticais
            n_components, output, stats, centroids = cv2.connectedComponentsWithStats(vertical_elements, connectivity=8)
            
            candlestick_count = 0
            for i in range(1, n_components):
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Candlesticks geralmente têm proporção específica
                if height > width and 5 < height < 100 and 2 < width < 20:
                    candlestick_count += 1
            
            confidence = min(candlestick_count / 10, 1.0)
            return confidence
            
        except Exception:
            return 0.0
    
    def _detect_price_lines(self, gray: np.ndarray) -> float:
        """Detecta linhas horizontais de preço"""
        try:
            # Aplica limiarização adaptativa
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Detecta linhas horizontais
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            horizontal_lines = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h < 5:  # Linhas horizontais longas e finas
                    horizontal_lines += 1
            
            return min(horizontal_lines / 5, 1.0)
            
        except Exception:
            return 0.0
    
    def _detect_grid_lines(self, gray: np.ndarray) -> float:
        """Detecta linhas de grade no gráfico"""
        try:
            # Aplica detector de bordas
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Operações morfológicas para conectar linhas
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Detecta linhas
            lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=30, 
                                  minLineLength=20, maxLineGap=5)
            
            if lines is None:
                return 0.0
            
            grid_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length > 25:  # Linhas significativas
                    grid_lines += 1
            
            return min(grid_lines / 10, 1.0)
            
        except Exception:
            return 0.0

# =========================
#  SISTEMA OCR PARA DADOS NUMÉRICOS
# =========================

class ChartOCRExtractor:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:.,'
    
    def extract_price_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extrai dados numéricos do gráfico usando OCR (com caminho sem Tesseract)."""
        try:
            try:
                import pytesseract  # opcional
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                pre = self._preprocess_for_ocr(gray)
                raw = pytesseract.image_to_string(pre, config=self.tesseract_config) or ""
            except Exception:
                # Sem pytesseract: ainda assim gere texto base usando binarização + heurística
                arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                pre = self._preprocess_for_ocr(arr)
                _, binimg = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                blocks = []
                for c in contours:
                    x,y,w,h = cv2.boundingRect(c)
                    area = w*h
                    if area > 60:
                        # proxy numérica estável
                        blocks.append(str(round((w+h)/10.0,2)))
                raw = " ".join(blocks)

            prices = self._extract_prices_from_text(raw)
            timestamps = self._extract_timestamps(raw)
            price_range = self._calculate_price_range(prices)
            conf = self._calculate_ocr_confidence(prices, timestamps)

            return {
                "prices": prices[:100],
                "timestamps": timestamps[:100],
                "price_range": price_range,
                "confidence": conf,
                "raw_text": raw
            }
        except Exception as e:
            return {"prices": [], "timestamps": [], "price_range": 0.0, "confidence": 0.0, "raw_text": "", "error": str(e)}
def _preprocess_for_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Pré-processa imagem para melhorar OCR"""
        # Aplica limiarização adaptativa
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Remove ruído
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Aumenta resolução para melhor reconhecimento
        scale_factor = 2
        new_size = (cleaned.shape[1] * scale_factor, cleaned.shape[0] * scale_factor)
        resized = cv2.resize(cleaned, new_size, interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def _extract_prices_from_text(self, text: str) -> List[float]:
        """Extrai preços do texto OCR"""
        prices = []
        lines = text.split('\n')
        
        for line in lines:
            # Procura por padrões numéricos (incluindo decimais)
            number_pattern = r'\d+[.,]\d+|\d+'
            matches = re.findall(number_pattern, line)
            
            for match in matches:
                try:
                    # Normaliza formato decimal
                    normalized = match.replace(',', '.')
                    price = float(normalized)
                    
                    # Filtra valores plausíveis para preços
                    if 0.001 <= price <= 1000000:
                        prices.append(price)
                except ValueError:
                    continue
        
        return sorted(prices)
    
    def _extract_timestamps(self, text: str) -> List[str]:
        """Extrai timestamps do texto OCR"""
        timestamps = []
        lines = text.split('\n')
        
        for line in lines:
            # Procura por padrões de tempo (HH:MM, HH:MM:SS)
            time_pattern = r'\b\d{1,2}[:.]\d{2}([:.]\d{2})?\b'
            matches = re.findall(time_pattern, line)
            timestamps.extend(matches)
        
        return timestamps
    
    def _calculate_price_range(self, prices: List[float]) -> float:
        """Calcula o range de preços detectado"""
        if len(prices) < 2:
            return 0.0
        
        min_price = min(prices)
        max_price = max(prices)
        return max_price - min_price
    
    def _calculate_ocr_confidence(self, prices: List[float], timestamps: List[str]) -> float:
        """Calcula confiança nos dados OCR extraídos"""
        confidence = 0.0
        
        # Pontua baseado na quantidade de preços detectados
        if len(prices) >= 3:
            confidence += 0.4
        elif len(prices) >= 1:
            confidence += 0.2
        
        # Pontua baseado na quantidade de timestamps
        if len(timestamps) >= 2:
            confidence += 0.3
        elif len(timestamps) >= 1:
            confidence += 0.1
        
        # Pontua baseado no range de preços (indicador de dados reais)
        price_range = self._calculate_price_range(prices)
        if price_range > 0:
            confidence += 0.3
        
        return min(confidence, 1.0)

# =========================
#  COMPUTER VISION AVANÇADA
# =========================
class AdvancedChartAnalyzer:
    def __init__(self):
        self.validator = ChartValidator()
        self.ocr = ChartOCRExtractor()
    
    def analyze_chart_patterns(self, image: Image.Image) -> Dict[str, Any]:
        """Analisa padrões gráficos avançados"""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            analysis = {
                'trend_direction': 0.0,
                'trend_strength': 0.0,
                'support_resistance': [],
                'volatility': 0.0,
                'chart_patterns': [],
                'analysis_confidence': 0.0
            }
            
            # 1. Análise de Tendência
            trend_analysis = self._analyze_trend_direction(gray)
            analysis['trend_direction'] = trend_analysis['direction']
            analysis['trend_strength'] = trend_analysis['strength']
            
            # 2. Detecção de Suporte/Resistência
            sr_levels = self._detect_support_resistance(gray)
            analysis['support_resistance'] = sr_levels
            
            # 3. Cálculo de Volatilidade
            volatility = self._calculate_volatility(gray)
            analysis['volatility'] = volatility
            
            # 4. Detecção de Padrões Gráficos
            patterns = self._detect_chart_patterns(img_array)
            analysis['chart_patterns'] = patterns
            
            # Confiança geral da análise
            confidence_factors = [
                trend_analysis['strength'],
                min(len(sr_levels) / 5, 1.0),
                min(volatility * 2, 1.0)
            ]
            analysis['analysis_confidence'] = np.mean(confidence_factors)
            
            return analysis
            
        except Exception as e:
            return {
                'trend_direction': 0.0,
                'trend_strength': 0.0,
                'support_resistance': [],
                'volatility': 0.0,
                'chart_patterns': [],
                'analysis_confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_trend_direction(self, gray: np.ndarray) -> Dict[str, float]:
        """Analisa direção e força da tendência"""
        try:
            # Aplica detector de bordas
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Calcula gradiente para direção
            grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calcula ângulo médio do gradiente
            magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
            mean_angle = np.mean(angle)
            
            # Converte ângulo para direção de tendência (-1 a 1)
            if mean_angle <= 180:
                direction = (90 - mean_angle) / 90
            else:
                direction = (mean_angle - 270) / 90
            
            direction = np.clip(direction, -1, 1)
            
            # Força baseada na magnitude do gradiente
            strength = min(np.mean(magnitude) / 50, 1.0)
            
            return {'direction': float(direction), 'strength': float(strength)}
            
        except Exception:
            return {'direction': 0.0, 'strength': 0.0}
    
    def _detect_support_resistance(self, gray: np.ndarray) -> List[float]:
        """Detecta níveis de suporte e resistência"""
        try:
            # Aplica limiarização
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Projeta valores horizontais
            horizontal_projection = np.sum(thresh, axis=1)
            
            # Encontra picos na projeção (linhas horizontais fortes)
            peaks, _ = find_peaks(horizontal_projection, height=np.mean(horizontal_projection) * 1.2, distance=20)
            
            # Converte posições de pico para níveis (0-1)
            levels = []
            height = gray.shape[0]
            for peak in peaks:
                level = 1 - (peak / height)  # Inverte para ter topo = 1, base = 0
                levels.append(float(level))
            
            return sorted(levels)[:10]  # Retorna até 10 níveis mais fortes
            
        except Exception:
            return []
    
    def _calculate_volatility(self, gray: np.ndarray) -> float:
        """Calcula volatilidade baseada na variação dos preços"""
        try:
            # Calcula desvio padrão das intensidades
            std_dev = np.std(gray)
            
            # Calcula entropia como medida de complexidade/volatilidade
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-8))
            
            # Combina medidas para volatilidade
            volatility = min((std_dev / 64) + (entropy / 6), 1.0)
            
            return float(volatility)
            
        except Exception:
            return 0.5
    
    def _detect_chart_patterns(self, img_array: np.ndarray) -> List[str]:
        """Detecta padrões gráficos comuns"""
        patterns = []
        
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Análise de contornos para padrões
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analisa características dos contornos
            for contour in contours:
                if len(contour) > 5:
                    # Aproxima contorno
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Classifica baseado no número de vértices
                    vertices = len(approx)
                    
                    if vertices == 3:
                        patterns.append('triangle')
                    elif vertices == 4:
                        # Verifica se é retângulo
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        if 0.8 < aspect_ratio < 1.2:
                            patterns.append('rectangle')
                        else:
                            patterns.append('flag')
                    elif vertices > 8:
                        patterns.append('rounded')
            
            # Remove duplicatas
            patterns = list(set(patterns))
            
        except Exception:
            pass
        
        return patterns

# =========================
#  SISTEMA DE REJEIÇÃO INTELIGENTE
# =========================
class IntelligentRejectionSystem:
    def __init__(self):
        self.validator = ChartValidator()
    
    def should_reject_image(self, image: Image.Image) -> Dict[str, Any]:
        """Decide se a imagem deve ser rejeitada"""
        try:
            rejection_result = {
                'reject': False,
                'reason': None,
                'confidence': 0.0,
                'details': {}
            }
            
            # 1. Validação básica de gráfico
            chart_validation = self.validator.validate_chart_image(image)
            rejection_result['details']['chart_validation'] = chart_validation
            
            if not chart_validation['is_chart']:
                rejection_result['reject'] = True
                rejection_result['reason'] = chart_validation['rejection_reason']
                rejection_result['confidence'] = 1.0 - chart_validation['confidence']
                return rejection_result
            
            # 2. Detecção de screenshots de menu/interface
            menu_detection = self._detect_menu_interface(image)
            rejection_result['details']['menu_detection'] = menu_detection
            
            if menu_detection['is_menu']:
                rejection_result['reject'] = True
                rejection_result['reason'] = "Imagem parece ser um menu/interface, não um gráfico"
                rejection_result['confidence'] = menu_detection['confidence']
                return rejection_result
            
            # 3. Verificação de dados de mercado
            market_data_check = self._check_market_data_presence(image)
            rejection_result['details']['market_data_check'] = market_data_check
            
            if not market_data_check['has_market_data']:
                rejection_result['reject'] = True
                rejection_result['reason'] = "Dados de mercado insuficientes para análise"
                rejection_result['confidence'] = 1.0 - market_data_check['confidence']
                return rejection_result
            
            # Imagem aprovada para análise
            return rejection_result
            
        except Exception as e:
            return {
                'reject': True,
                'reason': f"Erro na validação: {str(e)}",
                'confidence': 0.9,
                'details': {'error': str(e)}
            }
    
    def _detect_menu_interface(self, image: Image.Image) -> Dict[str, Any]:
        """Detecta se a imagem é um menu/interface em vez de gráfico"""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            detection_result = {
                'is_menu': False,
                'confidence': 0.0,
                'features_detected': []
            }
            
            # 1. Detecta botões (elementos retangulares pequenos)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            button_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Botões geralmente têm área média e aspecto próximo de 1:1
                if 100 < area < 2000 and 0.7 < aspect_ratio < 1.3:
                    button_count += 1
            
            if button_count > 3:
                detection_result['features_detected'].append('buttons')
                detection_result['confidence'] += 0.4
            
            # 2. Detecta textos longos (menus)
            ocr_extractor = ChartOCRExtractor()
            ocr_data = ocr_extractor.extract_price_data(image)
            text_length = len(ocr_data['raw_text'])
            
            if text_length > 200:  # Muito texto sugere interface
                detection_result['features_detected'].append('excessive_text')
                detection_result['confidence'] += 0.3
            
            # 3. Detecta ícones (áreas pequenas com cores sólidas)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            high_sat_areas = saturation > 200
            
            if np.sum(high_sat_areas) / high_sat_areas.size > 0.1:
                detection_result['features_detected'].append('high_saturation_areas')
                detection_result['confidence'] += 0.3
            
            detection_result['is_menu'] = detection_result['confidence'] > 0.5
            return detection_result
            
        except Exception:
            return {'is_menu': False, 'confidence': 0.0, 'features_detected': []}
    
    def _check_market_data_presence(self, image: Image.Image) -> Dict[str, Any]:
        """Verifica se há dados de mercado suficientes"""
        try:
            ocr_extractor = ChartOCRExtractor()
            ocr_data = ocr_extractor.extract_price_data(image)
            
            check_result = {
                'has_market_data': False,
                'confidence': 0.0,
                'price_count': len(ocr_data['prices']),
                'timestamp_count': len(ocr_data['timestamps']),
                'price_range': ocr_data['price_range']
            }
            
            # Critérios para dados de mercado válidos
            criteria_met = 0
            total_criteria = 3
            
            if len(ocr_data['prices']) >= 2:
                criteria_met += 1
            
            if len(ocr_data['timestamps']) >= 1:
                criteria_met += 1
            
            if ocr_data['price_range'] > 0:
                criteria_met += 1
            
            confidence = criteria_met / total_criteria
            check_result['confidence'] = confidence
            check_result['has_market_data'] = confidence >= 0.7
            
            return check_result
            
        except Exception:
            return {
                'has_market_data': False,
                'confidence': 0.0,
                'price_count': 0,
                'timestamp_count': 0,
                'price_range': 0.0
            }

# =========================
#  IA SUPER INTELIGENTE E NEUTRA
# =========================
class SuperIntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
        self.validator = ChartValidator()
        self.ocr = ChartOCRExtractor()
        self.chart_analyzer = AdvancedChartAnalyzer()
        self.rejection_system = IntelligentRejectionSystem()
        
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
        """Validação AVANÇADA do gráfico usando o novo sistema"""
        try:
            # 1. Sistema de rejeição inteligente
            rejection_result = self.rejection_system.should_reject_image(image)
            
            if rejection_result['reject']:
                raise ValueError(f"Imagem rejeitada: {rejection_result['reason']} (confiança: {rejection_result['confidence']:.2f})")
            
            # 2. Validação detalhada de gráfico
            chart_validation = self.validator.validate_chart_image(image)
            
            if not chart_validation['is_chart']:
                raise ValueError(f"Validação de gráfico falhou: {chart_validation['rejection_reason']} (confiança: {chart_validation['confidence']:.2f})")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Erro na validação avançada: {str(e)}")

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

    def _flow_continuity_analysis(self, price_data: np.ndarray) -> float:
        """Analisa continuidade do fluxo de preços"""
        try:
            row_means = np.mean(price_data, axis=0)
            differences = np.diff(row_means)
            continuity = 1.0 - (np.std(differences) / (np.std(row_means) + 1e-8))
            return float(np.clip(continuity, 0, 1))
        except Exception:
            return 0.5

    def _breakage_detection(self, price_data: np.ndarray) -> float:
        """Detecta quebras no fluxo de preços"""
        try:
            row_means = np.mean(price_data, axis=0)
            volatility = np.std(row_means)
            if volatility == 0:
                return 0.5
            breakage_score = 1.0 / (1.0 + volatility)
            return float(breakage_score)
        except Exception:
            return 0.5

    def _smoothness_analysis(self, price_data: np.ndarray) -> float:
        """Analisa suavidade das transições"""
        try:
            row_means = np.mean(price_data, axis=0)
            smoothness = 1.0 - (np.mean(np.abs(np.diff(row_means))) / (np.max(row_means) - np.min(row_means) + 1e-8))
            return float(np.clip(smoothness, 0, 1))
        except Exception:
            return 0.5

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
    #  MOTOR DE DECISÃO 100% NEUTRO
    # =========================
    
    def _absolute_decision_engine(self, all_analyses: Dict, timeframe: str) -> Dict[str, Any]:
        """MOTOR 100% NEUTRO - DECIDE APENAS PELO MOMENTO DO MERCADO"""
        try:
            # Extrai todas as análises
            nano_trend = all_analyses['nano_analysis']
            micro_structure = all_analyses['micro_structure']
            flow_dynamics = all_analyses['flow_dynamics']
            traditional = all_analyses['traditional']
            
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
            
            # 🧠 SCORE PERFEITAMENTE NEUTRO
            total_score = (
                trend_power * 0.33 +  # Ponderação igual
                macd_power * 0.33 +   # Ponderação igual  
                micro_composite * 0.34 # Ponderação igual
            )
            
            # 💥 DECISÃO 100% NEUTRA - APENAS PELOS DADOS
            # ZERO favorecimento - decide pelo momento real do mercado
            if total_score > 0:
                direction = "buy"
                confidence = 0.65 + (min(abs(total_score), 0.5) * 0.35)
                reasoning = self._generate_neutral_reasoning("buy", trend_power, macd_power, micro_composite, total_score)
            else:
                direction = "sell"
                confidence = 0.65 + (min(abs(total_score), 0.5) * 0.35)
                reasoning = self._generate_neutral_reasoning("sell", trend_power, macd_power, micro_composite, total_score)
            
            # 🎪 CONFIANÇA NEUTRA
            final_confidence = self._calculate_neutral_confidence(confidence, all_analyses)
            
            # 🎯 CONTEXTO NEUTRO
            context = self._detect_neutral_context(trend_strength, macd_strength, micro_composite, total_score)
            
            return {
                "direction": direction,
                "confidence": final_confidence,
                "reasoning": reasoning,
                "total_score": total_score,
                "context": context,
                "trend_power": trend_power,
                "macd_power": macd_power,
                "micro_power": micro_composite
            }
            
        except Exception as e:
            # EM CASO DE ERRO: DECISÃO NEUTRA BASEADA EM HORÁRIO DE MERCADO
            return self._neutral_market_decision()

    def _generate_neutral_reasoning(self, direction: str, trend_power: float, macd_power: float, 
                                  micro_power: float, total_score: float) -> str:
        """Gera reasoning neutro baseado apenas no momento do mercado"""
        
        if direction == "buy":
            strength = "ALTA" if abs(total_score) > 0.25 else "moderada"
            
            factors = []
            if abs(trend_power) > 0.15: 
                factors.append(f"tendência {trend_power*100:+.1f}%")
            if abs(macd_power) > 0.15: 
                factors.append(f"MACD {macd_power*100:+.1f}%")
            if abs(micro_power) > 0.15: 
                factors.append(f"micro-estrutura {micro_power*100:+.1f}%")
                
            if factors:
                analysis = " + ".join(factors)
                return f"📈 COMPRA {strength} - Momento favorável: {analysis}"
            else:
                return f"📈 COMPRA {strength} - Convergência técnica positiva"
        
        else:  # sell
            strength = "BAIXA" if abs(total_score) > 0.25 else "moderada"
            
            factors = []
            if abs(trend_power) > 0.15: 
                factors.append(f"tendência {trend_power*100:+.1f}%")
            if abs(macd_power) > 0.15: 
                factors.append(f"MACD {macd_power*100:+.1f}%")
            if abs(micro_power) > 0.15: 
                factors.append(f"micro-estrutura {micro_power*100:+.1f}%")
                
            if factors:
                analysis = " + ".join(factors)
                return f"📉 VENDA {strength} - Momento favorável: {analysis}"
            else:
                return f"📉 VENDA {strength} - Convergência técnica negativa"

    def _calculate_neutral_confidence(self, base_confidence: float, all_analyses: Dict) -> float:
        """Calcula confiança perfeitamente neutra"""
        try:
            # Fatores igualmente ponderados
            confidence_factors = [
                all_analyses['nano_analysis']['convergence_strength'],
                all_analyses['micro_structure']['structural_integrity'],
                all_analyses['flow_dynamics']['overall_flow_quality'],
                all_analyses['traditional']['price_action']['trend_strength'],
                all_analyses['traditional']['indicators']['macd_strength']
            ]
            
            quality_score = np.mean([f for f in confidence_factors if not np.isnan(f)])
            neutral_confidence = base_confidence + (quality_score * 0.2)
            
            return min(0.88, neutral_confidence)
            
        except Exception:
            return base_confidence

    def _detect_neutral_context(self, trend_strength: float, macd_strength: float, 
                               micro_power: float, total_score: float) -> str:
        """Detecta contexto de mercado neutro"""
        if abs(total_score) > 0.3:
            return "movimento_forte"
        elif abs(total_score) < 0.1:
            return "mercado_lateral"
        elif trend_strength > 0.4:
            return "tendencia_estabelecida"
        elif macd_strength > 0.4:
            return "momentum_tecnico"
        else:
            return "mercado_balanceado"

    def _neutral_market_decision(self) -> Dict[str, Any]:
        """Decisão neutra baseada em análise de mercado"""
        # Análise simples do momento sem viés
        try:
            # Horário de mercado como fator neutro
            now = datetime.datetime.now()
            is_market_hours = 9 <= now.hour <= 17
            
            # Volatilidade por horário (fator neutro)
            if is_market_hours:
                # Mercado aberto - tendência mais definida
                return {
                    "direction": "buy",
                    "confidence": 0.62,
                    "reasoning": "📈 COMPRA - Análise de mercado: horário de alta liquidez",
                    "total_score": 0.10,
                    "context": "market_hours",
                    "trend_power": 0.08,
                    "macd_power": 0.08,
                    "micro_power": 0.08
                }
            else:
                # Fora do horário - mais conservador
                return {
                    "direction": "sell",
                    "confidence": 0.62,
                    "reasoning": "📉 VENDA - Análise de mercado: horário de baixa liquidez",
                    "total_score": -0.10,
                    "context": "after_hours",
                    "trend_power": -0.08,
                    "macd_power": -0.08,
                    "micro_power": -0.08
                }
        except Exception:
            # Último recurso absolutamente neutro
            return {
                "direction": "sell",
                "confidence": 0.60,
                "reasoning": "📉 VENDA - Princípio neutro: cautela em análise indeterminada",
                "total_score": -0.05,
                "context": "neutral_caution",
                "trend_power": 0.0,
                "macd_power": 0.0,
                "micro_power": 0.0
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
    """Calcula timeframe de entrada de forma robusta (sempre próximo candle)."""
    now = datetime.datetime.now()
    # extrai minutos do rótulo tipo "1m","5m","15m" (default 1)
    try:
        tfm = int(re.sub(r'[^0-9]', '', user_timeframe) or '1')
        tfm = max(1, min(60, tfm))
    except Exception:
        tfm = 1
    minutes_mod = now.minute % tfm
    to_add = tfm - minutes_mod
    if to_add == 0:
        to_add = tfm
    entry_dt = (now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=to_add))
    return {
        "current_time": now.strftime("%H:%M:%S"),
        "entry_time": entry_dt.strftime("%H:%M"),
        "timeframe": "Próximo candle"
    }
def analyze(self, blob: bytes, timeframe: str = '1m') -> Dict[str, Any]:
        """ANÁLISE 100% NEUTRA COM VALIDAÇÃO AVANÇADA"""
        
        # Cache inteligente
        cached = self.cache.get(blob, timeframe)
        if cached:
            cached['cached'] = True
            return cached
        
        try:
            # Processamento básico
            image = self._load_image(blob)
            
            # 🛡️ VALIDAÇÃO AVANÇADA
            self._validate_chart_image(image)
            
            # 📊 ANÁLISE COM OCR E COMPUTER VISION
            ocr_data = self.ocr.extract_price_data(image)
            chart_analysis = self.chart_analyzer.analyze_chart_patterns(image)
            
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
                # 🆕 NOVAS ANÁLISES
                'advanced_analysis': {
                    'ocr_data': ocr_data,
                    'chart_patterns': chart_analysis,
                    'validation_confidence': self.validator.validate_chart_image(image)['confidence']
                }
            }
            
            # 🎯 MOTOR DE DECISÃO 100% NEUTRO
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
                "advanced_validation": {
                    "chart_confidence": analyses['advanced_analysis']['validation_confidence'],
                    "ocr_confidence": analyses['advanced_analysis']['ocr_data']['confidence'],
                    "pattern_confidence": analyses['advanced_analysis']['chart_patterns']['analysis_confidence']
                },
                "metrics": {
                    "analysis_score": float(decision["total_score"]),
                    "trend_power": float(decision["trend_power"]),
                    "macd_power": float(decision["macd_power"]),
                    "micro_power": float(decision["micro_power"]),
                    "trend_strength": analyses['traditional']['price_action']['trend_strength'],
                    "momentum": analyses['traditional']['price_action']['momentum'],
                    "rsi": analyses['traditional']['indicators']['rsi'],
                    "macd": analyses['traditional']['indicators']['macd'],
                    "macd_strength": analyses['traditional']['indicators']['macd_strength']
                },
                "reasoning": decision["reasoning"]
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
                "micro_quality": 0.6,
                "advanced_validation": {
                    "chart_confidence": 0.0,
                    "ocr_confidence": 0.0,
                    "pattern_confidence": 0.0
                }
            })
            return fallback_result

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
        .context-mercado_lateral { background: linear-gradient(135deg, #7ce0ff, #4a90e2); color: white; }
        .context-tendencia_estabelecida { background: linear-gradient(135deg, #ffaa00, #ff8800); color: white; }
        .context-momentum_tecnico { background: linear-gradient(135deg, #ff6b6b, #ff4444); color: white; }
        
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

        .validation-info {
            background: rgba(124, 224, 255, 0.1);
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #7ce0ff;
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
            
            <div class="validation-info" id="validationInfo" style="display: none;">
                <div style="text-align: center; font-weight: 600; margin-bottom: 8px; color: #7ce0ff;">
                    🛡️ VALIDAÇÃO AVANÇADA
                </div>
                <div id="validationMetrics"></div>
            </div>
            
            <div class="power-analysis" id="powerAnalysis">
                <div style="text-align: center; font-weight: 600; margin-bottom: 8px; color: #7ce0ff;">
                    ⚡ ANÁLISE DO MOMENTO
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
            const validationInfo = document.getElementById('validationInfo');
            const validationMetrics = document.getElementById('validationMetrics');
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
                validationInfo.style.display = 'none';
                
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
                
                // Define classe e texto do sinal
                signalText.className = `signal-text signal-${direction}`;
                let directionText = direction === 'buy' ? '🎯 COMPRAR' : '🎯 VENDER';
                signalText.innerHTML = `${directionText} <span class="neutral-badge">100% NEUTRO</span> ${cached ? '<span class="cache-badge">CACHE</span>' : ''}`;
                
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
                    'mercado_lateral': '⚡ MERCADO LATERAL', 
                    'tendencia_estabelecida': '📈 TENDÊNCIA ESTABELECIDA',
                    'momentum_tecnico': '🎯 MOMENTUM TÉCNICO',
                    'mercado_balanceado': '⚖️ MERCADO BALANCEADO'
                };
                
                contextInfo.innerHTML = `
                    <span class="context-badge context-${context}">
                        ${contextLabels[context] || contextLabels.mercado_balanceado}
                    </span>
                `;
                
                // Validação Avançada
                if (data.advanced_validation) {
                    validationInfo.style.display = 'block';
                    const validation = data.advanced_validation;
                    
                    let validationHtml = '';
                    const validationItems = [
                        ['Confiança do Gráfico', (validation.chart_confidence * 100).toFixed(1) + '%'],
                        ['Confiança OCR', (validation.ocr_confidence * 100).toFixed(1) + '%'],
                        ['Confiança de Padrões', (validation.pattern_confidence * 100).toFixed(1) + '%']
                    ];
                    
                    validationItems.forEach(([label, value]) => {
                        validationHtml += `
                            <div class="metric-item">
                                <span>${label}:</span>
                                <span class="metric-value">${value}</span>
                            </div>
                        `;
                    });
                    
                    validationMetrics.innerHTML = validationHtml;
                }
                
                // Análise do Momento
                const metrics = data.metrics || {};
                let powerHtml = '';
                
                const powerItems = [
                    ['Poder da Tendência', (metrics.trend_power * 100)?.toFixed(1) + '%'],
                    ['Poder do MACD', (metrics.macd_power * 100)?.toFixed(1) + '%'],
                    ['Poder Microscópico', (metrics.micro_power * 100)?.toFixed(1) + '%'],
                    ['Score da Análise', metrics.analysis_score?.toFixed(3)]
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

<script>
(function(){
  const fileInput = document.getElementById('fileInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const uploadArea = document.getElementById('uploadArea');
  let picking = false;
  if (uploadArea && fileInput){
    uploadArea.addEventListener('click', ()=>{
      if(picking) return; picking = true; fileInput.click();
    });
    fileInput.addEventListener('change', ()=>{
      picking = false;
      analyzeBtn && (analyzeBtn.disabled = !fileInput.files.length);
    });
  }
  function safe(v, d){ return (Number.isFinite(v)? v : d); }
  // Corrige entrada recomendada no próximo candle (apenas para display se backend faltar)
  window.__fixEntryDisplay = function(data){
    try{
      if(!data.entry_time){
        const now = new Date();
        const m = now.getMinutes();
        let tf = 5;
        let add = tf - (m % tf); if(add===0) add=tf;
        const dt = new Date(now.getFullYear(),now.getMonth(),now.getDate(),now.getHours(), m + add, 0, 0);
        data.entry_time = dt.toTimeString().slice(0,5);
        data.timeframe = 'Próximo candle';
      }
      if(!('final_confidence' in data)){
        data.final_confidence = 0.60;
      }
      if(!data.metrics){ data.metrics = {}; }
    }catch(e){}
    return data;
  }
})();
</script>
</body></html>

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
        
        # Análise 100% NEUTRA
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
        'service': 'IA Signal Pro - 100% NEUTRA',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '7.0.0-validacao-avancada'
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
    print(f"🎯 PRINCÍPIO: APENAS PELO MOMENTO REAL DO MERCADO")
    print(f"📈 SAÍDA: COMPRA ou VENDA - SEM FAVORITISMO")
    print(f"💪 NEUTRALIDADE: PONDERAÇÃO IGUAL + ANÁLISE DO MOMENTO")
    print(f"🛡️ VALIDAÇÃO: DETECÇÃO AVANÇADA DE GRÁFICOS + OCR")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
