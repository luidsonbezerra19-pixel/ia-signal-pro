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
import cv2
import numpy as np
import pytesseract
from typing import Any, Dict, Optional, List, Tuple
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageFilter, ImageEnhance

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
#  FASE 1: SISTEMA DE OCR AVANÇADO
# =========================
# =========================
#  FASE 1: SISTEMA DE OCR AVANÇADO - CORRIGIDO
# =========================
# =========================
#  FASE 1: SISTEMA DE OCR AVANÇADO - TOTALMENTE CORRIGIDO
# =========================

class TextDataExtractor:

def _scan_and_ocr(self, image: Image.Image, y_slices=((0.78,1.00),(0.58,0.80)), x_steps=4):
    """
    Varre horizontal/verticalmente as faixas inferiores (RSI/ADX) e médias (MACD),
    tentando múltiplos recortes para maximizar a leitura de texto.
    Retorna o texto mais longo obtido (melhor heurística).
    """
    w, h = image.size
    texts = []
    for y0, y1 in y_slices:
        y_top = int(h*min(y0,y1)); y_bot = int(h*max(y0,y1))
        for i in range(x_steps):
            x0 = int((w/x_steps)*i)
            x1 = int((w/x_steps)*(i+1))
            crop = image.crop((x0, y_top, x1, y_bot))
            t = self._ocr_text(crop, psm=6)
            if t and len(t.strip())>0:
                texts.append(t)
    # fallback: usa toda a faixa do rodapé
    full_bot = image.crop((0, int(h*0.76), w, h))
    texts.append(self._ocr_text(full_bot, psm=6))
    # escolhe o mais informativo
    texts = [t for t in texts if t]
    if not texts:
        return ""
    return max(texts, key=lambda s: len(s.strip()))
    """
    Substituição do OCR antigo por OCR inteligente para prints do Ebinex (tema escuro).
    Mantém o mesmo nome da classe e a mesma assinatura pública para NÃO quebrar o restante do app.
    """
    def __init__(self):
        try:
            import pytesseract  # garante import em runtime
        except Exception:
            pass

    
def _preprocess(self, image: Image.Image) -> Image.Image:
    """
    Pré-processamento v2 para tema escuro (Ebinex):
    - Converte para escala de cinza
    - Detecta tema escuro e inverte
    - Aumenta contraste
    - Adaptive threshold + dilatação leve para reforçar fonte fina
    - Upscale forte (x2) para OCR
    """
    import numpy as np, cv2
    from PIL import ImageOps, ImageFilter

    img = image.convert('L')
    np_img = np.array(img)

    # Detecta tema escuro pela média
    dark = np.mean(np_img) < 120
    if dark:
        np_img = 255 - np_img  # inverte para fundo claro / texto escuro

    # Autocontraste e nitidez
    img = Image.fromarray(np_img).convert('L')
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)

    # Adaptive threshold para destacar texto
    np_img = np.array(img)
    thr = cv2.adaptiveThreshold(np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)

    # Dilatação leve para engrossar caracteres finos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thr = cv2.dilate(thr, kernel, iterations=1)

    # Upscale forte
    h, w = thr.shape
    scale = 2
    thr = cv2.resize(thr, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(thr)

    
def _segment_panels(self, image: Image.Image):
    """
    Segmentação proporcional ajustada:
    - topo: 0.00–0.58 (candles)
    - meio: 0.58–0.80 (MACD)
    - rodapé: 0.78–1.00 (RSI/ADX) com leve sobreposição
    """
    w, h = image.size
    top = image.crop((0, 0, w, int(h*0.58)))
    mid = image.crop((0, int(h*0.58), w, int(h*0.80)))
    bot = image.crop((0, int(h*0.78), w, h))  # sobrepõe um pouco o mid
    return top, mid, bot

    
def _ocr_text(self, img: Image.Image, psm: int = 6) -> str:
    import pytesseract
    cfg_base = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%.,:- '
    configs = [f'--oem 3 --psm {psm} ' + cfg_base]
    if psm != 7:
        configs.append('--oem 3 --psm 7 ' + cfg_base)
    text = ""
    for cfg in configs:
        try:
            t = pytesseract.image_to_string(img, config=cfg) or ""
            if len(t.strip()) > len(text.strip()):
                text = t
        except Exception:
            continue
    return text

    def _parse_values(self, text: str) -> dict:
        import re
        t = (text or "")
        t = t.replace("−", "-").replace("–", "-").replace("—", "-")
        t = t.replace("RSI I4", "RSI 14")

        def pick(patterns):
            for pat in patterns:
                m = re.search(pat, t, flags=re.IGNORECASE)
                if m:
                    v = m.group(1).replace(",", ".")
                    try:
                        return float(v)
                    except Exception:
                        continue
            m = re.search(r'(-?\d+[.,]\d+)', t)
            if m:
                try:
                    return float(m.group(1).replace(",", "."))
                except Exception:
                    return None
            return None

        rsi = pick([r'RSI\s*\d*\s*[:=-]?\s*(-?\d+[.,]\d+)'])
        macd = pick([r'MACD[^\d-]*(-?\d+[.,]\d+)', r'MACD\s*[:=-]?\s*(-?\d+[.,]\d+)'])
        adx = pick([r'ADX[^\d-]*(-?\d+[.,]\d+)', r'ADX\s*[:=-]?\s*(-?\d+[.,]\d+)'])

        if rsi is not None and not (0.0 <= rsi <= 100.0):
            rsi = None

        hits = int(rsi is not None) + int(macd is not None) + int(adx is not None)
        conf = 0.0 if hits == 0 else (0.6 if hits == 1 else (0.8 if hits == 2 else 0.9))

        return {
            "raw_text": t,
            "rsi_value": rsi,
            "macd_value": macd,
            "adx_value": adx,
            "confidence": conf
        }

    
def extract_text_data(self, image: Image.Image) -> Dict[str, Any]:
    """
    Lê a imagem com pré-processamento, varredura adaptativa e retorna os valores parseados.
    """
    pre = self._preprocess(image)
    top, mid, bot = self._segment_panels(pre)

    # Primeira tentativa: OCR focal nos painéis
    txt_mid = self._ocr_text(mid, psm=6)
    txt_bot = self._ocr_text(bot, psm=6)
    combined = (txt_mid or "") + "\\n" + (txt_bot or "")

    # Se vazio/curto, faz varredura adaptativa no rodapé e meio
    if len(combined.strip()) < 5:
        scanned = self._scan_and_ocr(pre, y_slices=((0.76,1.00),(0.56,0.82)), x_steps=5)
        combined = (combined + "
" + scanned).strip()

    parsed = self._parse_values(combined)

    # Logs de debug: texto cru para validar leituras (aparece no console)
    try:
        print("📄 RAW OCR (len={}):".format(len(combined)))
        print(combined[:800])
    except Exception:
        pass

    # Ajuste de confiança
    if (parsed.get("rsi_value") is not None) or (parsed.get("macd_value") is not None) or (parsed.get("adx_value") is not None):
        parsed["confidence"] = max(parsed.get("confidence", 0.0), 0.6)
    if parsed.get("rsi_value") is not None and parsed.get("macd_value") is not None:
        parsed["confidence"] = max(parsed.get("confidence", 0.0), 0.88)

    return parsed

        except Exception as e:
            return {
                "raw_text": f"ERROR: {e}",
                "rsi_value": None,
                "macd_value": None,
                "adx_value": None,
                "confidence": 0.0
            }

class ChartPatternAnalyzer:
    def __init__(self):
        self.min_contour_area = 50
    
    def detect_price_levels(self, price_data: np.ndarray) -> Dict[str, Any]:
        """Detecta suportes e resistências automaticamente"""
        try:
            height, width = price_data.shape
            
            # Encontra linhas horizontais de densidade
            horizontal_profile = np.mean(price_data, axis=1)
            peaks = self._find_peaks(horizontal_profile)
            
            support_levels = []
            resistance_levels = []
            
            if len(peaks) >= 2:
                # Separa em suportes (parte inferior) e resistências (parte superior)
                mid_point = len(peaks) // 2
                support_levels = peaks[:mid_point]
                resistance_levels = peaks[mid_point:]
            
            return {
                'supports': support_levels[:3],  # Top 3 suportes
                'resistances': resistance_levels[:3],  # Top 3 resistências
                'key_levels_count': len(support_levels) + len(resistance_levels),
                'level_strength': self._calculate_level_strength(price_data, support_levels + resistance_levels)
            }
        except Exception:
            return {'supports': [], 'resistances': [], 'key_levels_count': 0, 'level_strength': 0.0}
    
    def _find_peaks(self, signal: np.ndarray, min_distance: int = 5) -> List[int]:
        """Encontra picos no sinal"""
        peaks = []
        for i in range(min_distance, len(signal) - min_distance):
            if (signal[i] > np.max(signal[i-min_distance:i]) and 
                signal[i] > np.max(signal[i+1:i+min_distance+1])):
                peaks.append(i)
        return peaks
    
    def _calculate_level_strength(self, price_data: np.ndarray, levels: List[int]) -> float:
        """Calcula a força dos níveis de preço"""
        if not levels:
            return 0.0
        
        strength_scores = []
        for level in levels:
            # Verifica quantas vezes o preço tocou este nível
            tolerance = price_data.shape[0] * 0.05  # 5% de tolerância
            touches = np.sum(np.abs(np.argmax(price_data, axis=0) - level) < tolerance)
            strength_scores.append(touches)
        
        return float(np.mean(strength_scores) / price_data.shape[1])
    
    def analyze_trend_strength(self, price_data: np.ndarray) -> Dict[str, float]:
        """Analisa força e direção da tendência"""
        try:
            height, width = price_data.shape
            
            # Análise de regressão linear
            x = np.arange(width)
            y = np.argmax(price_data, axis=0)
            
            if len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = self._calculate_r_squared(x, y, slope, intercept)
                
                # Normaliza slope para -1 a 1
                normalized_slope = -slope / (height / 2)  # Invertido porque y=0 é topo
                
                return {
                    'trend_direction': float(normalized_slope),
                    'trend_strength': float(r_squared),
                    'trend_consistency': float(1.0 - np.std(y) / height)
                }
            else:
                return {'trend_direction': 0.0, 'trend_strength': 0.0, 'trend_consistency': 0.0}
                
        except Exception:
            return {'trend_direction': 0.0, 'trend_strength': 0.0, 'trend_consistency': 0.0}
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        """Calcula R² para qualidade do ajuste"""
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

# =========================
#  FASE 3: ANÁLISE DOS SUBGRÁFICOS DE INDICADORES
# =========================
class IndicatorChartAnalyzer:
    def __init__(self):
        self.rsi_overbought = 70
        self.rsi_oversold = 30
    
    def analyze_rsi_position(self, extracted_data: Dict) -> Dict[str, float]:
        """Analisa posição do RSI baseado em dados extraídos e análise visual"""
        try:
            rsi_value = extracted_data.get('rsi_value')
            
            if rsi_value is not None:
                # Usa valor exato do OCR
                normalized_rsi = (rsi_value - 50) / 50  # Normaliza para -1 a 1
                position_strength = abs(normalized_rsi)
                
                if rsi_value > self.rsi_overbought:
                    position = 'overbought'
                elif rsi_value < self.rsi_oversold:
                    position = 'oversold'
                else:
                    position = 'neutral'
            else:
                # Fallback para análise visual
                normalized_rsi = 0.0
                position_strength = 0.5
                position = 'unknown'
            
            return {
                'rsi_normalized': float(normalized_rsi),
                'rsi_strength': float(position_strength),
                'rsi_position': position,
                'confidence': 1.0 if rsi_value is not None else 0.3
            }
        except Exception:
            return {'rsi_normalized': 0.0, 'rsi_strength': 0.5, 'rsi_position': 'unknown', 'confidence': 0.0}
    
    def analyze_macd_signal(self, extracted_data: Dict) -> Dict[str, float]:
        """Analisa sinal do MACD"""
        try:
            macd_value = extracted_data.get('macd_value')
            
            if macd_value is not None:
                # Normaliza MACD (assumindo faixa típica -5 a +5)
                normalized_macd = np.clip(macd_value / 5.0, -1.0, 1.0)
                macd_strength = abs(normalized_macd)
                
                if macd_value > 0:
                    signal = 'bullish'
                else:
                    signal = 'bearish'
            else:
                normalized_macd = 0.0
                macd_strength = 0.5
                signal = 'neutral'
            
            return {
                'macd_normalized': float(normalized_macd),
                'macd_strength': float(macd_strength),
                'macd_signal': signal,
                'confidence': 1.0 if macd_value is not None else 0.3
            }
        except Exception:
            return {'macd_normalized': 0.0, 'macd_strength': 0.5, 'macd_signal': 'neutral', 'confidence': 0.0}

# =========================
#  FASE 4: SISTEMA DE FUSÃO DE DADOS
# =========================
class DataFusionEngine:
    def __init__(self):
        self.weights = {
            'price_action': 0.25,
            'technical_indicators': 0.25,
            'market_structure': 0.25,
            'momentum': 0.25
        }
    
    def fuse_all_data(self, 
                     text_data: Dict, 
                     chart_analysis: Dict, 
                     indicator_analysis: Dict,
                     traditional_analysis: Dict) -> Dict[str, Any]:
        """Combina todas as fontes de dados para decisão final"""
        
        # 🎯 ANÁLISE DE PRICE ACTION (25%)
        price_action_score = self._analyze_price_action(chart_analysis, traditional_analysis)
        
        # 📊 INDICADORES TÉCNICOS (25%)
        technical_score = self._analyze_technical_indicators(text_data, indicator_analysis)
        
        # 🏗️ ESTRUTURA DE MERCADO (25%)
        structure_score = self._analyze_market_structure(chart_analysis)
        
        # 🚀 MOMENTUM (25%)
        momentum_score = self._analyze_momentum(indicator_analysis, traditional_analysis)
        
        # 🧮 SCORE FINAL PERFEITAMENTE EQUILIBRADO
        total_score = (
            price_action_score * self.weights['price_action'] +
            technical_score * self.weights['technical_indicators'] +
            structure_score * self.weights['market_structure'] +
            momentum_score * self.weights['momentum']
        )
        
        return {
            'total_score': float(total_score),
            'component_scores': {
                'price_action': float(price_action_score),
                'technical_indicators': float(technical_score),
                'market_structure': float(structure_score),
                'momentum': float(momentum_score)
            },
            'weighted_analysis': self._get_weighted_analysis(
                price_action_score, technical_score, structure_score, momentum_score
            )
        }
    
    def _analyze_price_action(self, chart_analysis: Dict, traditional: Dict) -> float:
        """Analisa price action combinando dados visuais e tradicionais"""
        # ✅ Corrige: extrai da subchave 'trend_analysis'
        trend_dict = chart_analysis.get('trend_analysis', {})
        trend_direction = trend_dict.get('trend_direction', 0.0)
        trend_strength = trend_dict.get('trend_strength', 0.0)
        traditional_trend = traditional.get('price_action', {}).get('trend_direction', 0.0)

        # Combina análise visual e tradicional
        combined_trend = (trend_direction + traditional_trend) / 2.0
        strength_factor = trend_strength

        return combined_trend * strength_factor

    
    def _analyze_technical_indicators(self, text_data: Dict, indicator_analysis: Dict) -> float:
        """Analisa indicadores técnicos com dados OCR"""
        rsi_analysis = indicator_analysis.get('rsi', {})
        macd_analysis = indicator_analysis.get('macd', {})
        
        rsi_score = rsi_analysis.get('rsi_normalized', 0.0)
        macd_score = macd_analysis.get('macd_normalized', 0.0)
        
        # Confiança baseada na qualidade dos dados OCR
        rsi_confidence = rsi_analysis.get('confidence', 0.5)
        macd_confidence = macd_analysis.get('confidence', 0.5)
        
        # Score ponderado pela confiança
        if rsi_confidence > 0.7 or macd_confidence > 0.7:
            # Dados confiáveis disponíveis
            if rsi_confidence > macd_confidence:
                return rsi_score
            else:
                return macd_score
        else:
            # Fallback: usar valores do OCR se existirem
            rsi_ocr = text_data.get('rsi_value')
            macd_ocr = text_data.get('macd_value')
            if rsi_ocr is not None:
                return float(np.clip((rsi_ocr - 50) / 50.0, -1.0, 1.0))
            if macd_ocr is not None:
                return float(np.clip(macd_ocr / 5.0, -1.0, 1.0))
            return 0.0

    
    def _analyze_market_structure(self, chart_analysis: Dict) -> float:
        """Analisa estrutura de mercado (suportes/resistências)"""
        levels_data = chart_analysis.get('price_levels', {})
        supports = levels_data.get('supports', [])
        resistances = levels_data.get('resistances', [])
        level_strength = levels_data.get('level_strength', 0.0)

        if level_strength <= 0:
            return 0.0

        # Heurística estável: sinal leva em conta quantidade de níveis
        score = 0.0
        if resistances and (not supports or len(resistances) >= len(supports)):
            score -= 0.3 * level_strength
        if supports and (not resistances or len(supports) > len(resistances)):
            score += 0.3 * level_strength

        return float(np.clip(score, -1.0, 1.0))

    def _analyze_momentum(self, indicator_analysis: Dict, traditional: Dict) -> float:
        """Analisa momentum combinado"""
        rsi_strength = indicator_analysis.get('rsi', {}).get('rsi_strength', 0.5)
        macd_strength = indicator_analysis.get('macd', {}).get('macd_strength', 0.5)
        traditional_momentum = traditional.get('price_action', {}).get('momentum', 0.0)
        
        momentum_components = [
            rsi_strength * traditional_momentum,
            macd_strength * traditional_momentum
        ]
        
        valid_components = [c for c in momentum_components if abs(c) > 0.1]
        if valid_components:
            return np.mean(valid_components)
        else:
            return traditional_momentum
    
    def _get_weighted_analysis(self, price_action: float, technical: float, 
                              structure: float, momentum: float) -> str:
        """Gera análise baseada nos componentes ponderados"""
        components = [
            (abs(price_action), "Price Action"),
            (abs(technical), "Indicadores Técnicos"),
            (abs(structure), "Estrutura de Mercado"),
            (abs(momentum), "Momentum")
        ]
        
        # Ordena por força do sinal
        components.sort(reverse=True)
        top_components = [name for score, name in components[:2] if score > 0.1]
        
        if top_components:
            return " + ".join(top_components)
        else:
            return "Análise Balanceada"

# =========================
#  IA SUPER INTELIGENTE E NEUTRA - ATUALIZADA
# =========================
class SuperIntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
        self.text_extractor = TextDataExtractor()
        self.chart_analyzer = ChartPatternAnalyzer()
        self.indicator_analyzer = IndicatorChartAnalyzer()
        self.data_fusion = DataFusionEngine()
        
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

    # =========================
    #  ANÁLISE MULTI-CAMADAS ATUALIZADA
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
    #  MOTOR DE DECISÃO 100% NEUTRO - ATUALIZADO
    # =========================
    
    def _absolute_decision_engine(self, all_analyses: Dict, timeframe: str) -> Dict[str, Any]:
        """MOTOR 100% NEUTRO ATUALIZADO - COM FUSÃO DE DADOS"""
        try:
            # 🎯 FUSÃO DE TODOS OS DADOS
            fusion_result = self.data_fusion.fuse_all_data(
                all_analyses['text_data'],
                all_analyses['chart_analysis'],
                all_analyses['indicator_analysis'],
                all_analyses['traditional']
            )
            
            total_score = fusion_result['total_score']
            component_scores = fusion_result['component_scores']
            
            # 💥 DECISÃO 100% NEUTRA - APENAS PELOS DADOS
            if total_score > 0:
                direction = "buy"
                base_confidence = 0.65 + (min(abs(total_score), 0.5) * 0.35)
                reasoning = self._generate_enhanced_reasoning("buy", component_scores, fusion_result['weighted_analysis'])
            else:
                direction = "sell"
                base_confidence = 0.65 + (min(abs(total_score), 0.5) * 0.35)
                reasoning = self._generate_enhanced_reasoning("sell", component_scores, fusion_result['weighted_analysis'])
            
            # 🎪 CONFIANÇA NEUTRA MELHORADA
            final_confidence = self._calculate_enhanced_confidence(base_confidence, all_analyses, fusion_result)
            
            # 🎯 CONTEXTO NEUTRO ATUALIZADO
            context = self._detect_enhanced_context(component_scores, total_score, all_analyses)
            
            return {
                "direction": direction,
                "confidence": final_confidence,
                "reasoning": reasoning,
                "total_score": total_score,
                "context": context,
                "component_scores": component_scores,
                "fusion_analysis": fusion_result['weighted_analysis'],
                "trend_power": component_scores['price_action'],
                "macd_power": component_scores['technical_indicators'],
                "micro_power": component_scores['momentum']
            }
            
        except Exception as e:
            # EM CASO DE ERRO: DECISÃO NEUTRA BASEADA EM HORÁRIO DE MERCADO
            return self._neutral_market_decision()

    def _generate_enhanced_reasoning(self, direction: str, component_scores: Dict, weighted_analysis: str) -> str:
        """Gera reasoning melhorado baseado na fusão de dados"""
        
        if direction == "buy":
            strength = "FORTE" if abs(component_scores['price_action']) > 0.2 else "moderada"
            
            # Analisa componentes mais fortes
            strong_components = []
            for comp_name, score in component_scores.items():
                if abs(score) > 0.15:
                    comp_label = {
                        'price_action': 'Tendência',
                        'technical_indicators': 'Indicadores',
                        'market_structure': 'Estrutura',
                        'momentum': 'Momentum'
                    }.get(comp_name, comp_name)
                    strong_components.append(f"{comp_label} {score*100:+.1f}%")
            
            if strong_components:
                analysis = " + ".join(strong_components)
                return f"📈 COMPRA {strength} - {weighted_analysis}: {analysis}"
            else:
                return f"📈 COMPRA {strength} - Convergência técnica positiva"
        
        else:  # sell
            strength = "FORTE" if abs(component_scores['price_action']) > 0.2 else "moderada"
            
            strong_components = []
            for comp_name, score in component_scores.items():
                if abs(score) > 0.15:
                    comp_label = {
                        'price_action': 'Tendência',
                        'technical_indicators': 'Indicadores', 
                        'market_structure': 'Estrutura',
                        'momentum': 'Momentum'
                    }.get(comp_name, comp_name)
                    strong_components.append(f"{comp_label} {score*100:+.1f}%")
            
            if strong_components:
                analysis = " + ".join(strong_components)
                return f"📉 VENDA {strength} - {weighted_analysis}: {analysis}"
            else:
                return f"📉 VENDA {strength} - Convergência técnica negativa"

    def _calculate_enhanced_confidence(self, base_confidence: float, all_analyses: Dict, fusion_result: Dict) -> float:
        """Calcula confiança melhorada considerando todas as fontes"""
        try:
            # Fatores de confiança de todas as análises
            confidence_factors = [
                all_analyses['nano_analysis']['convergence_strength'],
                all_analyses['micro_structure']['structural_integrity'],
                all_analyses['flow_dynamics']['overall_flow_quality'],
                all_analyses['traditional']['price_action']['trend_strength'],
                all_analyses['traditional']['indicators']['macd_strength'],
                all_analyses['text_data'].get('confidence', 0.5),  # Confiança do OCR
                np.mean([abs(score) for score in fusion_result['component_scores'].values()])  # Força dos sinais
            ]
            
            quality_score = np.mean([f for f in confidence_factors if not np.isnan(f)])
            enhanced_confidence = base_confidence + (quality_score * 0.25)
            
            return min(0.92, enhanced_confidence)
            
        except Exception:
            return base_confidence

    def _detect_enhanced_context(self, component_scores: Dict, total_score: float, all_analyses: Dict) -> str:
        """Detecta contexto de mercado melhorado"""
        text_data = all_analyses['text_data']
        
        # Verifica se temos dados OCR confiáveis
        if text_data.get('confidence', 0) > 0.7:
            if text_data.get('rsi_value'):
                rsi = text_data['rsi_value']
                if rsi > 70:
                    return "rsi_sobrecomprado"
                elif rsi < 30:
                    return "rsi_sobrevendido"
        
        # Contexto baseado em scores
        if abs(total_score) > 0.3:
            return "movimento_forte"
        elif abs(total_score) < 0.1:
            return "mercado_lateral"
        elif component_scores['price_action'] > 0.2:
            return "tendencia_de_alta"
        elif component_scores['price_action'] < -0.2:
            return "tendencia_de_baixa"
        elif component_scores['technical_indicators'] > 0.2:
            return "indicadores_favoraveis"
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
                    "component_scores": {"price_action": 0.08, "technical_indicators": 0.08, "market_structure": 0.08, "momentum": 0.08},
                    "fusion_analysis": "Horário de Mercado",
                    "trend_power": 0.08,
                    "macd_power": 0.08,
                    "micro_power": 0.08
                }
            else:
                return {
                    "direction": "sell", 
                    "confidence": 0.62,
                    "reasoning": "📉 VENDA - Análise de mercado: horário de baixa liquidez",
                    "total_score": -0.10,
                    "context": "after_hours",
                    "component_scores": {"price_action": -0.08, "technical_indicators": -0.08, "market_structure": -0.08, "momentum": -0.08},
                    "fusion_analysis": "Horário de Mercado",
                    "trend_power": -0.08,
                    "macd_power": -0.08,
                    "micro_power": -0.08
                }
        except Exception:
            return {
                "direction": "sell",
                "confidence": 0.60,
                "reasoning": "📉 VENDA - Princípio neutro: cautela em análise indeterminada",
                "total_score": -0.05,
                "context": "neutral_caution",
                "component_scores": {"price_action": 0.0, "technical_indicators": 0.0, "market_structure": 0.0, "momentum": 0.0},
                "fusion_analysis": "Análise Conservadora",
                "trend_power": 0.0,
                "macd_power": 0.0,
                "micro_power": 0.0
            }

    def _calculate_signal_quality(self, analyses: Dict) -> float:
        """Calcula qualidade do sinal considerando todas as fontes"""
        try:
            factors = [
                analyses['nano_analysis']['convergence_strength'] * 0.15,
                analyses['micro_structure']['structural_integrity'] * 0.15,
                analyses['flow_dynamics']['overall_flow_quality'] * 0.15,
                analyses['traditional']['price_action']['trend_strength'] * 0.15,
                analyses['traditional']['indicators']['macd_strength'] * 0.15,
                analyses['text_data'].get('confidence', 0.5) * 0.15,  # Confiança do OCR
                np.mean([abs(s) for s in analyses['decision']['component_scores'].values()]) * 0.10  # Força dos sinais
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

    def analyze(self, blob: bytes, timeframe: str = '1m') -> Dict[str, Any]:
        """ANÁLISE 100% NEUTRA ATUALIZADA - COM 4 FASES DE ANÁLISE"""
        
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
            price_data = self._extract_price_data(img_array)
            
            # 🎯 FASE 1: EXTRAÇÃO DE TEXTO (OCR)
            text_data = self.text_extractor.extract_text_data(image)
            
            # 🎯 FASE 2: ANÁLISE VISUAL DO GRÁFICO
            chart_analysis = {
                'price_levels': self.chart_analyzer.detect_price_levels(price_data),
                'trend_analysis': self.chart_analyzer.analyze_trend_strength(price_data)
            }
            
            # 🎯 FASE 3: ANÁLISE DE INDICADORES
            indicator_analysis = {
                'rsi': self.indicator_analyzer.analyze_rsi_position(text_data),
                'macd': self.indicator_analyzer.analyze_macd_signal(text_data)
            }
            
            

            # 🔥 Integra valores OCR diretamente e reforça confiança
            if text_data.get('rsi_value') is not None:
                indicator_analysis['rsi']['rsi_value_ocr'] = float(text_data['rsi_value'])
            if text_data.get('macd_value') is not None:
                indicator_analysis['macd']['macd_value_ocr'] = float(text_data['macd_value'])
            if text_data.get('confidence', 0) > 0.6:
                indicator_analysis['rsi']['confidence'] = max(indicator_analysis['rsi'].get('confidence', 0), text_data['confidence'])
                indicator_analysis['macd']['confidence'] = max(indicator_analysis['macd'].get('confidence', 0), text_data['confidence'])
# 🧠 ANÁLISE MULTI-CAMADAS ORIGINAL
            traditional_analyses = {
                'traditional': {
                    'price_action': self._analyze_price_action(price_data, timeframe),
                    'indicators': self._calculate_advanced_indicators(price_data)
                },
                'nano_analysis': self._microscopic_trend_analysis(price_data),
                'micro_structure': self._analyze_micro_structure(price_data),
                'flow_dynamics': self._analyze_flow_dynamics(price_data)
            }
            
            # 🎯 FASE 4: FUSÃO DE TODOS OS DADOS
            all_analyses = {
                'text_data': text_data,
                'chart_analysis': chart_analysis,
                'indicator_analysis': indicator_analysis,
                **traditional_analyses
            }
            
            # 🎯 MOTOR DE DECISÃO 100% NEUTRO ATUALIZADO
            decision = self._absolute_decision_engine(all_analyses, timeframe)
            time_info = self._get_entry_timeframe(timeframe)
            
            # 📊 QUALIDADE DA ANÁLISE MELHORADA
            signal_quality = self._calculate_signal_quality(all_analyses)
            
            # 🎨 RESULTADO SUPER NEUTRO ATUALIZADO
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
                "micro_quality": all_analyses['nano_analysis']['convergence_strength'],
                "text_data_confidence": text_data.get('confidence', 0.0),
                "fusion_analysis": decision.get('fusion_analysis', 'Análise Integrada'),
                "metrics": {
                    "analysis_score": float(decision["total_score"]),
                    "trend_power": float(decision["trend_power"]),
                    "macd_power": float(decision["macd_power"]),
                    "micro_power": float(decision["micro_power"]),
                    "trend_strength": all_analyses['traditional']['price_action']['trend_strength'],
                    "momentum": all_analyses['traditional']['price_action']['momentum'],
                    "rsi": all_analyses['traditional']['indicators']['rsi'],
                    "macd": all_analyses['traditional']['indicators']['macd'],
                    "macd_strength": all_analyses['traditional']['indicators']['macd_strength'],
                    "component_scores": decision.get('component_scores', {})
                },
                "reasoning": decision["reasoning"],
                "text_data": {
                    "rsi_value": text_data.get('rsi_value'),
                    "macd_value": text_data.get('macd_value'),
                    "ema_value": text_data.get('ema_value'),
                    "has_technical_data": text_data.get('confidence', 0) > 0.5
                }
            }
            
            
            # 🔋 Exporta poderes (0–100%) para a UI
            comp = decision.get("component_scores", {})
            def _to_pct(x):
                try:
                    return float(round(abs(x) * 100.0, 1))
                except Exception:
                    return 0.0
            result.update({
                "powers": {
                    "trend_power_pct": _to_pct(comp.get("price_action", 0.0)),
                    "indicators_power_pct": _to_pct(comp.get("technical_indicators", 0.0)),
                    "structure_power_pct": _to_pct(comp.get("market_structure", 0.0)),
                    "momentum_power_pct": _to_pct(comp.get("momentum", 0.0))
                }
            })

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
                "text_data_confidence": 0.0,
                "fusion_analysis": "Análise de Contingência"
            })
            return fallback_result

# =========================
#  APLICAÇÃO FLASK COMPLETA (MANTIDA)
# =========================
app = Flask(__name__)
analyzer = SuperIntelligentAnalyzer()

# Configurações para produção
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False

# HTML TEMPLATE (MANTIDO EXATAMENTE IGUAL)
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
        .context-rsi_sobrecomprado { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
        .context-rsi_sobrevendido { background: linear-gradient(135deg, #00ff88, #00cc66); color: white; }
        .context-tendencia_de_alta { background: linear-gradient(135deg, #00ff88, #00cc66); color: white; }
        .context-tendencia_de_baixa { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
        .context-indicadores_favoraveis { background: linear-gradient(135deg, #7ce0ff, #4a90e2); color: white; }
        
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
        
        .technical-data {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠⚖️ IA SIGNAL PRO - 100% NEUTRA</div>
            <div class="subtitle">ZERO VIÉS - DECISÕES APENAS PELO MOMENTO DO MERCADO</div>
            <div class="subtitle" style="color: #7ce0ff; font-size: 11px;">🎯 SISTEMA ATUALIZADO: OCR + ANÁLISE VISUAL + FUSÃO DE DADOS</div>
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
            
            <div id="technicalData" class="technical-data" style="display: none;">
                <div style="text-align: center; font-weight: 600; margin-bottom: 5px; color: #00ff88;">
                    📊 DADOS TÉCNICOS EXTRAÍDOS
                </div>
                <div id="technicalDataContent"></div>
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
                    ⚡ ANÁLISE DO MOMENTO - FUSÃO DE DADOS
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
            const technicalData = document.getElementById('technicalData');
            const technicalDataContent = document.getElementById('technicalDataContent');
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
                technicalData.style.display = 'none';
                
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
                reasoningText.textContent = 'Processando análise 100% neutra com OCR...';
                confidenceText.textContent = '';
                progressFill.style.width = '10%';
                
                metricsText.innerHTML = '<div class="loading">Iniciando análise multi-camadas...</div>';

                try {
                    const formData = new FormData();
                    formData.append('image', selectedFile);
                    formData.append('timeframe', currentTimeframe);
                    
                    progressFill.style.width = '30%';
                    
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    progressFill.style.width = '70%';
                    
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
                const fusionAnalysis = data.fusion_analysis || 'Análise Integrada';
                
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
                
                // Mostra dados técnicos se disponíveis
                if (data.text_data && data.text_data.has_technical_data) {
                    let technicalHtml = '';
                    const textData = data.text_data;
                    
                    if (textData.rsi_value) {
                        technicalHtml += `<div class="metric-item"><span>RSI:</span><span class="metric-value">${textData.rsi_value}</span></div>`;
                    }
                    if (textData.macd_value) {
                        technicalHtml += `<div class="metric-item"><span>MACD:</span><span class="metric-value">${textData.macd_value}</span></div>`;
                    }
                    if (textData.ema_value) {
                        technicalHtml += `<div class="metric-item"><span>EMA:</span><span class="metric-value">${textData.ema_value}</span></div>`;
                    }
                    
                    if (technicalHtml) {
                        technicalDataContent.innerHTML = technicalHtml;
                        technicalData.style.display = 'block';
                    }
                }
                
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
                    'mercado_balanceado': '⚖️ MERCADO BALANCEADO',
                    'rsi_sobrecomprado': '📛 RSI SOBRECOMPRADO',
                    'rsi_sobrevendido': '✅ RSI SOBREVENDIDO',
                    'tendencia_de_alta': '📈 TENDÊNCIA DE ALTA',
                    'tendencia_de_baixa': '📉 TENDÊNCIA DE BAIXA',
                    'indicadores_favoraveis': '🎯 INDICADORES FAVORÁVEIS'
                };
                
                contextInfo.innerHTML = `
                    <span class="context-badge context-${context}">
                        ${contextLabels[context] || contextLabels.mercado_balanceado}
                    </span>
                    <div style="margin-top: 5px; font-size: 11px; color: #7ce0ff;">
                        ${fusionAnalysis}
                    </div>
                `;
                
                // Análise do Momento
                const metrics = data.metrics || {};
                const componentScores = metrics.component_scores || {};
                let powerHtml = '';
                
                const powerItems = [
                    ['Poder da Tendência', (componentScores.price_action * 100)?.toFixed(1) + '%'],
                    ['Poder dos Indicadores', (componentScores.technical_indicators * 100)?.toFixed(1) + '%'],
                    ['Poder da Estrutura', (componentScores.market_structure * 100)?.toFixed(1) + '%'],
                    ['Poder do Momentum', (componentScores.momentum * 100)?.toFixed(1) + '%'],
                    ['Score Final', metrics.analysis_score?.toFixed(3)]
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
                    ['Qualidade do Sinal', (data.signal_quality * 100)?.toFixed(1) + '%'],
                    ['Confiança do OCR', (data.text_data_confidence * 100)?.toFixed(1) + '%']
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
        
        # Análise 100% NEUTRA ATUALIZADA
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
        'service': 'IA Signal Pro - 100% NEUTRA - SISTEMA ATUALIZADO',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '7.0.0-fusao-dados',
        'features': ['OCR', 'Análise Visual', 'Fusão de Dados', '4 Fases']
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
    
    print(f"🚀 IA Signal Pro - 100% NEUTRA ATUALIZADA iniciando na porta {port}")
    print(f"🧠⚖️ SISTEMA: ZERO VIÉS - 4 FASES DE ANÁLISE")
    print(f"🎯 PRINCÍPIO: OCR + ANÁLISE VISUAL + FUSÃO DE DADOS")
    print(f"📈 SAÍDA: COMPRA ou VENDA - SEM FAVORITISMO")
    print(f"💪 NEUTRALIDADE: PONDERAÇÃO IGUAL + ANÁLISE DO MOMENTO")
    print(f"🔍 RECURSOS: Detecção de RSI/MACD + Suportes/Resistências + Tendências")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
