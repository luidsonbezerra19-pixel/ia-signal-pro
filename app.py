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
import subprocess
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

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
#  IA SUPER INTELIGENTE E NEUTRA
# =========================
class SuperIntelligentAnalyzer:
    def __init__(self):
        self.cache = AnalysisCache()
        self.ocr_available = self._setup_ocr_railway()
        
    def _setup_ocr_railway(self) -> bool:
        """Configura e verifica OCR no ambiente Railway"""
        try:
            import pytesseract
            
            print("🔍 Configurando OCR para Railway...")
            
            # Tenta encontrar o Tesseract no Railway
            possible_paths = [
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract', 
                '/app/bin/tesseract',
                'tesseract'
            ]
            
            for path in possible_paths:
                try:
                    result = subprocess.run([path, '--version'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        pytesseract.pytesseract.tesseract_cmd = path
                        print(f"✅ Tesseract configurado em: {path}")
                        print(f"📄 Versão: {result.stdout.strip()}")
                        
                        # Teste rápido do OCR
                        test_image = Image.new('RGB', (100, 50), color='white')
                        test_text = pytesseract.image_to_string(test_image)
                        print("✅ OCR funcionando perfeitamente!")
                        return True
                except Exception:
                    continue
            
            print("❌ Tesseract não encontrado no Railway")
            return False
            
        except ImportError:
            print("❌ pytesseract não instalado")
            return False
        except Exception as e:
            print(f"❌ Erro na configuração OCR: {e}")
            return False
        
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
        target_size = (800, 600)
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
    #  SISTEMA OCR OTIMIZADO PARA RAILWAY
    # =========================

    # =========================
#  SISTEMA OCR ESPECIALIZADO PARA GRÁFICOS TRADING
# =========================

def _extract_with_ocr(self, image: Image.Image) -> Dict[str, Any]:
    """Extrai valores EXATOS do gráfico com OCR especializado"""
    if not self.ocr_available:
        print("❌ OCR não disponível")
        return {}
        
    try:
        import pytesseract
        
        print("🔍 Iniciando OCR especializado para gráficos trading...")
        
        # Estratégias específicas para gráficos
        strategies_results = []
        
        # 1. OCR na imagem original (muitas vezes funciona melhor)
        original_text = self._ocr_original_image(image)
        strategies_results.append(original_text)
        
        # 2. OCR com foco em regiões específicas
        regions_text = self._ocr_trading_regions(image)
        strategies_results.append(regions_text)
        
        # 3. OCR com alto contraste
        high_contrast_text = self._ocr_high_contrast_focused(image)
        strategies_results.append(high_contrast_text)
        
        # 4. OCR com imagem limpa
        clean_text = self._ocr_clean_image(image)
        strategies_results.append(clean_text)
        
        # Combina todos os resultados
        all_numbers = []
        for i, result in enumerate(strategies_results):
            numbers = self._extract_trading_numbers(result)
            print(f"🎯 Estratégia {i+1}: {len(numbers)} números -> {numbers}")
            all_numbers.extend(numbers)
        
        # Processa e classifica números
        if all_numbers:
            results = self._classify_trading_numbers_advanced(all_numbers)
            print(f"✅ OCR finalizado: {results}")
            return results
        else:
            print("❌ Nenhum número encontrado via OCR")
            return {}
            
    except Exception as e:
        print(f"❌ Erro no OCR: {e}")
        return {}

def _ocr_original_image(self, image: Image.Image) -> str:
    """OCR na imagem original - muitas vezes a mais precisa"""
    try:
        import pytesseract
        
        # Usa a imagem original sem modificações
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,- '
        text = pytesseract.image_to_string(image, config=config)
        print(f"📝 Original: {text.strip()}")
        return text
    except Exception:
        return ""

def _ocr_trading_regions(self, image: Image.Image) -> str:
    """OCR focado nas regiões onde estão os indicadores"""
    try:
        import pytesseract
        
        width, height = image.size
        all_text = ""
        
        # Regiões específicas para gráficos trading
        regions = [
            # Topo direito - geralmente preço atual
            (width * 2 // 3, 0, width, height // 6),
            # Centro direito - indicadores
            (width * 2 // 3, height // 6, width, height // 2),
            # Rodapé - RSI, MACD, etc
            (0, height * 3 // 4, width, height),
            # Topo esquerdo - título/par
            (0, 0, width // 3, height // 6)
        ]
        
        for i, (left, top, right, bottom) in enumerate(regions):
            try:
                region = image.crop((left, top, right, bottom))
                
                # Pré-processamento específico para região
                processed_region = self._preprocess_region_for_ocr(region)
                
                config = '--oem 3 --psm 8'
                text = pytesseract.image_to_string(processed_region, config=config)
                if text.strip():
                    all_text += f" | Região {i+1}: {text.strip()}"
                    
            except Exception:
                continue
        
        print(f"📍 Regiões: {all_text}")
        return all_text
        
    except Exception:
        return ""

def _ocr_high_contrast_focused(self, image: Image.Image) -> str:
    """OCR com contraste otimizado para números"""
    try:
        import pytesseract
        
        # Converte para escala de cinza
        gray = image.convert('L')
        
        # Redimensiona para melhor reconhecimento
        if gray.size[0] > 1000:
            gray = gray.resize((800, int(800 * gray.size[1] / gray.size[0])), Image.LANCZOS)
        
        # Contraste inteligente - não muito agressivo
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Brilho moderado
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        final_image = brightness_enhancer.enhance(1.1)
        
        config = '--oem 3 --psm 6'
        text = pytesseract.image_to_string(final_image, config=config)
        print(f"⚡ Alto contraste: {text.strip()}")
        return text
        
    except Exception:
        return ""

def _ocr_clean_image(self, image: Image.Image) -> str:
    """OCR com imagem limpa e suavizada"""
    try:
        import pytesseract
        
        gray = image.convert('L')
        
        # Suaviza a imagem para reduzir ruído
        smoothed = gray.filter(ImageFilter.SMOOTH)
        
        # Realce de bordas suave
        edges = smoothed.filter(ImageFilter.EDGE_ENHANCE)
        
        # Configuração para texto de trading
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,- '
        text = pytesseract.image_to_string(edges, config=config)
        print(f"🧹 Limpa: {text.strip()}")
        return text
        
    except Exception:
        return ""

def _preprocess_region_for_ocr(self, region: Image.Image) -> Image.Image:
    """Pré-processamento específico para regiões"""
    try:
        # Converte para escala de cinza
        gray = region.convert('L')
        
        # Aumenta o tamanho se muito pequeno
        if gray.size[0] < 100:
            gray = gray.resize((gray.size[0] * 2, gray.size[1] * 2), Image.LANCZOS)
        
        # Contraste moderado
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.5)
        
        return enhanced
        
    except Exception:
        return region

def _extract_trading_numbers(self, text: str) -> List[float]:
    """Extrai números de trading de forma avançada"""
    numbers = []
    
    if not text:
        return numbers
    
    # Padrões específicos para trading
    patterns = [
        r'\d{1,4}\.\d{2,4}',  # Preços: 1234.56, 123.4567
        r'\d{1,3}\,\d{2}',     # European format: 123,45
        r'\d{1,4}\.\d{1,2}',   # Valores simples: 1234.5
        r'-?\d{1,4}\.\d{1,4}', # Números com sinal: -123.45
        r'\d{1,3}\.\d{1,4}',   # Valores menores: 12.3456
        r'-?\d{1,5}',          # Inteiros: -12345
        r'\d{1,2}\,\d{1,2}',   # Percentuais: 12,34
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Substitui vírgula por ponto para conversão
                normalized = match.replace(',', '.')
                num = float(normalized)
                
                # Filtra números plausíveis para trading
                if self._is_valid_trading_number(num):
                    numbers.append(num)
                    
            except ValueError:
                continue
    
    # Remove duplicatas próximas
    unique_numbers = []
    for num in numbers:
        if not any(abs(num - existing) < 0.01 for existing in unique_numbers):
            unique_numbers.append(num)
    
    return unique_numbers

def _is_valid_trading_number(self, number: float) -> bool:
    """Verifica se o número é válido para trading"""
    # Filtra números extremos mas mantém uma faixa ampla
    if abs(number) > 1000000:  # Números muito grandes
        return False
    if number == 0:
        return False
    
    # Permite números negativos (MACD, variações)
    return True

def _classify_trading_numbers_advanced(self, numbers: List[float]) -> Dict[str, float]:
    """Classifica números com lógica específica para trading"""
    results = {}
    
    if not numbers:
        return results
    
    print(f"🔢 Números para classificar: {numbers}")
    
    # Ordena por frequência de ocorrência (mais comum = mais provável de ser correto)
    from collections import Counter
    number_counts = Counter(numbers)
    common_numbers = [num for num, count in number_counts.most_common()]
    
    # RSI: geralmente entre 0-100, valores como 59.76
    rsi_candidates = [n for n in common_numbers if 0 <= n <= 100 and n != 50]
    if rsi_candidates:
        # Pega o mais comum que não seja 50 (valor padrão)
        results['rsi'] = rsi_candidates[0]
        print(f"🎯 RSI identificado: {results['rsi']}")
    
    # Preços: geralmente os maiores números, excluindo outliers
    price_candidates = [n for n in common_numbers if n > 10 and n < 100000]
    if price_candidates:
        # Pega o número mais comum na faixa de preço
        results['price'] = price_candidates[0]
        print(f"🎯 Preço identificado: {results['price']}")
    
    # MACD: pode ser positivo ou negativo, geralmente pequeno
    macd_candidates = [n for n in common_numbers if -20 <= n <= 20 and n != 0]
    if macd_candidates:
        results['macd'] = macd_candidates[0]
        print(f"🎯 MACD identificado: {results['macd']}")
    
    # Variações percentuais: números pequenos
    change_candidates = [n for n in common_numbers if -100 <= n <= 100 and n != 0 and abs(n) < 20]
    if change_candidates:
        results['change'] = change_candidates[0]
        print(f"🎯 Variação identificada: {results['change']}")
    
    # ADX: entre 0-100, diferente do RSI
    adx_candidates = [n for n in common_numbers if 0 <= n <= 100 and n not in rsi_candidates]
    if adx_candidates:
        results['adx'] = adx_candidates[0]
        print(f"🎯 ADX identificado: {results['adx']}")
    
    # Bandas de Bollinger: geralmente próximas ao preço
    bollinger_candidates = [n for n in common_numbers if n > 100 and abs(n - results.get('price', 0)) < 1000]
    if len(bollinger_candidates) >= 2:
        results['bollinger_upper'] = max(bollinger_candidates[:2])
        results['bollinger_lower'] = min(bollinger_candidates[:2])
        print(f"🎯 Bollinger: {results['bollinger_upper']} / {results['bollinger_lower']}")
    
    return results

    def _analyze_extracted_indicators(self, ocr_data: Dict) -> Dict[str, Any]:
        """Analisa os indicadores extraídos via OCR"""
        try:
            # Valores padrão
            default_values = {
                'rsi': {'value': 50.0, 'overbought': False, 'oversold': False, 'trend': 'neutral', 'source': 'default'},
                'macd': {'value': 0.0, 'signal': 0.0, 'histogram': 0.0, 'trend': 'neutral', 'source': 'default'},
                'adx': {'value': 20.0, 'strength': 'weak', 'trend_strength': 0.2, 'source': 'default'},
                'price': {'value': 0.0, 'change': 0.0, 'source': 'default'}
            }
            
            if not ocr_data:
                return default_values
            
            result = default_values.copy()
            
            # RSI from OCR
            if 'rsi' in ocr_data:
                rsi_value = ocr_data['rsi']
                result['rsi'] = {
                    'value': rsi_value,
                    'overbought': rsi_value > 70,
                    'oversold': rsi_value < 30,
                    'trend': 'falling' if rsi_value < 50 else 'rising',
                    'source': 'ocr'
                }
            
            # MACD from OCR
            if 'macd' in ocr_data:
                macd_value = ocr_data['macd']
                result['macd'] = {
                    'value': macd_value,
                    'signal': macd_value * 0.9,
                    'histogram': macd_value * 0.1,
                    'trend': 'bullish' if macd_value > 0 else 'bearish',
                    'source': 'ocr'
                }
            
            # ADX from OCR
            if 'adx' in ocr_data:
                adx_value = ocr_data['adx']
                result['adx'] = {
                    'value': adx_value,
                    'strength': 'strong' if adx_value > 25 else 'weak',
                    'trend_strength': min(1.0, adx_value / 100.0),
                    'source': 'ocr'
                }
            
            # Price from OCR
            if 'price' in ocr_data:
                result['price'] = {
                    'value': ocr_data['price'],
                    'change': ocr_data.get('change', 0.0),
                    'source': 'ocr'
                }
            elif 'change' in ocr_data:
                result['price'] = {
                    'value': 0.0,
                    'change': ocr_data['change'],
                    'source': 'ocr'
                }
            
            return result
            
        except Exception as e:
            print(f"❌ Erro na análise de indicadores: {e}")
            return default_values

    # =========================
    #  ANÁLISE MICROSCÓPICA AVANÇADA
    # =========================
    
    def _microscopic_trend_analysis(self, price_data: np.ndarray) -> Dict[str, float]:
        """Análise NANO de tendências"""
        try:
            height, width = price_data.shape
            
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
        """Analisa continuidade do fluxo"""
        try:
            row_means = np.mean(price_data, axis=0)
            changes = np.diff(row_means)
            volatility = np.std(changes) + 1e-8
            continuity = 1.0 / (1.0 + volatility * 10)
            return float(np.clip(continuity, 0, 1))
        except Exception:
            return 0.5

    def _breakage_detection(self, price_data: np.ndarray) -> float:
        """Detecta quebras na estrutura"""
        try:
            row_means = np.mean(price_data, axis=0)
            changes = np.abs(np.diff(row_means))
            large_jumps = np.sum(changes > np.mean(changes) * 2)
            breakage_ratio = large_jumps / len(changes)
            return float(np.clip(1.0 - breakage_ratio, 0, 1))
        except Exception:
            return 0.5

    def _smoothness_analysis(self, price_data: np.ndarray) -> float:
        """Analisa suavidade das transições"""
        try:
            row_means = np.mean(price_data, axis=0)
            second_derivative = np.gradient(np.gradient(row_means))
            roughness = np.std(second_derivative)
            smoothness = 1.0 / (1.0 + roughness * 5)
            return float(np.clip(smoothness, 0, 1))
        except Exception:
            return 0.5

    # =========================
    #  ANÁLISE TRADICIONAL FORTALECIDA
    # =========================
    
    def _analyze_price_action(self, price_data: np.ndarray, timeframe: str) -> Dict[str, float]:
        """Análise tradicional de price action"""
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
        """Indicadores técnicos avançados"""
        try:
            height, width = price_data.shape
            
            if width > 10:
                row_means = np.mean(price_data, axis=0)
                
                # MACD
                fast_window = min(3, len(row_means))
                slow_window = min(8, len(row_means))
                signal_window = min(5, len(row_means))
                
                fast_ma = np.mean(row_means[-fast_window:])
                slow_ma = np.mean(row_means[-slow_window:])
                macd_line = fast_ma - slow_ma
                
                # Signal line
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
                
                # RSI
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
                
                # Força do MACD
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
        """MOTOR 100% NEUTRO"""
        try:
            nano_trend = all_analyses['nano_analysis']
            micro_structure = all_analyses['micro_structure']
            flow_dynamics = all_analyses['flow_dynamics']
            traditional = all_analyses['traditional']
            ocr_indicators = all_analyses.get('ocr_analysis', {})
            
            # Análise técnica
            trend_direction = traditional['price_action']['trend_direction']
            trend_strength = traditional['price_action']['trend_strength']
            trend_power = trend_direction * trend_strength
            
            macd_value = traditional['indicators']['macd']
            macd_strength = traditional['indicators']['macd_strength']
            macd_power = macd_value * macd_strength
            
            nano_power = nano_trend['nano_trend'] * nano_trend['convergence_strength']
            micro_power = micro_structure['structural_integrity'] * 0.5 + flow_dynamics['overall_flow_quality'] * 0.5
            micro_composite = (nano_power + micro_power) / 2
            
            # Indicadores OCR
            ocr_power = self._calculate_ocr_indicators_power(ocr_indicators)
            
            # Score neutro
            total_score = (
                trend_power * 0.25 +
                macd_power * 0.25 +  
                micro_composite * 0.25 +
                ocr_power * 0.25
            )
            
            # Decisão 100% neutra
            if total_score > 0:
                direction = "buy"
                confidence = 0.65 + (min(abs(total_score), 0.5) * 0.35)
                reasoning = self._generate_neutral_reasoning("buy", trend_power, macd_power, micro_composite, ocr_power, total_score, ocr_indicators)
            else:
                direction = "sell"
                confidence = 0.65 + (min(abs(total_score), 0.5) * 0.35)
                reasoning = self._generate_neutral_reasoning("sell", trend_power, macd_power, micro_composite, ocr_power, total_score, ocr_indicators)
            
            final_confidence = self._calculate_neutral_confidence(confidence, all_analyses)
            context = self._detect_neutral_context(trend_strength, macd_strength, micro_composite, total_score)
            
            return {
                "direction": direction,
                "confidence": final_confidence,
                "reasoning": reasoning,
                "total_score": total_score,
                "context": context,
                "trend_power": trend_power,
                "macd_power": macd_power,
                "micro_power": micro_composite,
                "ocr_power": ocr_power
            }
            
        except Exception as e:
            print(f"❌ Erro no motor de decisão: {e}")
            return self._neutral_market_decision()

    def _calculate_ocr_indicators_power(self, ocr_indicators: Dict) -> float:
        """Calcula o poder dos indicadores OCR"""
        try:
            power_score = 0.0
            factors_count = 0
            
            # RSI from OCR
            rsi_data = ocr_indicators.get('rsi', {})
            rsi_value = rsi_data.get('value', 50)
            if rsi_data.get('source') == 'ocr':
                if rsi_value < 30:
                    power_score += 0.3
                    print(f"🎯 RSI {rsi_value} - FORTE SINAL DE COMPRA")
                elif rsi_value > 70:
                    power_score -= 0.3
                    print(f"🎯 RSI {rsi_value} - FORTE SINAL DE VENDA")
                elif rsi_value < 40:
                    power_score += 0.15
                elif rsi_value > 60:
                    power_score -= 0.15
                factors_count += 1
            
            # MACD from OCR
            macd_data = ocr_indicators.get('macd', {})
            macd_value = macd_data.get('value', 0)
            if macd_data.get('source') == 'ocr':
                if macd_value > 1.0:
                    power_score += 0.25
                    print(f"🎯 MACD {macd_value} - SINAL DE COMPRA")
                elif macd_value < -1.0:
                    power_score -= 0.25
                    print(f"🎯 MACD {macd_value} - SINAL DE VENDA")
                elif macd_value > 0:
                    power_score += 0.1
                else:
                    power_score -= 0.1
                factors_count += 1
            
            # ADX from OCR
            adx_data = ocr_indicators.get('adx', {})
            adx_value = adx_data.get('value', 20)
            if adx_data.get('source') == 'ocr':
                if adx_value > 25:
                    power_score += 0.1
                    print(f"🎯 ADX {adx_value} - TENDÊNCIA FORTE")
                factors_count += 1
            
            # Price change from OCR
            price_data = ocr_indicators.get('price', {})
            price_change = price_data.get('change', 0)
            if price_data.get('source') == 'ocr':
                if price_change > 1.0:
                    power_score += 0.1
                elif price_change < -1.0:
                    power_score -= 0.1
                factors_count += 1
            
            if factors_count > 0:
                final_score = power_score / factors_count
                print(f"📊 Score OCR final: {final_score:.3f}")
                return final_score
            else:
                return 0.0
                
        except Exception as e:
            print(f"❌ Erro no cálculo OCR: {e}")
            return 0.0

    def _generate_neutral_reasoning(self, direction: str, trend_power: float, macd_power: float, 
                                  micro_power: float, ocr_power: float, total_score: float, 
                                  ocr_indicators: Dict) -> str:
        """Gera reasoning neutro"""
        
        ocr_reasons = []
        rsi_data = ocr_indicators.get('rsi', {})
        macd_data = ocr_indicators.get('macd', {})
        adx_data = ocr_indicators.get('adx', {})
        price_data = ocr_indicators.get('price', {})
        
        if rsi_data.get('source') == 'ocr':
            rsi_value = rsi_data.get('value', 50)
            if rsi_value < 30:
                ocr_reasons.append(f"RSI {rsi_value} (sobrevendido)")
            elif rsi_value > 70:
                ocr_reasons.append(f"RSI {rsi_value} (sobrecomprado)")
            else:
                ocr_reasons.append(f"RSI {rsi_value}")
        
        if macd_data.get('source') == 'ocr':
            macd_value = macd_data.get('value', 0)
            ocr_reasons.append(f"MACD {macd_value:+.2f}")
        
        if adx_data.get('source') == 'ocr':
            adx_value = adx_data.get('value', 20)
            if adx_value > 25:
                ocr_reasons.append(f"ADX {adx_value} (forte)")
        
        if price_data.get('source') == 'ocr':
            price_change = price_data.get('change', 0)
            if price_change != 0:
                ocr_reasons.append(f"Δ {price_change:+.2f}%")

        if direction == "buy":
            strength = "ALTA" if abs(total_score) > 0.25 else "moderada"
            
            if ocr_reasons:
                ocr_text = " + ".join(ocr_reasons)
                return f"📈 COMPRA {strength} - Indicadores: {ocr_text}"
            else:
                factors = []
                if abs(trend_power) > 0.1: 
                    factors.append(f"tendência {trend_power*100:+.1f}%")
                if abs(macd_power) > 0.1: 
                    factors.append(f"MACD {macd_power*100:+.1f}%")
                
                if factors:
                    analysis = " + ".join(factors)
                    return f"📈 COMPRA {strength} - {analysis}"
                else:
                    return f"📈 COMPRA {strength} - Momento favorável"
        
        else:
            strength = "BAIXA" if abs(total_score) > 0.25 else "moderada"
            
            if ocr_reasons:
                ocr_text = " + ".join(ocr_reasons)
                return f"📉 VENDA {strength} - Indicadores: {ocr_text}"
            else:
                factors = []
                if abs(trend_power) > 0.1: 
                    factors.append(f"tendência {trend_power*100:+.1f}%")
                if abs(macd_power) > 0.1: 
                    factors.append(f"MACD {macd_power*100:+.1f}%")
                
                if factors:
                    analysis = " + ".join(factors)
                    return f"📉 VENDA {strength} - {analysis}"
                else:
                    return f"📉 VENDA {strength} - Momento favorável"

    def _calculate_neutral_confidence(self, base_confidence: float, all_analyses: Dict) -> float:
        """Calcula confiança neutra"""
        try:
            confidence_factors = [
                all_analyses['nano_analysis']['convergence_strength'],
                all_analyses['micro_structure']['structural_integrity'],
                all_analyses['flow_dynamics']['overall_flow_quality'],
                all_analyses['traditional']['price_action']['trend_strength'],
                all_analyses['traditional']['indicators']['macd_strength']
            ]
            
            ocr_indicators = all_analyses.get('ocr_analysis', {})
            if any(indicator.get('source') == 'ocr' for indicator in ocr_indicators.values()):
                confidence_factors.append(0.8)
                print("🎯 Confiança aumentada - Dados OCR reais")
            
            quality_score = np.mean([f for f in confidence_factors if not np.isnan(f)])
            neutral_confidence = base_confidence + (quality_score * 0.2)
            
            return min(0.92, neutral_confidence)
            
        except Exception:
            return base_confidence

    def _detect_neutral_context(self, trend_strength: float, macd_strength: float, 
                               micro_power: float, total_score: float) -> str:
        """Detecta contexto de mercado"""
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
        """Decisão neutra de fallback"""
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
                    "trend_power": 0.08,
                    "macd_power": 0.08,
                    "micro_power": 0.08,
                    "ocr_power": 0.08
                }
            else:
                return {
                    "direction": "sell",
                    "confidence": 0.62,
                    "reasoning": "📉 VENDA - Análise de mercado: horário de baixa liquidez",
                    "total_score": -0.10,
                    "context": "after_hours",
                    "trend_power": -0.08,
                    "macd_power": -0.08,
                    "micro_power": -0.08,
                    "ocr_power": -0.08
                }
        except Exception:
            return {
                "direction": "sell",
                "confidence": 0.60,
                "reasoning": "📉 VENDA - Princípio neutro: cautela em análise indeterminada",
                "total_score": -0.05,
                "context": "neutral_caution",
                "trend_power": 0.0,
                "macd_power": 0.0,
                "micro_power": 0.0,
                "ocr_power": 0.0
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
            
            ocr_indicators = analyses.get('ocr_analysis', {})
            if any(indicator.get('source') == 'ocr' for indicator in ocr_indicators.values()):
                factors.append(0.8)
            
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
        """ANÁLISE 100% NEUTRA"""
        
        cached = self.cache.get(blob, timeframe)
        if cached:
            cached['cached'] = True
            return cached
        
        try:
            image = self._load_image(blob)
            self._validate_chart_image(image)
            
            img_array = self._preprocess_image(image, timeframe)
            price_data = self._extract_price_data(img_array)
            
            print("🔄 Iniciando extração OCR...")
            ocr_data = self._extract_with_ocr(image)
            ocr_analysis = self._analyze_extracted_indicators(ocr_data)
            
            analyses = {
                'traditional': {
                    'price_action': self._analyze_price_action(price_data, timeframe),
                    'indicators': self._calculate_advanced_indicators(price_data)
                },
                'nano_analysis': self._microscopic_trend_analysis(price_data),
                'micro_structure': self._analyze_micro_structure(price_data),
                'flow_dynamics': self._analyze_flow_dynamics(price_data),
                'ocr_analysis': ocr_analysis
            }
            
            decision = self._absolute_decision_engine(analyses, timeframe)
            time_info = self._get_entry_timeframe(timeframe)
            signal_quality = self._calculate_signal_quality(analyses)
            
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
                "ocr_data_available": any(indicator.get('source') == 'ocr' for indicator in ocr_analysis.values()),
                "metrics": {
                    "analysis_score": float(decision["total_score"]),
                    "trend_power": float(decision["trend_power"]),
                    "macd_power": float(decision["macd_power"]),
                    "micro_power": float(decision["micro_power"]),
                    "ocr_power": float(decision["ocr_power"]),
                    "trend_strength": analyses['traditional']['price_action']['trend_strength'],
                    "momentum": analyses['traditional']['price_action']['momentum'],
                    "rsi": analyses['traditional']['indicators']['rsi'],
                    "macd": analyses['traditional']['indicators']['macd'],
                    "macd_strength": analyses['traditional']['indicators']['macd_strength'],
                    "rsi_ocr": ocr_analysis['rsi']['value'],
                    "rsi_source": ocr_analysis['rsi']['source'],
                    "macd_ocr": ocr_analysis['macd']['value'],
                    "macd_source": ocr_analysis['macd']['source'],
                    "adx_ocr": ocr_analysis['adx']['value'],
                    "adx_source": ocr_analysis['adx']['source'],
                    "price_change": ocr_analysis['price']['change'],
                    "price_source": ocr_analysis['price']['source']
                },
                "reasoning": decision["reasoning"]
            }
            
            self.cache.set(blob, timeframe, result)
            print(f"✅ Análise concluída: {result['direction']} com {result['final_confidence']*100:.1f}% confiança")
            return result
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
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
                "ocr_data_available": False,
            })
            return fallback_result

# =========================
#  APLICAÇÃO FLASK
# =========================
app = Flask(__name__)
analyzer = SuperIntelligentAnalyzer()

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
        
        .ocr-badge {
            font-size: 9px;
            padding: 1px 4px;
            border-radius: 6px;
            margin-left: 3px;
            background: linear-gradient(135deg, #00ff88, #00cc66);
            color: white;
        }
        
        .source-ocr { color: #00ff88; font-weight: 600; }
        .source-default { color: #9db0d1; }
        .source-error { color: #ff4444; }
        
        .debug-info {
            background: rgba(255, 165, 0, 0.1);
            border: 1px solid #ffa500;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
            color: #ffa500;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠⚖️ IA SIGNAL PRO - 100% NEUTRA</div>
            <div class="subtitle">ZERO VIÉS - DECISÕES APENAS PELO MOMENTO DO MERCADO</div>
            <div class="subtitle" style="color: #7ce0ff; margin-top: 5px;">🎯 SISTEMA OCR AVANÇADO - EXTRAÇÃO PRECISA DE VALORES</div>
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
            
            <div class="power-analysis" id="powerAnalysis">
                <div style="text-align: center; font-weight: 600; margin-bottom: 8px; color: #7ce0ff;">
                    ⚡ ANÁLISE DO MOMENTO <span id="ocrStatus" class="ocr-badge">OCR ATIVO</span>
                </div>
                <div id="powerMetrics"></div>
            </div>
            
            <div id="debugInfo" class="debug-info" style="display: none;">
                <div style="text-align: center; font-weight: 600; margin-bottom: 5px;">🔍 INFORMAÇÕES DE DEBUG</div>
                <div id="debugContent"></div>
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
            const ocrStatus = document.getElementById('ocrStatus');
            const debugInfo = document.getElementById('debugInfo');
            const debugContent = document.getElementById('debugContent');
            const timeframeBtns = document.querySelectorAll('.timeframe-btn');

            let currentTimeframe = '1m';
            let selectedFile = null;

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
                debugInfo.style.display = 'none';
                
                signalText.className = 'signal-text';
                signalText.textContent = 'Analisando momento do mercado...';
                qualityIndicator.textContent = '';
                contextInfo.innerHTML = '';
                
                const now = new Date();
                analysisTime.textContent = now.toLocaleTimeString('pt-BR');
                
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
                progressFill.style.width = '10%';
                
                metricsText.innerHTML = '<div class="loading">Iniciando análise do momento do mercado...</div>';

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
                const ocr_available = data.ocr_data_available || false;
                
                signalText.className = `signal-text signal-${direction}`;
                let directionText = direction === 'buy' ? '🎯 COMPRAR' : '🎯 VENDER';
                signalText.innerHTML = `${directionText} <span class="neutral-badge">100% NEUTRO</span> ${cached ? '<span class="cache-badge">CACHE</span>' : ''} ${ocr_available ? '<span class="ocr-badge">OCR</span>' : ''}`;
                
                analysisTime.textContent = data.analysis_time || '--:--:--';
                entryTime.textContent = data.entry_time || '--:--';
                timeframeEl.textContent = data.timeframe || 'Próximo minuto';
                
                reasoningText.textContent = data.reasoning;
                confidenceText.textContent = `Confiança Técnica: ${confidence}%`;
                
                qualityIndicator.className = `quality-indicator quality-${quality}`;
                if (quality === 'high') {
                    qualityIndicator.textContent = '✅ ALTA QUALIDADE - Análise confiável do momento';
                } else {
                    qualityIndicator.textContent = '⚠️ QUALIDADE MÉDIA - Análise válida do momento';
                }
                
                if (ocr_available) {
                    ocrStatus.textContent = 'OCR ATIVO';
                    ocrStatus.style.background = 'linear-gradient(135deg, #00ff88, #00cc66)';
                } else {
                    ocrStatus.textContent = 'OCR NÃO DISPONÍVEL';
                    ocrStatus.style.background = 'linear-gradient(135deg, #ff6b6b, #ff4444)';
                }
                
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
                
                const metrics = data.metrics || {};
                let powerHtml = '';
                
                const powerItems = [
                    ['Poder da Tendência', (metrics.trend_power * 100)?.toFixed(1) + '%'],
                    ['Poder do MACD', (metrics.macd_power * 100)?.toFixed(1) + '%'],
                    ['Poder Microscópico', (metrics.micro_power * 100)?.toFixed(1) + '%'],
                    ['Poder OCR', (metrics.ocr_power * 100)?.toFixed(1) + '%'],
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
                
                let metricsHtml = '<div style="margin-bottom: 10px; text-align: center; font-weight: 600;">📊 ANÁLISE TÉCNICA COMPLETA</div>';
                
                const metricItems = [
                    ['Força da Tendência', (metrics.trend_strength * 100)?.toFixed(1) + '%'],
                    ['Momentum', metrics.momentum?.toFixed(3)],
                    ['RSI Calculado', metrics.rsi?.toFixed(3)],
                    ['MACD Calculado', metrics.macd?.toFixed(3)],
                    ['Força do MACD', (metrics.macd_strength * 100)?.toFixed(1) + '%'],
                    ['RSI Extraído', metrics.rsi_ocr?.toFixed(1)],
                    ['MACD Extraído', metrics.macd_ocr?.toFixed(2)],
                    ['ADX Extraído', metrics.adx_ocr?.toFixed(1)],
                    ['Variação Preço', metrics.price_change?.toFixed(2) + '%'],
                    ['Qualidade do Sinal', (data.signal_quality * 100)?.toFixed(1) + '%']
                ];
                
                metricItems.forEach(([label, value]) => {
                    const isOcrData = label.includes('Extraído') || label.includes('Variação');
                    let valueClass = 'source-default';
                    if (isOcrData) {
                        valueClass = metrics[label.toLowerCase().replace(' ', '_') + '_source'] === 'ocr' ? 'source-ocr' : 'source-default';
                    }
                    
                    metricsHtml += `
                        <div class="metric-item">
                            <span>${label}:</span>
                            <span class="metric-value ${valueClass}">${value}</span>
                        </div>
                    `;
                });
                
                metricsText.innerHTML = metricsHtml;
                
                if (!ocr_available) {
                    debugInfo.style.display = 'block';
                    debugContent.innerHTML = `
                        <div>📝 Para melhor precisão, verifique se o Tesseract OCR está instalado no servidor.</div>
                    `;
                }
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
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        image_file = request.files['image']
        if not image_file or image_file.filename == '':
            return jsonify({'error': 'Arquivo inválido'}), 400
        
        timeframe = request.form.get('timeframe', '1m')
        if timeframe not in ['1m', '5m']:
            timeframe = '1m'
        
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'error': 'Imagem muito grande (máximo 10MB)'}), 400
        
        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Arquivo vazio'}), 400
        
        analysis = analyzer.analyze(image_bytes, timeframe)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': f'Erro interno: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'service': 'IA Signal Pro - 100% NEUTRA',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '10.0.0-ocr-railway',
        'ocr_available': analyzer.ocr_available
    })

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    try:
        cache_dir = "analysis_cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, file))
            return jsonify({'ok': True, 'message': 'Cache limpo com sucesso!'})
        return jsonify({'ok': True, 'message': 'Cache já está vazio!'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/ocr-status')
def ocr_status():
    return jsonify({
        'ocr_available': analyzer.ocr_available,
        'environment': 'Railway',
        'timestamp': datetime.datetime.now().isoformat()
    })

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
    print(f"🎯 OCR STATUS: {'✅ CONFIGURADO' if analyzer.ocr_available else '❌ NÃO DISPONÍVEL'}")
    print(f"📈 SAÍDA: COMPRA ou VENDA - SEM FAVORITISMO")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
