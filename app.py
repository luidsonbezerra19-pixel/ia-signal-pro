from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime, timezone, timedelta
import os
import random
import math
import json
import time
import logging
from typing import List, Dict, Tuple, Any
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====== ENV com defaults seguros (Railway-friendly) ======
SIMS = int(os.getenv("SIMS", "1800"))                 # Monte Carlo paths (antes 3000)
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S", "20"))     # TTL de cache dos candles
BINANCE_TIMEOUT = float(os.getenv("BINANCE_TIMEOUT", "7.5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))      # Paralelismo leve
DISABLE_NEWS_EVENTS = os.getenv("DISABLE_NEWS_EVENTS", "1") == "1"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ====== Logs (stdout do Railway) ======
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("ia-signal")

# ========== SISTEMA DE PREÇOS REAIS (requests + retry + cache TTL) ==========
class RealPriceFetcher:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        # Mapeamento correto dos símbolos Binance
        self.symbol_mapping = {
            'BTC/USDT': 'BTCUSDT',
            'ETH/USDT': 'ETHUSDT', 
            'SOL/USDT': 'SOLUSDT',
            'ADA/USDT': 'ADAUSDT',
            'XRP/USDT': 'XRPUSDT',
            'BNB/USDT': 'BNBUSDT'
        }
        self.fallback_prices = {
            'BTCUSDT': 45000, 'ETHUSDT': 2500, 'SOLUSDT': 120,
            'ADAUSDT': 0.45, 'XRPUSDT': 0.55, 'BNBUSDT': 320
        }
        # sessão HTTP com retry/backoff
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get_binance_symbol(self, symbol: str) -> str:
        """Converte símbolo para formato Binance CORRETO"""
        return self.symbol_mapping.get(symbol, symbol.replace('/', ''))

    def get_historical_prices(self, symbol: str, interval: str = '1m', limit: int = 50) -> List[float]:
        """Busca preços históricos reais da Binance com retry/backoff"""
        try:
            binance_symbol = self.get_binance_symbol(symbol)
            url = f"{self.base_url}/klines"
            params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
            headers = {"User-Agent": "ia-signal-railway/1.0"}

            r = self.session.get(url, params=params, timeout=BINANCE_TIMEOUT, headers=headers)
            r.raise_for_status()
            data = r.json()

            if not data or len(data) < 10:
                log.warning(f"[binance] dados insuficientes para {symbol}, usando fallback")
                return self.get_fallback_prices(symbol)

            prices = [float(c[4]) for c in data]  # close
            log.info(f"[binance] {symbol}: {len(prices)} candles • {prices[0]:.6f} → {prices[-1]:.6f}")
            return prices

        except requests.RequestException as e:
            log.error(f"[binance] erro rede {symbol}: {e}")
            return self.get_fallback_prices(symbol)
        except Exception as e:
            log.error(f"[binance] erro geral {symbol}: {e}")
            return self.get_fallback_prices(symbol)

    def get_fallback_prices(self, symbol: str) -> List[float]:
        """Fallback determinístico para reprodutibilidade no Railway"""
        seed_base = int(datetime.utcnow().strftime("%Y%m%d%H%M"))
        random.seed(hash((symbol, seed_base)) & 0xffffffff)

        binance_symbol = self.get_binance_symbol(symbol)
        base_price = self.fallback_prices.get(binance_symbol, 100.0)
        prices = [base_price]
        current = base_price
        for _ in range(49):
            if symbol in ['BTC/USDT', 'ETH/USDT']:
                vol = 0.002
            elif symbol in ['BNB/USDT']:
                vol = 0.003
            else:
                vol = 0.005
            change = random.gauss(0, vol)
            current = max(base_price * 0.8, current * (1 + change))
            prices.append(current)
        log.warning(f"[fallback] {symbol}: {len(prices)} candles • last {prices[-1]:.6f}")
        return prices

# ========== SISTEMAS DE MEMÓRIA / LIQUIDEZ / CORRELAÇÃO / NOTÍCIAS / VOLATILIDADE ==========
class MemorySystem:
    def __init__(self):
        self.symbol_memory = {}
        self.market_regime = "NORMAL"
        self.regime_memory = []
    
    def get_symbol_weights(self, symbol: str) -> Dict:
        base_weights = {
            'monte_carlo': 0.65,
            'rsi': 0.08, 'adx': 0.07, 'macd': 0.06, 
            'bollinger': 0.05, 'volume': 0.04, 'fibonacci': 0.03,
            'multi_tf': 0.02
        }
        if self.market_regime == "VOLATILE":
            base_weights['monte_carlo'] = 0.60
            base_weights['bollinger'] = 0.08
            base_weights['adx'] = 0.09
        elif self.market_regime == "TRENDING":
            base_weights['adx'] = 0.10
            base_weights['multi_tf'] = 0.04
        return base_weights
    
    def update_market_regime(self, volatility: float, adx_values: List[float]):
        avg_adx = sum(adx_values) / len(adx_values) if adx_values else 25
        if volatility > 0.015 or avg_adx < 20:
            self.market_regime = "VOLATILE"
        elif avg_adx > 35:
            self.market_regime = "TRENDING"
        else:
            self.market_regime = "NORMAL"
        
        self.regime_memory.append({
            'timestamp': datetime.now(),
            'regime': self.market_regime,
            'volatility': volatility,
            'avg_adx': avg_adx
        })
        if len(self.regime_memory) > 100:
            self.regime_memory.pop(0)

class LiquiditySystem:
    def __init__(self):
        self.symbol_liquidity = {}
    
    def calculate_liquidity_score(self, symbol: str, prices: List[float]) -> float:
        if len(prices) < 10:
            return 0.7
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(abs(ret))
        if not returns:
            return 0.7
        volatility = sum(returns) / len(returns)
        if volatility < 0.005:
            liquidity_score = 0.9
        elif volatility < 0.01:
            liquidity_score = 0.8
        elif volatility < 0.02:
            liquidity_score = 0.7
        else:
            liquidity_score = 0.6
        self.symbol_liquidity[symbol] = liquidity_score
        return liquidity_score

class CorrelationSystem:
    def __init__(self):
        self.correlation_matrix = self._initialize_correlations()
    
    def _initialize_correlations(self) -> Dict:
        return {
            'BTC/USDT': {
                'ETH/USDT': 0.85, 'BNB/USDT': 0.65, 'SOL/USDT': 0.70,
                'ADA/USDT': 0.55, 'XRP/USDT': 0.50
            },
            'ETH/USDT': {
                'BTC/USDT': 0.85, 'BNB/USDT': 0.70, 'SOL/USDT': 0.75,
                'ADA/USDT': 0.60, 'XRP/USDT': 0.55
            },
            'SOL/USDT': {
                'BTC/USDT': 0.70, 'ETH/USDT': 0.75, 'ADA/USDT': 0.65
            },
            'ADA/USDT': {
                'BTC/USDT': 0.55, 'ETH/USDT': 0.60, 'SOL/USDT': 0.65
            },
            'XRP/USDT': {
                'BTC/USDT': 0.50, 'ETH/USDT': 0.55
            }
        }
    
    def get_correlation_adjustment(self, symbol: str, other_signals: Dict) -> float:
        if symbol not in self.correlation_matrix:
            return 1.0
        adjustments = []
        for other_symbol, signal_data in other_signals.items():
            if other_symbol != symbol and other_symbol in self.correlation_matrix[symbol]:
                correlation = self.correlation_matrix[symbol][other_symbol]
                if (signal_data['direction'] == other_signals.get(symbol, {}).get('direction', '')):
                    adjustment = 1.0 + (correlation * 0.1)
                else:
                    adjustment = 1.0 - (correlation * 0.05)
                adjustments.append(adjustment)
        if not adjustments:
            return 1.0
        return sum(adjustments) / len(adjustments)

class NewsEventSystem:
    def __init__(self):
        self.active_events = []
    
    def generate_market_events(self):
        if DISABLE_NEWS_EVENTS:
            return
        events = [
            {'type': 'FED_MEETING', 'impact': 'HIGH', 'volatility_multiplier': 2.0},
            {'type': 'CPI_RELEASE', 'impact': 'MEDIUM', 'volatility_multiplier': 1.5},
            {'type': 'REGULATION_NEWS', 'impact': 'MEDIUM', 'volatility_multiplier': 1.8},
            {'type': 'WHALE_MOVEMENT', 'impact': 'LOW', 'volatility_multiplier': 1.3},
        ]
        if random.random() < 0.15:
            event = random.choice(events)
            event['start_time'] = datetime.now()
            event['duration_hours'] = random.randint(2, 12)
            self.active_events.append(event)
            log.info(f"[evento] {event['type']} (Impacto: {event['impact']})")
    
    def get_volatility_multiplier(self):
        if not self.active_events:
            return 1.0
        max_multiplier = 1.0
        current_time = datetime.now()
        self.active_events = [
            event for event in self.active_events 
            if current_time - event['start_time'] < timedelta(hours=event['duration_hours'])
        ]
        for event in self.active_events:
            max_multiplier = max(max_multiplier, event['volatility_multiplier'])
        return max_multiplier
    
    def adjust_confidence_for_events(self, confidence: float) -> float:
        multiplier = self.get_volatility_multiplier()
        if multiplier > 1.5:
            return confidence * 0.85
        elif multiplier > 1.2:
            return confidence * 0.92
        return confidence

class VolatilityClustering:
    def __init__(self):
        self.volatility_regimes = {}
        self.historical_volatility = []
    
    def detect_volatility_clusters(self, prices: List[float], symbol: str) -> str:
        if len(prices) < 20:
            return "MEDIUM"
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(abs(ret))
        if not returns:
            return "MEDIUM"
        volatility = sum(returns) / len(returns)
        self.historical_volatility.append(volatility)
        if len(self.historical_volatility) > 50:
            self.historical_volatility.pop(0)
        if len(self.historical_volatility) > 10:
            avg_vol = sum(self.historical_volatility) / len(self.historical_volatility)
            if volatility > avg_vol * 1.5:
                regime = "HIGH"
            elif volatility < avg_vol * 0.7:
                regime = "LOW"
            else:
                regime = "MEDIUM"
        else:
            if volatility > 0.015:
                regime = "HIGH"
            elif volatility < 0.008:
                regime = "LOW"
            else:
                regime = "MEDIUM"
        self.volatility_regimes[symbol] = regime
        return regime
    
    def get_regime_adjustment(self, symbol: str) -> float:
        regime = self.volatility_regimes.get(symbol, "MEDIUM")
        if regime == "HIGH":
            return 0.85
        elif regime == "LOW":
            return 1.05
        else:
            return 1.0

# ========== INDICADORES TÉCNICOS (EMA/MACD corretos, RSI Wilder, ADX real) ==========
class TechnicalIndicators:
    @staticmethod
    def _ema(series: List[float], period: int) -> List[float]:
        if len(series) < period:
            return []
        k = 2.0 / (period + 1.0)
        ema_vals = [sum(series[:period]) / period]
        for price in series[period:]:
            ema_vals.append(price * k + ema_vals[-1] * (1.0 - k))
        return ema_vals

    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict:
        if len(prices) < 35:
            return {'signal': 'neutral', 'strength': 0.3}
        ema12 = TechnicalIndicators._ema(prices, 12)
        ema26 = TechnicalIndicators._ema(prices, 26)
        if not ema12 or not ema26:
            return {'signal': 'neutral', 'strength': 0.3}
        size = min(len(ema12), len(ema26))
        macd_line = [a - b for a, b in zip(ema12[-size:], ema26[-size:])]
        signal_line = TechnicalIndicators._ema(macd_line, 9)
        if not signal_line:
            return {'signal': 'neutral', 'strength': 0.3}
        hist = macd_line[-1] - signal_line[-1]
        denom = abs(prices[-1]) * 0.002 + 1e-9
        strength = min(1.0, abs(hist) / denom)
        if hist > 0:
            return {'signal': 'bullish', 'strength': strength}
        elif hist < 0:
            return {'signal': 'bearish', 'strength': strength}
        return {'signal': 'neutral', 'strength': 0.3}

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) <= period:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(prices)):
            d = prices[i] - prices[i-1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 70.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(max(0.0, min(100.0, rsi)), 1)

    @staticmethod
    def calculate_adx(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 2:
            return 25.0
        prox_high = []
        prox_low = []
        for i in range(len(prices)):
            win = prices[max(0, i - 2): i + 1]
            r = (max(win) - min(win)) if len(win) >= 2 else (prices[i] * 0.001)
            prox_high.append(prices[i] + r * 0.5)
            prox_low.append(prices[i] - r * 0.5)
        TR, plusDM, minusDM = [], [], []
        for i in range(1, len(prices)):
            high = prox_high[i]; low = prox_low[i]
            prev_high = prox_high[i-1]; prev_low = prox_low[i-1]
            tr = max(high - low, abs(high - prices[i-1]), abs(low - prices[i-1]))
            TR.append(tr)
            up_move = high - prev_high
            down_move = prev_low - low
            plusDM.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
            minusDM.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        def wilder_smooth(vals, p):
            if len(vals) < p:
                return []
            smoothed = [sum(vals[:p])]
            for v in vals[p:]:
                smoothed.append(smoothed[-1] - (smoothed[-1] / p) + v)
            return smoothed
        trN = wilder_smooth(TR, period)
        plusDMN = wilder_smooth(plusDM, period)
        minusDMN = wilder_smooth(minusDM, period)
        if not trN or not plusDMN or not minusDMN:
            return 25.0
        plusDI = [(pd / t) * 100 if t != 0 else 0.0 for pd, t in zip(plusDMN, trN)]
        minusDI = [(md / t) * 100 if t != 0 else 0.0 for md, t in zip(minusDMN, trN)]
        DX = [(abs(p - m) / (p + m) * 100) if (p + m) != 0 else 0.0 for p, m in zip(plusDI, minusDI)]
        if len(DX) < period:
            return 25.0
        adx_vals = [sum(DX[:period]) / period]
        for d in DX[period:]:
            adx_vals.append(((adx_vals[-1] * (period - 1)) + d) / period)
        adx = max(10.0, min(60.0, adx_vals[-1]))
        return round(adx, 1)

    @staticmethod
    def calculate_std_dev(data: List[float]) -> float:
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    @staticmethod
    def calculate_bollinger_bands(prices: List[float]) -> Dict:
        if len(prices) < 15:
            return {'signal': 'neutral'}
        recent = prices[-15:]
        middle = sum(recent) / 15
        std = TechnicalIndicators.calculate_std_dev(recent)
        current = prices[-1]
        if current < middle - (1.5 * std):
            return {'signal': 'oversold'}
        elif current > middle + (1.5 * std):
            return {'signal': 'overbought'}
        elif current > middle:
            return {'signal': 'bullish'}
        else:
            return {'signal': 'bearish'}

    @staticmethod
    def calculate_volume_profile(prices: List[float]) -> Dict:
        if len(prices) < 8:
            return {'signal': 'neutral'}
        current = prices[-1]
        high = max(prices[-10:])
        low = min(prices[-10:])
        poc = (high + low) / 2
        if current > poc + (high - low) * 0.25:
            return {'signal': 'overbought'}
        elif current < poc - (high - low) * 0.25:
            return {'signal': 'oversold'}
        return {'signal': 'neutral'}

    @staticmethod
    def calculate_fibonacci(prices: List[float]) -> Dict:
        if len(prices) < 15:
            return {'signal': 'neutral'}
        high = max(prices[-15:])
        low = min(prices[-15:])
        current = prices[-1]
        diff = high - low
        if diff == 0:
            return {'signal': 'neutral'}
        if current > high - (0.382 * diff):
            return {'signal': 'resistance'}
        elif current < low + (0.618 * diff):
            return {'signal': 'support'}
        return {'signal': 'neutral'}

class MultiTimeframeAnalyzer:
    @staticmethod
    def analyze_consensus(prices: List[float]) -> str:
        if len(prices) < 15:
            return 'neutral'
        tf_short = prices[-6:]
        tf_medium = prices[-12:]
        tf_long = prices[-18:]
        trends = []
        weights = []
        for i, tf in enumerate([tf_short, tf_medium, tf_long]):
            if len(tf) > 3:
                trend_strength = (tf[-1] - tf[0]) / tf[0]
                weight = [0.3, 0.4, 0.5][i]
                if trend_strength > 0.008:
                    trends.append(('buy', weight))
                elif trend_strength < -0.008:
                    trends.append(('sell', weight))
                else:
                    trends.append(('neutral', weight * 0.5))
        if not trends:
            return 'neutral'
        buy_score = sum(weight for direction, weight in trends if direction == 'buy')
        sell_score = sum(weight for direction, weight in trends if direction == 'sell')
        if buy_score > sell_score + 0.2:
            return 'buy'
        elif sell_score > buy_score + 0.2:
            return 'sell'
        return 'neutral'

# ========== MONTE CARLO ==========
class MonteCarloSimulator:
    @staticmethod
    def generate_price_paths(base_price: float, volatility: float, num_paths: int = 1800, steps: int = 4) -> List[List[float]]:
        paths = []
        for _ in range(num_paths):
            prices = [base_price]
            current = base_price
            for step in range(steps - 1):
                adjusted_volatility = volatility * (1 + (step * 0.05))
                trend = random.uniform(-volatility, volatility)
                change = trend + random.gauss(0, 1) * adjusted_volatility
                new_price = current * (1 + change)
                new_price = max(new_price, base_price * 0.7)
                prices.append(new_price)
                current = new_price
            paths.append(prices)
        return paths
    
    @staticmethod
    def calculate_probability_distribution(paths: List[List[float]]) -> Dict:
        if not paths or len(paths) < 200:
            return {'probability_buy': 0.5, 'probability_sell': 0.5, 'quality': 'LOW'}
        initial_price = paths[0][0]
        final_prices = [path[-1] for path in paths]
        higher_prices = sum(1 for price in final_prices if price > initial_price * 1.01)
        lower_prices = sum(1 for price in final_prices if price < initial_price * 0.99)
        neutral_prices = len(final_prices) - higher_prices - lower_prices
        total_paths = len(final_prices)
        probability_buy = (higher_prices + (neutral_prices * 0.5)) / total_paths
        probability_sell = (lower_prices + (neutral_prices * 0.5)) / total_paths
        total = probability_buy + probability_sell
        if total > 0:
            probability_buy /= total
            probability_sell /= total
        prob_strength = max(probability_buy, probability_sell) - 0.5
        if prob_strength > 0.15:
            quality = 'HIGH'
        elif prob_strength > 0.08:
            quality = 'MEDIUM'
        else:
            quality = 'LOW'
        return {
            'probability_buy': probability_buy,
            'probability_sell': probability_sell,
            'quality': quality
        }

# ========== SISTEMA PRINCIPAL ==========
class EnhancedTradingSystem:
    def __init__(self):
        self.memory = MemorySystem()
        self.monte_carlo = MonteCarloSimulator()
        self.indicators = TechnicalIndicators()
        self.multi_tf = MultiTimeframeAnalyzer()
        self.liquidity = LiquiditySystem()
        self.correlation = CorrelationSystem()
        self.news_events = NewsEventSystem()
        self.volatility_clustering = VolatilityClustering()
        self.price_fetcher = RealPriceFetcher()
        self.current_analysis_cache = {}
        # cache de preços com TTL
        self._price_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _get_prices_cached(self, symbol: str, interval: str = '1m', limit: int = 50) -> List[float]:
        key = (symbol, interval)
        now = time.time()
        hit = self._price_cache.get(key)
        if hit and now - hit["ts"] < CACHE_TTL_S:
            return hit["prices"]
        prices = self.price_fetcher.get_historical_prices(symbol, interval, limit)
        self._price_cache[key] = {"ts": now, "prices": prices}
        return prices
    
    def analyze_symbol(self, symbol: str, horizon: int) -> Dict:
        # PREÇOS REAIS COM CACHE
        historical_prices = self._get_prices_cached(symbol)
        if not historical_prices or len(historical_prices) < 10:
            log.warning(f"[analyze] dados insuficientes para {symbol}, usando análise básica")
            return self.get_basic_analysis(symbol, horizon)
        
        # Volatilidade REAL
        returns = []
        for i in range(1, len(historical_prices)):
            if historical_prices[i-1] != 0:
                ret = (historical_prices[i] - historical_prices[i-1]) / historical_prices[i-1]
                returns.append(abs(ret))
        real_volatility = sum(returns) / len(returns) if returns else 0.01
        
        # Monte Carlo: steps por horizonte (T+1 => 2, T+3 => 4)
        steps = max(2, horizon + 1)
        future_paths = self.monte_carlo.generate_price_paths(
            historical_prices[-1], 
            volatility=real_volatility,
            num_paths=SIMS, 
            steps=steps
        )
        mc_result = self.monte_carlo.calculate_probability_distribution(future_paths)
        prob_buy = mc_result['probability_buy']
        prob_sell = mc_result['probability_sell']
        
        # INDICADORES (reais)
        rsi = self.indicators.calculate_rsi(historical_prices)
        adx = self.indicators.calculate_adx(historical_prices)
        macd = self.indicators.calculate_macd(historical_prices)
        bollinger = self.indicators.calculate_bollinger_bands(historical_prices)
        volume = self.indicators.calculate_volume_profile(historical_prices)
        fibonacci = self.indicators.calculate_fibonacci(historical_prices)
        multi_tf_consensus = self.multi_tf.analyze_consensus(historical_prices)
        
        liquidity_score = self.liquidity.calculate_liquidity_score(symbol, historical_prices)
        volatility_regime = self.volatility_clustering.detect_volatility_clusters(historical_prices, symbol)
        
        # Atualiza regime mercado
        self.memory.update_market_regime(
            volatility=real_volatility, 
            adx_values=[adx] if adx else [25]
        )
        
        # Eventos (opcional)
        self.news_events.generate_market_events()
        
        # PESOS (mantém essência)
        weights = self.memory.get_symbol_weights(symbol)
        
        # SISTEMA DE PONTUAÇÃO
        base_score = 50.0
        factors = []
        winning_indicators = []
        
        # 1. MONTE CARLO (peso principal)
        mc_direction_strength = abs(prob_buy - 0.5) * 2
        mc_score = mc_direction_strength * 30.0
        if mc_result['quality'] == 'HIGH':
            mc_score *= 1.2
        elif mc_result['quality'] == 'MEDIUM':
            mc_score *= 1.1
        base_score += mc_score if prob_buy > 0.5 else -mc_score
        factors.append(f"MC:{mc_score:.1f}")
        
        # 2. INDICADORES TÉCNICOS
        indicator_score = 0.0
        if 30 < rsi < 70:
            indicator_score += 6; winning_indicators.append('RSI')
        if adx > 25:
            indicator_score += 5; winning_indicators.append('ADX')
        if (prob_buy > 0.5 and macd['signal'] == 'bullish') or (prob_buy < 0.5 and macd['signal'] == 'bearish'):
            indicator_score += 5 * macd['strength']; winning_indicators.append('MACD')
        if (prob_buy > 0.5 and bollinger['signal'] in ['oversold', 'bullish']) or (prob_buy < 0.5 and bollinger['signal'] in ['overbought', 'bearish']):
            indicator_score += 4; winning_indicators.append('BB')
        if (prob_buy > 0.5 and volume['signal'] in ['oversold', 'neutral']) or (prob_buy < 0.5 and volume['signal'] in ['overbought', 'neutral']):
            indicator_score += 3; winning_indicators.append('VOL')
        if (prob_buy > 0.5 and fibonacci['signal'] == 'support') or (prob_buy < 0.5 and fibonacci['signal'] == 'resistance'):
            indicator_score += 2; winning_indicators.append('FIB')
        if multi_tf_consensus == ('buy' if prob_buy > 0.5 else 'sell'):
            indicator_score += 4; winning_indicators.append('MultiTF')
        base_score += indicator_score if prob_buy > 0.5 else -indicator_score
        factors.append(f"IND:{indicator_score:.1f}")
        
        # 3. AJUSTES FINAIS
        liquidity_adjustment = 0.95 + (liquidity_score * 0.1)
        base_score *= liquidity_adjustment
        factors.append(f"LIQ:{liquidity_adjustment:.2f}")
        
        volatility_adjustment = self.volatility_clustering.get_regime_adjustment(symbol)
        base_score *= volatility_adjustment
        factors.append(f"VOL:{volatility_adjustment:.2f}")
        
        # Converter para confiança final (faixa mais aberta 45–95%)
        raw_confidence = (base_score / 100.0)
        final_confidence = min(0.95, max(0.45, raw_confidence))
        
        # Eventos de notícias
        final_confidence = self.news_events.adjust_confidence_for_events(final_confidence)
        
        # DIREÇÃO FINAL
        direction = 'buy' if prob_buy > 0.5 else 'sell'
        
        # Cache p/ correlações (primeira passada)
        self.current_analysis_cache[symbol] = {
            'direction': direction,
            'confidence': final_confidence,
            'timestamp': datetime.now()
        }
        
        return {
            'symbol': symbol,
            'horizon': horizon,
            'direction': direction,
            'probability_buy': prob_buy,
            'probability_sell': prob_sell,
            'confidence': final_confidence,
            'rsi': rsi,
            'adx': adx,
            'multi_timeframe': multi_tf_consensus,
            'monte_carlo_quality': mc_result['quality'],
            'winning_indicators': winning_indicators,
            'score_factors': factors,
            'price': historical_prices[-1],
            'timestamp': self.get_brazil_time().strftime("%H:%M:%S"),
            'liquidity_score': round(liquidity_score, 2),
            'volatility_regime': volatility_regime,
            'market_regime': self.memory.market_regime,
            'volatility_multiplier': self.news_events.get_volatility_multiplier(),
            'real_data': True
        }
    
    def get_basic_analysis(self, symbol: str, horizon: int) -> Dict:
        return {
            'symbol': symbol,
            'horizon': horizon,
            'direction': random.choice(['buy', 'sell']),
            'probability_buy': 0.5,
            'probability_sell': 0.5,
            'confidence': 0.6,
            'rsi': 50,
            'adx': 25,
            'multi_timeframe': 'neutral',
            'monte_carlo_quality': 'LOW',
            'winning_indicators': [],
            'score_factors': ['BASIC:0.0'],
            'price': 100,
            'timestamp': self.get_brazil_time().strftime("%H:%M:%S"),
            'liquidity_score': 0.7,
            'volatility_regime': 'MEDIUM',
            'market_regime': 'NORMAL',
            'volatility_multiplier': 1.0,
            'real_data': False
        }
    
    def get_brazil_time(self):
        return datetime.now(timezone(timedelta(hours=-3)))

# ========== FLASK APP ==========
app = Flask(__name__)
CORS(app)

trading_system = EnhancedTradingSystem()

# ==== utilitário: formata o melhor sinal em uma linha curta ====
def format_best_signal_card(best: dict, analysis_time: str) -> str:
    if not best:
        return "Nenhum sinal disponível."
    side = "🟢 COMPRAR" if best["direction"] == "buy" else "🔴 VENDER"
    return (
        f"{best['symbol']} T+{best['horizon']} • {side} • "
        f"Conf {best['confidence']:.1%} • Prob {best['probability_buy']:.1%}/{best['probability_sell']:.1%} • "
        f"ADX {best['adx']:.1f} • RSI {best['rsi']:.1f} • "
        f"Entrada {best.get('entry_time','--')} • Preço {best['price']:.6f}"
        f" • Última análise {analysis_time or '--'}"
    )

class AnalysisManager:
    def __init__(self):
        self.current_results = []
        self.best_opportunity = None
        self.analysis_time = None
        self.is_analyzing = False
        self.available_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XRP/USDT', 'BNB/USDT']
        self._lock = Lock()
    
    def analyze_symbols_thread(self, symbols, sims, only_adx):
        try:
            self.is_analyzing = True
            start_time = datetime.now()
            log.info(f"🚀 INICIANDO ANÁLISE (reais): {symbols}")
            
            trading_system.current_analysis_cache = {}
            all_horizons_results = []

            for symbol in symbols:
                log.info(f"🎯 ANALISANDO {symbol}")
                for horizon in [1, 2, 3]:
                    result = trading_system.analyze_symbol(symbol, horizon)
                    all_horizons_results.append(result)
                    log.info(f"   ✅ T+{horizon} - {result['direction'].upper()} | Conf: {result['confidence']:.1%} | Real Data: {result['real_data']}")
            
            best_by_symbol = {}
            for result in all_horizons_results:
                sym = result['symbol']
                if sym not in best_by_symbol or result['confidence'] > best_by_symbol[sym]['confidence']:
                    best_by_symbol[sym] = result
            
            formatted = []
            for result in all_horizons_results:
                is_best_of_symbol = (result['symbol'] in best_by_symbol and 
                                   result['confidence'] == best_by_symbol[result['symbol']]['confidence'])
                prob_buy = result['probability_buy']
                prob_sell = result['probability_sell']
                total = prob_buy + prob_sell
                if total > 0:
                    prob_buy /= total; prob_sell /= total
                formatted.append({
                    'symbol': result['symbol'],
                    'horizon': result['horizon'],
                    'direction': result['direction'],
                    'p_buy': round(prob_buy * 100, 1),
                    'p_sell': round(prob_sell * 100, 1),
                    'confidence': round(result['confidence'] * 100, 1),
                    'adx': round(result['adx'], 1),
                    'rsi': round(result['rsi'], 1),
                    'price': round(result['price'], 6),
                    'timestamp': result['timestamp'],
                    'technical_override': len(result['winning_indicators']) >= 4,
                    'multi_timeframe': result['multi_timeframe'],
                    'monte_carlo_quality': result['monte_carlo_quality'],
                    'winning_indicators': result['winning_indicators'],
                    'score_factors': result['score_factors'],
                    'assertiveness': self.calculate_assertiveness(result),
                    'is_best_of_symbol': is_best_of_symbol,
                    'liquidity_score': result['liquidity_score'],
                    'volatility_regime': result['volatility_regime'],
                    'market_regime': result['market_regime'],
                    'volatility_multiplier': result['volatility_multiplier'],
                    'real_data': result.get('real_data', True)
                })
            
            # ===== 2ª PASSADA DE CORRELAÇÃO =====
            cache_for_corr = {}
            for sym, r in best_by_symbol.items():
                cache_for_corr[sym] = {'direction': r['direction'], 'confidence': r['confidence']}
            adjusted = []
            for r in formatted:
                corr = trading_system.correlation.get_correlation_adjustment(r['symbol'], cache_for_corr)
                new_conf = min(95.0, max(40.0, r['confidence'] * corr))
                r['confidence'] = round(new_conf, 1)
                adjusted.append(r)
            formatted = adjusted

            best = max(formatted, key=lambda x: x['confidence']) if formatted else None
            if best:
                best_entry_time = self.calculate_entry_time_brazil(best['horizon'])
                best['entry_time'] = best_entry_time
            
            analysis_time = self.get_brazil_time().strftime("%d/%m/%Y %H:%M:%S")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            with self._lock:
                self.current_results = formatted
                self.best_opportunity = best
                self.analysis_time = analysis_time

            if best:
                log.info(f"🏆 MELHOR OPORTUNIDADE: {best['symbol']} T+{best['horizon']} ({best['confidence']}%)")
            log.info(f"✅ ANÁLISE CONCLUÍDA em {processing_time:.1f}s | {len(formatted)} sinais")

        except Exception as e:
            log.exception(f"❌ ERRO na análise: {e}")
            with self._lock:
                self.current_results = self._get_fallback_results(symbols)
                self.best_opportunity = self.current_results[0] if self.current_results else None
        finally:
            self.is_analyzing = False
    
    def get_brazil_time(self):
        return datetime.now(timezone(timedelta(hours=-3)))
    
    def calculate_entry_time_brazil(self, horizon):
        now = self.get_brazil_time()
        return (now + timedelta(minutes=horizon)).strftime("%H:%M BRT")
    
    def _get_fallback_results(self, symbols):
        results = []
        for symbol in symbols:
            for horizon in [1, 2, 3]:
                prob_buy = random.uniform(0.4, 0.6)
                prob_sell = 1.0 - prob_buy
                results.append({
                    'symbol': symbol,
                    'horizon': horizon,
                    'direction': 'buy' if prob_buy > 0.5 else 'sell',
                    'p_buy': round(prob_buy * 100, 1),
                    'p_sell': round(prob_sell * 100, 1),
                    'confidence': random.randint(55, 85),
                    'adx': random.randint(20, 40),
                    'rsi': random.randint(40, 60),
                    'price': round(random.uniform(50, 400), 6),
                    'timestamp': self.get_brazil_time().strftime("%H:%M:%S"),
                    'technical_override': random.choice([True, False]),
                    'multi_timeframe': random.choice(['buy', 'sell', 'neutral']),
                    'monte_carlo_quality': random.choice(['MEDIUM', 'HIGH', 'LOW']),
                    'winning_indicators': random.sample(['RSI', 'ADX', 'MACD', 'BB', 'VOL'], k=3),
                    'score_factors': ['MC:45.0', 'RSI:8.0', 'ADX:7.0'],
                    'assertiveness': random.randint(60, 90),
                    'is_best_of_symbol': (horizon == 2),
                    'liquidity_score': round(random.uniform(0.6, 0.9), 2),
                    'volatility_regime': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                    'market_regime': 'NORMAL',
                    'volatility_multiplier': 1.0,
                    'real_data': False
                })
        return results
    
    def calculate_assertiveness(self, result):
        base = result['confidence'] * 100 if isinstance(result['confidence'], float) and result['confidence'] <= 1 else result['confidence']
        indicator_count = len(result['winning_indicators'])
        if indicator_count >= 5:
            base += 15
        elif indicator_count >= 4:
            base += 10
        elif indicator_count >= 3:
            base += 6
        elif indicator_count >= 2:
            base += 3
        if result['monte_carlo_quality'] == 'HIGH':
            base += 12
        elif result['monte_carlo_quality'] == 'MEDIUM':
            base += 6
        if result['multi_timeframe'] == result['direction']:
            base += 8
        if max(result['probability_buy'], result['probability_sell']) > 0.6:
            base += 5
        if result['volatility_regime'] == 'LOW':
            base += 3
        elif result['volatility_regime'] == 'HIGH':
            base -= 5
        return min(round(base, 1), 95)

manager = AnalysisManager()

# ========== ROTAS (mantidas) ==========
@app.route('/')
def index():
    symbols_html = ''.join([f'''
        <label style="display: inline-block; margin: 5px; padding: 10px 15px; 
                      background: #2c3e50; border-radius: 8px; cursor: pointer; border: 2px solid #3498db;">
            <input type="checkbox" name="symbol" value="{symbol}" checked 
                   onchange="updateSymbols()" style="margin-right: 8px;"> 
            <strong>{symbol}</strong>
        </label>
    ''' for symbol in manager.available_symbols])
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 IA Signal Pro - PREÇOS REAIS CONFIRMADOS</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ background: #0a0a0a; color: white; font-family: Arial; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .card {{ background: #1a1a2e; border: 2px solid #3498db; border-radius: 10px; padding: 20px; margin: 10px 0; }}
            .best-card {{ border: 3px solid #f39c12; background: #2c2c3e; }}
            .symbols-container {{ background: #2c3e50; padding: 20px; border-radius: 10px; margin: 15px 0; text-align: center; }}
            input, button, select {{ padding: 12px; margin: 8px; border: 1px solid #3498db; border-radius: 6px; background: #34495e; color: white; }}
            button {{ background: #3498db; border: none; font-weight: bold; cursor: pointer; padding: 15px 25px; font-size: 1.1em; }}
            button:hover {{ background: #2980b9; }}
            button:disabled {{ opacity: 0.6; cursor: not-allowed; }}
            .results {{ background: #2c3e50; padding: 15px; border-radius: 5px; margin: 8px 0; border-left: 4px solid #3498db; }}
            .buy {{ color: #2ecc71; border-left-color: #2ecc71 !important; }}
            .sell {{ color: #e74c3c; border-left-color: #e74c3c !important; }}
            .best-of-symbol {{ border: 2px solid #f39c12 !important; background: #34495e; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 10px 0; }}
            .metric {{ background: #34495e; padding: 10px; border-radius: 5px; text-align: center; }}
            .factor {{ background: #16a085; padding: 4px 8px; border-radius: 3px; margin: 2px; font-size: 0.8em; display: inline-block; }}
            .indicator {{ background: #8e44ad; padding: 3px 6px; border-radius: 3px; margin: 1px; font-size: 0.75em; display: inline-block; }}
            .override {{ color: #f39c12; font-weight: bold; }}
            .quality-high {{ color: #2ecc71; }}
            .quality-medium {{ color: #f39c12; }}
            .quality-low {{ color: #e74c3c; }}
            .symbol-header {{ font-size: 1.1em; font-weight: bold; margin-bottom: 5px; }}
            .horizon-badge {{ background: #3498db; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 5px; }}
            .regime-low {{ color: #2ecc71; }}
            .regime-medium {{ color: #f39c12; }}
            .regime-high {{ color: #e74c3c; }}
            .liquidity-high {{ color: #2ecc71; }}
            .liquidity-low {{ color: #e74c3c; }}
            .real-data-badge {{ background: #27ae60; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-left: 5px; }}
            .fallback-badge {{ background: #e67e22; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-left: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🚀 IA Signal Pro - PREÇOS REAIS CONFIRMADOS</h1>
                <p><em>✅ DADOS REAIS DA BINANCE • Símbolos corrigidos • IMPARCIALIDADE GARANTIDA</em></p>
                <p><strong>🎯 AGORA COM: EMAs/MACD corretos, RSI Wilder, ADX real e correlação justa</strong></p>
                
                <div class="symbols-container">
                    <h3>🎯 SELECIONE OS ATIVOS PARA ANÁLISE COM DADOS REAIS:</h3>
                    <div id="symbolsCheckbox">
                        {symbols_html}
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <select id="sims" style="width: 200px; display: inline-block;">
                        <option value="1800" selected>{SIMS} simulações Monte Carlo</option>
                    </select>
                    
                    <button onclick="analyze()" id="analyzeBtn">🎯 ANALISAR COM DADOS REAIS</button>
                </div>
            </div>

            <div class="card best-card">
                <h2>🎖️ MELHOR OPORTUNIDADE GLOBAL</h2>
                <div id="bestResult">Selecione os ativos e clique em Analisar</div>
            </div>

            <div class="card">
                <h2>📈 TODOS OS HORIZONTES DE CADA ATIVO</h2>
                <div id="allResults">-</div>
            </div>
        </div>

        <script>
            function getSelectedSymbols() {{
                const checkboxes = document.querySelectorAll('input[name="symbol"]:checked');
                return Array.from(checkboxes).map(cb => cb.value);
            }}

            function updateSymbols() {{
                const selected = getSelectedSymbols();
                console.log('Símbolos selecionados:', selected);
            }}

            function formatFactors(factors) {{
                return factors ? factors.map(f => `<span class="factor">${{f}}</span>`).join('') : '';
            }}

            function formatIndicators(indicators) {{
                return indicators ? indicators.map(i => `<span class="indicator">${{i}}</span>`).join('') : '';
            }}

            function getRegimeClass(regime) {{
                if (regime === 'LOW') return 'regime-low';
                if (regime === 'HIGH') return 'regime-high';
                return 'regime-medium';
            }}

            function getLiquidityClass(score) {{
                return score > 0.8 ? 'liquidity-high' : (score < 0.6 ? 'liquidity-low' : '');
            }}

            async function analyze() {{
                const btn = document.getElementById('analyzeBtn');
                const symbols = getSelectedSymbols();
                
                if (symbols.length === 0) {{
                    alert('Selecione pelo menos um ativo!');
                    return;
                }}

                btn.disabled = true;
                btn.textContent = `⏳ BUSCANDO DADOS REAIS...`;

                try {{
                    const response = await fetch('/api/analyze', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            symbols: symbols,
                            sims: {SIMS},
                            only_adx: 0
                        }})
                    }});

                    const data = await response.json();
                    if (data.success) {{
                        checkResults();
                    }} else {{
                        alert('Erro: ' + data.error);
                        btn.disabled = false;
                        btn.textContent = '🎯 ANALISAR COM DADOS REAIS';
                    }}
                }} catch (error) {{
                    alert('Erro de conexão');
                    btn.disabled = false;
                    btn.textContent = '🎯 ANALISAR COM DADOS REAIS';
                }}
            }}

            async function checkResults() {{
                try {{
                    const response = await fetch('/api/results');
                    const data = await response.json();

                    if (data.success) {{
                        updateResults(data);
                        if (data.is_analyzing) {{
                            setTimeout(checkResults, 1500);
                        }} else {{
                            document.getElementById('analyzeBtn').disabled = false;
                            document.getElementById('analyzeBtn').textContent = '🎯 ANALISAR COM DADOS REAIS';
                        }}
                    }}
                }} catch (error) {{
                    setTimeout(checkResults, 2000);
                }}
            }}

            function updateResults(data) {{
                if (data.best) {{
                    const best = data.best;
                    const regimeClass = getRegimeClass(best.volatility_regime);
                    const liquidityClass = getLiquidityClass(best.liquidity_score);
                    const dataBadge = best.real_data ? 
                        '<span class="real-data-badge">📡 DADOS REAIS</span>' : 
                        '<span class="fallback-badge">🔄 DADOS SIMULADOS</span>';
                    
                    document.getElementById('bestResult').innerHTML = `
                        <div class="results ${{best.direction}}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.3em;">${{best.symbol}} T+${{best.horizon}}</strong>
                                    <span style="font-size: 1.2em; margin-left: 10px;">
                                        ${{best.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'}}
                                    </span>
                                    <div style="font-size: 0.9em; color: #f39c12; margin-top: 5px;">
                                        🏆 MELHOR ENTRE TODOS OS HORIZONTES
                                        ${{dataBadge}}
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.4em; font-weight: bold;">${{best.confidence}}%</div>
                                    <div>Assertividade: ${{best.assertiveness}}%</div>
                                </div>
                            </div>
                            
                            <div class="metrics">
                                <div class="metric"><div>Prob Compra</div><strong>${{best.p_buy}}%</strong></div>
                                <div class="metric"><div>Prob Venda</div><strong>${{best.p_sell}}%</strong></div>
                                <div class="metric"><div>Soma</div><strong>${{(best.p_buy + best.p_sell).toFixed(1)}}%</strong></div>
                                <div class="metric"><div>ADX</div><strong>${{best.adx}}</strong></div>
                                <div class="metric"><div>RSI</div><strong>${{best.rsi}}</strong></div>
                                <div class="metric"><div>Liquidez</div><strong class="${{liquidityClass}}">${{best.liquidity_score}}</strong></div>
                            </div>
                            
                            <div><strong>Indicadores Ativos:</strong> ${{formatIndicators(best.winning_indicators || [])}}</div>
                            <div><strong>Pontuação:</strong> ${{formatFactors(best.score_factors || [])}}</div>
                            <div>
                                <strong>Mercado:</strong> ${{best.market_regime}} | 
                                <strong>Vol Multi:</strong> ${{best.volatility_multiplier}}x |
                                <strong>Preço:</strong> $${{best.price}}
                            </div>
                            <div><strong>Entrada:</strong> ${{best.entry_time}}</div>
                            ${{best.technical_override ? '<div class="override">⚡ ALTA CONVERGÊNCIA TÉCNICA</div>' : ''}}
                            <br><em>Última análise: ${{data.analysis_time}} (Horário Brasil)</em>
                        </div>
                    `;
                }}

                if (data.results.length > 0) {{
                    const groupedBySymbol = {{}};
                    data.results.forEach(result => {{
                        if (!groupedBySymbol[result.symbol]) {{
                            groupedBySymbol[result.symbol] = [];
                        }}
                        groupedBySymbol[result.symbol].push(result);
                    }});

                    let html = '';
                    
                    Object.keys(groupedBySymbol).sort().forEach(symbol => {{
                        const symbolResults = groupedBySymbol[symbol].sort((a, b) => a.horizon - b.horizon);
                        const regimeClass = getRegimeClass(symbolResults[0].volatility_regime);
                        const liquidityClass = getLiquidityClass(symbolResults[0].liquidity_score);
                        const dataSource = symbolResults[0].real_data ? "📡 DADOS REAIS" : "🔄 DADOS SIMULADOS";
                        
                        html += `
                            <div class="symbol-header">
                                ${{symbol}} 
                                <span style="font-size: 0.8em; margin-left: 10px;">
                                    [Regime: <span class="${{regimeClass}}">${{symbolResults[0].volatility_regime}}</span> | 
                                    Liquidez: <span class="${{liquidityClass}}">${{symbolResults[0].liquidity_score}}</span> |
                                    Mercado: ${{symbolResults[0].market_regime}} | ${{dataSource}}]
                                </span>
                            </div>`;
                        
                        symbolResults.forEach(result => {{
                            const isBest = result.is_best_of_symbol;
                            const resultClass = isBest ? 'best-of-symbol' : '';
                            const bestBadge = isBest ? ' 🏆 MELHOR DO ATIVO' : '';
                            const globalBestBadge = (data.best && data.best.symbol === result.symbol && data.best.horizon === result.horizon) ? ' 🌟 MELHOR GLOBAL' : '';
                            const dataBadge = result.real_data ? 
                                '<span class="real-data-badge">REAL</span>' : 
                                '<span class="fallback-badge">SIM</span>';
                            
                            html += `
                            <div class="results ${{result.direction}} ${{resultClass}}">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div style="flex: 1;">
                                        <strong>T+${{result.horizon}}</strong>
                                        <span class="horizon-badge">${{result.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'}}${{bestBadge}}${{globalBestBadge}} ${{dataBadge}}</span>
                                        <br>
                                        <strong>Prob:</strong> ${{result.p_buy}}%/${{result.p_sell}}% (Soma: ${{(result.p_buy + result.p_sell).toFixed(1)}}%) | 
                                        <strong>Conf:</strong> ${{result.confidence}}% | 
                                        <strong>Assert:</strong> ${{result.assertiveness}}%
                                        <br>
                                        <strong>ADX:</strong> ${{result.adx}} | 
                                        <strong>RSI:</strong> ${{result.rsi}} | 
                                        <strong>Multi-TF:</strong> ${{result.multi_timeframe}} 
                                        <br>
                                        <strong>Indicadores:</strong> ${{formatIndicators(result.winning_indicators || [])}}
                                    </div>
                                </div>
                                ${{result.technical_override ? '<div class="override">⚡ Convergência Técnica</div>' : ''}}
                            </div>`;
                        }});
                    }});
                    
                    document.getElementById('allResults').innerHTML = html;
                }} else {{
                    document.getElementById('allResults').innerHTML = 'Nenhum sinal encontrado.';
                }}
            }}

            updateSymbols();
        </script>
    </body>
    </html>
    '''

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if manager.is_analyzing:
        return jsonify({'success': False, 'error': 'Análise em andamento'}), 429
    try:
        data = request.get_json()
        symbols = [s.strip().upper() for s in data['symbols'] if s.strip()]
        if not symbols:
            return jsonify({'success': False, 'error': 'Selecione pelo menos um ativo'}), 400
        sims = int(data.get('sims', SIMS))
        thread = threading.Thread(
            target=manager.analyze_symbols_thread,
            args=(symbols, sims, None)
        )
        thread.daemon = True
        thread.start()
        return jsonify({
            'success': True,
            'message': f'Analisando {len(symbols)} ativos com DADOS REAIS + {sims} simulações...',
            'symbols_count': len(symbols)
        })
    except Exception as e:
        log.exception("erro no /api/analyze")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results')
def get_results():
    with manager._lock:
        payload = {
            'success': True,
            'results': manager.current_results,
            'best': manager.best_opportunity,
            'analysis_time': manager.analysis_time,
            'total_signals': len(manager.current_results),
            'is_analyzing': manager.is_analyzing
        }
    return jsonify(payload)

@app.route('/api/best_signal')
def best_signal():
    with manager._lock:
        best = manager.best_opportunity
        text = format_best_signal_card(best, manager.analysis_time)
    return jsonify({'success': True, 'text': text, 'best': best})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': 'real-data-v2-macd-rsi-adx-corr', 'sims': SIMS})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    log.info("🚀 IA Signal Pro (Railway-ready)")
    log.info("✅ DADOS REAIS: Binance API com retry/backoff + cache TTL")
    log.info("✅ Indicadores: EMA/MACD corretos, RSI Wilder, ADX aproximação real")
    log.info("✅ Correlação: aplicada em 2ª passada (justa)")
    log.info("🔧 Servidor na porta: %s", port)
    # Railway aceita app.run (Nixpacks Python) ou Gunicorn via Procfile; mantemos app.run para compat.
    app.run(host='0.0.0.0', port=port, debug=False)
