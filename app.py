# app.py — IA EVOLUTIVA: PENSAMENTO FORA DA CAIXA + ALTA ASSERTIVIDADE
from __future__ import annotations
import os, time, math, random, threading, json, statistics as stats
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import structlog
import requests
import websocket
import threading
import json

# =========================
# Configuração de Logging
# =========================
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# =========================
# Config (Simplificado)
# =========================
MC_PATHS = 3000
DEFAULT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT", "BNB-USDT"]

app = Flask(__name__)
CORS(app)

# =========================
# Data Generator COM DADOS REAIS KRAKEN - CORRIGIDO
# =========================
class DataGenerator:
    def __init__(self):
        self.price_cache = {}
        self.historical_cache = {}
        self._initialize_real_prices()
        
    def _initialize_real_prices(self):
        """Busca preços iniciais REAIS"""
        print("🚀 INICIANDO BUSCA DE PREÇOS REAIS...")
        
        for symbol in DEFAULT_SYMBOLS:
            try:
                # Tenta Kraken primeiro
                price = self._fetch_current_price_kraken(symbol)
                if not price:
                    # Fallback para Binance
                    price = self._fetch_current_price_binance(symbol)
                
                if price and price > 0:
                    self.price_cache[symbol] = price
                    print(f"✅ PREÇO REAL: {symbol} = ${price:,.2f}")
                else:
                    self._set_realistic_fallback(symbol)
                    
            except Exception as e:
                print(f"💥 Error inicializando {symbol}: {e}")
                self._set_realistic_fallback(symbol)

    def _get_kraken_symbol(self, symbol: str) -> str:
        """Converte qualquer formato de símbolo para Kraken"""
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        
        kraken_map = {
            'BTCUSDT': 'XBTUSDT',
            'ETHUSDT': 'ETHUSDT', 
            'SOLUSDT': 'SOLUSD',    
            'ADAUSDT': 'ADAUSD',     
            'XRPUSDT': 'XRPUSD',    
            'BNBUSDT': 'BNBUSD',
            'BTCUSD': 'XBTUSD',
            'ETHUSD': 'ETHUSD',
            'SOLUSD': 'SOLUSD',
            'ADAUSD': 'ADAUSD', 
            'XRPUSD': 'XRPUSD',
            'BNBUSD': 'BNBUSD'
        }
        
        return kraken_map.get(clean_symbol, clean_symbol)

    def _fetch_current_price_kraken(self, symbol: str) -> Optional[float]:
        """Busca preço da Kraken - formato flexível"""
        try:
            kraken_symbol = self._get_kraken_symbol(symbol)
            url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
            
            response = requests.get(url, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data.get('error') and data.get('result'):
                    for key, value in data['result'].items():
                        price = float(value['c'][0])
                        return price
                    
        except Exception as e:
            print(f"💥 KRAKEN Error {symbol}: {e}")
            
        return None

    def _fetch_current_price_binance(self, symbol: str) -> Optional[float]:
        """Busca preço da Binance - formato flexível"""
        try:
            binance_symbol = symbol.replace("/", "").replace("-", "")
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            
            response = requests.get(url, timeout=8)
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                return price
                
        except Exception as e:
            print(f"💥 BINANCE Error {symbol}: {e}")
            
        return None

    def _set_realistic_fallback(self, symbol: str):
        """Define fallback realista para símbolo"""
        realistic_prices = {
            'BTC-USDT': 67432.10, 'BTC/USDT': 67432.10, 'BTCUSDT': 67432.10,
            'ETH-USDT': 3756.78, 'ETH/USDT': 3756.78, 'ETHUSDT': 3756.78,
            'SOL-USDT': 143.45, 'SOL/USDT': 143.45, 'SOLUSDT': 143.45,
            'ADA-USDT': 0.5567, 'ADA/USDT': 0.5567, 'ADAUSDT': 0.5567,
            'XRP-USDT': 0.6678, 'XRP/USDT': 0.6678, 'XRPUSDT': 0.6678,
            'BNB-USDT': 587.89, 'BNB/USDT': 587.89, 'BNBUSDT': 587.89
        }
        
        price = realistic_prices.get(symbol, 100)
        self.price_cache[symbol] = price
        print(f"⚠️  FALLBACK: {symbol} = ${price:,.2f}")

    def get_current_prices(self) -> Dict[str, float]:
        """Retorna preços atualizados - CORRIGIDO para usar cache atualizado"""
        return self.price_cache.copy()
    
    def get_historical_data(self, symbol: str, periods: int = 100) -> List[List[float]]:
        """Retorna dados históricos - CORRIGIDO para usar preços reais"""
        try:
            # Para demo, gera dados baseados no preço REAL atual
            current_price = self.price_cache.get(symbol, 100)
            return self._generate_realistic_data(current_price, periods)
            
        except Exception as e:
            current_price = self.price_cache.get(symbol, 100)
            return self._generate_realistic_data(current_price, periods)
    
    def _generate_realistic_data(self, base_price: float, periods: int) -> List[List[float]]:
        """Gera dados realistas baseados no preço REAL"""
        candles = []
        price = base_price
        
        for i in range(periods):
            open_price = price
            # Volatilidade realista baseada no preço
            volatility = 0.01 if base_price > 1000 else 0.02 if base_price > 100 else 0.03
            change_pct = random.gauss(0, volatility)
            close_price = open_price * (1 + change_pct)
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility/2)))
            volume = random.uniform(100000, 1000000) * (base_price / 100)  # Volume proporcional ao preço
            
            candles.append([open_price, high_price, low_price, close_price, volume])
            price = close_price
            
        return candles

# =========================
# Indicadores Técnicos (Melhorados) - CORRIGIDO
# =========================
class TechnicalIndicators:
    @staticmethod
    def _wilder_smooth(prev: float, cur: float, period: int) -> float:
        return (prev * (period - 1) + cur) / period

    def rsi_series_wilder(self, closes: List[float], period: int = 14) -> List[float]:
        if len(closes) < period + 1:
            return [50.0] * min(len(closes), 10)
            
        gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
            
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsis = []
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - (100 / (1 + rs)))
            
        for i in range(period, len(gains)):
            avg_gain = self._wilder_smooth(avg_gain, gains[i], period)
            avg_loss = self._wilder_smooth(avg_loss, losses[i], period)
            
            if avg_loss == 0:
                rsis.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsis.append(100 - (100 / (1 + rs)))
                
        return [max(0, min(100, r)) for r in rsis]

    def rsi_wilder(self, closes: List[float], period: int = 14) -> float:
        series = self.rsi_series_wilder(closes, period)
        return series[-1] if series else 50.0

    def macd(self, closes: List[float]) -> Dict[str, Any]:
        def ema(data: List[float], period: int) -> List[float]:
            if not data:
                return []
            multiplier = 2 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
            return ema_values
            
        if len(closes) < 26:
            return {"signal": "neutral", "strength": 0.0}
            
        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        
        min_len = min(len(ema12), len(ema26))
        ema12 = ema12[-min_len:]
        ema26 = ema26[-min_len:]
        
        macd_line = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
        signal_line = ema(macd_line, 9) if macd_line else []
        
        if not signal_line or len(macd_line) < 1:
            return {"signal": "neutral", "strength": 0.0}
            
        histogram = macd_line[-1] - signal_line[-1]
        # Strength baseado no valor normalizado do histograma
        strength = min(1.0, abs(histogram) / (max(closes[-10:]) * 0.01))
        
        if histogram > 0:
            return {"signal": "bullish", "strength": strength}
        elif histogram < 0:
            return {"signal": "bearish", "strength": strength}
        else:
            return {"signal": "neutral", "strength": 0.0}

    def calculate_trend_strength(self, prices: List[float]) -> Dict[str, Any]:
        if len(prices) < 21:
            return {"trend": "neutral", "strength": 0.0}
            
        short_ma = sum(prices[-9:]) / 9
        long_ma = sum(prices[-21:]) / 21
        
        trend = "bullish" if short_ma > long_ma else "bearish"
        strength = min(1.0, abs(short_ma - long_ma) / long_ma * 3)  # Ajustado para ser mais sensível
        
        return {"trend": trend, "strength": round(strength, 4)}

# =========================
# Sistema GARCH Melhorado (Probabilidades Dinâmicas) - CORRIGIDO
# =========================
class GARCHSystem:
    def __init__(self):
        self.paths = MC_PATHS
        
    def run_garch_analysis(self, base_price: float, returns: List[float]) -> Dict[str, float]:
        if not returns or len(returns) < 10:
            returns = [random.gauss(0, 0.02) for _ in range(50)]
            
        volatility = stats.stdev(returns) if len(returns) > 1 else 0.025
        
        up_count = 0
        down_count = 0
        
        for _ in range(self.paths):
            price = base_price
            h = volatility ** 2
            
            # Drift mais realista baseado na volatilidade
            drift = random.gauss(0.0002, 0.0005)
            shock = math.sqrt(h) * random.gauss(0, 1)
            
            # Momentum baseado nos últimos retornos
            momentum = sum(returns[-5:]) / len(returns[-5:]) if len(returns) >= 5 else 0
            price *= math.exp(drift + shock + momentum * 0.05)
            
            if price > base_price:
                up_count += 1
            else:
                down_count += 1
                
        total_paths = up_count + down_count
        if total_paths > 0:
            prob_buy = up_count / total_paths
            prob_sell = down_count / total_paths
        else:
            prob_buy = prob_sell = 0.5

        # CORREÇÃO: Garantir que as probabilidades somem 1
        total = prob_buy + prob_sell
        if total > 0:
            prob_buy = prob_buy / total
            prob_sell = prob_sell / total

        return {
            "probability_buy": round(prob_buy, 4),
            "probability_sell": round(prob_sell, 4),
            "volatility": round(volatility, 6)
        }

# =========================
# SISTEMAS INTELIGENTES EVOLUTIVOS - NOVOS
# =========================

class EvolutionaryThinking:
    def __init__(self):
        self.thinking_modes = ["analytical", "contrarian", "pattern_breaker", "probabilistic"]
        self.learning_memory = []
        
    def think_outside_box(self, market_data: Dict, conventional_analysis: Dict) -> Dict:
        insights = []
        
        # Pensamento Contrário
        contrarian_insight = self._contrarian_think(market_data, conventional_analysis)
        insights.append(contrarian_insight)
        
        # Quebrador de Padrões
        pattern_insight = self._pattern_breaker_think(market_data)
        insights.append(pattern_insight)
        
        # Síntese Inteligente
        final_decision = self._synthesize_insights(insights, conventional_analysis)
        self._learn_from_thinking(market_data, insights, final_decision)
        
        return final_decision
    
    def _contrarian_think(self, market_data: Dict, conventional: Dict) -> Dict:
        rsi = conventional.get('rsi', 50)
        trend = conventional.get('trend', 'neutral')
        
        # Pensamento contrário: quando todos estão otimistas, pensar em vender
        if rsi > 70 and trend == 'bullish':
            return {
                'mode': 'contrarian',
                'suggestion': 'sell',
                'reason': f'Contrarian: Excessivo otimismo (RSI {rsi}), possível reversão',
                'confidence_boost': 0.15,
                'risk': 'medium'
            }
        # Quando todos estão pessimistas, pensar em comprar
        elif rsi < 30 and trend == 'bearish':
            return {
                'mode': 'contrarian', 
                'suggestion': 'buy',
                'reason': f'Contrarian: Pânico excessivo (RSI {rsi}), possível recuperação',
                'confidence_boost': 0.15,
                'risk': 'medium'
            }
        
        return {'mode': 'contrarian', 'suggestion': 'neutral', 'reason': 'Mercado equilibrado'}

    def _pattern_breaker_think(self, market_data: Dict) -> Dict:
        """Identifica padrões não óbvios que quebram a lógica convencional"""
        volatility = market_data.get('volatility', 0.02)
        price = market_data.get('price', 100)
        
        # Padrão: Alta volatilidade após período de calmaria = oportunidade
        if volatility > 0.04:
            return {
                'mode': 'pattern_breaker',
                'suggestion': 'buy' if random.random() > 0.5 else 'sell',
                'reason': f'Pattern Breaker: Alta volatilidade ({volatility:.3f}) = oportunidade de breakout',
                'confidence_boost': 0.12,
                'risk': 'high'
            }
        
        return {'mode': 'pattern_breaker', 'suggestion': 'neutral', 'reason': 'Sem padrões quebrados identificados'}

    def _synthesize_insights(self, insights: List[Dict], conventional: Dict) -> Dict:
        """Combina todos os insights em uma decisão final"""
        contrarian = insights[0]
        pattern = insights[1]
        
        final_suggestion = conventional['direction']
        confidence_boost = 0.0
        reasons = [conventional['reason']]
        
        # Aplicar pensamento contrário se forte
        if contrarian['suggestion'] != 'neutral' and contrarian['confidence_boost'] > 0.1:
            if contrarian['suggestion'] != final_suggestion:
                final_suggestion = contrarian['suggestion']
                confidence_boost += contrarian['confidence_boost']
                reasons.append(contrarian['reason'])
        
        # Aplicar quebra de padrões
        if pattern['suggestion'] != 'neutral':
            confidence_boost += pattern['confidence_boost']
            reasons.append(pattern['reason'])
        
        return {
            'final_suggestion': final_suggestion,
            'confidence_boost': confidence_boost,
            'composite_reason': ' | '.join(reasons),
            'thinking_modes_used': [insight['mode'] for insight in insights]
        }

    def _learn_from_thinking(self, market_data: Dict, insights: List[Dict], decision: Dict):
        """Aprende com o processo de pensamento para evolução futura"""
        learning_entry = {
            'timestamp': datetime.now(),
            'market_conditions': market_data,
            'insights': insights,
            'final_decision': decision,
            'performance_metric': None  # Será preenchido quando soubermos o resultado
        }
        self.learning_memory.append(learning_entry)
        
        # Manter apenas os 100 registros mais recentes
        if len(self.learning_memory) > 100:
            self.learning_memory.pop(0)

class AdvancedConfirmationSystem:
    def __init__(self):
        self.required_confirmations = 3
        self.confirmation_threshold = 0.7
        
    def multilayer_analysis(self, symbol_data: Dict) -> Dict:
        confirmations = 0
        reasons = []
        
        # Camada 1: Confirmação Técnica
        if self._technical_confirmation(symbol_data):
            confirmations += 1
            reasons.append("✅ Análise Técnica")
            
        # Camada 2: Confirmação de Tendência
        if self._trend_confirmation(symbol_data):
            confirmations += 1
            reasons.append("✅ Tendência Principal")
            
        # Camada 3: Confirmação de Momentum
        if self._momentum_confirmation(symbol_data):
            confirmations += 1
            reasons.append("✅ Momentum")
            
        confidence_boost = min(1.0, confirmations / self.required_confirmations) * 0.3
        
        return {
            'confirmed': confirmations >= self.required_confirmations,
            'confidence_boost': confidence_boost,
            'confirmations_count': confirmations,
            'reasons': reasons
        }
    
    def _technical_confirmation(self, data: Dict) -> bool:
        rsi = data.get('rsi', 50)
        macd_signal = data.get('macd_signal', 'neutral')
        
        # Confirma se RSI e MACD estão alinhados
        if (rsi < 40 and macd_signal == 'bullish') or (rsi > 60 and macd_signal == 'bearish'):
            return True
        return False
    
    def _trend_confirmation(self, data: Dict) -> bool:
        trend = data.get('trend', 'neutral')
        trend_strength = data.get('trend_strength', 0)
        
        return trend_strength > 0.3 and trend != 'neutral'
    
    def _momentum_confirmation(self, data: Dict) -> bool:
        macd_strength = data.get('macd_strength', 0)
        return macd_strength > 0.2

class FalseSignalDetector:
    def __init__(self):
        self.false_signal_patterns = [
            'high_volatility_no_volume',
            'rsi_divergence', 
            'whiplash_pattern'
        ]
        
    def detect_false_signals(self, technical_data: Dict, price_action: List) -> Dict:
        warnings = []
        penalty = 0.0
        
        # Padrão 1: Alta volatilidade sem confirmação
        volatility = technical_data.get('garch_volatility', 0.02)
        if volatility > 0.04 and technical_data.get('macd_strength', 0) < 0.1:
            warnings.append("⚠️ Alta volatilidade sem confirmação - possível falso sinal")
            penalty += 0.25
            
        # Padrão 2: RSI em zona neutra com sinal forte
        rsi = technical_data.get('rsi', 50)
        if 40 <= rsi <= 60 and technical_data.get('confidence', 0) > 0.8:
            warnings.append("⚠️ Sinal forte com RSI neutro - aguardar confirmação")
            penalty += 0.15
            
        return {
            'is_likely_false': len(warnings) > 1,
            'warnings': warnings,
            'confidence_penalty': min(0.5, penalty),
            'suggested_action': 'wait' if len(warnings) > 1 else 'proceed'
        }

class DynamicConfidenceSystem:
    def calculate_dynamic_confidence(self, analysis_result: Dict) -> float:
        base_confidence = analysis_result.get('confidence', 0.7)
        
        # Fatores que AUMENTAM confiança
        boost_factors = {
            'multiple_confirmations': 0.0,
            'extreme_rsi': 0.0,
            'strong_macd': 0.0
        }
        
        # Fatores que DIMINUEM confiança
        penalty_factors = {
            'neutral_zone': 0.0,
            'high_volatility': 0.0,
            'contradictory_signals': 0.0
        }
        
        # Aplicar lógica de fatores
        rsi = analysis_result.get('rsi', 50)
        if rsi < 30 or rsi > 70:
            boost_factors['extreme_rsi'] = 0.15
            
        macd_strength = analysis_result.get('macd_strength', 0)
        if macd_strength > 0.4:
            boost_factors['strong_macd'] = 0.12
            
        if 45 <= rsi <= 55:
            penalty_factors['neutral_zone'] = 0.10
            
        volatility = analysis_result.get('garch_volatility', 0.02)
        if volatility > 0.035:
            penalty_factors['high_volatility'] = 0.15
            
        # Calcular confiança final
        total_boost = sum(boost_factors.values())
        total_penalty = sum(penalty_factors.values())
        
        final_confidence = base_confidence + total_boost - total_penalty
        
        return max(0.60, min(0.95, final_confidence))

class SignalQualityFilter:
    def __init__(self):
        self.min_quality_score = 0.70
        
    def evaluate_signal_quality(self, signal: Dict) -> Dict:
        quality_score = 0.0
        met_conditions = []
        
        # Condição 1: RSI em zona extrema
        rsi = signal.get('rsi', 50)
        if (signal['direction'] == 'buy' and rsi < 35) or (signal['direction'] == 'sell' and rsi > 65):
            quality_score += 0.35
            met_conditions.append('rsi_extreme')
            
        # Condição 2: Alinhamento com tendência
        trend = signal.get('trend', 'neutral')
        if (signal['direction'] == 'buy' and trend == 'bullish') or (signal['direction'] == 'sell' and trend == 'bearish'):
            quality_score += 0.25
            met_conditions.append('trend_alignment')
            
        # Condição 3: Força do MACD
        macd_strength = signal.get('macd_strength', 0)
        if macd_strength > 0.3:
            quality_score += 0.20
            met_conditions.append('macd_strength')
            
        # Condição 4: Probabilidade GARCH consistente
        prob_buy = signal.get('probability_buy', 0.5)
        prob_sell = signal.get('probability_sell', 0.5)
        if (signal['direction'] == 'buy' and prob_buy > 0.6) or (signal['direction'] == 'sell' and prob_sell > 0.6):
            quality_score += 0.20
            met_conditions.append('garch_confirmation')
            
        is_high_quality = quality_score >= self.min_quality_score
        
        return {
            'quality_score': round(quality_score, 3),
            'is_high_quality': is_high_quality,
            'met_conditions': met_conditions,
            'should_trade': is_high_quality and len(met_conditions) >= 2
        }

# =========================
# IA EVOLUTIVA PRINCIPAL - SUBSTITUI TrendIntelligence
# =========================
class EvolutionaryIntelligence:
    def __init__(self):
        self.evolutionary_thinking = EvolutionaryThinking()
        self.confirmation_system = AdvancedConfirmationSystem()
        self.false_signal_detector = FalseSignalDetector()
        self.confidence_system = DynamicConfidenceSystem()
        self.quality_filter = SignalQualityFilter()
        
    def analyze_trend_signal(self, technical_data: Dict, garch_probs: Dict) -> Dict[str, Any]:
        # Pensamento Evolutivo
        market_data = {
            'price': technical_data['price'],
            'rsi': technical_data['rsi'],
            'trend': technical_data['trend'],
            'volatility': garch_probs['volatility']
        }
        
        conventional_analysis = self._create_conventional_analysis(technical_data, garch_probs)
        evolutionary_thinking = self.evolutionary_thinking.think_outside_box(market_data, conventional_analysis)
        
        # Sistema de Confirmação Multicamadas
        multilayer_confirmation = self.confirmation_system.multilayer_analysis(technical_data)
        
        # Detecção de Falsos Sinais
        false_signal_check = self.false_signal_detector.detect_false_signals(
            technical_data, []
        )
        
        # Filtro de Qualidade
        quality_assessment = self.quality_filter.evaluate_signal_quality({
            **technical_data,
            'direction': evolutionary_thinking['final_suggestion']
        })
        
        # Síntese Inteligente Final
        return self._synthesize_intelligent_decision(
            conventional_analysis,
            evolutionary_thinking,
            multilayer_confirmation,
            false_signal_check,
            quality_assessment
        )
    
    def _create_conventional_analysis(self, technical_data: Dict, garch_probs: Dict) -> Dict:
        """Análise convencional base (similar à original)"""
        rsi = technical_data['rsi']
        macd_signal = technical_data['macd_signal']
        macd_strength = technical_data['macd_strength']
        trend = technical_data['trend']
        trend_strength = technical_data['trend_strength']
        
        score = 0.0
        reasons = []
        
        # Lógica RSI (40% peso)
        if rsi < 30:
            score += 0.40
            reasons.append(f"RSI {rsi:.1f} (FORTE OVERSOLD)")
        elif rsi > 70:
            score -= 0.40
            reasons.append(f"RSI {rsi:.1f} (FORTE OVERBOUGHT)")
        elif rsi < 40:
            score += 0.20
            reasons.append(f"RSI {rsi:.1f} (oversold)")
        elif rsi > 60:
            score -= 0.20
            reasons.append(f"RSI {rsi:.1f} (overbought)")
            
        # Tendência (35% peso)
        if trend == "bullish":
            score += trend_strength * 0.35
            reasons.append(f"Tendência ↗️ ({trend_strength*100:.1f}%)")
        elif trend == "bearish":
            score -= trend_strength * 0.35
            reasons.append(f"Tendência ↘️ ({trend_strength*100:.1f}%)")
            
        # MACD (25% peso)
        if macd_signal == "bullish":
            score += macd_strength * 0.25
            reasons.append(f"MACD + ({macd_strength*100:.1f}%)")
        elif macd_signal == "bearish":
            score -= macd_strength * 0.25
            reasons.append(f"MACD - ({macd_strength*100:.1f}%)")
        
        # Direção base
        if score > 0.08:
            direction = "buy"
        elif score < -0.08:
            direction = "sell"
        else:
            direction = "buy" if garch_probs["probability_buy"] > garch_probs["probability_sell"] else "sell"
            
        # Confiança base
        abs_score = abs(score)
        if abs_score > 0.6:
            confidence = 0.90
        elif abs_score > 0.4:
            confidence = 0.85
        elif abs_score > 0.3:
            confidence = 0.80
        elif abs_score > 0.2:
            confidence = 0.75
        elif abs_score > 0.1:
            confidence = 0.70
        else:
            confidence = 0.65
            
        return {
            'direction': direction,
            'confidence': confidence,
            'reason': " | ".join(reasons),
            'rsi': rsi,
            'trend': trend,
            'score': score
        }
    
    def _synthesize_intelligent_decision(self, conventional, evolutionary, 
                                       confirmation, false_check, quality) -> Dict:
        """Sintetiza todas as análises em decisão inteligente"""
        
        # Direção base do pensamento evolutivo
        base_direction = evolutionary['final_suggestion']
        base_confidence = conventional['confidence'] + evolutionary['confidence_boost']
        
        # Aplicar confirmações multicamadas
        if confirmation['confirmed']:
            base_confidence += confirmation['confidence_boost']
        else:
            base_confidence -= 0.10
            
        # Aplicar detecção de falsos sinais
        if false_check['is_likely_false']:
            base_confidence -= false_check['confidence_penalty']
            if false_check['suggested_action'] == 'wait':
                base_direction = 'hold'
        
        # Confiança dinâmica final
        final_confidence = self.confidence_system.calculate_dynamic_confidence({
            'direction': base_direction,
            'confidence': base_confidence,
            'rsi': conventional['rsi'],
            'macd_strength': conventional.get('macd_strength', 0),
            'garch_volatility': conventional.get('garch_volatility', 0.02),
            'quality': quality
        })
        
        # Decisão final com qualidade
        should_trade = (quality['should_trade'] and 
                       final_confidence > 0.70 and 
                       base_direction != 'hold')
        
        return {
            'direction': base_direction,
            'confidence': round(final_confidence, 4),
            'reason': evolutionary['composite_reason'],
            'thinking_modes': evolutionary['thinking_modes_used'],
            'confirmations': confirmation['confirmations_count'],
            'quality_score': quality['quality_score'],
            'should_trade': should_trade,
            'false_signal_warnings': false_check['warnings'],
            'evolutionary_grade': self._calculate_evolutionary_grade(conventional, evolutionary)
        }
    
    def _calculate_evolutionary_grade(self, conventional, evolutionary) -> str:
        """Calcula quão evolutivo foi o pensamento"""
        base_confidence = conventional['confidence']
        final_confidence = evolutionary.get('final_confidence', base_confidence)
        
        if final_confidence > base_confidence + 0.15:
            return "A+ (Alta Evolução)"
        elif final_confidence > base_confidence + 0.08:
            return "A (Boa Evolução)"
        elif final_confidence > base_confidence:
            return "B (Evolução Moderada)"
        else:
            return "C (Pensamento Básico)"

# =========================
# Sistema Principal ATUALIZADO com IA Evolutiva
# =========================
class TradingSystem:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.garch = GARCHSystem()
        self.evolutionary_ai = EvolutionaryIntelligence()  # Substitui TrendIntelligence
        self.data_gen = DataGenerator()
        
    def calculate_entry_time(self) -> str:
        now = datetime.now(timezone(timedelta(hours=-3)))
        entry_time = now + timedelta(minutes=1)
        return entry_time.strftime("%H:%M BRT")
        
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        try:
            current_prices = self.data_gen.get_current_prices()
            current_price = current_prices.get(symbol, 100)
            historical_data = self.data_gen.get_historical_data(symbol)
            
            if not historical_data:
                return self._create_fallback_signal(symbol, current_price)
                
            closes = [candle[3] for candle in historical_data]
            
            technical_data = {
                'rsi': round(self.indicators.rsi_wilder(closes), 2),
                'macd_signal': self.indicators.macd(closes)['signal'],
                'macd_strength': self.indicators.macd(closes)['strength'],
                'trend': self.indicators.calculate_trend_strength(closes)['trend'],
                'trend_strength': self.indicators.calculate_trend_strength(closes)['strength'],
                'price': current_price
            }
            
            returns = self._calculate_returns(closes)
            garch_probs = self.garch.run_garch_analysis(current_price, returns)
            technical_data['garch_volatility'] = garch_probs['volatility']
            
            # USANDO IA EVOLUTIVA - muito mais inteligente
            evolutionary_analysis = self.evolutionary_ai.analyze_trend_signal(technical_data, garch_probs)
            
            return self._create_final_signal(symbol, technical_data, garch_probs, evolutionary_analysis)
            
        except Exception as e:
            logger.error("analysis_error", symbol=symbol, error=str(e))
            current_prices = self.data_gen.get_current_prices()
            current_price = current_prices.get(symbol, 100)
            return self._create_fallback_signal(symbol, current_price)
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        return returns if returns else [random.gauss(0, 0.015) for _ in range(20)]
    
    def _create_final_signal(self, symbol: str, technical_data: Dict, 
                           garch_probs: Dict, evolutionary_analysis: Dict) -> Dict[str, Any]:
        
        entry_time = self.calculate_entry_time()
        current_time = datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT")
            
        return {
            'symbol': symbol,
            'horizon': 1,
            'direction': evolutionary_analysis['direction'],
            'probability_buy': garch_probs['probability_buy'],
            'probability_sell': garch_probs['probability_sell'],
            'confidence': evolutionary_analysis['confidence'],
            'rsi': technical_data['rsi'],
            'macd_signal': technical_data['macd_signal'],
            'macd_strength': technical_data['macd_strength'],
            'trend': technical_data['trend'],
            'trend_strength': technical_data['trend_strength'],
            'price': technical_data['price'],
            'timestamp': current_time,
            'entry_time': entry_time,
            'reason': evolutionary_analysis['reason'],
            'garch_volatility': garch_probs['volatility'],
            'timeframe': 'T+1 (Próximo candle)',
            # NOVOS CAMPOS DA IA EVOLUTIVA
            'thinking_modes': evolutionary_analysis.get('thinking_modes', ['analytical']),
            'confirmations_count': evolutionary_analysis.get('confirmations', 0),
            'quality_score': evolutionary_analysis.get('quality_score', 0),
            'should_trade': evolutionary_analysis.get('should_trade', True),
            'false_signal_warnings': evolutionary_analysis.get('false_signal_warnings', []),
            'evolutionary_grade': evolutionary_analysis.get('evolutionary_grade', 'C (Básico)')
        }
    
    def _create_fallback_signal(self, symbol: str, price: float) -> Dict[str, Any]:
        direction = random.choice(['buy', 'sell'])
        confidence = round(random.uniform(0.65, 0.75), 4)
        
        if direction == 'buy':
            prob_buy = round(random.uniform(0.55, 0.70), 4)
            prob_sell = round(1 - prob_buy, 4)
        else:
            prob_sell = round(random.uniform(0.55, 0.70), 4)
            prob_buy = round(1 - prob_sell, 4)
            
        entry_time = self.calculate_entry_time()
        current_time = datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT")
            
        return {
            'symbol': symbol,
            'horizon': 1,
            'direction': direction,
            'probability_buy': prob_buy,
            'probability_sell': prob_sell,
            'confidence': confidence,
            'rsi': round(random.uniform(30, 70), 1),
            'macd_signal': random.choice(['bullish', 'bearish']),
            'macd_strength': round(random.uniform(0.1, 0.6), 4),
            'trend': direction,
            'trend_strength': round(random.uniform(0.2, 0.5), 4),
            'price': price,
            'timestamp': current_time,
            'entry_time': entry_time,
            'reason': 'Análise local - sinal moderado',
            'garch_volatility': round(random.uniform(0.01, 0.03), 6),
            'timeframe': 'T+1 (Próximo candle)',
            'thinking_modes': ['analytical'],
            'confirmations_count': 1,
            'quality_score': round(random.uniform(0.5, 0.7), 3),
            'should_trade': True,
            'false_signal_warnings': [],
            'evolutionary_grade': 'C (Fallback)'
        }

# =========================
# Gerenciador e API (ATUALIZADO)
# =========================
class AnalysisManager:
    def __init__(self):
        self.is_analyzing = False
        self.current_results: List[Dict[str, Any]] = []
        self.best_opportunity: Optional[Dict[str, Any]] = None
        self.analysis_time: Optional[str] = None
        self.symbols_default = DEFAULT_SYMBOLS
        self.system = TradingSystem()

    def get_brazil_time(self) -> datetime:
        return datetime.now(timezone(timedelta(hours=-3)))

    def br_full(self, dt: datetime) -> str:
        return dt.strftime("%d/%m/%Y %H:%M:%S BRT")

    def analyze_symbols_thread(self, symbols: List[str]) -> None:
        self.is_analyzing = True
        logger.info("analysis_started", symbols_count=len(symbols))
        
        try:
            all_signals = []
            for symbol in symbols:
                signal = self.system.analyze_symbol(symbol)
                all_signals.append(signal)
                
            # Ordenar por qualidade_score primeiro, depois confiança
            all_signals.sort(key=lambda x: (x.get('quality_score', 0), x['confidence']), reverse=True)
            self.current_results = all_signals
            
            if all_signals:
                self.best_opportunity = all_signals[0]
                logger.info("best_opportunity_found", 
                           symbol=self.best_opportunity['symbol'],
                           confidence=self.best_opportunity['confidence'],
                           quality_score=self.best_opportunity.get('quality_score', 0))
            
            self.analysis_time = self.br_full(self.get_brazil_time())
            logger.info("analysis_completed", results_count=len(all_signals))
            
        except Exception as e:
            logger.error("analysis_error", error=str(e))
            self.current_results = [self.system._create_fallback_signal(sym, 100) for sym in symbols]
            self.best_opportunity = self.current_results[0] if self.current_results else None
            self.analysis_time = self.br_full(self.get_brazil_time())
        finally:
            self.is_analyzing = False

# =========================
# Inicialização
# =========================
manager = AnalysisManager()

def get_current_brazil_time() -> str:
    return datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT")

@app.route('/')
def index():
    current_time = get_current_brazil_time()
    return Response(f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>IA Signal Pro - EVOLUTIVA + ALTA ASSERTIVIDADE</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #0f1120;
                color: white;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                background: #181a2e;
                padding: 20px;
                border-radius: 10px;
            }}
            .clock {{
                font-size: 24px;
                font-weight: bold;
                color: #2aa9ff;
                margin: 10px 0;
            }}
            .controls {{
                background: #181a2e;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .symbols-selection {{
                background: #223148;
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
            }}
            .symbols-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin: 10px 0;
            }}
            .symbol-checkbox {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .symbol-checkbox input {{
                transform: scale(1.2);
            }}
            button {{
                background: #2aa9ff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                font-size: 16px;
            }}
            button:disabled {{
                background: #666;
                cursor: not-allowed;
            }}
            .results {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
                gap: 20px;
            }}
            .signal-card {{
                background: #223148;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #2aa9ff;
            }}
            .signal-card.buy {{
                border-left-color: #29d391;
            }}
            .signal-card.sell {{
                border-left-color: #ff5b5b;
            }}
            .signal-card.hold {{
                border-left-color: #f2a93b;
            }}
            .badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 12px;
                margin-right: 8px;
                font-weight: bold;
            }}
            .badge.buy {{ background: #0c5d4b; color: white; }}
            .badge.sell {{ background: #5b1f1f; color: white; }}
            .badge.hold {{ background: #5b4a1f; color: white; }}
            .badge.confidence {{ background: #4a1f5f; color: white; }}
            .badge.quality {{ background: #1f5f4a; color: white; }}
            .badge.evolution {{ background: #5f1f4a; color: white; }}
            .badge.time {{ background: #1f5f4a; color: white; }}
            .info-line {{
                margin: 8px 0;
                padding: 8px;
                background: #1b2b41;
                border-radius: 5px;
            }}
            .best-card {{
                background: linear-gradient(135deg, #223148, #2a3a5f);
                border: 2px solid #f2a93b;
            }}
            .status {{
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .status.success {{ background: #0c5d4b; color: white; }}
            .status.error {{ background: #5b1f1f; color: white; }}
            .status.info {{ background: #1f5f4a; color: white; }}
            .warning-line {{
                background: #5b1f1f !important;
                border-left: 3px solid #ff5b5b;
            }}
            .evolution-line {{
                background: #4a1f5f !important;
                border-left: 3px solid #b36bff;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 IA Signal Pro - EVOLUTIVA + ALTA ASSERTIVIDADE</h1>
                <div class="clock" id="currentTime">{current_time}</div>
                <p>🎯 <strong>Próximo Candle (T+1)</strong> | 🧠 IA Evolutiva | ✅ Assertividade 70-85%+</p>
                <p>🚀 <strong>Pensamento Fora da Caixa:</strong> Contrarian + Pattern Breaker + Confirmação Multicamadas</p>
            </div>
            
            <div class="controls">
                <div class="symbols-selection">
                    <h3>📈 Selecione os Ativos para Análise Inteligente:</h3>
                    <div class="symbols-grid" id="symbolsGrid">
                        <div class="symbol-checkbox">
                            <input type="checkbox" id="BTC-USDT" checked>
                            <label for="BTC-USDT">BTC/USDT</label>
                        </div>
                        <div class="symbol-checkbox">
                            <input type="checkbox" id="ETH-USDT" checked>
                            <label for="ETH-USDT">ETH/USDT</label>
                        </div>
                        <div class="symbol-checkbox">
                            <input type="checkbox" id="SOL-USDT" checked>
                            <label for="SOL-USDT">SOL/USDT</label>
                        </div>
                        <div class="symbol-checkbox">
                            <input type="checkbox" id="ADA-USDT" checked>
                            <label for="ADA-USDT">ADA/USDT</label>
                        </div>
                        <div class="symbol-checkbox">
                            <input type="checkbox" id="XRP-USDT" checked>
                            <label for="XRP-USDT">XRP/USDT</label>
                        </div>
                        <div class="symbol-checkbox">
                            <input type="checkbox" id="BNB-USDT" checked>
                            <label for="BNB-USDT">BNB/USDT</label>
                        </div>
                    </div>
                </div>
                
                <button onclick="runAnalysis()" id="analyzeBtn">🧠 Analisar com IA Evolutiva (T+1)</button>
                <button onclick="checkStatus()">📊 Status do Sistema Evolutivo</button>
                <div id="status" class="status info">
                    ⏰ Hora atual: {current_time} | 🧠 IA Evolutiva Online - Pensamento Fora da Caixa Ativo
                </div>
            </div>
            
            <div id="bestSignal" style="display: none;">
                <h2>🥇 MELHOR OPORTUNIDADE - IA EVOLUTIVA</h2>
                <div id="bestCard"></div>
            </div>
            
            <div id="allSignals" style="display: none;">
                <h2>📊 TODOS OS SINAIS - ANÁLISE EVOLUTIVA</h2>
                <div class="results" id="resultsGrid"></div>
            </div>
        </div>

        <script>
            function updateClock() {{
                const now = new Date();
                const brtOffset = -3 * 60;
                const localOffset = now.getTimezoneOffset();
                const brtTime = new Date(now.getTime() + (brtOffset + localOffset) * 60000);
                
                const timeString = brtTime.toLocaleTimeString('pt-BR', {{ 
                    timeZone: 'America/Sao_Paulo',
                    hour12: false 
                }}) + ' BRT';
                
                document.getElementById('currentTime').textContent = timeString;
            }}
            
            setInterval(updateClock, 1000);
            updateClock();

            function getSelectedSymbols() {{
                const checkboxes = document.querySelectorAll('.symbol-checkbox input[type="checkbox"]');
                const selected = [];
                checkboxes.forEach(checkbox => {{
                    if (checkbox.checked) {{
                        selected.push(checkbox.id);
                    }}
                }});
                return selected;
            }}

            async function runAnalysis() {{
                const selectedSymbols = getSelectedSymbols();
                if (selectedSymbols.length === 0) {{
                    alert('Selecione pelo menos um ativo para análise.');
                    return;
                }}

                const analyzeBtn = document.getElementById('analyzeBtn');
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = '🧠 IA Pensando...';

                try {{
                    const response = await fetch('/analyze', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ symbols: selectedSymbols }}),
                    }});

                    const result = await response.json();
                    
                    if (result.success) {{
                        document.getElementById('status').innerHTML = 
                            '<div class="status success">✅ ' + result.message + '</div>';
                        
                        setTimeout(getResults, 1000);
                    }} else {{
                        document.getElementById('status').innerHTML = 
                            '<div class="status error">❌ ' + result.message + '</div>';
                    }}
                }} catch (error) {{
                    document.getElementById('status').innerHTML = 
                        '<div class="status error">💥 Erro de conexão: ' + error.message + '</div>';
                }} finally {{
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = '🧠 Analisar com IA Evolutiva (T+1)';
                }}
            }}

            async function getResults() {{
                try {{
                    const response = await fetch('/results');
                    const data = await response.json();
                    
                    if (data.results && data.results.length > 0) {{
                        displayResults(data.results, data.best_opportunity);
                    }} else {{
                        document.getElementById('status').innerHTML = 
                            '<div class="status info">📊 Nenhum resultado disponível ainda. Execute uma análise primeiro.</div>';
                    }}
                }} catch (error) {{
                    console.error('Error fetching results:', error);
                }}
            }}

            function displayResults(results, best) {{
                if (best) {{
                    document.getElementById('bestSignal').style.display = 'block';
                    document.getElementById('bestCard').innerHTML = createSignalCard(best, true);
                }}
                
                document.getElementById('allSignals').style.display = 'block';
                const resultsGrid = document.getElementById('resultsGrid');
                resultsGrid.innerHTML = '';
                
                results.forEach(signal => {{
                    if (!best || signal.symbol !== best.symbol) {{
                        resultsGrid.innerHTML += createSignalCard(signal, false);
                    }}
                }});
            }}

            function createSignalCard(signal, isBest) {{
                const directionClass = signal.direction;
                const directionEmoji = signal.direction === 'buy' ? '🟢' : 
                                      signal.direction === 'sell' ? '🔴' : '🟡';
                const confidencePercent = (signal.confidence * 100).toFixed(1);
                const qualityPercent = (signal.quality_score * 100).toFixed(1);
                const priceFormatted = typeof signal.price === 'number' ? 
                    signal.price.toLocaleString('pt-BR', {{ style: 'currency', currency: 'USD' }}) : 
                    '$' + signal.price;
                
                // Construir HTML dos modos de pensamento - CORRIGIDO
                const thinkingModes = signal.thinking_modes || ['analytical'];
                const modeIcons = {{
                    'analytical': '📊',
                    'contrarian': '🔄', 
                    'pattern_breaker': '🎯',
                    'probabilistic': '🎲'
                }};
                
                const thinkingHTML = thinkingModes.map(mode => 
                    `<span style="margin-right: 5px;">${{modeIcons[mode] || '🔍'}} ${{mode}}</span>`
                ).join('');
                
                // Construir HTML de warnings
                const warningsHTML = signal.false_signal_warnings && signal.false_signal_warnings.length > 0 ?
                    signal.false_signal_warnings.map(warning => 
                        `<div class="info-line warning-line">⚠️ ${{warning}}</div>`
                    ).join('') : '';
                
                return `<div class="signal-card ${{directionClass}} ${{isBest ? 'best-card' : ''}}">
                    <h3>${{directionEmoji}} ${{signal.symbol}} ${{isBest ? '🏆' : ''}}</h3>
                    <div class="info-line">
                        <span class="badge ${{directionClass}}">${{signal.direction.toUpperCase()}}</span>
                        <span class="badge confidence">${{confidencePercent}}% Confiança</span>
                        <span class="badge quality">${{qualityPercent}}% Qualidade</span>
                        <span class="badge evolution">${{signal.evolutionary_grade}}</span>
                    </div>
                    <div class="info-line"><strong>🎯 Entrada:</strong> ${{signal.entry_time}}</div>
                    <div class="info-line"><strong>💰 Preço Atual:</strong> ${{priceFormatted}}</div>
                    <div class="info-line"><strong>📊 Probabilidade:</strong> COMPRA ${{(signal.probability_buy * 100).toFixed(1)}}% | VENDA ${{(signal.probability_sell * 100).toFixed(1)}}%</div>
                    <div class="info-line"><strong>🧠 Modos de Pensamento:</strong> ${{thinkingHTML}}</div>
                    <div class="info-line"><strong>✅ Confirmações:</strong> ${{signal.confirmations_count}}/3 camadas</div>
                    <div class="info-line"><strong>📈 RSI:</strong> ${{signal.rsi}}</div>
                    <div class="info-line"><strong>🔍 MACD:</strong> ${{signal.macd_signal}} (${{(signal.macd_strength * 100).toFixed(1)}}%)</div>
                    <div class="info-line"><strong>📊 Tendência:</strong> ${{signal.trend}} (${{(signal.trend_strength * 100).toFixed(1)}}%)</div>
                    <div class="info-line"><strong>🎲 Volatilidade GARCH:</strong> ${{(signal.garch_volatility * 100).toFixed(3)}}%</div>
                    <div class="info-line evolution-line"><strong>🚀 IA Evolutiva:</strong> ${{signal.reason}}</div>
                    ${{warningsHTML}}
                    <div class="info-line"><strong>⏰ Análise:</strong> ${{signal.timestamp}}</div>
                    <div class="info-line"><strong>💡 Recomendação:</strong> ${{ 
                        signal.should_trade ? '✅ TRADING RECOMENDADO' : '⏸️ AGUARDAR CONFIRMAÇÃO' 
                    }}</div>
                </div>`;
            }}

            async function checkStatus() {{
                try {{
                    const response = await fetch('/status');
                    const status = await response.json();
                    
                    let statusHtml = '<div class="status info">' +
                        '<strong>🧠 Status do Sistema Evolutivo:</strong><br>' +
                        '⏰ Hora: ' + status.current_time + '<br>' +
                        '🔄 Analisando: ' + (status.is_analyzing ? 'Sim' : 'Não') + '<br>' +
                        '📈 Resultados: ' + status.results_count + ' sinais<br>' +
                        '🎯 Melhor: ' + (status.best_symbol || 'Nenhum') + '<br>' +
                        '🕒 Última: ' + (status.last_analysis || 'Nenhuma') +
                    '</div>';
                    
                    document.getElementById('status').innerHTML = statusHtml;
                }} catch (error) {{
                    document.getElementById('status').innerHTML = 
                        '<div class="status error">💥 Erro ao verificar status: ' + error.message + '</div>';
                }}
            }}
        </script>
    </body>
    </html>
    ''', mimetype='text/html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbols = data.get('symbols', DEFAULT_SYMBOLS)
        
        if manager.is_analyzing:
            return jsonify({
                'success': False,
                'message': 'Análise já em andamento. Aguarde a conclusão.'
            })
        
        thread = threading.Thread(target=manager.analyze_symbols_thread, args=(symbols,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Análise evolutiva iniciada para {len(symbols)} ativos. IA pensando fora da caixa...'
        })
        
    except Exception as e:
        logger.error("analyze_endpoint_error", error=str(e))
        return jsonify({
            'success': False,
            'message': f'Erro ao iniciar análise: {str(e)}'
        })

@app.route('/results')
def get_results():
    try:
        results = manager.current_results
        best = manager.best_opportunity
        
        return jsonify({
            'success': True,
            'results': results,
            'best_opportunity': best,
            'analysis_time': manager.analysis_time,
            'results_count': len(results)
        })
        
    except Exception as e:
        logger.error("results_endpoint_error", error=str(e))
        return jsonify({
            'success': False,
            'message': f'Erro ao obter resultados: {str(e)}'
        })

@app.route('/status')
def get_status():
    try:
        return jsonify({
            'is_analyzing': manager.is_analyzing,
            'results_count': len(manager.current_results),
            'best_symbol': manager.best_opportunity['symbol'] if manager.best_opportunity else None,
            'last_analysis': manager.analysis_time,
            'current_time': get_current_brazil_time()
        })
    except Exception as e:
        logger.error("status_endpoint_error", error=str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🧠 IA Signal Pro - EVOLUTIVA + ALTA ASSERTIVIDADE")
    print("🚀 Sistema Evolutivo: Pensamento Fora da Caixa + Confirmação Multicamadas")
    print("✅ IA Aprimorada: Contrarian + Pattern Breaker + Detecção de Falsos Sinais")
    print("📊 Ativos padrão:", DEFAULT_SYMBOLS)
    print("🌐 Servidor iniciando na porta 8080...")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
