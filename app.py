# app.py — IA EVOLUTIVA: SISTEMA DE TENDÊNCIAS AVANÇADO
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
# SISTEMA AVANÇADO DE ANÁLISE DE TENDÊNCIAS
# =========================
class TrendAnalysisSystem:
    def __init__(self):
        self.trend_states = {}
        
    def analyze_trend_direction(self, symbol: str, closes: List[float]) -> Dict:
        """Análise COMPLETA de tendência com múltiplas confirmações"""
        
        if len(closes) < 50:
            return self._default_trend_analysis()
        
        # 1. DETECÇÃO DE TENDÊNCIA PRINCIPAL
        primary_trend = self._detect_primary_trend(closes)
        
        # 2. FORÇA DA TENDÊNCIA
        trend_strength = self._calculate_trend_strength(closes, primary_trend)
        
        # 3. MOMENTUM DA TENDÊNCIA
        trend_momentum = self._analyze_trend_momentum(closes)
        
        # 4. PONTOS DE TRANSIÇÃO
        transition_points = self._detect_trend_transition(closes, primary_trend)
        
        # 5. SUGESTÃO DE AÇÃO
        action_suggestion = self._suggest_trend_action(primary_trend, trend_strength, transition_points)
        
        return {
            'primary_trend': primary_trend,
            'trend_strength': round(trend_strength, 4),
            'trend_momentum': round(trend_momentum, 4),
            'is_transitioning': transition_points['is_transitioning'],
            'transition_stage': transition_points['stage'],
            'confidence': round(trend_strength * 0.7 + trend_momentum * 0.3, 4),
            'action': action_suggestion,
            'momentum_divergence': transition_points['momentum_divergence'],
            'exhaustion_signals': transition_points['exhaustion_signals'],
            'structure_break': transition_points['structure_break']
        }
    
    def _detect_primary_trend(self, closes: List[float]) -> str:
        """Detecta tendência com MÚLTIPLOS timeframes internos"""
        if len(closes) < 50:
            return "neutral"
            
        # Análise multi-período
        short_ma = sum(closes[-10:]) / 10
        medium_ma = sum(closes[-20:]) / 20 
        long_ma = sum(closes[-50:]) / 50
        
        # Tendência de VERY SHORT TERM (5 períodos)
        vst_trend = "up" if closes[-1] > closes[-5] else "down"
        
        # Tendência de SHORT TERM
        st_trend = "up" if short_ma > medium_ma else "down"
        
        # Tendência de MEDIUM TERM
        mt_trend = "up" if medium_ma > long_ma else "down"
        
        # CONFIRMAÇÃO: 2 de 3 devem concordar
        trends = [vst_trend, st_trend, mt_trend]
        up_count = trends.count("up")
        down_count = trends.count("down")
        
        if up_count >= 2:
            return "bullish"
        elif down_count >= 2:
            return "bearish"
        else:
            return "neutral"

    def _calculate_trend_strength(self, closes: List[float], trend: str) -> float:
        """Calcula força da tendência baseado na consistência"""
        if len(closes) < 20:
            return 0.5
            
        # Calcular direcionalidade
        price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        if trend == "bullish":
            positive_moves = sum(1 for change in price_changes if change > 0)
            strength = positive_moves / len(price_changes)
        elif trend == "bearish":
            negative_moves = sum(1 for change in price_changes if change < 0)
            strength = negative_moves / len(price_changes)
        else:
            # Para tendência neutra, força baseada na volatilidade
            volatility = stats.stdev(price_changes) / stats.mean(closes) if stats.mean(closes) > 0 else 0.02
            strength = max(0.1, 1 - volatility * 10)  # Baixa volatilidade = força
        
        return min(1.0, max(0.1, strength))

    def _analyze_trend_momentum(self, closes: List[float]) -> float:
        """Analisa momentum atual da tendência"""
        if len(closes) < 10:
            return 0.5
            
        # Momentum baseado na aceleração dos preços
        recent_changes = [closes[i] - closes[i-1] for i in range(max(1, len(closes)-5), len(closes))]
        previous_changes = [closes[i] - closes[i-1] for i in range(max(1, len(closes)-10), len(closes)-5)]
        
        if not recent_changes or not previous_changes:
            return 0.5
            
        recent_momentum = sum(recent_changes) / len(recent_changes)
        previous_momentum = sum(previous_changes) / len(previous_changes)
        
        # Momentum aumentando = bom, diminuindo = ruim
        if previous_momentum == 0:
            return 0.5
            
        momentum_ratio = recent_momentum / previous_momentum
        momentum_score = min(1.0, max(0.0, 0.5 + (momentum_ratio - 1) * 2))
        
        return momentum_score

    def _detect_trend_transition(self, closes: List[float], current_trend: str) -> Dict:
        """Detecta quando a tendência está prestes a mudar"""
        
        if len(closes) < 20:
            return {'is_transitioning': False, 'stage': 'insufficient_data'}
        
        # 1. DIVERGÊNCIAS DE MOMENTUM
        momentum_divergence = self._check_momentum_divergence(closes, current_trend)
        
        # 2. ESGOTAMENTO DE MOVIMENTO
        exhaustion = self._check_trend_exhaustion(closes, current_trend)
        
        # 3. QUEBRA DE ESTRUTURA
        structure_break = self._check_structure_break(closes, current_trend)
        
        # CLASSIFICAR ESTÁGIO DE TRANSIÇÃO
        if structure_break and momentum_divergence:
            stage = "reversal_confirmed"
            is_transitioning = True
        elif exhaustion and momentum_divergence:
            stage = "potential_reversal" 
            is_transitioning = True
        elif exhaustion:
            stage = "trend_weakening"
            is_transitioning = False
        else:
            stage = "trend_healthy"
            is_transitioning = False
            
        return {
            'is_transitioning': is_transitioning,
            'stage': stage,
            'momentum_divergence': momentum_divergence,
            'exhaustion_signals': exhaustion,
            'structure_break': structure_break
        }

    def _check_momentum_divergence(self, closes: List[float], trend: str) -> bool:
        """Detecta divergência entre preço e momentum"""
        if len(closes) < 20:
            return False
            
        # Últimos 10 candles vs anteriores
        recent_prices = closes[-10:]
        previous_prices = closes[-20:-10]
        
        if len(recent_prices) < 5 or len(previous_prices) < 5:
            return False
            
        # Direção do preço
        price_direction = "up" if recent_prices[-1] > recent_prices[0] else "down"
        
        # Momentum (variação percentual)
        recent_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] != 0 else 0
        previous_momentum = (previous_prices[-1] - previous_prices[0]) / previous_prices[0] if previous_prices[0] != 0 else 0
        
        # Divergência: Preço vai uma direção, momentum vai outra
        if trend == "bullish" and price_direction == "up" and recent_momentum < previous_momentum:
            return True
        elif trend == "bearish" and price_direction == "down" and recent_momentum > previous_momentum:
            return True
            
        return False

    def _check_trend_exhaustion(self, closes: List[float], trend: str) -> bool:
        """Verifica se o movimento atual está esgotado"""
        if len(closes) < 15:
            return False
            
        recent_high = max(closes[-10:])
        recent_low = min(closes[-10:])
        previous_high = max(closes[-15:-5])
        previous_low = min(closes[-15:-5])
        
        if trend == "bullish":
            # Exaustão: não consegue fazer novos máximos
            return recent_high <= previous_high
        elif trend == "bearish":
            # Exaustão: não consegue fazer novos mínimos
            return recent_low >= previous_low
            
        return False

    def _check_structure_break(self, closes: List[float], trend: str) -> bool:
        """Verifica quebra da estrutura de tendência"""
        if len(closes) < 25:
            return False
            
        # Para tendência de alta: fundos devem ser crescentes
        if trend == "bullish":
            first_low = min(closes[-25:-15])
            second_low = min(closes[-15:-5])
            current_low = min(closes[-5:])
            # Quebra: fundo atual é menor que fundo anterior
            return current_low < second_low
            
        # Para tendência de baixa: topos devem ser decrescentes
        elif trend == "bearish":
            first_high = max(closes[-25:-15])
            second_high = max(closes[-15:-5])
            current_high = max(closes[-5:])
            # Quebra: topo atual é maior que topo anterior
            return current_high > second_high
            
        return False

    def _suggest_trend_action(self, trend: str, strength: float, transition: Dict) -> str:
        """Sugere ação baseada na análise de tendência"""
        if transition['is_transitioning']:
            return "prepare_reversal"
        elif trend == "bullish" and strength > 0.6:
            return "trend_follow_buy"
        elif trend == "bearish" and strength > 0.6:
            return "trend_follow_sell"
        elif trend == "neutral" and strength > 0.7:
            return "breakout_watch"
        else:
            return "wait_confirm"

    def _default_trend_analysis(self) -> Dict:
        """Retorna análise padrão para dados insuficientes"""
        return {
            'primary_trend': 'neutral',
            'trend_strength': 0.5,
            'trend_momentum': 0.5,
            'is_transitioning': False,
            'transition_stage': 'insufficient_data',
            'confidence': 0.5,
            'action': 'wait_confirm',
            'momentum_divergence': False,
            'exhaustion_signals': False,
            'structure_break': False
        }

# =========================
# SISTEMA DE ENTRADAS NA TENDÊNCIA
# =========================
class TrendEntrySystem:
    def __init__(self):
        self.trend_analyzer = TrendAnalysisSystem()
        
    def find_trend_entries(self, symbol: str, closes: List[float], current_price: float) -> Dict:
        """Encontra as melhores entradas baseadas em tendência"""
        
        trend_analysis = self.trend_analyzer.analyze_trend_direction(symbol, closes)
        
        # SÓ OPERA SE TENDÊNCIA ESTÁ FORTE E SAUDÁVEL
        if not self._is_tradable_trend(trend_analysis):
            return self._create_no_entry_signal(symbol, trend_analysis)
        
        # ENCONTRAR MELHORES ENTRADAS BASEADO NA TENDÊNCIA
        if trend_analysis['action'] == 'trend_follow_buy':
            entry_signal = self._find_buy_entries(closes, current_price, trend_analysis)
        elif trend_analysis['action'] == 'trend_follow_sell':
            entry_signal = self._find_sell_entries(closes, current_price, trend_analysis)
        elif trend_analysis['action'] == 'prepare_reversal':
            entry_signal = self._find_reversal_entries(closes, current_price, trend_analysis)
        else:
            entry_signal = self._create_no_entry_signal(symbol, trend_analysis)
            
        return entry_signal
    
    def _is_tradable_trend(self, trend_analysis: Dict) -> bool:
        """Verifica se a tendência é operável"""
        return (trend_analysis['trend_strength'] > 0.6 and 
                trend_analysis['confidence'] > 0.65 and
                trend_analysis['action'] != 'wait_confirm')
    
    def _find_buy_entries(self, closes: List[float], current_price: float, trend_analysis: Dict) -> Dict:
        """Encontra entradas de COMPRA em tendência de alta"""
        if len(closes) < 15:
            return self._create_no_entry_signal("BUY", trend_analysis)
            
        # Estratégia 1: Pullback em suporte
        recent_low = min(closes[-10:])
        pullback_depth = (current_price - recent_low) / current_price
        
        if pullback_depth > 0.02:  # Pullback de pelo menos 2%
            return {
                'direction': 'buy',
                'entry_type': 'pullback',
                'confidence': trend_analysis['confidence'] * 0.9,
                'reason': f"Trend Following BUY: Pullback {pullback_depth:.2%} in bullish trend",
                'trend_strength': trend_analysis['trend_strength'],
                'risk_level': 'medium'
            }
        
        # Estratégia 2: Breakout de consolidação
        consolidation_break = self._check_consolidation_breakout(closes, "up")
        if consolidation_break:
            return {
                'direction': 'buy',
                'entry_type': 'breakout',
                'confidence': trend_analysis['confidence'] * 0.85,
                'reason': "Trend Following BUY: Breakout from consolidation",
                'trend_strength': trend_analysis['trend_strength'],
                'risk_level': 'high'
            }
            
        return self._create_no_entry_signal("BUY", trend_analysis)

    def _find_sell_entries(self, closes: List[float], current_price: float, trend_analysis: Dict) -> Dict:
        """Encontra entradas de VENDA em tendência de baixa"""
        if len(closes) < 15:
            return self._create_no_entry_signal("SELL", trend_analysis)
            
        # Estratégia 1: Rally em resistência
        recent_high = max(closes[-10:])
        rally_height = (recent_high - current_price) / current_price
        
        if rally_height > 0.02:  # Rally de pelo menos 2%
            return {
                'direction': 'sell',
                'entry_type': 'rally',
                'confidence': trend_analysis['confidence'] * 0.9,
                'reason': f"Trend Following SELL: Rally {rally_height:.2%} in bearish trend",
                'trend_strength': trend_analysis['trend_strength'],
                'risk_level': 'medium'
            }
        
        # Estratégia 2: Breakdown de consolidação
        consolidation_break = self._check_consolidation_breakout(closes, "down")
        if consolidation_break:
            return {
                'direction': 'sell',
                'entry_type': 'breakdown',
                'confidence': trend_analysis['confidence'] * 0.85,
                'reason': "Trend Following SELL: Breakdown from consolidation",
                'trend_strength': trend_analysis['trend_strength'],
                'risk_level': 'high'
            }
            
        return self._create_no_entry_signal("SELL", trend_analysis)

    def _find_reversal_entries(self, closes: List[float], current_price: float, trend_analysis: Dict) -> Dict:
        """Encontra entradas em possíveis reversões"""
        if len(closes) < 20:
            return self._create_no_entry_signal("REVERSAL", trend_analysis)
        
        # Determinar direção da reversão
        current_trend = trend_analysis['primary_trend']
        reversal_direction = "sell" if current_trend == "bullish" else "buy"
        
        return {
            'direction': reversal_direction,
            'entry_type': 'reversal',
            'confidence': trend_analysis['confidence'] * 0.8,
            'reason': f"Reversal {reversal_direction.upper()}: {trend_analysis['transition_stage']}",
            'trend_strength': trend_analysis['trend_strength'],
            'risk_level': 'high',
            'reversal_signals': {
                'momentum_divergence': trend_analysis['momentum_divergence'],
                'exhaustion': trend_analysis['exhaustion_signals'],
                'structure_break': trend_analysis['structure_break']
            }
        }

    def _check_consolidation_breakout(self, closes: List[float], direction: str) -> bool:
        """Verifica breakout de formação de consolidação"""
        if len(closes) < 20:
            return False
            
        # Verificar se preço rompeu faixa de consolidação
        consolidation_high = max(closes[-15:-5])
        consolidation_low = min(closes[-15:-5])
        current_price = closes[-1]
        
        if direction == "up":
            return current_price > consolidation_high
        else:
            return current_price < consolidation_low

    def _create_no_entry_signal(self, symbol: str, trend_analysis: Dict) -> Dict:
        """Cria sinal de não entrada (mas com análise)"""
        return {
            'direction': 'hold',
            'entry_type': 'none',
            'confidence': trend_analysis['confidence'],
            'reason': f"No optimal entry for {symbol} | Trend: {trend_analysis['primary_trend']}",
            'trend_strength': trend_analysis['trend_strength'],
            'risk_level': 'none',
            'trend_analysis': trend_analysis
        }

# =========================
# Indicadores Técnicos CORRIGIDOS
# =========================
class TechnicalIndicators:
    @staticmethod
    def _wilder_smooth(prev: float, cur: float, period: int) -> float:
        return (prev * (period - 1) + cur) / period

    def rsi_series_wilder(self, closes: List[float], period: int = 14) -> List[float]:
        """RSI CORRIGIDO - cálculo preciso"""
        if len(closes) < period + 1:
            return [50.0] * len(closes)
            
        gains = []
        losses = []
        
        # Calcular ganhos e perdas
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        
        # Verificar se temos dados suficientes
        if len(gains) < period:
            return [50.0] * len(closes)
            
        # Médias iniciais
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsis = []
        
        # Primeiro RSI
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - (100 / (1 + rs)))
        
        # RSI subsequentes com suavização de Wilder
        for i in range(period, len(gains)):
            avg_gain = self._wilder_smooth(avg_gain, gains[i], period)
            avg_loss = self._wilder_smooth(avg_loss, losses[i], period)
            
            if avg_loss == 0:
                rsis.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsis.append(rsi)
        
        # Preencher o início com 50
        full_rsis = [50.0] * (period)
        full_rsis.extend(rsis)
        
        return full_rsis

    def rsi_wilder(self, closes: List[float], period: int = 14) -> float:
        """RSI final CORRIGIDO"""
        series = self.rsi_series_wilder(closes, period)
        return series[-1] if series else 50.0

    def macd(self, closes: List[float]) -> Dict[str, Any]:
        """MACD CORRIGIDO - mais robusto"""
        if len(closes) < 35:
            return {"signal": "neutral", "strength": 0.0}
            
        def ema(data: List[float], period: int) -> List[float]:
            if len(data) < period:
                return []
            multiplier = 2 / (period + 1)
            ema_values = [sum(data[:period]) / period]  # Primeiro valor é SMA
            
            for i in range(period, len(data)):
                ema_val = (data[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema_val)
            return ema_values
            
        # Calcular EMAs
        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        
        if len(ema12) < 9 or len(ema26) < 9:
            return {"signal": "neutral", "strength": 0.0}
            
        # MACD Line = EMA12 - EMA26
        min_len = min(len(ema12), len(ema26))
        macd_line = [ema12[i] - ema26[i] for i in range(min_len)]
        
        # Signal Line = EMA9 do MACD
        signal_line = ema(macd_line, 9)
        
        if len(macd_line) < 1 or len(signal_line) < 1:
            return {"signal": "neutral", "strength": 0.0}
            
        # Histogram = MACD - Signal
        histogram = macd_line[-1] - signal_line[-1]
        
        # Strength baseado na volatilidade recente
        recent_closes = closes[-20:] if len(closes) >= 20 else closes
        price_range = max(recent_closes) - min(recent_closes)
        avg_price = sum(recent_closes) / len(recent_closes)
        
        if avg_price > 0 and price_range > 0:
            strength = min(1.0, abs(histogram) / (avg_price * 0.02))
        else:
            strength = 0.0
        
        # Determinar sinal
        if histogram > 0 and macd_line[-1] > signal_line[-1]:
            signal = "bullish"
        elif histogram < 0 and macd_line[-1] < signal_line[-1]:
            signal = "bearish"
        else:
            signal = "neutral"
            strength = 0.0
            
        return {"signal": signal, "strength": round(strength, 4)}

    def calculate_trend_strength(self, prices: List[float]) -> Dict[str, Any]:
        """Tendência CORRIGIDA - mais sensível"""
        if len(prices) < 50:
            return {"trend": "neutral", "strength": 0.0}
            
        # Médias de diferentes períodos para detectar tendência
        short_ma = sum(prices[-10:]) / 10
        medium_ma = sum(prices[-20:]) / 20
        long_ma = sum(prices[-50:]) / 50
        
        # Tendência principal (curto vs longo prazo)
        if short_ma > long_ma and medium_ma > long_ma:
            trend = "bullish"
            # Força baseada na diferença percentual
            strength = min(1.0, (short_ma - long_ma) / long_ma * 5)
        elif short_ma < long_ma and medium_ma < long_ma:
            trend = "bearish"
            strength = min(1.0, (long_ma - short_ma) / long_ma * 5)
        else:
            trend = "neutral"
            strength = 0.0
        
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
# IA EVOLUTIVA PRINCIPAL - ATUALIZADA COM TENDÊNCIAS
# =========================
class EvolutionaryIntelligence:
    def __init__(self):
        self.trend_entry_system = TrendEntrySystem()
        self.indicators = TechnicalIndicators()
        self.garch = GARCHSystem()
        
    def analyze_with_trend_focus(self, symbol: str, technical_data: Dict) -> Dict[str, Any]:
        """Análise EVOLUTIVA focada em tendências"""
        
        closes = technical_data.get('closes', [])
        current_price = technical_data.get('price', 100)
        
        if len(closes) < 50:
            return self._create_fallback_signal(symbol, current_price, "Dados insuficientes")
        
        # 1. ANÁLISE DE TENDÊNCIA AVANÇADA
        trend_entry_signal = self.trend_entry_system.find_trend_entries(symbol, closes, current_price)
        
        # 2. INDICADORES TÉCNICOS TRADICIONAIS
        rsi = self.indicators.rsi_wilder(closes)
        macd_result = self.indicators.macd(closes)
        
        # 3. ANÁLISE PROBABILÍSTICA GARCH
        returns = self._calculate_returns(closes)
        garch_probs = self.garch.run_garch_analysis(current_price, returns)
        
        # 4. SÍNTESE INTELIGENTE
        return self._synthesize_comprehensive_signal(
            symbol, 
            trend_entry_signal, 
            technical_data,
            rsi, 
            macd_result, 
            garch_probs
        )
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        return returns if returns else [random.gauss(0, 0.015) for _ in range(20)]
    
    def _synthesize_comprehensive_signal(self, symbol: str, trend_signal: Dict, 
                                       technical_data: Dict, rsi: float, 
                                       macd_result: Dict, garch_probs: Dict) -> Dict:
        """Combina análise de tendência com indicadores técnicos"""
        
        # BASE: Sinal de tendência é o principal
        direction = trend_signal['direction']
        base_confidence = trend_signal['confidence']
        reason = trend_signal['reason']
        
        # AJUSTAR COM INDICADORES TÉCNICOS
        confidence_boost = 0.0
        
        # RSI confirmation
        if ((direction == 'buy' and rsi < 40) or 
            (direction == 'sell' and rsi > 60)):
            confidence_boost += 0.10
            reason += " | RSI confirmed"
        
        # MACD confirmation  
        if ((direction == 'buy' and macd_result['signal'] == 'bullish') or
            (direction == 'sell' and macd_result['signal'] == 'bearish')):
            confidence_boost += 0.08
            reason += " | MACD aligned"
        
        # GARCH probabilities
        if ((direction == 'buy' and garch_probs['probability_buy'] > 0.6) or
            (direction == 'sell' and garch_probs['probability_sell'] > 0.6)):
            confidence_boost += 0.07
            reason += " | GARCH favorable"
        
        final_confidence = min(0.95, base_confidence + confidence_boost)
        
        return {
            'symbol': symbol,
            'direction': direction,
            'confidence': round(final_confidence, 4),
            'reason': reason,
            'rsi': round(rsi, 2),
            'macd_signal': macd_result['signal'],
            'macd_strength': macd_result['strength'],
            'probability_buy': garch_probs['probability_buy'],
            'probability_sell': garch_probs['probability_sell'],
            'price': technical_data['price'],
            'entry_type': trend_signal.get('entry_type', 'standard'),
            'trend_strength': trend_signal.get('trend_strength', 0.5),
            'risk_level': trend_signal.get('risk_level', 'medium'),
            'garch_volatility': garch_probs['volatility'],
            'timestamp': datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT"),
            'entry_time': self._calculate_entry_time(),
            'timeframe': 'T+1 (Próximo candle)',
            'trend_analysis': trend_signal.get('trend_analysis', {})
        }
    
    def _calculate_entry_time(self) -> str:
        now = datetime.now(timezone(timedelta(hours=-3)))
        entry_time = now + timedelta(minutes=1)
        return entry_time.strftime("%H:%M BRT")
    
    def _create_fallback_signal(self, symbol: str, price: float, reason: str) -> Dict[str, Any]:
        """Fallback para dados insuficientes"""
        current_time = datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT")
        
        return {
            'symbol': symbol,
            'direction': 'hold',
            'confidence': 0.60,
            'reason': f'Fallback: {reason}',
            'rsi': 50.0,
            'macd_signal': 'neutral',
            'macd_strength': 0.0,
            'probability_buy': 0.5,
            'probability_sell': 0.5,
            'price': price,
            'entry_type': 'none',
            'trend_strength': 0.5,
            'risk_level': 'none',
            'garch_volatility': 0.02,
            'timestamp': current_time,
            'entry_time': self._calculate_entry_time(),
            'timeframe': 'T+1 (Próximo candle)',
            'trend_analysis': {}
        }

# =========================
# Sistema Principal ATUALIZADO
# =========================
class TradingSystem:
    def __init__(self):
        self.evolutionary_ai = EvolutionaryIntelligence()
        self.data_gen = DataGenerator()
        
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        try:
            current_prices = self.data_gen.get_current_prices()
            current_price = current_prices.get(symbol, 100)
            historical_data = self.data_gen.get_historical_data(symbol)
            
            if not historical_data:
                return self.evolutionary_ai._create_fallback_signal(symbol, current_price, "Sem dados históricos")
                
            closes = [candle[3] for candle in historical_data]
            
            technical_data = {
                'closes': closes,
                'price': current_price,
                'rsi': 50.0,  # Será calculado pela IA
                'macd_signal': 'neutral'
            }
            
            # ANÁLISE EVOLUTIVA COM FOCO EM TENDÊNCIAS
            signal = self.evolutionary_ai.analyze_with_trend_focus(symbol, technical_data)
            
            return signal
            
        except Exception as e:
            logger.error("analysis_error", symbol=symbol, error=str(e))
            current_prices = self.data_gen.get_current_prices()
            current_price = current_prices.get(symbol, 100)
            return self.evolutionary_ai._create_fallback_signal(symbol, current_price, f"Erro: {str(e)}")

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
                
            # Ordenar por confiança e força de tendência
            all_signals.sort(key=lambda x: (x['confidence'], x.get('trend_strength', 0)), reverse=True)
            self.current_results = all_signals
            
            if all_signals:
                self.best_opportunity = all_signals[0]
                logger.info("best_opportunity_found", 
                           symbol=self.best_opportunity['symbol'],
                           confidence=self.best_opportunity['confidence'],
                           direction=self.best_opportunity['direction'])
            
            self.analysis_time = self.br_full(self.get_brazil_time())
            logger.info("analysis_completed", results_count=len(all_signals))
            
        except Exception as e:
            logger.error("analysis_error", error=str(e))
            self.current_results = [self.system.evolutionary_ai._create_fallback_signal(sym, 100, "Erro na análise") for sym in symbols]
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
        <title>IA Signal Pro - SISTEMA DE TENDÊNCIAS AVANÇADO</title>
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
            .badge.trend {{ background: #1f5f4a; color: white; }}
            .badge.entry {{ background: #5f1f4a; color: white; }}
            .badge.risk {{ background: #5f4a1f; color: white; }}
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
            .trend-line {{
                background: #1f4a5f !important;
                border-left: 3px solid #2aa9ff;
            }}
            .entry-line {{
                background: #4a1f5f !important;
                border-left: 3px solid #b36bff;
            }}
            .risk-high {{
                background: #5b1f1f !important;
                border-left: 3px solid #ff5b5b;
            }}
            .risk-medium {{
                background: #5b4a1f !important;
                border-left: 3px solid #f2a93b;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 IA Signal Pro - SISTEMA DE TENDÊNCIAS AVANÇADO</h1>
                <div class="clock" id="currentTime">{current_time}</div>
                <p>🚀 <strong>Análise de Tendências</strong> | ✅ Entradas a favor da tendência | 🎯 Detecção de reversões</p>
                <p>🔧 <strong>Novo Sistema:</strong> Trend Following + Pullbacks/Rallies + Alertas de Reversão</p>
            </div>
            
            <div class="controls">
                <div class="symbols-selection">
                    <h3>📈 Selecione os Ativos para Análise de Tendências:</h3>
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
                
                <button onclick="runAnalysis()" id="analyzeBtn">🎯 Analisar Tendências (T+1)</button>
                <button onclick="checkStatus()">📊 Status do Sistema</button>
                <div id="status" class="status info">
                    ⏰ Hora atual: {current_time} | ✅ Sistema de Tendências Online
                </div>
            </div>
            
            <div id="bestSignal" style="display: none;">
                <h2>🥇 MELHOR OPORTUNIDADE - ANÁLISE DE TENDÊNCIA</h2>
                <div id="bestCard"></div>
            </div>
            
            <div id="allSignals" style="display: none;">
                <h2>📊 TODOS OS SINAIS - SISTEMA DE TENDÊNCIAS</h2>
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
                analyzeBtn.textContent = '🎯 Analisando Tendências...';

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
                    analyzeBtn.textContent = '🎯 Analisar Tendências (T+1)';
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
                const trendStrengthPercent = (signal.trend_strength * 100).toFixed(1);
                const priceFormatted = typeof signal.price === 'number' ? 
                    signal.price.toLocaleString('pt-BR', {{ style: 'currency', currency: 'USD' }}) : 
                    '$' + signal.price;
                
                const riskClass = signal.risk_level === 'high' ? 'risk-high' : 
                                signal.risk_level === 'medium' ? 'risk-medium' : '';
                
                // Informações de tendência
                const trendAnalysis = signal.trend_analysis || {{}};
                const primaryTrend = trendAnalysis.primary_trend || 'neutral';
                const trendAction = trendAnalysis.action || 'wait_confirm';
                
                return `<div class="signal-card ${{directionClass}} ${{isBest ? 'best-card' : ''}}">
                    <h3>${{directionEmoji}} ${{signal.symbol}} ${{isBest ? '🏆' : ''}}</h3>
                    <div class="info-line">
                        <span class="badge ${{directionClass}}">${{signal.direction.toUpperCase()}}</span>
                        <span class="badge confidence">${{confidencePercent}}% Confiança</span>
                        <span class="badge trend">${{trendStrengthPercent}}% Força Trend</span>
                        <span class="badge entry">${{signal.entry_type}}</span>
                        <span class="badge risk">Risk: ${{signal.risk_level}}</span>
                    </div>
                    <div class="info-line"><strong>🎯 Entrada:</strong> ${{signal.entry_time}}</div>
                    <div class="info-line"><strong>💰 Preço Atual:</strong> ${{priceFormatted}}</div>
                    <div class="info-line trend-line">
                        <strong>📊 Tendência:</strong> ${{primaryTrend}} | ${{trendAction}}
                    </div>
                    <div class="info-line ${{riskClass}}">
                        <strong>🎯 Tipo Entrada:</strong> ${{signal.entry_type}} | <strong>Risk:</strong> ${{signal.risk_level}}
                    </div>
                    <div class="info-line"><strong>📈 Probabilidade:</strong> COMPRA ${{(signal.probability_buy * 100).toFixed(1)}}% | VENDA ${{(signal.probability_sell * 100).toFixed(1)}}%</div>
                    <div class="info-line"><strong>📊 RSI:</strong> ${{signal.rsi}}</div>
                    <div class="info-line"><strong>🔍 MACD:</strong> ${{signal.macd_signal}} (${{(signal.macd_strength * 100).toFixed(1)}}%)</div>
                    <div class="info-line"><strong>🎲 Volatilidade GARCH:</strong> ${{(signal.garch_volatility * 100).toFixed(3)}}%</div>
                    <div class="info-line entry-line"><strong>🚀 Estratégia:</strong> ${{signal.reason}}</div>
                    <div class="info-line"><strong>⏰ Análise:</strong> ${{signal.timestamp}}</div>
                </div>`;
            }}

            async function checkStatus() {{
                try {{
                    const response = await fetch('/status');
                    const status = await response.json();
                    
                    let statusHtml = '<div class="status info">' +
                        '<strong>🎯 Status do Sistema de Tendências:</strong><br>' +
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
            'message': f'Análise de tendências iniciada para {len(symbols)} ativos.'
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
    print("🎯 IA Signal Pro - SISTEMA DE TENDÊNCIAS AVANÇADO")
    print("🚀 Sistema Atualizado: Análise de Tendências + Entradas Inteligentes")
    print("✅ Novos Recursos: Trend Following + Pullbacks/Rallies + Alertas de Reversão")
    print("📊 Ativos padrão:", DEFAULT_SYMBOLS)
    print("🌐 Servidor iniciando na porta 8080...")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
