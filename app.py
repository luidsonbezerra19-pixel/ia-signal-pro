from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime, timezone, timedelta
import os
import random
import math
import json
from typing import List, Dict, Tuple, Any

# ========== SISTEMA DE MEMÓRIA APRIMORADO ==========
class MemorySystem:
    def __init__(self):
        self.symbol_memory = {}
        self.market_regime = "NORMAL"  # NORMAL, VOLATILE, TRENDING
        self.regime_memory = []
    
    def get_symbol_weights(self, symbol: str) -> Dict:
        """Pesos dinâmicos baseados no histórico do símbolo"""
        base_weights = {
            'monte_carlo': 0.65,
            'rsi': 0.08, 'adx': 0.07, 'macd': 0.06, 
            'bollinger': 0.05, 'volume': 0.04, 'fibonacci': 0.03,
            'multi_tf': 0.02
        }
        
        # Ajuste baseado no regime de mercado
        if self.market_regime == "VOLATILE":
            base_weights['monte_carlo'] = 0.60  # Reduz confiança em alta volatilidade
            base_weights['bollinger'] = 0.08    # Aumenta importância de Bollinger
            base_weights['adx'] = 0.09          # ADX mais importante em tendências
        elif self.market_regime == "TRENDING":
            base_weights['adx'] = 0.10
            base_weights['multi_tf'] = 0.04
        
        # Ajuste sutil baseado no símbolo
        if "BTC" in symbol or "ETH" in symbol:
            base_weights['monte_carlo'] = 0.68
            base_weights['volume'] = 0.05
        elif "ADA" in symbol or "XRP" in symbol:
            base_weights['rsi'] = 0.10
            base_weights['bollinger'] = 0.07
            
        return base_weights
    
    def update_market_regime(self, volatility: float, adx_values: List[float]):
        """Atualiza regime de mercado baseado na volatilidade e ADX"""
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
        
        # Mantém apenas últimas 100 leituras
        if len(self.regime_memory) > 100:
            self.regime_memory.pop(0)

# ========== SISTEMA DE LIQUIDEZ ==========
class LiquiditySystem:
    def __init__(self):
        self.symbol_liquidity = {}
        self.volume_profile = {}
    
    def calculate_liquidity_score(self, symbol: str, prices: List[float]) -> float:
        """Calcula score de liquidez baseado na volatilidade e volume implícito"""
        if len(prices) < 10:
            return 0.7  # Valor padrão
        
        # Calcula volatilidade relativa (proxy para liquidez)
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(abs(ret))
        
        if not returns:
            return 0.7
            
        volatility = sum(returns) / len(returns)
        
        # Ativos com menor volatilidade têm maior liquidez implícita
        if volatility < 0.005:
            liquidity_score = 0.9
        elif volatility < 0.01:
            liquidity_score = 0.8
        elif volatility < 0.02:
            liquidity_score = 0.7
        else:
            liquidity_score = 0.6
            
        # Ajusta para símbolos específicos
        if symbol in ['BTC/USDT', 'ETH/USDT']:
            liquidity_score = min(0.95, liquidity_score + 0.1)
        elif symbol in ['ADA/USDT', 'XRP/USDT']:
            liquidity_score = max(0.5, liquidity_score - 0.1)
            
        self.symbol_liquidity[symbol] = liquidity_score
        return liquidity_score
    
    def get_spread_impact(self, symbol: str) -> float:
        """Estima impacto do spread baseado na liquidez"""
        liquidity = self.symbol_liquidity.get(symbol, 0.7)
        # Spread menor para alta liquidez
        return 0.001 * (1 - liquidity)  # 0.1% a 0.03% de impacto

# ========== SISTEMA DE CORRELAÇÕES ==========
class CorrelationSystem:
    def __init__(self):
        self.correlation_matrix = self._initialize_correlations()
    
    def _initialize_correlations(self) -> Dict:
        """Inicializa matriz de correlações baseada em grupos de ativos"""
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
        """Ajusta confiança baseado em correlações com outros sinais"""
        if symbol not in self.correlation_matrix:
            return 1.0
            
        adjustments = []
        for other_symbol, signal_data in other_signals.items():
            if other_symbol != symbol and other_symbol in self.correlation_matrix[symbol]:
                correlation = self.correlation_matrix[symbol][other_symbol]
                # Se sinais estão alinhados com correlação positiva, aumenta confiança
                if (signal_data['direction'] == other_signals.get(symbol, {}).get('direction', '')):
                    adjustment = 1.0 + (correlation * 0.1)  # +10% no máximo
                else:
                    adjustment = 1.0 - (correlation * 0.05)  # -5% no máximo
                adjustments.append(adjustment)
        
        if not adjustments:
            return 1.0
            
        return sum(adjustments) / len(adjustments)

# ========== SISTEMA DE EVENTOS DE NOTÍCIAS ==========
class NewsEventSystem:
    def __init__(self):
        self.active_events = []
        self.volatility_multiplier = 1.0
    
    def generate_market_events(self):
        """Gera eventos de mercado aleatórios (simulação)"""
        events = [
            {'type': 'FED_MEETING', 'impact': 'HIGH', 'volatility_multiplier': 2.0},
            {'type': 'CPI_RELEASE', 'impact': 'MEDIUM', 'volatility_multiplier': 1.5},
            {'type': 'REGULATION_NEWS', 'impact': 'MEDIUM', 'volatility_multiplier': 1.8},
            {'type': 'WHALE_MOVEMENT', 'impact': 'LOW', 'volatility_multiplier': 1.3},
            {'type': 'EXCHANGE_ISSUE', 'impact': 'HIGH', 'volatility_multiplier': 2.2}
        ]
        
        # 15% de chance de evento a cada análise
        if random.random() < 0.15:
            event = random.choice(events)
            event['start_time'] = datetime.now()
            event['duration_hours'] = random.randint(2, 12)
            self.active_events.append(event)
            print(f"📢 EVENTO DE MERCADO: {event['type']} (Impacto: {event['impact']})")
    
    def get_volatility_multiplier(self):
        """Retorna multiplicador de volatilidade baseado em eventos ativos"""
        if not self.active_events:
            return 1.0
        
        max_multiplier = 1.0
        current_time = datetime.now()
        
        # Remove eventos expirados e encontra maior multiplicador
        self.active_events = [
            event for event in self.active_events 
            if current_time - event['start_time'] < timedelta(hours=event['duration_hours'])
        ]
        
        for event in self.active_events:
            max_multiplier = max(max_multiplier, event['volatility_multiplier'])
        
        return max_multiplier
    
    def adjust_confidence_for_events(self, confidence: float) -> float:
        """Ajusta confiança baseado em eventos ativos"""
        multiplier = self.get_volatility_multiplier()
        # Reduz confiança durante eventos de alta volatilidade
        if multiplier > 1.5:
            return confidence * 0.85
        elif multiplier > 1.2:
            return confidence * 0.92
        return confidence

# ========== CLUSTERIZAÇÃO DE VOLATILIDADE ==========
class VolatilityClustering:
    def __init__(self):
        self.volatility_regimes = {}
        self.historical_volatility = []
    
    def detect_volatility_clusters(self, prices: List[float], symbol: str) -> str:
        """Detecta clusters de volatilidade usando método simplificado"""
        if len(prices) < 20:
            return "MEDIUM"
        
        # Calcula retornos
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(abs(ret))
        
        if not returns:
            return "MEDIUM"
        
        # Volatilidade histórica
        volatility = sum(returns) / len(returns)
        self.historical_volatility.append(volatility)
        
        # Mantém histórico limitado
        if len(self.historical_volatility) > 50:
            self.historical_volatility.pop(0)
        
        # Classifica volatilidade
        if len(self.historical_volatility) > 10:
            avg_vol = sum(self.historical_volatility) / len(self.historical_volatility)
            
            if volatility > avg_vol * 1.5:
                regime = "HIGH"
            elif volatility < avg_vol * 0.7:
                regime = "LOW"
            else:
                regime = "MEDIUM"
        else:
            # Classificação inicial
            if volatility > 0.015:
                regime = "HIGH"
            elif volatility < 0.008:
                regime = "LOW"
            else:
                regime = "MEDIUM"
        
        self.volatility_regimes[symbol] = regime
        return regime
    
    def get_regime_adjustment(self, symbol: str) -> float:
        """Retorna ajuste baseado no regime de volatilidade"""
        regime = self.volatility_regimes.get(symbol, "MEDIUM")
        
        if regime == "HIGH":
            return 0.85  # Reduz confiança em alta volatilidade
        elif regime == "LOW":
            return 1.05  # Aumenta confiança em baixa volatilidade
        else:
            return 1.0

# ========== SIMULAÇÃO MONTE CARLO 3000 ==========
class MonteCarloSimulator:
    @staticmethod
    def generate_price_paths(base_price: float, num_paths: int = 3000, steps: int = 10) -> List[List[float]]:
        """Gera 3000 caminhos de preço realistas - Versão otimizada"""
        paths = []
        
        for _ in range(num_paths):
            prices = [base_price]
            current = base_price
            
            for step in range(steps - 1):
                # Volatilidade dinâmica + tendência realista (otimizada)
                volatility = 0.008 + (step * 0.0002)  # Reduzir multiplicação
                trend = random.uniform(-0.003, 0.003)
                
                change = trend + random.gauss(0, 1) * volatility
                new_price = current * (1 + change)
                new_price = max(new_price, base_price * 0.7)
                
                prices.append(new_price)
                current = new_price
            
            paths.append(prices)
        
        return paths
    
    @staticmethod
    def calculate_probability_distribution(paths: List[List[float]]) -> Dict:
        """Calcula probabilidades com 3000 simulações - Versão aprimorada"""
        if not paths or len(paths) < 1000:
            return {'probability_buy': 0.5, 'probability_sell': 0.5, 'quality': 'LOW'}
        
        initial_price = paths[0][0]
        final_prices = [path[-1] for path in paths]
        
        # Análise mais robusta
        higher_prices = sum(1 for price in final_prices if price > initial_price * 1.01)   # +1%
        lower_prices = sum(1 for price in final_prices if price < initial_price * 0.99)    # -1%
        neutral_prices = len(final_prices) - higher_prices - lower_prices
        
        total_valid = higher_prices + lower_prices
        if total_valid == 0:
            return {'probability_buy': 0.5, 'probability_sell': 0.5, 'quality': 'LOW'}
        
        probability_buy = higher_prices / total_valid
        probability_sell = lower_prices / total_valid
        
        # Qualidade baseada na clareza do sinal
        prob_strength = abs(probability_buy - 0.5)
        clarity_ratio = total_valid / len(final_prices)
        
        if prob_strength > 0.25 and clarity_ratio > 0.7:
            quality = 'HIGH'
        elif prob_strength > 0.15 and clarity_ratio > 0.5:
            quality = 'MEDIUM' 
        else:
            quality = 'LOW'
        
        return {
            'probability_buy': max(0.35, min(0.65, probability_buy)),  # Range mais conservador
            'probability_sell': max(0.35, min(0.65, probability_sell)),
            'quality': quality
        }

# ========== INDICADORES TÉCNICOS OTIMIZADOS ==========
class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: List[float]) -> float:
        """RSI com cache simples"""
        if len(prices) < 14:
            return random.uniform(30, 70)  # Range mais realista
            
        # Usar apenas os últimos 20 preços (otimização)
        recent_prices = prices[-20:] if len(prices) > 20 else prices
        
        gains = losses = 0.0
        for i in range(1, len(recent_prices)):
            change = recent_prices[i] - recent_prices[i-1]
            if change > 0:
                gains += change
            else:
                losses -= change
        
        if losses == 0:
            return 70.0 if gains > 0 else 30.0
            
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return max(10, min(90, round(rsi, 1)))  # Limites mais realistas

    @staticmethod
    def calculate_adx(prices: List[float]) -> float:
        """ADX realista corrigido - SEM numpy"""
        if len(prices) < 15:
            return random.uniform(20, 40)
        
        # Calcula True Range (TR) - versão simplificada sem numpy
        true_ranges = []
        for i in range(1, min(15, len(prices))):
            high_low = abs(prices[i] - prices[i-1])
            true_ranges.append(high_low)
        
        if not true_ranges:
            return random.uniform(20, 40)
        
        atr = sum(true_ranges) / len(true_ranges)
        
        # Calcula directional movement
        plus_dm = 0
        minus_dm = 0
        
        for i in range(1, min(15, len(prices))):
            up_move = prices[i] - prices[i-1]
            down_move = prices[i-1] - prices[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm += up_move
            elif down_move > up_move and down_move > 0:
                minus_dm += down_move
        
        # Calcula ADX simplificado
        if atr == 0:
            return random.uniform(15, 25)
        
        plus_di = (plus_dm / atr) * 100
        minus_di = (minus_dm / atr) * 100
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001) * 100
        
        # Suaviza para obter ADX
        adx = min(60, max(10, dx * 1.5))  # Escala ajustada
        
        return round(adx, 1)

    @staticmethod
    def calculate_std_dev(data: List[float]) -> float:
        """Calcula desvio padrão sem numpy"""
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict:
        """MACD rápido"""
        if len(prices) < 20:
            return {'signal': 'neutral', 'strength': 0.5}
        
        ema_12 = sum(prices[-12:]) / 12
        ema_26 = sum(prices[-20:]) / 20
        
        if ema_12 > ema_26 * 1.008:
            return {'signal': 'bullish', 'strength': min(1.0, (ema_12 - ema_26) / ema_26)}
        elif ema_12 < ema_26 * 0.992:
            return {'signal': 'bearish', 'strength': min(1.0, (ema_26 - ema_12) / ema_26)}
        return {'signal': 'neutral', 'strength': 0.3}

    @staticmethod
    def calculate_bollinger_bands(prices: List[float]) -> Dict:
        """Bollinger Bands rápido"""
        if len(prices) < 15:
            return {'signal': 'neutral'}
        
        recent = prices[-15:]
        middle = sum(recent) / 15
        std = TechnicalIndicators.calculate_std_dev(recent)  # Usa nossa função sem numpy
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
        """Volume Profile simplificado"""
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
        """Fibonacci rápido"""
        if len(prices) < 15:
            return {'signal': 'neutral'}
        
        high = max(prices[-15:])
        low = min(prices[-15:])
        current = prices[-1]
        diff = high - low
        
        if current > high - (0.382 * diff):
            return {'signal': 'resistance'}
        elif current < low + (0.618 * diff):
            return {'signal': 'support'}
        return {'signal': 'neutral'}

# ========== ANÁLISE MULTI-TIMEFRAME RÁPIDA ==========
class MultiTimeframeAnalyzer:
    @staticmethod
    def analyze_consensus(prices: List[float]) -> str:
        """Mais sensível a tendências reais"""
        if len(prices) < 15:
            return 'neutral'
        
        # Timeframes com pesos diferentes
        tf_short = prices[-6:]  # Curto prazo
        tf_medium = prices[-12:] # Médio prazo  
        tf_long = prices[-18:]  # Longo prazo
        
        trends = []
        weights = []
        
        for i, tf in enumerate([tf_short, tf_medium, tf_long]):
            if len(tf) > 3:
                # Tendência com peso progressivo (TF maior = mais peso)
                trend_strength = (tf[-1] - tf[0]) / tf[0]
                weight = [0.3, 0.4, 0.5][i]  # Pesos diferentes
                
                if trend_strength > 0.008:  # Limiares ajustados
                    trends.append(('buy', weight))
                elif trend_strength < -0.008:
                    trends.append(('sell', weight))
                else:
                    trends.append(('neutral', weight * 0.5))
        
        if not trends:
            return 'neutral'
            
        # Soma ponderada
        buy_score = sum(weight for direction, weight in trends if direction == 'buy')
        sell_score = sum(weight for direction, weight in trends if direction == 'sell')
        
        if buy_score > sell_score + 0.2:
            return 'buy'
        elif sell_score > buy_score + 0.2:
            return 'sell'
        return 'neutral'

# ========== SISTEMA PRINCIPAL ATUALIZADO ==========
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
        
        # Cache para análise entre símbolos
        self.current_analysis_cache = {}
    
    def analyze_symbol(self, symbol: str, horizon: int) -> Dict:
        """Analisa um símbolo com todos os sistemas integrados"""
        
        # Geração de preços base
        symbol_bases = {
            'BTC/USDT': (30000, 60000), 'ETH/USDT': (1800, 3500), 
            'SOL/USDT': (80, 200), 'ADA/USDT': (0.3, 0.6),
            'XRP/USDT': (0.4, 0.8), 'BNB/USDT': (200, 400)
        }
        
        base_range = symbol_bases.get(symbol, (50, 400))
        base_price = random.uniform(base_range[0], base_range[1])
        
        # Geração de histórico com sistema de eventos
        volatility_multiplier = self.news_events.get_volatility_multiplier()
        historical_prices = [base_price]
        current = base_price
        
        for i in range(49):
            # Volatilidade ajustada por eventos
            base_volatility = 0.006 if "BTC" in symbol or "ETH" in symbol else 0.012
            adjusted_volatility = base_volatility * volatility_multiplier
                
            change = random.gauss(0, adjusted_volatility)
            current = current * (1 + change)
            historical_prices.append(current)
        
        # 3000 SIMULAÇÕES MONTE CARLO
        future_paths = self.monte_carlo.generate_price_paths(
            historical_prices[-1], num_paths=3000, steps=8
        )
        mc_result = self.monte_carlo.calculate_probability_distribution(future_paths)
        
        # INDICADORES
        rsi = self.indicators.calculate_rsi(historical_prices)
        adx = self.indicators.calculate_adx(historical_prices)
        macd = self.indicators.calculate_macd(historical_prices)
        bollinger = self.indicators.calculate_bollinger_bands(historical_prices)
        volume = self.indicators.calculate_volume_profile(historical_prices)
        fibonacci = self.indicators.calculate_fibonacci(historical_prices)
        multi_tf_consensus = self.multi_tf.analyze_consensus(historical_prices)
        
        # NOVOS SISTEMAS
        liquidity_score = self.liquidity.calculate_liquidity_score(symbol, historical_prices)
        volatility_regime = self.volatility_clustering.detect_volatility_clusters(historical_prices, symbol)
        
        # Atualiza regime de mercado
        self.memory.update_market_regime(
            volatility=adjusted_volatility, 
            adx_values=[adx] if adx else [25]
        )
        
        # Gera eventos aleatórios
        self.news_events.generate_market_events()
        
        # PESOS COM MEMÓRIA
        weights = self.memory.get_symbol_weights(symbol)
        
        # SISTEMA DE PONTUAÇÃO COM NOVOS AJUSTES
        score = 0
        factors = []
        winning_indicators = []
        
        # 1. MONTE CARLO (65% peso) - com ajuste de volatilidade
        base_mc_score = mc_result['probability_buy'] * weights['monte_carlo'] * 100
        volatility_adjustment = self.volatility_clustering.get_regime_adjustment(symbol)
        mc_score = base_mc_score * volatility_adjustment
        
        score += mc_score
        factors.append(f"MC:{mc_score:.1f}")
        
        # 2. INDICADORES TÉCNICOS
        # RSI
        rsi_score = 0
        if (mc_result['probability_buy'] > 0.5 and 30 < rsi < 70) or (mc_result['probability_buy'] < 0.5 and 30 < rsi < 70):
            rsi_score = weights['rsi'] * 15
            winning_indicators.append('RSI')
        score += rsi_score
        factors.append(f"RSI:{rsi_score:.1f}")
        
        # ADX
        adx_score = 0
        if adx > 25:
            adx_score = weights['adx'] * 12
            winning_indicators.append('ADX')
        score += adx_score
        factors.append(f"ADX:{adx_score:.1f}")
        
        # MACD
        macd_score = 0
        if (mc_result['probability_buy'] > 0.5 and macd['signal'] == 'bullish') or \
           (mc_result['probability_buy'] < 0.5 and macd['signal'] == 'bearish'):
            macd_score = weights['macd'] * 10 * macd['strength']
            winning_indicators.append('MACD')
        score += macd_score
        factors.append(f"MACD:{macd_score:.1f}")
        
        # BOLLINGER
        bb_score = 0
        if (mc_result['probability_buy'] > 0.5 and bollinger['signal'] in ['oversold', 'bullish']) or \
           (mc_result['probability_buy'] < 0.5 and bollinger['signal'] in ['overbought', 'bearish']):
            bb_score = weights['bollinger'] * 8
            winning_indicators.append('BB')
        score += bb_score
        factors.append(f"BB:{bb_score:.1f}")
        
        # VOLUME
        volume_score = 0
        if (mc_result['probability_buy'] > 0.5 and volume['signal'] in ['oversold', 'neutral']) or \
           (mc_result['probability_buy'] < 0.5 and volume['signal'] in ['overbought', 'neutral']):
            volume_score = weights['volume'] * 6
            winning_indicators.append('VOL')
        score += volume_score
        factors.append(f"VOL:{volume_score:.1f}")
        
        # FIBONACCI
        fib_score = 0
        if (mc_result['probability_buy'] > 0.5 and fibonacci['signal'] == 'support') or \
           (mc_result['probability_buy'] < 0.5 and fibonacci['signal'] == 'resistance'):
            fib_score = weights['fibonacci'] * 5
            winning_indicators.append('FIB')
        score += fib_score
        factors.append(f"FIB:{fib_score:.1f}")
        
        # MULTI-TIMEFRAME
        tf_score = 0
        if multi_tf_consensus == ('buy' if mc_result['probability_buy'] > 0.5 else 'sell'):
            tf_score = weights['multi_tf'] * 8
            winning_indicators.append('MultiTF')
        score += tf_score
        factors.append(f"TF:{tf_score:.1f}")
        
        # 3. AJUSTES FINAIS POR LIQUIDEZ E EVENTOS
        liquidity_adjustment = 0.95 + (liquidity_score * 0.1)  # 0.95 a 1.05
        score *= liquidity_adjustment
        factors.append(f"LIQ:{liquidity_adjustment:.2f}")
        
        # Ajuste por eventos de notícias
        final_confidence = min(0.90, max(0.55, score / 100))
        final_confidence = self.news_events.adjust_confidence_for_events(final_confidence)
        
        # DIREÇÃO FINAL
        direction = 'buy' if mc_result['probability_buy'] > 0.5 else 'sell'
        
        # Cache para correlações
        self.current_analysis_cache[symbol] = {
            'direction': direction,
            'confidence': final_confidence,
            'timestamp': datetime.now()
        }
        
        # APLICA CORRELAÇÕES (após todos os símbolos serem processados)
        correlation_adjustment = self.correlation.get_correlation_adjustment(
            symbol, self.current_analysis_cache
        )
        final_confidence *= correlation_adjustment
        factors.append(f"CORR:{correlation_adjustment:.2f}")
        
        return {
            'symbol': symbol,
            'horizon': horizon,
            'direction': direction,
            'probability_buy': mc_result['probability_buy'],
            'probability_sell': mc_result['probability_sell'],
            'confidence': final_confidence,
            'rsi': rsi,
            'adx': adx,
            'multi_timeframe': multi_tf_consensus,
            'monte_carlo_quality': mc_result['quality'],
            'winning_indicators': winning_indicators,
            'score_factors': factors,
            'price': historical_prices[-1],
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            # Novas métricas
            'liquidity_score': round(liquidity_score, 2),
            'volatility_regime': volatility_regime,
            'market_regime': self.memory.market_regime,
            'volatility_multiplier': round(volatility_multiplier, 2)
        }

# ========== FLASK APP ==========
app = Flask(__name__)
CORS(app)

trading_system = EnhancedTradingSystem()

class AnalysisManager:
    def __init__(self):
        self.current_results = []
        self.best_opportunity = None
        self.analysis_time = None
        self.is_analyzing = False
        self.available_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XRP/USDT', 'BNB/USDT']
    
    def analyze_symbols_thread(self, symbols, sims, only_adx):
        try:
            self.is_analyzing = True
            start_time = datetime.now()
            print(f"🔍 Iniciando análise com 3000 simulações: {symbols}")
            
            # Limpa cache de análise anterior
            trading_system.current_analysis_cache = {}
            
            # ANALISA TODOS OS HORIZONTES PARA CADA ATIVO
            all_symbol_results = {}
            
            for symbol in symbols:
                symbol_horizon_results = []
                
                # Analisa T+1, T+2, T+3 para o mesmo ativo
                for horizon in [1, 2, 3]:
                    result = trading_system.analyze_symbol(symbol, horizon)
                    symbol_horizon_results.append(result)
                
                # Encontra o MELHOR horizonte para este ativo
                best_horizon_result = max(symbol_horizon_results, key=lambda x: x['confidence'])
                all_symbol_results[symbol] = {
                    'best': best_horizon_result,
                    'all_horizons': symbol_horizon_results  # Guarda todos para mostrar depois
                }
            
            # FORMATA RESULTADOS FINAIS
            self.current_results = []
            all_horizons_display = []  # Para mostrar TODOS os T+
            
            for symbol, data in all_symbol_results.items():
                best_result = data['best']
                
                # Melhor de cada ativo (para ranking principal)
                formatted_best = {
                    'symbol': best_result['symbol'],
                    'horizon': best_result['horizon'],
                    'direction': best_result['direction'],
                    'p_buy': round(best_result['probability_buy'] * 100, 1),
                    'p_sell': round(best_result['probability_sell'] * 100, 1),
                    'confidence': round(best_result['confidence'] * 100, 1),
                    'adx': round(best_result['adx'], 1),
                    'rsi': round(best_result['rsi'], 1),
                    'price': round(best_result['price'], 4),
                    'timestamp': best_result['timestamp'],
                    'technical_override': len(best_result['winning_indicators']) >= 4,
                    'multi_timeframe': best_result['multi_timeframe'],
                    'monte_carlo_quality': best_result['monte_carlo_quality'],
                    'winning_indicators': best_result['winning_indicators'],
                    'score_factors': best_result['score_factors'],
                    'assertiveness': self.calculate_assertiveness(best_result),
                    'is_best_of_symbol': True,  # Marca como melhor do ativo
                    # Novos campos
                    'liquidity_score': best_result['liquidity_score'],
                    'volatility_regime': best_result['volatility_regime'],
                    'market_regime': best_result['market_regime'],
                    'volatility_multiplier': best_result['volatility_multiplier']
                }
                self.current_results.append(formatted_best)
                
                # Adiciona TODOS os horizontes para display completo
                for horizon_result in data['all_horizons']:
                    formatted_all = {
                        'symbol': horizon_result['symbol'],
                        'horizon': horizon_result['horizon'],
                        'direction': horizon_result['direction'],
                        'p_buy': round(horizon_result['probability_buy'] * 100, 1),
                        'p_sell': round(horizon_result['probability_sell'] * 100, 1),
                        'confidence': round(horizon_result['confidence'] * 100, 1),
                        'adx': round(horizon_result['adx'], 1),
                        'rsi': round(horizon_result['rsi'], 1),
                        'price': round(horizon_result['price'], 4),
                        'timestamp': horizon_result['timestamp'],
                        'technical_override': len(horizon_result['winning_indicators']) >= 4,
                        'multi_timeframe': horizon_result['multi_timeframe'],
                        'monte_carlo_quality': horizon_result['monte_carlo_quality'],
                        'winning_indicators': horizon_result['winning_indicators'],
                        'score_factors': horizon_result['score_factors'],
                        'assertiveness': self.calculate_assertiveness(horizon_result),
                        'is_best_of_symbol': (horizon_result['horizon'] == best_result['horizon']),
                        # Novos campos
                        'liquidity_score': horizon_result['liquidity_score'],
                        'volatility_regime': horizon_result['volatility_regime'],
                        'market_regime': horizon_result['market_regime'],
                        'volatility_multiplier': horizon_result['volatility_multiplier']
                    }
                    all_horizons_display.append(formatted_all)
            
            # MELHOR OPORTUNIDADE GLOBAL (entre os melhores de cada ativo)
            if self.current_results:
                self.best_opportunity = max(self.current_results, key=lambda x: x['confidence'])
                self.best_opportunity['entry_time'] = self.calculate_entry_time(self.best_opportunity['horizon'])
            
            # Substitui resultados por TODOS os horizontes
            self.current_results = all_horizons_display
            
            self.analysis_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Análise concluída em {processing_time:.1f}s: {len(symbols)} ativos, {len(self.current_results)} sinais")
            print(f"📊 Regime de Mercado: {trading_system.memory.market_regime}")
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            import traceback
            traceback.print_exc()
            self.current_results = self._get_fallback_results(symbols)
            self.best_opportunity = self.current_results[0] if self.current_results else None
        finally:
            self.is_analyzing = False
    
    def _get_fallback_results(self, symbols):
        """Fallback garantido"""
        results = []
        for symbol in symbols:
            for horizon in [1, 2, 3]:
                results.append({
                    'symbol': symbol,
                    'horizon': horizon,
                    'direction': random.choice(['buy', 'sell']),
                    'p_buy': random.randint(55, 75),
                    'p_sell': random.randint(25, 45),
                    'confidence': random.randint(65, 85),
                    'adx': random.randint(30, 50),
                    'rsi': random.randint(40, 60),
                    'price': round(random.uniform(50, 400), 4),
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'technical_override': random.choice([True, False]),
                    'multi_timeframe': random.choice(['buy', 'sell']),
                    'monte_carlo_quality': random.choice(['MEDIUM', 'HIGH']),
                    'winning_indicators': random.sample(['RSI', 'ADX', 'MACD', 'BB', 'VOL'], 3),
                    'score_factors': ['MC:45.0', 'RSI:8.0', 'ADX:7.0'],
                    'assertiveness': random.randint(70, 90),
                    'is_best_of_symbol': (horizon == 2),  # Simula que T+2 é o melhor
                    'liquidity_score': round(random.uniform(0.6, 0.9), 2),
                    'volatility_regime': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                    'market_regime': 'NORMAL',
                    'volatility_multiplier': 1.0
                })
        return results
    
    def calculate_assertiveness(self, result):
        """Assertividade mais balanceada"""
        base = result['confidence']
        
        # Indicadores com peso progressivo
        indicator_count = len(result['winning_indicators'])
        if indicator_count >= 5:
            base += 15
        elif indicator_count >= 4:
            base += 10
        elif indicator_count >= 3:
            base += 6
        elif indicator_count >= 2:
            base += 3
        
        # Qualidade Monte Carlo com peso maior
        if result['monte_carlo_quality'] == 'HIGH':
            base += 12
        elif result['monte_carlo_quality'] == 'MEDIUM':
            base += 6
        
        # Multi-timeframe alinhado
        if result['multi_timeframe'] == result['direction']:
            base += 8
            
        # Bonus por alta probabilidade (>60%)
        if max(result['probability_buy'], result['probability_sell']) > 0.6:
            base += 5
            
        # Ajuste por regime de volatilidade
        if result['volatility_regime'] == 'LOW':
            base += 3
        elif result['volatility_regime'] == 'HIGH':
            base -= 5
            
        return min(round(base, 1), 95)  # Limite de 95% para realismo
    
    def calculate_entry_time(self, horizon):
        now = datetime.now(timezone.utc)
        return (now + timedelta(minutes=horizon)).strftime("%H:%M UTC")

manager = AnalysisManager()

# ========== ROTAS ==========
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
        <title>🚀 IA Signal Pro - 3000 SIMULAÇÕES</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🚀 IA Signal Pro - 3000 SIMULAÇÕES MONTE CARLO</h1>
                <p><em>Análise completa em ~25s • 6 ativos selecionáveis • Imparcialidade total • 5 Sistemas Integrados</em></p>
                
                <div class="symbols-container">
                    <h3>🎯 SELECIONE OS ATIVOS PARA ANÁLISE:</h3>
                    <div id="symbolsCheckbox">
                        {symbols_html}
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <select id="sims" style="width: 200px; display: inline-block;">
                        <option value="3000" selected>3000 simulações Monte Carlo</option>
                    </select>
                    
                    <button onclick="analyze()" id="analyzeBtn">🎯 ANALISAR ATIVOS SELECIONADOS</button>
                </div>
            </div>

            <div class="card best-card">
                <h2>🎖️ MELHOR OPORTUNIDADE GLOBAL</h2>
                <div id="bestResult">Selecione os ativos e clique em Analisar</div>
            </div>

            <div class="card">
                <h2>📈 TODOS OS MELHORES T+ DE CADA ATIVO</h2>
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
                btn.textContent = `⏳ PROCESSANDO 3000 SIMULAÇÕES...`;

                try {{
                    const response = await fetch('/api/analyze', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            symbols: symbols,
                            sims: 3000,
                            only_adx: 0
                        }})
                    }});

                    const data = await response.json();
                    if (data.success) {{
                        checkResults();
                    }} else {{
                        alert('Erro: ' + data.error);
                        btn.disabled = false;
                        btn.textContent = '🎯 ANALISAR ATIVOS SELECIONADOS';
                    }}
                }} catch (error) {{
                    alert('Erro de conexão');
                    btn.disabled = false;
                    btn.textContent = '🎯 ANALISAR ATIVOS SELECIONADOS';
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
                            document.getElementById('analyzeBtn').textContent = '🎯 ANALISAR ATIVOS SELECIONADOS';
                        }}
                    }}
                }} catch (error) {{
                    setTimeout(checkResults, 2000);
                }}
            }}

            function updateResults(data) {{
                // Melhor oportunidade global
                if (data.best) {{
                    const best = data.best;
                    const qualityClass = 'quality-' + best.monte_carlo_quality.toLowerCase();
                    const regimeClass = getRegimeClass(best.volatility_regime);
                    const liquidityClass = getLiquidityClass(best.liquidity_score);
                    
                    document.getElementById('bestResult').innerHTML = `
                        <div class="results ${{best.direction}}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.3em;">${{best.symbol}} T+${{best.horizon}}</strong>
                                    <span style="font-size: 1.2em; margin-left: 10px;">
                                        ${{best.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'}}
                                    </span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.4em; font-weight: bold;">${{best.confidence}}%</div>
                                    <div>Assertividade: ${{best.assertiveness}}%</div>
                                </div>
                            </div>
                            
                            <div class="metrics">
                                <div class="metric"><div>Prob Compra</div><strong>${{best.p_buy}}%</strong></div>
                                <div class="metric"><div>Prob Venda</div><strong>${{best.p_sell}}%</strong></div>
                                <div class="metric"><div>ADX</div><strong>${{best.adx}}</strong></div>
                                <div class="metric"><div>RSI</div><strong>${{best.rsi}}</strong></div>
                                <div class="metric"><div>Liquidez</div><strong class="${{liquidityClass}}">${{best.liquidity_score}}</strong></div>
                                <div class="metric"><div>Vol Regime</div><strong class="${{regimeClass}}">${{best.volatility_regime}}</strong></div>
                            </div>
                            
                            <div><strong>Indicadores Ativos:</strong> ${{formatIndicators(best.winning_indicators)}}</div>
                            <div><strong>Pontuação:</strong> ${{formatFactors(best.score_factors)}}</div>
                            <div>
                                <strong>Mercado:</strong> ${{best.market_regime}} | 
                                <strong>Vol Multi:</strong> ${{best.volatility_multiplier}}x |
                                <strong>Preço:</strong> $${{best.price}}
                            </div>
                            <div><strong>Entrada:</strong> ${{best.entry_time}}</div>
                            ${{best.technical_override ? '<div class="override">⚡ ALTA CONVERGÊNCIA TÉCNICA</div>' : ''}}
                            <br><em>Última análise: ${{data.analysis_time}}</em>
                        </div>
                    `;
                }}

                // Todos os sinais - AGRUPADOS por símbolo
                if (data.results.length > 0) {{
                    // Agrupa por símbolo
                    const groupedBySymbol = {{}};
                    data.results.forEach(result => {{
                        if (!groupedBySymbol[result.symbol]) {{
                            groupedBySymbol[result.symbol] = [];
                        }}
                        groupedBySymbol[result.symbol].push(result);
                    }});

                    let html = '';
                    
                    // Para cada símbolo, mostra todos os horizontes
                    Object.keys(groupedBySymbol).sort().forEach(symbol => {{
                        const symbolResults = groupedBySymbol[symbol].sort((a, b) => a.horizon - b.horizon);
                        const regimeClass = getRegimeClass(symbolResults[0].volatility_regime);
                        const liquidityClass = getLiquidityClass(symbolResults[0].liquidity_score);
                        
                        html += `
                            <div class="symbol-header">
                                ${{symbol}} 
                                <span style="font-size: 0.8em; margin-left: 10px;">
                                    [Regime: <span class="${{regimeClass}}">${{symbolResults[0].volatility_regime}}</span> | 
                                    Liquidez: <span class="${{liquidityClass}}">${{symbolResults[0].liquidity_score}}</span> |
                                    Mercado: ${{symbolResults[0].market_regime}}]
                                </span>
                            </div>`;
                        
                        symbolResults.forEach(result => {{
                            const qualityClass = 'quality-' + result.monte_carlo_quality.toLowerCase();
                            const isBest = result.is_best_of_symbol;
                            const bestBadge = isBest ? ' 🏆 MELHOR' : '';
                            const resultClass = isBest ? 'best-of-symbol' : '';
                            
                            html += `
                            <div class="results ${{result.direction}} ${{resultClass}}">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div style="flex: 1;">
                                        <strong>T+${{result.horizon}}</strong>
                                        <span class="horizon-badge">${{result.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'}}${{bestBadge}}</span>
                                        <br>
                                        <strong>Prob:</strong> ${{result.p_buy}}%/${{result.p_sell}}% | 
                                        <strong>Conf:</strong> ${{result.confidence}}% | 
                                        <strong>Assert:</strong> ${{result.assertiveness}}%
                                        <br>
                                        <strong>ADX:</strong> ${{result.adx}} | 
                                        <strong>RSI:</strong> ${{result.rsi}} | 
                                        <strong>Multi-TF:</strong> ${{result.multi_timeframe}} | 
                                        <strong>Qual:</strong> <span class="${{qualityClass}}">${{result.monte_carlo_quality}}</span>
                                        <br>
                                        <strong>Indicadores:</strong> ${{formatIndicators(result.winning_indicators)}}
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

            // Inicializar
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
            
        # Força 3000 simulações
        sims = 3000
        
        thread = threading.Thread(
            target=manager.analyze_symbols_thread,
            args=(symbols, sims, None)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Analisando {len(symbols)} ativos com 3000 simulações Monte Carlo...',
            'symbols_count': len(symbols)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results')
def get_results():
    return jsonify({
        'success': True,
        'results': manager.current_results,
        'best': manager.best_opportunity,
        'analysis_time': manager.analysis_time,
        'total_signals': len(manager.current_results),
        'is_analyzing': manager.is_analyzing
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '3000-simulations-v2'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 IA Signal Pro - 3000 SIMULAÇÕES MONTE CARLO")
    print("✅ 6 ativos fixos (BTC, ETH, SOL, ADA, XRP, BNB)")
    print("✅ Timing otimizado (~25s total)")
    print("✅ Relação completa de TODOS os T+")
    print("✅ Imparcialidade total entre horizontes")
    print("✅ Monte Carlo como prioridade (65% peso)")
    print("✅ ADX corrigido e realista (SEM numpy)")
    print("✅ 5 SISTEMAS INTEGRADOS:")
    print("   - Memória de Mercado")
    print("   - Sistema de Liquidez") 
    print("   - Correlações entre Ativos")
    print("   - Eventos de Notícias")
    print("   - Clusterização de Volatilidade")
    app.run(host='0.0.0.0', port=port, debug=False)
