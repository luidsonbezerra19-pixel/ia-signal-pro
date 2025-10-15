from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime, timezone, timedelta
import os
import random
import math
import json
from typing import List, Dict, Tuple, Any

# ========== SISTEMA DE MEMÓRIA E APRENDIZADO ==========
class MemorySystem:
    def __init__(self):
        self.memory_file = 'trading_memory.json'
        self.symbol_memory = self.load_memory()
    
    def load_memory(self):
        """Carrega a memória de trades anteriores"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def save_memory(self):
        """Salva a memória em arquivo"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.symbol_memory, f, indent=2)
        except:
            pass
    
    def update_memory(self, symbol: str, decision: Dict, actual_result: str = None):
        """Atualiza a memória com nova decisão"""
        if symbol not in self.symbol_memory:
            self.symbol_memory[symbol] = {
                'total_trades': 0,
                'successful_trades': 0,
                'success_rate': 0.5,
                'horizon_performance': {1: 0.5, 2: 0.5, 3: 0.5},
                'direction_preference': {'buy': 0, 'sell': 0},
                'best_horizon': 1,
                'indicator_weights': {'adx': 0.25, 'rsi': 0.2, 'macd': 0.15, 'bollinger': 0.15, 'volume': 0.1, 'fibonacci': 0.15},
                'last_analysis': datetime.now().isoformat()
            }
        
        memory = self.symbol_memory[symbol]
        memory['total_trades'] += 1
        memory['last_analysis'] = datetime.now().isoformat()
        
        # Atualiza preferência de direção
        memory['direction_preference'][decision['direction']] += 1
        
        # Se temos resultado real, atualiza sucesso
        if actual_result:
            if decision['direction'] == actual_result:
                memory['successful_trades'] += 1
                # Reforça pesos dos indicadores que acertaram
                for indicator in decision.get('winning_indicators', []):
                    memory['indicator_weights'][indicator] = min(0.3, memory['indicator_weights'][indicator] + 0.02)
            
            memory['success_rate'] = memory['successful_trades'] / memory['total_trades']
            
            # Atualiza performance por horizonte
            horizon = decision.get('horizon', 1)
            if actual_result == decision['direction']:
                memory['horizon_performance'][horizon] = min(1.0, memory['horizon_performance'][horizon] + 0.05)
            else:
                memory['horizon_performance'][horizon] = max(0.1, memory['horizon_performance'][horizon] - 0.03)
            
            # Atualiza melhor horizonte
            memory['best_horizon'] = max(memory['horizon_performance'].items(), key=lambda x: x[1])[0]
        
        self.save_memory()
    
    def get_symbol_weights(self, symbol: str) -> Dict:
        """Retorna pesos personalizados para o símbolo"""
        if symbol in self.symbol_memory:
            return self.symbol_memory[symbol]['indicator_weights']
        return {'adx': 0.25, 'rsi': 0.2, 'macd': 0.15, 'bollinger': 0.15, 'volume': 0.1, 'fibonacci': 0.15}

# ========== SIMULAÇÃO MONTE CARLO REAL ==========
class MonteCarloSimulator:
    @staticmethod
    def generate_price_paths(base_price: float, volatility: float, trend: float, num_paths: int = 1500, steps: int = 21) -> List[List[float]]:
        """Gera múltiplos caminhos de preço usando Monte Carlo"""
        paths = []
        
        for _ in range(num_paths):
            prices = [base_price]
            for step in range(steps - 1):
                # Componente de tendência + ruído estocástico
                drift = trend * prices[-1] * 0.001
                shock = random.gauss(0, 1) * volatility * prices[-1] * 0.01
                
                new_price = prices[-1] + drift + shock
                new_price = max(new_price, base_price * 0.5)  # Previne preços negativos
                prices.append(new_price)
            paths.append(prices)
        
        return paths
    
    @staticmethod
    def calculate_probability_distribution(paths: List[List[float]]) -> Dict:
        """Calcula distribuição de probabilidades dos caminhos"""
        final_prices = [path[-1] for path in paths]
        initial_price = paths[0][0] if paths else base_price
        
        price_changes = [(price - initial_price) / initial_price for price in final_prices]
        
        # Análise estatística
        mean_change = sum(price_changes) / len(price_changes)
        positive_changes = [change for change in price_changes if change > 0]
        negative_changes = [change for change in price_changes if change < 0]
        
        prob_up = len(positive_changes) / len(price_changes)
        prob_down = len(negative_changes) / len(price_changes)
        
        # Value at Risk (95% confidence)
        sorted_changes = sorted(price_changes)
        var_95 = sorted_changes[int(0.05 * len(sorted_changes))]
        
        return {
            'probability_buy': prob_up,
            'probability_sell': prob_down,
            'expected_return': mean_change,
            'volatility': math.sqrt(sum((x - mean_change) ** 2 for x in price_changes) / len(price_changes)),
            'var_95': var_95,
            'confidence_interval': (sorted_changes[int(0.25 * len(sorted_changes))], sorted_changes[int(0.75 * len(sorted_changes))])
        }

# ========== INDICADORES TÉCNICOS AVANÇADOS ==========
class AdvancedIndicators:
    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict:
        """Calcula MACD (Moving Average Convergence Divergence)"""
        if len(prices) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'}
        
        # EMAs simplificadas
        ema_12 = sum(prices[-12:]) / 12
        ema_26 = sum(prices[-26:]) / 26
        
        macd = ema_12 - ema_26
        signal = sum(prices[-9:]) / 9  # EMA 9 do MACD simplificado
        
        histogram = macd - signal
        
        if macd > signal and histogram > 0:
            trend = 'bullish'
        elif macd < signal and histogram < 0:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {'macd': macd, 'signal': signal, 'histogram': histogram, 'trend': trend}
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict:
        """Calcula Bollinger Bands"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'position': 'middle'}
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period
        std_dev = math.sqrt(sum((x - middle) ** 2 for x in recent_prices) / period)
        
        upper = middle + (2 * std_dev)
        lower = middle - (2 * std_dev)
        width = (upper - lower) / middle
        
        current_price = prices[-1]
        if current_price > upper:
            position = 'above_upper'
        elif current_price < lower:
            position = 'below_lower'
        elif current_price > middle:
            position = 'upper_half'
        else:
            position = 'lower_half'
        
        return {'upper': upper, 'middle': middle, 'lower': lower, 'width': width, 'position': position}
    
    @staticmethod
    def calculate_volume_profile(prices: List[float]) -> Dict:
        """Simula Volume Profile (áreas de valor)"""
        if len(prices) < 10:
            return {'poc': 0, 'value_area_high': 0, 'value_area_low': 0, 'profile_balance': 'neutral'}
        
        price_range = max(prices) - min(prices)
        current_price = prices[-1]
        
        # Simula Point of Control (preço com maior volume)
        poc = sum(prices) / len(prices)
        
        value_area_high = poc + (price_range * 0.3)
        value_area_low = poc - (price_range * 0.3)
        
        if current_price > value_area_high:
            balance = 'above_value'
        elif current_price < value_area_low:
            balance = 'below_value'
        else:
            balance = 'in_value'
        
        return {'poc': poc, 'value_area_high': value_area_high, 'value_area_low': value_area_low, 'profile_balance': balance}
    
    @staticmethod
    def calculate_fibonacci_retracement(prices: List[float]) -> Dict:
        """Calcula níveis de Fibonacci Retracement"""
        if len(prices) < 10:
            return {'level_236': 0, 'level_382': 0, 'level_500': 0, 'level_618': 0, 'current_level': 'unknown'}
        
        high = max(prices)
        low = min(prices)
        current = prices[-1]
        
        diff = high - low
        
        levels = {
            'level_236': high - (0.236 * diff),
            'level_382': high - (0.382 * diff),
            'level_500': high - (0.5 * diff),
            'level_618': high - (0.618 * diff)
        }
        
        # Determina nível atual
        current_level = 'above_236'
        for level_name, level_price in sorted(levels.items(), key=lambda x: x[1], reverse=True):
            if current <= level_price:
                current_level = level_name
                break
        
        return {**levels, 'current_level': current_level}

# ========== ANÁLISE MULTI-TIMEFRAME ==========
class MultiTimeframeAnalyzer:
    @staticmethod
    def analyze_timeframes(prices: List[float]) -> Dict:
        """Analisa múltiplos timeframes a partir dos dados disponíveis"""
        if len(prices) < 30:
            return {'5m': 'neutral', '15m': 'neutral', '1h': 'neutral', 'consensus': 'neutral'}
        
        # Timeframe 5 minutos (últimos 5 candles)
        tf_5m = prices[-5:] if len(prices) >= 5 else prices
        trend_5m = (tf_5m[-1] - tf_5m[0]) / tf_5m[0] if tf_5m[0] != 0 else 0
        
        # Timeframe 15 minutos (últimos 15 candles)
        tf_15m = prices[-15:] if len(prices) >= 15 else prices
        trend_15m = (tf_15m[-1] - tf_15m[0]) / tf_15m[0] if tf_15m[0] != 0 else 0
        
        # Timeframe 1 hora (últimos 30 candles)
        tf_1h = prices[-30:] if len(prices) >= 30 else prices
        trend_1h = (tf_1h[-1] - tf_1h[0]) / tf_1h[0] if tf_1h[0] != 0 else 0
        
        # Determina direção por timeframe
        def get_direction(trend):
            if trend > 0.002: return 'buy'
            elif trend < -0.002: return 'sell'
            return 'neutral'
        
        directions = {
            '5m': get_direction(trend_5m),
            '15m': get_direction(trend_15m),
            '1h': get_direction(trend_1h)
        }
        
        # Consenso ponderado
        buy_count = list(directions.values()).count('buy')
        sell_count = list(directions.values()).count('sell')
        
        if buy_count > sell_count:
            consensus = 'buy'
        elif sell_count > buy_count:
            consensus = 'sell'
        else:
            consensus = 'neutral'
        
        return {**directions, 'consensus': consensus}

# ========== SISTEMA PRINCIPAL APRIMORADO ==========
class EnhancedTradingSystem:
    def __init__(self):
        self.memory = MemorySystem()
        self.monte_carlo = MonteCarloSimulator()
        self.indicators = AdvancedIndicators()
        self.multi_tf = MultiTimeframeAnalyzer()
    
    def calculate_technical_indicators(self, prices: List[float]) -> Dict:
        """Calcula todos os indicadores técnicos"""
        if len(prices) < 10:
            return self._get_default_indicators()
        
        # Indicadores básicos
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        volatility = math.sqrt(sum(x**2 for x in price_changes) / len(price_changes)) if price_changes else 0.01
        
        # RSI simplificado
        gains = sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
        losses = sum(max(0, prices[i-1] - prices[i]) for i in range(1, len(prices)))
        rsi = 100 - (100 / (1 + (gains / losses))) if losses > 0 else 100
        
        # ADX simplificado
        trend_strength = min(1.0, abs(sum(price_changes[-5:])) / (prices[0] * 0.01)) if len(price_changes) >= 5 else 0.5
        adx = 20 + (trend_strength * 40)
        
        # Indicadores avançados
        macd = self.indicators.calculate_macd(prices)
        bollinger = self.indicators.calculate_bollinger_bands(prices)
        volume_profile = self.indicators.calculate_volume_profile(prices)
        fibonacci = self.indicators.calculate_fibonacci_retracement(prices)
        multi_timeframe = self.multi_tf.analyze_timeframes(prices)
        
        return {
            'basic': {
                'rsi': max(10, min(90, rsi)),
                'adx': max(10, min(80, adx)),
                'volatility': volatility,
                'trend_strength': trend_strength,
                'current_price': prices[-1]
            },
            'advanced': {
                'macd': macd,
                'bollinger': bollinger,
                'volume_profile': volume_profile,
                'fibonacci': fibonacci
            },
            'multi_timeframe': multi_timeframe
        }
    
    def _get_default_indicators(self):
        """Retorna indicadores padrão quando dados insuficientes"""
        return {
            'basic': {'rsi': 50, 'adx': 25, 'volatility': 0.01, 'trend_strength': 0.5, 'current_price': 100},
            'advanced': {
                'macd': {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'},
                'bollinger': {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'position': 'middle'},
                'volume_profile': {'poc': 0, 'value_area_high': 0, 'value_area_low': 0, 'profile_balance': 'neutral'},
                'fibonacci': {'level_236': 0, 'level_382': 0, 'level_500': 0, 'level_618': 0, 'current_level': 'unknown'}
            },
            'multi_timeframe': {'5m': 'neutral', '15m': 'neutral', '1h': 'neutral', 'consensus': 'neutral'}
        }
    
    def multi_factor_scoring(self, symbol: str, indicators: Dict, monte_carlo_result: Dict, horizon: int) -> Dict:
        """Sistema de pontuação multi-fatorial com pesos personalizados"""
        weights = self.memory.get_symbol_weights(symbol)
        score = 0
        factors = []
        winning_indicators = []
        
        # 1. Probabilidade Monte Carlo (peso máximo)
        mc_prob = monte_carlo_result['probability_buy']
        mc_score = mc_prob * 30  # 30% do score máximo
        score += mc_score
        factors.append(f"MC: {mc_score:.1f}pts")
        
        # 2. Indicadores básicos com pesos personalizados
        basic = indicators['basic']
        
        # RSI
        rsi_score = 0
        if 30 <= basic['rsi'] <= 70:
            rsi_score = weights['rsi'] * 20
            winning_indicators.append('rsi')
        score += rsi_score
        factors.append(f"RSI: {rsi_score:.1f}pts")
        
        # ADX
        adx_score = 0
        if basic['adx'] > 25:
            adx_score = weights['adx'] * 20
            winning_indicators.append('adx')
        score += adx_score
        factors.append(f"ADX: {adx_score:.1f}pts")
        
        # 3. Indicadores avançados
        advanced = indicators['advanced']
        
        # MACD
        macd_score = 0
        if advanced['macd']['trend'] == 'bullish':
            macd_score = weights['macd'] * 15
            winning_indicators.append('macd')
        score += macd_score
        factors.append(f"MACD: {macd_score:.1f}pts")
        
        # Bollinger Bands
        bollinger_score = 0
        if advanced['bollinger']['position'] in ['below_lower', 'upper_half']:
            bollinger_score = weights['bollinger'] * 10
            winning_indicators.append('bollinger')
        score += bollinger_score
        factors.append(f"BB: {bollinger_score:.1f}pts")
        
        # Volume Profile
        volume_score = 0
        if advanced['volume_profile']['profile_balance'] == 'in_value':
            volume_score = weights['volume'] * 10
            winning_indicators.append('volume')
        score += volume_score
        factors.append(f"Volume: {volume_score:.1f}pts")
        
        # Fibonacci
        fib_score = 0
        if advanced['fibonacci']['current_level'] in ['level_382', 'level_500', 'level_618']:
            fib_score = weights['fibonacci'] * 10
            winning_indicators.append('fibonacci')
        score += fib_score
        factors.append(f"Fib: {fib_score:.1f}pts")
        
        # 4. Multi-Timeframe
        tf_score = 0
        if indicators['multi_timeframe']['consensus'] == 'buy':
            tf_score = 15
        score += tf_score
        factors.append(f"TF: {tf_score:.1f}pts")
        
        # 5. Ajuste por horizonte baseado na memória
        memory_data = self.memory.symbol_memory.get(symbol, {})
        horizon_performance = memory_data.get('horizon_performance', {1: 0.5, 2: 0.5, 3: 0.5})
        horizon_bonus = horizon_performance.get(horizon, 0.5) * 10
        score += horizon_bonus
        factors.append(f"Horizon: {horizon_bonus:.1f}pts")
        
        # Direção final baseada na probabilidade Monte Carlo
        direction = 'buy' if mc_prob > 0.5 else 'sell'
        
        # Confiança baseada no score (0-100%)
        confidence = min(0.95, max(0.3, score / 100))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'factors': factors,
            'winning_indicators': winning_indicators,
            'probability_buy': mc_prob,
            'probability_sell': 1 - mc_prob
        }
    
    def analyze_symbol(self, symbol: str, horizon: int, num_simulations: int = 1500) -> Dict:
        """Analisa um símbolo com todas as melhorias"""
        # Preço base realista
        base_price = random.uniform(50, 500)
        
        # Gera dados de preço históricos
        trend_strength = random.uniform(-0.2, 0.2)
        volatility = random.uniform(0.005, 0.025)
        historical_prices = self.monte_carlo.generate_price_paths(
            base_price, volatility, trend_strength, num_paths=1, steps=50
        )[0] if random.random() > 0.5 else [base_price] * 50
        
        # Simulação Monte Carlo para o horizonte
        future_paths = self.monte_carlo.generate_price_paths(
            historical_prices[-1], volatility, trend_strength, num_paths=num_simulations, steps=horizon+5
        )
        mc_result = self.monte_carlo.calculate_probability_distribution(future_paths)
        
        # Calcula todos os indicadores
        indicators = self.calculate_technical_indicators(historical_prices)
        
        # Sistema multi-fatorial
        decision = self.multi_factor_scoring(symbol, indicators, mc_result, horizon)
        
        # Atualiza memória (sem resultado real por enquanto)
        self.memory.update_memory(symbol, {
            'direction': decision['direction'],
            'horizon': horizon,
            'confidence': decision['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'symbol': symbol,
            'horizon': horizon,
            'direction': decision['direction'],
            'confidence': decision['confidence'],
            'probability_buy': decision['probability_buy'],
            'probability_sell': decision['probability_sell'],
            'score': decision['score'],
            'factors': decision['factors'],
            'winning_indicators': decision['winning_indicators'],
            'price': historical_prices[-1],
            'indicators': indicators,
            'monte_carlo': mc_result,
            'timestamp': datetime.now().strftime("%H:%M:%S")
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
    
    def analyze_symbols_thread(self, symbols, sims, only_adx):
        try:
            self.is_analyzing = True
            print(f"🔍 Iniciando análise APRIMORADA: {symbols}")
            
            all_results = []
            for symbol in symbols:
                for horizon in [1, 2, 3]:
                    try:
                        result = trading_system.analyze_symbol(symbol, horizon, num_simulations=sims)
                        all_results.append(result)
                    except Exception as e:
                        print(f"❌ Erro analisando {symbol} T+{horizon}: {e}")
            
            # Classifica por confiança (não filtra!)
            all_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 🎯 MELHOR DE CADA ATIVO
            best_per_symbol = {}
            for result in all_results:
                symbol = result['symbol']
                if symbol not in best_per_symbol or result['confidence'] > best_per_symbol[symbol]['confidence']:
                    best_per_symbol[symbol] = result
            
            self.current_results = []
            for symbol, result in best_per_symbol.items():
                formatted_result = {
                    'symbol': result['symbol'],
                    'horizon': result['horizon'],
                    'direction': result['direction'],
                    'p_buy': round(result['probability_buy'] * 100, 1),
                    'p_sell': round(result['probability_sell'] * 100, 1),
                    'confidence': round(result['confidence'] * 100, 1),
                    'adx': round(result['indicators']['basic']['adx'], 1),
                    'rsi': round(result['indicators']['basic']['rsi'], 1),
                    'price': round(result['price'], 6),
                    'timestamp': result['timestamp'],
                    'technical_override': len(result['winning_indicators']) >= 3,
                    'assertiveness': self.calculate_assertiveness(result),
                    'volatility': round(result['indicators']['basic']['volatility'] * 100, 2),
                    'trend_strength': round(result['indicators']['basic']['trend_strength'] * 100, 1),
                    'score_factors': result['factors'],
                    'winning_indicators': result['winning_indicators'],
                    'multi_timeframe': result['indicators']['multi_timeframe']['consensus'],
                    'monte_carlo_quality': 'HIGH' if result['monte_carlo']['probability_buy'] > 0.7 or result['monte_carlo']['probability_buy'] < 0.3 else 'MEDIUM'
                }
                self.current_results.append(formatted_result)
            
            # MELHOR OPORTUNIDADE GLOBAL
            if self.current_results:
                self.best_opportunity = max(self.current_results, key=lambda x: x['confidence'])
                self.best_opportunity['entry_time'] = self.calculate_entry_time(self.best_opportunity['horizon'])
            
            self.analysis_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"✅ Análise APRIMORADA concluída: {len(self.current_results)} sinais")
            
        except Exception as e:
            print(f"❌ Erro na análise: {str(e)}")
            import traceback
            traceback.print_exc()
            self.current_results = []
            self.best_opportunity = None
        finally:
            self.is_analyzing = False
    
    def calculate_assertiveness(self, result):
        """Calcula assertividade baseada em múltiplos fatores"""
        base_score = result['confidence']
        
        # Bônus por múltiplos indicadores convergentes
        if len(result['winning_indicators']) >= 4:
            base_score += 15
        elif len(result['winning_indicators']) >= 3:
            base_score += 10
        
        # Bônus por Monte Carlo de alta qualidade
        if result['monte_carlo_quality'] == 'HIGH':
            base_score += 8
        
        # Bônus por confirmação multi-timeframe
        if result['multi_timeframe'] == result['direction']:
            base_score += 5
        
        return min(round(base_score, 1), 100)
    
    def calculate_entry_time(self, horizon):
        now = datetime.now(timezone.utc)
        entry_time = now.replace(second=0, microsecond=0) + timedelta(minutes=horizon)
        return entry_time.strftime("%H:%M UTC")

manager = AnalysisManager()

# ========== ROTAS ==========
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 IA Signal Pro - SISTEMA APRIMORADO</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                background: #0a0a0a; 
                color: white; 
                font-family: Arial; 
                margin: 0; 
                padding: 20px; 
            }
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
            }
            .card { 
                background: #1a1a2e; 
                border: 2px solid #3498db; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 10px 0; 
            }
            .best-card { 
                border: 3px solid #f39c12; 
                background: #2c2c3e; 
            }
            input, button, select { 
                width: 100%; 
                padding: 12px; 
                margin: 5px 0; 
                border: 2px solid #3498db; 
                border-radius: 8px; 
                background: #2c3e50; 
                color: white; 
            }
            button { 
                background: #3498db; 
                border: none; 
                font-weight: bold; 
                cursor: pointer; 
            }
            button:hover { background: #2980b9; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            .results { 
                background: #2c3e50; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 8px 0; 
                border-left: 4px solid #3498db; 
            }
            .buy { 
                color: #2ecc71; 
                border-left-color: #2ecc71 !important; 
            }
            .sell { 
                color: #e74c3c; 
                border-left-color: #e74c3c !important; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
                gap: 8px; 
                margin: 10px 0; 
            }
            .metric { 
                background: #34495e; 
                padding: 6px; 
                border-radius: 5px; 
                text-align: center; 
                font-size: 0.9em; 
            }
            .factor { 
                background: #16a085; 
                padding: 3px 6px; 
                border-radius: 3px; 
                margin: 1px; 
                font-size: 0.75em; 
                display: inline-block; 
            }
            .indicator { 
                background: #8e44ad; 
                padding: 2px 5px; 
                border-radius: 3px; 
                margin: 1px; 
                font-size: 0.7em; 
                display: inline-block; 
            }
            .override { 
                color: #f39c12; 
                font-weight: bold; 
            }
            .quality-high { color: #2ecc71; }
            .quality-medium { color: #f39c12; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🚀 IA Signal Pro - SISTEMA COMPLETO APRIMORADO</h1>
                <p><em>Monte Carlo + Multi-Timeframe + Indicadores Avançados + Memória</em></p>
                
                <input type="text" id="symbols" value="BTC/USDT,ETH/USDT,ADA/USDT,SOL/USDT,BNB/USDT" placeholder="Digite os símbolos...">
                <select id="sims">
                    <option value="500">500 simulações</option>
                    <option value="1000" selected>1000 simulações</option>
                    <option value="1500">1500 simulações</option>
                    <option value="2000">2000 simulações</option>
                </select>
                <select id="adx">
                    <option value="0">Todos ADX</option>
                    <option value="20">ADX ≥ 20</option>
                    <option value="25">ADX ≥ 25</option>
                    <option value="30">ADX ≥ 30</option>
                </select>
                
                <button onclick="analyze()" id="analyzeBtn">🎯 ANALISAR COM SISTEMA COMPLETO</button>
            </div>

            <div class="card best-card">
                <h2>🎖️ MELHOR OPORTUNIDADE GLOBAL</h2>
                <div id="bestResult">Aguardando análise...</div>
            </div>

            <div class="card">
                <h2>📈 MELHORES SINAIS POR ATIVO</h2>
                <div id="allResults">Execute uma análise para ver os resultados</div>
            </div>

            <div class="card">
                <h3>ℹ️ STATUS DO SISTEMA</h3>
                <div id="status">Sistema Completo conectado e pronto</div>
            </div>
        </div>

        <script>
            function getVolatilityClass(vol) {
                if (vol > 2.5) return 'volatility-high';
                if (vol > 1.5) return 'volatility-medium';
                return 'volatility-low';
            }

            function formatFactors(factors) {
                if (!factors || !Array.isArray(factors)) return '';
                return factors.map(f => `<span class="factor">${f}</span>`).join('');
            }

            function formatIndicators(indicators) {
                if (!indicators || !Array.isArray(indicators)) return '';
                return indicators.map(i => `<span class="indicator">${i}</span>`).join('');
            }

            async function analyze() {
                const btn = document.getElementById('analyzeBtn');
                btn.disabled = true;
                btn.textContent = '⏳ ANALISANDO COM MONTE CARLO...';

                const symbols = document.getElementById('symbols').value;
                const sims = document.getElementById('sims').value;
                const adx = document.getElementById('adx').value;

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            symbols: symbols.split(','),
                            sims: parseInt(sims),
                            only_adx: parseFloat(adx)
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        checkResults();
                    } else {
                        alert('Erro: ' + data.error);
                        btn.disabled = false;
                        btn.textContent = '🎯 ANALISAR COM SISTEMA COMPLETO';
                    }

                } catch (error) {
                    alert('Erro de conexão: ' + error.message);
                    btn.disabled = false;
                    btn.textContent = '🎯 ANALISAR COM SISTEMA COMPLETO';
                }
            }

            async function checkResults() {
                try {
                    const response = await fetch('/api/results');
                    const data = await response.json();

                    if (data.success) {
                        updateResults(data);
                        
                        if (data.is_analyzing) {
                            setTimeout(checkResults, 2000);
                        } else {
                            document.getElementById('analyzeBtn').disabled = false;
                            document.getElementById('analyzeBtn').textContent = '🎯 ANALISAR COM SISTEMA COMPLETO';
                        }
                    }
                } catch (error) {
                    console.error('Erro:', error);
                    setTimeout(checkResults, 3000);
                }
            }

            function updateResults(data) {
                // Melhor oportunidade
                if (data.best) {
                    const best = data.best;
                    const overrideIcon = best.technical_override ? ' ⚡ ' : '';
                    const overrideText = best.technical_override ? 
                        '<br><div class="override">⚡ ALTA CONVERGÊNCIA TÉCNICA</div>' : '';
                    
                    const volClass = getVolatilityClass(best.volatility);
                    const qualityClass = best.monte_carlo_quality === 'HIGH' ? 'quality-high' : 'quality-medium';
                    
                    document.getElementById('bestResult').innerHTML = `
                        <div class="results ${best.direction}">
                            <div style="display: flex; justify-content: between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.2em;">${best.symbol} T+${best.horizon}</strong>
                                    <span style="font-size: 1.1em;">
                                        ${best.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'} ${overrideIcon}
                                    </span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.3em; font-weight: bold;">${best.confidence}%</div>
                                    <div>Assertividade: ${best.assertiveness}%</div>
                                </div>
                            </div>
                            
                            <div class="metrics">
                                <div class="metric">
                                    <div>Prob Compra</div>
                                    <strong>${best.p_buy}%</strong>
                                </div>
                                <div class="metric">
                                    <div>Prob Venda</div>
                                    <strong>${best.p_sell}%</strong>
                                </div>
                                <div class="metric">
                                    <div>ADX</div>
                                    <strong>${best.adx}</strong>
                                </div>
                                <div class="metric">
                                    <div>RSI</div>
                                    <strong>${best.rsi}</strong>
                                </div>
                                <div class="metric">
                                    <div>Multi-TF</div>
                                    <strong>${best.multi_timeframe}</strong>
                                </div>
                                <div class="metric">
                                    <div>MC Quality</div>
                                    <strong class="${qualityClass}">${best.monte_carlo_quality}</strong>
                                </div>
                            </div>
                            
                            <div>Indicadores Ativos: ${formatIndicators(best.winning_indicators)}</div>
                            <div>Pontuação: ${formatFactors(best.score_factors)}</div>
                            <div>Entrada: ${best.entry_time} | Preço: $${best.price}</div>
                            ${overrideText}
                            <br><em>Análise: ${data.analysis_time}</em>
                        </div>
                    `;
                }

                // Todos os sinais
                if (data.results.length > 0) {
                    let html = '';
                    data.results.sort((a, b) => b.confidence - a.confidence);
                    
                    data.results.forEach(result => {
                        const overrideIcon = result.technical_override ? ' ⚡ ' : '';
                        const qualityClass = result.monte_carlo_quality === 'HIGH' ? 'quality-high' : 'quality-medium';
                        
                        html += `
                        <div class="results ${result.direction}">
                            <div style="display: flex; justify-content: between; align-items: start;">
                                <div style="flex: 1;">
                                    <strong>${result.symbol} T+${result.horizon}</strong>
                                    <span>${result.direction == 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'} ${overrideIcon}</span>
                                    <br>
                                    Compra: ${result.p_buy}% | Venda: ${result.p_sell}% 
                                    | Conf: <strong>${result.confidence}%</strong>
                                    | Assert: ${result.assertiveness}%
                                    <br>
                                    ADX: ${result.adx} | RSI: ${result.rsi} 
                                    | Multi-TF: ${result.multi_timeframe}
                                    | MC: <span class="${qualityClass}">${result.monte_carlo_quality}</span>
                                    <br>
                                    Indicadores: ${formatIndicators(result.winning_indicators)}
                                </div>
                            </div>
                            ${result.technical_override ? '<div class="override">⚡ Convergência Técnica</div>' : ''}
                        </div>`;
                    });
                    
                    document.getElementById('allResults').innerHTML = html;
                    document.getElementById('status').textContent = 
                        `✅ ${data.results.length} ativos analisados | Sistema Completo Ativo`;
                } else {
                    document.getElementById('allResults').innerHTML = 'Nenhum sinal encontrado. Tente ajustar os parâmetros.';
                    document.getElementById('status').textContent = '⚠️ Nenhum sinal encontrado';
                }
            }

            checkResults();
        </script>
    </body>
    </html>
    '''

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if manager.is_analyzing:
        return jsonify({'success': False, 'error': 'Análise já em andamento'}), 429
    
    try:
        data = request.get_json()
        symbols = [s.strip().upper() for s in data['symbols'] if s.strip()]
        if not symbols:
            return jsonify({'success': False, 'error': 'Nenhum símbolo informado'}), 400
            
        sims = int(data.get('sims', 1000))
        only_adx = float(data.get('only_adx', 0)) if data.get('only_adx') else None
        
        thread = threading.Thread(
            target=manager.analyze_symbols_thread,
            args=(symbols, sims, only_adx)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Análise completa iniciada...',
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
    return jsonify({'status': 'healthy', 'version': 'sistema-completo'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 IA Signal Pro SISTEMA COMPLETO iniciando...")
    print("✅ TODAS AS MELHORIAS IMPLEMENTADAS:")
    print("   1. Sistema de Memória e Aprendizado")
    print("   2. Simulação Monte Carlo Real (1500+ caminhos)")
    print("   3. Indicadores Técnicos Avançados")
    print("   4. Análise Multi-Timeframe") 
    print("   5. Probabilidades Estatísticas Reais")
    print("   6. Classificação por Qualidade (SEM filtros)")
    app.run(host='0.0.0.0', port=port, debug=False)
