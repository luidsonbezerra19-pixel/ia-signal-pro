# app.py — RODA IA COM GARCH EXPANDIDO + IA AVANÇADA
from __future__ import annotations
import os, re, time, math, random, threading, json, statistics as stats
from typing import Any, Dict, List, Tuple, Optional, Deque
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import structlog
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

# =========================
# Configuração de Logging Estruturado
# =========================
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
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
# Config (com GARCH Expandido - MODIFICADO)
# =========================
TZ_STR = "America/Maceio"
MC_PATHS = 5000  # ✅ AUMENTADO para 5000 simulações
USE_CLOSED_ONLY = True
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,XRP/USDT,BNB/USDT".split(",")
DEFAULT_SYMBOLS = [s.strip().upper() for s in DEFAULT_SYMBOLS if s.strip()]

USE_WS = 1
WS_BUFFER_MINUTES = 720
WS_SYMBOLS = DEFAULT_SYMBOLS[:]
REALTIME_PROVIDER = "okx"

OKX_URL = "wss://ws.okx.com:8443/ws/v5/business"
OKX_CHANNEL = "candle1m"

COINAPI_KEY = "COLE_SUA_COINAPI_KEY_AQUI"
COINAPI_URL = "wss://ws.coinapi.io/v1/"
COINAPI_PERIOD = "1MIN"

app = Flask(__name__)
CORS(app)

# =========================
# NOVO: Sistema de GARCH Expandido - MODIFICADO
# =========================

class ExpandedGARCHSystem:
    def __init__(self):
        self.horizons = [1]  # ✅ APENAS T+1
        self.paths_per_horizon = MC_PATHS  # ✅ 5000 simulações
        
    def run_multi_horizon_garch(self, base_price: float, returns: List[float]) -> Dict:
        """Executa GARCH apenas para T+1 com 5000 simulações"""
        horizon_results = {}
        
        for horizon in self.horizons:  # ✅ Apenas T+1
            result = self.simulate_garch11_single(
                base_price, returns, horizon, self.paths_per_horizon
            )
            horizon_results[f"T{horizon}"] = {
                'probability_buy': result['probability_buy'],
                'probability_sell': result['probability_sell'],
                'volatility_forecast': result.get('volatility_forecast', 0.02),
                'confidence': result.get('fit_quality', 0.7),
                'garch_params': result.get('garch_params', {}),
                'market_regime': result.get('market_regime', 'normal')
            }
        
        return horizon_results
    
    def simulate_garch11_single(self, base_price: float, returns: List[float], 
                               steps: int, num_paths: int) -> Dict[str, Any]:
        """Versão simplificada do GARCH para múltiplas execuções"""
        import math
        import random
        
        if not returns or len(returns) < 10:
            returns = [random.gauss(0.0, 0.002) for _ in range(100)]
        
        # Parâmetros GARCH adaptativos
        volatility = stats.stdev(returns) if len(returns) > 1 else 0.02
        if volatility > 0.03:
            omega, alpha, beta = 1e-5, 0.12, 0.80
        elif volatility < 0.01:
            omega, alpha, beta = 1e-6, 0.05, 0.92
        else:
            omega, alpha, beta = 1e-6, 0.08, 0.90
            
        h_last = volatility ** 2
        
        up_count = 0
        total_count = 0
        
        for _ in range(num_paths):
            try:
                h = h_last
                price = base_price
                
                for step in range(steps):
                    epsilon = math.sqrt(h) * random.gauss(0.0, 1.0)
                    price *= math.exp(epsilon)
                    h = omega + alpha * (epsilon ** 2) + beta * h
                    h = max(1e-12, h)
                
                total_count += 1
                if price > base_price:
                    up_count += 1
                    
            except Exception:
                continue
        
        if total_count == 0:
            prob_buy = 0.5
        else:
            prob_buy = up_count / total_count
        
        prob_buy = min(0.95, max(0.05, prob_buy))
        
        return {
            "probability_buy": prob_buy,
            "probability_sell": 1.0 - prob_buy,
            "paths_used": total_count,
            "garch_params": {"omega": omega, "alpha": alpha, "beta": beta},
            "market_regime": "high_volatility" if volatility > 0.03 else "low_volatility" if volatility < 0.01 else "normal",
            "fit_quality": 0.7 + (min(volatility, 0.05) / 0.05 * 0.3)  # Qualidade baseada na volatilidade
        }

# =========================
# NOVO: Agregador Inteligente de Sinais - MODIFICADO
# =========================

class IntelligentSignalAggregator:
    def __init__(self):
        self.min_trajectory_quality = 0.45  # ✅ REDUZIDO para mais sinais
        self.quality_filter = SignalQualityFilter()
        
    def aggregate_signals(self, symbol: str, multi_horizon_data: Dict, 
                         technical_data: Dict, trajectory_analysis: Dict) -> List[Dict]:
        """Agrega sinais apenas do T+1"""
        signals = []
        
        # ✅ Apenas T+1
        horizon = "T1"
        if horizon in multi_horizon_data:
            garch_data = multi_horizon_data[horizon]
            base_signal = self._create_base_signal(symbol, horizon, garch_data, technical_data)
            
            # ✅ Filtros mais permissivos
            if self._passes_expanded_filters(base_signal, trajectory_analysis):
                enhanced_signal = self._enhance_with_trajectory(
                    base_signal, trajectory_analysis
                )
                signals.append(enhanced_signal)
                    
        return signals
    
    def _create_base_signal(self, symbol: str, horizon: str, garch_data: Dict, 
                           technical_data: Dict) -> Dict:
        """Cria sinal base com dados técnicos e GARCH"""
        horizon_num = int(horizon[1:])  # Remove "T" do horizonte
        
        prob_buy = garch_data['probability_buy']
        direction = 'buy' if prob_buy > 0.5 else 'sell'
        prob_directional = prob_buy if direction == 'buy' else garch_data['probability_sell']
        
        # Confiança base combinada
        base_confidence = min(0.95, max(0.3, 
            prob_directional * 0.6 + 
            technical_data.get('liquidity_score', 0.5) * 0.2 +
            garch_data['confidence'] * 0.2
        ))
        
        return {
            'symbol': symbol,
            'horizon': horizon_num,
            'direction': direction,
            'probability_buy': prob_buy,
            'probability_sell': garch_data['probability_sell'],
            'confidence': base_confidence,
            'rsi': technical_data.get('rsi', 50),
            'adx': technical_data.get('adx', 20),
            'liquidity_score': technical_data.get('liquidity_score', 0.5),
            'multi_timeframe': technical_data.get('multi_timeframe', 'neutral'),
            'price': technical_data.get('price', 0),
            'garch_confidence': garch_data['confidence'],
            'market_regime': garch_data.get('market_regime', 'normal'),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
    
    def _passes_expanded_filters(self, signal: Dict, trajectory: Dict) -> bool:
        """Filtros expandidos - MAIS PERMISSIVOS"""
        base_checks = [
            signal.get('confidence', 0) > 0.45,  # ✅ REDUZIDO de 0.55 para 0.45
            signal.get('liquidity_score', 0) > 0.25,  # ✅ REDUZIDO
            trajectory['trajectory_quality'] > self.min_trajectory_quality,
            signal.get('probability_buy', 0.5) > 0.35,  # ✅ REDUZIDO
            signal.get('probability_buy', 0.5) < 0.92   # ✅ AUMENTADO
        ]
        
        # ✅ Requer apenas 3 de 5 critérios (mais permissivo)
        return sum(base_checks) >= 3
    
    def _enhance_with_trajectory(self, signal: Dict, trajectory: Dict) -> Dict:
        """Aprimora sinal com análise de trajetória"""
        # Boost de confiança baseado na qualidade da trajetória
        trajectory_boost = trajectory['trajectory_quality'] * 0.2
        enhanced_confidence = min(0.95, signal['confidence'] + trajectory_boost)
        
        signal.update({
            'intelligent_confidence': enhanced_confidence,
            'trajectory_analysis': trajectory,
            'is_trajectory_enhanced': True,
            'recommended_horizon': trajectory['recommended_horizon'],
            'trajectory_quality': trajectory['trajectory_quality']
        })
        
        return signal

# =========================
# Feature Flags Atualizadas - MODIFICADO
# =========================
FEATURE_FLAGS = {
    "enable_adaptive_garch": True,
    "enable_smart_cache": True,
    "enable_circuit_breaker": True,
    "websocket_provider": "okx",
    "maintenance_mode": False,
    "enable_ai_intelligence": True,
    "enable_learning": True,
    "enable_self_check": True,
    "enable_expanded_garch": True,
    "enable_trajectory_analysis": True,
    "focus_t1_only": True  # ✅ NOVO: Foco apenas no T+1
}

# =========================
# Sistema de Inteligência Avançada (MANTIDO ORIGINAL)
# =========================

class MarketMemory:
    """Memória de padrões de mercado e sua efetividade"""
    
    def __init__(self, max_patterns: int = 1000):
        self.pattern_success: Dict[str, float] = {}
        self.regime_patterns: Dict[str, Dict[str, float]] = {}
        self.false_signals: set = set()
        self.recent_outcomes: Deque[Tuple[str, bool]] = deque(maxlen=500)
        
    def _extract_pattern_key(self, signal: Dict) -> str:
        """Extrai chave única para identificar padrão"""
        elements = [
            f"rsi_{int(signal.get('rsi', 0))}",
            f"adx_{int(signal.get('adx', 0))}",
            f"macd_{signal.get('macd_signal', 'neutral')}",
            f"boll_{signal.get('boll_signal', 'neutral')}",
            f"rev_{signal.get('reversal', False)}",
            f"liq_{signal.get('liquidity_score', 0):.1f}"
        ]
        return "|".join(elements)
    
    def learn_from_signal(self, signal: Dict, actual_outcome: bool, price_movement: float):
        """Aprende com resultado real do sinal"""
        pattern_key = self._extract_pattern_key(signal)
        self.recent_outcomes.append((pattern_key, actual_outcome))
        
        # Atualiza efetividade do padrão
        if pattern_key not in self.pattern_success:
            self.pattern_success[pattern_key] = 0.5
            
        adjustment = 0.1 if actual_outcome else -0.1
        self.pattern_success[pattern_key] = max(0.1, min(0.9, 
            self.pattern_success[pattern_key] + adjustment))
        
        logger.debug("pattern_learned", 
                    pattern=pattern_key, 
                    success_rate=self.pattern_success[pattern_key],
                    outcome=actual_outcome)
    
    def get_pattern_effectiveness(self, signal: Dict) -> float:
        """Retorna efetividade histórica do padrão"""
        pattern_key = self._extract_pattern_key(signal)
        return self.pattern_success.get(pattern_key, 0.5)
    
    def get_recent_accuracy(self) -> float:
        """Calcula acurácia recente geral"""
        if not self.recent_outcomes:
            return 0.5
        successes = sum(1 for _, outcome in self.recent_outcomes if outcome)
        return successes / len(self.recent_outcomes)

class AdaptiveIntelligence:
    """Sistema de adaptação e calibração contínua"""
    
    def __init__(self):
        self.confidence_calibration: Dict[str, float] = {}
        self.regime_effectiveness: Dict[str, float] = {
            "high_volatility": 0.5,
            "low_volatility": 0.5, 
            "trending": 0.5,
            "ranging": 0.5
        }
        self.performance_history: Deque[bool] = deque(maxlen=100)
        
    def calibrate_confidence(self, signal: Dict, actual_outcome: bool):
        """Calibra confiança baseado em resultados reais"""
        regime = signal.get('market_regime', 'normal')
        direction = signal.get('direction', 'neutral')
        
        key = f"{regime}_{direction}"
        current_calibration = self.confidence_calibration.get(key, 1.0)
        
        # Ajusta calibration factor
        if actual_outcome:
            new_calibration = min(1.5, current_calibration * 1.05)
        else:
            new_calibration = max(0.5, current_calibration * 0.95)
            
        self.confidence_calibration[key] = new_calibration
        self.performance_history.append(actual_outcome)
        
    def get_confidence_multiplier(self, signal: Dict) -> float:
        """Retorna multiplicador de confiança calibrado"""
        regime = signal.get('market_regime', 'normal')
        direction = signal.get('direction', 'neutral')
        key = f"{regime}_{direction}"
        return self.confidence_calibration.get(key, 1.0)
    
    def get_overall_accuracy(self) -> float:
        """Retorna acurácia geral recente"""
        if not self.performance_history:
            return 0.5
        return sum(self.performance_history) / len(self.performance_history)

class IntelligentReasoning:
    """Sistema de raciocínio multicamadas"""
    
    def __init__(self):
        self.condition_weights = {
            "high_volatility": {"rsi": 0.15, "adx": 0.25, "volume": 0.20, "garch": 0.40},
            "low_volatility": {"rsi": 0.25, "bollinger": 0.30, "garch": 0.25, "liquidity": 0.20},
            "trending": {"adx": 0.35, "macd": 0.25, "multi_tf": 0.20, "garch": 0.20},
            "ranging": {"rsi": 0.30, "bollinger": 0.35, "reversal": 0.25, "garch": 0.10}
        }
        
    def _technical_analysis(self, raw_data: Dict) -> Dict:
        """Camada 1: Análise técnica tradicional"""
        score = 0.0
        reasons = []
        
        # RSI Analysis
        rsi = raw_data.get('rsi', 50)
        if rsi < 35:
            score += 0.2
            reasons.append("RSI em oversold")
        elif rsi > 65:
            score -= 0.2
            reasons.append("RSI em overbought")
            
        # ADX Analysis
        adx = raw_data.get('adx', 20)
        if adx > 25:
            score += 0.15
            reasons.append("Tendência forte")
            
        # MACD Analysis
        macd_signal = raw_data.get('macd_signal', 'neutral')
        if macd_signal == 'bullish':
            score += 0.15
            reasons.append("MACD bullish")
        elif macd_signal == 'bearish':
            score -= 0.15
            reasons.append("MACD bearish")
            
        return {"technical_score": score, "technical_reasons": reasons}
    
    def _market_context_analysis(self, raw_data: Dict) -> Dict:
        """Camada 2: Análise de contexto de mercado"""
        score = 0.0
        reasons = []
        
        # Liquidity Analysis
        liquidity = raw_data.get('liquidity_score', 0.5)
        if liquidity > 0.7:
            score += 0.15
            reasons.append("Alta liquidez")
        elif liquidity < 0.3:
            score -= 0.1
            reasons.append("Baixa liquidez")
            
        # Reversal Analysis
        if raw_data.get('reversal', False):
            reversal_proximity = raw_data.get('reversal_proximity', 0)
            score += 0.2 * reversal_proximity
            reasons.append(f"Reversão detectada ({reversal_proximity:.1f})")
            
        # Multi-timeframe Analysis
        multi_tf = raw_data.get('multi_timeframe', 'neutral')
        if multi_tf == 'buy':
            score += 0.1
            reasons.append("Consenso multi-timeframe positivo")
        elif multi_tf == 'sell':
            score -= 0.1
            reasons.append("Consenso multi-timeframe negativo")
            
        return {"context_score": score, "context_reasons": reasons}
    
    def _pattern_recognition(self, raw_data: Dict, market_memory: MarketMemory) -> Dict:
        """Camada 3: Reconhecimento de padrões baseado em memória"""
        pattern_effectiveness = market_memory.get_pattern_effectiveness(raw_data)
        
        score_adjustment = (pattern_effectiveness - 0.5) * 0.3
        
        reasons = []
        if pattern_effectiveness > 0.6:
            reasons.append(f"Padrão historicamente efetivo ({pattern_effectiveness:.1%})")
        elif pattern_effectiveness < 0.4:
            reasons.append(f"Padrão historicamente problemático ({pattern_effectiveness:.1%})")
            
        return {
            "pattern_score": score_adjustment, 
            "pattern_reasons": reasons,
            "pattern_effectiveness": pattern_effectiveness
        }
    
    def _synthesize_decision(self, technical: Dict, context: Dict, pattern: Dict, 
                           raw_data: Dict) -> Dict:
        """Camada 4: Síntese final inteligente"""
        
        # Determina regime para pesos dinâmicos
        regime = self._determine_market_regime(raw_data)
        weights = self.condition_weights.get(regime, self.condition_weights["trending"])
        
        # Calcula score final com pesos dinâmicos
        total_score = (
            technical["technical_score"] * weights.get("rsi", 0.25) +
            context["context_score"] * weights.get("adx", 0.25) + 
            pattern["pattern_score"] * weights.get("garch", 0.25) +
            (raw_data.get('probability_buy', 0.5) - 0.5) * weights.get("garch", 0.25)
        )
        
        # Determina direção
        direction = 'buy' if total_score > 0 else 'sell'
        
        # Calcula confiança baseada na força do sinal
        base_confidence = min(0.95, max(0.3, 0.5 + abs(total_score)))
        
        # Combina todas as razões
        all_reasons = (technical["technical_reasons"] + 
                      context["context_reasons"] + 
                      pattern["pattern_reasons"])
        
        return {
            'direction': direction,
            'confidence': base_confidence,
            'reasoning': all_reasons,
            'market_regime': regime,
            'synthesis_score': total_score
        }
    
    def _determine_market_regime(self, raw_data: Dict) -> str:
        """Detecta o regime atual de mercado"""
        volatility = raw_data.get('volatility', 0.02)
        adx = raw_data.get('adx', 20)
        rsi = raw_data.get('rsi', 50)
        
        if volatility > 0.03:
            return "high_volatility"
        elif volatility < 0.01:
            return "low_volatility"
        elif adx > 25 and (rsi < 40 or rsi > 60):
            return "trending"
        else:
            return "ranging"
    
    def process(self, raw_data: Dict, market_memory: MarketMemory) -> Dict:
        """Processamento completo do raciocínio inteligente"""
        technical = self._technical_analysis(raw_data)
        context = self._market_context_analysis(raw_data)
        pattern = self._pattern_recognition(raw_data, market_memory)
        
        return self._synthesize_decision(technical, context, pattern, raw_data)

class IntelligentTradingAI:
    """IA de Trading Inteligente - Núcleo Principal"""
    
    def __init__(self):
        self.memory = MarketMemory()
        self.adaptation = AdaptiveIntelligence()
        self.reasoning = IntelligentReasoning()
        self.learning_enabled = True
        
    def analyze_with_intelligence(self, raw_analysis: Dict) -> Dict:
        """Aplica inteligência avançada à análise bruta"""
        
        # Processa com sistema de raciocínio
        intelligent_analysis = self.reasoning.process(raw_analysis, self.memory)
        
        # Aplica calibração adaptativa
        confidence_multiplier = self.adaptation.get_confidence_multiplier(intelligent_analysis)
        calibrated_confidence = min(0.95, intelligent_analysis['confidence'] * confidence_multiplier)
        
        # Adiciona métricas de inteligência
        intelligent_analysis.update({
            'intelligent_confidence': calibrated_confidence,
            'pattern_effectiveness': self.memory.get_pattern_effectiveness(raw_analysis),
            'system_accuracy': self.adaptation.get_overall_accuracy(),
            'learning_enabled': self.learning_enabled,
            'reasoning_depth': 'multilayer_intelligence'
        })
        
        return intelligent_analysis
    
    def learn_from_result(self, signal: Dict, actual_price_movement: float, 
                         expected_direction: str):
        """Aprende com o resultado real da operação"""
        if not self.learning_enabled:
            return
            
        # Determina se a previsão foi correta
        movement_direction = 'buy' if actual_price_movement > 0 else 'sell'
        was_correct = movement_direction == expected_direction
        
        # Aprende com o resultado
        self.memory.learn_from_signal(signal, was_correct, actual_price_movement)
        self.adaptation.calibrate_confidence(signal, was_correct)
        
        logger.info("ai_learned_from_result", 
                   symbol=signal.get('symbol', 'unknown'),
                   expected=expected_direction,
                   actual=movement_direction,
                   correct=was_correct,
                   system_accuracy=self.adaptation.get_overall_accuracy())
    
    def intelligent_self_check(self, signal: Dict) -> Dict:
        """Realiza auto-crítica inteligente"""
        checks = []
        
        # Check 1: Consistência de indicadores
        rsi = signal.get('rsi', 50)
        macd = signal.get('macd_signal', 'neutral')
        if (rsi < 40 and macd == 'bearish') or (rsi > 60 and macd == 'bullish'):
            checks.append(False)
        else:
            checks.append(True)
            
        # Check 2: Confirmação de contexto
        liquidity = signal.get('liquidity_score', 0.5)
        if liquidity < 0.3:
            checks.append(False)
        else:
            checks.append(True)
            
        # Check 3: Força do sinal
        confidence = signal.get('intelligent_confidence', 0.5)
        if confidence < 0.6:
            checks.append(False)
        else:
            checks.append(True)
            
        # Check 4: Efetividade do padrão
        pattern_effectiveness = signal.get('pattern_effectiveness', 0.5)
        if pattern_effectiveness < 0.4:
            checks.append(False)
        else:
            checks.append(True)
            
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        return {
            'self_check_passed': passed_checks >= 3,
            'checks_passed': passed_checks,
            'total_checks': total_checks,
            'check_details': [
                f"Consistência: {'PASS' if checks[0] else 'FAIL'}",
                f"Liquidez: {'PASS' if checks[1] else 'FAIL'}", 
                f"Confiança: {'PASS' if checks[2] else 'FAIL'}",
                f"Padrão: {'PASS' if checks[3] else 'FAIL'}"
            ]
        }
    
    def explain_uncertainty(self, signal: Dict) -> Dict:
        """Explica fontes de incerteza no sinal"""
        uncertainty_factors = []
        
        rsi = signal.get('rsi', 50)
        if 40 <= rsi <= 60:
            uncertainty_factors.append("RSI em zona neutra")
            
        adx = signal.get('adx', 20)
        if adx < 25:
            uncertainty_factors.append("Tendência fraca (ADX baixo)")
            
        liquidity = signal.get('liquidity_score', 0.5)
        if liquidity < 0.4:
            uncertainty_factors.append("Liquidez abaixo do ideal")
            
        confidence = signal.get('intelligent_confidence', 0.5)
        if confidence < 0.65:
            uncertainty_factors.append("Confiança moderada")
            
        return {
            'uncertainty_level': len(uncertainty_factors),
            'uncertainty_factors': uncertainty_factors,
            'suggested_action': 'wait' if uncertainty_factors else 'consider'
        }

# =========================
# RESTANTE DO CÓDIGO ORIGINAL (MANTIDO)
# =========================

# [Mantém todas as outras classes: RateLimiter, CircuitBreaker, 
#  TechnicalIndicators, SpotMarket, WSRealtimeFeed, TrajectoryIntelligence, etc...]

# [Mantém todas as funções utilitárias]

# =========================
# Enhanced Trading System - MODIFICADO
# =========================

class EnhancedTradingSystem:
    def __init__(self)->None:
        self.indicators=TechnicalIndicators()
        self.revdet=ReversalDetector()
        self.monte_carlo=MonteCarloSimulator()
        self.multi_tf=MultiTimeframeAnalyzer()
        self.liquidity=LiquiditySystem()
        self.spot=SpotMarket()
        self.current_analysis_cache: Dict[str,Any]={}
        
        # Sistemas de IA
        self.intelligent_ai = IntelligentTradingAI()
        
        # Sistemas Expandidos
        self.expanded_garch = ExpandedGARCHSystem()  # ✅ Agora só T+1
        self.trajectory_intel = TrajectoryIntelligence()
        self.signal_aggregator = IntelligentSignalAggregator()  # ✅ Filtros mais permissivos

    def analyze_symbol_expanded(self, symbol: str) -> List[Dict]:
        """Analisa apenas T+1 com 5000 simulações"""
        start_time = time.time()
        logger.info("t1_analysis_started", symbol=symbol, simulations=5000)
        
        # Coleta dados (código original mantido)
        raw = self.spot.fetch_ohlcv(symbol, "1m", max(800, 720 + 50))
        if len(raw) < 60:
            base = random.uniform(50, 400)
            raw = []
            t = int(time.time() * 1000)
            for i in range(800):
                if not raw:
                    o, h, l, c = base * 0.999, base * 1.001, base * 0.999, base
                else:
                    c_prev = raw[-1][4]
                    c = max(1e-9, c_prev * (1.0 + random.gauss(0, 0.003)))
                    o = c_prev; h = max(o, c) * (1.0 + 0.0007); l = min(o, c) * (1.0 - 0.0007)
                raw.append([t + i * 60000, o, h, l, c, 0.0])

        ohlcv_closed = raw[:-1] if (USE_CLOSED_ONLY and len(raw)>=2) else raw
        highs  = [x[2] for x in ohlcv_closed]
        lows   = [x[3] for x in ohlcv_closed]
        closes = [x[4] for x in ohlcv_closed]
        volumes = [x[5] for x in ohlcv_closed]

        price_display = raw[-1][4] if raw else 0
        volume_display = raw[-1][5] if raw and len(raw[-1]) >= 6 else 0.0
        
        try:
            ws_last = WS_FEED.get_last_candle(symbol)
            if ws_last:
                price_display  = float(ws_last[4])
                volume_display = float(ws_last[5])
        except Exception:
            pass

        # Indicadores técnicos (código original mantido)
        rsi_series = self.indicators.rsi_series_wilder(closes, 14)
        rsi = rsi_series[-1] if rsi_series else 50.0
        adx = self.indicators.adx_wilder(highs, lows, closes)
        macd = self.indicators.macd(closes)
        boll = self.indicators.calculate_bollinger_bands(closes)
        tf_cons = self.multi_tf.analyze_consensus(closes)
        liq = self.liquidity.calculate_liquidity_score(highs, lows, closes)

        # Reversão
        levels = self.revdet.compute_extremes_levels(rsi_series, 720, 6) if rsi_series else {"avg_peak":70.0,"avg_trough":30.0}
        rev_sig = self.revdet.signal_from_levels(rsi, levels, 2.5)

        # Dados técnicos consolidados
        technical_data = {
            'rsi': rsi,
            'adx': adx,
            'macd_signal': macd['signal'],
            'boll_signal': boll['signal'],
            'multi_timeframe': tf_cons,
            'liquidity_score': liq,
            'reversal': rev_sig['reversal'],
            'reversal_side': rev_sig['side'],
            'reversal_proximity': rev_sig['proximity'],
            'price': price_display,
            'volume': volume_display
        }

        # ✅ GARCH apenas para T+1 com 5000 simulações
        empirical_returns = _safe_returns_from_prices(closes)
        base_price = closes[-1] if closes else price_display

        if FEATURE_FLAGS["enable_expanded_garch"]:
            multi_horizon_garch = self.expanded_garch.run_multi_horizon_garch(
                base_price, empirical_returns
            )
            
            # IA de Trajetória Temporal
            trajectory_analysis = self.trajectory_intel.analyze_trajectory_consistency(
                multi_horizon_garch
            )
            
            # Agregação Inteligente de Sinais (apenas T+1)
            signals = self.signal_aggregator.aggregate_signals(
                symbol, multi_horizon_garch, technical_data, trajectory_analysis
            )
            
            # Aplica IA Inteligente nos sinais filtrados
            final_signals = []
            for signal in signals:
                if FEATURE_FLAGS["enable_ai_intelligence"]:
                    try:
                        intelligent_signal = self.intelligent_ai.analyze_with_intelligence({
                            **signal,
                            'volatility': stats.stdev(empirical_returns) if empirical_returns else 0.02,
                            'volume_quality': self.signal_aggregator.quality_filter.evaluate_volume_quality(volumes)
                        })
                        final_signals.append(intelligent_signal)
                    except Exception as e:
                        logger.error("ai_processing_error", symbol=symbol, error=str(e))
                        final_signals.append(signal)
                else:
                    final_signals.append(signal)
                    
            analysis_duration = (time.time() - start_time) * 1000
            logger.info("t1_analysis_completed", 
                       symbol=symbol, 
                       signals_count=len(final_signals),
                       duration_ms=analysis_duration)
            
            return final_signals
        else:
            return []

    def scan_symbols_expanded(self, symbols: List[str]) -> Dict[str, Any]:
        """Scan apenas para T+1"""
        all_signals = []
        
        for symbol in symbols:
            try:
                signals = self.analyze_symbol_expanded(symbol)
                all_signals.extend(signals)
                logger.debug("symbol_t1_analysis_completed", symbol=symbol, signals_count=len(signals))
            except Exception as e:
                logger.error("symbol_analysis_error", symbol=symbol, error=str(e))
        
        # Ordena por qualidade (trajetória + confiança)
        if all_signals:
            all_signals.sort(key=lambda x: (
                x.get('trajectory_quality', 0.5) * 0.6 +
                x.get('intelligent_confidence', x.get('confidence', 0.5)) * 0.4
            ), reverse=True)
        
        return {
            'signals': all_signals,
            'total_signals': len(all_signals),
            'symbols_analyzed': len(symbols),
            'analysis_type': 'T1_ONLY_5000_SIMS',
            'best_global': all_signals[0] if all_signals else None
        }

# =========================
# Manager / API / UI - MODIFICADO
# =========================

class AnalysisManager:
    def __init__(self)->None:
        self.is_analyzing=False
        self.current_results: List[Dict[str,Any]]=[]
        self.best_opportunity: Optional[Dict[str,Any]]=None
        self.analysis_time: Optional[str]=None
        self.symbols_default=DEFAULT_SYMBOLS
        self.system=EnhancedTradingSystem()

    def calculate_entry_time_brazil(self, horizon: int) -> str:
        dt = brazil_now() + timedelta(minutes=int(horizon))
        return br_hm_brt(dt)

    def get_brazil_time(self)->datetime:
        return brazil_now()

    def analyze_symbols_thread(self, symbols: List[str], sims: int, _unused=None)->None:
        self.is_analyzing=True
        logger.info("batch_analysis_started", symbols_count=len(symbols), simulations=sims)
        try:
            # ✅ Scan apenas para T+1
            result = self.system.scan_symbols_expanded(symbols)
            self.current_results = result['signals']
            
            if self.current_results:
                # ✅ Melhor oportunidade global T+1
                best = result.get('best_global') or max(self.current_results, key=_rank_key_directional)
                best = dict(best)
                best["entry_time"] = self.calculate_entry_time_brazil(best.get("horizon",1))
                self.best_opportunity = best
                logger.info("best_t1_opportunity_found", 
                           symbol=best['symbol'], 
                           direction=best['direction'],
                           confidence=best.get('intelligent_confidence', best.get('confidence', 0.5)),
                           probability=best['probability_buy'] if best['direction'] == 'buy' else best['probability_sell'])
            else:
                self.best_opportunity = None
                logger.info("no_t1_signals_found")
                
            self.analysis_time = br_full(self.get_brazil_time())
            logger.info("t1_batch_analysis_completed", 
                       results_count=len(self.current_results),
                       best_symbol=self.best_opportunity['symbol'] if self.best_opportunity else 'none')
        except Exception as e:
            logger.error("batch_analysis_error", error=str(e))
            self.current_results=[]
            self.best_opportunity={"error":str(e)}
            self.analysis_time = br_full(self.get_brazil_time())
        finally:
            self.is_analyzing=False

# =========================
# INICIALIZAÇÃO
# =========================

manager=AnalysisManager()
WS_FEED = WSRealtimeFeed()
WS_FEED.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("application_starting_t1_focus", 
                port=port, 
                simulations=MC_PATHS,
                symbols_count=len(DEFAULT_SYMBOLS),
                focus="T+1_ONLY")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
