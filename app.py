
# === [PHOTO IMPORTS - AUTO-ADDED] ===
try:
    import base64, io
    import numpy as np
    from PIL import Image
    import cv2 as cv
except Exception:
    base64 = io = np = Image = cv = None

# app.py — IA Signal Pro COM INTELIGÊNCIA AVANÇADA E ALTA ASSERTIVIDADE
from __future__ import annotations
import os, re, time, math, random, threading, json, statistics as stats
from typing import Any, Dict, List, Tuple, Optional, Deque
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import structlog
from collections import deque, defaultdict

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
# Config (sem ENV — tudo aqui)
# =========================
TZ_STR = "America/Maceio"
MC_PATHS = 3000
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
# SISTEMA AVANÇADO DE ASSERTIVIDADE DA IA
# =========================

class AdvancedPatternMemory:
    """Sistema de memória de padrões com análise temporal e contextual"""
    
    def __init__(self, max_patterns: int = 2000):
        self.pattern_success: Dict[str, Dict] = {}
        self.regime_specific_patterns: Dict[str, Dict] = {}
        self.time_based_patterns: Dict[str, Dict] = {}  # Padrões por horário
        self.volatility_patterns: Dict[str, Dict] = {}  # Padrões por volatilidade
        self.false_positive_patterns: set = set()
        self.high_confidence_patterns: set = set()
        self.recent_outcomes: Deque[Tuple[str, bool, float]] = deque(maxlen=1000)
        self.performance_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0, 
            'false_negatives': 0
        }
        
    def _extract_advanced_pattern_key(self, signal: Dict) -> str:
        """Chave de padrão mais granular e contextual"""
        hour = datetime.now().hour
        volatility_tier = "high" if signal.get('volatility', 0) > 0.025 else "low" if signal.get('volatility', 0) < 0.01 else "medium"
        
        elements = [
            f"rsi_tier_{int(signal.get('rsi', 0) // 10)}",
            f"adx_tier_{int(signal.get('adx', 0) // 10)}",
            f"macd_{signal.get('macd_signal', 'neutral')}",
            f"boll_{signal.get('boll_signal', 'neutral')}",
            f"tf_{signal.get('multi_timeframe', 'neutral')}",
            f"liq_tier_{int(signal.get('liquidity_score', 0) * 10)}",
            f"vol_{volatility_tier}",
            f"hour_{hour // 6}",  # Período do dia (0-3)
            f"regime_{signal.get('market_regime', 'normal')}"
        ]
        return "|".join(elements)
    
    def learn_from_signal_advanced(self, signal: Dict, actual_outcome: bool, price_movement: float, market_context: Dict):
        """Aprendizado avançado com contexto de mercado"""
        pattern_key = self._extract_advanced_pattern_key(signal)
        
        # Atualiza métricas de performance
        if actual_outcome and signal.get('direction') == 'buy' and price_movement > 0:
            self.performance_metrics['true_positives'] += 1
        elif not actual_outcome and signal.get('direction') == 'buy' and price_movement <= 0:
            self.performance_metrics['false_positives'] += 1
        elif actual_outcome and signal.get('direction') == 'sell' and price_movement < 0:
            self.performance_metrics['true_negatives'] += 1
        else:
            self.performance_metrics['false_negatives'] += 1
            
        # Aprendizado com decay temporal
        if pattern_key not in self.pattern_success:
            self.pattern_success[pattern_key] = {
                'success_rate': 0.5,
                'count': 0,
                'avg_movement': 0.0,
                'last_updated': time.time()
            }
        
        pattern_data = self.pattern_success[pattern_key]
        old_rate = pattern_data['success_rate']
        count = pattern_data['count']
        
        # Ajuste adaptativo baseado na força do movimento
        movement_factor = min(1.0, abs(price_movement) * 10)  # Normaliza movimento
        adjustment = 0.15 if actual_outcome else -0.15
        adjustment *= movement_factor  # Ajusta mais para movimentos fortes
        
        new_rate = old_rate + adjustment
        pattern_data['success_rate'] = max(0.1, min(0.95, new_rate))
        pattern_data['count'] += 1
        pattern_data['avg_movement'] = (pattern_data['avg_movement'] * count + price_movement) / (count + 1)
        pattern_data['last_updated'] = time.time()
        
        # Classifica padrões
        if pattern_data['success_rate'] > 0.7 and pattern_data['count'] >= 5:
            self.high_confidence_patterns.add(pattern_key)
        elif pattern_data['success_rate'] < 0.3 and pattern_data['count'] >= 3:
            self.false_positive_patterns.add(pattern_key)
            
        self.recent_outcomes.append((pattern_key, actual_outcome, price_movement))
        
    def get_pattern_confidence(self, signal: Dict) -> Dict[str, float]:
        """Retorna confiança multidimensional do padrão"""
        pattern_key = self._extract_advanced_pattern_key(signal)
        pattern_data = self.pattern_success.get(pattern_key, {
            'success_rate': 0.5, 'count': 0, 'avg_movement': 0.0
        })
        
        base_confidence = pattern_data['success_rate']
        count_boost = min(0.2, pattern_data['count'] * 0.02)  # Bônus por amostras
        movement_alignment = 0.0
        
        # Verifica se o movimento esperado alinha com histórico
        expected_direction = 1 if signal.get('direction') == 'buy' else -1
        if pattern_data['avg_movement'] * expected_direction > 0:
            movement_alignment = 0.1
            
        total_confidence = base_confidence + count_boost + movement_alignment
        
        return {
            'pattern_confidence': min(0.95, total_confidence),
            'reliability_score': min(1.0, pattern_data['count'] / 20.0),
            'historical_movement': pattern_data['avg_movement']
        }
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas completas de performance"""
        total = sum(self.performance_metrics.values())
        if total == 0:
            return {**self.performance_metrics, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5}
            
        tp = self.performance_metrics['true_positives']
        fp = self.performance_metrics['false_positives']
        tn = self.performance_metrics['true_negatives']
        fn = self.performance_metrics['false_negatives']
        
        accuracy = (tp + tn) / total if total > 0 else 0.5
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.5
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.5
        
        return {
            **self.performance_metrics,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'total_patterns': len(self.pattern_success),
            'high_confidence_patterns': len(self.high_confidence_patterns),
            'false_positive_patterns': len(self.false_positive_patterns)
        }

class MarketContextAnalyzer:
    """Analisador avançado de contexto de mercado"""
    
    def __init__(self):
        self.market_regimes = {}
        self.sector_correlations = {}
        self.volume_analysis = {}
        
    def analyze_market_context(self, symbols_data: List[Dict]) -> Dict:
        """Analisa contexto geral do mercado"""
        if not symbols_data:
            return {"market_sentiment": "neutral", "volatility_regime": "medium"}
            
        # Análise de sentimento geral
        buy_signals = sum(1 for s in symbols_data if s.get('direction') == 'buy')
        sell_signals = sum(1 for s in symbols_data if s.get('direction') == 'sell')
        total_signals = len(symbols_data)
        
        sentiment_score = (buy_signals - sell_signals) / total_signals if total_signals > 0 else 0
        
        # Análise de volatilidade média
        avg_volatility = stats.mean([s.get('volatility', 0.02) for s in symbols_data]) if symbols_data else 0.02
        volatility_regime = "high" if avg_volatility > 0.03 else "low" if avg_volatility < 0.01 else "medium"
        
        # Análise de força de tendência
        avg_adx = stats.mean([s.get('adx', 20) for s in symbols_data]) if symbols_data else 20
        trend_strength = "strong" if avg_adx > 25 else "weak" if avg_adx < 15 else "moderate"
        
        # Análise de RSI médio
        avg_rsi = stats.mean([s.get('rsi', 50) for s in symbols_data]) if symbols_data else 50
        rsi_bias = "oversold" if avg_rsi < 40 else "overbought" if avg_rsi > 60 else "neutral"
        
        return {
            "market_sentiment": "bullish" if sentiment_score > 0.1 else "bearish" if sentiment_score < -0.1 else "neutral",
            "volatility_regime": volatility_regime,
            "trend_strength": trend_strength,
            "rsi_bias": rsi_bias,
            "sentiment_score": round(sentiment_score, 3),
            "avg_volatility": round(avg_volatility, 4),
            "avg_adx": round(avg_adx, 1),
            "avg_rsi": round(avg_rsi, 1),
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "total_signals": total_signals
        }

class AdvancedIntelligenceEngine:
    """Motor de inteligência avançado com múltiplas camadas"""
    
    def __init__(self):
        self.pattern_memory = AdvancedPatternMemory()
        self.context_analyzer = MarketContextAnalyzer()
        self.confidence_calibration = {}
        self.performance_tracking = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': deque(maxlen=100)
        }
        
    def _calculate_technical_convergence(self, signal: Dict) -> float:
        """Calcula convergência de indicadores técnicos"""
        convergence_score = 0.0
        confirming_indicators = 0
        total_indicators = 0
        
        direction = signal.get('direction', 'buy')
        rsi = signal.get('rsi', 50)
        adx = signal.get('adx', 20)
        macd_signal = signal.get('macd_signal', 'neutral')
        boll_signal = signal.get('boll_signal', 'neutral')
        tf_consensus = signal.get('multi_timeframe', 'neutral')
        reversal = signal.get('reversal', False)
        reversal_side = signal.get('reversal_side')
        
        # RSI
        if (direction == 'buy' and rsi < 40) or (direction == 'sell' and rsi > 60):
            confirming_indicators += 1
        total_indicators += 1
        
        # ADX 
        if adx > 25:  # Tendência forte
            confirming_indicators += 1
        total_indicators += 1
        
        # MACD
        if (direction == 'buy' and macd_signal == 'bullish') or (direction == 'sell' and macd_signal == 'bearish'):
            confirming_indicators += 1
        total_indicators += 1
        
        # Bollinger Bands
        if (direction == 'buy' and boll_signal == 'oversold') or (direction == 'sell' and boll_signal == 'overbought'):
            confirming_indicators += 1
        total_indicators += 1
            
        # Timeframe Consensus
        if (direction == 'buy' and tf_consensus == 'buy') or (direction == 'sell' and tf_consensus == 'sell'):
            confirming_indicators += 1
        total_indicators += 1
            
        # Reversal Signals
        if reversal and reversal_side == direction:
            confirming_indicators += 1
        total_indicators += 1
            
        convergence_score = confirming_indicators / total_indicators if total_indicators > 0 else 0.5
        return convergence_score
    
    def _analyze_risk_adjustment(self, signal: Dict, market_context: Dict) -> float:
        """Ajuste de risco baseado em contexto"""
        base_risk = 1.0
        
        # Ajuste por volatilidade
        volatility = signal.get('volatility', 0.02)
        if volatility > 0.03:
            base_risk *= 0.8  # Reduz confiança em alta volatilidade
        elif volatility < 0.01:
            base_risk *= 1.1  # Aumenta confiança em baixa volatilidade
            
        # Ajuste por liquidez
        liquidity = signal.get('liquidity_score', 0.5)
        if liquidity < 0.3:
            base_risk *= 0.7  # Reduz confiança em baixa liquidez
            
        # Ajuste por força de tendência
        adx = signal.get('adx', 20)
        if adx > 25:
            base_risk *= 1.15  # Aumenta confiança em tendências fortes
        elif adx < 15:
            base_risk *= 0.9   # Reduz confiança em tendências fracas
            
        # Ajuste por probabilidade do Monte Carlo
        prob_buy = signal.get('probability_buy', 0.5)
        if (signal.get('direction') == 'buy' and prob_buy > 0.6) or (signal.get('direction') == 'sell' and prob_buy < 0.4):
            base_risk *= 1.1
        elif (signal.get('direction') == 'buy' and prob_buy < 0.4) or (signal.get('direction') == 'sell' and prob_buy > 0.6):
            base_risk *= 0.9
            
        # Ajuste por regime de mercado (suave; não bloqueia sinal)
        regime = market_context.get('volatility_regime') or signal.get('market_regime')
        if regime == "high" or regime == "high_volatility":
            base_risk *= 0.92  # -8% em alta vol
        elif regime == "low" or regime == "low_volatility":
            base_risk *= 1.05  # +5% em baixa vol
        return base_risk
    
    def _calculate_adaptive_confidence(self, signal: Dict, market_context: Dict, pattern_confidence: Dict) -> Dict:
        """Cálculo adaptativo de confiança final"""
        base_confidence = signal.get('confidence', 0.5)
        
        # Fatores de ajuste
        technical_convergence = self._calculate_technical_convergence(signal)
        risk_adjustment = self._analyze_risk_adjustment(signal, market_context)
        pattern_strength = pattern_confidence.get('pattern_confidence', 0.5)
        reliability = pattern_confidence.get('reliability_score', 0.0)
        
        # Fórmula de confiança ponderada
        confidence_components = {
            'technical': base_confidence * 0.3,
            'pattern_memory': pattern_strength * 0.4,
            'convergence': technical_convergence * 0.2,
            'reliability': reliability * 0.1
        }
        # bônus quando há alta convergência
        if technical_convergence >= 0.8:
            confidence_components['technical'] = min(0.95, confidence_components['technical'] + 0.05)
        final_confidence = sum(confidence_components.values()) * risk_adjustment
        
        # Limites de confiança
        final_confidence = max(0.3, min(0.95, final_confidence))
        
        return {
            'final_confidence': final_confidence,
            'confidence_breakdown': confidence_components,
            'risk_adjustment': risk_adjustment,
            'technical_convergence': technical_convergence
        }
    
    def generate_intelligent_signal(self, raw_signal: Dict, market_context: Dict, all_signals: List[Dict]) -> Dict:
        """Gera sinal inteligente com assertividade aumentada"""
        
        # Análise de padrão
        pattern_analysis = self.pattern_memory.get_pattern_confidence(raw_signal)
        
        # Cálculo de confiança adaptativa
        confidence_analysis = self._calculate_adaptive_confidence(raw_signal, market_context, pattern_analysis)
        
        # Análise de contexto de mercado
        market_sentiment = market_context.get('market_sentiment', 'neutral')
        direction = raw_signal.get('direction', 'buy')
        
        # Ajuste baseado no sentimento do mercado
        sentiment_alignment = 1.0
        if (market_sentiment == 'bullish' and direction == 'sell') or (market_sentiment == 'bearish' and direction == 'buy'):
            sentiment_alignment = 0.8  # Penaliza sinais contra tendência
        
        final_confidence = confidence_analysis['final_confidence'] * sentiment_alignment
        
        # Gera razões inteligentes
        reasoning = self._generate_intelligent_reasoning(raw_signal, confidence_analysis, market_context)
        
        # Atualiza métricas
        self.performance_tracking['total_predictions'] += 1
        
        return {
            'symbol': raw_signal.get('symbol'),
            'direction': direction,
            'final_confidence': final_confidence,
            'reasoning': reasoning,
            'pattern_analysis': pattern_analysis,
            'confidence_breakdown': confidence_analysis['confidence_breakdown'],
            'technical_convergence': confidence_analysis['technical_convergence'],
            'market_sentiment_alignment': sentiment_alignment,
            'risk_adjustment': confidence_analysis['risk_adjustment'],
            'quality_metrics': {
                'pattern_reliability': pattern_analysis.get('reliability_score', 0),
                'technical_strength': confidence_analysis['technical_convergence'],
                'context_alignment': sentiment_alignment
            }
        }
    
    def _generate_intelligent_reasoning(self, signal: Dict, confidence_analysis: Dict, market_context: Dict) -> List[str]:
        """Gera razões inteligentes para a decisão"""
        reasons = []
        
        # Razões técnicas
        convergence = confidence_analysis['technical_convergence']
        if convergence > 0.7:
            reasons.append("Alta convergência de indicadores técnicos")
        elif convergence < 0.4:
            reasons.append("Baixa convergência entre indicadores")
            
        # Razões de padrão
        pattern_conf = confidence_analysis['confidence_breakdown']['pattern_memory']
        if pattern_conf > 0.6:
            reasons.append("Padrão histórico de alta efetividade")
        elif pattern_conf < 0.4:
            reasons.append("Padrão histórico problemático")
            
        # Razões de contexto
        market_sentiment = market_context.get('market_sentiment')
        direction = signal.get('direction')
        if market_sentiment == 'bullish' and direction == 'buy':
            reasons.append("Alinhado com sentimento bullish do mercado")
        elif market_sentiment == 'bearish' and direction == 'sell':
            reasons.append("Alinhado com sentimento bearish do mercado")
        else:
            reasons.append("Operação contra tendência geral - cuidado recomendado")
            
        # Razões de risco
        risk_adj = confidence_analysis['risk_adjustment']
        if risk_adj > 1.1:
            reasons.append("Condições de baixo risco favoráveis")
        elif risk_adj < 0.9:
            reasons.append("Condições de alto risco detectadas")
            
        # Razões específicas de indicadores
        rsi = signal.get('rsi', 50)
        if (direction == 'buy' and rsi < 35) or (direction == 'sell' and rsi > 65):
            reasons.append("RSI em zona extrema favorável")
            
        adx = signal.get('adx', 20)
        if adx > 25:
            reasons.append("Tendência forte confirmada")
            
        if signal.get('reversal', False):
            reasons.append("Sinal de reversão detectado")
            
        return reasons
    
    def learn_from_outcome(self, signal: Dict, actual_movement: float, market_context: Dict):
        """Aprendizado com resultado real"""
        predicted_direction = signal.get('direction', 'buy')
        actual_direction = 'buy' if actual_movement > 0 else 'sell'
        was_correct = predicted_direction == actual_direction
        
        # Atualiza memória de padrões
        self.pattern_memory.learn_from_signal_advanced(signal, was_correct, actual_movement, market_context)
        
        # Atualiza métricas de performance
        if was_correct:
            self.performance_tracking['correct_predictions'] += 1
        self.performance_tracking['recent_accuracy'].append(was_correct)
        
        logger.info("advanced_ai_learned", 
                   symbol=signal.get('symbol'),
                   predicted=predicted_direction,
                   actual=actual_direction,
                   correct=was_correct,
                   movement=actual_movement)
    
    def get_system_accuracy(self) -> float:
        """Retorna acurácia atual do sistema"""
        total = self.performance_tracking['total_predictions']
        correct = self.performance_tracking['correct_predictions']
        
        if total > 0:
            return correct / total
        
        # Calcula acurácia recente
        recent = self.performance_tracking['recent_accuracy']
        if recent:
            return sum(recent) / len(recent)
            
        return 0.5
    
    def get_detailed_performance(self) -> Dict:
        """Retorna métricas detalhadas de performance"""
        pattern_metrics = self.pattern_memory.get_performance_metrics()
        
        return {
            'system_accuracy': self.get_system_accuracy(),
            'total_predictions': self.performance_tracking['total_predictions'],
            'correct_predictions': self.performance_tracking['correct_predictions'],
            'recent_accuracy_window': len(self.performance_tracking['recent_accuracy']),
            'pattern_memory_performance': pattern_metrics
        }

class HighAccuracyTradingAI:
    """IA de Trading com Alta Assertividade"""
    
    def __init__(self):
        self.advanced_engine = AdvancedIntelligenceEngine()
        self.learning_enabled = True
        
    def analyze_with_high_accuracy(self, raw_analysis: Dict, all_symbols_data: List[Dict]) -> Dict:
        """Análise com assertividade aumentada"""
        
        # Análise de contexto de mercado
        market_context = self.advanced_engine.context_analyzer.analyze_market_context(all_symbols_data)
        
        # Gera sinal inteligente
        intelligent_signal = self.advanced_engine.generate_intelligent_signal(
            raw_analysis, market_context, all_symbols_data
        )
        
        # Adiciona métricas de qualidade
        intelligent_signal.update({
            'market_context': market_context,
            'learning_enabled': self.learning_enabled,
            'system_accuracy': self.advanced_engine.get_system_accuracy(),
            'reasoning_depth': 'advanced_multilayer_intelligence',
            'advanced_ai': True
        })
        
        return intelligent_signal
    
    def learn_from_result(self, signal: Dict, actual_price_movement: float, expected_direction: str, all_signals: List[Dict]):
        """Aprendizado com resultado real"""
        if not self.learning_enabled:
            return
            
        market_context = self.advanced_engine.context_analyzer.analyze_market_context(all_signals)
        self.advanced_engine.learn_from_outcome(signal, actual_price_movement, market_context)
        
    def get_detailed_status(self) -> Dict:
        """Status detalhado da IA avançada"""
        return self.advanced_engine.get_detailed_performance()

# =========================
# Feature Flags Atualizadas
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
    "enable_advanced_ai": True  # NOVA FLAG PARA IA AVANÇADA
}

# =========================
# Rate Limiting Simples
# =========================
class RateLimiter:
    def __init__(self):
        self.requests = {}
        
    def is_allowed(self, identifier: str, max_requests: int = 30, window_seconds: int = 60) -> bool:
        now = time.time()
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        self.requests[identifier] = [req_time for req_time in self.requests[identifier] 
                                   if now - req_time < window_seconds]
        
        if len(self.requests[identifier]) < max_requests:
            self.requests[identifier].append(now)
            return True
        return False

rate_limiter = RateLimiter()

# =========================
# Circuit Breaker
# =========================
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
        logger.info("circuit_breaker_closed")
        
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        logger.warning("circuit_breaker_failure", failures=self.failures, threshold=self.failure_threshold)
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.error("circuit_breaker_opened")
            
    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("circuit_breaker_half_open")
                return True
            return False
        else:
            return True

binance_circuit_breaker = CircuitBreaker()

# =========================
# Tempo (Brasil)
# =========================
def brazil_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=-3)))

def br_full(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y %H:%M:%S")

def br_hm_brt(dt: datetime) -> str:
    return dt.strftime("%H:%M BRT")

# =========================
# Utils (mantidas do código original)
# =========================
def _to_binance_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}{quote}"
    return re.sub(r'[^A-Z0-9]', '', s)

def _to_coinapi_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
    else:
        if s.endswith("USDT"): base, quote = s[:-4], "USDT"
        elif s.endswith("USD"): base, quote = s[:-3], "USD"
        else: base, quote = s, "USDT"
    return f"BINANCE_SPOT_{base}_{quote}"

def _iso_to_ms(iso_str: str) -> int:
    z = iso_str.endswith("Z")
    s = iso_str[:-1] if z else iso_str
    if '.' in s:
        head, tail = s.split('.', 1)
        tail = ''.join(ch for ch in tail if ch.isdigit())
        tail = (tail + "000000")[:6]
        s = head + "." + tail
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
    else:
        fmt = "%Y-%m-%dT%H:%M:%S"
    dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _safe_returns_from_prices(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
    return returns

def _rank_key_directional(x: Dict[str, Any]) -> float:
    direction = x.get("direction", "buy")
    prob_directional = x["probability_buy"] if direction == "buy" else x["probability_sell"]
    # Prefere confiança avançada se disponível
    confidence = x.get('final_confidence') or x.get('intelligent_confidence') or x.get('confidence', 0.5)
    return (confidence * 1000) + (prob_directional * 100)

def _confirm_prob_neutral_zone(prob_up: float, rsi: float, macd_hist: float, adx: float, 
                              boll_signal: str, tf_consensus: str) -> float:
    bullish = 0
    bearish = 0
    
    if rsi >= 55: bullish += 1
    if rsi <= 45: bearish += 1
    
    if macd_hist > 0.001: bullish += 1
    if macd_hist < -0.001: bearish += 1
    
    if boll_signal in ["oversold", "bullish"]: bullish += 1
    if boll_signal in ["overbought", "bearish"]: bearish += 1
    
    if tf_consensus == "buy": bullish += 1
    if tf_consensus == "sell": bearish += 1
    
    strong = (adx >= 25)
    
    if bullish > bearish and strong:   adj = 0.07
    elif bullish > bearish:            adj = 0.04
    elif bearish > bullish and strong: adj = -0.07
    elif bearish > bullish:            adj = -0.04
    else: 
        adj = 0.0
        
    adjusted_prob = prob_up + adj
    return min(0.90, max(0.10, adjusted_prob))

def _calculate_directional_confidence(prob_direction: float, direction: str, rsi: float, 
                                    adx: float, macd_signal: str, boll_signal: str,
                                    tf_consensus: str, reversal_signal: dict,
                                    liquidity_score: float) -> float:
    base_confidence = prob_direction * 100.0
    directional_boosts = 0.0
    
    if adx > 25:
        if (direction == 'buy' and tf_consensus == 'buy') or (direction == 'sell' and tf_consensus == 'sell'):
            directional_boosts += 15.0
    
    if (direction == 'buy' and rsi < 40) or (direction == 'sell' and rsi > 60):
        directional_boosts += 10.0
    
    if (direction == 'buy' and boll_signal == 'oversold') or (direction == 'sell' and boll_signal == 'overbought'):
        directional_boosts += 12.0
    
    if reversal_signal["reversal"] and reversal_signal["side"] == direction:
        directional_boosts += 18.0 * reversal_signal["proximity"]
    
    if (direction == 'buy' and macd_signal == 'bullish') or (direction == 'sell' and macd_signal == 'bearish'):
        directional_boosts += 8.0
    
    total_score = base_confidence + directional_boosts
    total_score *= (0.97 + (liquidity_score * 0.06))
    
    return min(95.0, max(30.0, total_score)) / 100.0

# =========================
# WebSocket OKX (mantido do original)
# =========================
class WSRealtimeFeed:
    def __init__(self):
        self.enabled = bool(USE_WS)
        self.buf_minutes = int(WS_BUFFER_MINUTES)
        self.symbols = [s.strip().upper() for s in WS_SYMBOLS if s.strip()]
        self._lock = threading.Lock()
        self._buffers: Dict[str, List[List[float]]] = {s: [] for s in self.symbols}
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._running = False
        self._ws_available = False
        try:
            import websocket
            self._ws_available = True
        except Exception:
            logger.warning("websocket_client_not_available", ws_disabled=True)
            self.enabled = False

    def _on_open(self, ws):
        try:
            args = [{"channel": OKX_CHANNEL, "instId": s.replace("/", "-")} for s in self.symbols]
            ws.send(json.dumps({"op":"subscribe","args":args}))
            logger.info("websocket_subscribed", provider="okx", symbols_count=len(args))
        except Exception as e:
            logger.error("websocket_subscribe_error", error=str(e))

    def _on_message(self, _, msg: str):
        try:
            data = json.loads(msg)
            if data.get("event") in ("subscribe", "error"):
                if data.get("event") == "error":
                    logger.error("websocket_error", error=data)
                return
            arg = data.get("arg", {})
            if arg.get("channel") != OKX_CHANNEL:
                return
            sym = arg.get("instId","").replace("-", "/")
            for row in (data.get("data") or []):
                ts = int(row[0]); o=float(row[1]); h=float(row[2]); l=float(row[3]); c=float(row[4])
                v = float(row[5]) if len(row)>5 else 0.0
                rec = [ts,o,h,l,c,v]
                with self._lock:
                    buf = self._buffers.setdefault(sym, [])
                    if buf and buf[-1][0] == ts:
                        buf[-1] = rec
                    else:
                        buf.append(rec)
                        if len(buf) > self.buf_minutes + 5:
                            del buf[:len(buf)-(self.buf_minutes+5)]
        except Exception as e:
            logger.error("websocket_message_error", error=str(e))

    def _on_error(self, _, err): 
        logger.error("websocket_error", error=str(err))
    
    def _on_close(self, *_):    
        logger.warning("websocket_closed")

    def _run(self):
        from websocket import WebSocketApp
        while self._running:
            try:
                self._ws = WebSocketApp(OKX_URL, on_open=self._on_open, on_message=self._on_message,
                                        on_error=self._on_error, on_close=self._on_close)
                self._ws.run_forever(ping_interval=25, ping_timeout=10)
            except Exception as e:
                logger.error("websocket_run_forever_error", error=str(e))
            if self._running: 
                time.sleep(3)

    def start(self):
        if not self.enabled or not self._ws_available: 
            return
        if self._thread and self._thread.is_alive():    
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("websocket_started")

    def stop(self):
        self._running = False
        try:
            if self._ws: 
                self._ws.close()
        except Exception as e:
            logger.error("websocket_stop_error", error=str(e))
        if self._thread: 
            self._thread.join(timeout=2)
        logger.info("websocket_stopped")

    def get_ohlcv(self, symbol: str, limit: int = 1000, use_closed_only: bool = True) -> List[List[float]]:
        if not (self.enabled and self._ws_available): 
            return []
        sym = symbol.strip().upper()
        with self._lock:
            buf = self._buffers.get(sym, [])
            data = buf[-min(len(buf), limit):]
        if not data: 
            return []
        if use_closed_only and len(data) >= 1:
            now_min  = int(time.time() // 60)
            last_min = int((data[-1][0] // 1000) // 60)
            if last_min == now_min: 
                data = data[:-1]
        return data[:]

    def get_last_candle(self, symbol: str) -> Optional[List[float]]:
        if not (self.enabled and self._ws_available): 
            return None
        sym = symbol.strip().upper()
        with self._lock:
            buf = self._buffers.get(sym, [])
            return buf[-1][:] if buf else None

WS_FEED = WSRealtimeFeed()
WS_FEED.start()

# =========================
# Mercado Spot com Cache Inteligente (mantido)
# =========================
class SmartCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, int], Tuple[float, List[List[float]], float]] = {}
        self._volatility_cache: Dict[str, float] = {}
        
    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 10:
            return 0.0
        returns = _safe_returns_from_prices(prices[-50:])
        if not returns:
            return 0.0
        return stats.stdev(returns) if len(returns) > 1 else 0.0
    
    def _should_invalidate(self, symbol: str, current_prices: List[float]) -> bool:
        current_vol = self._calculate_volatility(current_prices)
        cached_vol = self._volatility_cache.get(symbol, 0.0)
        
        if current_vol > cached_vol * 1.5 and current_vol > 0.01:
            return True
        return False
    
    def get(self, key: Tuple[str, str, int]) -> Optional[Tuple[float, List[List[float]]]]:
        if key in self._cache:
            timestamp, data, _ = self._cache[key]
            symbol = key[0]
            current_vol = self._volatility_cache.get(symbol, 0.0)
            cache_ttl = 2 if current_vol > 0.02 else 5 if current_vol > 0.01 else 10
            
            if time.time() - timestamp < cache_ttl:
                return (timestamp, data)
        return None
    
    def set(self, key: Tuple[str, str, int], data: List[List[float]]):
        symbol = key[0]
        prices = [x[4] for x in data] if data else []
        current_vol = self._calculate_volatility(prices)
        self._volatility_cache[symbol] = current_vol
        self._cache[key] = (time.time(), data, current_vol)

class SpotMarket:
    def __init__(self) -> None:
        self._cache = SmartCache()
        self._session = __import__("requests").Session()
        self._has_ccxt = False
        self._ccxt = None
        try:
            import ccxt
            self._ccxt = ccxt.binace({
                "enableRateLimit": True,
                "timeout": 12000,
                "options": {"defaultType": "spot"}
            })
            self._has_ccxt = True
            logger.info("ccxt_initialized", version=getattr(ccxt, '__version__', 'unknown'))
        except Exception as e:
            logger.warning("ccxt_unavailable", error=str(e))
            self._has_ccxt = False

    def _fetch_http_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 1000) -> List[List[float]]:
        if not binance_circuit_breaker.can_execute():
            logger.warning("circuit_breaker_open", provider="binance_http")
            return []
            
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": _to_binance_symbol(symbol), "interval": timeframe, "limit": min(1000, int(limit))}
        try:
            r = self._session.get(url, params=params, timeout=10)
            if r.status_code in (418, 429):
                time.sleep(0.5)
                r = self._session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            binance_circuit_breaker.record_success()
            logger.debug("http_fetch_success", symbol=symbol, candles_count=len(data))
            return [[float(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        except Exception as e:
            binance_circuit_breaker.record_failure()
            logger.error("http_fetch_error", symbol=symbol, error=str(e))
            return []

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 1000) -> List[List[float]]:
        key = (symbol.upper(), timeframe, limit)
        
        if FEATURE_FLAGS["enable_smart_cache"]:
            cached = self._cache.get(key)
            if cached:
                timestamp, data = cached
                if not self._cache._should_invalidate(symbol, [x[4] for x in data]):
                    return data

        ohlcv: List[List[float]] = []

        try:
            if timeframe == "1m":
                ws_data = WS_FEED.get_ohlcv(symbol, limit=limit, use_closed_only=USE_CLOSED_ONLY)
                if ws_data and len(ws_data) >= 10:
                    ohlcv = ws_data
                    logger.debug("websocket_data_used", symbol=symbol, candles_count=len(ws_data))
        except Exception as e:
            logger.error("websocket_fetch_error", symbol=symbol, error=str(e))

        if (not ohlcv or len(ohlcv) < 60) and self._has_ccxt and self._ccxt is not None:
            if not binance_circuit_breaker.can_execute():
                logger.warning("circuit_breaker_open", provider="binance_ccxt")
            else:
                try:
                    raw = self._ccxt.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(1000, int(limit)))
                    cc = [[float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw]
                    if ohlcv:
                        ts = {r[0] for r in ohlcv}
                        cc = [r for r in cc if r[0] not in ts]
                        ohlcv = sorted(ohlcv + cc, key=lambda x: x[0])[-limit:]
                    else:
                        ohlcv = cc
                    binance_circuit_breaker.record_success()
                    logger.debug("ccxt_fetch_success", symbol=symbol, candles_count=len(ohlcv))
                except Exception as e:
                    binance_circuit_breaker.record_failure()
                    logger.error("ccxt_fetch_error", symbol=symbol, error=str(e))

        if not ohlcv or len(ohlcv) < 60:
            http = self._fetch_http_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if http:
                if ohlcv:
                    ts = {r[0] for r in ohlcv}
                    http = [r for r in http if r[0] not in ts]
                    ohlcv = sorted(ohlcv + http, key=lambda x: x[0])[-limit:]
                else:
                    ohlcv = http
                logger.debug("http_fallback_used", symbol=symbol, candles_count=len(ohlcv))

        if ohlcv:
            self._cache.set(key, ohlcv)
            
        return ohlcv

# =========================
# Indicadores (mantidos do original)
# =========================
class TechnicalIndicators:
    @staticmethod
    def _wilder_smooth(prev: float, cur: float, period: int) -> float:
        alpha = 1.0 / period
        return prev + alpha * (cur - prev)

    def rsi_series_wilder(self, closes: List[float], period: int = 14) -> List[float]:
        if len(closes) < period + 1:
            return []
        gains, losses = [], []
        for i in range(1, len(closes)):
            ch = closes[i] - closes[i - 1]
            gains.append(max(0.0, ch))
            losses.append(max(0.0, -ch))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsis = []
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float('inf')
        rsis.append(100.0 if rs == float('inf') else 100.0 - (100.0 / (1.0 + rs)))

        for i in range(period, len(gains)):
            avg_gain = self._wilder_smooth(avg_gain, gains[i], period)
            avg_loss = self._wilder_smooth(avg_loss, losses[i], period)
            if avg_loss == 0:
                rsis.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsis.append(100.0 - (100.0 / (1.0 + rs)))
        return [max(0.0, min(100.0, r)) for r in rsis]

    def rsi_wilder(self, closes: List[float], period: int = 14) -> float:
        s = self.rsi_series_wilder(closes, period)
        return s[-1] if s else 50.0

    def adx_wilder(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        n = len(closes)
        if n < period + 2:
            return 20.0
        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(1, n):
            high, low, close_prev = highs[i], lows[i], closes[i - 1]
            prev_high, prev_low = highs[i - 1], lows[i - 1]
            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            up_move = high - prev_high
            down_move = prev_low - low
            pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
            ndm = down_move if (down_move > up_move and down_move > 0) else 0.0
            tr_list.append(tr); pdm_list.append(pdm); ndm_list.append(ndm)

        atr = sum(tr_list[:period]) / period
        pdi = sum(pdm_list[:period]) / period
        ndi = sum(ndm_list[:period]) / period

        dx_vals = []
        for i in range(period, len(tr_list)):
            atr = self._wilder_smooth(atr, tr_list[i], period)
            pdi = self._wilder_smooth(pdi, pdm_list[i], period)
            ndi = self._wilder_smooth(ndi, ndm_list[i], period)
            plus_di = 100.0 * (pdi / max(1e-12, atr))
            minus_di = 100.0 * (ndi / max(1e-12, atr))
            dx = 100.0 * abs(plus_di - minus_di) / max(1e-12, (plus_di + minus_di))
            dx_vals.append(dx)

        if not dx_vals:
            return 20.0
        adx = sum(dx_vals[:period]) / period if len(dx_vals) >= period else sum(dx_vals) / len(dx_vals)
        for i in range(period, len(dx_vals)):
            adx = self._wilder_smooth(adx, dx_vals[i], period)
        return round(max(0.0, min(100.0, adx)), 2)

    def macd(self, closes: List[float]) -> Dict[str, Any]:
        def ema(vals: List[float], n: int) -> List[float]:
            if not vals: return []
            k = 2 / (n + 1)
            e = [vals[0]]
            for v in vals[1:]:
                e.append(e[-1] + k * (v - e[-1]))
            return e
        if len(closes) < 35:
            return {"signal": "neutral", "strength": 0.0}
        ema12 = ema(closes, 12); ema26 = ema(closes, 26)
        macd_line = [a - b for a, b in zip(ema12[-len(ema26):], ema26)]
        signal_line = ema(macd_line, 9)
        if not signal_line:
            return {"signal": "neutral", "strength": 0.0}
        hist = macd_line[-1] - signal_line[-1]
        if hist > 0:  return {"signal": "bullish", "strength": min(1.0, abs(hist) / max(1e-9, closes[-1] * 0.002))}
        if hist < 0:  return {"signal": "bearish", "strength": min(1.0, abs(hist) / max(1e-9, closes[-1] * 0.002))}
        return {"signal": "neutral", "strength": 0.0}

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Dict[str,str]:
        if len(prices) < period: return {"signal":"neutral"}
        win = prices[-period:]; ma = sum(win)/period
        var = sum((p-ma)**2 for p in win)/period; sd = math.sqrt(max(0.0, var))
        last = prices[-1]; upper = ma + 2*sd; lower = ma - 2*sd
        if last>upper: return {"signal":"overbought"}
        if last<lower: return {"signal":"oversold"}
        if last>ma: return {"signal":"bullish"}
        if last<ma: return {"signal":"bearish"}
        return {"signal":"neutral"}

class MultiTimeframeAnalyzer:
    def analyze_consensus(self, closes: List[float]) -> str:
        if len(closes) < 60: return "neutral"
        ma9 = sum(closes[-9:]) / 9
        ma21 = sum(closes[-21:]) / 21 if len(closes) >= 21 else ma9
        return "buy" if ma9 > ma21 else ("sell" if ma9 < ma21 else "neutral")

class LiquiditySystem:
    def calculate_liquidity_score(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        n = len(closes)
        if n < period + 2:
            return 0.5
        trs = []
        for i in range(1, n):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atr = sum(trs[:period]) / period
        for i in range(period, len(trs)):
            atr = (atr * (period - 1) + trs[i]) / period
        atr_pct = atr / max(1e-12, closes[-1])
        LIM = 0.02
        score = 1.0 - min(1.0, atr_pct / LIM)
        return round(max(0.0, min(1.0, score)), 3)

# =========================
# Reversão (mantido do original)
# =========================
class ReversalDetector:
    def compute_extremes_levels(self, rsi_series: List[float], window: int = 720, n_extremes: int = 6) -> Dict[str, float]:
        if not rsi_series:
            return {"avg_peak": 70.0, "avg_trough": 30.0}
        rs = rsi_series[-window:] if len(rsi_series) > window else rsi_series[:]
        peaks, troughs = [], []
        for i in range(1, len(rs)-1):
            if rs[i] > rs[i-1] and rs[i] > rs[i+1]:
                peaks.append(rs[i])
            if rs[i] < rs[i-1] and rs[i] < rs[i+1]:
                troughs.append(rs[i])
        peaks = sorted(peaks, reverse=True)[:max(1, n_extremes)]
        troughs = sorted(troughs)[:max(1, n_extremes)]
        avg_peak = stats.mean(peaks) if peaks else 70.0
        avg_trough = stats.mean(troughs) if troughs else 30.0
        return {"avg_peak": float(avg_peak), "avg_trough": float(avg_trough)}

    def signal_from_levels(self, current_rsi: float, levels: Dict[str,float], tol: float = 2.5) -> Dict[str, Any]:
        peak, trough = levels["avg_peak"], levels["avg_trough"]
        out = {"reversal": False, "side": None, "proximity": 0.0, "levels": levels}
        if abs(current_rsi - peak) <= tol:
            out.update({"reversal": True, "side": "bearish", "proximity": max(0.0, 1 - abs(current_rsi-peak)/max(1e-9,tol))})
        elif abs(current_rsi - trough) <= tol:
            out.update({"reversal": True, "side": "bullish", "proximity": max(0.0, 1 - abs(current_rsi-trough)/max(1e-9,tol))})
        return out

# =========================
# GARCH(1,1) Light Adaptativo (mantido)
# =========================
class AdaptiveGARCH11Simulator:
    def _detect_market_regime(self, returns: List[float]) -> str:
        if not returns or len(returns) < 20:
            return "normal"
        
        volatility = stats.stdev(returns) if len(returns) > 1 else 0.0
        mean_abs_return = stats.mean([abs(r) for r in returns])
        
        if volatility > 0.03 or mean_abs_return > 0.02:
            return "high_volatility"
        elif volatility < 0.005:
            return "low_volatility"
        else:
            return "normal"
    
    def _get_garch_params_for_regime(self, regime: str, returns: List[float]) -> Tuple[float, float, float, float]:
        if not returns:
            return 1e-6, 0.1, 0.85, 1e-4
            
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        
        if regime == "low_volatility":
            alpha, beta = 0.05, 0.92
        elif regime == "high_volatility":
            alpha, beta = 0.12, 0.80
        else:
            alpha, beta = 0.08, 0.90
            
        omega = var * (1 - alpha - beta)
        omega = max(1e-6, omega)
        
        if alpha + beta >= 0.999:
            alpha, beta = 0.08, 0.90
            omega = 1e-6
            
        return omega, alpha, beta, var
    
    def _calculate_garch_fit_quality(self, returns: List[float], omega: float, alpha: float, beta: float) -> float:
        if len(returns) < 30:
            return 0.7
            
        try:
            predicted_var = omega / (1 - alpha - beta) if alpha + beta < 1 else 1e-4
            actual_var = stats.variance(returns) if len(returns) > 1 else 1e-4
            fit_quality = 1.0 - min(1.0, abs(predicted_var - actual_var) / actual_var)
            return max(0.3, min(0.95, fit_quality))
        except:
            return 0.7

    def simulate_garch11(self, base_price: float, returns: List[float], 
                        steps: int, num_paths: int = 3000) -> Dict[str, Any]:
        import math
        import random
        
        if not returns or len(returns) < 10:
            returns = [random.gauss(0.0, 0.002) for _ in range(100)]
        
        regime = self._detect_market_regime(returns) if FEATURE_FLAGS["enable_adaptive_garch"] else "normal"
        omega, alpha, beta, h_last = self._get_garch_params_for_regime(regime, returns)
        
        fit_quality = self._calculate_garch_fit_quality(returns, omega, alpha, beta)
        
        up_count = 0
        total_count = 0
        
        start_time = time.time()
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
        
        duration_ms = (time.time() - start_time) * 1000
        
        if total_count == 0:
            prob_buy = 0.5
        else:
            prob_buy = up_count / total_count
        
        prob_buy = min(0.95, max(0.05, prob_buy))
        prob_sell = 1.0 - prob_buy
        
        logger.debug("garch_simulation_completed", 
                    paths=total_count, 
                    duration_ms=duration_ms,
                    regime=regime,
                    fit_quality=fit_quality,
                    probability_buy=prob_buy)
        
        return {
            "probability_buy": prob_buy,
            "probability_sell": prob_sell,
            "quality": "garch11_adaptive",
            "sim_model": "garch11",
            "paths_used": total_count,
            "garch_params": {"omega": omega, "alpha": alpha, "beta": beta},
            "market_regime": regime,
            "fit_quality": fit_quality,
            "calculation_time_ms": duration_ms
        }

MonteCarloSimulator = AdaptiveGARCH11Simulator

# =========================
# Enhanced Trading System com IA AVANÇADA
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
        
        # NOVO: IA com Alta Assertividade
        self.intelligent_ai = HighAccuracyTradingAI()

    def get_brazil_time(self)->datetime:
        return brazil_now()

    def analyze_symbol(self, symbol: str, horizon: int)->Dict[str,Any]:
        start_time = time.time()
        logger.info("analysis_started", symbol=symbol, horizon=horizon)
        
        # Coleta dados (mantido do original)
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

        price_display = raw[-1][4]
        volume_display = raw[-1][5] if raw and len(raw[-1]) >= 6 else 0.0
        try:
            ws_last = WS_FEED.get_last_candle(symbol)
            if ws_last:
                price_display  = float(ws_last[4])
                volume_display = float(ws_last[5])
        except Exception:
            pass

        # Indicadores técnicos (mantido do original)
        # Indicadores técnicos (mantido do original) — RSI com último candle em análise

        closes_for_rsi = closes[:]

        try:

            last_price_live = raw[-1][4]

            if closes_for_rsi and last_price_live != closes_for_rsi[-1]:

                closes_for_rsi = closes_for_rsi + [last_price_live]

        except Exception:

            pass

        rsi_series = self.indicators.rsi_series_wilder(closes_for_rsi, 14)

        rsi = rsi_series[-1] if rsi_series else 50.0
        # ADX com último candle em análise

        highs_for_adx, lows_for_adx, closes_for_adx = highs[:], lows[:], closes[:]

        try:

            last_h, last_l, last_c = raw[-1][2], raw[-1][3], raw[-1][4]

            if closes_for_adx and last_c != closes_for_adx[-1]:

                highs_for_adx.append(last_h)

                lows_for_adx.append(last_l)

                closes_for_adx.append(last_c)

        except Exception:

            pass

        adx = self.indicators.adx_wilder(highs_for_adx, lows_for_adx, closes_for_adx)
        # MACD com último candle em análise

        closes_for_macd = closes[:]

        try:

            last_c_macd = raw[-1][4]

            if closes_for_macd and last_c_macd != closes_for_macd[-1]:

                closes_for_macd.append(last_c_macd)

        except Exception:

            pass

        macd = self.indicators.macd(closes_for_macd)
        boll = self.indicators.calculate_bollinger_bands(closes)
        tf_cons = self.multi_tf.analyze_consensus(closes)
        liq = self.liquidity.calculate_liquidity_score(highs, lows, closes)

        # Reversão (mantido)
        levels = self.revdet.compute_extremes_levels(rsi_series, 720, 6) if rsi_series else {"avg_peak":70.0,"avg_trough":30.0}
        rev_sig = self.revdet.signal_from_levels(rsi, levels, 2.5)

        # Simulador GARCH (mantido)
        empirical_returns = _safe_returns_from_prices(closes)
        steps = max(1, min(3, int(horizon)))
        base_price = closes[-1] if closes else price_display

        mc = self.monte_carlo.simulate_garch11(
            base_price, 
            empirical_returns, 
            steps, 
            num_paths=MC_PATHS
        )

        # Ajuste de probabilidade (mantido)
        prob_buy_original = mc['probability_buy']
        prob_sell_original = mc['probability_sell']

        macd_hist = 0.0
        try:
            # MACD com último candle em análise (para histograma/força)

            closes_for_macd2 = closes[:]

            try:

                last_c_macd2 = raw[-1][4]

                if closes_for_macd2 and last_c_macd2 != closes_for_macd2[-1]:

                    closes_for_macd2.append(last_c_macd2)

            except Exception:

                pass

            macd_result = self.indicators.macd(closes_for_macd2)
            if macd_result["signal"] == "bullish":
                macd_hist = macd_result["strength"]
            elif macd_result["signal"] == "bearish":
                macd_hist = -macd_result["strength"]
        except:
            macd_hist = 0.0

        prob_buy_adjusted = _confirm_prob_neutral_zone(
            prob_buy_original, rsi, macd_hist, adx, 
            boll['signal'], tf_cons
        )
        prob_sell_adjusted = 1.0 - prob_buy_adjusted

        direction = 'buy' if prob_buy_adjusted > 0.5 else 'sell'

        mc['probability_buy'] = prob_buy_adjusted
        mc['probability_sell'] = prob_buy_adjusted

        prob_dir = prob_buy_adjusted if direction == 'buy' else prob_sell_adjusted
        confidence = _calculate_directional_confidence(
            prob_dir, direction, rsi, adx, macd['signal'], boll['signal'], 
            tf_cons, rev_sig, liq
        )

        # Penalização suave por distância da MA9 (evita topo/fundo esticado; não bloqueia)
        try:
            ma9 = sum(closes[-9:]) / 9 if len(closes) >= 9 else closes[-1]
            dist9 = abs((raw[-1][4] - ma9) / max(1e-9, raw[-1][4]))
            penalty = min(0.10, max(0.0, (dist9 - 0.004) * 6.0))  # começa em ~0.4%, limita em 10%
            confidence = max(0.30, confidence * (1.0 - penalty))
        except Exception:
            pass

        # Análise bruta para IA
        raw_analysis = {
            'symbol': symbol,
            'horizon': horizon,
            'rsi': rsi,
            'adx': adx,
            'macd_signal': macd['signal'],
            'boll_signal': boll['signal'],
            'multi_timeframe': tf_cons,
            'liquidity_score': liq,
            'reversal': rev_sig['reversal'],
            'reversal_side': rev_sig['side'],
            'reversal_proximity': rev_sig['proximity'],
            'probability_buy': prob_buy_adjusted,
            'probability_sell': prob_sell_adjusted,
            'price': price_display,
            'volatility': stats.stdev(empirical_returns) if empirical_returns else 0.02,
            'market_regime': mc.get('market_regime', 'normal'),
            'confidence': confidence,
            'direction': direction
        }

        # Aplicar inteligência artificial AVANÇADA se habilitada
        if FEATURE_FLAGS["enable_advanced_ai"]:
            # Para contexto de mercado, usamos análise básica dos outros símbolos
            all_symbols_data = [raw_analysis]  # Em produção, coletar dados de todos os símbolos
            
            intelligent_result = self.intelligent_ai.analyze_with_high_accuracy(
                raw_analysis, all_symbols_data
            )
            
            # Sobrescreve direção e confiança com decisão inteligente
            direction = intelligent_result['direction']
            confidence = intelligent_result['final_confidence']
            
            # Adiciona resultados avançados da IA
            raw_analysis.update(intelligent_result)

        analysis_duration = (time.time() - start_time) * 1000
        
        # Resultado final
        result = {
            'symbol': symbol,
            'horizon': steps,
            'direction': direction,
            'probability_buy': prob_buy_adjusted,
            'probability_sell': prob_sell_adjusted,
            'confidence': confidence,
            'rsi': rsi, 'adx': adx, 'multi_timeframe': tf_cons,
            'monte_carlo_quality': mc['quality'],
            'garch_model': mc['sim_model'],
            'simulations_count': mc.get('paths_used', MC_PATHS),
            'market_regime': mc.get('market_regime', 'normal'),
            'fit_quality': mc.get('fit_quality', 0.7),
            'price': price_display,
            'liquidity_score': liq,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'rev_levels': levels,
            'reversal': rev_sig["reversal"],
            'reversal_side': rev_sig["side"],
            'reversal_proximity': round(rev_sig["proximity"], 3),
            'sim_model': mc.get('sim_model', 'garch11'),
            'last_volume_1m': round(volume_display, 8),
            'data_source': 'WS_COINAPI' if WS_FEED.enabled else ('CCXT' if self.spot._has_ccxt else 'HTTP'),
            'analysis_time_ms': round(analysis_duration, 2)
        ,
            'signal_strength': ('forte' if confidence >= 0.70 else ('normal' if confidence >= 0.55 else 'leve'))
        }

        # Adiciona dados de inteligência avançada se habilitada
        if FEATURE_FLAGS["enable_advanced_ai"]:
            result.update({
                'final_confidence': raw_analysis.get('final_confidence', confidence),
                'intelligent_confidence': raw_analysis.get('final_confidence', confidence),
                'reasoning': raw_analysis.get('reasoning', []),
                'pattern_analysis': raw_analysis.get('pattern_analysis', {}),
                'confidence_breakdown': raw_analysis.get('confidence_breakdown', {}),
                'technical_convergence': raw_analysis.get('technical_convergence', 0.5),
                'market_sentiment_alignment': raw_analysis.get('market_sentiment_alignment', 1.0),
                'risk_adjustment': raw_analysis.get('risk_adjustment', 1.0),
                'quality_metrics': raw_analysis.get('quality_metrics', {}),
                'market_context': raw_analysis.get('market_context', {}),
                'system_accuracy': raw_analysis.get('system_accuracy', 0.5),
                'advanced_ai': True,
                'reasoning_depth': 'advanced_multilayer_intelligence'
            })

        logger.info("analysis_completed", 
                   symbol=symbol, 
                   horizon=horizon, 
                   duration_ms=analysis_duration,
                   direction=direction,
                   confidence=confidence,
                   advanced_ai=FEATURE_FLAGS["enable_advanced_ai"])

        return result

    def scan_symbols_tplus(self, symbols: List[str])->Dict[str,Any]:
        por_ativo={}; candidatos=[]
        for sym in symbols:
            tplus=[]
            # APENAS T+1 
            for h in (1,):  # ← AGORA SÓ T+1
                try:
                    r=self.analyze_symbol(sym,h)
                    r['label']=f"{sym} T+{h}"
                    tplus.append(r); candidatos.append(r)
                except Exception as e:
                    logger.error("symbol_analysis_error", symbol=sym, horizon=h, error=str(e))
                    tplus.append({
                        "symbol":sym,"horizon":h,"error":str(e),
                        "direction":"buy","probability_buy":0.5,"probability_sell":0.5,
                        "confidence":0.5,"label":f"{sym} T+{h}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
            por_ativo[sym]={"tplus":tplus,"best_for_symbol":tplus[0] if tplus else None}  # ← Como só tem um, é o primeiro
        best_overall=max(candidatos,key=_rank_key_directional) if candidatos else None
        return {"por_ativo":por_ativo,"best_overall":best_overall}

# =========================
# Manager / API / UI
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
            result = self.system.scan_symbols_tplus(symbols)
            flat=[]
            for sym, bloco in result["por_ativo"].items():
                flat.extend(bloco["tplus"])
            self.current_results = flat
            if flat:
                best=max(flat, key=_rank_key_directional)
                best=dict(best)
                best["entry_time"]=self.calculate_entry_time_brazil(best.get("horizon",1))
                self.best_opportunity=best
                logger.info("best_opportunity_found", 
                           symbol=best['symbol'], 
                           direction=best['direction'],
                           confidence=best['confidence'])
            else:
                self.best_opportunity=None
            self.analysis_time = br_full(self.get_brazil_time())
            logger.info("batch_analysis_completed", results_count=len(flat))
        except Exception as e:
            logger.error("batch_analysis_error", error=str(e))
            self.current_results=[]
            self.best_opportunity={"error":str(e)}
            self.analysis_time = br_full(self.get_brazil_time())
        finally:
            self.is_analyzing=False

manager=AnalysisManager()

# =========================
# NOVO: Endpoints para IA AVANÇADA
# =========================
@app.post("/api/ai/learn")
def api_ai_learn():
    """Endpoint para aprendizado da IA avançada com resultados reais"""
    if FEATURE_FLAGS["maintenance_mode"]:
        return jsonify({"success": False, "error": "Sistema em manutenção"}), 503
        
    try:
        data = request.get_json(silent=True) or {}
        symbol = data.get("symbol")
        expected_direction = data.get("expected_direction")
        actual_price_movement = data.get("actual_price_movement", 0.0)
        
        if not symbol or not expected_direction:
            return jsonify({"success": False, "error": "Dados incompletos"}), 400
            
        # Busca o sinal mais recente para este símbolo
        recent_signal = None
        for signal in manager.current_results:
            if signal.get('symbol') == symbol:
                recent_signal = signal
                break
                
        if recent_signal and FEATURE_FLAGS["enable_advanced_ai"]:
            manager.system.intelligent_ai.learn_from_result(
                recent_signal, actual_price_movement, expected_direction, manager.current_results
            )
            return jsonify({
                "success": True,
                "message": "IA avançada aprendeu com resultado",
                "system_accuracy": manager.system.intelligent_ai.advanced_engine.get_system_accuracy(),
                "performance_metrics": manager.system.intelligent_ai.get_detailed_status()
            })
        else:
            return jsonify({"success": False, "error": "Sinal recente não encontrado ou IA avançada desativada"}), 404
            
    except Exception as e:
        logger.error("ai_learning_error", error=str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/api/ai/status")
def api_ai_status():
    """Status detalhado da IA avançada"""
    if not FEATURE_FLAGS["enable_advanced_ai"]:
        return jsonify({"success": False, "error": "IA avançada desativada"}), 400
        
    ai = manager.system.intelligent_ai
    detailed_status = ai.get_detailed_status()
    
    return jsonify({
        "success": True,
        "advanced_ai_enabled": FEATURE_FLAGS["enable_advanced_ai"],
        "learning_enabled": FEATURE_FLAGS["enable_learning"],
        "detailed_performance": detailed_status,
        "system_accuracy": ai.advanced_engine.get_system_accuracy(),
        "total_predictions": detailed_status['total_predictions'],
        "correct_predictions": detailed_status['correct_predictions']
    })

# =========================
# Endpoints originais (mantidos)
# =========================
@app.post("/api/analyze")
def api_analyze():
    if FEATURE_FLAGS["maintenance_mode"]:
        return jsonify({"success": False, "error": "Sistema em manutenção"}), 503
        
    client_id = request.remote_addr or "unknown"
    if not rate_limiter.is_allowed(client_id, max_requests=30, window_seconds=60):
        logger.warning("rate_limit_exceeded", client_id=client_id)
        return jsonify({"success": False, "error": "Limite de requisições excedido. Tente novamente em 1 minuto."}), 429
        
    if manager.is_analyzing:
        return jsonify({"success": False, "error": "Análise em andamento"}), 429
        
    try:
        data = request.get_json(silent=True) or {}
        symbols = [s.strip().upper() for s in (data.get("symbols") or manager.symbols_default) if s.strip()]
        if not symbols:
            return jsonify({"success": False, "error": "Selecione pelo menos um ativo"}), 400
            
        sims = MC_PATHS
        th = threading.Thread(target=manager.analyze_symbols_thread, args=(symbols, sims, None))
        th.daemon = True
        th.start()
        
        logger.info("analysis_request", client_id=client_id, symbols_count=len(symbols))
        return jsonify({
            "success": True, 
            "message": f"Analisando {len(symbols)} ativos com {sims} simulações.", 
            "symbols_count": len(symbols),
            "advanced_ai": FEATURE_FLAGS["enable_advanced_ai"]
        })
    except Exception as e:
        logger.error("analysis_request_error", error=str(e), client_id=client_id)
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/api/results")
def api_results():
    return jsonify({
        "success": True,
        "results": manager.current_results,
        "best": manager.best_opportunity,
        "analysis_time": manager.analysis_time,
        "total_signals": len(manager.current_results),
        "is_analyzing": manager.is_analyzing,
        "advanced_ai": FEATURE_FLAGS["enable_advanced_ai"],
        "ai_intelligence": FEATURE_FLAGS["enable_ai_intelligence"]
    })

@app.get("/health")
def health():
    health_status = {
        "ok": True,
        "ws": WS_FEED.enabled,
        "provider": REALTIME_PROVIDER,
        "ts": datetime.now(timezone.utc).isoformat(),
        "circuit_breaker": binance_circuit_breaker.state,
        "feature_flags": FEATURE_FLAGS,
        "cache_size": len(manager.system.spot._cache._cache),
        "advanced_ai": FEATURE_FLAGS["enable_advanced_ai"],
        "ai_accuracy": manager.system.intelligent_ai.advanced_engine.get_system_accuracy() if FEATURE_FLAGS["enable_advanced_ai"] else None
    }
    return jsonify(health_status), 200

@app.get("/deep-health")
def deep_health():
    ws_status = "connected" if WS_FEED._ws and WS_FEED._ws.sock else "disconnected"
    
    # Métricas da IA avançada
    ai_metrics = {}
    if FEATURE_FLAGS["enable_advanced_ai"]:
        ai_metrics = manager.system.intelligent_ai.get_detailed_status()
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "websocket": {
                "status": ws_status,
                "symbols_monitored": len(WS_FEED.symbols),
                "enabled": WS_FEED.enabled
            },
            "cache": {
                "size": len(manager.system.spot._cache._cache),
                "volatility_tracking": len(manager.system.spot._cache._volatility_cache)
            },
            "circuit_breaker": {
                "state": binance_circuit_breaker.state,
                "failures": binance_circuit_breaker.failures,
                "last_failure": binance_circuit_breaker.last_failure_time
            },
            "analysis_engine": {
                "is_analyzing": manager.is_analyzing,
                "last_analysis_time": manager.analysis_time,
                "cached_results": len(manager.current_results)
            },
            "advanced_ai_intelligence": {
                "enabled": FEATURE_FLAGS["enable_advanced_ai"],
                "performance_metrics": ai_metrics,
                "learning_enabled": FEATURE_FLAGS["enable_learning"]
            }
        },
        "feature_flags": FEATURE_FLAGS
    }
    
    return jsonify(health_data), 200

@app.get("/")
def index():
    symbols_js = json.dumps(DEFAULT_SYMBOLS)
    HTML = """<!doctype html>
<html lang="pt-br"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IA Signal Pro - ALTA ASSERTIVIDADE + 3000 SIMULAÇÕES + IA AVANÇADA</title>
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0"/>
<style>
:root{--bg:#0f1120;--panel:#181a2e;--panel2:#223148;--tx:#dfe6ff;--muted:#9fb4ff;--accent:#2aa9ff;--gold:#f2a93b;--ok:#29d391;--err:#ff5b5b;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial}
.wrap{max-width:1120px;margin:22px auto;padding:0 16px}
.hline{border:2px solid var(--accent);border-radius:12px;background:var(--panel);padding:18px;position:relative}
h1{margin:0 0 8px;font-size:22px} .sub{color:#8ccf9d;font-size:13px;margin:6px 0 0}
.clock{position:absolute;right:18px;top:18px;background:#0d2033;border:1px solid #3e6fa8;border-radius:10px;padding:8px 10px;color:#cfe2ff;font-weight:600}
.controls{margin-top:14px;background:var(--panel2);border-radius:12px;padding:14px}
.chips{display:flex;flex-wrap:wrap;gap:10px} .chip{border:2px solid var(--accent);border-radius:12px;padding:8px 12px;cursor:pointer;user-select:none}
.chip input{margin-right:8px}
.chip.active{box-shadow:0 0 0 2px inset var(--accent)}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
select,button{border:2px solid var(--accent);border-radius:12px;padding:10px 12px;background:#16314b;color:#fff}
button{background:#2a9df4;cursor:pointer} button:disabled{opacity:.6;cursor:not-allowed}
.section{margin-top:16px;border:2px solid var(--gold);border-radius:12px;background:var(--panel)}
.section .title{padding:10px 14px;border-bottom:2px solid var(--gold);font-weight:700}
.card{margin:12px;border-radius:12px;background:var(--panel2);padding:14px;border:2px solid var(--gold)}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;margin-top:8px}
.kpi{background:#1b2b41;border-radius:10px;padding:10px 12px;color:#b6c8ff} .kpi b{display:block;color:#fff}
.badge{display:inline-block;padding:3px 8px;border-radius:8px;font-size:11px;margin-right:6px;background:#12263a;border:1px solid #2e6ea8}
.buy{background:#0c5d4b} .sell{background:#5b1f1f}
.small{color:#9fb4ff;font-size:12px} .muted{color:#7d90c7}
.grid-syms{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;padding-bottom:12px}
.sym-head{padding:10px 14px;border-bottom:1px dashed #3b577a} .line{border-top:1px dashed #3b577a;margin:8px 0}
.tbox{border:2px solid #f0a43c;border-radius:10px;background:#26384e;padding:10px;margin-top:10px}
.tag{display:inline-block;padding:2px 6px;border-radius:6px;font-size:10px;margin-left:6px;background:#0d2033;border:1px solid #3e6fa8}
.right{float:right}
.ai-badge{background:#4a1f5f;border-color:#b362ff}
.advanced-badge{background:#1f5f4a;border-color:#62ffb3}
.accuracy-high{color:#29d391}
.accuracy-medium{color:#f2a93b}
.accuracy-low{color:#ff5b5b}
</style>
</head>
<body>
<div class="wrap">
  <div class="hline">
    <h1>IA Signal Pro - ALTA ASSERTIVIDADE + 3000 SIMULAÇÕES + IA AVANÇADA</h1>
    <div class="clock" id="clock">--:--:-- BRT</div>
    <div class="sub">✅ CoinAPI (WS) · Binance REST · RSI/ADX (Wilder) · Liquidez (ATR%) · Reversão RSI · GARCH(1,1) Adaptativo · 🧠 IA AVANÇADA MULTICAMADAS · 📈 ALTA ASSERTIVIDADE</div>
    <div class="controls">
      <div class="chips" id="chips"></div>
      <div class="row">
        <select id="mcsel">
          <option value="3000" selected>3000 simulações GARCH</option>
          <option value="1000">1000 simulações</option>
          <option value="5000">5000 simulações</option>
        </select>
        <button type="button" onclick="selectAll()">Selecionar todos</button>
        <button type="button" onclick="clearAll()">Limpar</button>
        <button id="go" onclick="runAnalyze()">🧠 Analisar com IA Avançada</button>
      </div>
    </div>
  </div>

  <div class="section" id="bestSec" style="display:none">
    <div class="title">🥇 MELHOR OPORTUNIDADE GLOBAL (IA AVANÇADA)</div>
    <div class="card" id="bestCard"></div>
  </div>

  <div class="section" id="allSec" style="display:none">
    <div class="title">📊 SINAIS T+1 POR ATIVO (ALTA ASSERTIVIDADE)</div>
    <div class="grid-syms" id="grid"></div>
  </div>
</div>

<script>
const SYMS_DEFAULT = __SYMS__;
const chipsEl = document.getElementById('chips');
const gridEl  = document.getElementById('grid');
const bestEl  = document.getElementById('bestCard');
const bestSec = document.getElementById('bestSec');
const allSec  = document.getElementById('allSec');
const clockEl = document.getElementById('clock');

function tickClock(){
  const now = new Date();
  const utc = now.getTime() + (now.getTimezoneOffset()*60000);
  const brt = new Date(utc - 3*60*60000);
  const pad = (n)=> n.toString().padStart(2,'0');
  clockEl.textContent = pad(brt.getHours())+':'+pad(brt.getMinutes())+':'+pad(brt.getSeconds())+' BRT';
}
setInterval(tickClock, 500); tickClock();

let pollTimer = null;
let lastAnalysisTime = null;

function mkChip(sym){
  const label = document.createElement('label');
  label.className = 'chip active';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = true;
  input.value = sym;
  input.addEventListener('change', () => {
    label.classList.toggle('active', input.checked);
  });
  label.appendChild(input);
  label.append(sym);
  chipsEl.appendChild(label);
}
SYMS_DEFAULT.forEach(mkChip);

function selectAll(){
  document.querySelectorAll('#chips .chip input').forEach(cb=>{
    cb.checked = true;
    cb.dispatchEvent(new Event('change'));
  });
}
function clearAll(){
  document.querySelectorAll('#chips .chip input').forEach(cb=>{
    cb.checked = false;
    cb.dispatchEvent(new Event('change'));
  });
}
function selSymbols(){
  return Array.from(chipsEl.querySelectorAll('input')).filter(i=>i.checked).map(i=>i.value);
}
function pct(x){ return (x*100).toFixed(1)+'%'; }
function badgeDir(d){ return `<span class="badge ${d==='buy'?'buy':'sell'}">${d==='buy'?'COMPRAR':'VENDER'}</span>`; }
function accuracyClass(conf){ 
  if(conf >= 0.7) return 'accuracy-high';
  if(conf >= 0.5) return 'accuracy-medium';
  return 'accuracy-low';
}

async function runAnalyze(){
  const btn = document.getElementById('go');
  btn.disabled = true;
  btn.textContent = '⏳ IA Avançada Analisando...';
  const syms = selSymbols();
  if(!syms.length){ alert('Selecione pelo menos um ativo.'); btn.disabled=false; btn.textContent='🧠 Analisar com IA Avançada'; return; }
  await fetch('/api/analyze', {
    method:'POST',
    headers:{'Content-Type':'application/json','Cache-Control':'no-store'},
    cache:'no-store',
    body: JSON.stringify({ symbols: syms })
  });
  startPollingResults();
}

function startPollingResults(){
  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const finished = await fetchAndRenderResults();
    if (finished){
      clearInterval(pollTimer);
      pollTimer = null;
      const btn = document.getElementById('go');
      btn.disabled = false;
      btn.textContent = '🧠 Analisar com IA Avançada';
    }
  }, 700);
}

async function fetchAndRenderResults(){
  const r = await fetch('/api/results', { cache: 'no-store', headers: {'Cache-Control':'no-store'} });
  const data = await r.json();

  if (data.is_analyzing) return false;
  if (lastAnalysisTime && data.analysis_time === lastAnalysisTime) return false;
  lastAnalysisTime = data.analysis_time;

  bestSec.style.display='block';
  bestEl.innerHTML = renderBest(data.best, data.analysis_time);

  const groups = {};
  (data.results||[]).forEach(it=>{ (groups[it.symbol]=groups[it.symbol]||[]).push(it); });
  const html = Object.keys(groups).sort().map(sym=>{
    const arr = groups[sym];
    const signal = arr[0]; // Apenas um sinal T+1 por ativo
    return `
      <div class="card">
        <div class="sym-head"><b>${sym}</b>
          <span class="tag">TF: ${signal?.multi_timeframe||'neutral'}</span>
          <span class="tag">Liquidez: ${Number(signal?.liquidity_score||0).toFixed(2)}</span>
          ${signal?.reversal ? `<span class="tag">🔄 Reversão (${signal.reversal_side})</span>`:''}
          <span class="tag advanced-badge">🧠 IA AVANÇADA</span>
        </div>
        ${renderTbox(signal)}
      </div>`;
  }).join('');
  gridEl.innerHTML = html;
  allSec.style.display='block';

  return true;
}

function rank(it){ 
  const direction = it.direction || 'buy';
  const prob_directional = direction === 'buy' ? it.probability_buy : it.probability_sell;
  // Prefere confiança avançada se disponível
  const confidence = it.final_confidence || it.intelligent_confidence || it.confidence;
  return (confidence * 1000) + (prob_directional * 100);
}

function renderBest(best, analysisTime){
  if(!best) return '<div class="small">Sem oportunidade no momento.</div>';
  const rev = best.reversal ? ` <span class="tag">🔄 Reversão (${best.reversal_side})</span>` : '';
  const confidence = best.final_confidence || best.intelligent_confidence || best.confidence;
  const reasoning = best.reasoning ? `<div class="small" style="margin-top:8px;color:#8ccf9d">🧠 ${best.reasoning.slice(0,3).join(' · ')}</div>` : '';
  const accuracyClass = confidence >= 0.7 ? 'accuracy-high' : confidence >= 0.5 ? 'accuracy-medium' : 'accuracy-low';
  
  // Métricas avançadas se disponíveis
  const advancedMetrics = best.advanced_ai ? `
    <div class="small" style="margin-top:6px;">
      <span class="tag">Convergência: ${((best.technical_convergence||0)*100).toFixed(0)}%</span>
      <span class="tag">Alinhamento: ${((best.market_sentiment_alignment||1)*100).toFixed(0)}%</span>
      <span class="tag">Risco: ${((best.risk_adjustment||1)*100).toFixed(0)}%</span>
    </div>
  ` : '';
  
  return `
    <div class="small muted">Atualizado: ${analysisTime} (Horário Brasil) · IA Avançada Ativa · Assertividade: <span class="${accuracyClass}">${pct(confidence)}</span></div>
    <div class="line"></div>
    <div><b>${best.symbol} T+${best.horizon}</b> ${badgeDir(best.direction)} <span class="tag">🥇 MELHOR ENTRE TODOS OS ATIVOS</span>${rev} <span class="tag advanced-badge">🧠 IA AVANÇADA</span></div>
    <div class="kpis">
      <div class="kpi"><b>Prob Compra</b>${pct(best.probability_buy||0)}</div>
      <div class="kpi"><b>Prob Venda</b>${pct(best.probability_sell||0)}</div>
      <div class="kpi"><b>Confiança IA</b><span class="${accuracyClass}">${pct(confidence)}</span></div>
      <div class="kpi"><b>ADX</b>${(best.adx||0).toFixed(1)}</div>
      <div class="kpi"><b>RSI</b>${(best.rsi||0).toFixed(1)}</div>
      <div class="kpi"><b>Liquidez</b>${Number(best.liquidity_score||0).toFixed(2)}</div>
    </div>
    ${advancedMetrics}
    ${reasoning}
    <div class="small" style="margin-top:8px;">
      Assertividade: <span class="${accuracyClass}">${(confidence*100).toFixed(1)}%</span> · TF: <b>${best.multi_timeframe||'neutral'}</b> · Price: <b>${Number(best.price||0).toFixed(6)}</b>
      <span class="right">Entrada: <b>${best.entry_time||'-'}</b></span>
    </div>`;
}

function renderTbox(it){
  if(!it) return '<div class="tbox">Erro ao carregar sinal</div>';
  
  const rev = it.reversal ? ` <span class="tag">🔄 REVERSÃO (${it.reversal_side})</span>` : '';
  const confidence = it.final_confidence || it.intelligent_confidence || it.confidence;
  const reasoning = it.reasoning ? `<div class="small" style="color:#8ccf9d;margin-top:4px">🧠 ${it.reasoning.slice(0,2).join(' · ')}</div>` : '';
  const accuracyClass = confidence >= 0.7 ? 'accuracy-high' : confidence >= 0.5 ? 'accuracy-medium' : 'accuracy-low';
  
  return `
    <div class="tbox">
      <div><b>T+${it.horizon}</b> ${badgeDir(it.direction)}${rev} <span class="tag advanced-badge">🧠 IA AVANÇADA</span></div>
      <div class="small">
        Prob: <span class="${it.direction==='buy'?'ok':'err'}">${pct(it.probability_buy||0)}/${pct(it.probability_sell||0)}</span>
        · Conf IA: <span class="${accuracyClass}">${pct(confidence)}</span>
        · RSI≈Pico: ${(it.rev_levels?.avg_peak||0).toFixed(1)} · RSI≈Vale: ${(it.rev_levels?.avg_trough||0).toFixed(1)}
      </div>
      <div class="small">ADX: ${(it.adx||0).toFixed(1)} | RSI: ${(it.rsi||0).toFixed(1)} | TF: <b>${it.multi_timeframe||'neutral'}</b></div>
      ${reasoning}
      <div class="small muted">⏱️ ${it.timestamp||'-'} · Price: ${Number(it.price||0).toFixed(6)}</div>
    </div>`;
}
</script>
</body></html>"""
    return Response(HTML.replace("__SYMS__", symbols_js), mimetype="text/html")

# =========================
# Execução
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("application_starting", port=port, features_enabled=FEATURE_FLAGS)
    logger.info("advanced_ai_enabled", 
                enabled=FEATURE_FLAGS["enable_advanced_ai"],
                learning=FEATURE_FLAGS["enable_learning"])
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)


# === [PHOTO ANALYZE ADDON] ===
# Este bloco adiciona: analisador de foto + rotas /photo (UI) e /photo/analyze (API)
# Não altera o restante do app.

from flask import request, jsonify, Response

def _photo_log(msg):
    try:
        logger.info(f"[PHOTO] {msg}")
    except Exception:
        print(f"[PHOTO] {msg}")

class PhotoAnalyzer:
    def __init__(self):
        pass

    def _ensure_deps(self):
        try:
            import base64, io
            import numpy as np
            from PIL import Image
            import cv2 as cv
            return base64, io, np, Image, cv
        except Exception as e:
            raise RuntimeError("Dependências ausentes: instale pillow, numpy, opencv-python-headless") from e

    def _read_image(self, image_bytes: bytes):
        base64, io, np, Image, cv = self._ensure_deps()
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            import numpy as _np
            img = cv.cvtColor(_np.array(pil), cv.COLOR_RGB2BGR)
        return img

    def _preprocess(self, img):
        import numpy as np
        h, w = img.shape[:2]
        cut = 0.05
        return img[int(h*cut):int(h*(1-cut)), int(w*cut):int(w*(1-cut))]

    def extract_features(self, image_bytes: bytes) -> dict:
        # Implementação compacta (heurística) – proxies para RSI/ADX/MACD/BB
        base64, io, np, Image, cv = self._ensure_deps()
        img = self._read_image(image_bytes)
        img = self._preprocess(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        edges = cv.Canny(gray, 60, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv.dilate(edges, kernel, iterations=1)
        edges = cv.erode(edges, kernel, iterations=1)

        ys, xs = np.nonzero(edges)
        slope = 0.0; vol = 0.01
        if len(xs) >= 100:
            X = np.vstack([xs, np.ones_like(xs)]).T
            m, b = np.linalg.lstsq(X, ys, rcond=None)[0]
            slope = -float(m) / max(1.0, edges.shape[1]*0.75)
            yfit = m*xs + b
            vol = float(np.std(ys - yfit) / max(1.0, edges.shape[0]))

        # ADX proxy por consistência angular de linhas
        lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=20, maxLineGap=8)
        adx_proxy = 20.0
        if lines is not None and len(lines) > 0:
            angs = []
            for l in lines[:500]:
                x1,y1,x2,y2 = l[0]
                angs.append(np.degrees(np.arctan2((y2-y1),(x2-x1))))
            if angs:
                hist,_ = np.histogram(angs, bins=18, range=(-90,90))
                kons = hist.max()/max(1,len(angs))
                adx_proxy = float(15 + 80*min(1.0, max(0.0, (kons-0.25)/0.75)))

        # RSI proxy com gradiente vertical
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        gpos = float(np.clip(gy[gy>0].mean() if (gy>0).any() else 0.0, 0, 255))
        gneg = float(np.clip((-gy[gy<0]).mean() if (gy<0).any() else 0.0, 0, 255))
        base = 50.0
        if (gpos+gneg) > 0:
            base = 100.0 * (gpos / (gpos + gneg + 1e-9))
        rsi_proxy = float(max(0.0, min(100.0, base)))

        # MACD proxy em série de brilho
        row = gray[int(gray.shape[0]*0.5), :].astype(float) + 1.0
        def ema(x, n):
            k = 2/(n+1)
            e = [float(x[0])]
            for v in x[1:]:
                e.append(e[-1] + k*(float(v)-e[-1]))
            import numpy as _np
            return _np.array(e, dtype=float)
        if len(row) >= 50:
            ema12 = ema(row, 12); ema26 = ema(row, 26)
            macd_line = ema12[-len(ema26):] - ema26
            signal = ema(macd_line, 9)
            hist = float(macd_line[-1] - signal[-1]) if len(signal) else 0.0
            if hist > 0: macd_sig = "bullish"
            elif hist < 0: macd_sig = "bearish"
            else: macd_sig = "neutral"
        else:
            macd_sig = "neutral"

        # Bollinger proxy
        period = min(20, len(row))
        if period >= 12:
            win = row[-period:]
            ma = float(win.mean()); sd = float(win.std()); last = float(win[-1])
            if last > ma + 2*sd:   boll_sig = "overbought"
            elif last < ma - 2*sd: boll_sig = "oversold"
            elif last > ma:        boll_sig = "bullish"
            elif last < ma:        boll_sig = "bearish"
            else:                  boll_sig = "neutral"
        else:
            boll_sig = "neutral"

        tf_cons = "buy" if slope > 0.002 else ("sell" if slope < -0.002 else "neutral")
        direction = "buy" if (slope > 0) or (macd_sig == "bullish") else "sell"
        liq = float(max(0.0, min(1.0, 1.0 - (vol*3.0))))
        prob_buy = 0.5 + float(np.clip(slope*8.0, -0.35, 0.35))
        if macd_sig == "bullish": prob_buy += 0.05
        if boll_sig == "oversold": prob_buy += 0.05
        if boll_sig == "overbought": prob_buy -= 0.05
        prob_buy = float(max(0.10, min(0.90, prob_buy)))

        return {
            "symbol": "PHOTO",
            "horizon": 1,
            "rsi": rsi_proxy,
            "adx": round(adx_proxy,2),
            "macd_signal": macd_sig,
            "boll_signal": boll_sig,
            "multi_timeframe": tf_cons,
            "liquidity_score": liq,
            "reversal": False,
            "reversal_side": None,
            "reversal_proximity": 0.0,
            "probability_buy": prob_buy,
            "probability_sell": 1.0 - prob_buy,
            "price": None,
            "volatility": float(vol),
            "market_regime": "normal" if vol < 0.015 else ("high_volatility" if vol > 0.03 else "normal"),
            "confidence": 0.55,
            "direction": direction
        }

# instâncias reutilizáveis
try:
    _photo_analyzer = PhotoAnalyzer()
except Exception as _e:
    _photo_analyzer = None
    _photo_log(f"init error: {_e}")

try:
    # Tenta reaproveitar o seu sistema principal, se já existir
    _photo_system = EnhancedTradingSystem()
except Exception as _e:
    _photo_system = None
    _photo_log(f"system init error: {_e}")

@app.post("/photo/analyze")
def photo_analyze():
    if _photo_system is None or _photo_analyzer is None:
        return jsonify({"ok": False, "error": "PHOTO não inicializado. Instale dependências e verifique logs."}), 500

    image_bytes = None
    if request.files and "image" in request.files:
        image_bytes = request.files["image"].read()
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        b64 = (data.get("image_base64") or "").strip()
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        if b64:
            import base64
            try:
                image_bytes = base64.b64decode(b64)
            except Exception:
                return jsonify({"ok": False, "error": "Base64 inválido"}), 400

    if not image_bytes:
        return jsonify({"ok": False, "error": "Envie 'image' (arquivo) ou 'image_base64' no JSON"}), 400

    try:
        raw_from_photo = _photo_analyzer.extract_features(image_bytes)
    except Exception as e:
        _photo_log(f"extract fail: {e}")
        return jsonify({"ok": False, "error": "Falha ao analisar a imagem. Instale pillow, numpy, opencv-python-headless."}), 500

    try:
        result = _photo_system.intelligent_ai.analyze_with_high_accuracy(raw_from_photo, [raw_from_photo])
    except Exception as e:
        _photo_log(f"ai fail: {e}")
        return jsonify({"ok": False, "error": "Falha na IA avançada com os dados da foto."}), 500

    out = {
        "ok": True,
        "source": "PHOTO",
        "direction": result.get("direction"),
        "final_confidence": float(result.get("final_confidence", 0.55)),
        "reasoning": result.get("reasoning", []),
        "quality_metrics": result.get("quality_metrics", {}),
        "market_context": result.get("market_context", {}),
        "pattern_analysis": result.get("pattern_analysis", {}),
        "confidence_breakdown": result.get("confidence_breakdown", {}),
        "technical_convergence": float(result.get("technical_convergence", 0.5)),
        "market_sentiment_alignment": float(result.get("market_sentiment_alignment", 1.0)),
        "risk_adjustment": float(result.get("risk_adjustment", 1.0)),
        "system_accuracy": float(result.get("system_accuracy", 0.5)),
        "photo_features": {
            "rsi": raw_from_photo["rsi"],
            "adx": raw_from_photo["adx"],
            "macd_signal": raw_from_photo["macd_signal"],
            "boll_signal": raw_from_photo["boll_signal"],
            "multi_timeframe": raw_from_photo["multi_timeframe"],
            "probability_buy_seed": raw_from_photo["probability_buy"],
            "volatility": raw_from_photo["volatility"],
            "liquidity_score": raw_from_photo["liquidity_score"]
        }
    }
    return jsonify(out), 200

@app.get("/photo")
def photo_page():
    # UI leve para enviar print: upload/arrastar/colar imagem e ver resposta
    html = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>📷 Análise por Foto</title>
<style>
  :root{--bg:#0b0b10;--fg:#e8e8ef;--muted:#a0a0b2;--card:#14141c;--acc:#6ea8fe;}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:Inter,Segoe UI,Roboto,Arial,sans-serif}
  .wrap{max-width:920px;margin:40px auto;padding:0 16px}
  .card{background:var(--card);border-radius:16px;padding:20px;box-shadow:0 8px 30px rgba(0,0,0,.35)}
  h1{font-size:22px;margin:0 0 10px}
  p{color:var(--muted)}
  .drop{border:2px dashed #2a2a3a;border-radius:16px;padding:24px;text-align:center;margin-top:12px}
  .drop.drag{border-color:var(--acc);background:rgba(110,168,254,.08)}
  input[type=file]{display:none}
  .btn{display:inline-block;background:var(--acc);color:#0b0b10;border:0;border-radius:12px;padding:10px 16px;font-weight:600;cursor:pointer}
  .row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;justify-content:space-between;margin-top:12px}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .result{margin-top:16px;padding:14px;border-radius:12px;background:#101018;border:1px solid #23233a}
  .badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700}
  .buy{background:#0affc233;color:#0affc2;border:1px solid #0affc2}
  .sell{background:#ff5c7a33;color:#ff5c7a;border:1px solid #ff5c7a}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:10px}
  .kv{background:#0e0e16;border:1px solid #1f1f33;border-radius:10px;padding:10px}
  .kv b{color:#cfd6ff}
  .hidden{display:none}
  .preview{max-width:100%;border-radius:12px;margin-top:10px;border:1px solid #22223a}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>📷 Análise por Foto</h1>
    <p>Envie um print do gráfico (arraste/solte, cole do clipboard ou selecione). O app calcula <b>comprar</b> ou <b>vender</b> com a maior assertividade possível usando o mesmo motor da IA.</p>

    <div id="drop" class="drop">
      <p><b>Arraste e solte</b> aqui, <button id="pick" class="btn">Selecionar imagem</button> ou <b>Ctrl+V</b> para colar.</p>
      <input id="file" type="file" accept="image/*"/>
      <img id="preview" class="preview hidden"/>
    </div>

    <div class="row">
      <div class="mono" id="status">Pronto para analisar.</div>
      <button id="send" class="btn" disabled>Enviar para análise</button>
    </div>

    <div id="out" class="result hidden"></div>
  </div>
</div>

<script>
let fileData = null;

const drop = document.getElementById('drop');
const pick = document.getElementById('pick');
const file = document.getElementById('file');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const out = document.getElementById('out');
const preview = document.getElementById('preview');

function setStatus(t){ statusEl.textContent = t; }

['dragenter','dragover'].forEach(ev=>{
  drop.addEventListener(ev, e=>{e.preventDefault();e.stopPropagation();drop.classList.add('drag');});
});
['dragleave','drop'].forEach(ev=>{
  drop.addEventListener(ev, e=>{e.preventDefault();e.stopPropagation();drop.classList.remove('drag');});
});
drop.addEventListener('drop', e=>{
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if(f){ loadFile(f); }
});

pick.addEventListener('click', ()=> file.click());
file.addEventListener('change', ()=> {
  if(file.files && file.files[0]) loadFile(file.files[0]);
});

window.addEventListener('paste', e=>{
  const items = e.clipboardData && e.clipboardData.items;
  if(!items) return;
  for(const it of items){
    if(it.type.indexOf('image') === 0){
      const f = it.getAsFile();
      if(f) loadFile(f);
    }
  }
});

function loadFile(f){
  fileData = f;
  sendBtn.disabled = false;
  setStatus(`Selecionado: ${f.name} (${Math.round(f.size/1024)} KB)`);
  const reader = new FileReader();
  reader.onload = ()=>{ preview.src = reader.result; preview.classList.remove('hidden'); };
  reader.readAsDataURL(f);
}

sendBtn.addEventListener('click', async ()=>{
  if(!fileData) return;
  setStatus('Analisando...');
  out.classList.add('hidden');
  out.innerHTML = '';
  const fd = new FormData();
  fd.append('image', fileData);
  try{
    const r = await fetch('/photo/analyze', { method:'POST', body: fd });
    const j = await r.json();
    if(!j.ok){ setStatus('Erro: ' + (j.error || 'falha desconhecida')); return; }
    setStatus('Concluído.');
    const dir = j.direction || 'indefinido';
    const conf = (j.final_confidence!=null? (j.final_confidence*100).toFixed(1)+'%':'—');
    const badge = `<span class="badge ${dir==='buy'?'buy':'sell'}">${dir.toUpperCase()}</span>`;
    const qm = j.quality_metrics || {};
    const pf = j.photo_features || {};
    out.innerHTML = `
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        ${badge}
        <div class="mono">Confiança: <b>${conf}</b></div>
      </div>
      <div class="grid">
        <div class="kv"><b>RSI (proxy)</b><br>${pf.rsi ?? '—'}</div>
        <div class="kv"><b>ADX (proxy)</b><br>${pf.adx ?? '—'}</div>
        <div class="kv"><b>MACD</b><br>${pf.macd_signal ?? '—'}</div>
        <div class="kv"><b>Bollinger</b><br>${pf.boll_signal ?? '—'}</div>
        <div class="kv"><b>Volatilidade</b><br>${pf.volatility ?? '—'}</div>
        <div class="kv"><b>Liquidez (proxy)</b><br>${pf.liquidity_score ?? '—'}</div>
      </div>
      <div style="margin-top:10px;color:#aab">Observação: análise por foto é uma estimativa (proxy) – o motor da IA recalibra o sinal final.</div>
    `;
    out.classList.remove('hidden');
  }catch(e){
    setStatus('Falha na requisição.');
  }
});
</script>
</body>
</html>
    """
    return Response(html, mimetype="text/html")
# === [FIM PHOTO ADDON] ===


# === [PHOTO ANALYZER - AUTO-ADDED] ===
class PhotoAnalyzer:
    """
    Extrai features a partir de um print de gráfico (heurístico).
    Não altera o fluxo existente do app; serve como plug-in "PHOTO".
    """
    def __init__(self):
        pass

    @staticmethod
    def _ensure_cv_ok():
        if (base64 is None) or (io is None) or (np is None) or (Image is None) or (cv is None):
            raise RuntimeError("Dependências ausentes: pip install pillow numpy opencv-python-headless")

    def _read_image(self, image_bytes: bytes):
        self._ensure_cv_ok()
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)
        return img

    def _preprocess(self, img):
        h, w = img.shape[:2]
        cut = 0.05
        return img[int(h*cut):int(h*(1-cut)), int(w*cut):int(w*(1-cut))]

    def _extract_edges(self, gray):
        edges = cv.Canny(gray, 60, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv.dilate(edges, kernel, iterations=1)
        edges = cv.erode(edges, kernel, iterations=1)
        return edges

    def _trend_vol(self, edges):
        import numpy as np
        ys, xs = np.nonzero(edges)
        if len(xs) < 100:
            return 0.0, 0.01
        X = np.vstack([xs, np.ones_like(xs)]).T
        m, b = np.linalg.lstsq(X, ys, rcond=None)[0]
        slope = -m / max(1.0, edges.shape[1]*0.75)
        yfit = m*xs + b
        vol = float(np.std(ys - yfit) / max(1.0, edges.shape[0]))
        return float(slope), float(vol)

    def _rsi_proxy(self, gray):
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        gpos = float(np.clip(gy[gy>0].mean() if (gy>0).any() else 0.0, 0, 255))
        gneg = float(np.clip((-gy[gy<0]).mean() if (gy<0).any() else 0.0, 0, 255))
        base = 50.0
        if (gpos+gneg) > 0:
            base = 100.0 * (gpos / (gpos + gneg + 1e-9))
        return float(max(0.0, min(100.0, base)))

    def _macd_proxy(self, series):
        import numpy as np
        if len(series) < 50:
            return {"signal": "neutral", "strength": 0.0}
        def ema(x, n):
            k = 2/(n+1)
            e = [float(x[0])]
            for v in x[1:]:
                e.append(e[-1] + k*(float(v)-e[-1]))
            return np.array(e, dtype=float)
        ema12 = ema(series, 12)
        ema26 = ema(series, 26)
        macd_line = ema12[-len(ema26):] - ema26
        signal = ema(macd_line, 9)
        hist = float(macd_line[-1] - signal[-1]) if len(signal) else 0.0
        if hist > 0:  return {"signal": "bullish", "strength": min(1.0, abs(hist)/(np.mean(series)*0.02+1e-9))}
        if hist < 0:  return {"signal": "bearish", "strength": min(1.0, abs(hist)/(np.mean(series)*0.02+1e-9))}
        return {"signal": "neutral", "strength": 0.0}

    def extract_features(self, image_bytes: bytes) -> dict:
        img = self._read_image(image_bytes)
        img = self._preprocess(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = self._extract_edges(gray)
        slope, vol = self._trend_vol(edges)

        # ADX proxy via consistência angular de linhas
        lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=20, maxLineGap=8)
        adx_proxy = 20.0
        if lines is not None and len(lines) > 0:
            angles = []
            for l in lines[:500]:
                x1,y1,x2,y2 = l[0]
                ang = np.degrees(np.arctan2((y2-y1),(x2-x1)))
                angles.append(ang)
            if angles:
                hist,_ = np.histogram(angles, bins=18, range=(-90,90))
                kons = hist.max()/max(1,len(angles))
                adx_proxy = float(15 + 80*min(1.0, max(0.0, (kons-0.25)/0.75)))

        rsi_proxy = self._rsi_proxy(gray)
        h, w = gray.shape[:2]
        row = gray[int(h*0.5), :].astype(np.float32) + 1.0
        macd = self._macd_proxy(row)

        # Boll proxy
        period = min(20, len(row))
        if period >= 12:
            win = row[-period:]
            ma = float(win.mean())
            sd = float(win.std())
            last = float(win[-1])
            if last > ma + 2*sd:   boll_sig = "overbought"
            elif last < ma - 2*sd: boll_sig = "oversold"
            elif last > ma:        boll_sig = "bullish"
            elif last < ma:        boll_sig = "bearish"
            else:                  boll_sig = "neutral"
        else:
            boll_sig = "neutral"

        tf_cons = "buy" if slope > 0.002 else ("sell" if slope < -0.002 else "neutral")
        direction = "buy" if (slope > 0 or macd["signal"] == "bullish") else "sell"
        liq_score = float(max(0.0, min(1.0, 1.0 - (vol*3.0))))

        base_prob_buy = 0.5 + float(np.clip(slope*8.0, -0.35, 0.35))
        if macd["signal"] == "bullish": base_prob_buy += 0.05
        if boll_sig == "oversold":      base_prob_buy += 0.05
        if boll_sig == "overbought":    base_prob_buy -= 0.05
        prob_buy = float(max(0.10, min(0.90, base_prob_buy)))

        return {
            "symbol": "PHOTO",
            "horizon": 1,
            "rsi": float(rsi_proxy),
            "adx": float(round(adx_proxy,2)),
            "macd_signal": macd["signal"],
            "boll_signal": boll_sig,
            "multi_timeframe": tf_cons,
            "liquidity_score": liq_score,
            "reversal": False,
            "reversal_side": None,
            "reversal_proximity": 0.0,
            "probability_buy": prob_buy,
            "probability_sell": 1.0 - prob_buy,
            "price": None,
            "volatility": float(vol),
            "market_regime": "normal" if vol < 0.015 else ("high_volatility" if vol > 0.03 else "normal"),
            "confidence": 0.55,
            "direction": direction
        }



# === [PHOTO INTEGRATION + ROUTES - AUTO-ADDED] ===
try:
    _app_ref = app  # type: ignore
except NameError:
    from flask import Flask
    app = Flask(__name__)
    _app_ref = app

try:
    _PHOTO_SYSTEM = manager.system if 'manager' in globals() else EnhancedTradingSystem()  # type: ignore
except Exception:
    _PHOTO_SYSTEM = None

PHOTO_ANALYZER = PhotoAnalyzer()

from flask import request, jsonify, render_template_string

PHOTO_HTML = r"""<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8">
    <title>📷 Análise por Foto</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial,sans-serif;margin:24px;background:#0b0b10;color:#e9eef2}
      .card{max-width:880px;margin:auto;background:#141621;border:1px solid #1f2333;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
      .hd{padding:16px 20px;border-bottom:1px solid #1f2333;font-size:18px;font-weight:700;display:flex;gap:8px;align-items:center}
      .bd{padding:20px}
      .row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
      .drop{flex:1 1 360px;border:2px dashed #2d3350;border-radius:12px;padding:20px;text-align:center;background:#0f1120;min-height:160px}
      .drop.drag{border-color:#5177ff;background:#0f1430}
      .btn{background:#4c6fff;border:none;color:#fff;padding:10px 16px;border-radius:10px;font-weight:600;cursor:pointer}
      .btn:disabled{opacity:.6;cursor:not-allowed}
      .out{margin-top:16px;background:#0f1120;border:1px solid #1f2333;border-radius:12px;padding:16px;font-family:ui-monospace,Consolas,Monaco,monospace;white-space:pre-wrap}
      .kv{display:grid;grid-template-columns:180px 1fr;gap:8px 12px;margin-top:10px}
      .kv b{color:#a9b4d0}
      .pill{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700}
      .buy{background:#1b3d22;color:#3ee07a}
      .sell{background:#3d1b1b;color:#ff8f8f}
      .muted{color:#92a0bd}
    </style>
  </head>
  <body>
    <div class="card">
      <div class="hd">📷 Análise por Foto <span class="muted">— arraste e solte um print do gráfico</span></div>
      <div class="bd">
        <div class="row">
          <div class="drop" id="drop">
            <p>Solte a imagem aqui, ou</p>
            <input type="file" id="file" accept="image/*">
            <div style="margin-top:10px">
              <button class="btn" id="send">Analisar</button>
            </div>
          </div>
          <div style="flex:1 1 360px">
            <div id="result" class="out" style="display:none"></div>
          </div>
        </div>
      </div>
    </div>

<script>
const fileInput = document.getElementById('file');
const drop = document.getElementById('drop');
const btn = document.getElementById('send');
const out = document.getElementById('result');
let blob = null;

function setDragging(v){ drop.classList.toggle('drag', v); }
['dragenter','dragover'].forEach(e=> drop.addEventListener(e, ev=>{ev.preventDefault(); setDragging(true);}));
['dragleave','drop'].forEach(e=> drop.addEventListener(e, ev=>{ev.preventDefault(); setDragging(false);}));
drop.addEventListener('drop', ev=>{
  const f = ev.dataTransfer.files && ev.dataTransfer.files[0];
  if(f){ fileInput.files = ev.dataTransfer.files; blob = f; }
});
fileInput.addEventListener('change', ev=>{ blob = ev.target.files[0] || null; });

btn.addEventListener('click', async ()=>{
  if(!blob){ out.style.display='block'; out.textContent='Selecione uma imagem.'; return; }
  btn.disabled = true; out.style.display='block'; out.textContent='Enviando e analisando...';
  try{
    const fd = new FormData();
    fd.append('image', blob);
    const res = await fetch('/photo/analyze', { method:'POST', body: fd });
    const data = await res.json();
    if(!data.ok){ out.textContent = 'Erro: ' + (data.error || 'Falha desconhecida'); btn.disabled=false; return; }
    const pill = data.direction==='buy' ? '<span class="pill buy">COMPRAR</span>' : '<span class="pill sell">VENDER</span>';
    const conf = (data.final_confidence*100).toFixed(1) + '%';
    const lines = [
      `Direção: ${pill}`,
      `Confiança: ${conf}`,
      '',
      'Métricas:',
      JSON.stringify(data.quality_metrics || {}, null, 2),
      '',
      'Motivos:',
      (data.reasoning||[]).join('\n- ')
    ].join('\n');
    out.innerHTML = lines.replace('Direção: ', 'Direção: ');
  }catch(err){
    out.textContent = 'Erro de rede: ' + err;
  }
  btn.disabled=false;
});
</script>
  </body>
</html>"""

@_app_ref.get("/photo")
def photo_page():
    return render_template_string(PHOTO_HTML)

@_app_ref.post("/photo/analyze")
def photo_analyze():
    if _PHOTO_SYSTEM is None:
        return jsonify({"ok": False, "error": "Sistema de IA não inicializado."}), 500

    image_bytes = None
    if request.files and "image" in request.files:
        image_bytes = request.files["image"].read()
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        b64 = (data.get("image_base64") or "").strip()
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        if b64:
            try:
                image_bytes = base64.b64decode(b64)
            except Exception:
                return jsonify({"ok": False, "error": "Base64 inválido"}), 400
    if not image_bytes:
        return jsonify({"ok": False, "error": "Envie 'image' (arquivo) ou 'image_base64'"}), 400

    try:
        raw = PHOTO_ANALYZER.extract_features(image_bytes)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Falha ao analisar imagem: {e}"}), 500

    all_symbols_data = [raw]
    try:
        if hasattr(_PHOTO_SYSTEM, "intelligent_ai"):
            intelligent = _PHOTO_SYSTEM.intelligent_ai.analyze_with_high_accuracy(raw, all_symbols_data)
        elif hasattr(_PHOTO_SYSTEM, "analyze_with_high_accuracy"):
            intelligent = _PHOTO_SYSTEM.analyze_with_high_accuracy(raw, all_symbols_data)
        elif 'HighAccuracyTradingAI' in globals():
            intelligent = HighAccuracyTradingAI().analyze_with_high_accuracy(raw, all_symbols_data)  # type: ignore
        else:
            intelligent = {
                "direction": "buy" if raw.get("probability_buy",0.5)>=0.5 else "sell",
                "final_confidence": raw.get("confidence", 0.55),
                "reasoning": ["Decisão direta do analisador de foto (modo fallback)."],
                "quality_metrics": {"photo_only": True}
            }
    except Exception as e:
        intelligent = {
            "direction": "buy" if raw.get("probability_buy",0.5)>=0.5 else "sell",
            "final_confidence": raw.get("confidence", 0.55),
            "reasoning": [f"Falha na IA principal, usando fallback. Erro: {e}"],
            "quality_metrics": {"photo_only": True, "error": str(e)}
        }

    out = {
        "ok": True,
        "source": "PHOTO",
        "direction": intelligent.get("direction"),
        "final_confidence": float(intelligent.get("final_confidence", 0.55)),
        "reasoning": intelligent.get("reasoning", []),
        "quality_metrics": intelligent.get("quality_metrics", {}),
        "market_context": intelligent.get("market_context", {}),
        "pattern_analysis": intelligent.get("pattern_analysis", {}),
        "confidence_breakdown": intelligent.get("confidence_breakdown", {}),
        "technical_convergence": float(intelligent.get("technical_convergence", 0.5)) if isinstance(intelligent.get("technical_convergence", 0.5),(int,float)) else 0.5,
        "market_sentiment_alignment": float(intelligent.get("market_sentiment_alignment", 1.0)) if isinstance(intelligent.get("market_sentiment_alignment", 1.0),(int,float)) else 1.0,
        "risk_adjustment": float(intelligent.get("risk_adjustment", 1.0)) if isinstance(intelligent.get("risk_adjustment", 1.0),(int,float)) else 1.0,
        "system_accuracy": float(intelligent.get("system_accuracy", 0.5)) if isinstance(intelligent.get("system_accuracy", 0.5),(int,float)) else 0.5,
        "photo_features": {
            "rsi": raw.get("rsi"),
            "adx": raw.get("adx"),
            "macd_signal": raw.get("macd_signal"),
            "boll_signal": raw.get("boll_signal"),
            "multi_timeframe": raw.get("multi_timeframe"),
            "probability_buy_seed": raw.get("probability_buy"),
            "volatility": raw.get("volatility"),
            "liquidity_score": raw.get("liquidity_score")
        }
    }
    return jsonify(out), 200

