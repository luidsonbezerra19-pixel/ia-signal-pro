# app.py — IA Signal Pro COM INTELIGÊNCIA AVANÇADA
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
# NOVO: Sistema de Inteligência Avançada
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
            self.pattern_success[pattern_key] = 0.5  # Inicializa neutro
            
        # Ajusta baseado no resultado
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
        
        score_adjustment = (pattern_effectiveness - 0.5) * 0.3  # ±15%
        
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
# Feature Flags Atualizadas
# =========================
FEATURE_FLAGS = {
    "enable_adaptive_garch": True,
    "enable_smart_cache": True,
    "enable_circuit_breaker": True,
    "websocket_provider": "okx",
    "maintenance_mode": False,
    "enable_ai_intelligence": True,  # NOVO
    "enable_learning": True,
    "enable_self_check": True
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
    return (x["confidence"] * 1000) + (prob_directional * 100)

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
    total_score *= (0.90 + (liquidity_score * 0.15))
    
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
        return max(5.0, min(65.0, adx))

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
# NOVO: Enhanced Trading System com IA
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
        
        # NOVO: Sistema de IA Inteligente
        self.intelligent_ai = IntelligentTradingAI()

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
        rsi_series = self.indicators.rsi_series_wilder(closes, 14)
        rsi = rsi_series[-1] if rsi_series else 50.0
        adx = self.indicators.adx_wilder(highs, lows, closes)
        macd = self.indicators.macd(closes)
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
            macd_result = self.indicators.macd(closes)
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

        # NOVO: Análise bruta para IA
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
            'market_regime': mc.get('market_regime', 'normal')
        }

        # NOVO: Aplicar inteligência artificial se habilitada
        if FEATURE_FLAGS["enable_ai_intelligence"]:
            intelligent_result = self.intelligent_ai.analyze_with_intelligence(raw_analysis)
            
            # Sobrescreve direção e confiança com decisão inteligente
            direction = intelligent_result['direction']
            confidence = intelligent_result['intelligent_confidence']
            
            # NOVO: Auto-crítica
            if FEATURE_FLAGS["enable_self_check"]:
                self_check = self.intelligent_ai.intelligent_self_check(intelligent_result)
                intelligent_result['self_check'] = self_check
            
            # NOVO: Análise de incerteza
            uncertainty = self.intelligent_ai.explain_uncertainty(intelligent_result)
            intelligent_result['uncertainty'] = uncertainty
            
            # Adiciona resultados da IA ao analysis final
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
        }

        # NOVO: Adiciona dados de inteligência se habilitada
        if FEATURE_FLAGS["enable_ai_intelligence"]:
            result.update({
                'intelligent_confidence': raw_analysis.get('intelligent_confidence', confidence),
                'reasoning': raw_analysis.get('reasoning', []),
                'pattern_effectiveness': raw_analysis.get('pattern_effectiveness', 0.5),
                'system_accuracy': raw_analysis.get('system_accuracy', 0.5),
                'market_regime': raw_analysis.get('market_regime', 'normal'),
                'synthesis_score': raw_analysis.get('synthesis_score', 0),
                'self_check': raw_analysis.get('self_check', {}),
                'uncertainty': raw_analysis.get('uncertainty', {}),
                'reasoning_depth': 'multilayer_intelligence'
            })

        logger.info("analysis_completed", 
                   symbol=symbol, 
                   horizon=horizon, 
                   duration_ms=analysis_duration,
                   direction=direction,
                   confidence=confidence,
                   intelligent_ai=FEATURE_FLAGS["enable_ai_intelligence"])

        return result

    def scan_symbols_tplus(self, symbols: List[str])->Dict[str,Any]:
        por_ativo={}; candidatos=[]
        for sym in symbols:
            tplus=[]
            # APENAS T+1 - Modificação aqui
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
# NOVO: Endpoints para IA
# =========================
@app.post("/api/ai/learn")
def api_ai_learn():
    """Endpoint para aprendizado da IA com resultados reais"""
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
                
        if recent_signal:
            manager.system.intelligent_ai.learn_from_result(
                recent_signal, actual_price_movement, expected_direction
            )
            return jsonify({
                "success": True,
                "message": "IA aprendeu com resultado",
                "system_accuracy": manager.system.intelligent_ai.adaptation.get_overall_accuracy()
            })
        else:
            return jsonify({"success": False, "error": "Sinal recente não encontrado"}), 404
            
    except Exception as e:
        logger.error("ai_learning_error", error=str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/api/ai/status")
def api_ai_status():
    """Status da IA inteligente"""
    ai = manager.system.intelligent_ai
    return jsonify({
        "success": True,
        "ai_enabled": FEATURE_FLAGS["enable_ai_intelligence"],
        "learning_enabled": FEATURE_FLAGS["enable_learning"],
        "system_accuracy": ai.adaptation.get_overall_accuracy(),
        "patterns_learned": len(ai.memory.pattern_success),
        "recent_accuracy": ai.memory.get_recent_accuracy(),
        "confidence_calibration": ai.adaptation.confidence_calibration
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
            "symbols_count": len(symbols)
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
        "ai_intelligence": FEATURE_FLAGS["enable_ai_intelligence"],
        "ai_accuracy": manager.system.intelligent_ai.adaptation.get_overall_accuracy() if FEATURE_FLAGS["enable_ai_intelligence"] else None
    }
    return jsonify(health_status), 200

@app.get("/deep-health")
def deep_health():
    ws_status = "connected" if WS_FEED._ws and WS_FEED._ws.sock else "disconnected"
    
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
            "ai_intelligence": {
                "enabled": FEATURE_FLAGS["enable_ai_intelligence"],
                "patterns_learned": len(manager.system.intelligent_ai.memory.pattern_success),
                "system_accuracy": manager.system.intelligent_ai.adaptation.get_overall_accuracy(),
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
<title>IA Signal Pro - PREÇOS REAIS + SIMULAÇÕES + IA INTELIGENTE</title>
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
</style>
</head>
<body>
<div class="wrap">
  <div class="hline">
    <h1>IA Signal Pro - PREÇOS REAIS + 3000 SIMULAÇÕES + IA INTELIGENTE</h1>
    <div class="clock" id="clock">--:--:-- BRT</div>
    <div class="sub">✅ CoinAPI (WS) · Binance REST fallback · RSI/ADX (Wilder) · Liquidez (ATR%) · Reversão RSI (12h) · GARCH(1,1) Adaptativo · 🧠 IA MULTICAMADAS INTELIGENTE</div>
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
        <button id="go" onclick="runAnalyze()">🧠 Analisar com IA Inteligente</button>
      </div>
    </div>
  </div>

  <div class="section" id="bestSec" style="display:none">
    <div class="title">🥇 MELHOR OPORTUNIDADE GLOBAL (IA INTELIGENTE)</div>
    <div class="card" id="bestCard"></div>
  </div>

  <div class="section" id="allSec" style="display:none">
    <div class="title">📊 SINAIS T+1 POR ATIVO</div>
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

async function runAnalyze(){
  const btn = document.getElementById('go');
  btn.disabled = true;
  btn.textContent = '⏳ IA Analisando...';
  const syms = selSymbols();
  if(!syms.length){ alert('Selecione pelo menos um ativo.'); btn.disabled=false; btn.textContent='🧠 Analisar com IA Inteligente'; return; }
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
      btn.textContent = '🧠 Analisar com IA Inteligente';
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
          <span class="tag ai-badge">🧠 IA Inteligente</span>
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
  // Prefere confiança inteligente se disponível
  const confidence = it.intelligent_confidence || it.confidence;
  return (confidence * 1000) + (prob_directional * 100);
}

function renderBest(best, analysisTime){
  if(!best) return '<div class="small">Sem oportunidade no momento.</div>';
  const rev = best.reversal ? ` <span class="tag">🔄 Reversão (${best.reversal_side})</span>` : '';
  const confidence = best.intelligent_confidence || best.confidence;
  const reasoning = best.reasoning ? `<div class="small" style="margin-top:8px;color:#8ccf9d">🧠 ${best.reasoning.slice(0,3).join(' · ')}</div>` : '';
  
  return `
    <div class="small muted">Atualizado: ${analysisTime} (Horário Brasil) · IA Inteligente Ativa</div>
    <div class="line"></div>
    <div><b>${best.symbol} T+${best.horizon}</b> ${badgeDir(best.direction)} <span class="tag">🥇 MELHOR ENTRE TODOS OS ATIVOS</span>${rev} <span class="tag ai-badge">🧠 IA INTELIGENTE</span></div>
    <div class="kpis">
      <div class="kpi"><b>Prob Compra</b>${pct(best.probability_buy||0)}</div>
      <div class="kpi"><b>Prob Venda</b>${pct(best.probability_sell||0)}</div>
      <div class="kpi"><b>Confiança IA</b>${pct(confidence)}</div>
      <div class="kpi"><b>ADX</b>${(best.adx||0).toFixed(1)}</div>
      <div class="kpi"><b>RSI</b>${(best.rsi||0).toFixed(1)}</div>
      <div class="kpi"><b>Liquidez</b>${Number(best.liquidity_score||0).toFixed(2)}</div>
    </div>
    ${reasoning}
    <div class="small" style="margin-top:8px;">
      Pontuação IA: <span class="ok">${(confidence*100).toFixed(1)}%</span> · TF: <b>${best.multi_timeframe||'neutral'}</b> · Price: <b>${Number(best.price||0).toFixed(6)}</b>
      <span class="right">Entrada: <b>${best.entry_time||'-'}</b></span>
    </div>`;
}

function renderTbox(it){
  if(!it) return '<div class="tbox">Erro ao carregar sinal</div>';
  
  const rev = it.reversal ? ` <span class="tag">🔄 REVERSÃO (${it.reversal_side})</span>` : '';
  const confidence = it.intelligent_confidence || it.confidence;
  const reasoning = it.reasoning ? `<div class="small" style="color:#8ccf9d;margin-top:4px">🧠 ${it.reasoning.slice(0,2).join(' · ')}</div>` : '';
  
  return `
    <div class="tbox">
      <div><b>T+${it.horizon}</b> ${badgeDir(it.direction)}${rev} <span class="tag ai-badge">🧠 IA</span></div>
      <div class="small">
        Prob: <span class="${it.direction==='buy'?'ok':'err'}">${pct(it.probability_buy||0)}/${pct(it.probability_sell||0)}</span>
        · Conf IA: <span class="ok">${pct(confidence)}</span>
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
    logger.info("ai_intelligence_enabled", 
                enabled=FEATURE_FLAGS["enable_ai_intelligence"],
                learning=FEATURE_FLAGS["enable_learning"])
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
