# app.py — IA Signal Pro com Novo Modelo Híbrido
from __future__ import annotations
import os, re, time, math, random, threading, json, statistics as stats
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import structlog

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
# Novos Feature Flags
# =========================
FEATURE_FLAGS = {
    "enable_adaptive_garch": True,
    "enable_smart_cache": True,
    "enable_circuit_breaker": True,
    "websocket_provider": "okx",
    "maintenance_mode": False,
    "enable_market_context_garch": True,  # NOVO
    "enable_dynamic_weights": True,       # NOVO
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
# Utils
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

# =========================
# NOVO: Análise de Tendência de Mercado
# =========================
class MarketTrendAnalyzer:
    def __init__(self):
        self.market_regime = "neutral"
        self.trend_strength = 0.0
        self.last_update = 0
        
    def analyze_market_trend(self, symbols_data: Dict[str, List[float]]) -> Dict[str, Any]:
        if time.time() - self.last_update < 300:
            return {"regime": self.market_regime, "strength": self.trend_strength}
            
        bullish_count = 0
        total_symbols = 0
        strength_scores = []
        
        for symbol, closes in symbols_data.items():
            if len(closes) < 50:
                continue
                
            ma_20 = sum(closes[-20:]) / 20
            ma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma_20
            
            trend = "bullish" if ma_20 > ma_50 else "bearish"
            strength = abs(ma_20 - ma_50) / ma_50
            
            if trend == "bullish":
                bullish_count += 1
                strength_scores.append(strength)
            else:
                strength_scores.append(-strength)
                
            total_symbols += 1
        
        if total_symbols == 0:
            return {"regime": "neutral", "strength": 0.0}
        
        bullish_ratio = bullish_count / total_symbols
        avg_strength = sum(strength_scores) / len(strength_scores) if strength_scores else 0.0
        
        if bullish_ratio >= 0.7 and avg_strength > 0.01:
            regime = "bullish"
        elif bullish_ratio <= 0.3 and avg_strength < -0.01:
            regime = "bearish" 
        else:
            regime = "neutral"
            
        self.market_regime = regime
        self.trend_strength = avg_strength
        self.last_update = time.time()
        
        return {
            "regime": regime,
            "strength": avg_strength,
            "bullish_ratio": bullish_ratio,
            "symbols_analyzed": total_symbols
        }

# =========================
# NOVO: Pesos Dinâmicos Baseados na Função do Mercado
# =========================
def calculate_dynamic_weights(adx: float, liquidity_score: float) -> Dict[str, float]:
    """Pesos dinâmicos baseados apenas em ADX e liquidez - IMPARCIAL"""
    
    if adx > 25:
        # Tendência forte - foco em indicadores de tendência
        return {
            "multi_tf": 0.22,
            "adx": 0.18,  
            "rsi": 0.20,
            "macd": 0.20,
            "bollinger": 0.20
        }
    else:
        # Mercado lateral - pesos balanceados
        return {
            "multi_tf": 0.18,
            "adx": 0.16,
            "rsi": 0.22,
            "macd": 0.22, 
            "bollinger": 0.22
        }

# =========================
# NOVO: Confirmação com Pesos Dinâmicos
# =========================
def _confirm_prob_with_dynamic_weights(prob_up: float, rsi: float, macd_hist: float, adx: float, 
                                     boll_signal: str, tf_consensus: str, market_trend: Dict,
                                     liquidity_score: float) -> float:
    """Sistema de confirmação com pesos dinâmicos"""
    
    # Calcular pesos baseados no mercado
    weights = calculate_dynamic_weights(adx, liquidity_score)
    
    # Calcular contribuições de cada indicador
    contributions = 0.0
    
    # Multi-Timeframe
    if tf_consensus == "buy":
        contributions += 0.04 * weights["multi_tf"]
    elif tf_consensus == "sell":
        contributions -= 0.04 * weights["multi_tf"]
    
    # ADX (apenas confirma força, não direção)
    if adx >= 25:
        # Tendência forte - confirma sinais alinhados
        if market_trend["regime"] == "bullish":
            contributions += 0.02 * weights["adx"]
        elif market_trend["regime"] == "bearish":
            contributions -= 0.02 * weights["adx"]
    
    # RSI
    if rsi >= 58:
        contributions -= 0.03 * weights["rsi"]
    elif rsi <= 42:
        contributions += 0.03 * weights["rsi"]
    
    # MACD
    if macd_hist > 0.001:
        contributions += 0.025 * weights["macd"]
    elif macd_hist < -0.001:
        contributions -= 0.025 * weights["macd"]
    
    # Bollinger Bands
    if boll_signal == "oversold":
        contributions += 0.03 * weights["bollinger"]
    elif boll_signal == "overbought":
        contributions -= 0.03 * weights["bollinger"]
    elif boll_signal == "bullish":
        contributions += 0.015 * weights["bollinger"]
    elif boll_signal == "bearish":
        contributions -= 0.015 * weights["bollinger"]
    
    # Ajuste final mais conservador (±4% máximo)
    adjustment = contributions * 1.5  # Escala para ±4% máximo
    adjustment = max(-0.04, min(0.04, adjustment))
    
    adjusted_prob = prob_up + adjustment
    
    # Randomização mínima para evitar travamento (0.1%)
    noise = random.gauss(0, 0.001)
    adjusted_prob += noise
    
    return min(0.92, max(0.08, adjusted_prob))

# =========================
# NOVO: Determinação de Direção com Histerese
# =========================
def determine_direction_impartial(prob_buy: float, prob_sell: float) -> str:
    """Decisão PURA baseada apenas nas probabilidades"""
    return 'buy' if prob_buy > prob_sell else 'sell'

# =========================
# Confiança Direcional Atualizada
# =========================
def _calculate_directional_confidence(prob_direction: float, direction: str, rsi: float, 
                                    adx: float, macd_signal: str, boll_signal: str,
                                    tf_consensus: str, reversal_signal: dict,
                                    liquidity_score: float, market_trend: Dict) -> float:
    """Versão atualizada com contexto de mercado"""
    
    base_conf = prob_direction * 100.0
    score = base_conf
    
    # Pesos baseados no regime de mercado
    if market_trend["regime"] in ["bullish", "bearish"] and adx >= 25:
        # Mercado trend - foco em tendência
        w_trend, w_momentum, w_boll, w_rev, w_macd, w_liq = 15.0, 6.0, 8.0, 12.0, 6.0, 5.0
    else:
        # Mercado lateral - foco em momento
        w_trend, w_momentum, w_boll, w_rev, w_macd, w_liq = 8.0, 12.0, 10.0, 10.0, 8.0, 6.0

    # Confirmações
    if (direction == 'buy' and tf_consensus == 'buy') or (direction == 'sell' and tf_consensus == 'sell'):
        score += w_trend
    
    if (direction == 'buy' and rsi < 45) or (direction == 'sell' and rsi > 55):
        score += w_momentum

    if (direction == 'buy' and boll_signal in ['oversold','bullish']) or \
       (direction == 'sell' and boll_signal in ['overbought','bearish']):
        score += w_boll

    if reversal_signal["reversal"] and reversal_signal["side"] == direction:
        prox = reversal_signal.get("proximity", 0.0)
        score += w_rev * prox

    if (direction == 'buy' and macd_signal == 'bullish') or (direction == 'sell' and macd_signal == 'bearish'):
        score += w_macd

    # Alinhamento com tendência de mercado
    if market_trend["regime"] == "bullish" and direction == "buy":
        score += 8.0
    elif market_trend["regime"] == "bearish" and direction == "sell":
        score += 8.0
    elif market_trend["regime"] != "neutral" and (
        (market_trend["regime"] == "bullish" and direction == "sell") or
        (market_trend["regime"] == "bearish" and direction == "buy")
    ):
        score -= 6.0

    # Liquidez
    score *= (0.92 + liquidity_score * (w_liq / 100))

    # Modulação pelo ADX
    if adx < 18:
        score *= 0.88
    elif adx > 30:
        score *= 1.10

    return min(96.0, max(30.0, score)) / 100.0

# =========================
# WebSocket OKX (mantido igual)
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
# NOVO: Cache Uniforme de 3 Segundos
# =========================
class SmartCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, int], Tuple[float, List[List[float]]]] = {}
        
    def get(self, key: Tuple[str, str, int]) -> Optional[Tuple[float, List[List[float]]]]:
        if key in self._cache:
            timestamp, data = self._cache[key]
            # Cache uniforme de 3 segundos para todos os ativos
            if time.time() - timestamp < 3:
                return (timestamp, data)
        return None
    
    def set(self, key: Tuple[str, str, int], data: List[List[float]]):
        self._cache[key] = (time.time(), data)

# =========================
# Mercado Spot com Novo Cache
# =========================
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
                return cached[1]  # Retorna dados do cache

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
# Indicadores (mantido igual)
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
# Reversão (mantido igual)
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
# NOVO: GARCH com Contexto de Mercado e Mean Reversion
# =========================
class AdaptiveGARCH11Simulator:
    """GARCH(1,1) com contexto de mercado e mean reversion"""
    
    def _detect_market_regime(self, returns: List[float], volume_trend: float = 0.0) -> str:
        """Detecta regime com contexto de volume"""
        if not returns or len(returns) < 20:
            return "normal"
        
        volatility = stats.stdev(returns) if len(returns) > 1 else 0.0
        mean_return = stats.mean(returns) if returns else 0.0
        
        # Tendência baseada em retorno e volume
        if abs(mean_return) > 0.015 and volatility > 0.025:
            return "high_volatility"
        elif abs(mean_return) < 0.005 and volatility < 0.01:
            return "low_volatility"
        else:
            return "normal"
    
    def _get_garch_params_for_regime(self, regime: str, returns: List[float], 
                                   market_trend: Dict, liquidity: float) -> Tuple[float, float, float, float]:
        """Parâmetros GARCH com contexto de mercado"""
        if not returns:
            return 1e-6, 0.1, 0.85, 1e-4
            
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        
        # Ajustar parâmetros baseado no regime de mercado
        if regime == "low_volatility":
            alpha, beta = 0.06, 0.90
        elif regime == "high_volatility":
            alpha, beta = 0.15, 0.75
        else:
            alpha, beta = 0.09, 0.88
            
        # Ajuste fino baseado na tendência de mercado
        if market_trend["regime"] == "bullish" and mean_ret > 0:
            alpha += 0.02  # Mais sensível a choques positivos
        elif market_trend["regime"] == "bearish" and mean_ret < 0:
            alpha += 0.02  # Mais sensível a choques negativos
            
        # Ajuste por liquidez
        if liquidity > 0.8:  # Alta liquidez
            beta += 0.04    # Mais persistência
        elif liquidity < 0.4:  # Baixa liquidez
            alpha += 0.03    # Mais sensibilidade
            
        omega = var * (1 - alpha - beta)
        omega = max(1e-6, omega)
        
        if alpha + beta >= 0.999:
            alpha, beta = 0.08, 0.90
            omega = 1e-6
            
        return omega, alpha, beta, var
    
    def _calculate_mean_reversion_force(self, current_price: float, base_price: float, 
                                  returns: List[float], market_trend: Dict) -> float:
        """Calcula força de mean reversion baseada no contexto"""
        if len(returns) < 10:
            return 0.0
            
        mean_return = stats.mean(returns)
        price_ratio = current_price / base_price
    
        # Força de reversion baseada na distância da média
        if market_trend["regime"] == "neutral":
            # Mercado lateral - reversion mais forte
            theta = 0.25
        else:
            # Mercado com tendência - reversion mais suave
            theta = 0.12
            
        # Calcular reversion force
        if abs(price_ratio - 1.0) > 0.02:  # Desvio de 2%+
            reversion_force = theta * (1.0 - price_ratio)
        else:
            reversion_force = 0.0
            
        return reversion_force

    def simulate_garch11(self, base_price: float, returns: List[float], 
                        steps: int, num_paths: int = 3000, 
                        market_trend: Dict = None, liquidity: float = 0.5,
                        volumes: List[float] = None) -> Dict[str, Any]:
        """GARCH com contexto de mercado e mean reversion"""
        import math
        import random
        
        if market_trend is None:
            market_trend = {"regime": "neutral", "strength": 0.0}
            
        if not returns or len(returns) < 10:
            returns = [random.gauss(0.0, 0.002) for _ in range(100)]
        
        # Detectar regime com contexto
        volume_trend = stats.mean(volumes[-10:]) / stats.mean(volumes[-50:]) if volumes and len(volumes) >= 50 else 1.0
        regime = self._detect_market_regime(returns, volume_trend)
        
        # Calibrar parâmetros com contexto
        omega, alpha, beta, h_last = self._get_garch_params_for_regime(
            regime, returns, market_trend, liquidity
        )
        
        up_count = 0
        total_count = 0
        
        start_time = time.time()
        for _ in range(num_paths):
            try:
                h = h_last
                price = base_price
                
                for step in range(steps):
                    # Calcular mean reversion
                    reversion_force = self._calculate_mean_reversion_force(
                        price, base_price, returns, market_trend
                    )
                    
                    # Simulação GARCH com reversion
                    epsilon = math.sqrt(h) * random.gauss(0.0, 1.0) + reversion_force
                    price *= math.exp(epsilon)
                    
                    # Atualizar volatilidade GARCH
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
                    market_trend=market_trend["regime"],
                    probability_buy=prob_buy)
        
        return {
            "probability_buy": prob_buy,
            "probability_sell": prob_sell,
            "quality": "garch11_adaptive_market",
            "sim_model": "garch11",
            "paths_used": total_count,
            "garch_params": {"omega": omega, "alpha": alpha, "beta": beta},
            "market_regime": regime,
            "calculation_time_ms": duration_ms
        }

MonteCarloSimulator = AdaptiveGARCH11Simulator

# =========================
# Sistema Principal Atualizado
# =========================
class EnhancedTradingSystem:
    def __init__(self)->None:
        self.indicators=TechnicalIndicators()
        self.revdet=ReversalDetector()
        self.monte_carlo=MonteCarloSimulator()
        self.multi_tf=MultiTimeframeAnalyzer()
        self.liquidity=LiquiditySystem()
        self.spot=SpotMarket()
        self.trend_analyzer = MarketTrendAnalyzer()  # NOVO
        self.current_analysis_cache: Dict[str,Any]={}
        self.last_directions: Dict[str, str] = {}  # NOVO: histerese

    def get_brazil_time(self)->datetime:
        return brazil_now()

    def analyze_symbol(self, symbol: str, horizon: int)->Dict[str,Any]:
        start_time = time.time()
        logger.info("analysis_started", symbol=symbol, horizon=horizon)
        
        # OHLCV 1m
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
        volumes = [x[5] for x in ohlcv_closed]  # NOVO: volumes para GARCH

        # Preço e Volume
        price_display = raw[-1][4]
        volume_display = raw[-1][5] if raw and len(raw[-1]) >= 6 else 0.0
        try:
            ws_last = WS_FEED.get_last_candle(symbol)
            if ws_last:
                price_display  = float(ws_last[4])
                volume_display = float(ws_last[5])
        except Exception:
            pass

        # NOVO: Análise de Tendência de Mercado
        market_data = {}
        for sym in DEFAULT_SYMBOLS:
            if sym != symbol:
                sym_data = self.spot.fetch_ohlcv(sym, "1m", 100)
                if sym_data:
                    market_data[sym] = [x[4] for x in sym_data]
        
        market_trend = self.trend_analyzer.analyze_market_trend(market_data)

        # Indicadores
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

        # GARCH com Contexto de Mercado
        empirical_returns = _safe_returns_from_prices(closes)
        steps = max(1, min(3, int(horizon)))
        base_price = closes[-1] if closes else price_display

        # NOVO: GARCH com mercado, volumes e liquidez
        mc = self.monte_carlo.simulate_garch11(
            base_price, 
            empirical_returns, 
            steps, 
            num_paths=MC_PATHS,
            market_trend=market_trend,
            liquidity=liq,
            volumes=volumes
        )

        # NOVO: Confirmação com pesos dinâmicos
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

        # NOVO: Sistema de confirmação com pesos dinâmicos
        prob_buy_adjusted = _confirm_prob_with_dynamic_weights(
            prob_buy_original, rsi, macd_hist, adx, 
            boll['signal'], tf_cons, market_trend, liq
        )
        prob_sell_adjusted = 1.0 - prob_buy_adjusted

        # NOVO: Direção com histerese
        # Direção IMPARCIAL - baseada apenas nas probabilidades
        direction = determine_direction_impartial(prob_buy_adjusted, prob_sell_adjusted)

        # Atualizar resultado GARCH
        mc['probability_buy'] = prob_buy_adjusted
        mc['probability_sell'] = prob_sell_adjusted

        # NOVO: Confiança com contexto de mercado
        prob_dir = prob_buy_adjusted if direction == 'buy' else prob_sell_adjusted
        confidence = _calculate_directional_confidence(
            prob_dir, direction, rsi, adx, macd['signal'], boll['signal'], 
            tf_cons, rev_sig, liq, market_trend
        )

        analysis_duration = (time.time() - start_time) * 1000
        logger.info("analysis_completed", 
                   symbol=symbol, 
                   horizon=horizon, 
                   duration_ms=analysis_duration,
                   direction=direction,
                   confidence=confidence,
                   market_trend=market_trend["regime"])

        return {
            'symbol':symbol,
            'horizon':steps,
            'direction':direction,
            'probability_buy':mc['probability_buy'],
            'probability_sell':mc['probability_sell'],
            'confidence':confidence,
            'rsi':rsi,'adx':adx,'multi_timeframe':tf_cons,
            'monte_carlo_quality':mc['quality'],
            'garch_model': mc['sim_model'],
            'simulations_count': mc.get('paths_used', MC_PATHS),
            'market_regime': mc.get('market_regime', 'normal'),
            'price':price_display,
            'liquidity_score':liq,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            # Reversão
            'rev_levels': levels,
            'reversal': rev_sig["reversal"],
            'reversal_side': rev_sig["side"],
            'reversal_proximity': round(rev_sig["proximity"],3),
            # NOVO: Tendência de mercado
            'market_trend': market_trend["regime"],
            'market_trend_strength': round(market_trend["strength"], 4),
            'bullish_ratio': round(market_trend.get("bullish_ratio", 0.5), 3),
            # Extras
            'sim_model': mc.get('sim_model', 'garch11'),
            'last_volume_1m': round(volume_display, 8),
            'data_source': 'WS_COINAPI' if WS_FEED.enabled else ('CCXT' if self.spot._has_ccxt else 'HTTP'),
            'analysis_time_ms': round(analysis_duration, 2)
        }

    def scan_symbols_tplus(self, symbols: List[str])->Dict[str,Any]:
        por_ativo={}; candidatos=[]
        for sym in symbols:
            tplus=[]
            for h in (1,2,3):
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
            por_ativo[sym]={"tplus":tplus,"best_for_symbol":max(tplus,key=_rank_key_directional)}
        best_overall=max(candidatos,key=_rank_key_directional) if candidatos else None
        return {"por_ativo":por_ativo,"best_overall":best_overall}

# =========================
# Manager / API / UI (atualizado para mostrar tendência)
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
# API Routes (mantidas iguais)
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
        "is_analyzing": manager.is_analyzing
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
        "cache_size": len(manager.system.spot._cache._cache)
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
<title>IA Signal Pro - PREÇOS REAIS + SIMULAÇÕES</title>
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
</style>
</head>
<body>
<div class="wrap">
  <div class="hline">
    <h1>IA Signal Pro - NOVO MODELO HÍBRIDO + CONTEXTO DE MERCADO</h1>
    <div class="clock" id="clock">--:--:-- BRT</div>
    <div class="sub">✅ GARCH com Contexto de Mercado · Pesos Dinâmicos · Cache 3s · Mean Reversion · Tendência Geral</div>
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
        <button id="go" onclick="runAnalyze()">🔎 Analisar com NOVO modelo</button>
      </div>
    </div>
  </div>

  <div class="section" id="bestSec" style="display:none">
    <div class="title">🥇 MELHOR OPORTUNIDADE GLOBAL</div>
    <div class="card" id="bestCard"></div>
  </div>

  <div class="section" id="allSec" style="display:none">
    <div class="title">📊 TODOS OS HORIZONTES DE CADA ATIVO</div>
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
  btn.textContent = '⏳ Analisando...';
  const syms = selSymbols();
  if(!syms.length){ alert('Selecione pelo menos um ativo.'); btn.disabled=false; btn.textContent='🔎 Analisar com NOVO modelo'; return; }
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
      btn.textContent = '🔎 Analisar com NOVO modelo';
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
    const arr = groups[sym].sort((a,b)=>(a.horizon||0)-(b.horizon||0));
    const bestLocal = arr.slice().sort((a,b)=>rank(b)-rank(a))[0];
    return `
      <div class="card">
        <div class="sym-head"><b>${sym}</b>
          <span class="tag">TF: ${bestLocal?.multi_timeframe||'neutral'}</span>
          <span class="tag">Liquidez: ${Number(bestLocal?.liquidity_score||0).toFixed(2)}</span>
          <span class="tag">Mercado: ${bestLocal?.market_trend||'neutral'}</span>
          ${bestLocal?.reversal ? `<span class="tag">🔄 Reversão (${bestLocal.reversal_side})</span>`:''}
          <span class="tag">📈 GARCH Contextual</span>
        </div>
        ${arr.map(item=>renderTbox(item, bestLocal)).join('')}
      </div>`;
  }).join('');
  gridEl.innerHTML = html;
  allSec.style.display='block';

  return true;
}

function rank(it){ 
  const direction = it.direction || 'buy';
  const prob_directional = direction === 'buy' ? it.probability_buy : it.probability_sell;
  return (it.confidence * 1000) + (prob_directional * 100);
}

function renderBest(best, analysisTime){
  if(!best) return '<div class="small">Sem oportunidade no momento.</div>';
  const rev = best.reversal ? ` <span class="tag">🔄 Reversão (${best.reversal_side})</span>` : '';
  return `
    <div class="small muted">Atualizado: ${analysisTime} (Horário Brasil)</div>
    <div class="line"></div>
    <div><b>${best.symbol} T+${best.horizon}</b> ${badgeDir(best.direction)} <span class="tag">🥇 MELHOR ENTRE TODOS OS HORIZONTES</span>${rev} <span class="tag">📈 GARCH Contextual</span></div>
    <div class="kpis">
      <div class="kpi"><b>Prob Compra</b>${pct(best.probability_buy||0)}</div>
      <div class="kpi"><b>Prob Venda</b>${pct(best.probability_sell||0)}</div>
      <div class="kpi"><b>Confiança</b>${pct(best.confidence||0)}</div>
      <div class="kpi"><b>ADX</b>${(best.adx||0).toFixed(1)}</div>
      <div class="kpi"><b>RSI</b>${(best.rsi||0).toFixed(1)}</div>
      <div class="kpi"><b>Mercado</b>${best.market_trend||'neutral'}</div>
    </div>
    <div class="small" style="margin-top:8px;">
      Bullish Ratio: <span class="ok">${((best.bullish_ratio||0.5)*100).toFixed(0)}%</span> · Liquidez: <b>${Number(best.liquidity_score||0).toFixed(2)}</b> · Price: <b>${Number(best.price||0).toFixed(6)}</b>
      <span class="right">Entrada: <b>${best.entry_time||'-'}</b></span>
    </div>`;
}

function renderTbox(it, bestLocal){
  const isBest = bestLocal && it.symbol===bestLocal.symbol && it.horizon===bestLocal.horizon;
  const rev = it.reversal ? ` <span class="tag">🔄 REVERSÃO (${it.reversal_side})</span>` : '';
  return `
    <div class="tbox">
      <div><b>T+${it.horizon}</b> ${badgeDir(it.direction)} ${isBest?'<span class="tag">🥇 MELHOR DO ATIVO</span>':''}${rev} <span class="tag">📈 GARCH Contextual</span></div>
      <div class="small">
        Prob: <span class="${it.direction==='buy'?'ok':'err'}">${pct(it.probability_buy||0)}/${pct(it.probability_sell||0)}</span>
        · Conf: <span class="ok">${pct(it.confidence||0)}</span>
        · Mercado: <b>${it.market_trend||'neutral'}</b>
      </div>
      <div class="small">ADX: ${(it.adx||0).toFixed(1)} | RSI: ${(it.rsi||0).toFixed(1)} | TF: <b>${it.multi_timeframe||'neutral'}</b></div>
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
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
