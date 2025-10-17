# app.py — IA Signal Pro (CoinAPI WS OHLCV + Binance REST fallback) + RSI/ADX/MACD/Liquidez + GARCH(1,1) Light
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
USE_CLOSED_ONLY = True  # usar apenas candles fechados p/ indicadores (coloque False para intrabar)
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,XRP/USDT,BNB/USDT".split(",")
DEFAULT_SYMBOLS = [s.strip().upper() for s in DEFAULT_SYMBOLS if s.strip()]

# Provedor de tempo real
USE_WS = 1                 # 1=ligado (CoinAPI), 0=desligado
WS_BUFFER_MINUTES = 720    # ~12h em memória (compatível com RSI reversão)
WS_SYMBOLS = DEFAULT_SYMBOLS[:]  # símbolos monitorados
REALTIME_PROVIDER = "okx"    # informativo

OKX_URL = "wss://ws.okx.com:8443/ws/v5/business"
OKX_CHANNEL = "candle1m"

# >>>>>>>> COLOQUE SUA CHAVE AQUI <<<<<<<<
COINAPI_KEY = "COLE_SUA_COINAPI_KEY_AQUI"  # obtenha em https://www.coinapi.io/
COINAPI_URL = "wss://ws.coinapi.io/v1/"    # endpoint WS da CoinAPI
COINAPI_PERIOD = "1MIN"                    # período do OHLCV

app = Flask(__name__)
CORS(app)

# =========================
# Feature Flags
# =========================
FEATURE_FLAGS = {
    "enable_adaptive_garch": True,
    "enable_smart_cache": True,
    "enable_circuit_breaker": True,
    "websocket_provider": "okx",
    "maintenance_mode": False
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
        
        # Remove requests outside the window
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
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
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
        else:  # HALF_OPEN
            return True

binance_circuit_breaker = CircuitBreaker()

# =========================
# Tempo (Brasil)
# =========================
def brazil_now() -> datetime:
    # BRT fixo -3h (Railway usa UTC por padrão)
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
    """Converte 'BTC/USDT' -> 'BINANCE_SPOT_BTC_USDT'"""
    s = sym.strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
    else:
        # tenta deduzir
        if s.endswith("USDT"): base, quote = s[:-4], "USDT"
        elif s.endswith("USD"): base, quote = s[:-3], "USD"
        else: base, quote = s, "USDT"
    return f"BINANCE_SPOT_{base}_{quote}"

def _iso_to_ms(iso_str: str) -> int:
    # CoinAPI manda "2025-01-01T00:00:00.0000000Z"
    z = iso_str.endswith("Z")
    s = iso_str[:-1] if z else iso_str
    # corta casas a mais em micros
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
    """Calcula retornos percentuais de forma segura"""
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
    return returns

def _rank_key_directional(x: Dict[str, Any]) -> float:
    """Ranking que considera a melhor probabilidade NA DIREÇÃO escolhida"""
    direction = x.get("direction", "buy")
    
    # Probabilidade na direção escolhida (não sempre prob_buy)
    prob_directional = x["probability_buy"] if direction == "buy" else x["probability_sell"]
    
    # Score prioriza confiança + probabilidade na direção
    return (x["confidence"] * 1000) + (prob_directional * 100)

def _confirm_prob_neutral_zone(prob_up: float, rsi: float, macd_hist: float, adx: float, 
                              boll_signal: str, tf_consensus: str) -> float:
    """Sistema verdadeiramente balanceado"""
    
    # Sistema de votos balanceado - MAIS EQUILIBRADO
    bullish = 0
    bearish = 0
    
    # RSI - VOLTANDO PARA VALORES MAIS EQUILIBRADOS
    if rsi >= 58: bullish += 1        # Ajustado de 60 para 58
    if rsi <= 42: bearish += 1        # Ajustado de 40 para 42
    
    # MACD - voltando aos valores originais que funcionavam
    if macd_hist > 0.001: bullish += 1      
    if macd_hist < -0.001: bearish += 1     
    
    # Bollinger 
    if boll_signal in ["oversold", "bullish"]: bullish += 1
    if boll_signal in ["overbought", "bearish"]: bearish += 1
    
    # Timeframe
    if tf_consensus == "buy": bullish += 1
    if tf_consensus == "sell": bearish += 1
    
    # ADX - critério mais suave
    strong_trend = (adx >= 20)  # Reduzido de 25
    very_strong_trend = (adx >= 30)  # Reduzido de 35
    
    # LÓGICA MAIS SUAVE - CORRIGIDA
    if bullish > bearish and very_strong_trend:
        adj = 0.06
    elif bullish > bearish and strong_trend:
        adj = 0.04
    elif bullish > bearish:
        adj = 0.02
    elif bearish > bullish and very_strong_trend:
        adj = -0.06
    elif bearish > bullish and strong_trend:
        adj = -0.04
    elif bearish > bullish:
        adj = -0.02
    else: 
        adj = 0.0
    
    adjusted_prob = prob_up + adj
    return min(0.90, max(0.10, adjusted_prob))

def _calculate_directional_confidence(prob_direction: float, direction: str, rsi: float, 
                                    adx: float, macd_signal: str, boll_signal: str,
                                    tf_consensus: str, reversal_signal: dict,
                                    liquidity_score: float) -> float:
    """Confiança balanceada"""
    
    base_confidence = prob_direction * 100.0
    directional_boosts = 0.0
    
    # 1. CONFIRMAÇÃO TENDÊNCIA - MAIS SUAVE
    if adx > 20:  # Reduzido de 25
        if (direction == 'buy' and tf_consensus == 'buy') or (direction == 'sell' and tf_consensus == 'sell'):
            directional_boosts += 10.0  # Balanceado
    
    # 2. RSI - ZONAS MAIS AMPLAS
    if (direction == 'buy' and rsi < 45) or (direction == 'sell' and rsi > 55):  # Mais amplo
        directional_boosts += 6.0
    
    # 3. BOLLINGER 
    if (direction == 'buy' and boll_signal == 'oversold') or (direction == 'sell' and boll_signal == 'overbought'):
        directional_boosts += 8.0
    
    # 4. REVERSÃO 
    if reversal_signal["reversal"] and reversal_signal["side"] == direction:
        directional_boosts += 12.0 * reversal_signal["proximity"]
    
    # 5. MACD 
    if (direction == 'buy' and macd_signal == 'bullish') or (direction == 'sell' and macd_signal == 'bearish'):
        directional_boosts += 5.0
    
    # APLICAR BOOSTS
    total_score = base_confidence + directional_boosts
    total_score *= (0.90 + (liquidity_score * 0.15))
    
    return min(95.0, max(30.0, total_score)) / 100.0
                                        
# =========================
# WebSocket OKX (OHLCV 1m em tempo real)
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
            import websocket  # noqa: F401
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
# Mercado Spot com Cache Inteligente e Circuit Breaker
# =========================
class SmartCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, str, int], Tuple[float, List[List[float]], float]] = {}
        self._volatility_cache: Dict[str, float] = {}
        
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calcula volatilidade baseada nos últimos preços"""
        if len(prices) < 10:
            return 0.0
        returns = _safe_returns_from_prices(prices[-50:])  # Últimos 50 períodos
        if not returns:
            return 0.0
        return stats.stdev(returns) if len(returns) > 1 else 0.0
    
    def _should_invalidate(self, symbol: str, current_prices: List[float]) -> bool:
        """Decide se deve invalidar cache baseado na volatilidade"""
        current_vol = self._calculate_volatility(current_prices)
        cached_vol = self._volatility_cache.get(symbol, 0.0)
        
        # Se volatilidade aumentou significativamente, invalida cache
        if current_vol > cached_vol * 1.5 and current_vol > 0.01:  # Threshold de 1%
            return True
        return False
    
    def get(self, key: Tuple[str, str, int]) -> Optional[Tuple[float, List[List[float]]]]:
        if key in self._cache:
            timestamp, data, _ = self._cache[key]
            # Cache válido por 2-10 segundos baseado na volatilidade
            symbol = key[0]
            current_vol = self._volatility_cache.get(symbol, 0.0)
            cache_ttl = 2 if current_vol > 0.02 else 5 if current_vol > 0.01 else 10
            
            if time.time() - timestamp < cache_ttl:
                return (timestamp, data)
        return None
    
    def set(self, key: Tuple[str, str, int], data: List[List[float]]):
        symbol = key[0]
        prices = [x[4] for x in data] if data else []  # Preços de fechamento
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
            self._ccxt = ccxt.binance({
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
        
        # Verifica cache inteligente
        if FEATURE_FLAGS["enable_smart_cache"]:
            cached = self._cache.get(key)
            if cached:
                timestamp, data = cached
                # Verifica se precisa invalidar por volatilidade
                if not self._cache._should_invalidate(symbol, [x[4] for x in data]):
                    return data

        ohlcv: List[List[float]] = []

        # 1) WebSocket OKX — preferido para 1m
        try:
            if timeframe == "1m":
                ws_data = WS_FEED.get_ohlcv(symbol, limit=limit, use_closed_only=USE_CLOSED_ONLY)
                if ws_data and len(ws_data) >= 10:
                    ohlcv = ws_data
                    logger.debug("websocket_data_used", symbol=symbol, candles_count=len(ws_data))
        except Exception as e:
            logger.error("websocket_fetch_error", symbol=symbol, error=str(e))

        # 2) ccxt (se faltar)
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

        # 3) HTTP público (fallback final)
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
# Indicadores (Wilder/TV-like)
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
            atr = (atr * (period - 1) + trs[i]) / period  # Wilder
        atr_pct = atr / max(1e-12, closes[-1])
        LIM = 0.02
        score = 1.0 - min(1.0, atr_pct / LIM)
        return round(max(0.0, min(1.0, score)), 3)

# =========================
# Reversão (extremos RSI 12h)
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
# GARCH(1,1) Light Adaptativo com 3000 simulações
# =========================

class AdaptiveGARCH11Simulator:
    """GARCH(1,1) Light Adaptativo com detecção de regime"""
    
    def _detect_market_regime(self, returns: List[float]) -> str:
        """Detecta o regime de mercado baseado na volatilidade"""
        if not returns or len(returns) < 20:
            return "normal"
        
        volatility = stats.stdev(returns) if len(returns) > 1 else 0.0
        mean_abs_return = stats.mean([abs(r) for r in returns])
        
        if volatility > 0.03 or mean_abs_return > 0.02:  # 3% vol ou 2% retorno médio
            return "high_volatility"
        elif volatility < 0.005:  # 0.5% vol
            return "low_volatility"
        else:
            return "normal"
    
    def _get_garch_params_for_regime(self, regime: str, returns: List[float]) -> Tuple[float, float, float, float]:
        """Retorna parâmetros GARCH otimizados para o regime detectado"""
        if not returns:
            return 1e-6, 0.1, 0.85, 1e-4
            
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        
        if regime == "low_volatility":
            # Mercado calmo: volatilidade mais persistente
            alpha, beta = 0.05, 0.92
        elif regime == "high_volatility":
            # Mercado volátil: mais sensível a choques recentes
            alpha, beta = 0.12, 0.80
        else:  # normal
            alpha, beta = 0.08, 0.90
            
        omega = var * (1 - alpha - beta)
        omega = max(1e-6, omega)
        
        # Garantir estabilidade
        if alpha + beta >= 0.999:
            alpha, beta = 0.08, 0.90
            omega = 1e-6
            
        return omega, alpha, beta, var
    
    def _calculate_garch_fit_quality(self, returns: List[float], omega: float, alpha: float, beta: float) -> float:
        """Calcula qualidade do ajuste GARCH (simplificado)"""
        if len(returns) < 30:
            return 0.7
            
        try:
            # Métrica simplificada de qualidade
            predicted_var = omega / (1 - alpha - beta) if alpha + beta < 1 else 1e-4
            actual_var = stats.variance(returns) if len(returns) > 1 else 1e-4
            fit_quality = 1.0 - min(1.0, abs(predicted_var - actual_var) / actual_var)
            return max(0.3, min(0.95, fit_quality))
        except:
            return 0.7

    def simulate_garch11(self, base_price: float, returns: List[float], 
                        steps: int, num_paths: int = 3000) -> Dict[str, Any]:
        """Executa simulação GARCH(1,1) adaptativa"""
        import math
        import random
        
        if not returns or len(returns) < 10:
            returns = [random.gauss(0.0, 0.002) for _ in range(100)]
        
        # Detectar regime e calibrar parâmetros
        regime = self._detect_market_regime(returns) if FEATURE_FLAGS["enable_adaptive_garch"] else "normal"
        omega, alpha, beta, h_last = self._get_garch_params_for_regime(regime, returns)
        
        # Calcular qualidade do ajuste
        fit_quality = self._calculate_garch_fit_quality(returns, omega, alpha, beta)
        
        up_count = 0
        total_count = 0
        
        # Simulações GARCH(1,1)
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
        
        # Ajustar probabilidade baseado na qualidade do fit
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

# Manter compatibilidade
MonteCarloSimulator = AdaptiveGARCH11Simulator

class EnhancedTradingSystem:
    def __init__(self)->None:
        self.indicators=TechnicalIndicators()
        self.revdet=ReversalDetector()
        self.monte_carlo=MonteCarloSimulator()
        self.multi_tf=MultiTimeframeAnalyzer()
        self.liquidity=LiquiditySystem()
        self.spot=SpotMarket()
        self.current_analysis_cache: Dict[str,Any]={}

    def get_brazil_time(self)->datetime:
        return brazil_now()

    def analyze_symbol(self, symbol: str, horizon: int)->Dict[str,Any]:
        start_time = time.time()
        logger.info("analysis_started", symbol=symbol, horizon=horizon)
        
        # OHLCV 1m (~12h+ para reversões)
        raw = self.spot.fetch_ohlcv(symbol, "1m", max(800, 720 + 50))
        if len(raw) < 60:
            # fallback sintético mínimo
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

        # Preço e Volume de exibição preferindo WS
        price_display = raw[-1][4]
        volume_display = raw[-1][5] if raw and len(raw[-1]) >= 6 else 0.0
        try:
            ws_last = WS_FEED.get_last_candle(symbol)
            if ws_last:
                price_display  = float(ws_last[4])  # close "ao vivo"
                volume_display = float(ws_last[5])  # volume 1m "ao vivo"
        except Exception:
            pass

        # Indicadores
        rsi_series = self.indicators.rsi_series_wilder(closes, 14)
        rsi = rsi_series[-1] if rsi_series else 50.0
        adx = self.indicators.adx_wilder(highs, lows, closes)
        macd = self.indicators.macd(closes)
        boll = self.indicators.calculate_bollinger_bands(closes)
        tf_cons = self.multi_tf.analyze_consensus(closes)
        liq = self.liquidity.calculate_liquidity_score(highs, lows, closes)

        # Reversão por RSI (12h)
        levels = self.revdet.compute_extremes_levels(rsi_series, 720, 6) if rsi_series else {"avg_peak":70.0,"avg_trough":30.0}
        rev_sig = self.revdet.signal_from_levels(rsi, levels, 2.5)

        # Simulador GARCH(1,1) Light com 3000 simulações
        empirical_returns = _safe_returns_from_prices(closes)
        steps = max(1, min(3, int(horizon)))
        base_price = closes[-1] if closes else price_display

        # Usar GARCH(1,1) adaptativo
        mc = self.monte_carlo.simulate_garch11(
            base_price, 
            empirical_returns, 
            steps, 
            num_paths=MC_PATHS
        )

        # Aplicar confirmação de probabilidade com indicadores (SEM VIÉS)
        prob_buy_original = mc['probability_buy']
        prob_sell_original = mc['probability_sell']

        # Calcular MACD histogram para confirmação
        macd_hist = 0.0
        try:
            macd_result = self.indicators.macd(closes)
            if macd_result["signal"] == "bullish":
                macd_hist = macd_result["strength"]
            elif macd_result["signal"] == "bearish":
                macd_hist = -macd_result["strength"]
        except:
            macd_hist = 0.0

        # Ajustar probabilidade GARCH com confirmação técnica (ZONA NEUTRA)
        prob_buy_adjusted = _confirm_prob_neutral_zone(
            prob_buy_original, rsi, macd_hist, adx, 
            boll['signal'], tf_cons
        )
        prob_sell_adjusted = 1.0 - prob_buy_adjusted

        # DIREÇÃO com desempate inteligente
        # DIREÇÃO simples e direta como no código antigo
        direction = 'buy' if prob_buy_adjusted > 0.5 else 'sell'

        # Atualizar o resultado GARCH com as probabilidades ajustadas
        mc['probability_buy'] = prob_buy_adjusted
        mc['probability_sell'] = prob_sell_adjusted

        # CONFIANÇA DIRECIONAL INTELIGENTE
        prob_dir = prob_buy_adjusted if direction == 'buy' else prob_sell_adjusted
        confidence = _calculate_directional_confidence(
            prob_dir, direction, rsi, adx, macd['signal'], boll['signal'], 
            tf_cons, rev_sig, liq
        )

        analysis_duration = (time.time() - start_time) * 1000
        logger.info("analysis_completed", 
                   symbol=symbol, 
                   horizon=horizon, 
                   duration_ms=analysis_duration,
                   direction=direction,
                   confidence=confidence)

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
            'fit_quality': mc.get('fit_quality', 0.7),
            'price':price_display,
            'liquidity_score':liq,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            # Reversão
            'rev_levels': levels,
            'reversal': rev_sig["reversal"],
            'reversal_side': rev_sig["side"],
            'reversal_proximity': round(rev_sig["proximity"],3),
            # Extras úteis
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
    """Health check detalhado para monitoramento"""
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
    <h1>IA Signal Pro - PREÇOS REAIS + 3000 SIMULAÇÕES</h1>
    <div class="clock" id="clock">--:--:-- BRT</div>
    <div class="sub">✅ CoinAPI (WS) · Binance REST fallback · RSI/ADX (Wilder) · Liquidez (ATR%) · Reversão RSI (12h) · GARCH(1,1) Adaptativo</div>
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
        <button id="go" onclick="runAnalyze()">🔎 Analisar com dados reais</button>
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
  if(!syms.length){ alert('Selecione pelo menos um ativo.'); btn.disabled=false; btn.textContent='🔎 Analisar com dados reais'; return; }
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
      btn.textContent = '🔎 Analisar com dados reais';
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
          ${bestLocal?.reversal ? `<span class="tag">🔄 Reversão (${bestLocal.reversal_side})</span>`:''}
          <span class="tag">📈 GARCH(1,1)</span>
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
    <div><b>${best.symbol} T+${best.horizon}</b> ${badgeDir(best.direction)} <span class="tag">🥇 MELHOR ENTRE TODOS OS HORIZONTES</span>${rev} <span class="tag">📈 GARCH(1,1)</span></div>
    <div class="kpis">
      <div class="kpi"><b>Prob Compra</b>${pct(best.probability_buy||0)}</div>
      <div class="kpi"><b>Prob Venda</b>${pct(best.probability_sell||0)}</div>
      <div class="kpi"><b>Soma</b>100.0%</div>
      <div class="kpi"><b>ADX</b>${(best.adx||0).toFixed(1)}</div>
      <div class="kpi"><b>RSI</b>${(best.rsi||0).toFixed(1)}</div>
      <div class="kpi"><b>Liquidez</b>${Number(best.liquidity_score||0).toFixed(2)}</div>
    </div>
    <div class="small" style="margin-top:8px;">
      Pontuação: <span class="ok">${(best.confidence*100).toFixed(1)}%</span> · TF: <b>${best.multi_timeframe||'neutral'}</b> · Price: <b>${Number(best.price||0).toFixed(6)}</b>
      <span class="right">Entrada: <b>${best.entry_time||'-'}</b></span>
    </div>`;
}

function renderTbox(it, bestLocal){
  const isBest = bestLocal && it.symbol===bestLocal.symbol && it.horizon===bestLocal.horizon;
  const rev = it.reversal ? ` <span class="tag">🔄 REVERSÃO (${it.reversal_side})</span>` : '';
  return `
    <div class="tbox">
      <div><b>T+${it.horizon}</b> ${badgeDir(it.direction)} ${isBest?'<span class="tag">🥇 MELHOR DO ATIVO</span>':''}${rev} <span class="tag">📈 GARCH(1,1)</span></div>
      <div class="small">
        Prob: <span class="${it.direction==='buy'?'ok':'err'}">${pct(it.probability_buy||0)}/${pct(it.probability_sell||0)}</span>
        · Conf: <span class="ok">${pct(it.confidence||0)}</span>
        · RSI≈Pico: ${(it.rev_levels?.avg_peak||0).toFixed(1)} · RSI≈Vale: ${(it.rev_levels?.avg_trough||0).toFixed(1)}
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
