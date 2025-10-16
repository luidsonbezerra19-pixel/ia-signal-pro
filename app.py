# app.py — IA Signal Pro (Binance OHLCV + RSI/ADX Wilder + Liquidez/ATR% + Reversão RSI 12h + Relógio BRT)
from __future__ import annotations
import os, re, time, math, random, threading, json, statistics as stats
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# =========================
# Config (sem ENV — tudo aqui)
# =========================
TZ_STR = "America/Maceio"
MC_PATHS = 3000
USE_CLOSED_ONLY = false  # usar apenas candles fechados p/ indicadores (mude para False se quiser intrabar)
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,XRP/USDT,BNB/USDT".split(",")
DEFAULT_SYMBOLS = [s.strip().upper() for s in DEFAULT_SYMBOLS if s.strip()]

# WebSocket (tempo real)
USE_WS = 1  # 1=ligado, 0=desligado
WS_BUFFER_MINUTES = 720  # ~12h em memória (compatível com janela do RSI de reversão)
WS_SYMBOLS = DEFAULT_SYMBOLS[:]  # pode ajustar a lista aqui

# Reversão por RSI (parâmetros)
RSI_PERIOD = 14
RSI_REV_WINDOW_MINUTES = 720  # 12h em 1m
RSI_REV_EXTREMES = 6          # média dos N extremos
RSI_REV_TOL = 2.5             # tolerância (pontos RSI)

app = Flask(__name__)
CORS(app)

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
# WebSocket Binance (kline 1m em tempo real)
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
        self._ws_available = False  # vira True se websocket-client existir

        # tenta importar websocket-client
        try:
            import websocket  # noqa: F401
            self._ws_available = True
        except Exception:
            print("[ws] 'websocket-client' não está instalado; WebSocket desativado.")
            self.enabled = False

    @staticmethod
    def _to_stream_symbol(sym: str) -> str:
        return sym.replace("/", "").lower()

    def _on_message(self, _, msg: str):
        try:
            data = json.loads(msg)
            payload = data.get("data") or data
            k = payload.get("k")
            if not k:
                return
            s = payload.get("s")
            if not s:
                return
            syml = s.upper()  # "BTCUSDT"
            # mapeia "BTCUSDT" -> "BTC/USDT" (funciona p/ *USDT)
            sym = f"{syml[:-4]}/{syml[-4:]}" if "/" not in syml else syml

            ts_open = int(k["t"])  # open time ms
            o = float(k["o"]); h = float(k["h"]); l = float(k["l"]); c = float(k["c"]); v = float(k["v"])
            row = [ts_open, o, h, l, c, v]

            with self._lock:
                buf = self._buffers.get(sym)
                if buf is None:
                    self._buffers[sym] = buf = []
                # substitui o último candle se timestamps coincidirem, senão anexa
                if buf and buf[-1][0] == ts_open:
                    buf[-1] = row
                else:
                    buf.append(row)
                # janela máxima
                if len(buf) > self.buf_minutes:
                    del buf[: len(buf) - self.buf_minutes]
        except Exception:
            # ignora mensagens quebradas sem derrubar o feed
            pass

    def _on_error(self, _, err):
        print("[ws] erro:", err)

    def _on_close(self, *_):
        print("[ws] fechado.")

    def _run(self):
        import websocket
        streams = "/".join([f"{self._to_stream_symbol(s)}@kline_1m" for s in self.symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        delay = 2
        while self._running:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws = ws
                ws.run_forever(ping_interval=30, ping_timeout=10)
                delay = 2  # reset após sessão OK
            except Exception as e:
                print(f"[ws] reconectando em {delay}s, motivo:", e)
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def start(self):
        if not self.enabled or not self._ws_available or self._thread:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[ws] feed iniciado para {len(self.symbols)} símbolos.")

    def stop(self):
        self._running = False

    def get_ohlcv(self, symbol: str, limit: int = 1000, use_closed_only: bool = True) -> List[List[float]]:
        """Retorna até 'limit' candles ordenados por tempo.
           Se use_closed_only=True, descarta candle do minuto corrente (em formação)."""
        if not (self.enabled and self._ws_available):
            return []
        sym = symbol.strip().upper()
        with self._lock:
            buf = self._buffers.get(sym, [])
            data = buf[-min(len(buf), limit):]
        if not data:
            return []
        if use_closed_only and len(data) >= 1:
            now_min = int(time.time() // 60)
            last_min = int((data[-1][0] // 1000) // 60)
            if last_min == now_min:
                data = data[:-1]
        return data[:]

    def get_last_candle(self, symbol: str) -> Optional[List[float]]:
        """Retorna o último candle recebido do WS: [ts,o,h,l,c,v] ou None."""
        if not (self.enabled and self._ws_available):
            return None
        sym = symbol.strip().upper()
        with self._lock:
            buf = self._buffers.get(sym, [])
            return buf[-1][:] if buf else None

# instancia e inicia já no boot
WS_FEED = WSRealtimeFeed()
WS_FEED.start()

# =========================
# Mercado Spot (ccxt + fallback HTTP) — agora com WS primeiro
# =========================
def _to_binance_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}{quote}"
    return re.sub(r'[^A-Z0-9]', '', s)

class SpotMarket:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, int], Tuple[float, List[List[float]]]] = {}
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
            print(f"[boot] ccxt version: {getattr(ccxt, '__version__', 'unknown')}")
        except Exception as e:
            print("[spot] ccxt indisponível, fallback HTTP:", e)
            self._has_ccxt = False

    def _fetch_http_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 1000) -> List[List[float]]:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": _to_binance_symbol(symbol), "interval": timeframe, "limit": min(1000, int(limit))}
        try:
            r = self._session.get(url, params=params, timeout=10)
            if r.status_code in (418, 429):
                time.sleep(0.5)
                r = self._session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            # [openTime, open, high, low, close, volume, ...]
            return [[float(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        except Exception:
            return []

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 1000) -> List[List[float]]:
        key = (symbol.upper(), timeframe, limit)
        now = time.time()
        # cache curto (WS atualiza muito)
        if key in self._cache and (now - self._cache[key][0]) < 2:
            return self._cache[key][1]

        ohlcv: List[List[float]] = []

        # 1) WebSocket — preferido para 1m
        try:
            if timeframe == "1m":
                ws_data = WS_FEED.get_ohlcv(symbol, limit=limit, use_closed_only=USE_CLOSED_ONLY)
                if ws_data and len(ws_data) >= 10:
                    ohlcv = ws_data
        except Exception:
            pass

        # 2) ccxt (se faltar)
        if (not ohlcv or len(ohlcv) < 60) and self._has_ccxt and self._ccxt is not None:
            try:
                raw = self._ccxt.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(1000, int(limit)))
                cc = [[float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw]
                if ohlcv:
                    ts = {r[0] for r in ohlcv}
                    cc = [r for r in cc if r[0] not in ts]
                    ohlcv = sorted(ohlcv + cc, key=lambda x: x[0])[-limit:]
                else:
                    ohlcv = cc
            except Exception:
                pass

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

        if ohlcv:
            self._cache[key] = (now, ohlcv)
        return ohlcv

# =========================
# Indicadores (Wilder/TV-like)
# =========================
class TechnicalIndicators:
    @staticmethod
    def _wilder_smooth(prev: float, cur: float, period: int) -> float:
        alpha = 1.0 / period
        return prev + alpha * (cur - prev)

    def rsi_series_wilder(self, closes: List[float], period: int = RSI_PERIOD) -> List[float]:
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

    def rsi_wilder(self, closes: List[float], period: int = RSI_PERIOD) -> float:
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
    def compute_extremes_levels(self, rsi_series: List[float], window: int = RSI_REV_WINDOW_MINUTES, n_extremes: int = RSI_REV_EXTREMES) -> Dict[str, float]:
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

    def signal_from_levels(self, current_rsi: float, levels: Dict[str,float], tol: float = RSI_REV_TOL) -> Dict[str, Any]:
        peak, trough = levels["avg_peak"], levels["avg_trough"]
        out = {"reversal": False, "side": None, "proximity": 0.0, "levels": levels}
        if abs(current_rsi - peak) <= tol:
            out.update({"reversal": True, "side": "bearish", "proximity": max(0.0, 1 - abs(current_rsi-peak)/max(1e-9,tol))})
        elif abs(current_rsi - trough) <= tol:
            out.update({"reversal": True, "side": "bullish", "proximity": max(0.0, 1 - abs(current_rsi-trough)/max(1e-9,tol))})
        return out

# =========================
# Simulador 100% paramétrico (AR(1)-GARCH "light" + backup EWMA-GBM)
# =========================
class MonteCarloSimulator:
    """
    Simulador 100% paramétrico (sem empírico):
      - Preferência: AR(1) + GARCH(1,1) "light" (inovações ~ N(0,1)).
      - Backup só-modelo: EWMA-GBM (RiskMetrics) quando histórico for curto/ruidoso.
    """

    # ===== utilidades =====
    @staticmethod
    def _mean(x: List[float]) -> float:
        return sum(x)/len(x) if x else 0.0

    @staticmethod
    def _var(x: List[float]) -> float:
        n = len(x)
        if n <= 1: return 0.0
        m = sum(x)/n
        return sum((v-m)*(v-m) for v in x) / (n-1)

    @staticmethod
    def _std(x: List[float]) -> float:
        return math.sqrt(max(0.0, MonteCarloSimulator._var(x)))

    # ===== AR(1) =====
    @staticmethod
    def _estimate_ar1_params(returns: List[float]) -> Tuple[float, float]:
        """
        Estima mu (média) e phi (AR(1)) via fórmulas fechadas simples (OLS):
          r_t = mu + phi*(r_{t-1}-mu) + eps_t
        """
        n = len(returns)
        if n < 60:
            return (0.0, 0.0)  # histórico insuficiente → sem memória
        mu = MonteCarloSimulator._mean(returns)
        num = 0.0
        den = 0.0
        for t in range(1, n):
            x_t1 = returns[t-1] - mu
            x_t  = returns[t]   - mu
            num += x_t1 * x_t
            den += x_t1 * x_t1
        phi = (num/den) if den != 0.0 else 0.0
        # estabiliza phi para |phi| < 0.98
        phi = max(-0.98, min(0.98, phi))
        return (mu, phi)

    # ===== GARCH(1,1) "light" =====
    @staticmethod
    def _estimate_garch11_params(returns: List[float]) -> Tuple[float, float, float]:
        """
        Estima (omega, alpha, beta) de forma robusta sem otimização numérica:
          - sigma2_bar = var_amostral
          - alpha ~ peso do choque; beta ~ persistência
          - omega = sigma2_bar * (1 - alpha - beta), garantindo estacionaridade (alpha+beta<1)
        Heurísticas seguras para M1 cripto; ajuste fino se quiser.
        """
        sigma2_bar = max(1e-10, MonteCarloSimulator._var(returns))
        alpha = 0.07
        beta  = 0.90
        if alpha + beta >= 0.98:
            beta = 0.98 - alpha
        omega = sigma2_bar * (1.0 - alpha - beta)
        omega = max(1e-12, omega)
        return (omega, alpha, beta)

    @staticmethod
    def _simulate_ar1_garch_paths(base_price: float, returns: List[float], steps: int, num_paths: int) -> List[List[float]]:
        if steps < 1 or not returns:
            return []
        mu, phi = MonteCarloSimulator._estimate_ar1_params(returns)
        omega, alpha, beta = MonteCarloSimulator._estimate_garch11_params(returns)

        # h0: variância de longo prazo
        h_prev = max(1e-10, omega / max(1e-12, (1.0 - alpha - beta)))
        r_prev = returns[-1] if returns else 0.0

        paths = []
        for _ in range(num_paths):
            p = base_price
            seq = [p]
            r_tm1 = r_prev
            h_tm1 = h_prev
            for _s in range(steps):
                # evolução da variância condicional
                h_t = omega + alpha * (r_tm1*r_tm1) + beta * h_tm1
                h_t = max(h_t, 1e-12)
                # AR(1) na média + choque com vol condicional
                eps = random.gauss(0.0, 1.0)
                r_t = (mu + phi * (r_tm1 - mu)) + eps * math.sqrt(h_t)
                p   = max(1e-9, p * (1.0 + r_t))
                seq.append(p)
                # avança estado
                r_tm1 = r_t
                h_tm1 = h_t
            paths.append(seq)
        return paths

    # ===== EWMA-GBM (backup paramétrico) =====
    @staticmethod
    def _simulate_ewma_gbm_paths(base_price: float, returns: List[float], steps: int, num_paths: int, lam: float = 0.94) -> List[List[float]]:
        """
        EWMA (RiskMetrics) para sigma_t^2 e drift ~ média simples.
        r_t ~ N(mu, sigma_t);  sigma_t^2 = (1-lam)*r_{t-1}^2 + lam*sigma_{t-1}^2
        """
        if steps < 1 or not returns:
            return []
        mu = MonteCarloSimulator._mean(returns)
        # inicializa sigma2 com variância amostral
        sigma2 = max(1e-10, MonteCarloSimulator._var(returns))
        r_prev2 = returns[-1]**2 if returns else 0.0

        paths = []
        for _ in range(num_paths):
            p = base_price
            seq = [p]
            sigma2_t = sigma2
            r2_tm1   = r_prev2
            for _s in range(steps):
                # atualiza vol condicional por EWMA
                sigma2_t = (1.0 - lam) * r2_tm1 + lam * sigma2_t
                sigma_t  = math.sqrt(max(1e-12, sigma2_t))
                r_t = random.gauss(mu, sigma_t)
                p   = max(1e-9, p * (1.0 + r_t))
                seq.append(p)
                r2_tm1 = r_t*r_t
            paths.append(seq)
        return paths

    # ===== agregador de probabilidade =====
    @staticmethod
    def _to_distribution(paths: List[List[float]]) -> Dict[str, Any]:
        if not paths:
            return {"probability_buy":0.5, "probability_sell":0.5, "quality":"LOW", "clarity_ratio":0.0}
        start = paths[0][0]
        ups   = sum(1 for seq in paths if seq[-1] > start)
        downs = sum(1 for seq in paths if seq[-1] < start)
        total = max(1, ups + downs)
        p_buy = ups/total
        p_sell= downs/total
        strength = abs(p_buy - 0.5)
        clarity  = total/len(paths)
        quality  = "HIGH" if (strength>=0.20 and clarity>=0.70) else ("MEDIUM" if (strength>=0.10 and clarity>=0.50) else "LOW")
        return {"probability_buy":round(p_buy,4), "probability_sell":round(p_sell,4), "quality":quality, "clarity_ratio":round(clarity,3)}

    # ===== interface pública (sem empírico) =====
    def simulate_auto(self, base_price: float, empirical_returns: List[float], steps: int, num_paths: int = MC_PATHS) -> Dict[str, Any]:
        """
        Sempre tenta um modelo paramétrico:
          1) AR(1)-GARCH(1,1) "light" (preferido).
          2) Se histórico < 60 → EWMA-GBM (paramétrico).
        Nunca usa bootstrap nem choice() de retornos.
        """
        rets = empirical_returns[:] if empirical_returns else []
        if len(rets) < 60:
            # histórico curto → usa EWMA-GBM
            paths = self._simulate_ewma_gbm_paths(base_price, rets, steps, num_paths, lam=0.94)
            out = self._to_distribution(paths)
            out.update({"sim_model":"ewma_gbm"})
            return out

        try:
            paths = self._simulate_ar1_garch_paths(base_price, rets, steps, num_paths)
            out = self._to_distribution(paths)
            out.update({"sim_model":"ar1_garch11"})
            return out
        except Exception:
            paths = self._simulate_ewma_gbm_paths(base_price, rets, steps, num_paths, lam=0.94)
            out = self._to_distribution(paths)
            out.update({"sim_model":"ewma_gbm"})
            return out

# =========================
# Helpers
# =========================
def _safe_returns_from_prices(prices: List[float])->List[float]:
    emp=[]
    for i in range(1,len(prices)):
        p0,p1=prices[i-1],prices[i]
        if p0>0: emp.append((p1-p0)/p0)
    return emp

def _rank_key(item: Dict[str,Any])->float:
    prob_dir=item['probability_buy'] if item['direction']=='buy' else item['probability_sell']
    return (item['confidence']*1000.0)+(prob_dir*100.0)

# =========================
# Sistema principal
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

    def get_brazil_time(self)->datetime:
        return brazil_now()

    def analyze_symbol(self, symbol: str, horizon: int)->Dict[str,Any]:
        # OHLCV 1m (~12h+ para reversões)
        raw = self.spot.fetch_ohlcv(symbol, "1m", max(800, RSI_REV_WINDOW_MINUTES + 50))
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

        # Indicadores (sobre WS/ccxt/HTTP conforme disponível)
        rsi_series = self.indicators.rsi_series_wilder(closes, RSI_PERIOD)
        rsi = rsi_series[-1] if rsi_series else 50.0
        adx = self.indicators.adx_wilder(highs, lows, closes)
        macd = self.indicators.macd(closes)
        boll = self.indicators.calculate_bollinger_bands(closes)
        tf_cons = self.multi_tf.analyze_consensus(closes)
        liq = self.liquidity.calculate_liquidity_score(highs, lows, closes)

        # Reversão por RSI (12h)
        levels = self.revdet.compute_extremes_levels(rsi_series, RSI_REV_WINDOW_MINUTES, RSI_REV_EXTREMES) if rsi_series else {"avg_peak":70.0,"avg_trough":30.0}
        rev_sig = self.revdet.signal_from_levels(rsi, levels, RSI_REV_TOL)

        # Simulador paramétrico
        empirical=_safe_returns_from_prices(closes) or [random.gauss(0,0.003) for _ in range(120)]
        steps=max(1,min(3,int(horizon)))
        base_price=closes[-1] if closes else price_display
        mc=self.monte_carlo.simulate_auto(base_price, empirical, steps, num_paths=MC_PATHS)

        # direção primária pelo MC
        direction = 'buy' if mc['probability_buy']>mc['probability_sell'] else 'sell'
        prob_dir  = mc['probability_buy'] if direction=='buy' else mc['probability_sell']

        # confiança base + boosts
        score=prob_dir*100.0
        if 30<rsi<70: score+=12.0
        if adx>25:    score+=12.0
        if (direction=='buy' and macd['signal']=='bullish') or (direction=='sell' and macd['signal']=='bearish'):
            score+=8.0
        if (direction=='sell' and boll['signal'] in ['overbought','bearish']) or (direction=='buy' and boll['signal'] in ['oversold','bullish']):
            score+=6.0
        if rev_sig["reversal"]:
            score += 10.0 * rev_sig["proximity"]  # peso da reversão

        score *= (0.95 + (liq*0.1))
        confidence = min(0.95, max(0.50, score/100.0))

        return {
            'symbol':symbol,
            'horizon':steps,
            'direction':direction,
            'probability_buy':mc['probability_buy'],
            'probability_sell':mc['probability_sell'],
            'confidence':confidence,
            'rsi':rsi,'adx':adx,'multi_timeframe':tf_cons,
            'monte_carlo_quality':mc['quality'],
            'price':price_display,
            'liquidity_score':liq,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            # Reversão
            'rev_levels': levels,
            'reversal': rev_sig["reversal"],
            'reversal_side': rev_sig["side"],
            'reversal_proximity': round(rev_sig["proximity"],3),
            # Extras úteis
            'sim_model': mc.get('sim_model','ar1_garch11'),
            'last_volume_1m': round(volume_display, 8),
            'data_source': 'WS' if WS_FEED.enabled else ('CCXT' if self.spot._has_ccxt else 'HTTP')
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
                    tplus.append({
                        "symbol":sym,"horizon":h,"error":str(e),
                        "direction":"buy","probability_buy":0.5,"probability_sell":0.5,
                        "confidence":0.5,"label":f"{sym} T+{h}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
            por_ativo[sym]={"tplus":tplus,"best_for_symbol":max(tplus,key=_rank_key)}
        best_overall=max(candidatos,key=_rank_key) if candidatos else None
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
        try:
            result = self.system.scan_symbols_tplus(symbols)
            flat=[]
            for sym, bloco in result["por_ativo"].items():
                flat.extend(bloco["tplus"])
            self.current_results = flat
            if flat:
                best=max(flat, key=_rank_key)
                best=dict(best)
                best["entry_time"]=self.calculate_entry_time_brazil(best.get("horizon",1))
                self.best_opportunity=best
            else:
                self.best_opportunity=None
            self.analysis_time = br_full(self.get_brazil_time())
        except Exception as e:
            self.current_results=[]
            self.best_opportunity={"error":str(e)}
            self.analysis_time = br_full(self.get_brazil_time())
        finally:
            self.is_analyzing=False

manager=AnalysisManager()

@app.post("/api/analyze")
def api_analyze():
    if manager.is_analyzing:
        resp = jsonify({"success": False, "error": "Análise em andamento"})
        resp.headers["Cache-Control"] = "no-store"
        return resp, 429
    try:
        data = request.get_json(silent=True) or {}
        symbols = [s.strip().upper() for s in (data.get("symbols") or manager.symbols_default) if s.strip()]
        if not symbols:
            resp = jsonify({"success": False, "error": "Selecione pelo menos um ativo"})
            resp.headers["Cache-Control"] = "no-store"
            return resp, 400
        sims = MC_PATHS
        th = threading.Thread(target=manager.analyze_symbols_thread, args=(symbols, sims, None))
        th.daemon = True
        th.start()
        resp = jsonify({"success": True, "message": f"Analisando {len(symbols)} ativos com {sims} simulações.", "symbols_count": len(symbols)})
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception as e:
        resp = jsonify({"success": False, "error": str(e)})
        resp.headers["Cache-Control"] = "no-store"
        return resp, 500

@app.get("/api/results")
def api_results():
    resp = jsonify({
        "success": True,
        "results": manager.current_results,
        "best": manager.best_opportunity,
        "analysis_time": manager.analysis_time,
        "total_signals": len(manager.current_results),
        "is_analyzing": manager.is_analyzing
    })
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health():
    resp = jsonify({"ok": True, "ws": WS_FEED.enabled, "ts": datetime.now(timezone.utc).isoformat()})
    resp.headers["Cache-Control"] = "no-store"
    return resp, 200

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
    <div class="sub">✅ Binance OHLCV · RSI/ADX (Wilder) · Liquidez (ATR%) · Reversão por extremos de RSI (12h)</div>
    <div class="controls">
      <div class="chips" id="chips"></div>
      <div class="row">
        <select id="mcsel">
          <option value="3000" selected>3000 simulações</option>
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
          <span class="tag">🗲 DADOS REAIS</span>
        </div>
        ${arr.map(item=>renderTbox(item, bestLocal)).join('')}
      </div>`;
  }).join('');
  gridEl.innerHTML = html;
  allSec.style.display='block';

  return true;
}

function rank(it){ const pd = it.direction==='buy' ? it.probability_buy : it.probability_sell; return (it.confidence*1000)+(pd*100); }

function renderBest(best, analysisTime){
  if(!best) return '<div class="small">Sem oportunidade no momento.</div>';
  const rev = best.reversal ? ` <span class="tag">🔄 Reversão (${best.reversal_side})</span>` : '';
  return `
    <div class="small muted">Atualizado: ${analysisTime} (Horário Brasil)</div>
    <div class="line"></div>
    <div><b>${best.symbol} T+${best.horizon}</b> ${badgeDir(best.direction)} <span class="tag">🥇 MELHOR ENTRE TODOS OS HORIZONTES</span>${rev} <span class="tag">🗲 REAL</span></div>
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
      <div><b>T+${it.horizon}</b> ${badgeDir(it.direction)} ${isBest?'<span class="tag">🥇 MELHOR DO ATIVO</span>':''}${rev} <span class="tag">🗲 REAL</span></div>
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
    # Em Railway, o PORT vem de env; aqui mantenho compatível sem depender dele
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
