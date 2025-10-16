# app.py — IA Signal Pro (OKX WS OHLCV) + RSI/ADX/MACD + GARCH(1,1) 3000 sim (T+1..T+3)
from __future__ import annotations
import os, re, time, math, random, threading, json
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# =========================
# Config (sem ENV — tudo aqui)
# =========================
TZ_STR = "America/Maceio"
SIM_PATHS = 3000
USE_CLOSED_ONLY = False  # usar apenas candles fechados p/ indicadores
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,XRP/USDT,BNB/USDT".split(",")
DEFAULT_SYMBOLS = [s.strip().upper() for s in DEFAULT_SYMBOLS if s.strip()]

# Provedor de tempo real (OKX)
USE_WS = 1                 # 1=ligado (OKX), 0=desligado
WS_BUFFER_MINUTES = 720    # ~12h em memória
WS_SYMBOLS = DEFAULT_SYMBOLS[:]  # símbolos monitorados
REALTIME_PROVIDER = "okx"  # informativo

OKX_URL = "wss://ws.okx.com:8443/ws/v5/business"
OKX_CHANNEL = "candle1m"

# GARCH(1,1) Light
G_ALPHA = 0.06
G_BETA  = 0.94
HORIZONS = [1,2,3]

# Ajustes por indicadores
BOOST_STRONG = 0.07  # ADX >= 25 e consenso RSI/MACD
BOOST_MED    = 0.04
PENALTY_MIX  = 0.02

app = Flask(__name__)
CORS(app)

# =========================
# Tempo (Brasil)
# =========================
def brazil_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=-3)))

def br_full(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y %H:%M:%S")

# =========================
# Utils
# =========================
def _to_okx_inst_id(sym: str) -> str:
    """Converte 'BTC/USDT' -> 'BTC-USDT' (instId SPOT OKX)"""
    s = sym.strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
    else:
        if s.endswith("USDT"): base, quote = s[:-4], "USDT"
        elif s.endswith("USD"): base, quote = s[:-3], "USD"
        else: base, quote = s, "USDT"
    return f"{base}-{quote}"

# =========================
# WebSocket OKX (OHLCV 1m em tempo real)
# =========================
class WSRealtimeFeed:
    def __init__(self):
        self.enabled = bool(USE_WS)
        self.buf_minutes = int(WS_BUFFER_MINUTES)
        self.symbols = [s.strip().upper() for s in WS_SYMBOLS if s.strip()]
        self._lock = threading.Lock()
        # buffers: sym -> list[[ts,o,h,l,c,v]]
        self._buffers: Dict[str, List[List[float]]] = {s: [] for s in self.symbols}
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._running = False
        self._ws_available = False

        try:
            import websocket  # noqa: F401
            self._ws_available = True
        except Exception:
            print("[ws] 'websocket-client' não está instalado; WS desativado.")
            self.enabled = False

    def _on_open(self, ws):
        try:
            args = [{"channel": OKX_CHANNEL, "instId": _to_okx_inst_id(s)} for s in self.symbols]
            sub = {"op": "subscribe", "args": args}
            ws.send(json.dumps(sub))
            print("[ws] subscribe enviado para OKX; subs:", len(args))
        except Exception as e:
            print("[ws] erro ao enviar subscribe:", e)

    def _on_message(self, _, msg: str):
        # OKX candle: {"arg":{"channel":"candle1m","instId":"BTC-USDT"},"data":[["ts","o","h","l","c","vol",...]]}
        try:
            data = json.loads(msg)
            if data.get("event") in ("subscribe", "error"):
                if data.get("event") == "error":
                    print("[ws] erro OKX:", data)
                return
            arg = data.get("arg", {})
            if arg.get("channel") != OKX_CHANNEL:
                return
            inst = arg.get("instId", "")
            sym = inst.replace("-", "/")
            rows = data.get("data") or []
            for row in rows:
                ts_open = int(row[0])
                o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
                v = float(row[5]) if len(row)>5 else 0.0
                rec = [ts_open,o,h,l,c,v]
                with self._lock:
                    buf = self._buffers.get(sym)
                    if buf is None:
                        self._buffers[sym] = buf = []
                    if buf and buf[-1][0] == ts_open:
                        buf[-1] = rec
                    else:
                        buf.append(rec)
                        # limpeza: manter ~WS_BUFFER_MINUTES (assumindo 1/min)
                        if len(buf) > self.buf_minutes + 5:
                            del buf[:len(buf)-(self.buf_minutes+5)]
        except Exception:
            pass

    def _on_error(self, _, err):
        print("[ws] error:", err)

    def _on_close(self, *_):
        print("[ws] closed")

    def _run(self):
        from websocket import WebSocketApp
        while self._running:
            try:
                self._ws = WebSocketApp(
                    OKX_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self._ws.run_forever(ping_interval=25, ping_timeout=10)
            except Exception as e:
                print("[ws] run_forever exception:", e)
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

    def stop(self):
        self._running = False
        try:
            if self._ws: self._ws.close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2)

    # ====== APIs usadas pelo Manager ======
    def get_series(self, sym: str) -> List[List[float]]:
        with self._lock:
            return list(self._buffers.get(sym, []))

WS_FEED = WSRealtimeFeed()
WS_FEED.start()

# =========================
# Indicadores (do stream)
# =========================
def ema(values: List[float], period: int) -> List[float]:
    if not values: return []
    k = 2/(period+1)
    out = [values[0]]
    for v in values[1:]:
        out.append(k*v + (1-k)*out[-1])
    return out

def rsi(closes: List[float], period: int = 14) -> List[float]:
    n = len(closes)
    if n < period + 1: return []
    gains, losses = [], []
    for i in range(1, period+1):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0)); losses.append(abs(min(ch, 0.0)))
    avg_gain = sum(gains)/period
    avg_loss = sum(losses)/period
    rsis = []
    for i in range(period+1, n):
        ch = closes[i] - closes[i-1]
        gain = max(ch, 0.0); loss = abs(min(ch, 0.0))
        avg_gain = (avg_gain*(period-1) + gain)/period
        avg_loss = (avg_loss*(period-1) + loss)/period
        if avg_loss == 0: rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100.0 - 100.0/(1.0+rs))
    return rsis

def macd(closes: List[float], fast=12, slow=26, signal=9):
    if len(closes) < slow + signal: return [], [], []
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    m = min(len(ema_fast), len(ema_slow))
    macd_line = [ema_fast[-m+i] - ema_slow[-m+i] for i in range(m)]
    sig = ema(macd_line, signal)
    k = min(len(macd_line), len(sig))
    macd_line = macd_line[-k:]
    sig = sig[-k:]
    hist = [macd_line[i] - sig[i] for i in range(k)]
    return macd_line, sig, hist

def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    n = len(closes)
    if n < period + 1: return []
    tr_list, dmp_list, dmn_list = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i-1]
        dn = lows[i-1] - lows[i]
        dmp = up if (up > dn and up > 0) else 0.0
        dmn = dn if (dn > up and dn > 0) else 0.0
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        tr_list.append(tr); dmp_list.append(dmp); dmn_list.append(dmn)
    trn = sum(tr_list[:period]); dmpn = sum(dmp_list[:period]); dmn_n = sum(dmn_list[:period])
    dxs = []
    for i in range(period, len(tr_list)):
        trn  = trn  - (trn/period)  + tr_list[i]
        dmpn = dmpn - (dmpn/period) + dmp_list[i]
        dmn_n= dmn_n- (dmn_n/period)+ dmn_list[i]
        if trn == 0: dxs.append(0.0); continue
        dip = 100.0 * (dmpn / trn)
        din = 100.0 * (dmn_n / trn)
        dx = 100.0 * abs(dip - din) / max(1e-9, (dip + din))
        dxs.append(dx)
    if not dxs: return []
    return ema(dxs, period)

# =========================
# GARCH(1,1) Light + Simulação
# =========================
def log_returns(closes: List[float]) -> List[float]:
    out = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0:
            out.append(math.log(closes[i]/closes[i-1]))
    return out

def garch_calibrate(rets: List[float], alpha=G_ALPHA, beta=G_BETA):
    if not rets: return None
    mu = sum(rets)/len(rets)
    var = sum((x-mu)**2 for x in rets)/(len(rets)-1 if len(rets)>1 else 1)
    var = max(var, 1e-12)
    alpha = max(1e-6, min(alpha, 0.999))
    beta  = max(1e-6, min(beta,  0.999))
    if alpha + beta >= 0.9999:
        beta = 0.9998 - alpha
    omega = var * (1 - alpha - beta)
    h = var
    for r in rets:
        h = omega + alpha*(r*r) + beta*h
    return omega, alpha, beta, h

def garch_simulate_paths(p0: float, rets_hist: List[float], paths=SIM_PATHS, horizons=HORIZONS):
    calib = garch_calibrate(rets_hist)
    if not calib: return None
    omega, alpha, beta, h_last = calib
    out_prices = {h: [] for h in horizons}
    for _ in range(paths):
        h = h_last
        price = p0
        r_prev = 0.0
        for step in range(1, max(horizons)+1):
            # atualiza h com retorno anterior
            h = omega + alpha*(r_prev*r_prev) + beta*h
            sigma = math.sqrt(max(h, 1e-18))
            z = random.gauss(0.0, 1.0)
            r = sigma * z
            price = price * math.exp(r)
            if step in horizons:
                out_prices[step].append(price)
            r_prev = r
    probs = {h: (sum(1 for x in out_prices[h] if x > p0) / max(1,len(out_prices[h]))) for h in horizons}
    avgs  = {h: (sum(out_prices[h])/len(out_prices[h]) if out_prices[h] else p0) for h in horizons}
    return probs, avgs

def apply_indicator_confirmation(prob_up: float, rsi_v: Optional[float], macd_hist_v: Optional[float], adx_v: Optional[float]) -> float:
    bullish = 0; bearish = 0
    if rsi_v is not None:
        if rsi_v >= 55: bullish += 1
        elif rsi_v <= 45: bearish += 1
    if macd_hist_v is not None:
        if macd_hist_v > 0: bullish += 1
        elif macd_hist_v < 0: bearish += 1
    strong = (adx_v is not None and adx_v >= 25)
    adj = 0.0
    if bullish > bearish and strong: adj = BOOST_STRONG
    elif bullish > bearish: adj = BOOST_MED
    elif bearish > bullish and strong: adj = -BOOST_STRONG
    elif bearish > bullish: adj = -BOOST_MED
    else: adj = (BOOST_MED if (strong and prob_up>=0.5) else (-BOOST_MED if (strong and prob_up<0.5) else 0.0))
    return max(0.01, min(0.99, prob_up + adj))

def decide_from_prob(prob_up: float) -> Tuple[str, float]:
    return ("buy", prob_up) if prob_up >= 0.5 else ("sell", 1.0 - prob_up)

# =========================
# Analysis Manager
# =========================
class AnalysisManager:
    def __init__(self):
        self.symbols_default = DEFAULT_SYMBOLS[:]
        self.current_results: List[Dict[str, Any]] = []
        self.best_opportunity: Optional[Dict[str, Any]] = None
        self.analysis_time: str = ""
        self.is_analyzing = False

    def _calc_indicators(self, sym: str) -> Optional[Dict[str, Any]]:
        series = WS_FEED.get_series(sym)
        if len(series) < 60:
            return None
        closes = [x[4] for x in series]
        highs  = [x[2] for x in series]
        lows   = [x[3] for x in series]
        rsi_vals = rsi(closes, 14)
        macd_line, macd_sig, macd_hist = macd(closes, 12, 26, 9)
        adx_vals = adx(highs, lows, closes, 14)
        return {
            "price": closes[-1],
            "closes": closes,
            "rsi": rsi_vals[-1] if rsi_vals else None,
            "macd_hist": macd_hist[-1] if macd_hist else None,
            "adx": adx_vals[-1] if adx_vals else None,
            "ts": series[-1][0]
        }

    def _build_item(self, sym: str, horizon: int, direction: str,
                    p_buy: float, p_sell: float, price: float,
                    rsi_v: Optional[float], macd_hist_v: Optional[float], adx_v: Optional[float]) -> Dict[str, Any]:
        # campos compatíveis com renderTbox/renderBest
        return {
            "symbol": sym,
            "horizon": horizon,
            "direction": "buy" if direction=="buy" else "sell",
            "probability_buy": p_buy,
            "probability_sell": p_sell,
            "confidence": max(p_buy, p_sell),
            "rsi": rsi_v or 0.0,
            "adx": adx_v or 0.0,
            "liquidity_score": 1.0,  # placeholder (mantido p/ layout)
            "multi_timeframe": "neutral",
            "price": price,
            "timestamp": br_full(brazil_now()),
            "entry_time": br_full(brazil_now())
        }

    def analyze_symbols_thread(self, symbols: List[str], sims: int, _unused=None):
        self.is_analyzing = True
        try:
            results: List[Dict[str, Any]] = []
            best_global: Optional[Dict[str, Any]] = None

            for sym in symbols:
                ind = self._calc_indicators(sym)
                if not ind or len(ind["closes"]) < 240:
                    # Resultado placeholder para manter UX
                    for h in HORIZONS:
                        results.append({
                            "symbol": sym, "horizon": h, "direction": "buy",
                            "probability_buy": 0.5, "probability_sell": 0.5,
                            "confidence": 0.5, "rsi": 0.0, "adx": 0.0,
                            "liquidity_score": 0.0, "multi_timeframe": "neutral",
                            "price": ind["price"] if ind else 0.0,
                            "timestamp": br_full(brazil_now()),
                            "entry_time": "-",
                        })
                    continue

                closes = ind["closes"]
                rets = log_returns(closes)
                sim = garch_simulate_paths(closes[-1], rets, paths=sims, horizons=HORIZONS)
                if not sim:
                    continue
                probs_up, _ = sim

                # Ajuste por indicadores
                adjusted = {}
                for h in HORIZONS:
                    p_base = probs_up[h]
            # break exact 50/50 ties using last price move
            if abs(p_base - 0.5) < 1e-9:
                p_base += 0.001 if ind["closes"][-1] > ind["closes"][-2] else -0.001
            p_adj = apply_indicator_confirmation(p_base, ind["rsi"], ind["macd_hist"], ind["adx"])
                    action, prob_act = decide_from_prob(p_adj)
                    p_buy = p_adj if action=="buy" else (1.0 - prob_act)
                    p_sell = 1.0 - p_buy
                    adjusted[h] = (action, p_buy, p_sell, prob_act)

                # Criar itens T+1..T+3
                per_asset = []
                for h in HORIZONS:
                    action, p_buy, p_sell, prob_act = adjusted[h]
                    item = self._build_item(sym, h, action, p_buy, p_sell, ind["price"], ind["rsi"], ind["macd_hist"], ind["adx"])
                    per_asset.append(item)
                    results.append(item)

                # Melhor do ativo
                best_local = max(per_asset, key=lambda it: it["confidence"])
                # Candidato a melhor global
                if (best_global is None) or (best_local["confidence"] >= best_global["confidence"]):
                    best_global = best_local

            self.current_results = results
            self.best_opportunity = best_global
            self.analysis_time = br_full(brazil_now())
        finally:
            self.is_analyzing = False

manager = AnalysisManager()

# =========================
# API
# =========================
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
        sims = SIM_PATHS
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
    resp = jsonify({"ok": True, "ws": WS_FEED.enabled, "provider": REALTIME_PROVIDER, "ts": datetime.now(timezone.utc).isoformat()})
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
.kpi{background:#0e2140;border:1px solid #335f99;padding:8px;border-radius:10px}
.small{font-size:12px} .muted{color:var(--muted)} .right{float:right}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.tbox{border:1px solid #3b5c90;border-radius:12px;padding:10px;background:#0e1d34;margin-top:10px}
.tag{margin-left:6px;border:1px solid #557db8;border-radius:999px;padding:2px 7px;font-size:11px}
.ok{color:var(--ok)} .err{color:var(--err)} .line{border-top:1px dashed #3b5c90;margin:8px 0}
</style>
</head><body>
<div class="wrap">
  <div class="hline">
    <div class="clock" id="clock">--:--:--</div>
    <h1>IA Signal Pro - PREÇOS REAIS + SIMULAÇÕES</h1>
    <div class="sub">Dados: OKX • Horizons: T+1/T+2/T+3 • GARCH(1,1) 3000 sim • Prob sempre BUY/SELL</div>
    <div class="controls">
      <div class="chips" id="chips"></div>
      <div class="row">
        <button id="btnAll">Selecionar todos</button>
        <button id="btnNone">Limpar</button>
        <button id="btnAnalyze">Analisar</button>
        <span id="msg" class="muted"></span>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="title">Melhor entre todos os horizontes</div>
    <div class="card" id="bestBox"></div>
  </div>

  <div class="section">
    <div class="title">Resultados por Ativo</div>
    <div class="card">
      <div id="grid"></div>
    </div>
  </div>
</div>

<script>
const DEF = __SYMS__;
const chipsEl = document.getElementById('chips');
const btnAll = document.getElementById('btnAll');
const btnNone = document.getElementById('btnNone');
const btnAnalyze = document.getElementById('btnAnalyze');
const msgEl = document.getElementById('msg');
const gridEl = document.getElementById('grid');
const bestEl = document.getElementById('bestBox');
const clockEl = document.getElementById('clock');

function tick(){
  const d = new Date();
  const hh=String(d.getHours()).padStart(2,'0');
  const mm=String(d.getMinutes()).padStart(2,'0');
  const ss=String(d.getSeconds()).padStart(2,'0');
  clockEl.textContent = `${hh}:${mm}:${ss}`;
}
setInterval(tick,1000); tick();

function renderChips(){
  chipsEl.innerHTML = DEF.map(s => {
    const id = 'chk_'+s.replace('/','_');
    return `<label class="chip"><input type="checkbox" id="${id}" data-s="${s}" checked/> ${s}</label>`;
  }).join('');
}
renderChips();

btnAll.onclick = ()=>document.querySelectorAll('#chips input[type=checkbox]').forEach(c=>c.checked=true);
btnNone.onclick = ()=>document.querySelectorAll('#chips input[type=checkbox]').forEach(c=>c.checked=false);

function badgeDir(dir){ return dir==='buy' ? '<span class="tag ok">COMPRAR</span>' : '<span class="tag err">VENDER</span>'; }
function pct(x){ return (x*100).toFixed(1)+'%'; }

function renderBest(best, analysisTime){
  if(!best){ bestEl.innerHTML = '<div class="small">Sem oportunidade no momento.</div>'; return; }
  const html = `
    <div class="small muted">Atualizado: ${analysisTime} (Horário Brasil)</div>
    <div class="line"></div>
    <div><b>${best.symbol} T+${best.horizon}</b> ${badgeDir(best.direction)} <span class="tag">🥇 MELHOR ENTRE TODOS OS HORIZONTES</span> <span class="tag">🗲 REAL</span></div>
    <div class="kpis">
      <div class="kpi"><b>Prob Compra</b>${pct(best.probability_buy||0)}</div>
      <div class="kpi"><b>Prob Venda</b>${pct(best.probability_sell||0)}</div>
      <div class="kpi"><b>Soma</b>100.0%</div>
      <div class="kpi"><b>ADX</b>${(best.adx||0).toFixed(1)}</div>
      <div class="kpi"><b>RSI</b>${(best.rsi||0).toFixed(1)}</div>
      <div class="kpi"><b>TF</b>${best.multi_timeframe||'neutral'}</div>
    </div>
    <div class="small" style="margin-top:8px;">
      Pontuação: <span class="ok">${(best.confidence*100).toFixed(1)}%</span> · Price: <b>${Number(best.price||0).toFixed(6)}</b>
      <span class="right">Entrada: <b>${best.entry_time||'-'}</b></span>
    </div>`;
  bestEl.innerHTML = html;
}

function renderTbox(it, bestLocal){
  const isBest = bestLocal && it.symbol===bestLocal.symbol && it.horizon===bestLocal.horizon;
  const rev = ''; // mantido p/ compat
  return `
    <div class="tbox">
      <div><b>T+${it.horizon}</b> ${badgeDir(it.direction)} ${isBest?'<span class="tag">🥇 MELHOR DO ATIVO</span>':''}${rev} <span class="tag">🗲 REAL</span></div>
      <div class="small">
        Prob: <span class="${it.direction==='buy'?'ok':'err'}">${pct(it.probability_buy||0)}/${pct(it.probability_sell||0)}</span>
        · Conf: <span class="ok">${pct(it.confidence||0)}</span>
      </div>
      <div class="small">ADX: ${(it.adx||0).toFixed(1)} | RSI: ${(it.rsi||0).toFixed(1)} | TF: <b>${it.multi_timeframe||'neutral'}</b></div>
      <div class="small muted">⏱️ ${it.timestamp||'-'} · Price: ${Number(it.price||0).toFixed(6)}</div>
    </div>`;
}

function groupBySymbol(items){
  const m = new Map();
  items.forEach(it => {
    if(!m.has(it.symbol)) m.set(it.symbol, []);
    m.get(it.symbol).push(it);
  });
  return m;
}

async function analyze(){
  const sel = Array.from(document.querySelectorAll('#chips input[type=checkbox]')).filter(c=>c.checked).map(c=>c.dataset.s);
  if(sel.length===0){ alert('Selecione pelo menos um ativo'); return; }
  btnAnalyze.disabled = true; msgEl.textContent = 'Analisando...';
  try{
    const r = await fetch('/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({symbols: sel})});
    const j = await r.json();
    if(!j.success) { alert(j.error||'Falha'); }
    else { msgEl.textContent = j.message; }
  }catch(e){ alert('Falha ao iniciar análise'); }
  btnAnalyze.disabled=false;
  // poll até terminar
  let tries = 0;
  while(tries++ < 60){
    await new Promise(res=>setTimeout(res, 1000));
    const rr = await fetch('/api/results?t='+Date.now(), {cache:'no-store'});
    const jj = await rr.json();
    if(!jj.success) continue;
    renderBest(jj.best, jj.analysis_time);
    const groups = groupBySymbol(jj.results||[]);
    const html = Array.from(groups.entries()).map(([sym, arr]) => {
      // melhor do ativo
      const bestLocal = arr.reduce((a,b)=> (a && a.confidence>b.confidence? a:b), null);
      return `
      <div class="card">
        <div><b>${sym}</b></div>
        ${arr.sort((a,b)=>a.horizon-b.horizon).map(item=>renderTbox(item, bestLocal)).join('')}
      </div>`;
    }).join('');
    gridEl.innerHTML = html;
    if(!jj.is_analyzing) break;
  }
}

btnAnalyze.onclick = analyze;
</script>
</body></html>"""
    return Response(HTML.replace("__SYMS__", symbols_js), mimetype="text/html")

# =========================
# Execução
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
def compute_indicators_for(sym: str):
    ser = WS_FEED.get_series(sym)
    if len(ser) < 5:
        return None
    closes = [x[4] for x in ser]
    highs  = [x[2] for x in ser]
    lows   = [x[3] for x in ser]

    # Adaptive periods to avoid zeros
    rsi_period = min(14, max(3, len(closes)//6))
    macd_fast, macd_slow, macd_sig = 12, 26, 9
    if len(closes) < macd_slow + macd_sig:
        # shrink MACD when history is short
        macd_slow = max(6, len(closes)//3)
        macd_fast = max(4, macd_slow//2)
        macd_sig  = max(3, macd_fast//2)

    adx_period = min(14, max(5, len(closes)//6))

    rsi_vals = rsi(closes, rsi_period)
    macd_line, macd_sig_v, macd_hist = macd(closes, macd_fast, macd_slow, macd_sig)
    adx_vals = adx(highs, lows, closes, adx_period)

    rsi_v = (rsi_vals[-1] if rsi_vals else 50.0)
    macd_hist_v = (macd_hist[-1] if macd_hist else 0.0)
    adx_v = (adx_vals[-1] if adx_vals else 15.0)

    return {
        "price": closes[-1],
        "closes": closes,
        "rsi": float(rsi_v),
        "macd_hist": float(macd_hist_v),
        "adx": float(adx_v)
    }
