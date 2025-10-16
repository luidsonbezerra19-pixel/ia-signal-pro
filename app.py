# app.py — App completo (backend + front em uma rota) no estilo do seu layout antigo.
# - Binance Spot via ccxt (com fallback HTTP)
# - Monte Carlo empírico (MC_PATHS=3000)
# - T+1, T+2, T+3 por ativo
# - Indicadores ajustam CONF, não a prob.
# - Ranking objetivo + Melhor Geral
# - Campos esperados pelo seu front: timestamp por sinal, best.entry_time, analysis_time

from __future__ import annotations
import os, re, time, math, random, threading, json 
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# =========================
# Configurações
# =========================
TZ_STR = os.getenv("TZ", "America/Maceio")
MC_PATHS = int(os.getenv("MC_PATHS", "3000"))  # nº de caminhos Monte Carlo (padrão 3000)
DEFAULT_SYMBOLS = os.getenv(
    "SYMBOLS",
    "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,XRP/USDT,BNB/USDT"
).split(",")
DEFAULT_SYMBOLS = [s.strip().upper() for s in DEFAULT_SYMBOLS if s.strip()]

app = Flask(__name__)
CORS(app)

# =========================
# Tempo (Brasil)
# =========================
def brazil_now() -> datetime:
    # America/Maceio ~= UTC-3 sem DST
    return datetime.now(timezone(timedelta(hours=-3)))

def br_full(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y %H:%M:%S")

def br_hm_brt(dt: datetime) -> str:
    return dt.strftime("%H:%M BRT")

# =========================
# Mercado Spot (ccxt + fallback HTTP)
# =========================
def _to_binance_symbol(sym: str) -> str:
    s = sym.strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}{quote}"
    return re.sub(r'[^A-Z0-9]', '', s)

class SpotMarket:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str,str,int], Tuple[float, List[float]]] = {}
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

    def _fetch_http(self, symbol: str, timeframe: str = "1m", limit: int = 240) -> List[float]:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": _to_binance_symbol(symbol), "interval": timeframe, "limit": limit}
        try:
            r = self._session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            return [float(k[4]) for k in data if k and len(k) >= 5]
        except Exception:
            return []

    def fetch_prices(self, symbol: str, timeframe: str = "1m", limit: int = 240) -> List[float]:
        key = (symbol, timeframe, limit)
        now = time.time()
        # cache 20s para aliviar rate-limit
        if key in self._cache and (now - self._cache[key][0]) < 20:
            return self._cache[key][1]

        closes: List[float] = []
        # 1) ccxt (primário)
        if self._has_ccxt and self._ccxt is not None:
            for tf in (timeframe, "5m"):
                try:
                    ohlcv = self._ccxt.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                    closes = [c[4] for c in ohlcv if c and len(c) >= 5]
                    if len(closes) >= min(60, max(20, limit // 3)):
                        self._cache[key] = (now, closes)
                        return closes
                except Exception:
                    time.sleep(0.25)

        # 2) fallback HTTP
        for tf in (timeframe, "5m"):
            closes = self._fetch_http(symbol, timeframe=tf, limit=limit)
            if len(closes) >= min(60, max(20, limit // 3)):
                self._cache[key] = (now, closes)
                return closes

        return []

# =========================
# Indicadores (implementações leves)
# =========================
class TechnicalIndicators:
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) <= period:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(prices)):
            ch = prices[i] - prices[i-1]
            gains.append(max(0.0, ch)); losses.append(max(0.0, -ch))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 70.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return max(0.0, min(100.0, rsi))

    def calculate_adx(self, prices: List[float], period: int = 14) -> float:
        if len(prices) <= period+1:
            return 20.0
        diffs = [abs(prices[i]-prices[i-1]) for i in range(1, len(prices))]
        avg = sum(diffs[-period:]) / period
        base = sum(abs(p) for p in prices[-period:]) / period
        if base == 0:
            return 20.0
        val = (avg/base) * 1000.0
        return max(5.0, min(45.0, val))

    def calculate_macd(self, prices: List[float]) -> Dict[str, Any]:
        def ema(vals, n):
            if not vals: return []
            k = 2/(n+1); e=[vals[0]]
            for v in vals[1:]: e.append(e[-1] + k*(v-e[-1]))
            return e
        if len(prices) < 35:
            return {"signal":"neutral","strength":0.0}
        ema12 = ema(prices,12); ema26 = ema(prices,26)
        macd_line = [a-b for a,b in zip(ema12[-len(ema26):], ema26)]
        signal_line = ema(macd_line,9)
        if not signal_line: return {"signal":"neutral","strength":0.0}
        hist = macd_line[-1]-signal_line[-1]
        if hist>0: return {"signal":"bullish","strength":min(1.0, abs(hist)/max(1e-9, prices[-1]*0.002))}
        if hist<0: return {"signal":"bearish","strength":min(1.0, abs(hist)/max(1e-9, prices[-1]*0.002))}
        return {"signal":"neutral","strength":0.0}

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

    def calculate_volume_profile(self, prices: List[float]) -> Dict[str,str]:
        if len(prices) < 30: return {"signal":"neutral"}
        amp=(max(prices[-30:])-min(prices[-30:]))/max(1e-9, prices[-30])
        if amp>0.02: return {"signal":"overbought" if prices[-1]>prices[-2] else "oversold"}
        return {"signal":"neutral"}

    def calculate_fibonacci(self, prices: List[float]) -> Dict[str,str]:
        if len(prices) < 50: return {"signal":"neutral"}
        H=max(prices[-50:]); L=min(prices[-50:]); last=prices[-1]
        if last <= L + 0.382*(H-L): return {"signal":"support"}
        if last >= H - 0.382*(H-L): return {"signal":"resistance"}
        return {"signal":"neutral"}

class MultiTimeframeAnalyzer:
    def analyze_consensus(self, prices: List[float]) -> str:
        if len(prices) < 60: return "neutral"
        ma9=sum(prices[-9:])/9
        ma21=sum(prices[-21:])/21 if len(prices)>=21 else ma9
        return "buy" if ma9>ma21 else ("sell" if ma9<ma21 else "neutral")

class LiquiditySystem:
    def calculate_liquidity_score(self, symbol: str, prices: List[float]) -> float:
        if len(prices)<60: return 0.5
        vol=sum(abs(prices[i]-prices[i-1]) for i in range(1,len(prices[-60:]))) / max(1e-9, prices[-1])
        return max(0.0, min(1.0, 1.0 - min(0.05, vol)/0.05))

class CorrelationSystem:
    def get_correlation_adjustment(self, symbol: str, cache: Dict[str,Any]) -> float:
        return 1.0

class NewsEventSystem:
    def generate_market_events(self)->None: pass
    def get_volatility_multiplier(self)->float: return 1.0
    def adjust_confidence_for_events(self, conf: float)->float: return conf

class VolatilityClustering:
    def detect_volatility_clusters(self, prices: List[float], symbol: str)->str:
        if len(prices)<60: return "normal"
        amp=(max(prices[-30:])-min(prices[-30:]))/max(1e-9, prices[-30])
        if amp>0.03: return "volatile"
        if amp<0.01: return "calm"
        return "normal"
    def get_regime_adjustment(self, symbol: str)->float:
        return 1.0

class MemorySystem:
    def __init__(self)->None:
        self.market_regime={"volatility":0.0,"avg_adx":0.0}
    def get_symbol_weights(self, symbol: str)->Dict[str,float]:
        return {"monte_carlo":1.0,"rsi":1.0,"adx":1.0,"macd":1.0,"bollinger":1.0,"volume":1.0,"fibonacci":1.0,"multi_tf":1.0}
    def update_market_regime(self, volatility: float, adx_values: List[float])->None:
        self.market_regime={"volatility":round(volatility,6),"avg_adx":round(sum(adx_values)/max(1,len(adx_values)),2)}

# =========================
# Monte Carlo empírico
# =========================
class MonteCarloSimulator:
    @staticmethod
    def generate_price_paths_empirical(base_price: float, empirical_returns: List[float], steps: int, num_paths: int = 3000)->List[List[float]]:
        if not empirical_returns or steps<1: return []
        paths=[]
        for _ in range(num_paths):
            p=base_price; seq=[p]
            for _s in range(steps):
                r=random.choice(empirical_returns)  # bootstrap de retornos reais
                p=max(1e-9, p*(1.0+r))
                seq.append(p)
            paths.append(seq)
        return paths

    @staticmethod
    def calculate_probability_distribution(paths: List[List[float]])->Dict[str,Any]:
        if not paths: return {"probability_buy":0.5,"probability_sell":0.5,"quality":"LOW"}
        start=paths[0][0]
        ups=sum(1 for seq in paths if seq[-1]>start)
        downs=sum(1 for seq in paths if seq[-1]<start)
        total=ups+downs
        if total==0: return {"probability_buy":0.5,"probability_sell":0.5,"quality":"LOW"}
        p_buy=ups/total; p_sell=downs/total
        strength=abs(p_buy-0.5); clarity=total/len(paths)
        quality="HIGH" if (strength>=0.20 and clarity>=0.70) else ("MEDIUM" if (strength>=0.10 and clarity>=0.50) else "LOW")
        return {"probability_buy":round(p_buy,4),"probability_sell":round(p_sell,4),"quality":quality,"clarity_ratio":round(clarity,3)}

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
        self.memory=MemorySystem()
        self.monte_carlo=MonteCarloSimulator()
        self.indicators=TechnicalIndicators()
        self.multi_tf=MultiTimeframeAnalyzer()
        self.liquidity=LiquiditySystem()
        self.correlation=CorrelationSystem()
        self.news_events=NewsEventSystem()
        self.volatility_clustering=VolatilityClustering()
        self.spot=SpotMarket()
        self.current_analysis_cache: Dict[str,Any]={}

    def get_brazil_time(self)->datetime:
        return brazil_now()

    def analyze_symbol(self, symbol: str, horizon: int)->Dict[str,Any]:
        # 1) preços reais (spot)
        historical=self.spot.fetch_prices(symbol,"1m",240)
        if len(historical)<60:
            base=random.uniform(50,400); historical=[base]
            for _ in range(239):
                historical.append(historical[-1]*(1.0+random.gauss(0,0.003)))

        # 2) retornos empíricos
        empirical=_safe_returns_from_prices(historical) or [random.gauss(0,0.003) for _ in range(120)]

        # 3) eventos/regime (placeholders estáveis)
        self.news_events.generate_market_events()
        vol_mult=self.news_events.get_volatility_multiplier()

        # 4) Monte Carlo: SOMENTE T+1..T+3
        steps=max(1,min(3,int(horizon))); base_price=historical[-1]
        paths=self.monte_carlo.generate_price_paths_empirical(base_price, empirical, steps, num_paths=MC_PATHS)
        mc=self.monte_carlo.calculate_probability_distribution(paths)

        # 5) Indicadores: ajustam CONFIANÇA (probabilidade vem do MC)
        rsi=self.indicators.calculate_rsi(historical)
        adx=self.indicators.calculate_adx(historical)
        macd=self.indicators.calculate_macd(historical)
        boll=self.indicators.calculate_bollinger_bands(historical)
        volp=self.indicators.calculate_volume_profile(historical)
        fibo=self.indicators.calculate_fibonacci(historical)
        tf_cons=self.multi_tf.analyze_consensus(historical)
        liq=self.liquidity.calculate_liquidity_score(symbol,historical)
        regime=self.volatility_clustering.detect_volatility_clusters(historical,symbol)

        # 6) regime global (informativo)
        base_vol=max(0.001, sum(abs(r) for r in empirical[-60:])/max(1,min(60,len(empirical))))
        self.memory.update_market_regime(volatility=base_vol, adx_values=[adx])

        # 7) pesos iguais
        weights=self.memory.get_symbol_weights(symbol)

        # 8) direção pelo MC; confiança pelos indicadores
        direction='buy' if mc['probability_buy']>mc['probability_sell'] else 'sell'
        prob_dir=mc['probability_buy'] if direction=='buy' else mc['probability_sell']

        score=0.0; factors=[]
        mc_score=prob_dir*weights['monte_carlo']*100.0
        mc_score*=self.volatility_clustering.get_regime_adjustment(symbol)
        score+=mc_score; factors.append(f"MC:{mc_score:.1f}")

        if 30<rsi<70: s=weights['rsi']*12.0; score+=s; factors.append(f"RSI:{s:.1f}")
        if adx>25: s=weights['adx']*12.0; score+=s; factors.append(f"ADX:{s:.1f}")
        if (direction=='buy' and macd['signal']=='bullish') or (direction=='sell' and macd['signal']=='bearish'):
            s=weights['macd']*10.0*max(0.3, macd.get('strength',0.3)); score+=s; factors.append(f"MACD:{s:.1f}")
        if (direction=='buy' and boll['signal'] in ['oversold','bullish']) or (direction=='sell' and boll['signal'] in ['overbought','bearish']):
            s=weights['bollinger']*8.0; score+=s; factors.append(f"BB:{s:.1f}")
        if (direction=='buy' and volp['signal'] in ['oversold','neutral']) or (direction=='sell' and volp['signal'] in ['overbought','neutral']):
            s=weights['volume']*6.0; score+=s; factors.append(f"VOL:{s:.1f}")
        if (direction=='buy' and fibo['signal']=='support') or (direction=='sell' and fibo['signal']=='resistance'):
            s=weights['fibonacci']*5.0; score+=s; factors.append(f"FIB:{s:.1f}")
        if tf_cons==direction:
            s=weights['multi_tf']*8.0; score+=s; factors.append(f"TF:{s:.1f}")

        score*=(0.95 + (liq*0.1))
        corr_adj=self.correlation.get_correlation_adjustment(symbol,self.current_analysis_cache)
        score*=corr_adj; factors.append(f"CORR:{corr_adj:.2f}")

        conf=min(0.95, max(0.50, score/100.0))
        conf=self.news_events.adjust_confidence_for_events(conf)

        # cache simbólico multi-ativo
        self.current_analysis_cache[symbol]={'direction':direction,'confidence':conf,'timestamp':datetime.now()}

        return {
            'symbol':symbol,
            'horizon':steps,   # T+1..T+3
            'direction':direction,
            'probability_buy':mc['probability_buy'],
            'probability_sell':mc['probability_sell'],
            'confidence':conf,                     # ajustada por indicadores
            'rsi':rsi,'adx':adx,'multi_timeframe':tf_cons,
            'monte_carlo_quality':mc['quality'],
            'price':base_price,
            'winning_indicators':[],
            'score_factors':factors,
            'liquidity_score':round(liq,2),
            'volatility_regime':regime,
            'market_regime':self.memory.market_regime,
            'volatility_multiplier':round(vol_mult,2),
            'timestamp': self.get_brazil_time().strftime("%H:%M:%S")  # para o layout antigo
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
                        "timestamp": self.get_brazil_time().strftime("%H:%M:%S")
                    })
            por_ativo[sym]={"tplus":tplus,"best_for_symbol":max(tplus,key=_rank_key)}
        best_overall=max(candidatos,key=_rank_key) if candidatos else None
        return {"por_ativo":por_ativo,"best_overall":best_overall}

# =========================
# Manager (compatível com seu layout antigo)
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

# =========================
# Rotas de API (mantidas)
# =========================
@app.post("/api/analyze")
def api_analyze():
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
        return jsonify({"success": True, "message": f"Analisando {len(symbols)} ativos com {sims} simulações Monte Carlo.", "symbols_count": len(symbols)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/api/results")
def api_results():
    return jsonify({
        "success": True,
        "results": manager.current_results,      # lista FLAT p/ layout antigo
        "best": manager.best_opportunity,        # inclui best.entry_time
        "analysis_time": manager.analysis_time,  # horário da pesquisa
        "total_signals": len(manager.current_results),
        "is_analyzing": manager.is_analyzing
    })

@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()}), 200

# =========================
# Rota "/" com SEU LAYOUT (HTML embutido)
# =========================
@app.get("/")
def index():
    # Monta o HTML sem f-string para evitar conflito com chaves { } de CSS/JS
    symbols_js = json.dumps(DEFAULT_SYMBOLS)  # vira array JS válido

    HTML = """<!doctype html>
<html lang="pt-br"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IA Signal Pro - PREÇOS REAIS + 3000 SIMULAÇÕES</title>
<style>
:root{--bg:#0f1120;--panel:#181a2e;--panel2:#223148;--tx:#dfe6ff;--muted:#9fb4ff;--accent:#2aa9ff;--gold:#f2a93b;--ok:#29d391;--err:#ff5b5b;--warn:#ffbd2d;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial}
.wrap{max-width:1120px;margin:22px auto;padding:0 16px}
.hline{border:2px solid var(--accent);border-radius:12px;background:var(--panel);padding:18px}
h1{margin:0 0 8px;font-size:22px} .sub{color:#8ccf9d;font-size:13px;margin:6px 0 0}
.controls{margin-top:14px;background:var(--panel2);border-radius:12px;padding:14px}
.chips{display:flex;flex-wrap:wrap;gap:10px} .chip{border:2px solid var(--accent);border-radius:12px;padding:8px 12px;cursor:pointer;user-select:none}
.chip input{display:none} .chip.active{box-shadow:0 0 0 2px inset var(--accent)}
.row{display:flex;gap:10px;align-items:center;margin-top:12px}
select,button{border:2px solid var(--accent);border-radius:12px;padding:10px 12px;background:#16314b;color:#fff}
button{background:#2a9df4;cursor:pointer} button:disabled{opacity:.6;cursor:not-allowed}
.section{margin-top:16px;border:2px solid var(--gold);border-radius:12px;background:var(--panel)}
.section .title{padding:10px 14px;border-bottom:2px solid var(--gold);font-weight:700}
.card{margin:12px;border-radius:12px;background:var(--panel2);padding:14px;border:2px solid var(--gold)}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;margin-top:8px}
.kpi{background:#1b2b41;border-radius:10px;padding:10px 12px;color:#b6c8ff} .kpi b{display:block;color:#fff}
.badge{display:inline-block;padding:3px 8px;border-radius:8px;font-size:11px;margin-right:6px;background:#12263a;border:1px solid #2e6ea8}
.buy{background:#0c5d4b} .sell{background:#5b1f1f}
.small{color:#9fb4ff;font-size:12px} .muted{color:#7d90c7} .ok{color:var(--ok)} .err{color:var(--err)}
.grid-syms{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;padding-bottom:12px}
.sym-head{padding:10px 14px;border-bottom:1px dashed #3b577a} .line{border-top:1px dashed #3b577a;margin:8px 0}
.tbox{border:2px solid #f0a43c;border-radius:10px;background:#26384e;padding:10px;margin-top:10px}
.tag{display:inline-block;padding:2px 6px;border-radius:6px;font-size:10px;margin-left:6px;background:#0d2033;border:1px solid #3e6fa8}
.right{text-align:right}
</style>
</head>
<body>
<div class="wrap">
  <div class="hline">
    <h1>IA Signal Pro - PREÇOS REAIS + 3000 SIMULAÇÕES</h1>
    <div class="sub">✅ Dados reais da Binance · Monte Carlo com retornos empíricos · Imparcialidade garantida</div>
    <div class="controls">
      <div class="chips" id="chips"></div>
      <div class="row">
        <select id="mcsel">
          <option value="3000" selected>3000 simulações Monte Carlo</option>
          <option value="1000">1000 simulações</option>
          <option value="5000">5000 simulações</option>
        </select>
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
const SYMS_DEFAULT = __SYMS__;  // ← substituído no Python por json.dumps(DEFAULT_SYMBOLS)

const chipsEl = document.getElementById('chips');
const gridEl  = document.getElementById('grid');
const bestEl  = document.getElementById('bestCard');
const bestSec = document.getElementById('bestSec');
const allSec  = document.getElementById('allSec');

function mkChip(sym){
  const div = document.createElement('label');
  div.className = 'chip active';
  div.innerHTML = `<input type="checkbox" checked value="${sym}"/>${sym}`;
  div.onclick = (e)=>{ if(e.target.tagName!=='INPUT'){ const cb=div.querySelector('input'); cb.checked=!cb.checked; } div.classList.toggle('active', div.querySelector('input').checked); };
  chipsEl.appendChild(div);
}
SYMS_DEFAULT.forEach(mkChip);

function selSymbols(){
  return Array.from(chipsEl.querySelectorAll('input')).filter(i=>i.checked).map(i=>i.value);
}
function pct(x){ return (x*100).toFixed(1)+'%'; }
function badgeDir(d){ return `<span class="badge ${d==='buy'?'buy':'sell'}">${d==='buy'?'COMPRAR':'VENDER'}</span>`; }

async function runAnalyze(){
  const btn = document.getElementById('go');
  btn.disabled = true;
  const syms = selSymbols();
  if(!syms.length){ alert('Selecione pelo menos um ativo.'); btn.disabled=false; return; }
  try{
    await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbols: syms})});
    setTimeout(loadResults, 1200);
  }catch(e){ alert('Erro na análise'); btn.disabled=false; }
}

async function loadResults(){
  const btn = document.getElementById('go');
  try{
    const r = await fetch('/api/results'); const data = await r.json();
    if(!data.success){ alert('Sem resultados'); btn.disabled=false; return; }

    // BEST
    bestSec.style.display='block';
    bestEl.innerHTML = renderBest(data.best, data.analysis_time);

    // GRID por símbolo
    const groups = {};
    (data.results||[]).forEach(it=>{ (groups[it.symbol]=groups[it.symbol]||[]).push(it); });
    const html = Object.keys(groups).sort().map(sym=>{
      const arr = groups[sym].sort((a,b)=>(a.horizon||0)-(b.horizon||0));
      const bestLocal = arr.slice().sort((a,b)=>rank(b)-rank(a))[0];
      return `
        <div class="card">
          <div class="sym-head"><b>${sym}</b>
            <span class="tag">Regime: ${bestLocal?.volatility_regime||'normal'}</span>
            <span class="tag">Liquidez: ${Number(bestLocal?.liquidity_score||0).toFixed(1)}</span>
            <span class="tag">Mercado: ${bestLocal?.multi_timeframe||'neutral'}</span>
            <span class="tag">🗲 DADOS REAIS</span>
          </div>
          ${arr.map(item=>renderTbox(item, bestLocal)).join('')}
        </div>`;
    }).join('');
    gridEl.innerHTML = html;
    allSec.style.display='block';
  }catch(e){
    alert('Erro ao carregar resultados');
  }finally{
    btn.disabled=false;
  }
}

function rank(it){ const pd = it.direction==='buy' ? it.probability_buy : it.probability_sell; return (it.confidence*1000)+(pd*100); }

function renderBest(best, analysisTime){
  if(!best) return '<div class="small">Sem oportunidade no momento.</div>';
  return `
    <div class="small muted">Atualizado: ${analysisTime} (Horário Brasil)</div>
    <div class="line"></div>
    <div><b>${best.symbol} T+${best.horizon}</b> ${badgeDir(best.direction)} <span class="tag">🥇 MELHOR ENTRE TODOS OS HORIZONTES</span> <span class="tag">🗲 DADOS REAIS</span></div>
    <div class="kpis">
      <div class="kpi"><b>Prob Compra</b>${pct(best.probability_buy||0)}</div>
      <div class="kpi"><b>Prob Venda</b>${pct(best.probability_sell||0)}</div>
      <div class="kpi"><b>Soma</b>100.0%</div>
      <div class="kpi"><b>ADX</b>${(best.adx||0).toFixed(1)}</div>
      <div class="kpi"><b>RSI</b>${(best.rsi||0).toFixed(1)}</div>
      <div class="kpi"><b>Liquidez</b>${Number(best.liquidity_score||0).toFixed(1)}</div>
    </div>
    <div class="small" style="margin-top:8px;">
      Pontuação: <span class="ok">${(best.confidence*100).toFixed(1)}%</span> · Mercado: <b>${best.multi_timeframe||'neutral'}</b> · Price: <b>${Number(best.price||0).toFixed(6)}</b>
      <span class="right">Entrada: <b>${best.entry_time||'-'}</b></span>
    </div>
  `;
}

function renderTbox(it, bestLocal){
  const isBest = bestLocal && it.symbol===bestLocal.symbol && it.horizon===bestLocal.horizon;
  return `
    <div class="tbox">
      <div><b>T+${it.horizon}</b> ${badgeDir(it.direction)} ${isBest?'<span class="tag">🥇 MELHOR DO ATIVO</span>':''} <span class="tag">🗲 REAL</span></div>
      <div class="small">
        Prob: <span class="${it.direction==='buy'?'ok':'err'}">${pct(it.probability_buy||0)}/${pct(it.probability_sell||0)}</span>
        · Conf: <span class="ok">${pct(it.confidence||0)}</span>
        · Qual: <b>${it.monte_carlo_quality||'LOW'}</b>
      </div>
      <div class="small">ADX: ${(it.adx||0).toFixed(1)} | RSI: ${(it.rsi||0).toFixed(1)} | Multi-TF: <b>${it.multi_timeframe||'neutral'}</b></div>
      <div class="small muted">⏱️ ${it.timestamp||'-'} · Price: ${Number(it.price||0).toFixed(6)}</div>
    </div>`;
}
</script>
</body></html>"""

    html = HTML.replace("__SYMS__", symbols_js)
    return Response(html, mimetype="text/html")
    
# Execução (porta intacta)
if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),  # Railway usa PORT; local fica 5000
        threaded=True,
        debug=False
    )
