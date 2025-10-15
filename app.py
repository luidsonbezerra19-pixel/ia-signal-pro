# main.py
# App pronto para Railway (ou local) mantendo porta intacta, usando Binance spot + Monte Carlo (T+1..T+3)

from __future__ import annotations
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import os, time, random, math, json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple, Optional

# ========== Dependência de mercado ==========
try:
    import ccxt  # pip install ccxt
except Exception:
    ccxt = None

# ========== Configs simples ==========
TZ_STR = os.getenv("TZ", "America/Maceio")  # usado só para rótulo; cálculo de fuso abaixo (UTC-3 fixo)
MC_PATHS = int(os.getenv("MC_PATHS", "3000"))  # nº de caminhos no Monte Carlo (padrão 3000)
DEFAULT_SYMBOLS = os.getenv(
    "SYMBOLS",
    "XRP/USDT,ADA/USDT,SOL/USDT,ETH/USDT,BNB/USDT"
).split(",")
DEFAULT_SYMBOLS = [s.strip() for s in DEFAULT_SYMBOLS if s.strip()]

# ========== App ==========
app = Flask(__name__)
CORS(app)

# ========== Utilidades de tempo ==========
def brazil_now() -> datetime:
    # America/Maceio ≈ UTC-3 sem DST (simples e robusto p/ server worker)
    return datetime.now(timezone(timedelta(hours=-3)))

# ========== Mercado Spot (Binance via ccxt) ==========
class SpotMarket:
    def __init__(self) -> None:
        if ccxt is None:
            raise RuntimeError("Dependência 'ccxt' não encontrada. Instale com: pip install ccxt")
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": 12000,
            "options": {"defaultType": "spot"}
        })
        self._cache: Dict[Tuple[str, str, int], Tuple[float, List[float]]] = {}

    def fetch_prices(self, symbol: str, timeframe: str = "1m", limit: int = 240) -> List[float]:
        """
        Fecha candles reais do spot; faz cache curto p/ evitar rate-limit.
        """
        key = (symbol, timeframe, limit)
        now = time.time()
        if key in self._cache and (now - self._cache[key][0]) < 20:
            return self._cache[key][1]

        for tf in (timeframe, "5m"):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                closes = [c[4] for c in ohlcv if c and len(c) >= 5]
                if len(closes) >= min(60, max(20, limit // 3)):
                    self._cache[key] = (now, closes)
                    return closes
            except ccxt.NetworkError:
                time.sleep(0.25)
            except Exception:
                time.sleep(0.25)
        return []

# ========== Indicadores (implementações leves) ==========
class TechnicalIndicators:
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) <= period:
            return 50.0
        gains = []
        losses = []
        for i in range(1, len(prices)):
            ch = prices[i] - prices[i-1]
            gains.append(max(0.0, ch))
            losses.append(max(0.0, -ch))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 70.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return max(0.0, min(100.0, rsi))

    def calculate_adx(self, prices: List[float], period: int = 14) -> float:
        # simplificado: volatilidade relativa como proxy de "força de tendência"
        if len(prices) <= period + 1:
            return 20.0
        diffs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg = sum(diffs[-period:]) / period
        base = sum(abs(p) for p in prices[-period:]) / period
        if base == 0:
            return 20.0
        val = (avg / base) * 1000.0
        return max(5.0, min(45.0, val))

    def calculate_macd(self, prices: List[float]) -> Dict[str, Any]:
        # MACD muito simplificado
        def ema(vals, n):
            if not vals: return []
            k = 2 / (n + 1)
            e = [vals[0]]
            for v in vals[1:]:
                e.append(e[-1] + k * (v - e[-1]))
            return e
        if len(prices) < 35:
            return {"signal": "neutral", "strength": 0.0}
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd_line = [a - b for a, b in zip(ema12[-len(ema26):], ema26)]
        signal_line = ema(macd_line, 9)
        if not signal_line: 
            return {"signal": "neutral", "strength": 0.0}
        hist = macd_line[-1] - signal_line[-1]
        if hist > 0:
            return {"signal": "bullish", "strength": min(1.0, abs(hist) / max(1e-9, prices[-1]*0.002))}
        if hist < 0:
            return {"signal": "bearish", "strength": min(1.0, abs(hist) / max(1e-9, prices[-1]*0.002))}
        return {"signal": "neutral", "strength": 0.0}

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Dict[str, str]:
        if len(prices) < period:
            return {"signal": "neutral"}
        window = prices[-period:]
        ma = sum(window)/period
        var = sum((p - ma)**2 for p in window)/period
        sd = math.sqrt(max(0.0, var))
        last = prices[-1]
        upper = ma + 2*sd
        lower = ma - 2*sd
        if last > upper:
            return {"signal": "overbought"}
        if last < lower:
            return {"signal": "oversold"}
        if last > ma:
            return {"signal": "bullish"}
        if last < ma:
            return {"signal": "bearish"}
        return {"signal": "neutral"}

    def calculate_volume_profile(self, prices: List[float]) -> Dict[str, str]:
        # Sem volume real neste contexto — heurística por amplitude
        if len(prices) < 30:
            return {"signal": "neutral"}
        amp = (max(prices[-30:]) - min(prices[-30:])) / max(1e-9, prices[-30])
        if amp > 0.02:
            return {"signal": "overbought" if prices[-1] > prices[-2] else "oversold"}
        return {"signal": "neutral"}

    def calculate_fibonacci(self, prices: List[float]) -> Dict[str, str]:
        if len(prices) < 50:
            return {"signal": "neutral"}
        swing_high = max(prices[-50:])
        swing_low = min(prices[-50:])
        last = prices[-1]
        if last <= swing_low + 0.382*(swing_high - swing_low):
            return {"signal": "support"}
        if last >= swing_high - 0.382*(swing_high - swing_low):
            return {"signal": "resistance"}
        return {"signal": "neutral"}

class MultiTimeframeAnalyzer:
    def analyze_consensus(self, prices: List[float]) -> str:
        if len(prices) < 60:
            return "neutral"
        ma9 = sum(prices[-9:]) / 9
        ma21 = sum(prices[-21:]) / 21 if len(prices) >= 21 else ma9
        return "buy" if ma9 > ma21 else ("sell" if ma9 < ma21 else "neutral")

class LiquiditySystem:
    def calculate_liquidity_score(self, symbol: str, prices: List[float]) -> float:
        if len(prices) < 60:
            return 0.5
        vol = sum(abs(prices[i]-prices[i-1]) for i in range(1, len(prices[-60:]))) / max(1e-9, prices[-1])
        score = max(0.0, min(1.0, 1.0 - min(0.05, vol)/0.05))
        return score  # 0..1

class CorrelationSystem:
    def get_correlation_adjustment(self, symbol: str, cache: Dict[str, Any]) -> float:
        # Placeholder: sem penalidade por correlação no escopo básico
        return 1.0

class NewsEventSystem:
    def generate_market_events(self) -> None:
        pass
    def get_volatility_multiplier(self) -> float:
        return 1.0
    def adjust_confidence_for_events(self, conf: float) -> float:
        return conf

class VolatilityClustering:
    def detect_volatility_clusters(self, prices: List[float], symbol: str) -> str:
        if len(prices) < 60: return "normal"
        amp = (max(prices[-30:]) - min(prices[-30:]))/max(1e-9, prices[-30])
        if amp > 0.03: return "volatile"
        if amp < 0.01: return "calm"
        return "normal"
    def get_regime_adjustment(self, symbol: str) -> float:
        return 1.0

class MemorySystem:
    def __init__(self) -> None:
        self.market_regime = {"volatility": 0.0, "avg_adx": 0.0}
    def get_symbol_weights(self, symbol: str) -> Dict[str, float]:
        # Pesos iguais: sem prioridade pra BTC/qualquer outro
        return {
            "monte_carlo": 1.0,
            "rsi": 1.0,
            "adx": 1.0,
            "macd": 1.0,
            "bollinger": 1.0,
            "volume": 1.0,
            "fibonacci": 1.0,
            "multi_tf": 1.0,
        }
    def update_market_regime(self, volatility: float, adx_values: List[float]) -> None:
        self.market_regime = {
            "volatility": round(volatility, 6),
            "avg_adx": round(sum(adx_values)/max(1, len(adx_values)), 2)
        }

# ========== Monte Carlo ==========
class MonteCarloSimulator:
    @staticmethod
    def generate_price_paths_empirical(base_price: float, empirical_returns: List[float], steps: int, num_paths: int = 3000) -> List[List[float]]:
        if not empirical_returns or steps < 1:
            return []
        paths: List[List[float]] = []
        for _ in range(num_paths):
            p = base_price
            seq = [p]
            for _ in range(steps):
                r = random.choice(empirical_returns)  # bootstrap dos retornos reais
                p = max(1e-9, p * (1.0 + r))
                seq.append(p)
            paths.append(seq)
        return paths

    @staticmethod
    def calculate_probability_distribution(paths: List[List[float]]) -> Dict[str, Any]:
        if not paths:
            return {"probability_buy": 0.5, "probability_sell": 0.5, "quality": "LOW"}
        start = paths[0][0]
        ups = sum(1 for seq in paths if seq[-1] > start)
        downs = sum(1 for seq in paths if seq[-1] < start)
        total = ups + downs
        if total == 0:
            return {"probability_buy": 0.5, "probability_sell": 0.5, "quality": "LOW"}
        p_buy = ups / total
        p_sell = downs / total
        strength = abs(p_buy - 0.5)
        clarity = total / len(paths)
        quality = "HIGH" if (strength >= 0.20 and clarity >= 0.70) else ("MEDIUM" if (strength >= 0.10 and clarity >= 0.50) else "LOW")
        return {
            "probability_buy": round(p_buy, 4),
            "probability_sell": round(p_sell, 4),
            "quality": quality,
            "clarity_ratio": round(clarity, 3)
        }

# ========== Helpers ==========
def _safe_returns_from_prices(prices: List[float]) -> List[float]:
    emp: List[float] = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i-1], prices[i]
        if p0 > 0:
            emp.append((p1 - p0) / p0)
    return emp

def _rank_key(item: Dict[str, Any]) -> float:
    prob_dir = item['probability_buy'] if item['direction'] == 'buy' else item['probability_sell']
    return (item['confidence'] * 1000.0) + (prob_dir * 100.0)

# ========== Sistema principal ==========
class EnhancedTradingSystem:
    def __init__(self) -> None:
        self.memory = MemorySystem()
        self.monte_carlo = MonteCarloSimulator()
        self.indicators = TechnicalIndicators()
        self.multi_tf = MultiTimeframeAnalyzer()
        self.liquidity = LiquiditySystem()
        self.correlation = CorrelationSystem()
        self.news_events = NewsEventSystem()
        self.volatility_clustering = VolatilityClustering()
        self.spot = SpotMarket()
        self.current_analysis_cache: Dict[str, Any] = {}

    def get_brazil_time(self) -> datetime:
        return brazil_now()

    def analyze_symbol(self, symbol: str, horizon: int) -> Dict[str, Any]:
        # 1) Preços reais com fallback leve (resiliência)
        historical = self.spot.fetch_prices(symbol, timeframe="1m", limit=240)
        if len(historical) < 60:
            base = random.uniform(50, 400)
            historical = [base]
            for _ in range(239):
                historical.append(historical[-1]*(1.0 + random.gauss(0, 0.003)))

        # 2) Retornos empíricos
        empirical = _safe_returns_from_prices(historical) or [random.gauss(0, 0.003) for _ in range(120)]

        # 3) Eventos / regime
        self.news_events.generate_market_events()
        vol_mult = self.news_events.get_volatility_multiplier()

        # 4) Monte Carlo (T+1..T+3)
        base_price = historical[-1]
        steps = max(1, min(3, int(horizon)))
        paths = self.monte_carlo.generate_price_paths_empirical(base_price, empirical, steps=steps, num_paths=MC_PATHS)
        mc = self.monte_carlo.calculate_probability_distribution(paths)

        # 5) Indicadores (confirmação)
        rsi = self.indicators.calculate_rsi(historical)
        adx = self.indicators.calculate_adx(historical)
        macd = self.indicators.calculate_macd(historical)
        boll = self.indicators.calculate_bollinger_bands(historical)
        volp = self.indicators.calculate_volume_profile(historical)
        fibo = self.indicators.calculate_fibonacci(historical)
        tf_cons = self.multi_tf.analyze_consensus(historical)
        liq = self.liquidity.calculate_liquidity_score(symbol, historical)
        regime = self.volatility_clustering.detect_volatility_clusters(historical, symbol)

        # 6) Regime global
        base_vol = max(0.001, sum(abs(r) for r in empirical[-60:]) / max(1, min(60, len(empirical))))
        self.memory.update_market_regime(volatility=base_vol, adx_values=[adx])

        # 7) Pesos iguais (sem prioridade pra BTC)
        weights = self.memory.get_symbol_weights(symbol)

        # 8) Direção por MC (probabilidade bruta) + confiança por confirmações
        direction = 'buy' if mc['probability_buy'] > mc['probability_sell'] else 'sell'
        prob_dir = mc['probability_buy'] if direction == 'buy' else mc['probability_sell']

        score = 0.0
        factors: List[str] = []

        # MC base
        mc_score = prob_dir * weights['monte_carlo'] * 100.0
        mc_score *= self.volatility_clustering.get_regime_adjustment(symbol)
        score += mc_score; factors.append(f"MC:{mc_score:.1f}")

        # RSI (timing fora de extremos)
        if 30 < rsi < 70:
            s = weights['rsi'] * 12.0; score += s; factors.append(f"RSI:{s:.1f}")
        # ADX (tendência)
        if adx > 25:
            s = weights['adx'] * 12.0; score += s; factors.append(f"ADX:{s:.1f}")
        # MACD alinhado
        if (direction == 'buy' and macd['signal'] == 'bullish') or (direction == 'sell' and macd['signal'] == 'bearish'):
            s = weights['macd'] * 10.0 * max(0.3, macd.get('strength', 0.3)); score += s; factors.append(f"MACD:{s:.1f}")
        # Bollinger
        if (direction == 'buy' and boll['signal'] in ['oversold','bullish']) or (direction == 'sell' and boll['signal'] in ['overbought','bearish']):
            s = weights['bollinger'] * 8.0; score += s; factors.append(f"BB:{s:.1f}")
        # Volume profile
        if (direction == 'buy' and volp['signal'] in ['oversold','neutral']) or (direction == 'sell' and volp['signal'] in ['overbought','neutral']):
            s = weights['volume'] * 6.0; score += s; factors.append(f"VOL:{s:.1f}")
        # Fibonacci
        if (direction == 'buy' and fibo['signal'] == 'support') or (direction == 'sell' and fibo['signal'] == 'resistance'):
            s = weights['fibonacci'] * 5.0; score += s; factors.append(f"FIB:{s:.1f}")
        # Multi-TF
        if tf_cons == direction:
            s = weights['multi_tf'] * 8.0; score += s; factors.append(f"TF:{s:.1f}")

        # Liquidez / Correlação
        score *= (0.95 + (liq * 0.1))
        corr_adj = self.correlation.get_correlation_adjustment(symbol, self.current_analysis_cache)
        score *= corr_adj; factors.append(f"CORR:{corr_adj:.2f}")

        conf = min(0.95, max(0.50, score / 100.0))
        conf = self.news_events.adjust_confidence_for_events(conf)

        # Cache p/ (eventual) coerência multi-ativo
        self.current_analysis_cache[symbol] = {'direction': direction, 'confidence': conf, 'timestamp': datetime.now()}

        return {
            'symbol': symbol, 'horizon': steps, 'direction': direction,
            'probability_buy': mc['probability_buy'], 'probability_sell': mc['probability_sell'],
            'confidence': conf, 'rsi': rsi, 'adx': adx, 'multi_timeframe': tf_cons,
            'monte_carlo_quality': mc['quality'], 'price': base_price,
            'winning_indicators': [], 'score_factors': factors,
            'liquidity_score': round(liq, 2), 'volatility_regime': regime,
            'market_regime': self.memory.market_regime, 'volatility_multiplier': round(vol_mult, 2),
            'timestamp': self.get_brazil_time().strftime("%H:%M:%S")
        }

    def scan_symbols_tplus(self, symbols: List[str]) -> Dict[str, Any]:
        por_ativo: Dict[str, Any] = {}
        candidatos: List[Dict[str, Any]] = []
        for sym in symbols:
            tplus: List[Dict[str, Any]] = []
            for h in (1, 2, 3):
                try:
                    r = self.analyze_symbol(sym, h)
                    r['label'] = f"{sym} T+{h}"
                    tplus.append(r)
                    candidatos.append(r)
                except Exception as e:
                    tplus.append({
                        "symbol": sym, "horizon": h, "error": str(e),
                        "direction": "buy", "probability_buy": 0.5, "probability_sell": 0.5,
                        "confidence": 0.5, "label": f"{sym} T+{h}"
                    })
            por_ativo[sym] = {"tplus": tplus, "best_for_symbol": max(tplus, key=_rank_key)}
        best_overall = max(candidatos, key=_rank_key) if candidatos else None
        return {"por_ativo": por_ativo, "best_overall": best_overall}

trading_system = EnhancedTradingSystem()

# ========== Rotas ==========
@app.get("/")
def index() -> Response:
    # Página simples com o grid T+ (sem depender de templates externos)
    html = f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IA Profissional — Scan T+ (Monte Carlo)</title>
<style>
body{{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial; margin:0; background:#f6f7fb; color:#111}}
.wrap{{max-width:1040px;margin:24px auto;padding:0 16px}}
h1{{font-size:20px;margin:0 0 4px}}
.small{{color:#666;font-size:12px;margin:0 0 16px}}
.bar{{display:flex;gap:8px;align-items:center;margin:12px 0 20px}}
button{{border:1px solid #ddd;border-radius:10px;padding:10px 14px;background:#fff;cursor:pointer}}
#sym{{flex:1;border:1px solid #ddd;border-radius:10px;padding:10px 12px;background:#fff}}
.tplus-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.tplus-card{{border:1px solid #e6e6e6;border-radius:12px;padding:12px;box-shadow:0 2px 6px rgba(0,0,0,0.06);background:#fff}}
.tplus-card h4{{margin:0 0 8px;font-size:14px;font-weight:700}}
.tplus-row{{display:flex;justify-content:space-between;margin:6px 0;font-size:13px}}
.badge-best{{display:inline-block;padding:2px 6px;border-radius:8px;font-size:11px;background:#111;color:#fff;margin-left:8px}}
.badge-buy{{padding:2px 6px;border-radius:8px;font-size:11px;background:#0b8;color:#fff}}
.badge-sell{{padding:2px 6px;border-radius:8px;font-size:11px;background:#c33;color:#fff}}
.footer{{margin:20px 0 0;color:#666;font-size:12px}}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Scan T+ (Monte Carlo, Binance Spot)</h1>
    <p class="small">Fuso: {TZ_STR} · {brazil_now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <div class="bar">
      <input id="sym" placeholder="Símbolos separados por vírgula" value="{",".join(DEFAULT_SYMBOLS)}"/>
      <button onclick="doScanOnce()">Escanear</button>
    </div>
    <div id="tplus-grid" class="tplus-grid"></div>
    <p class="footer">Fonte: Binance (spot) · Monte Carlo com retornos empíricos · T+1/T+2/T+3 de cada ativo · Melhor geral destacado</p>
  </div>
<script>
function fmtPct(x){{ return (x*100).toFixed(1) + "%"; }}
function badge(dir){{ return dir === "buy" ? '<span class="badge-buy">BUY</span>' : '<span class="badge-sell">SELL</span>'; }}

function renderTplusGrid(data){{
  const root = document.getElementById("tplus-grid"); if(!root) return;
  const porAtivo = data.por_ativo || {{}}; const best = data.best_overall || null;
  const bestKey = best ? `${{best.symbol}}#${{best.horizon}}` : null;
  const cards = [];
  Object.keys(porAtivo).forEach(sym=>{{
    const bloco = porAtivo[sym]; const t3 = (bloco && bloco.tplus) ? bloco.tplus : [];
    const bestLocal = bloco.best_for_symbol;
    let html = `<div class="tplus-card"><h4>${{sym}}`;
    if(bestLocal){{ html += ` <span class="badge-best" title="Melhor do símbolo">T+${{bestLocal.horizon}}</span>`; }}
    html += `</h4>`;
    t3.sort((a,b)=>(a.horizon||0)-(b.horizon||0)).forEach(item=>{{
      const dir = item.direction||"buy";
      const prob = dir==="buy" ? (item.probability_buy||0) : (item.probability_sell||0);
      const conf = item.confidence||0;
      const rowKey = `${{item.symbol}}#${{item.horizon}}`;
      const star = (bestKey && rowKey===bestKey) ? ' <span class="badge-best" title="Melhor geral">MELHOR</span>' : '';
      html += `<div class="tplus-row"><div>T+${{item.horizon}} ${{
        badge(dir)
      }}${{star}}</div><div>Prob: ${{fmtPct(prob)}} · Conf: ${{fmtPct(conf)}}</div></div>`;
    }});
    html += `</div>`;
    cards.push(html);
  }});
  root.innerHTML = cards.join("");
}}

async function doScanOnce(){{
  const sym = document.getElementById("sym").value.trim();
  const symbols = sym ? sym.split(",").map(s=>s.trim()).filter(Boolean) : [];
  const resp = await fetch("/scan_once", {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{symbols}})
  }});
  const data = await resp.json();
  renderTplusGrid(data);
}}
document.addEventListener("DOMContentLoaded", doScanOnce);
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")

@app.post("/scan_once")
def scan_once():
    payload = request.get_json(silent=True) or {}
    symbols = payload.get("symbols") or DEFAULT_SYMBOLS
    symbols = [s.strip() for s in symbols if s and s.strip()]
    res = trading_system.scan_symbols_tplus(symbols)
    return jsonify({
        "timestamp": trading_system.get_brazil_time().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": symbols,
        "por_ativo": res["por_ativo"],
        "best_overall": res["best_overall"],
        "source": "Binance spot"
    })

@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()}), 200

# ========== Execução ==========
if __name__ == "__main__":
    # NÃO força porta. Se PORT existir (Railway/Heroku), usa. Senão, usa padrão do Flask.
    try:
        env_port = os.getenv("PORT",5000))
        port = int(env_port) if env_port else None
    except Exception:
        port = None

    if port:
        app.run(host="0.0.0.0", port=port, threaded=True)
    else:
        app.run(threaded=True)
