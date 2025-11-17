# ==============================================================
#  IA SIGNAL PRO - GPT-5.1 COMO CÉREBRO CENTRAL
# ==============================================================

import os
import json
import datetime
import requests
import numpy as np
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

# ==============================================================
#  CONFIGURAÇÃO OPENAI GPT-5.1
# ==============================================================

client = OpenAI(api_key="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")   # <-- COLOQUE SUA API KEY AQUI

# ==============================================================
#  ANALISADOR SUPER INTELIGENTE COM GPT-5.1 - CÉREBRO CENTRAL
# ==============================================================

class CentralIntelligenceAnalyzer:

    def calculate_comprehensive_analysis(self, symbol, timeframe, candles):
        """Calcula TODOS os indicadores para a IA analisar"""
        
        # Extrair dados dos candles
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])
        opens = np.array([c['open'] for c in candles])
        
        current_price = closes[-1]
        price_change_24h = ((closes[-1] - closes[0]) / closes[0]) * 100
        
        # ========== INDICADORES DE TENDÊNCIA ==========
        ema_20 = EMAIndicator(close=closes, window=20).ema_indicator()
        ema_50 = EMAIndicator(close=closes, window=50).ema_indicator()
        ema_100 = EMAIndicator(close=closes, window=100).ema_indicator()
        ema_200 = EMAIndicator(close=closes, window=200).ema_indicator()
        
        macd = MACD(close=closes)
        macd_line = macd.macd()
        macd_signal = macd.macd_signal()
        macd_histogram = macd.macd_diff()
        
        adx = ADXIndicator(high=highs, low=lows, close=closes, window=14)
        adx_value = adx.adx()
        adx_plus = adx.adx_pos()
        adx_minus = adx.adx_neg()
        
        # ========== INDICADORES DE MOMENTUM ==========
        rsi = RSIIndicator(close=closes, window=14).rsi()
        stoch = StochasticOscillator(high=highs, low=lows, close=closes, window=14, smooth_window=3)
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        
        williams = self.calculate_williams_r(highs, lows, closes)
        cci = self.calculate_cci(highs, lows, closes)
        momentum = self.calculate_momentum(closes)
        
        # ========== INDICADORES DE VOLATILIDADE ==========
        bollinger = BollingerBands(close=closes, window=20, window_dev=2)
        bb_upper = bollinger.bollinger_hband()
        bb_lower = bollinger.bollinger_lband()
        bb_middle = bollinger.bollinger_mavg()
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        atr = AverageTrueRange(high=highs, low=lows, close=closes, window=14).average_true_range()
        
        # ========== INDICADORES DE VOLUME ==========
        vwap = VolumeWeightedAveragePrice(high=highs, low=lows, close=closes, volume=volumes, window=20).volume_weighted_average_price()
        obv = OnBalanceVolumeIndicator(close=closes, volume=volumes).on_balance_volume()
        volume_sma = ta.volume.VolumeSMAIndicator(volume=volumes, window=20).volume_sma()
        volume_ratio = volumes[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
        
        # ========== ANÁLISE DE PRICE ACTION ==========
        support_resistance = self.find_support_resistance(highs, lows, closes)
        market_structure = self.analyze_market_structure(highs, lows, closes)
        trend_strength = self.calculate_trend_strength(closes)
        volatility = self.calculate_volatility(closes)
        
        # ========== PADRÕES DE CANDLES ==========
        recent_candles_analysis = self.analyze_recent_candles(candles[-10:])
        key_levels = self.find_key_levels(closes, highs, lows)
        
        # Compilar todos os dados para a IA
        comprehensive_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            
            # TENDÊNCIA
            'ema_20': ema_20.iloc[-1],
            'ema_50': ema_50.iloc[-1],
            'ema_100': ema_100.iloc[-1],
            'ema_200': ema_200.iloc[-1],
            'ema_alignment': self.get_ema_alignment(ema_20.iloc[-1], ema_50.iloc[-1], ema_100.iloc[-1], ema_200.iloc[-1]),
            
            'macd_line': macd_line.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'macd_histogram': macd_histogram.iloc[-1],
            'macd_trend': 'BULLISH' if macd_line.iloc[-1] > macd_signal.iloc[-1] else 'BEARISH',
            
            'adx': adx_value.iloc[-1],
            'adx_plus': adx_plus.iloc[-1],
            'adx_minus': adx_minus.iloc[-1],
            'adx_trend': 'BULLISH' if adx_plus.iloc[-1] > adx_minus.iloc[-1] else 'BEARISH',
            
            # MOMENTUM
            'rsi': rsi.iloc[-1],
            'rsi_trend': self.get_rsi_trend(rsi),
            'stoch_k': stoch_k.iloc[-1],
            'stoch_d': stoch_d.iloc[-1],
            'stoch_trend': 'BULLISH' if stoch_k.iloc[-1] > stoch_d.iloc[-1] else 'BEARISH',
            'williams_r': williams[-1] if len(williams) > 0 else 0,
            'cci': cci[-1] if len(cci) > 0 else 0,
            'momentum': momentum[-1] if len(momentum) > 0 else 0,
            
            # VOLATILIDADE
            'bb_upper': bb_upper.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'bb_middle': bb_middle.iloc[-1],
            'bb_position': (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5,
            'bb_width': bb_width.iloc[-1],
            'atr': atr.iloc[-1],
            'atr_percentage': (atr.iloc[-1] / current_price) * 100,
            
            # VOLUME
            'volume': volumes[-1],
            'volume_ratio': volume_ratio,
            'volume_trend': self.get_volume_trend(volumes),
            'vwap': vwap.iloc[-1],
            'vwap_position': 'ABOVE' if current_price > vwap.iloc[-1] else 'BELOW',
            'obv': obv.iloc[-1],
            'obv_trend': self.get_obv_trend(obv),
            
            # PRICE ACTION
            'support_levels': support_resistance['supports'][:3],
            'resistance_levels': support_resistance['resistances'][:3],
            'market_structure': market_structure,
            'trend_strength': trend_strength,
            'volatility_percentage': volatility,
            
            # CANDLES RECENTES
            'recent_candles': recent_candles_analysis,
            'key_levels': key_levels,
            
            # CONVERGÊNCIA DE SINAIS
            'bullish_signals': self.count_bullish_signals(locals()),
            'bearish_signals': self.count_bearish_signals(locals())
        }
        
        return comprehensive_data

    def calculate_williams_r(self, highs, lows, closes, period=14):
        """Calcula Williams %R"""
        if len(highs) < period:
            return [0]
        
        williams = []
        for i in range(period, len(highs)):
            highest_high = max(highs[i-period:i])
            lowest_low = min(lows[i-period:i])
            current_close = closes[i]
            
            if highest_high != lowest_low:
                wr = (highest_high - current_close) / (highest_high - lowest_low) * -100
            else:
                wr = 0
            williams.append(wr)
        
        return williams

    def calculate_cci(self, highs, lows, closes, period=20):
        """Calcula Commodity Channel Index"""
        if len(closes) < period:
            return [0]
        
        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
        cci_values = []
        
        for i in range(period, len(typical_prices)):
            sma = np.mean(typical_prices[i-period:i])
            mean_dev = np.mean([abs(tp - sma) for tp in typical_prices[i-period:i]])
            
            if mean_dev != 0:
                cci = (typical_prices[i] - sma) / (0.015 * mean_dev)
            else:
                cci = 0
            cci_values.append(cci)
        
        return cci_values

    def calculate_momentum(self, closes, period=10):
        """Calcula Momentum"""
        if len(closes) < period:
            return [0]
        
        momentum = []
        for i in range(period, len(closes)):
            mom = ((closes[i] - closes[i-period]) / closes[i-period]) * 100
            momentum.append(mom)
        
        return momentum

    def find_support_resistance(self, highs, lows, closes, window=20):
        """Encontra níveis de suporte e resistência"""
        supports = []
        resistances = []
        
        for i in range(window, len(highs) - window):
            # Resistências
            if highs[i] == max(highs[i-window:i+window]):
                resistances.append(highs[i])
            # Suportes
            if lows[i] == min(lows[i-window:i+window]):
                supports.append(lows[i])
        
        # Remover duplicatas próximas
        def filter_levels(levels, threshold_percent=0.001):
            filtered = []
            for level in sorted(levels):
                if not any(abs(level - existing) < (np.mean(closes) * threshold_percent) for existing in filtered):
                    filtered.append(level)
            return filtered
        
        return {
            'supports': filter_levels(supports)[-3:],
            'resistances': filter_levels(resistances)[-3:]
        }

    def analyze_market_structure(self, highs, lows, closes):
        """Analisa estrutura de mercado"""
        if len(highs) < 10:
            return "NEUTRAL"
        
        # Higher Highs / Higher Lows
        recent_highs = highs[-5:]
        recent_lows = lows[-5:]
        
        hh = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        lh = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
        ll = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        if hh and hl:
            return "UPTREND"
        elif lh and ll:
            return "DOWNTREND"
        else:
            return "RANGING"

    def calculate_trend_strength(self, closes, period=20):
        """Calcula força da tendência (0-100)"""
        if len(closes) < period:
            return 50
        
        returns = np.diff(closes[-period:]) / closes[-period-1:-1]
        trend_strength = min(100, max(0, abs(np.mean(returns)) * 1000))
        return trend_strength

    def calculate_volatility(self, closes, period=20):
        """Calcula volatilidade percentual"""
        returns = np.diff(closes) / closes[:-1]
        recent_volatility = np.std(returns[-period:]) * 100
        return recent_volatility

    def analyze_recent_candles(self, candles):
        """Analisa candles recentes"""
        if len(candles) < 2:
            return "INSUFFICIENT_DATA"
        
        bullish_count = sum(1 for c in candles if c['close'] > c['open'])
        bearish_count = len(candles) - bullish_count
        
        if bullish_count >= 8:
            return "VERY_STRONG_BULLISH"
        elif bearish_count >= 8:
            return "VERY_STRONG_BEARISH"
        elif bullish_count >= 6:
            return "STRONG_BULLISH"
        elif bearish_count >= 6:
            return "STRONG_BEARISH"
        elif bullish_count > bearish_count:
            return "BULLISH"
        else:
            return "BEARISH"

    def find_key_levels(self, closes, highs, lows):
        """Encontra níveis-chave"""
        return {
            'recent_high': max(highs[-20:]),
            'recent_low': min(lows[-20:]),
            'pivot_point': (max(highs[-1]) + min(lows[-1]) + closes[-1]) / 3,
            'psychological_levels': self.find_psychological_levels(closes[-1])
        }

    def find_psychological_levels(self, price):
        """Encontra níveis psicológicos"""
        base = round(price, -int(np.log10(price)) + 1)
        return [base * (1 + i * 0.01) for i in range(-2, 3)]

    def get_ema_alignment(self, ema20, ema50, ema100, ema200):
        """Verifica alinhamento das EMAs"""
        if ema20 > ema50 > ema100 > ema200:
            return "STRONG_BULLISH"
        elif ema20 < ema50 < ema100 < ema200:
            return "STRONG_BEARISH"
        elif ema20 > ema50:
            return "BULLISH"
        else:
            return "BEARISH"

    def get_rsi_trend(self, rsi):
        """Analisa tendência do RSI"""
        if rsi.iloc[-1] > 70:
            return "OVERBOUGHT"
        elif rsi.iloc[-1] < 30:
            return "OVERSOLD"
        elif rsi.iloc[-1] > rsi.iloc[-5]:
            return "BULLISH"
        else:
            return "BEARISH"

    def get_volume_trend(self, volumes):
        """Analisa tendência do volume"""
        if len(volumes) < 5:
            return "NEUTRAL"
        
        recent_avg = np.mean(volumes[-5:])
        previous_avg = np.mean(volumes[-10:-5])
        
        if recent_avg > previous_avg * 1.2:
            return "INCREASING"
        elif recent_avg < previous_avg * 0.8:
            return "DECREASING"
        else:
            return "STABLE"

    def get_obv_trend(self, obv):
        """Analisa tendência do OBV"""
        if len(obv) < 5:
            return "NEUTRAL"
        
        if obv.iloc[-1] > obv.iloc[-5]:
            return "BULLISH"
        else:
            return "BEARISH"

    def count_bullish_signals(self, data_dict):
        """Conta sinais de alta"""
        bullish_count = 0
        
        if data_dict['ema_alignment'] in ['BULLISH', 'STRONG_BULLISH']:
            bullish_count += 2
        if data_dict['macd_trend'] == 'BULLISH':
            bullish_count += 1
        if data_dict['adx_trend'] == 'BULLISH':
            bullish_count += 1
        if data_dict['rsi_trend'] in ['BULLISH', 'OVERSOLD']:
            bullish_count += 1
        if data_dict['stoch_trend'] == 'BULLISH':
            bullish_count += 1
        if data_dict['vwap_position'] == 'ABOVE':
            bullish_count += 1
        if data_dict['obv_trend'] == 'BULLISH':
            bullish_count += 1
        if 'BULLISH' in data_dict['recent_candles']:
            bullish_count += 1
            
        return bullish_count

    def count_bearish_signals(self, data_dict):
        """Conta sinais de baixa"""
        bearish_count = 0
        
        if data_dict['ema_alignment'] in ['BEARISH', 'STRONG_BEARISH']:
            bearish_count += 2
        if data_dict['macd_trend'] == 'BEARISH':
            bearish_count += 1
        if data_dict['adx_trend'] == 'BEARISH':
            bearish_count += 1
        if data_dict['rsi_trend'] in ['BEARISH', 'OVERBOUGHT']:
            bearish_count += 1
        if data_dict['stoch_trend'] == 'BEARISH':
            bearish_count += 1
        if data_dict['vwap_position'] == 'BELOW':
            bearish_count += 1
        if data_dict['obv_trend'] == 'BEARISH':
            bearish_count += 1
        if 'BEARISH' in data_dict['recent_candles']:
            bearish_count += 1
            
        return bearish_count

    def analyze_raw_data(self, symbol, timeframe, candles):
        """
        Análise COMPLETA usando GPT-5.1 como cérebro central
        """
        
        # 1. Calcular TODOS os indicadores (25+)
        comprehensive_data = self.calculate_comprehensive_analysis(symbol, timeframe, candles)
        
        # 2. Preparar prompt SUPER DETALHADO para a IA
        prompt = f"""
# 🎯 ANÁLISE DE TRADING PROFISSIONAL - GPT-5.1

## DADOS COMPLETOS DO MERCADO:
**Ativo:** {symbol} | **Timeframe:** {timeframe}
**Preço Atual:** {comprehensive_data['current_price']:.6f}
**Variação 24h:** {comprehensive_data['price_change_24h']:.2f}%

## 📊 ANÁLISE TÉCNICA COMPLETA:

### 🔥 TENDÊNCIA:
- EMA 20: {comprehensive_data['ema_20']:.6f}
- EMA 50: {comprehensive_data['ema_50']:.6f}
- EMA 100: {comprehensive_data['ema_100']:.6f} 
- EMA 200: {comprehensive_data['ema_200']:.6f}
- Alinhamento EMAs: {comprehensive_data['ema_alignment']}

- MACD: {comprehensive_data['macd_line']:.6f} | Signal: {comprehensive_data['macd_signal']:.6f}
- Histogram MACD: {comprehensive_data['macd_histogram']:.6f} | Tendência: {comprehensive_data['macd_trend']}

- ADX: {comprehensive_data['adx']:.1f} | +DI: {comprehensive_data['adx_plus']:.1f} | -DI: {comprehensive_data['adx_minus']:.1f}
- Tendência ADX: {comprehensive_data['adx_trend']}

### ⚡ MOMENTUM:
- RSI: {comprehensive_data['rsi']:.1f} | Tendência: {comprehensive_data['rsi_trend']}
- Stochastic: K={comprehensive_data['stoch_k']:.1f} | D={comprehensive_data['stoch_d']:.1f} | Tendência: {comprehensive_data['stoch_trend']}
- Williams %R: {comprehensive_data['williams_r']:.1f}
- CCI: {comprehensive_data['cci']:.1f}
- Momentum: {comprehensive_data['momentum']:.2f}%

### 🌪️ VOLATILIDADE:
- Bollinger Upper: {comprehensive_data['bb_upper']:.6f}
- Bollinger Lower: {comprehensive_data['bb_lower']:.6f}
- Posição nas BB: {comprehensive_data['bb_position']:.1%}
- Largura das BB: {comprehensive_data['bb_width']:.4f}
- ATR: {comprehensive_data['atr']:.6f} | ATR %: {comprehensive_data['atr_percentage']:.2f}%

### 📈 VOLUME:
- Volume Atual: {comprehensive_data['volume']:.2f}
- Volume Ratio: {comprehensive_data['volume_ratio']:.2f}x
- Tendência Volume: {comprehensive_data['volume_trend']}
- VWAP: {comprehensive_data['vwap']:.6f} | Posição: {comprehensive_data['vwap_position']}
- OBV: {comprehensive_data['obv']:.0f} | Tendência: {comprehensive_data['obv_trend']}

### 🎯 PRICE ACTION:
- Estrutura de Mercado: {comprehensive_data['market_structure']}
- Força da Tendência: {comprehensive_data['trend_strength']:.1f}/100
- Volatilidade: {comprehensive_data['volatility_percentage']:.2f}%
- Candles Recentes: {comprehensive_data['recent_candles']}

### 🛡️ NÍVEIS CRÍTICOS:
- Suportes: {comprehensive_data['support_levels']}
- Resistências: {comprehensive_data['resistance_levels']}
- Recent High: {comprehensive_data['key_levels']['recent_high']:.6f}
- Recent Low: {comprehensive_data['key_levels']['recent_low']:.6f}

### ⚖️ CONVERGÊNCIA DE SINAIS:
- Sinais de Alta: {comprehensive_data['bullish_signals']}/10
- Sinais de Baixa: {comprehensive_data['bearish_signals']}/10

## 🧠 SUA MISSÃO COMO ESPECIALISTA:
Analise TODOS os dados acima de forma PROFISSIONAL e determine:

**DECISÃO FINAL: COMPRAR ou VENDER**

Forneça um relatório detalhado incluindo:

1. **DECISÃO** (COMPRAR/VENDER)
2. **NÍVEL DE CONFIANÇA** (0-100%)
3. **ANÁLISE TÉCNICA DETALHADA** - Explicação completa baseada em convergência de indicadores
4. **PRÓXIMOS MOVIMENTOS ESPERADOS**
5. **FATORES CHAVE** que influenciaram a decisão

**BASEIE-SE EM:**
- Convergência/Divergência de indicadores
- Força e direção da tendência
- Confirmação de volume
- Estrutura de mercado
- Análise de price action
- Níveis de suporte e resistência
"""

        # 3. CONSULTAR IA GPT-5.1 PARA DECISÃO FINAL
        response = client.chat.completions.create(
            model="gpt-4",  # Altere para "gpt-5.1" quando disponível
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )

        return response.choices[0].message.content


# ==============================================================
#  SERVIÇO FLASK (MANTIDO)
# ==============================================================

def create_app():
    app = Flask(__name__)
    app.analyzer = CentralIntelligenceAnalyzer()
    return app


app = create_app()

# ==============================================================
#  HTML (MANTIDO - JÁ ESTÁ PERFEITO)
# ==============================================================

# SEU HTML ATUAL PERMANECE EXATAMENTE O MESMO!

# ... (seu HTML completo aqui - mantido igual)


# ==============================================================
#  BACKEND — ROTA PARA ANÁLISE (200 CANDLES)
# ==============================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/analisar_ativo", methods=["POST"])
def analisar_ativo():
    try:
        data = request.json
        symbol = data["symbol"]
        timeframe = data["timeframe"]

        # Mapear timeframe para o formato da Binance
        tf = timeframe  # já vem como "1m" ou "5m"

        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit=200"
        candles_raw = requests.get(url, timeout=10).json()

        candles = []
        for c in candles_raw:
            candles.append({
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time": c[6]
            })

        resultado = app.analyzer.analyze_raw_data(symbol, timeframe, candles)

        return jsonify({
            "symbol": symbol,
            "timeframe": timeframe,
            "resultado": resultado
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================================================
#  START SERVER (RAILWAY)
# ==============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
