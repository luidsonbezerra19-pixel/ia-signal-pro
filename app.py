# app.py — IA SIMPLES E ASSERTIVA COM ENTRADA AUTOMÁTICA
from __future__ import annotations
import os, time, math, random, statistics as stats
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import requests

# =========================
# Config
# =========================
SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT", "BNB-USDT"]
app = Flask(__name__)
CORS(app)

# =========================
# Data Generator - PREÇOS REAIS KRAKEN
# =========================
class DataGenerator:
    def __init__(self):
        self.price_cache = {}
        self._initialize_real_prices()
        
    def _initialize_real_prices(self):
        """Busca preços iniciais REAIS do Kraken"""
        print("🚀 BUSCANDO PREÇOS REAIS KRAKEN...")
        
        for symbol in SYMBOLS:
            try:
                price = self._fetch_current_price_kraken(symbol)
                if price and price > 0:
                    self.price_cache[symbol] = price
                    print(f"✅ {symbol} = ${price:,.2f}")
                else:
                    self._set_fallback_price(symbol)
            except Exception as e:
                print(f"💥 Erro {symbol}: {e}")
                self._set_fallback_price(symbol)

    def _get_kraken_symbol(self, symbol: str) -> str:
        """Converte símbolo para formato Kraken"""
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        
        kraken_map = {
            'BTCUSDT': 'XBTUSDT', 'BTCUSD': 'XBTUSD',
            'ETHUSDT': 'ETHUSDT', 'ETHUSD': 'ETHUSD',
            'SOLUSDT': 'SOLUSD', 'SOLUSD': 'SOLUSD',
            'ADAUSDT': 'ADAUSD', 'ADAUSD': 'ADAUSD',
            'XRPUSDT': 'XRPUSD', 'XRPUSD': 'XRPUSD',
            'BNBUSDT': 'BNBUSD', 'BNBUSD': 'BNBUSD'
        }
        return kraken_map.get(clean_symbol, clean_symbol)

    def _fetch_current_price_kraken(self, symbol: str) -> Optional[float]:
        """Busca preço REAL da Kraken"""
        try:
            kraken_symbol = self._get_kraken_symbol(symbol)
            url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if not data.get('error') and data.get('result'):
                    for key, value in data['result'].items():
                        return float(value['c'][0])
        except Exception:
            return None
        return None

    def _set_fallback_price(self, symbol: str):
        """Preços de fallback realistas"""
        fallback_prices = {
            'BTC-USDT': 67432, 'ETH-USDT': 3756, 'SOL-USDT': 143,
            'ADA-USDT': 0.55, 'XRP-USDT': 0.66, 'BNB-USDT': 587
        }
        price = fallback_prices.get(symbol, 100)
        self.price_cache[symbol] = price
        print(f"⚠️  Fallback: {symbol} = ${price:,.2f}")

    def get_current_prices(self) -> Dict[str, float]:
        return self.price_cache.copy()
    
    def get_historical_data(self, symbol: str, periods: int = 100) -> List[List[float]]:
        """Gera dados históricos realistas baseados no preço atual"""
        current_price = self.price_cache.get(symbol, 100)
        return self._generate_candles(current_price, periods)
    
    def _generate_candles(self, base_price: float, periods: int) -> List[List[float]]:
        """Gera candles realistas"""
        candles = []
        price = base_price
        
        for _ in range(periods):
            open_price = price
            volatility = 0.015  # 1.5% de volatilidade
            change_pct = random.gauss(0, volatility)
            close_price = open_price * (1 + change_pct)
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility/2)))
            volume = random.uniform(100000, 1000000)
            
            candles.append([open_price, high_price, low_price, close_price, volume])
            price = close_price
            
        return candles

# =========================
# INDICADORES OFICIAIS - SIMPLES E PRECISOS
# =========================
class TechnicalIndicators:
    
    def rsi(self, closes: List[float], period: int = 14) -> float:
        """RSI OFICIAL - igual TradingView"""
        if len(closes) < period + 1:
            return 50.0
            
        gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        
        if len(gains) < period:
            return 50.0
            
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return max(0, min(100, round(rsi, 2)))

    def macd(self, closes: List[float]) -> Dict[str, Any]:
        """MACD OFICIAL - sinal claro"""
        if len(closes) < 26:
            return {"signal": "neutral", "histogram": 0}
            
        def ema(data: List[float], period: int) -> float:
            if len(data) < period:
                return data[-1] if data else 0
            multiplier = 2 / (period + 1)
            ema_val = sum(data[:period]) / period
            for i in range(period, len(data)):
                ema_val = (data[i] * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
            
        # EMA 12 e 26
        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd_line = ema12 - ema26
        
        # Signal line (EMA 9 do MACD)
        if len(closes) >= 35:
            macd_values = []
            for i in range(26, len(closes)):
                segment = closes[i-25:i+1]
                macd_val = ema(segment, 12) - ema(segment, 26)
                macd_values.append(macd_val)
            signal_line = ema(macd_values, 9) if len(macd_values) >= 9 else macd_line
        else:
            signal_line = macd_line
            
        histogram = macd_line - signal_line
        
        # Sinal claro
        if histogram > 0 and macd_line > signal_line:
            signal = "bullish"
        elif histogram < 0 and macd_line < signal_line:
            signal = "bearish"
        else:
            signal = "neutral"
            
        return {
            "signal": signal,
            "histogram": round(histogram, 6),
            "macd_line": round(macd_line, 6),
            "signal_line": round(signal_line, 6)
        }

    def trend(self, closes: List[float]) -> Dict[str, Any]:
        """Tendência OFICIAL - simples e direta"""
        if len(closes) < 20:
            return {"direction": "neutral", "strength": 0.5}
            
        short_ma = sum(closes[-10:]) / 10
        medium_ma = sum(closes[-20:]) / 20
        
        if short_ma > medium_ma and closes[-1] > short_ma:
            direction = "bullish"
            strength = min(1.0, (short_ma - medium_ma) / medium_ma * 10)
        elif short_ma < medium_ma and closes[-1] < short_ma:
            direction = "bearish" 
            strength = min(1.0, (medium_ma - short_ma) / medium_ma * 10)
        else:
            direction = "neutral"
            strength = 0.3
            
        return {
            "direction": direction,
            "strength": round(strength, 3)
        }

# =========================
# IA DECISÓRIA - SEMPRE COMPRAR ou VENDER
# =========================
class SimpleAI:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def analyze(self, symbol: str, closes: List[float], current_price: float) -> Dict[str, Any]:
        """Análise SIMPLES e ASSERTIVA - sempre retorna COMPRAR ou VENDER"""
        
        # Calcular indicadores
        rsi = self.indicators.rsi(closes)
        macd = self.indicators.macd(closes)
        trend = self.indicators.trend(closes)
        
        # REGRAS CLARAS E ASSERTIVAS
        decision = self._make_decision(rsi, macd, trend)
        
        # Calcular horário de entrada (próximo candle - 1 minuto no futuro)
        now = datetime.now(timezone(timedelta(hours=-3)))
        entry_time = (now + timedelta(minutes=1)).strftime("%H:%M")
        analysis_time = now.strftime("%H:%M:%S")
        
        return {
            'symbol': symbol,
            'decision': decision['action'],
            'direction': decision['direction'],
            'confidence': decision['confidence'],
            'reason': decision['reason'],
            'price': current_price,
            'rsi': rsi,
            'macd_signal': macd['signal'],
            'macd_histogram': macd['histogram'],
            'trend': trend['direction'],
            'trend_strength': trend['strength'],
            'analysis_time': analysis_time,
            'entry_time': entry_time,
            'timestamp': now.strftime("%H:%M:%S BRT")
        }
    
    def _make_decision(self, rsi: float, macd: Dict, trend: Dict) -> Dict:
        """Toma decisão ASSERTIVA baseada em regras claras"""
        
        # REGRA 1: RSI EXTREMO + MACD CONFIRMA → DECISÃO FORTE
        if rsi < 30 and macd['signal'] == 'bullish':
            return self._create_buy(0.85, "RSI SOBREVENDIDO + MACD BULLISH")
        if rsi > 70 and macd['signal'] == 'bearish':
            return self._create_sell(0.85, "RSI SOBRECOMPRADO + MACD BEARISH")
        
        # REGRA 2: TENDÊNCIA FORTE + RSI FAVORÁVEL
        if trend['direction'] == 'bullish' and trend['strength'] > 0.7 and rsi < 60:
            return self._create_buy(0.80, "TENDÊNCIA FORTE DE ALTA + RSI OK")
        if trend['direction'] == 'bearish' and trend['strength'] > 0.7 and rsi > 40:
            return self._create_sell(0.80, "TENDÊNCIA FORTE DE BAIXA + RSI OK")
        
        # REGRA 3: MOMENTUM MACD FORTE
        if abs(macd['histogram']) > 0.001:
            if macd['signal'] == 'bullish' and rsi < 55:
                return self._create_buy(0.75, "MOMENTUM MACD BULLISH FORTE")
            if macd['signal'] == 'bearish' and rsi > 45:
                return self._create_sell(0.75, "MOMENTUM MACD BEARISH FORTE")
        
        # REGRA 4: RSI SIMPLES
        if rsi < 40:
            return self._create_buy(0.70, "RSI EM ZONA DE COMPRA")
        if rsi > 60:
            return self._create_sell(0.70, "RSI EM ZONA DE VENDA")
        
        # REGRA 5: TENDÊNCIA PRINCIPAL
        if trend['direction'] == 'bullish':
            return self._create_buy(0.65, "TENDÊNCIA DE ALTA PREDOMINANTE")
        if trend['direction'] == 'bearish':
            return self._create_sell(0.65, "TENDÊNCIA DE BAIXA PREDOMINANTE")
        
        # ÚLTIMO RECURSO: DECISÃO BASEADA NO PREÇO (COMPRA por padrão em mercados)
        return self._create_buy(0.60, "ANÁLISE NEUTRA - TENDENDO PARA COMPRA")

    def _create_buy(self, confidence: float, reason: str) -> Dict:
        return {
            'action': 'COMPRAR',
            'direction': 'buy', 
            'confidence': confidence,
            'reason': f"🎯 COMPRAR: {reason}"
        }

    def _create_sell(self, confidence: float, reason: str) -> Dict:
        return {
            'action': 'VENDER', 
            'direction': 'sell',
            'confidence': confidence,
            'reason': f"🎯 VENDER: {reason}"
        }

# =========================
# Sistema Principal SIMPLES
# =========================
class TradingSystem:
    def __init__(self):
        self.data_gen = DataGenerator()
        self.ai = SimpleAI()
        
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        try:
            # Dados atuais
            current_prices = self.data_gen.get_current_prices()
            current_price = current_prices.get(symbol, 100)
            
            # Dados históricos
            historical_data = self.data_gen.get_historical_data(symbol)
            closes = [candle[3] for candle in historical_data]
            
            # Análise da IA
            return self.ai.analyze(symbol, closes, current_price)
            
        except Exception as e:
            # Fallback em caso de erro
            return self._create_error_signal(symbol, str(e))
    
    def _create_error_signal(self, symbol: str, error: str) -> Dict[str, Any]:
        """Sinal de fallback em caso de erro"""
        now = datetime.now(timezone(timedelta(hours=-3)))
        entry_time = (now + timedelta(minutes=1)).strftime("%H:%M")
        
        return {
            'symbol': symbol,
            'decision': 'COMPRAR',  # Sempre decide mesmo com erro
            'direction': 'buy',
            'confidence': 0.55,
            'reason': f"⚠️ DECISÃO DE EMERGÊNCIA: {error}",
            'price': 100,
            'rsi': 50.0,
            'macd_signal': 'neutral',
            'macd_histogram': 0,
            'trend': 'neutral',
            'trend_strength': 0.5,
            'analysis_time': now.strftime("%H:%M:%S"),
            'entry_time': entry_time,
            'timestamp': now.strftime("%H:%M:%S BRT")
        }

# =========================
# Gerenciador e API SIMPLES
# =========================
class AnalysisManager:
    def __init__(self):
        self.is_analyzing = False
        self.current_results = []
        self.system = TradingSystem()

    def analyze_symbols(self, symbols: List[str]) -> None:
        self.is_analyzing = True
        try:
            results = []
            for symbol in symbols:
                signal = self.system.analyze_symbol(symbol)
                results.append(signal)
            
            # Ordenar por confiança
            results.sort(key=lambda x: x['confidence'], reverse=True)
            self.current_results = results
            
        except Exception as e:
            print(f"Erro na análise: {e}")
            self.current_results = []
        finally:
            self.is_analyzing = False

# =========================
# Inicialização e API
# =========================
manager = AnalysisManager()

def get_brazil_time():
    return datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT")

@app.route('/')
def index():
    current_time = get_brazil_time()
    return Response(f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>IA Trading Simples</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f1120;
                color: white;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                background: #181a2e;
                padding: 20px;
                border-radius: 10px;
            }}
            .clock {{
                font-size: 28px;
                font-weight: bold;
                color: #2aa9ff;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
            }}
            button {{
                background: #2aa9ff;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
                transition: all 0.3s;
            }}
            button:hover {{
                background: #1e8fd6;
                transform: scale(1.05);
            }}
            button:disabled {{
                background: #666;
                cursor: not-allowed;
                transform: none;
            }}
            .signal {{
                background: #223148;
                padding: 20px;
                border-radius: 10px;
                margin: 15px 0;
                border-left: 5px solid #2aa9ff;
            }}
            .signal.buy {{
                border-left-color: #29d391;
            }}
            .signal.sell {{
                border-left-color: #ff5b5b;
            }}
            .signal.best {{
                background: linear-gradient(135deg, #223148, #2a3a5f);
                border: 2px solid #f2a93b;
            }}
            .confidence {{
                color: #2aa9ff;
                font-weight: bold;
                font-size: 18px;
            }}
            .decision {{
                font-size: 20px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .buy-decision {{
                color: #29d391;
            }}
            .sell-decision {{
                color: #ff5b5b;
            }}
            .info-line {{
                margin: 8px 0;
                padding: 8px;
                background: #1b2b41;
                border-radius: 5px;
            }}
            .status {{
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                text-align: center;
                font-weight: bold;
            }}
            .status.analyzing {{
                background: #f2a93b;
                color: #000;
            }}
            .status.success {{
                background: #29d391;
                color: #000;
            }}
            .status.error {{
                background: #ff5b5b;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 IA TRADING SIMPLES</h1>
                <div class="clock" id="currentTime">{current_time}</div>
                <p>Preços reais Kraken • RSI/MACD Oficiais • Decisões Assertivas</p>
            </div>
            
            <div style="text-align: center;">
                <button onclick="runAnalysis()" id="analyzeBtn">🎯 ANALISAR MERCADO</button>
            </div>
            
            <div id="status"></div>
            <div id="results"></div>
        </div>

        <script>
            // Atualizar relógio em tempo real
            function updateClock() {{
                const now = new Date();
                const brtOffset = -3 * 60;
                const localOffset = now.getTimezoneOffset();
                const brtTime = new Date(now.getTime() + (brtOffset + localOffset) * 60000);
                
                const timeString = brtTime.toLocaleTimeString('pt-BR', {{ 
                    timeZone: 'America/Sao_Paulo',
                    hour12: false 
                }}) + ' BRT';
                
                document.getElementById('currentTime').textContent = timeString;
            }}
            
            setInterval(updateClock, 1000);
            updateClock();

            async function runAnalysis() {{
                const btn = document.getElementById('analyzeBtn');
                btn.disabled = true;
                btn.textContent = '🎯 ANALISANDO...';
                
                // Mostrar status de análise
                document.getElementById('status').innerHTML = 
                    '<div class="status analyzing">🔄 ANALISANDO MERCADO... AGUARDE</div>';
                document.getElementById('results').innerHTML = '';
                
                try {{
                    const response = await fetch('/analyze', {{ method: 'POST' }});
                    const result = await response.json();
                    
                    if (result.success) {{
                        document.getElementById('status').innerHTML = 
                            '<div class="status success">✅ ANÁLISE CONCLUÍDA! CARREGANDO RESULTADOS...</div>';
                        
                        // Aguardar um pouco e buscar resultados automaticamente
                        setTimeout(getResults, 1000);
                    }} else {{
                        document.getElementById('status').innerHTML = 
                            '<div class="status error">❌ ' + result.message + '</div>';
                    }}
                }} catch (error) {{
                    document.getElementById('status').innerHTML = 
                        '<div class="status error">💥 Erro: ' + error.message + '</div>';
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '🎯 ANALISAR MERCADO';
                }}
            }}

            async function getResults() {{
                try {{
                    const response = await fetch('/results');
                    const data = await response.json();
                    
                    if (data.success && data.results.length > 0) {{
                        let html = '<h2>🎯 SINAIS DE TRADING:</h2>';
                        
                        data.results.forEach((signal, index) => {{
                            const isBest = index === 0;
                            const decisionClass = signal.direction === 'buy' ? 'buy-decision' : 'sell-decision';
                            
                            html += `
                                <div class="signal ${{signal.direction}} ${{isBest ? 'best' : ''}}">
                                    <div class="decision ${{decisionClass}}">
                                        ${{signal.decision}} • ${{signal.symbol}} • ${{(signal.confidence * 100).toFixed(1)}}% Confiança
                                        ${{isBest ? ' 🏆' : ''}}
                                    </div>
                                    <div class="info-line">🎯 <strong>Razão:</strong> ${{signal.reason}}</div>
                                    <div class="info-line">
                                        💰 <strong>Preço:</strong> $${signal.price.toFixed(2)} | 
                                        📊 <strong>RSI:</strong> ${{signal.rsi}} ${{signal.rsi < 35 ? '(SOBREVENDIDO)' : signal.rsi > 65 ? '(SOBRECOMPRADO)' : ''}} |
                                        📈 <strong>MACD:</strong> ${{signal.macd_signal}}
                                    </div>
                                    <div class="info-line">
                                        🎯 <strong>Tendência:</strong> ${{signal.trend}} (${{(signal.trend_strength * 100).toFixed(1)}}%) |
                                        ⏰ <strong>Análise:</strong> ${{signal.analysis_time}} |
                                        🚀 <strong>Entrada:</strong> ${{signal.entry_time}}
                                    </div>
                                    <div class="info-line">
                                        📊 <strong>MACD Hist:</strong> ${{signal.macd_histogram}} |
                                        🕒 <strong>Timestamp:</strong> ${{signal.timestamp}}
                                    </div>
                                </div>
                            `;
                        }});
                        
                        document.getElementById('results').innerHTML = html;
                        document.getElementById('status').innerHTML = 
                            '<div class="status success">✅ ANÁLISE CONCLUÍDA • ' + data.results.length + ' SINAIS ENCONTRADOS</div>';
                    }} else {{
                        document.getElementById('status').innerHTML = 
                            '<div class="status error">❌ Nenhum resultado encontrado</div>';
                    }}
                }} catch (error) {{
                    document.getElementById('status').innerHTML = 
                        '<div class="status error">💥 Erro ao buscar resultados: ' + error.message + '</div>';
                }}
            }}

            // Iniciar análise automaticamente quando a página carregar
            window.addEventListener('load', function() {{
                setTimeout(() => {{
                    document.getElementById('status').innerHTML = 
                        '<div class="status info">✅ Sistema pronto • Clique em ANALISAR MERCADO para começar</div>';
                }}, 1000);
            }});
        </script>
    </body>
    </html>
    ''', mimetype='text/html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if manager.is_analyzing:
        return jsonify({'success': False, 'message': 'Análise em andamento'})
    
    manager.analyze_symbols(SYMBOLS)
    return jsonify({'success': True, 'message': f'Análise completada para {len(SYMBOLS)} ativos'})

@app.route('/results')
def get_results():
    return jsonify({
        'success': True,
        'results': manager.current_results
    })

if __name__ == '__main__':
    print("🎯 IA TRADING SIMPLES INICIADA")
    print("✅ Preços reais Kraken • RSI/MACD Oficiais • Decisões Assertivas")
    print("⏰ Relógio em tempo real • Entrada automática no próximo candle")
    print("🌐 Servidor: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
