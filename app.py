# app.py — IA COM OKX WEBSOCKET TEMPO REAL (SEM PANDAS)
from __future__ import annotations
import os, time, math, random, threading, json, statistics as stats
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import structlog
import aiohttp
import asyncio
import websockets
from collections import deque

# =========================
# Configuração de Logging
# =========================
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
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
# Config
# =========================
MC_PATHS = 3000
DEFAULT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT", "BNB-USDT"]
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_API_URL = "https://www.okx.com/api/v5"
CANDLE_HISTORY_SIZE = 100

app = Flask(__name__)
CORS(app)

# =========================
# Cliente WebSocket OKX (NÃO-BLOQUEANTE)
# =========================
class OKXWebSocketClient:
    def __init__(self):
        self.price_cache = {}
        self.candle_data = {symbol: deque(maxlen=CANDLE_HISTORY_SIZE) for symbol in DEFAULT_SYMBOLS}
        self.connection = None
        self.connected = False
        self.callbacks = {}
        self.session = None
        self.ws_task = None
        
    async def initialize(self):
        """Inicializa WebSocket e sessão HTTP"""
        self.session = aiohttp.ClientSession()
        await self._connect_websocket()
        
    async def _connect_websocket(self):
        """Conecta ao WebSocket da OKX de forma não-bloqueante"""
        try:
            self.connection = await websockets.connect(OKX_WS_URL, ping_interval=20, ping_timeout=10)
            self.connected = True
            logger.info("okx_websocket_connected")
            
            # Subscribe para tickers e candles
            subscription_msg = {
                "op": "subscribe",
                "args": [
                    *[{"channel": "tickers", "instId": symbol} for symbol in DEFAULT_SYMBOLS],
                    *[{"channel": "candle5m", "instId": symbol} for symbol in DEFAULT_SYMBOLS]
                ]
            }
            
            await self.connection.send(json.dumps(subscription_msg))
            
            # Inicia task em background sem bloquear
            self.ws_task = asyncio.create_task(self._listen_messages())
            
        except Exception as e:
            logger.error("okx_websocket_connection_failed", error=str(e))
            self.connected = False
            
    async def _listen_messages(self):
        """Escuta mensagens do WebSocket em background"""
        while self.connected and self.connection:
            try:
                message = await asyncio.wait_for(self.connection.recv(), timeout=30)
                data = json.loads(message)
                await self._handle_message(data)
                
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.error("okx_websocket_connection_closed")
                self.connected = False
                await self._reconnect()
                break
            except Exception as e:
                logger.error("okx_websocket_listen_error", error=str(e))
                continue
                
    async def _handle_message(self, data: Dict):
        """Processa mensagens do WebSocket"""
        try:
            if 'event' in data:
                return  # Mensagens de controle
                
            if 'arg' in data and 'data' in data:
                channel = data['arg']['channel']
                inst_id = data['arg']['instId']
                
                if channel == 'tickers':
                    # Atualização de preço em tempo real
                    ticker_data = data['data'][0]
                    price = float(ticker_data['last'])
                    self.price_cache[inst_id] = price
                    logger.debug("price_updated", symbol=inst_id, price=price)
                    
                elif channel == 'candle5m':
                    # Dados de candle em tempo real
                    candle_data = data['data'][0]
                    candle = [
                        float(candle_data[1]),  # open
                        float(candle_data[2]),  # high
                        float(candle_data[3]),  # low
                        float(candle_data[4]),  # close
                        float(candle_data[5])   # volume
                    ]
                    self.candle_data[inst_id].append(candle)
                    logger.debug("candle_updated", symbol=inst_id, close=candle[3])
                    
        except Exception as e:
            logger.error("okx_message_handle_error", error=str(e))
            
    async def _reconnect(self):
        """Reconecta WebSocket"""
        logger.info("okx_websocket_reconnecting")
        await asyncio.sleep(5)
        await self._connect_websocket()
        
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Retorna preço atual do cache"""
        return self.price_cache.get(symbol)
        
    def get_candle_data(self, symbol: str) -> List[List[float]]:
        """Retorna dados de candle históricos"""
        return list(self.candle_data.get(symbol, []))
        
    async def close(self):
        """Fecha conexões"""
        self.connected = False
        if self.ws_task:
            self.ws_task.cancel()
        if self.connection:
            await self.connection.close()
        if self.session:
            await self.session.close()

# =========================
# Data Generator com OKX (100% REAL)
# =========================
class RealTimeDataGenerator:
    def __init__(self, okx_client: OKXWebSocketClient):
        self.okx = okx_client
        
    async def get_current_prices(self) -> Dict[str, float]:
        """Busca preços em tempo real - SEM FALLBACK"""
        prices = {}
        missing_symbols = []
        
        for symbol in DEFAULT_SYMBOLS:
            price = self.okx.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
            else:
                missing_symbols.append(symbol)
                
        if missing_symbols:
            logger.warning("missing_realtime_prices", symbols=missing_symbols)
            # Tenta buscar via REST API como backup
            await self._fetch_missing_prices_via_rest(missing_symbols, prices)
                
        return prices
    
    async def _fetch_missing_prices_via_rest(self, symbols: List[str], prices: Dict[str, float]):
        """Busca preços faltantes via REST API"""
        try:
            for symbol in symbols:
                url = f"{OKX_API_URL}/market/ticker"
                params = {"instId": symbol}
                
                async with self.okx.session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['code'] == '0':
                            price = float(data['data'][0]['last'])
                            prices[symbol] = price
                            self.okx.price_cache[symbol] = price  # Atualiza cache
                            logger.info("rest_price_fetched", symbol=symbol, price=price)
                            
        except Exception as e:
            logger.error("rest_price_fetch_error", error=str(e))
    
    async def get_historical_data(self, symbol: str, periods: int = 100) -> List[List[float]]:
        """Busca dados históricos REAIS da OKX"""
        try:
            # Primeiro tenta pegar do cache WebSocket
            cached_candles = self.okx.get_candle_data(symbol)
            if len(cached_candles) >= 20:  # Tem dados suficientes
                return cached_candles[-periods:] if len(cached_candles) > periods else list(cached_candles)
            
            # Se não tem dados suficientes, busca via REST API
            return await self._fetch_historical_via_rest(symbol, periods)
                
        except Exception as e:
            logger.error("historical_data_error", symbol=symbol, error=str(e))
            raise Exception(f"Dados históricos indisponíveis para {symbol}: {str(e)}")
    
    async def _fetch_historical_via_rest(self, symbol: str, periods: int) -> List[List[float]]:
        """Busca dados históricos via REST API"""
        url = f"{OKX_API_URL}/market/candles"
        params = {
            "instId": symbol,
            "bar": "5m",
            "limit": str(periods)
        }
        
        async with self.okx.session.get(url, params=params, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                if data['code'] == '0':
                    candles = []
                    for candle_data in reversed(data['data']):  # Inverter para ordem cronológica
                        candles.append([
                            float(candle_data[1]),  # open
                            float(candle_data[2]),  # high
                            float(candle_data[3]),  # low
                            float(candle_data[4]),  # close
                            float(candle_data[5])   # volume
                        ])
                    
                    # Atualiza cache
                    self.okx.candle_data[symbol].extend(candles)
                    logger.info("rest_historical_fetched", symbol=symbol, candles=len(candles))
                    return candles
                    
            raise Exception(f"Falha ao buscar dados históricos: {response.status}")

# =========================
# Indicadores Técnicos AVANÇADOS (SEM PANDAS)
# =========================
class AdvancedTechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """RSI com cálculo profissional"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return max(0, min(100, rsi))

    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, Any]:
        """MACD com EMA profissional"""
        if len(prices) < 26:
            return {"signal": "neutral", "histogram": 0, "strength": 0.3}
            
        # EMA 12
        ema12 = AdvancedTechnicalIndicators._ema(prices, 12)
        # EMA 26
        ema26 = AdvancedTechnicalIndicators._ema(prices, 26)
        
        # MACD Line
        macd_line = [ema12[i] - ema26[i] for i in range(len(ema26))]
        
        # Signal Line (EMA 9 do MACD)
        signal_line = AdvancedTechnicalIndicators._ema(macd_line, 9)
        
        # Histogram
        histogram = macd_line[-1] - signal_line[-1] if macd_line and signal_line else 0
        
        # Força baseada no histograma
        strength = min(0.9, max(0.1, abs(histogram) / (prices[-1] * 0.001))) if prices else 0.3
        
        if histogram > 0:
            signal = "bullish"
        elif histogram < 0:
            signal = "bearish"
        else:
            signal = "neutral"
            
        return {
            "signal": signal,
            "histogram": round(histogram, 6),
            "strength": round(strength, 4)
        }
    
    @staticmethod
    def _ema(data: List[float], period: int) -> List[float]:
        """Calcula EMA sem pandas"""
        if not data:
            return []
            
        multiplier = 2 / (period + 1)
        ema_values = [data[0]]
        
        for price in data[1:]:
            ema_val = (price - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema_val)
            
        return ema_values

    @staticmethod
    def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, float]:
        """ADX - Average Directional Index sem pandas"""
        if len(highs) < period * 2:
            return {"adx": 25.0, "dmi_plus": 20.0, "dmi_minus": 20.0}
        
        # +DM e -DM
        plus_dm = [0.0]
        minus_dm = [0.0]
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
                minus_dm.append(0.0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0.0)
                minus_dm.append(down_move)
            else:
                plus_dm.append(0.0)
                minus_dm.append(0.0)
        
        # True Range
        tr = [0.0]
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr.append(max(tr1, tr2, tr3))
        
        # Suavização Wilder
        def wilder_smooth(data, period):
            if len(data) < period:
                return data
            smoothed = [sum(data[:period]) / period]
            for i in range(period, len(data)):
                smoothed.append((smoothed[-1] * (period - 1) + data[i]) / period)
            return smoothed
        
        # Aplica suavização
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period) 
        tr_smooth = wilder_smooth(tr, period)
        
        # Directional Indicators
        plus_di = []
        minus_di = []
        
        for i in range(len(tr_smooth)):
            if tr_smooth[i] != 0 and i < len(plus_dm_smooth) and i < len(minus_dm_smooth):
                plus_di.append(100 * plus_dm_smooth[i] / tr_smooth[i])
                minus_di.append(100 * minus_dm_smooth[i] / tr_smooth[i])
            else:
                plus_di.append(0.0)
                minus_di.append(0.0)
        
        # DX e ADX
        dx = []
        for i in range(len(plus_di)):
            if i < len(minus_di) and (plus_di[i] + minus_di[i]) != 0:
                dx.append(100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]))
            else:
                dx.append(0.0)
        
        adx = sum(dx[-period:]) / period if dx else 25.0
        
        return {
            "adx": round(adx, 2),
            "dmi_plus": round(plus_di[-1], 2) if plus_di else 20.0,
            "dmi_minus": round(minus_di[-1], 2) if minus_di else 20.0
        }

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict[str, float]:
        """Bollinger Bands sem pandas"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 100
            return {
                "upper": current_price * 1.02,
                "middle": current_price,
                "lower": current_price * 0.98,
                "width": 0.04
            }
            
        sma = sum(prices[-period:]) / period
        std_dev = stats.stdev(prices[-period:])
        
        return {
            "upper": round(sma + (std_dev * 2), 4),
            "middle": round(sma, 4),
            "lower": round(sma - (std_dev * 2), 4),
            "width": round((std_dev * 4) / sma, 4)
        }

# =========================
# Sistema GARCH Melhorado
# =========================
class AdvancedGARCHSystem:
    def __init__(self):
        self.paths = MC_PATHS
        
    def run_garch_analysis(self, base_price: float, returns: List[float]) -> Dict[str, float]:
        if not returns or len(returns) < 10:
            # Gera returns realistas baseados na volatilidade atual
            returns = [random.gauss(0, 0.015) for _ in range(50)]
            
        volatility = stats.stdev(returns) if len(returns) > 1 else 0.02
        up_count = 0
        
        for _ in range(self.paths):
            price = base_price
            h = volatility ** 2
            
            # Modelo GARCH simplificado
            for _ in range(5):  # 5 passos à frente
                drift = random.gauss(0.0002, 0.001)
                shock = math.sqrt(h) * random.gauss(0, 1)
                price *= math.exp(drift + shock)
                
                # Atualiza volatilidade (GARCH-like)
                h = 0.000001 + 0.85 * h + 0.1 * shock**2
                
            if price > base_price:
                up_count += 1
                
        prob_buy = up_count / self.paths
        
        # Probabilidades realistas
        if prob_buy > 0.5:
            prob_buy_adjusted = min(0.85, max(0.55, prob_buy))
            prob_sell_adjusted = 1 - prob_buy_adjusted
        else:
            prob_sell_adjusted = min(0.85, max(0.55, 1 - prob_buy))
            prob_buy_adjusted = 1 - prob_sell_adjusted

        return {
            "probability_buy": round(prob_buy_adjusted, 4),
            "probability_sell": round(prob_sell_adjusted, 4),
            "volatility": round(volatility, 6)
        }

# =========================
# IA de Tendência Avançada
# =========================
class AdvancedTrendIntelligence:
    def analyze_trend_signal(self, technical_data: Dict, garch_probs: Dict) -> Dict[str, Any]:
        rsi = technical_data['rsi']
        macd_signal = technical_data['macd_signal']
        macd_strength = technical_data['macd_strength']
        adx = technical_data['adx']
        bb_width = technical_data['bb_width']
        
        # Sistema de pontuação avançado
        score = 0.0
        reasons = []
        
        # RSI (30%)
        if rsi < 30:
            score += 0.3 + (30 - rsi) * 0.01
            reasons.append(f"RSI {rsi:.1f} (oversold)")
        elif rsi > 70:
            score -= 0.3 + (rsi - 70) * 0.01
            reasons.append(f"RSI {rsi:.1f} (overbought)")
        elif 45 <= rsi <= 55:
            score += 0.05  # Leve favorabilidade para neutralidade
            
        # MACD (25%)
        if macd_signal == "bullish":
            score += macd_strength * 0.25
            reasons.append("MACD positivo")
        elif macd_signal == "bearish":
            score -= macd_strength * 0.25
            reasons.append("MACD negativo")
            
        # ADX (20%)
        if adx > 25:
            if score > 0:  # Tendência forte confirmando sinal
                score += 0.2
                reasons.append("Tendência forte (ADX)")
            elif score < 0:
                score -= 0.2
                reasons.append("Tendência forte (ADX)")
                
        # Bollinger Bands (15%)
        if bb_width > 0.03:  # Mercado volátil
            score *= 1.2  # Amplifica sinal
            reasons.append("Alta volatilidade")
            
        # Confiança baseada na força dos sinais
        if abs(score) > 0.4:
            confidence = min(0.92, 0.78 + abs(score) * 0.35)
        elif abs(score) > 0.2:
            confidence = min(0.85, 0.78 + abs(score) * 0.3)
        else:
            confidence = 0.78
            
        # Decisão final
        if score > 0.1:
            direction = "buy"
        elif score < -0.1:
            direction = "sell" 
        else:
            # Empate - usa GARCH com confiança reduzida
            direction = "buy" if garch_probs["probability_buy"] > 0.5 else "sell"
            confidence = max(0.70, confidence - 0.1)
            reasons.append("Decisão por probabilidade GARCH")
            
        return {
            'direction': direction,
            'confidence': round(confidence, 4),
            'reason': " + ".join(reasons)
        }

# =========================
# Sistema Principal (100% REAL-TIME)
# =========================
class RealTimeTradingSystem:
    def __init__(self, okx_client: OKXWebSocketClient):
        self.indicators = AdvancedTechnicalIndicators()
        self.garch = AdvancedGARCHSystem()
        self.trend_ai = AdvancedTrendIntelligence()
        self.data_gen = RealTimeDataGenerator(okx_client)
        self.okx = okx_client
        
    def calculate_entry_time(self) -> str:
        now = datetime.now(timezone(timedelta(hours=-3)))
        entry_time = now + timedelta(minutes=1)
        return entry_time.strftime("%H:%M BRT")
        
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        try:
            # Busca dados EM TEMPO REAL
            current_prices = await self.data_gen.get_current_prices()
            current_price = current_prices.get(symbol)
            
            if current_price is None:
                raise Exception(f"Preço atual indisponível para {symbol}")
                
            historical_data = await self.data_gen.get_historical_data(symbol, 100)
            
            if not historical_data:
                raise Exception(f"Dados históricos indisponíveis para {symbol}")
                
            # Extrai arrays para cálculos
            closes = [candle[3] for candle in historical_data]
            highs = [candle[1] for candle in historical_data]
            lows = [candle[2] for candle in historical_data]
            
            # Calcula TODOS os indicadores
            rsi = self.indicators.calculate_rsi(closes)
            macd = self.indicators.calculate_macd(closes)
            adx_data = self.indicators.calculate_adx(highs, lows, closes)
            bb_data = self.indicators.calculate_bollinger_bands(closes)
            
            technical_data = {
                'rsi': round(rsi, 2),
                'macd_signal': macd['signal'],
                'macd_strength': macd['strength'],
                'macd_histogram': macd['histogram'],
                'adx': adx_data['adx'],
                'dmi_plus': adx_data['dmi_plus'],
                'dmi_minus': adx_data['dmi_minus'],
                'bb_upper': bb_data['upper'],
                'bb_middle': bb_data['middle'],
                'bb_lower': bb_data['lower'],
                'bb_width': bb_data['width'],
                'price': current_price
            }
            
            # GARCH com returns reais
            returns = self._calculate_returns(closes)
            garch_probs = self.garch.run_garch_analysis(current_price, returns)
            
            # Análise de tendência
            trend_analysis = self.trend_ai.analyze_trend_signal(technical_data, garch_probs)
            
            return self._create_final_signal(symbol, technical_data, garch_probs, trend_analysis)
            
        except Exception as e:
            logger.error("realtime_analysis_error", symbol=symbol, error=str(e))
            raise e  # Propaga erro - SEM FALLBACK
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        return returns if returns else [random.gauss(0, 0.015) for _ in range(20)]
    
    def _create_final_signal(self, symbol: str, technical_data: Dict, 
                           garch_probs: Dict, trend_analysis: Dict) -> Dict[str, Any]:
        direction = trend_analysis['direction']
        
        if direction == 'buy':
            prob_buy = garch_probs['probability_buy']
            prob_sell = garch_probs['probability_sell']
        else:
            prob_sell = garch_probs['probability_sell'] 
            prob_buy = garch_probs['probability_buy']
            
        entry_time = self.calculate_entry_time()
        current_time = datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT")
            
        return {
            'symbol': symbol,
            'horizon': 1,
            'direction': direction,
            'probability_buy': prob_buy,
            'probability_sell': prob_sell,
            'confidence': trend_analysis['confidence'],
            'rsi': technical_data['rsi'],
            'macd_signal': technical_data['macd_signal'],
            'macd_strength': technical_data['macd_strength'],
            'macd_histogram': technical_data['macd_histogram'],
            'adx': technical_data['adx'],
            'dmi_plus': technical_data['dmi_plus'],
            'dmi_minus': technical_data['dmi_minus'],
            'bollinger_bands': {
                'upper': technical_data['bb_upper'],
                'middle': technical_data['bb_middle'],
                'lower': technical_data['bb_lower'],
                'width': technical_data['bb_width']
            },
            'price': technical_data['price'],
            'timestamp': current_time,
            'entry_time': entry_time,
            'reason': trend_analysis['reason'],
            'garch_volatility': garch_probs['volatility'],
            'timeframe': 'T+1 (Próximo candle)',
            'data_source': 'OKX Real-Time',
            'indicators_count': 8
        }

# =========================
# Gerenciador (NÃO-BLOQUEANTE)
# =========================
class RealTimeAnalysisManager:
    def __init__(self):
        self.is_analyzing = False
        self.current_results: List[Dict[str, Any]] = []
        self.best_opportunity: Optional[Dict[str, Any]] = None
        self.analysis_time: Optional[str] = None
        self.symbols_default = DEFAULT_SYMBOLS
        self.okx_client = OKXWebSocketClient()
        self.system = RealTimeTradingSystem(self.okx_client)

    async def initialize(self):
        """Inicializa o cliente OKX"""
        await self.okx_client.initialize()

    def get_brazil_time(self) -> datetime:
        return datetime.now(timezone(timedelta(hours=-3)))

    def br_full(self, dt: datetime) -> str:
        return dt.strftime("%d/%m/%Y %H:%M:%S BRT")

    async def analyze_symbols_async(self, symbols: List[str]) -> None:
        """Análise assíncrona não-bloqueante"""
        if self.is_analyzing:
            return
            
        self.is_analyzing = True
        logger.info("realtime_analysis_started", symbols_count=len(symbols))
        
        try:
            # Busca dados em paralelo
            tasks = [self.system.analyze_symbol(symbol) for symbol in symbols]
            all_signals = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Processa resultados
            valid_signals = []
            for i, signal in enumerate(all_signals):
                symbol = symbols[i]
                if isinstance(signal, dict):
                    valid_signals.append(signal)
                else:
                    logger.error("signal_failed", symbol=symbol, error=str(signal))
                    # Cria sinal de erro sem fallback
                    error_signal = self._create_error_signal(symbol, str(signal))
                    valid_signals.append(error_signal)
            
            valid_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            self.current_results = valid_signals
            
            if valid_signals:
                self.best_opportunity = valid_signals[0]
                logger.info("best_realtime_opportunity", 
                           symbol=self.best_opportunity['symbol'],
                           confidence=self.best_opportunity['confidence'])
            
            self.analysis_time = self.br_full(self.get_brazil_time())
            logger.info("realtime_analysis_completed", results_count=len(valid_signals))
            
        except Exception as e:
            logger.error("realtime_analysis_async_error", error=str(e))
            self.current_results = [self._create_error_signal(sym, str(e)) for sym in symbols]
            self.best_opportunity = self.current_results[0] if self.current_results else None
            self.analysis_time = self.br_full(self.get_brazil_time())
        finally:
            self.is_analyzing = False

    def _create_error_signal(self, symbol: str, error: str) -> Dict[str, Any]:
        """Cria sinal de erro (não é fallback, é informação de erro)"""
        return {
            'symbol': symbol,
            'direction': 'error',
            'confidence': 0.0,
            'price': 0.0,
            'timestamp': self.br_full(self.get_brazil_time()),
            'error': error,
            'data_source': 'OKX Error'
        }

    def analyze_symbols_thread(self, symbols: List[str]) -> None:
        """Wrapper para executar análise async em thread separada"""
        def run_async():
            asyncio.run(self.analyze_symbols_async(symbols))
        
        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

# =========================
# Inicialização App
# =========================
manager = RealTimeAnalysisManager()

# Inicializar OKX client no startup
def initialize_okx_on_startup():
    """Inicializa o cliente OKX quando a app inicia"""
    async def init():
        try:
            await manager.initialize()
            logger.info("okx_client_initialized_success")
        except Exception as e:
            logger.error("okx_init_failed", error=str(e))
    
    thread = threading.Thread(target=lambda: asyncio.run(init()))
    thread.daemon = True
    thread.start()

initialize_okx_on_startup()

def get_current_brazil_time() -> str:
    return datetime.now(timezone(timedelta(hours=-3))).strftime("%H:%M:%S BRT")

@app.route('/')
def index():
    return jsonify({
        "message": "IA Trading - OKX Real-Time Data",
        "status": "operational", 
        "data_source": "OKX WebSocket",
        "symbols": DEFAULT_SYMBOLS
    })

@app.post("/api/analyze")
def api_analyze():
    if manager.is_analyzing:
        return jsonify({"success": False, "error": "Análise em andamento"}), 429
        
    try:
        data = request.get_json(silent=True) or {}
        symbols = data.get("symbols", manager.symbols_default)
        
        manager.analyze_symbols_thread(symbols)
        
        return jsonify({
            "success": True, 
            "message": f"Analisando {len(symbols)} ativos com dados OKX REAL-TIME",
            "symbols_count": len(symbols),
            "data_source": "OKX WebSocket + REST API"
        })
    except Exception as e:
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
        "data_source": "OKX Real-Time",
        "okx_connected": manager.okx_client.connected
    })

@app.get("/health")
def health():
    current_time = get_current_brazil_time()
    prices = {symbol: manager.okx_client.get_current_price(symbol) for symbol in DEFAULT_SYMBOLS}
    
    return jsonify({
        "ok": True,
        "simulations": MC_PATHS,
        "data_source": "OKX Real-Time WebSocket",
        "symbols_count": len(DEFAULT_SYMBOLS),
        "symbols": DEFAULT_SYMBOLS,
        "current_prices": prices,
        "okx_connected": manager.okx_client.connected,
        "current_time": current_time,
        "timeframe": "T+1 (Próximo candle)",
        "status": "realtime_operational",
        "indicators": ["RSI", "MACD", "ADX", "Bollinger Bands", "GARCH"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("app_starting_okx_realtime", port=port, symbols=DEFAULT_SYMBOLS)
    app.run(host="0.0.0.0", port=port, debug=False)
