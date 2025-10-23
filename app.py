import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import time

# Configuração da página
st.set_page_config(
    page_title="Sinais Trading - IA Direta",
    page_icon="📈",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .signal-buy {
        background-color: #C8E6C9;
        padding: 20px;
        border-radius: 10px;
        border: 3px solid #4CAF50;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .signal-sell {
        background-color: #FFCDD2;
        padding: 20px;
        border-radius: 10px;
        border: 3px solid #F44336;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .asset-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: white;
    }
    .price-up {
        color: #4CAF50;
        font-weight: bold;
    }
    .price-down {
        color: #F44336;
        font-weight: bold;
    }
    .update-btn {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Lista de ativos
ASSETS = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'ADA/USD', 'BNB/USD']

class TradingSignalAI:
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        
    def get_ohlc_data(self, pair, interval=60):
        """Busca dados OHLC da Kraken"""
        try:
            url = f"{self.base_url}/OHLC"
            kraken_pair = pair.replace('/USD', 'USD')
            params = {'pair': kraken_pair, 'interval': interval}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'result' in data and data['result']:
                key = list(data['result'].keys())[0]
                df = pd.DataFrame(data['result'][key], 
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'count'])
                df['close'] = pd.to_numeric(df['close'])
                return df
        except Exception as e:
            st.error(f"Erro ao buscar dados para {pair}: {str(e)}")
        return None

    def calculate_ema(self, prices, period):
        """Calcula EMA"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_macd(self, prices):
        """Calcula MACD"""
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]

    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def analyze_trend(self, prices):
        """Analisa tendência"""
        if len(prices) < 50:
            return "NEUTRO"
        
        ema9 = self.calculate_ema(prices, 9).iloc[-1]
        ema21 = self.calculate_ema(prices, 21).iloc[-1]
        current_price = prices.iloc[-1]
        
        if current_price > ema9 and ema9 > ema21:
            return "ALTA"
        elif current_price < ema9 and ema9 < ema21:
            return "BAIXA"
        else:
            return "NEUTRO"

    def generate_signal(self, pair):
        """Gera sinal COMPRAR ou VENDER - LÓGICA DIRETA"""
        try:
            df = self.get_ohlc_data(pair)
            if df is None or len(df) < 50:
                return "ERRO", 0, 0, 0, 0, 0, "ERRO"
            
            prices = df['close']
            current_price = prices.iloc[-1]
            previous_price = prices.iloc[-2] if len(prices) > 1 else current_price
            
            # Calcula indicadores
            macd, signal, histogram = self.calculate_macd(prices)
            rsi = self.calculate_rsi(prices)
            trend = self.analyze_trend(prices)
            
            # LÓGICA ULTRA SIMPLES - APENAS COMPRAR OU VENDER
            buy_signals = 0
            sell_signals = 0
            
            # MACD (Sinal Principal)
            if macd > signal and histogram > 0:
                buy_signals += 2
            elif macd < signal and histogram < 0:
                sell_signals += 2
            
            # RSI (Confirmação)
            if rsi < 35:
                buy_signals += 1
            elif rsi > 65:
                sell_signals += 1
            
            # Tendência (Contexto)
            if trend == "ALTA":
                buy_signals += 1
            elif trend == "BAIXA":
                sell_signals += 1
            
            # DECISÃO FINAL
            if buy_signals > sell_signals:
                return "COMPRAR", current_price, current_price - previous_price, rsi, macd, histogram, trend
            else:
                return "VENDER", current_price, current_price - previous_price, rsi, macd, histogram, trend
                
        except Exception as e:
            return "ERRO", 0, 0, 0, 0, 0, f"ERRO"

def main():
    st.markdown('<div class="main-header">🚀 SINAIS TRADING - IA DIRETA</div>', unsafe_allow_html=True)
    
    # Inicializa a IA
    ai = TradingSignalAI()
    
    # Botão para atualizar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 ATUALIZAR SINAIS", use_container_width=True, type="primary"):
            st.rerun()
    
    st.markdown("---")
    
    # Analisa cada ativo
    for asset in ASSETS:
        signal, price, change, rsi, macd, histogram, trend = ai.generate_signal(asset)
        
        # Layout do card
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Informações do ativo
            price_class = "price-up" if change >= 0 else "price-down"
            change_symbol = "↗" if change >= 0 else "↘"
            change_percent = (change / price) * 100 if price > 0 else 0
            
            st.markdown(f"""
            <div class="asset-card">
                <h3>{asset}</h3>
                <p><span class="{price_class}">${price:,.2f} {change_symbol} ({change_percent:+.2f}%)</span></p>
                <p><strong>RSI:</strong> {rsi:.1f} | <strong>MACD:</strong> {macd:.4f}</p>
                <p><strong>Histograma:</strong> {histogram:.4f} | <strong>Tendência:</strong> {trend}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Sinal PRINCIPAL
            if signal == "COMPRAR":
                st.markdown('<div class="signal-buy">🎯 COMPRAR</div>', unsafe_allow_html=True)
            elif signal == "VENDER":
                st.markdown('<div class="signal-sell">🎯 VENDER</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color: #FFE0B2; padding: 20px; border-radius: 10px; border: 3px solid #FF9800; margin: 10px 0; text-align: center; font-weight: bold; font-size: 1.4rem;">⚠️ ERRO</div>', unsafe_allow_html=True)
    
        st.markdown("---")
    
    # Timestamp
    st.markdown(f"<small>Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
