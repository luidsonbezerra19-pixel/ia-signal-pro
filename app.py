import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time

# Configuração da página
st.set_page_config(
    page_title="Analisador Cripto - Sinais em Tempo Real",
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
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .signal-sell {
        background-color: #FFCDD2;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #F44336;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .asset-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
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
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
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
                df['high'] = pd.to_numeric(df['high'])
                df['low'] = pd.to_numeric(df['low'])
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
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def analyze_trend(self, prices):
        """Analisa tendência usando médias móveis"""
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
        """Gera sinal COMPRAR ou VENDER apenas"""
        df = self.get_ohlc_data(pair)
        if df is None or len(df) < 50:
            return "ERRO", 0, 0, 0, 0, 0
        
        prices = df['close']
        current_price = prices.iloc[-1]
        previous_price = prices.iloc[-2] if len(prices) > 1 else current_price
        
        # Calcula indicadores
        macd, signal, histogram = self.calculate_macd(prices)
        rsi = self.calculate_rsi(prices)
        trend = self.analyze_trend(prices)
        
        # Lógica SIMPLES e DIRETA - apenas COMPRAR ou VENDER
        buy_score = 0
        sell_score = 0
        
        # MACD
        if macd > signal:
            buy_score += 2
        else:
            sell_score += 2
        
        # Tendência
        if trend == "ALTA":
            buy_score += 1
        elif trend == "BAIXA":
            sell_score += 1
        
        # RSI
        if rsi < 40:
            buy_score += 1
        elif rsi > 60:
            sell_score += 1
        
        # Decisão FINAL - apenas COMPRAR ou VENDER
        if buy_score > sell_score:
            return "COMPRAR", current_price, current_price - previous_price, rsi, macd, trend
        else:
            return "VENDER", current_price, current_price - previous_price, rsi, macd, trend

def main():
    st.markdown('<div class="main-header">🚀 ANALISADOR CRIPTO - SINAIS EM TEMPO REAL</div>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <strong>📊 SISTEMA DE ANÁLISE AUTOMÁTICA</strong><br>
    Análise em tempo real com base em MACD, RSI e Tendência para gerar sinais de COMPRAR ou VENDER
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializa a IA
    ai = TradingSignalAI()
    
    # Botão para atualizar dados
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Atualizar Dados", use_container_width=True, type="primary"):
            st.rerun()
    
    st.markdown("---")
    
    # Cria colunas para os cards
    cols = st.columns(2)
    
    # Analisa cada ativo
    for i, asset in enumerate(ASSETS):
        with cols[i % 2]:
            with st.container():
                with st.spinner(f"Analisando {asset}..."):
                    signal, price, change, rsi, macd, trend = ai.generate_signal(asset)
                    
                    # Determina a cor do preço
                    price_class = "price-up" if change >= 0 else "price-down"
                    change_symbol = "↗" if change >= 0 else "↘"
                    change_percent = (change / price) * 100 if price > 0 else 0
                    
                    # Card do ativo
                    st.markdown(f"""
                    <div class="asset-card">
                        <h3>{asset}</h3>
                        <p><span class="{price_class}">${price:,.2f} {change_symbol} ({change_percent:+.2f}%)</span></p>
                        <p><strong>RSI:</strong> {rsi:.1f}</p>
                        <p><strong>MACD:</strong> {macd:.4f}</p>
                        <p><strong>Tendência:</strong> {trend}</p>
                    """, unsafe_allow_html=True)
                    
                    # Sinal
                    if signal == "COMPRAR":
                        st.markdown('<div class="signal-buy">🎯 SINAL: COMPRAR</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="signal-sell">🎯 SINAL: VENDER</div>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Legenda
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🎯 LEGENDA DOS SINAIS:**
        - **COMPRAR**: Condições favoráveis para entrada comprada
        - **VENDER**: Condições favoráveis para entrada vendida
        """)
    
    with col2:
        st.markdown("""
        **📊 INDICADORES:**
        - **MACD**: Momentum e direção da tendência
        - **RSI**: Força relativa do preço
        - **Tendência**: Direção geral do mercado
        """)
    
    # Timestamp da última atualização
    st.markdown(f"<small>Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
