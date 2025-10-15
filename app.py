from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime, timezone, timedelta
import os
import random
import numpy as np
from typing import List, Dict, Tuple

# SIMULAÇÃO APRIMORADA com todas as melhorias
class EnhancedCore:
    @staticmethod
    def generate_realistic_candles(base_price: float, trend_strength: float, volatility: float, num_candles: int = 21) -> List[float]:
        """Gera candles realistas com tendência, momentum e ruído"""
        prices = [base_price]
        
        for i in range(num_candles - 1):
            # Componente de tendência (direcional)
            trend_component = trend_strength * random.uniform(0.001, 0.005) * base_price
            
            # Componente de momentum (aceleração)
            momentum = 0
            if len(prices) > 3:
                recent_change = (prices[-1] - prices[-3]) / prices[-3]
                momentum = recent_change * 0.3  # 30% do momentum recente
            
            # Componente de ruído (volatilidade)
            noise = random.gauss(0, 1) * volatility * base_price * 0.01
            
            # Volatilidade dinâmica baseada no horizonte
            time_factor = min(1.0, (i + 1) / num_candles)  # Aumenta com o tempo
            dynamic_volatility = volatility * (1 + time_factor * 0.5)
            
            # Preço final com todos os componentes
            price_change = trend_component + momentum + noise * dynamic_volatility
            new_price = prices[-1] + price_change
            
            # Garantir que o preço não fique negativo
            new_price = max(new_price, base_price * 0.1)
            prices.append(new_price)
        
        return prices

    @staticmethod
    def calculate_technical_indicators(prices: List[float]) -> Dict:
        """Calcula indicadores técnicos realistas a partir dos preços"""
        if len(prices) < 14:
            return {'adx': 25, 'rsi': 50, 'trend_strength': 0.5}
        
        # Cálculo do RSI
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains[-14:]) if gains else 0
        avg_loss = np.mean(losses[-14:]) if losses else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Cálculo simplificado do ADX (força da tendência)
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        volatility = np.std(price_changes) if price_changes else 0
        trend_direction = np.mean(price_changes[-5:]) if len(price_changes) >= 5 else 0
        
        # ADX baseado na volatilidade e direção da tendência
        base_adx = min(60, volatility * 1000 / (prices[0] if prices[0] > 0 else 1))
        trend_strength = min(1.0, abs(trend_direction) / (prices[0] * 0.01) if prices[0] > 0 else 0.5)
        adx = base_adx * (1 + trend_strength)
        
        return {
            'adx': max(10, min(80, adx)),
            'rsi': max(10, min(90, rsi)),
            'trend_strength': trend_strength,
            'volatility': volatility
        }

    @staticmethod
    def intelligent_probability_system(prices: List[float], indicators: Dict, horizon: int) -> Dict:
        """Sistema inteligente de probabilidades baseado em ação de preço real"""
        # Análise de tendência a partir dos candles
        price_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        
        # Momentum baseado nos últimos preços
        if len(prices) >= 5:
            short_momentum = (prices[-1] - prices[-3]) / prices[-3]
            medium_momentum = (prices[-1] - prices[-5]) / prices[-5]
        else:
            short_momentum = medium_momentum = price_trend
        
        # Probabilidade base ajustada pela tendência real
        base_prob = 0.5 + (price_trend * 2)  # Tendência influencia diretamente
        
        # Ajustes por indicadores técnicos
        rsi_factor = 0
        if indicators['rsi'] < 30:  # Sobrevendido
            rsi_factor = 0.15
        elif indicators['rsi'] > 70:  # Sobrecomprado
            rsi_factor = -0.15
        elif 40 <= indicators['rsi'] <= 60:  # Neutro
            rsi_factor = 0.05
        
        # Ajuste por força da tendência (ADX)
        adx_factor = 0
        if indicators['adx'] > 40:  # Tendência forte
            adx_factor = 0.1 * (1 if price_trend > 0 else -1)
        elif indicators['adx'] < 20:  # Tendência fraca
            adx_factor = -0.05  # Penalidade por baixa definição
        
        # Penalidade por alta volatilidade
        volatility_penalty = -min(0.2, indicators['volatility'] * 10)
        
        # Ajuste por horizonte temporal
        horizon_factor = -0.05 * (horizon - 1)  # Horizonte maior = mais incerteza
        
        # Probabilidade final
        raw_prob = base_prob + rsi_factor + adx_factor + volatility_penalty + horizon_factor
        buy_prob = max(0.3, min(0.85, 0.5 + raw_prob))
        sell_prob = 1 - buy_prob
        
        return {
            'p_buy': buy_prob,
            'p_sell': sell_prob,
            'trend': price_trend,
            'momentum': medium_momentum
        }

    @staticmethod
    def multi_factor_decision_system(probabilities: Dict, indicators: Dict, prices: List[float]) -> Dict:
        """Sistema de decisão multi-fatorial com pontuação"""
        score = 0
        factors = []
        
        # 1. Fator de Probabilidade (peso 40%)
        prob_strength = abs(probabilities['p_buy'] - 0.5) * 2
        prob_score = prob_strength * 40
        score += prob_score
        factors.append(f"Prob: {prob_score:.1f}pts")
        
        # 2. Fator de Tendência ADX (peso 25%)
        adx_score = 0
        if indicators['adx'] > 40:
            adx_score = 25
        elif indicators['adx'] > 25:
            adx_score = 15
        elif indicators['adx'] > 20:
            adx_score = 10
        score += adx_score
        factors.append(f"ADX: {adx_score:.1f}pts")
        
        # 3. Fator RSI (peso 20%)
        rsi_score = 0
        if 30 <= indicators['rsi'] <= 70:
            rsi_score = 20  # Zona neutra = bom
        elif 25 <= indicators['rsi'] <= 75:
            rsi_score = 10  # Zona aceitável
        score += rsi_score
        factors.append(f"RSI: {rsi_score:.1f}pts")
        
        # 4. Fator Momentum (peso 15%)
        momentum_strength = abs(probabilities['momentum'])
        momentum_score = min(15, momentum_strength * 100)
        score += momentum_score
        factors.append(f"Momentum: {momentum_score:.1f}pts")
        
        # Direção baseada nas probabilidades
        direction = "buy" if probabilities['p_buy'] > 0.5 else "sell"
        
        # Overrides técnicos inteligentes
        technical_override = False
        final_direction = direction
        
        # REGRAS DE INVERSÃO APRIMORADAS
        # 1. RSI extremo + ADX forte
        if (indicators['rsi'] > 75 and indicators['adx'] > 35 and direction == "buy"):
            final_direction = "sell"
            technical_override = True
            score += 10  # Bônus por identificar reversão
        elif (indicators['rsi'] < 25 and indicators['adx'] > 35 and direction == "sell"):
            final_direction = "buy"
            technical_override = True
            score += 10
        
        # 2. Alta volatilidade + tendência fraca
        elif (indicators['volatility'] > 0.02 and indicators['adx'] < 20):
            # Em alta volatilidade com tendência fraca, ser conservador
            final_direction = "sell" if probabilities['p_buy'] < 0.55 else "buy"
            technical_override = True
        
        # Confiança baseada na pontuação
        confidence = min(0.95, max(0.4, score / 100))
        
        return {
            'direction': final_direction,
            'original_direction': direction,
            'technical_override': technical_override,
            'confidence': confidence,
            'score': score,
            'factors': factors,
            'volatility': indicators['volatility'],
            'trend_strength': indicators['trend_strength']
        }

    @staticmethod
    def analyze_symbols(symbols, sims=1000, only_adx=None):
        print(f"🔍 Analisando: {symbols} com simulação realista")
        
        mock_results = []
        symbols_list = [s.strip() for s in symbols if s.strip()]
        
        for i, symbol in enumerate(symbols_list):
            # Preço base realista
            base_price = random.uniform(50, 500)
            
            for horizon in [1, 2, 3]:
                # Geração realista de candles
                trend_strength = random.uniform(-0.3, 0.3)  # Tendência mais variada
                volatility = random.uniform(0.005, 0.03)  # Volatilidade realista
                
                prices = EnhancedCore.generate_realistic_candles(
                    base_price, trend_strength, volatility, num_candles=21
                )
                
                # Cálculo de indicadores a partir dos preços
                indicators = EnhancedCore.calculate_technical_indicators(prices)
                
                # Sistema inteligente de probabilidades
                probabilities = EnhancedCore.intelligent_probability_system(
                    prices, indicators, horizon
                )
                
                # Decisão multi-fatorial
                decision = EnhancedCore.multi_factor_decision_system(
                    probabilities, indicators, prices
                )
                
                # Filtro ADX se solicitado
                if only_adx and indicators['adx'] < only_adx:
                    continue
                
                class Row:
                    def __init__(self):
                        self.symbol = symbol
                        self.h = horizon
                        self.direction = decision['direction']
                        self.original_direction = decision['original_direction']
                        self.technical_override = decision['technical_override']
                        self.p_buy = probabilities['p_buy']
                        self.p_sell = probabilities['p_sell']
                        self.conf = decision['confidence']
                        self.adx = indicators['adx']
                        self.rsi = indicators['rsi']
                        self.price = prices[-1]
                        self.ts = datetime.now().strftime("%H:%M:%S")
                        self.volatility = decision['volatility']
                        self.trend_strength = decision['trend_strength']
                        self.score_factors = decision['factors']
                        self.assertiveness = EnhancedCore.calculate_assertiveness(self)
                
                mock_results.append(Row())
        
        # Filtrar apenas oportunidades com confiança decente
        quality_results = [r for r in mock_results if r.conf >= 0.55]
        best = max(quality_results, key=lambda x: x.conf) if quality_results else None
        
        print(f"   📊 Resultados: {len(quality_results)}/{len(mock_results)} com confiança ≥55%")
        return quality_results, best

    @staticmethod
    def calculate_assertiveness(row):
        """Cálculo de assertividade mais preciso"""
        base_score = row.conf * 80  # Base da confiança
        
        # Bônus por força de tendência
        if row.trend_strength > 0.1:
            base_score += 10
        elif row.trend_strength > 0.05:
            base_score += 5
            
        # Bônus por ADX forte
        if row.adx > 35:
            base_score += 8
        elif row.adx > 25:
            base_score += 4
            
        # Bônus por RSI neutro
        if 40 <= row.rsi <= 60:
            base_score += 5
            
        # Penalidade por alta volatilidade
        if row.volatility > 0.02:
            base_score -= 5
            
        return min(round(base_score, 1), 100)

# TODO: Quando quiser usar seu código real, DESCOMENTE a linha abaixo:
# from core import analyze_symbols

app = Flask(__name__)
CORS(app)

# Use a simulação aprimorada
analyze_symbols_real = EnhancedCore.analyze_symbols
# analyze_symbols_real = analyze_symbols  # Use esta linha com seu código real

class AnalysisManager:
    def __init__(self):
        self.current_results = []
        self.best_opportunity = None
        self.analysis_time = None
        self.is_analyzing = False
    
    def analyze_symbols_thread(self, symbols, sims, only_adx):
        try:
            self.is_analyzing = True
            print(f"🔍 Iniciando análise aprimorada: {symbols}")
            
            # USA a função de análise aprimorada
            all_rows, best = analyze_symbols_real(symbols, sims=sims, only_adx=only_adx)
            
            # 🎯 APENAS MELHOR T+ DE CADA ATIVO (maior confiança)
            best_per_symbol = {}
            for row in all_rows:
                symbol = row.symbol
                if symbol not in best_per_symbol or row.conf > best_per_symbol[symbol].conf:
                    best_per_symbol[symbol] = row
            
            # Converter resultados (apenas os melhores)
            self.current_results = []
            for symbol, row in best_per_symbol.items():
                result = {
                    'symbol': row.symbol,
                    'horizon': row.h,
                    'direction': row.direction,
                    'p_buy': round(row.p_buy * 100, 1),
                    'p_sell': round(row.p_sell * 100, 1),
                    'confidence': round(row.conf * 100, 1),
                    'adx': round(getattr(row, 'adx', 0), 1),
                    'rsi': round(getattr(row, 'rsi', 0), 1),
                    'price': round(row.price, 6),
                    'timestamp': getattr(row, 'ts', ''),
                    'technical_override': getattr(row, 'technical_override', False),
                    'assertiveness': getattr(row, 'assertiveness', 0),
                    'volatility': round(getattr(row, 'volatility', 0) * 100, 2),  # Em porcentagem
                    'trend_strength': round(getattr(row, 'trend_strength', 0) * 100, 1),  # Em porcentagem
                    'score_factors': getattr(row, 'score_factors', [])
                }
                self.current_results.append(result)
            
            # Melhor oportunidade global
            if best_per_symbol:
                best_overall = max(best_per_symbol.values(), key=lambda x: x.conf)
                self.best_opportunity = {
                    'symbol': best_overall.symbol,
                    'horizon': best_overall.h,
                    'direction': best_overall.direction,
                    'confidence': round(best_overall.conf * 100, 1),
                    'p_buy': round(best_overall.p_buy * 100, 1),
                    'p_sell': round(best_overall.p_sell * 100, 1),
                    'adx': round(getattr(best_overall, 'adx', 0), 1),
                    'rsi': round(getattr(best_overall, 'rsi', 0), 1),
                    'price': round(best_overall.price, 6),
                    'technical_override': getattr(best_overall, 'technical_override', False),
                    'assertiveness': getattr(best_overall, 'assertiveness', 0),
                    'volatility': round(getattr(best_overall, 'volatility', 0) * 100, 2),
                    'trend_strength': round(getattr(best_overall, 'trend_strength', 0) * 100, 1),
                    'score_factors': getattr(best_overall, 'score_factors', []),
                    'entry_time': self.calculate_entry_time(best_overall.h)
                }
            
            self.analysis_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"✅ Análise aprimorada concluída: {len(self.current_results)} sinais")
            
        except Exception as e:
            print(f"❌ Erro na análise: {str(e)}")
            import traceback
            traceback.print_exc()
            self.current_results = []
            self.best_opportunity = None
        finally:
            self.is_analyzing = False
    
    def calculate_entry_time(self, horizon):
        now = datetime.now(timezone.utc)
        entry_time = now.replace(second=0, microsecond=0) + timedelta(minutes=horizon)
        return entry_time.strftime("%H:%M UTC")

manager = AnalysisManager()

# Página principal APRIMORADA
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 IA Signal Pro - APRIMORADO</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                background: #0a0a0a; 
                color: white; 
                font-family: Arial; 
                margin: 0; 
                padding: 20px; 
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto; 
            }
            .card { 
                background: #1a1a2e; 
                border: 2px solid #3498db; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 10px 0; 
            }
            .best-card { 
                border: 3px solid #f39c12; 
                background: #2c2c3e; 
            }
            input, button, select { 
                width: 100%; 
                padding: 12px; 
                margin: 5px 0; 
                border: 2px solid #3498db; 
                border-radius: 8px; 
                background: #2c3e50; 
                color: white; 
            }
            button { 
                background: #3498db; 
                border: none; 
                font-weight: bold; 
                cursor: pointer; 
            }
            button:hover { background: #2980b9; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            .results { 
                background: #2c3e50; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 8px 0; 
                border-left: 4px solid #3498db; 
            }
            .buy { 
                color: #2ecc71; 
                border-left-color: #2ecc71 !important; 
            }
            .sell { 
                color: #e74c3c; 
                border-left-color: #e74c3c !important; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                gap: 10px; 
                margin: 10px 0; 
            }
            .metric { 
                background: #34495e; 
                padding: 8px; 
                border-radius: 5px; 
                text-align: center; 
            }
            .factor { 
                background: #16a085; 
                padding: 4px 8px; 
                border-radius: 3px; 
                margin: 2px; 
                font-size: 0.8em; 
                display: inline-block; 
            }
            .override { 
                color: #f39c12; 
                font-weight: bold; 
            }
            .volatility-high { color: #e74c3c; }
            .volatility-medium { color: #f39c12; }
            .volatility-low { color: #2ecc71; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🚀 IA Signal Pro - APRIMORADO</h1>
                <p><em>Versão com Simulação Realista + Análise Multi-fatorial</em></p>
                
                <input type="text" id="symbols" value="BTC/USDT,ETH/USDT,ADA/USDT,SOL/USDT" placeholder="Digite os símbolos...">
                <select id="sims">
                    <option value="500">500 simulações</option>
                    <option value="1000" selected>1000 simulações</option>
                    <option value="1500">1500 simulações</option>
                </select>
                <select id="adx">
                    <option value="0">Todos ADX</option>
                    <option value="20">ADX ≥ 20</option>
                    <option value="25" selected>ADX ≥ 25</option>
                    <option value="30">ADX ≥ 30</option>
                </select>
                
                <button onclick="analyze()" id="analyzeBtn">🎯 ANALISAR COM NOVO SISTEMA</button>
            </div>

            <div class="card best-card">
                <h2>🎖️ MELHOR OPORTUNIDADE GLOBAL</h2>
                <div id="bestResult">Aguardando análise...</div>
            </div>

            <div class="card">
                <h2>📈 MELHORES SINAIS POR ATIVO</h2>
                <div id="allResults">Execute uma análise para ver os resultados</div>
            </div>

            <div class="card">
                <h3>ℹ️ STATUS DO SISTEMA</h3>
                <div id="status">Sistema aprimorado conectado e pronto</div>
            </div>
        </div>

        <script>
            function getVolatilityClass(vol) {
                if (vol > 2.5) return 'volatility-high';
                if (vol > 1.5) return 'volatility-medium';
                return 'volatility-low';
            }

            function formatFactors(factors) {
                if (!factors || !Array.isArray(factors)) return '';
                return factors.map(f => `<span class="factor">${f}</span>`).join('');
            }

            async function analyze() {
                const btn = document.getElementById('analyzeBtn');
                btn.disabled = true;
                btn.textContent = '⏳ ANALISANDO COM NOVO SISTEMA...';

                const symbols = document.getElementById('symbols').value;
                const sims = document.getElementById('sims').value;
                const adx = document.getElementById('adx').value;

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            symbols: symbols.split(','),
                            sims: parseInt(sims),
                            only_adx: parseFloat(adx)
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        checkResults();
                    } else {
                        alert('Erro: ' + data.error);
                        btn.disabled = false;
                        btn.textContent = '🎯 ANALISAR COM NOVO SISTEMA';
                    }

                } catch (error) {
                    alert('Erro de conexão: ' + error.message);
                    btn.disabled = false;
                    btn.textContent = '🎯 ANALISAR COM NOVO SISTEMA';
                }
            }

            async function checkResults() {
                try {
                    const response = await fetch('/api/results');
                    const data = await response.json();

                    if (data.success) {
                        updateResults(data);
                        
                        if (data.is_analyzing) {
                            setTimeout(checkResults, 2000);
                        } else {
                            document.getElementById('analyzeBtn').disabled = false;
                            document.getElementById('analyzeBtn').textContent = '🎯 ANALISAR COM NOVO SISTEMA';
                        }
                    }
                } catch (error) {
                    console.error('Erro:', error);
                    setTimeout(checkResults, 3000);
                }
            }

            function updateResults(data) {
                // Melhor oportunidade
                if (data.best) {
                    const best = data.best;
                    const overrideIcon = best.technical_override ? ' ⚡ ' : '';
                    const overrideText = best.technical_override ? 
                        '<br><div class="override">⚡ DECISÃO TÉCNICA AVANÇADA</div>' : '';
                    
                    const volClass = getVolatilityClass(best.volatility);
                    
                    document.getElementById('bestResult').innerHTML = `
                        <div class="results ${best.direction}">
                            <div style="display: flex; justify-content: between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.2em;">${best.symbol} T+${best.horizon}</strong>
                                    <span style="font-size: 1.1em;">
                                        ${best.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'} ${overrideIcon}
                                    </span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.3em; font-weight: bold;">${best.confidence}%</div>
                                    <div>Assertividade: ${best.assertiveness}%</div>
                                </div>
                            </div>
                            
                            <div class="metrics">
                                <div class="metric">
                                    <div>Compra</div>
                                    <strong>${best.p_buy}%</strong>
                                </div>
                                <div class="metric">
                                    <div>Venda</div>
                                    <strong>${best.p_sell}%</strong>
                                </div>
                                <div class="metric">
                                    <div>ADX</div>
                                    <strong>${best.adx}</strong>
                                </div>
                                <div class="metric">
                                    <div>RSI</div>
                                    <strong>${best.rsi}</strong>
                                </div>
                                <div class="metric">
                                    <div>Volatilidade</div>
                                    <strong class="${volClass}">${best.volatility}%</strong>
                                </div>
                                <div class="metric">
                                    <div>Força Trend</div>
                                    <strong>${best.trend_strength}%</strong>
                                </div>
                            </div>
                            
                            <div>Pontuação: ${formatFactors(best.score_factors)}</div>
                            <div>Entrada: ${best.entry_time} | Preço: $${best.price}</div>
                            ${overrideText}
                            <br><em>Análise: ${data.analysis_time}</em>
                        </div>
                    `;
                }

                // Todos os sinais (APENAS MELHOR DE CADA ATIVO)
                if (data.results.length > 0) {
                    let html = '';
                    data.results.sort((a, b) => b.confidence - a.confidence);
                    
                    data.results.forEach(result => {
                        const overrideIcon = result.technical_override ? ' ⚡ ' : '';
                        const volClass = getVolatilityClass(result.volatility);
                        
                        html += `
                        <div class="results ${result.direction}">
                            <div style="display: flex; justify-content: between; align-items: start;">
                                <div style="flex: 1;">
                                    <strong>${result.symbol} T+${result.horizon}</strong>
                                    <span>${result.direction == 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'} ${overrideIcon}</span>
                                    <br>
                                    Compra: ${result.p_buy}% | Venda: ${result.p_sell}% 
                                    | Conf: <strong>${result.confidence}%</strong>
                                    | Assert: ${result.assertiveness}%
                                    <br>
                                    ADX: ${result.adx} | RSI: ${result.rsi} 
                                    | Vol: <span class="${volClass}">${result.volatility}%</span>
                                    | Trend: ${result.trend_strength}%
                                </div>
                            </div>
                            ${result.technical_override ? '<div class="override">⚡ Decisão Técnica</div>' : ''}
                        </div>`;
                    });
                    
                    document.getElementById('allResults').innerHTML = html;
                    document.getElementById('status').textContent = 
                        `✅ ${data.results.length} ativos analisados | Sistema Aprimorado Ativo`;
                }
            }

            // Verificar status inicial
            checkResults();
        </script>
    </body>
    </html>
    '''

# API Routes (mantidas as mesmas)
@app.route('/api/analyze', methods=['POST'])
def analyze():
    if manager.is_analyzing:
        return jsonify({
            'success': False,
            'error': 'Análise já em andamento'
        }), 429
    
    try:
        data = request.get_json()
        
        symbols = [s.strip().upper() for s in data['symbols'] if s.strip()]
        if not symbols:
            return jsonify({
                'success': False,
                'error': 'Nenhum símbolo informado'
            }), 400
            
        sims = int(data.get('sims', 1200))
        only_adx = float(data.get('only_adx', 25)) if data.get('only_adx') else None
        
        # Executar análise em thread
        thread = threading.Thread(
            target=manager.analyze_symbols_thread,
            args=(symbols, sims, only_adx)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Análise aprimorada iniciada...',
            'symbols_count': len(symbols)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/results')
def get_results():
    return jsonify({
        'success': True,
        'results': manager.current_results,
        'best': manager.best_opportunity,
        'analysis_time': manager.analysis_time,
        'total_signals': len(manager.current_results),
        'is_analyzing': manager.is_analyzing
    })

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'version': 'aprimorada-1.0',
        'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        'results_available': len(manager.current_results) > 0,
        'is_analyzing': manager.is_analyzing
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': 'aprimorada'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 IA Signal Pro APRIMORADO iniciando...")
    print("📍 Será disponível em: https://seu-app.up.railway.app")
    print("✅ Melhorias implementadas:")
    print("   - Simulação realista de candles")
    print("   - Sistema de probabilidades inteligente") 
    print("   - Indicadores técnicos correlacionados")
    print("   - Decisão multi-fatorial")
    print("   - Frontend aprimorado")
    print("   - Cálculos mais precisos")
    app.run(host='0.0.0.0', port=port, debug=False)
