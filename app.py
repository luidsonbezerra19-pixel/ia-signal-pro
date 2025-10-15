from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime, timezone, timedelta
import os
import random
import math

# SIMULAÇÃO APRIMORADA com candles mais realistas (sem numpy)
class MockCore:
    @staticmethod
    def analyze_symbols(symbols, sims=1000, only_adx=None):
        print(f"🔍 Analisando: {symbols} (simulações: {sims})")
        
        mock_results = []
        symbols_list = [s.strip() for s in symbols if s.strip()]
        
        for i, symbol in enumerate(symbols_list):
            # 🎯 SISTEMA DE TENDÊNCIA POR SÍMBOLO (mais consistente)
            symbol_hash = sum(ord(c) for c in symbol)
            random.seed(symbol_hash)
            base_trend = random.uniform(0.4, 0.7)  # Tendência base mais conservadora
            
            for horizon in [1, 2, 3]:
                # 🔄 SIMULAÇÃO DE CANDLES MAIS REALISTA
                price_action = MockCore.simulate_price_action(symbol, horizon, base_trend)
                
                # Probabilidades baseadas na ação de preço simulada
                trend_strength = price_action['trend_strength']
                volatility = price_action['volatility']
                
                # Cálculo mais sofisticado de probabilidades
                p_buy, p_sell = MockCore.calculate_probabilities(
                    trend_strength, volatility, price_action['momentum']
                )
                
                # 🎯 INDICADORES TÉCNICOS CORRELACIONADOS
                adx, rsi = MockCore.generate_technical_indicators(
                    trend_strength, volatility, price_action['momentum']
                )
                
                # 🚨 DECISÃO TÉCNICA INTELIGENTE APRIMORADA
                final_direction, technical_override, confidence = MockCore.make_trading_decision(
                    p_buy, p_sell, adx, rsi, trend_strength, volatility, symbol, horizon
                )
                
                # Preço baseado na simulação real
                current_price = price_action['current_price']
                
                class Row:
                    def __init__(self):
                        self.symbol = symbol
                        self.h = horizon
                        self.direction = final_direction
                        self.original_direction = "buy" if p_buy > p_sell else "sell"
                        self.technical_override = technical_override
                        self.p_buy = p_buy
                        self.p_sell = p_sell
                        self.conf = confidence
                        self.adx = adx
                        self.rsi = rsi
                        self.price = current_price
                        self.ts = datetime.now().strftime("%H:%M:%S")
                        self.volatility = volatility
                        self.trend_strength = trend_strength
                
                mock_results.append(Row())
        
        # Filtrar apenas oportunidades com confiança decente
        quality_results = [r for r in mock_results if r.conf >= 0.6]
        best = max(quality_results, key=lambda x: x.conf) if quality_results else None
        
        print(f"   📊 Resultados: {len(quality_results)}/{len(mock_results)} com confiança ≥60%")
        return quality_results, best
    
    @staticmethod
    def simulate_price_action(symbol, horizon, base_trend):
        """Simula ação de preço mais realista baseada em candles (sem numpy)"""
        # Seed consistente por símbolo e horizonte
        seed = sum(ord(c) for c in symbol) + horizon
        random.seed(seed)
        
        # Parâmetros de mercado realistas
        base_volatility = random.uniform(0.01, 0.05)  # 1-5% de volatilidade
        trend_direction = 1 if base_trend > 0.5 else -1
        trend_strength = abs(base_trend - 0.5) * 2  # 0-1
        
        # Simulação de série temporal (20 candles)
        prices = [100.0]  # Preço inicial
        for i in range(20):
            # Volatilidade dinâmica (maior em horizontes maiores)
            dynamic_volatility = base_volatility * (1 + horizon * 0.2)
            
            # Componentes do movimento de preço (substituindo numpy.random.normal)
            # Usando Box-Muller transform para distribuição normal
            def normal_random(mean=0, std=1):
                u1 = random.random()
                u2 = random.random()
                z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                return mean + z0 * std
            
            trend_component = trend_direction * trend_strength * dynamic_volatility * 0.3
            random_component = normal_random(0, dynamic_volatility * 0.7)
            momentum = prices[-1] - prices[0] if len(prices) > 1 else 0
            
            # Movimento final do preço
            price_change = trend_component + random_component + (momentum * 0.1)
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        # Análise da simulação
        current_price = prices[-1]
        high = max(prices)
        low = min(prices)
        
        return {
            'current_price': round(current_price, 6),
            'high': round(high, 6),
            'low': round(low, 6),
            'trend_strength': min(trend_strength + random.uniform(-0.1, 0.1), 1.0),
            'volatility': dynamic_volatility,
            'momentum': (current_price - prices[0]) / prices[0],
            'candle_pattern': MockCore.analyze_candle_pattern(prices)
        }
    
    @staticmethod
    def analyze_candle_pattern(prices):
        """Analisa padrões de candles para decisões mais assertivas (sem numpy)"""
        if len(prices) < 5:
            return "neutral"
        
        recent = prices[-5:]
        body_sizes = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        avg_body = sum(body_sizes) / len(body_sizes)
        
        # Calcula desvio padrão manualmente (substituindo numpy.std)
        mean_price = sum(prices) / len(prices)
        variance = sum((x - mean_price) ** 2 for x in prices) / len(prices)
        std_dev = math.sqrt(variance)
        
        # Detecta tendência forte
        if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
            return "strong_uptrend"
        elif all(recent[i] < recent[i-1] for i in range(1, len(recent))):
            return "strong_downtrend"
        elif avg_body > std_dev * 0.1:
            return "high_volatility"
        else:
            return "consolidation"
    
    @staticmethod
    def calculate_probabilities(trend_strength, volatility, momentum):
        """Cálculo mais sofisticado de probabilidades"""
        # Base da tendência
        base_p_buy = 0.5 + (trend_strength * 0.4)
        
        # Ajuste por momentum
        momentum_effect = momentum * 2
        base_p_buy += momentum_effect
        
        # Ajuste por volatilidade (alta volatilidade reduz confiança)
        volatility_penalty = volatility * 0.5
        base_p_buy = base_p_buy * (1 - volatility_penalty)
        
        # Garantir limites razoáveis
        p_buy = max(0.3, min(0.85, base_p_buy))
        p_sell = 1 - p_buy
        
        # Normalizar para soma 1
        total = p_buy + p_sell
        p_buy /= total
        p_sell /= total
        
        return round(p_buy, 3), round(p_sell, 3)
    
    @staticmethod
    def generate_technical_indicators(trend_strength, volatility, momentum):
        """Gera indicadores técnicos correlacionados com a ação de preço"""
        # ADX baseado na força da tendência e volatilidade
        base_adx = trend_strength * 40 + 10  # 10-50
        adx_variation = random.uniform(-5, 5)
        adx = max(10, min(60, base_adx + adx_variation))
        
        # RSI baseado no momentum e tendência
        if momentum > 0.02:  # Momentum positivo forte
            base_rsi = 60 + (momentum * 200)
        elif momentum < -0.02:  # Momentum negativo forte
            base_rsi = 40 + (momentum * 200)
        else:
            base_rsi = 50 + (random.uniform(-10, 10))
        
        rsi = max(20, min(80, base_rsi))
        
        return round(adx, 1), round(rsi, 1)
    
    @staticmethod
    def make_trading_decision(p_buy, p_sell, adx, rsi, trend_strength, volatility, symbol, horizon):
        """Sistema de decisão de trading mais assertivo"""
        base_direction = "buy" if p_buy > p_sell else "sell"
        confidence_base = abs(p_buy - p_sell)
        technical_override = False
        
        # 🎯 REGRAS DE DECISÃO APRIMORADAS
        
        # 1. CONFIRMAÇÃO POR MULTIPLOS INDICADORES
        buy_signals = 0
        sell_signals = 0
        
        # Regra ADX + Tendência
        if adx > 25 and trend_strength > 0.3:
            if base_direction == "buy":
                buy_signals += 2
            else:
                sell_signals += 2
        
        # Regra RSI
        if rsi > 70:
            sell_signals += 1
        elif rsi < 30:
            buy_signals += 1
        
        # Regra de Volatilidade (evitar mercados muito voláteis)
        if volatility > 0.04:
            confidence_base *= 0.8  # Reduz confiança em alta volatilidade
        
        # 🚨 DECISÃO FINAL
        final_direction = base_direction
        signal_diff = buy_signals - sell_signals
        
        if abs(signal_diff) >= 2:  # Confirmação técnica forte
            if signal_diff > 0 and base_direction == "sell":
                final_direction = "buy"
                technical_override = True
                print(f"   ⚡ {symbol} T+{horizon}: COMPRA (Confirmação técnica)")
            elif signal_diff < 0 and base_direction == "buy":
                final_direction = "sell"
                technical_override = True
                print(f"   ⚡ {symbol} T+{horizon}: VENDA (Confirmação técnica)")
        
        # 📊 CÁLCULO DE CONFIANÇA APRIMORADO
        confidence = confidence_base
        
        # Bonus por confirmação técnica
        if technical_override and abs(signal_diff) >= 2:
            confidence += 0.15
        
        # Bonus por ADX forte
        if adx > 30:
            confidence += 0.1
        
        # Bonus por RSI neutro
        if 40 <= rsi <= 60:
            confidence += 0.05
        
        # Penalidade por alta volatilidade
        if volatility > 0.03:
            confidence -= 0.1
        
        confidence = max(0.4, min(0.95, confidence))
        
        return final_direction, technical_override, round(confidence, 3)

# TODO: Quando quiser usar seu código real, DESCOMENTE a linha abaixo:
# from core import analyze_symbols

app = Flask(__name__)
CORS(app)

# Use a simulação por enquanto - TROQUE depois pelo seu código
analyze_symbols_real = MockCore.analyze_symbols
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
            print(f"🔍 Iniciando análise: {symbols}")
            
            # USA a função de análise (mock ou real)
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
                    'assertiveness': self.calculate_assertiveness(row),
                    'volatility': round(getattr(row, 'volatility', 0) * 100, 2),
                    'trend_strength': round(getattr(row, 'trend_strength', 0) * 100, 1)
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
                    'assertiveness': self.calculate_assertiveness(best_overall),
                    'entry_time': self.calculate_entry_time(best_overall.h),
                    'volatility': round(getattr(best_overall, 'volatility', 0) * 100, 2),
                    'trend_strength': round(getattr(best_overall, 'trend_strength', 0) * 100, 1)
                }
            
            self.analysis_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"✅ Análise concluída: {len(self.current_results)} sinais (melhor de cada ativo)")
            
        except Exception as e:
            print(f"❌ Erro na análise: {str(e)}")
            self.current_results = []
            self.best_opportunity = None
        finally:
            self.is_analyzing = False
    
    def calculate_assertiveness(self, row):
        """Cálculo de assertividade mais preciso"""
        base_score = row.conf * 100
        
        # Fatores de melhoria
        buy_sell_diff = abs(row.p_buy - row.p_sell) * 100
        if buy_sell_diff > 25: base_score += 12
        elif buy_sell_diff > 20: base_score += 8
        elif buy_sell_diff > 15: base_score += 4
            
        # Indicadores técnicos
        if hasattr(row, 'adx') and row.adx > 30: base_score += 8
        if hasattr(row, 'rsi') and 40 <= row.rsi <= 60: base_score += 5
        
        # Volatilidade (menos é melhor)
        if hasattr(row, 'volatility') and row.volatility < 0.02: base_score += 5
        
        return min(round(base_score, 1), 100)
    
    def calculate_entry_time(self, horizon):
        now = datetime.now(timezone.utc)
        entry_time = now.replace(second=0, microsecond=0) + timedelta(minutes=horizon)
        return entry_time.strftime("%H:%M UTC")

manager = AnalysisManager()

# Página principal
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 IA Signal Pro</title>
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
                max-width: 800px; 
                margin: 0 auto; 
            }
            .card { 
                background: #1a1a2e; 
                border: 2px solid #3498db; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 10px 0; 
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
                padding: 10px; 
                border-radius: 5px; 
                margin: 5px 0; 
            }
            .buy { color: #2ecc71; }
            .sell { color: #e74c3c; }
            .technical-override { 
                border-left: 4px solid #f39c12 !important; 
            }
            .volatility-high { color: #e74c3c; }
            .volatility-low { color: #2ecc71; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🚀 IA Signal Pro - Web</h1>
                <p><em>Versão de teste - Conectado ao Railway</em></p>
                
                <input type="text" id="symbols" value="BTC/USDT,ETH/USDT,ADA/USDT" placeholder="Digite os símbolos...">
                <select id="sims">
                    <option value="500">500 simulações</option>
                    <option value="1000" selected>1000 simulações</option>
                    <option value="1500">1500 simulações</option>
                </select>
                <select id="adx">
                    <option value="20">ADX ≥ 20</option>
                    <option value="25" selected>ADX ≥ 25</option>
                    <option value="30">ADX ≥ 30</option>
                </select>
                
                <button onclick="analyze()" id="analyzeBtn">🎯 ANALISAR AGORA</button>
            </div>

            <div class="card">
                <h2>🎖️ MELHOR OPORTUNIDADE</h2>
                <div id="bestResult">Aguardando análise...</div>
            </div>

            <div class="card">
                <h2>📈 TODOS OS SINAIS</h2>
                <div id="allResults">Execute uma análise para ver os resultados</div>
            </div>

            <div class="card">
                <h3>ℹ️ STATUS</h3>
                <div id="status">Conectado ao servidor</div>
            </div>
        </div>

        <script>
            let isChecking = false;
            
            async function analyze() {
                const btn = document.getElementById('analyzeBtn');
                btn.disabled = true;
                btn.textContent = '⏳ ANALISANDO...';

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
                        isChecking = true;
                        checkResults();
                    } else {
                        alert('Erro: ' + data.error);
                        resetAnalyzeButton();
                    }

                } catch (error) {
                    alert('Erro de conexão: ' + error.message);
                    resetAnalyzeButton();
                }
            }

            function resetAnalyzeButton() {
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('analyzeBtn').textContent = '🎯 ANALISAR AGORA';
                isChecking = false;
            }

            async function checkResults() {
                if (!isChecking) return;
                
                try {
                    const response = await fetch('/api/results');
                    const data = await response.json();

                    if (data.success) {
                        updateResults(data);
                        
                        if (data.is_analyzing) {
                            // Continua verificando se ainda está analisando
                            setTimeout(checkResults, 2000);
                        } else {
                            // Análise concluída - para de verificar e reativa botão
                            resetAnalyzeButton();
                            document.getElementById('status').textContent = '✅ Análise concluída - Pronto para nova análise';
                        }
                    }
                } catch (error) {
                    console.error('Erro:', error);
                    // Em caso de erro, tenta mais algumas vezes depois para
                    setTimeout(checkResults, 3000);
                }
            }

            function updateResults(data) {
                // Melhor oportunidade
                if (data.best) {
                    const best = data.best;
                    const overrideClass = best.technical_override ? 'technical-override' : '';
                    const volatilityClass = best.volatility > 3 ? 'volatility-high' : 'volatility-low';
                    
                    document.getElementById('bestResult').innerHTML = `
                        <div class="results ${overrideClass}" style="border-left: 4px solid ${best.direction === 'buy' ? '#2ecc71' : '#e74c3c'}">
                            <strong>${best.symbol} T+${best.horizon}</strong>
                            <span class="${best.direction}">
                                ${best.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'} 
                                ${best.technical_override ? '⚠️' : ''}
                            </span>
                            <br>
                            Compra: <strong>${best.p_buy}%</strong> | Venda: <strong>${best.p_sell}%</strong><br>
                            Confiança: <strong>${best.confidence}%</strong> | Assertividade: <strong>${best.assertiveness}%</strong><br>
                            ADX: ${best.adx} | RSI: ${best.rsi}<br>
                            Volatilidade: <span class="${volatilityClass}">${best.volatility}%</span> | 
                            Tendência: ${best.trend_strength}%<br>
                            Entrada: ${best.entry_time}
                            ${best.technical_override ? '<br><small style="color: #f39c12;">⚠ Decisão Técnica Aplicada</small>' : ''}
                            <br><em>Análise: ${data.analysis_time}</em>
                        </div>
                    `;
                }

                // Todos os sinais (APENAS MELHOR DE CADA ATIVO)
                if (data.results.length > 0) {
                    let html = '';
                    data.results.sort((a, b) => b.confidence - a.confidence);
                    
                    data.results.forEach(result => {
                        const overrideClass = result.technical_override ? 'technical-override' : '';
                        const volatilityClass = result.volatility > 3 ? 'volatility-high' : 'volatility-low';
                        
                        html += `
                        <div class="results ${overrideClass}">
                            <strong>${result.symbol} T+${result.horizon}</strong>
                            <span class="${result.direction}">
                                ${result.direction == 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'} 
                                ${result.technical_override ? '⚠️' : ''}
                            </span>
                            | Compra: ${result.p_buy}% | Venda: ${result.p_sell}% |
                            Conf: ${result.confidence}% | Assert: ${result.assertiveness}% |
                            ADX: ${result.adx} | RSI: ${result.rsi} |
                            Vol: <span class="${volatilityClass}">${result.volatility}%</span>
                            ${result.technical_override ? '<br><small style="color: #f39c12;">⚠ Decisão Técnica</small>' : ''}
                        </div>`;
                    });
                    
                    document.getElementById('allResults').innerHTML = html;
                }
                
                // Atualiza status
                if (data.is_analyzing) {
                    document.getElementById('status').textContent = '⏳ Analisando...';
                } else if (data.results.length > 0) {
                    document.getElementById('status').textContent = `✅ ${data.results.length} ativos analisados - Pronto para nova análise`;
                }
            }

            // Verificar status inicial
            checkResults();
        </script>
    </body>
    </html>
    '''

# API Routes
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
            'message': 'Análise iniciada...',
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
        'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        'results_available': len(manager.current_results) > 0,
        'is_analyzing': manager.is_analyzing
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 IA Signal Pro iniciando...")
    print("📍 Será disponível em: https://seu-app.up.railway.app")
    app.run(host='0.0.0.0', port=port, debug=False)
