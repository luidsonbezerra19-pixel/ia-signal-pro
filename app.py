from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime, timezone, timedelta
import os
import random

# SIMULAÇÃO do seu sistema - funciona sem seu código core
class MockCore:
    @staticmethod
    def analyze_symbols(symbols, sims=1000, only_adx=None):
        print(f"🔍 Analisando: {symbols}")
        
        # Dados de exemplo - REMOVA depois quando colocar seu código real
        mock_results = []
        symbols_list = [s.strip() for s in symbols if s.strip()]
        
        for i, symbol in enumerate(symbols_list):
            for horizon in [1, 2, 3]:
                # Gera dados aleatórios para teste
                direction = "buy" if random.random() > 0.5 else "sell"
                confidence = round(0.5 + (random.random() * 0.4), 3)  # 50% a 90%
                
                class Row:
                    def __init__(self):
                        self.symbol = symbol
                        self.h = horizon
                        self.direction = direction
                        self.p_buy = round(0.3 + (random.random() * 0.5), 3)
                        self.p_sell = round(1 - self.p_buy, 3)
                        self.conf = confidence
                        self.adx = round(20 + (random.random() * 20), 1)
                        self.rsi = round(30 + (random.random() * 40), 1)
                        self.price = round(100 + (random.random() * 100), 6)
                        self.ts = datetime.now().strftime("%H:%M:%S")
                
                mock_results.append(Row())
        
        best = max(mock_results, key=lambda x: x.conf) if mock_results else None
        return mock_results, best

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
            
            # Converter resultados
            self.current_results = []
            for row in all_rows:
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
                    'assertiveness': self.calculate_assertiveness(row)
                }
                self.current_results.append(result)
            
            # Melhor oportunidade
            if best:
                self.best_opportunity = {
                    'symbol': best.symbol,
                    'horizon': best.h,
                    'direction': best.direction,
                    'confidence': round(best.conf * 100, 1),
                    'p_buy': round(best.p_buy * 100, 1),
                    'p_sell': round(best.p_sell * 100, 1),
                    'adx': round(getattr(best, 'adx', 0), 1),
                    'rsi': round(getattr(best, 'rsi', 0), 1),
                    'price': round(best.price, 6),
                    'assertiveness': self.calculate_assertiveness(best),
                    'entry_time': self.calculate_entry_time(best.h)
                }
            
            self.analysis_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"✅ Análise concluída: {len(self.current_results)} sinais")
            
        except Exception as e:
            print(f"❌ Erro na análise: {str(e)}")
            self.current_results = []
            self.best_opportunity = None
        finally:
            self.is_analyzing = False
    
    def calculate_assertiveness(self, row):
        score = row.conf * 100
        buy_sell_diff = abs(row.p_buy - row.p_sell) * 100
        
        if buy_sell_diff > 20: score += 10
        elif buy_sell_diff > 15: score += 5
            
        if hasattr(row, 'adx') and row.adx > 25: score += 5
            
        return min(round(score, 1), 100)
    
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
                        checkResults();
                    } else {
                        alert('Erro: ' + data.error);
                        btn.disabled = false;
                        btn.textContent = '🎯 ANALISAR AGORA';
                    }

                } catch (error) {
                    alert('Erro de conexão: ' + error.message);
                    btn.disabled = false;
                    btn.textContent = '🎯 ANALISAR AGORA';
                }
            }

            async function checkResults() {
                try {
                    const response = await fetch('/api/results');
                    const data = await response.json();

                    if (data.success) {
                        updateResults(data);
                        
                        if (data.is_analyzing) {
                            // Continua verificando se ainda está analisando
                            setTimeout(checkResults, 2000);
                        } else {
                            document.getElementById('analyzeBtn').disabled = false;
                            document.getElementById('analyzeBtn').textContent = '🎯 ANALISAR AGORA';
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
                    document.getElementById('bestResult').innerHTML = `
                        <div class="results" style="border-left: 4px solid ${best.direction === 'buy' ? '#2ecc71' : '#e74c3c'}">
                            <strong>${best.symbol} T+${best.horizon}</strong><br>
                            Ação: <span class="${best.direction}">${best.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'}</span><br>
                            Confiança: <strong>${best.confidence}%</strong><br>
                            ADX: ${best.adx} | RSI: ${best.rsi}<br>
                            Entrada: ${best.entry_time}<br>
                            <em>Análise: ${data.analysis_time}</em>
                        </div>
                    `;
                }

                // Todos os sinais
                if (data.results.length > 0) {
                    let html = '';
                    data.results.sort((a, b) => b.confidence - a.confidence);
                    
                    data.results.forEach(result => {
                        html += `
                            <div class="results">
                                <strong>${result.symbol} T+${result.horizon}</strong> | 
                                <span class="${result.direction}">${result.direction === 'buy' ? '🟢 COMPRAR' : '🔴 VENDER'}</span> | 
                                Conf: ${result.confidence}% | 
                                ADX: ${result.adx}
                            </div>
                        `;
                    });
                    
                    document.getElementById('allResults').innerHTML = html;
                    document.getElementById('status').textContent = `✅ ${data.results.length} sinais encontrados`;
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