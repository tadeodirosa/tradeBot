#!/usr/bin/env python3
"""
🔥 SIMULADOR AGRESIVO DE FUTUROS
Diseñado para generar 10-20% retornos con 45x leverage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.futures_simulator import FuturesSimulator, FUTURES_CONFIG
from config_aggressive import FUTURES_CONFIG_AGGRESSIVE, calculate_aggressive_position_size
import argparse
from datetime import datetime, timedelta

def create_aggressive_simulator(symbol="ETH_USDT", days=180):
    """Crea simulador con configuración agresiva"""
    
    # Configuración agresiva
    config = FUTURES_CONFIG_AGGRESSIVE.copy()
    
    print(f"🔥 CONFIGURACIÓN AGRESIVA ACTIVADA")
    print(f"   Posición por trade: ${config['position_size_usd']} USD")
    print(f"   Leverage: {config['leverage']}x")
    print(f"   Take Profit: {config['take_profit_atr_mult']}x ATR")
    print(f"   Stop Loss: {config['stop_loss_atr_mult']}x ATR")
    print(f"   Min Score: {config['min_buy_score']}")
    print(f"   Target mensual: 8%+ (96%+ anual)")
    print()
    
    # Crear simulador con configuración agresiva
    simulator = FuturesSimulator(
        initial_balance=config['initial_balance'],
        leverage=config['leverage'],
        position_size_usd=config['position_size_usd']
    )
    
    # Actualizar configuración interna
    FUTURES_CONFIG.update({
        'take_profit_atr_mult': config['take_profit_atr_mult'],
        'stop_loss_atr_mult': config['stop_loss_atr_mult'],
        'min_buy_score': config['min_buy_score'],
        'min_sell_score': config['min_sell_score'],
        'min_confidence': config['min_confidence']
    })
    
    return simulator

def run_aggressive_backtest(symbol="ETH_USDT", days=180):
    """Ejecuta backtest agresivo"""
    
    simulator = create_aggressive_simulator(symbol, days)
    
    # Cargar datos
    from core.data_handler import load_ohlcv_data
    
    try:
        df = load_ohlcv_data(symbol)
        if df is None or len(df) == 0:
            print(f"❌ No se pudieron cargar datos para {symbol}")
            return None
            
        # Limitar datos a los últimos X días
        if days > 0:
            end_date = df.index[-1]
            start_date = end_date - timedelta(days=days)
            df = df[df.index >= start_date]
            print(f"📊 LIMITADO A ÚLTIMOS {days} DÍAS")
            
        print(f"✅ Datos OHLCV cargados: {len(df)} barras desde {df.index[0]} hasta {df.index[-1]}")
        
        # Ejecutar backtest agresivo
        from src.analyzer_v10 import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        
        total_bars = len(df)
        signals_generated = 0
        bars_analyzed = 0
        
        print(f"📊 Período: {df.index[0]} a {df.index[-1]} ({len(df)} barras)")
        
        # Procesar cada barra
        for i in range(50, len(df)):  # Empezar en barra 50 para tener historia suficiente
            bars_analyzed += 1
            current_data = df.iloc[:i+1]
            
            try:
                # Análisis técnico
                analysis = analyzer.analyze(current_data)
                
                if analysis and 'signal' in analysis:
                    signal = analysis['signal']
                    
                    # Filtros agresivos
                    if signal['action'] == 'BUY' and signal['score'] >= FUTURES_CONFIG['min_buy_score']:
                        signals_generated += 1
                        
                        # Calcular stops dinámicos
                        current_price = current_data['close'].iloc[-1]
                        atr = current_data['high'].rolling(10).max() - current_data['low'].rolling(10).min()
                        current_atr = atr.iloc[-1] if not atr.empty else current_price * 0.02
                        
                        take_profit = current_price + (current_atr * FUTURES_CONFIG['take_profit_atr_mult'])
                        stop_loss = current_price - (current_atr * FUTURES_CONFIG['stop_loss_atr_mult'])
                        
                        # Abrir posición
                        simulator.open_position(
                            'LONG',
                            current_price,
                            datetime.now(),
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"Score: {signal['score']:.0f} | Conf: {signal['confidence']:.2f}"
                        )
                        
                        print(f"[{i:3d}] BUY | Score: {signal['score']:.0f} | Conf: {signal['confidence']:.2f} | Price: ${current_price:.2f}")
                        
                    elif signal['action'] == 'SELL' and signal['score'] <= FUTURES_CONFIG['min_sell_score']:
                        signals_generated += 1
                        
                        # Calcular stops para SHORT
                        current_price = current_data['close'].iloc[-1]
                        atr = current_data['high'].rolling(10).max() - current_data['low'].rolling(10).min()
                        current_atr = atr.iloc[-1] if not atr.empty else current_price * 0.02
                        
                        take_profit = current_price - (current_atr * FUTURES_CONFIG['take_profit_atr_mult'])
                        stop_loss = current_price + (current_atr * FUTURES_CONFIG['stop_loss_atr_mult'])
                        
                        # Abrir posición SHORT
                        simulator.open_position(
                            'SHORT',
                            current_price,
                            datetime.now(),
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"Score: {signal['score']:.0f} | Conf: {signal['confidence']:.2f}"
                        )
                        
                        print(f"[{i:3d}] SELL | Score: {signal['score']:.0f} | Conf: {signal['confidence']:.2f} | Price: ${current_price:.2f}")
                
                # Actualizar posiciones existentes
                current_price = current_data['close'].iloc[-1]
                current_high = current_data['high'].iloc[-1]
                current_low = current_data['low'].iloc[-1]
                
                simulator.update_positions(current_price, current_high, current_low)
                
            except Exception as e:
                print(f"⚠️ Error en barra {i}: {str(e)}")
                continue
        
        # Mostrar estadísticas
        print(f"\n📊 ESTADÍSTICAS DE ANÁLISIS:")
        print(f"   Barras analizadas: {bars_analyzed} de {total_bars} barras totales")
        print(f"   Señales generadas: {signals_generated}")
        print(f"   Ratio señales: {signals_generated/bars_analyzed*100:.1f}% de las barras analizadas")
        
        # Resultados finales
        results = simulator.get_results()
        
        # Calcular retorno anualizado 
        total_return = results['total_return']
        period_months = days / 30.44  # Promedio días por mes
        monthly_return = total_return / period_months if period_months > 0 else 0
        annual_return = monthly_return * 12
        
        print(f"\n🔥 RESULTADOS AGRESIVOS - BACKTEST DE FUTUROS:")
        print("=" * 60)
        print(f"🔧 CONFIGURACIÓN AGRESIVA:")
        print(f"   Balance inicial: ${results['initial_balance']:,.2f}")
        print(f"   Leverage: {simulator.leverage}x")
        print(f"   Tamaño posición: ${simulator.position_size_usd} USD")
        print(f"   Margen por posición: ${simulator.margin_per_position:.2f} USD")
        print()
        print(f"💰 RENDIMIENTO AGRESIVO:")
        print(f"   P&L Total: ${results['total_pnl']:+.2f}")
        print(f"   Retorno {days} días: {total_return:.2f}%")
        print(f"   Retorno mensual: {monthly_return:.2f}%")
        print(f"   Retorno anualizado: {annual_return:.1f}%")
        print(f"   Balance final: ${results['final_balance']:,.2f}")
        print()
        print(f"📊 ESTADÍSTICAS DE TRADING:")
        print(f"   Total trades: {results['total_trades']}")
        print(f"   Trades ganadores: {results['winning_trades']}")
        print(f"   Trades perdedores: {results['losing_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Mejor trade: ${results['best_trade']:+.2f}")
        print(f"   Peor trade: ${results['worst_trade']:+.2f}")
        print(f"   Trade promedio: ${results['average_trade']:+.2f}")
        print()
        print(f"🛡️ MÉTRICAS DE RIESGO:")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print()
        print(f"💸 COSTOS:")
        print(f"   Comisiones totales: ${results['total_commissions']:.2f}")
        print()
        
        # Evaluación agresiva
        grade = "F"
        if annual_return >= 96 and results['sharpe_ratio'] >= 2.0:
            grade = "A+ (EXCELENTE - Target alcanzado)"
        elif annual_return >= 60 and results['sharpe_ratio'] >= 1.5:
            grade = "A (MUY BUENO)"
        elif annual_return >= 30 and results['sharpe_ratio'] >= 1.0:
            grade = "B (BUENO)"
        elif annual_return >= 15:
            grade = "C (REGULAR)"
        elif annual_return >= 5:
            grade = "D (MALO)"
            
        print(f"🏆 EVALUACIÓN AGRESIVA:")
        print(f"   Calificación: {grade}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"❌ Error durante el backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='🔥 Simulador Agresivo de Futuros')
    parser.add_argument('symbol', nargs='?', default='ETH_USDT', help='Símbolo a analizar (ej: ETH_USDT)')
    parser.add_argument('--leverage', type=int, default=45, help='Apalancamiento (default: 45)')
    parser.add_argument('--days', type=int, default=180, help='Días a analizar (default: 180)')
    
    args = parser.parse_args()
    
    print(f"🔥 Simulador Agresivo de futuros para {args.symbol} en timeframe 4h\n")
    
    # Ejecutar backtest agresivo
    results = run_aggressive_backtest(args.symbol, args.days)
    
    if results:
        print(f"\n✅ Backtest agresivo completado para {args.symbol}")
    else:
        print(f"\n❌ Error en backtest agresivo para {args.symbol}")