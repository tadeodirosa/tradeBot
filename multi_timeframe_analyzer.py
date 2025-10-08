"""
Multi-Timeframe Analyzer - La evolución del TradeBot
===================================================

ESTRATEGIA: 4H para dirección + 1H para timing preciso
- Análisis de tendencia en 4H (dirección principal)
- Condiciones de entrada en 1H (timing sniper)
- Gestión de riesgo basada en ATR 4H (objetivos realistas)

OBJETIVO: Reducir drawdown manteniendo o mejorando ROI
- Target: Drawdown <35% (vs 69.2% actual)
- Target: ROI >300% (vs 427.86% actual)
- Target: Win Rate >55% (vs 50.8% actual)
"""

import requests
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar componentes existentes
from signal_tracker import SignalTracker

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.config = {
            'timeframe_trend': '4h',      # Para análisis de tendencia
            'timeframe_entry': '1h',      # Para timing de entrada
            'limit_4h': 50,               # Datos suficientes para tendencia
            'limit_1h': 100,              # Datos suficientes para entrada
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'atr_period': 14,
            'stop_loss_atr_multiplier': 0.8,  # Más ajustado con entradas precisas
            'take_profit_atr_multiplier': 2.0,  # Mantener targets realistas
            'risk_per_trade': 0.03,
            'max_position_size': 0.25
        }
        
        # Mapeo de timeframes para API de Binance
        self.binance_intervals = {
            '1h': '1h',
            '4h': '4h'
        }
        
        # Inicializar tracker de señales
        self.signal_tracker = SignalTracker()
    
    def get_binance_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Obtener datos de Binance para un timeframe específico."""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': self.binance_intervals[timeframe],
                'limit': limit
            }
            
            print(f"📡 Obteniendo datos {timeframe}: {symbol} ({limit} velas)...")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Error API Binance: {response.status_code}")
            
            data = response.json()
            
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos y timestamp
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Datos {timeframe} obtenidos: {len(df)} velas")
            print(f"📅 Última vela {timeframe}: {df.index[-1]}")
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos {timeframe}: {e}")
            raise
    
    def calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Calcular indicadores técnicos para un timeframe específico."""
        
        # EMA usando pandas
        ema_9 = df['close'].ewm(span=self.config['ema_fast']).mean().iloc[-1]
        ema_21 = df['close'].ewm(span=self.config['ema_slow']).mean().iloc[-1]
        
        # RSI
        close_prices = df['close']
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.config['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.config['rsi_period']).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # ATR (método Wilder)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr_values = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        # ATR usando smoothing de Wilder
        first_atr = np.mean(tr_values[:self.config['atr_period']])
        atr_current = first_atr
        
        for i in range(self.config['atr_period'], len(tr_values)):
            atr_current = (atr_current * (self.config['atr_period'] - 1) + tr_values[i]) / self.config['atr_period']
        
        # ATR como porcentaje del precio
        current_price = df['close'].iloc[-1]
        atr_percentage = (atr_current / current_price) * 100
        
        # Momentum (últimas 3 velas)
        momentum = 0
        if len(close_prices) >= 3:
            momentum = (current_price - close_prices.iloc[-3]) / close_prices.iloc[-3] * 100
        
        return {
            'timeframe': timeframe,
            'price': current_price,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'rsi': current_rsi,
            'atr': atr_current,
            'atr_percentage': atr_percentage,
            'momentum': momentum
        }
    
    def analyze_trend_4h(self, indicators_4h: dict) -> dict:
        """Analizar tendencia en 4H para determinar dirección principal."""
        
        trend_analysis = {
            'direction': None,
            'strength': 0,
            'reasons': [],
            'conditions_met': 0
        }
        
        price = indicators_4h['price']
        ema_9 = indicators_4h['ema_9']
        ema_21 = indicators_4h['ema_21']
        rsi = indicators_4h['rsi']
        atr_pct = indicators_4h['atr_percentage']
        momentum = indicators_4h['momentum']
        
        print(f"\n🔍 ANÁLISIS DE TENDENCIA 4H:")
        print(f"💰 Precio: ${price:.4f}")
        print(f"📈 EMA9: ${ema_9:.4f} | EMA21: ${ema_21:.4f}")
        print(f"⚡ RSI: {rsi:.1f} | ATR: {atr_pct:.2f}%")
        print(f"🚀 Momentum: {momentum:.2f}%")
        
        # Condiciones para TENDENCIA ALCISTA (más flexibles)
        bullish_conditions = 0
        bullish_reasons = []
        
        if ema_9 > ema_21:
            bullish_conditions += 1
            bullish_reasons.append("✅ EMA9 > EMA21 (tendencia alcista)")
        else:
            bullish_reasons.append("❌ EMA9 ≤ EMA21")
        
        if rsi < 70:  # No sobrecomprado
            bullish_conditions += 1
            bullish_reasons.append(f"✅ RSI no sobrecomprado ({rsi:.1f})")
        else:
            bullish_reasons.append(f"❌ RSI sobrecomprado ({rsi:.1f})")
        
        if atr_pct > 1.0:  # Volatilidad mínima
            bullish_conditions += 1
            bullish_reasons.append("✅ Volatilidad presente")
        else:
            bullish_reasons.append("❌ Volatilidad insuficiente")
        
        # Condiciones para TENDENCIA BAJISTA (más flexibles)
        bearish_conditions = 0
        bearish_reasons = []
        
        if ema_9 < ema_21:
            bearish_conditions += 1
            bearish_reasons.append("✅ EMA9 < EMA21 (tendencia bajista)")
        else:
            bearish_reasons.append("❌ EMA9 ≥ EMA21")
        
        if rsi > 30:  # No sobrevendido
            bearish_conditions += 1
            bearish_reasons.append(f"✅ RSI no sobrevendido ({rsi:.1f})")
        else:
            bearish_reasons.append(f"❌ RSI sobrevendido ({rsi:.1f})")
        
        if atr_pct > 1.0:  # Volatilidad mínima
            bearish_conditions += 1
            bearish_reasons.append("✅ Volatilidad presente")
        else:
            bearish_reasons.append("❌ Volatilidad insuficiente")
        
        # Determinar tendencia (2+ condiciones requeridas)
        if bullish_conditions >= 2:
            trend_analysis['direction'] = 'BULLISH'
            trend_analysis['strength'] = bullish_conditions * 33
            trend_analysis['reasons'] = bullish_reasons
            trend_analysis['conditions_met'] = bullish_conditions
            print(f"📈 TENDENCIA 4H: ALCISTA ({bullish_conditions}/3)")
            
        elif bearish_conditions >= 2:
            trend_analysis['direction'] = 'BEARISH'
            trend_analysis['strength'] = bearish_conditions * 33
            trend_analysis['reasons'] = bearish_reasons
            trend_analysis['conditions_met'] = bearish_conditions
            print(f"📉 TENDENCIA 4H: BAJISTA ({bearish_conditions}/3)")
            
        else:
            print(f"😐 TENDENCIA 4H: NEUTRAL (Bull:{bullish_conditions}/3, Bear:{bearish_conditions}/3)")
            trend_analysis['reasons'] = bullish_reasons + bearish_reasons
        
        return trend_analysis
    
    def analyze_entry_1h(self, indicators_1h: dict, trend_direction: str) -> dict:
        """Analizar condiciones de entrada en 1H según tendencia 4H."""
        
        entry_analysis = {
            'signal': None,
            'strength': 0,
            'reasons': [],
            'conditions_met': 0
        }
        
        if not trend_direction:
            print(f"\n⏳ SIN ANÁLISIS 1H: Tendencia 4H no definida")
            return entry_analysis
        
        price = indicators_1h['price']
        ema_9 = indicators_1h['ema_9']
        ema_21 = indicators_1h['ema_21']
        rsi = indicators_1h['rsi']
        atr_pct = indicators_1h['atr_percentage']
        momentum = indicators_1h['momentum']
        
        print(f"\n🎯 ANÁLISIS DE ENTRADA 1H (Tendencia {trend_direction}):")
        print(f"💰 Precio: ${price:.4f}")
        print(f"📈 EMA9: ${ema_9:.4f} | EMA21: ${ema_21:.4f}")
        print(f"⚡ RSI: {rsi:.1f} | ATR: {atr_pct:.2f}%")
        print(f"🚀 Momentum: {momentum:.2f}%")
        
        if trend_direction == 'BULLISH':
            # Condiciones de entrada LONG (más selectivas)
            long_conditions = 0
            long_reasons = []
            
            # RSI en zona de entrada (más amplio que original)
            if 20 <= rsi <= 50:
                long_conditions += 1
                long_reasons.append(f"✅ RSI zona entrada ({rsi:.1f})")
            else:
                long_reasons.append(f"❌ RSI fuera zona entrada ({rsi:.1f})")
            
            # Momentum no muy negativo
            if momentum > -5.0:
                long_conditions += 1
                long_reasons.append(f"✅ Momentum controlado ({momentum:.2f}%)")
            else:
                long_reasons.append(f"❌ Momentum muy negativo ({momentum:.2f}%)")
            
            # Volatilidad suficiente (más permisivo)
            if atr_pct > 0.8:
                long_conditions += 1
                long_reasons.append("✅ Volatilidad adecuada")
            else:
                long_reasons.append("❌ Volatilidad insuficiente")
            
            # Generar señal LONG (2+ condiciones requeridas)
            if long_conditions >= 2:
                entry_analysis['signal'] = 'LONG'
                entry_analysis['strength'] = long_conditions * 50
                entry_analysis['reasons'] = long_reasons
                entry_analysis['conditions_met'] = long_conditions
                print(f"🚨 SEÑAL 1H: LONG ({long_conditions}/3)")
            else:
                print(f"⏳ Sin señal LONG ({long_conditions}/3)")
                entry_analysis['reasons'] = long_reasons
        
        elif trend_direction == 'BEARISH':
            # Condiciones de entrada SHORT (más selectivas)
            short_conditions = 0
            short_reasons = []
            
            # RSI en zona de entrada (más amplio que original)
            if 50 <= rsi <= 80:
                short_conditions += 1
                short_reasons.append(f"✅ RSI zona entrada ({rsi:.1f})")
            else:
                short_reasons.append(f"❌ RSI fuera zona entrada ({rsi:.1f})")
            
            # Momentum no muy positivo
            if momentum < 5.0:
                short_conditions += 1
                short_reasons.append(f"✅ Momentum controlado ({momentum:.2f}%)")
            else:
                short_reasons.append(f"❌ Momentum muy positivo ({momentum:.2f}%)")
            
            # Volatilidad suficiente (más permisivo)
            if atr_pct > 0.8:
                short_conditions += 1
                short_reasons.append("✅ Volatilidad adecuada")
            else:
                short_reasons.append("❌ Volatilidad insuficiente")
            
            # Generar señal SHORT (2+ condiciones requeridas)
            if short_conditions >= 2:
                entry_analysis['signal'] = 'SHORT'
                entry_analysis['strength'] = short_conditions * 50
                entry_analysis['reasons'] = short_reasons
                entry_analysis['conditions_met'] = short_conditions
                print(f"🚨 SEÑAL 1H: SHORT ({short_conditions}/3)")
            else:
                print(f"⏳ Sin señal SHORT ({short_conditions}/3)")
                entry_analysis['reasons'] = short_reasons
        
        return entry_analysis
    
    def analyze_signal(self, symbol: str) -> dict:
        """Análisis completo multi-timeframe."""
        
        # 1. Obtener datos de ambos timeframes
        df_4h = self.get_binance_data(symbol, '4h', self.config['limit_4h'])
        df_1h = self.get_binance_data(symbol, '1h', self.config['limit_1h'])
        
        # 2. Calcular indicadores
        indicators_4h = self.calculate_indicators(df_4h, '4h')
        indicators_1h = self.calculate_indicators(df_1h, '1h')
        
        # 3. Analizar tendencia en 4H
        trend_analysis = self.analyze_trend_4h(indicators_4h)
        
        # 4. Analizar entrada en 1H según tendencia 4H
        entry_analysis = self.analyze_entry_1h(indicators_1h, trend_analysis['direction'])
        
        # 5. Compilar resultado final
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': indicators_1h['price'],  # Precio actual (1H más preciso)
            'signal': entry_analysis['signal'],
            'strength': entry_analysis['strength'],
            'reasons': entry_analysis['reasons'],
            'trend_4h': trend_analysis,
            'entry_1h': entry_analysis,
            'indicators_4h': indicators_4h,
            'indicators_1h': indicators_1h
        }
        
        # 6. Calcular niveles de trading si hay señal
        if result['signal']:
            # Usar ATR 4H para niveles (más estables)
            atr_4h = indicators_4h['atr']
            current_price = indicators_1h['price']
            
            stop_loss_distance = atr_4h * self.config['stop_loss_atr_multiplier']
            
            if result['signal'] == 'LONG':
                stop_loss = current_price - stop_loss_distance
                take_profit = current_price + (stop_loss_distance * self.config['take_profit_atr_multiplier'])
            else:  # SHORT
                stop_loss = current_price + stop_loss_distance
                take_profit = current_price - (stop_loss_distance * self.config['take_profit_atr_multiplier'])
            
            # Risk management
            balance = 10000  # Balance ejemplo
            risk_amount = balance * self.config['risk_per_trade']
            position_size = min(
                risk_amount / stop_loss_distance,
                balance * self.config['max_position_size'] / current_price
            )
            
            result.update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'atr_4h_used': atr_4h
            })
        
        return result
    
    def print_signal_report(self, result: dict):
        """Imprimir reporte completo multi-timeframe."""
        
        print(f"\n" + "="*70)
        print(f"🎯 ANÁLISIS MULTI-TIMEFRAME - {result['symbol']}")
        print(f"📅 {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Precio actual: ${result['price']:.4f}")
        print("="*70)
        
        # Mostrar análisis de tendencia 4H
        trend = result['trend_4h']
        print(f"\n📊 TENDENCIA 4H:")
        if trend['direction']:
            print(f"   Dirección: {trend['direction']} ({trend['strength']}%)")
            print(f"   Condiciones: {trend['conditions_met']}/3")
        else:
            print(f"   Dirección: NEUTRAL")
        
        # Mostrar análisis de entrada 1H
        entry = result['entry_1h']
        print(f"\n🎯 ENTRADA 1H:")
        if entry['signal']:
            print(f"   Señal: {entry['signal']} ({entry['strength']}%)")
            print(f"   Condiciones: {entry['conditions_met']}/3")
        else:
            print(f"   Señal: SIN ENTRADA")
        
        # Resultado final
        if result['signal']:
            print(f"\n🚨 SEÑAL FINAL: {result['signal']}")
            print(f"💪 Fuerza combinada: {result['strength']}%")
            print(f"🎯 Stop Loss: ${result['stop_loss']:.4f}")
            print(f"🏆 Take Profit: ${result['take_profit']:.4f}")
            print(f"📦 Tamaño posición: {result['position_size']:.4f} {result['symbol'].replace('USDT', '')}")
            print(f"⚠️ Riesgo: ${result['risk_amount']:.2f}")
            print(f"📏 ATR 4H usado: ${result['atr_4h_used']:.4f}")
            
            print(f"\n📋 RAZONES DE LA SEÑAL:")
            for reason in result['reasons']:
                if "✅" in reason:
                    print(f"   {reason}")
            
            # Guardar señal
            signal_id = self.signal_tracker.save_signal(result)
            print(f"\n💾 Señal guardada para tracking: {signal_id}")
                    
        else:
            print(f"\n⏳ SIN SEÑAL MULTI-TIMEFRAME")
            print(f"💡 Esperando alineación de tendencia 4H + entrada 1H")
        
        print(f"\n📊 SISTEMA MULTI-TIMEFRAME:")
        print(f"   🎯 Objetivo: Reducir drawdown <35% manteniendo ROI >300%")
        print(f"   ⚖️ Balance: Precisión (1H) + Estabilidad (4H)")
        print("="*70)


def main():
    """Función principal para análisis multi-timeframe."""
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='TradeBot Multi-Timeframe Analyzer')
    parser.add_argument('--symbol', '-s',
                       default='LINKUSDT',
                       help='Símbolo a analizar (default: LINKUSDT)')
    
    # Si hay argumentos de línea de comandos, usarlos
    if len(sys.argv) > 1:
        args = parser.parse_args()
        symbol = args.symbol
        
        print(f"🚀 ANALIZADOR MULTI-TIMEFRAME")
        print(f"🔧 Modo línea de comandos:")
        print(f"   Símbolo: {symbol}")
        
    else:
        # Modo interactivo
        print(f"🚀 ANALIZADOR MULTI-TIMEFRAME")
        print(f"🎯 Estrategia: 4H tendencia + 1H entrada precisa")
        
        symbol_input = input("💰 Símbolo (enter para LINKUSDT): ").strip().upper()
        symbol = symbol_input or "LINKUSDT"
        
        if symbol.endswith('/USDT'):
            symbol = symbol.replace('/', '')  # Convertir BTC/USDT → BTCUSDT
    
    print(f"🎯 Objetivo: Drawdown <35% | ROI >300% | Win Rate >55%")
    print(f"⚖️ Balance: Precisión (1H) + Estabilidad (4H)")
    
    try:
        # Crear analyzer multi-timeframe
        analyzer = MultiTimeframeAnalyzer()
        
        # Analizar señal
        result = analyzer.analyze_signal(symbol)
        
        # Mostrar reporte
        analyzer.print_signal_report(result)
        
    except Exception as e:
        print(f"❌ Error en análisis multi-timeframe: {e}")


if __name__ == "__main__":
    main()