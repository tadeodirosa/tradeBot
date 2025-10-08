"""
Analizador en Tiempo Real para Señales de Trading
Usa las MISMAS condiciones optimizadas del backtester verified_backtester.py
ROI: 427.86% | Win Rate: 50.8% | Profit Factor: 1.46

NUEVO: Auto-guarda señales para tracking de performance
ACTUALIZADO: Soporte para múltiples timeframes (1h, 4h)
"""

import requests
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar el tracker de señales
from signal_tracker import SignalTracker

class RealTimeAnalyzer:
    def __init__(self, timeframe: str = '4h'):
        # Validar timeframe
        if timeframe not in ['1h', '4h']:
            raise ValueError(f"Timeframe '{timeframe}' no soportado. Use '1h' o '4h'")
        
        self.config = {
            'timeframe': timeframe,
            'limit': 100,  # Suficientes datos para indicadores
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'atr_period': 14,
            'stop_loss_atr_multiplier': 1.2,
            'risk_per_trade': 0.03,  # 3%
            'max_position_size': 0.25  # 25%
        }
        
        # Inicializar tracker de señales
        self.signal_tracker = SignalTracker()
        
        # Mapeo de timeframes para API de Binance
        self.binance_intervals = {
            '1h': '1h',
            '4h': '4h'
        }
    
    def get_binance_data(self, symbol: str) -> pd.DataFrame:
        """Obtener datos en tiempo real de Binance."""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': self.binance_intervals[self.config['timeframe']],
                'limit': self.config['limit']
            }
            
            print(f"📡 Obteniendo datos de {symbol} ({self.config['timeframe']})...")
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
            
            print(f"✅ Datos obtenidos: {len(df)} velas")
            print(f"📅 Última vela: {df.index[-1]}")
            print(f"💰 Precio actual: ${df['close'].iloc[-1]:.4f}")
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            raise
    
    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calcular indicadores técnicos (MISMO método que backtester)."""
        
        # EMA usando pandas (CORREGIDO del bug original)
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
        
        return {
            'price': current_price,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'rsi': current_rsi,
            'atr': atr_current,
            'atr_percentage': atr_percentage
        }
    
    def analyze_signal(self, symbol: str) -> dict:
        """Analizar señal actual para un símbolo."""
        
        # Obtener datos
        df = self.get_binance_data(symbol)
        
        # Calcular indicadores
        indicators = self.calculate_indicators(df)
        
        # Extraer valores
        current_price = indicators['price']
        ema_9 = indicators['ema_9']
        ema_21 = indicators['ema_21']
        rsi = indicators['rsi']
        atr = indicators['atr']
        atr_percentage = indicators['atr_percentage']
        
        print(f"\n📊 INDICADORES TÉCNICOS:")
        print(f"💰 Precio: ${current_price:.4f}")
        print(f"📈 EMA 9: ${ema_9:.4f}")
        print(f"📈 EMA 21: ${ema_21:.4f}")
        print(f"⚡ RSI: {rsi:.1f}")
        print(f"📏 ATR: ${atr:.4f} ({atr_percentage:.2f}%)")
        
        # APLICAR LAS MISMAS CONDICIONES OPTIMIZADAS DEL BACKTESTER
        close_prices = df['close']
        
        # Condiciones LONG (exactamente igual al backtester)
        long_conditions = 0
        long_reasons = []
        
        if current_price > ema_21:
            long_conditions += 1
            long_reasons.append("✅ Precio > EMA21")
        else:
            long_reasons.append("❌ Precio ≤ EMA21")
        
        if ema_9 > ema_21:
            long_conditions += 1
            long_reasons.append("✅ EMA9 > EMA21")
        else:
            long_reasons.append("❌ EMA9 ≤ EMA21")
        
        if 25 <= rsi <= 45:  # Zona de sobreventa controlada
            long_conditions += 1
            long_reasons.append(f"✅ RSI zona sobreventa ({rsi:.1f})")
        else:
            long_reasons.append(f"❌ RSI fuera zona sobreventa ({rsi:.1f})")
        
        if atr_percentage > 1.5:  # Mayor volatilidad
            long_conditions += 1
            long_reasons.append("✅ Volatilidad adecuada")
        else:
            long_reasons.append("❌ Volatilidad insuficiente")
        
        # Momentum positivo
        if len(close_prices) >= 3:
            recent_momentum = (current_price - close_prices.iloc[-3]) / close_prices.iloc[-3] * 100
            if recent_momentum > -2.0:
                long_conditions += 1
                long_reasons.append(f"✅ Momentum controlado ({recent_momentum:.2f}%)")
            else:
                long_reasons.append(f"❌ Momentum negativo ({recent_momentum:.2f}%)")
        
        # Condiciones SHORT (exactamente igual al backtester)
        short_conditions = 0
        short_reasons = []
        
        if current_price < ema_21:
            short_conditions += 1
            short_reasons.append("✅ Precio < EMA21")
        else:
            short_reasons.append("❌ Precio ≥ EMA21")
        
        if ema_9 < ema_21:
            short_conditions += 1
            short_reasons.append("✅ EMA9 < EMA21")
        else:
            short_reasons.append("❌ EMA9 ≥ EMA21")
        
        if 55 <= rsi <= 75:  # Zona de sobrecompra controlada
            short_conditions += 1
            short_reasons.append(f"✅ RSI zona sobrecompra ({rsi:.1f})")
        else:
            short_reasons.append(f"❌ RSI fuera zona sobrecompra ({rsi:.1f})")
        
        if atr_percentage > 1.5:
            short_conditions += 1
            short_reasons.append("✅ Volatilidad adecuada")
        else:
            short_reasons.append("❌ Volatilidad insuficiente")
        
        # Momentum negativo para SHORT
        if len(close_prices) >= 3:
            recent_momentum = (current_price - close_prices.iloc[-3]) / close_prices.iloc[-3] * 100
            if recent_momentum < 2.0:
                short_conditions += 1
                short_reasons.append(f"✅ Momentum controlado ({recent_momentum:.2f}%)")
            else:
                short_reasons.append(f"❌ Momentum muy positivo ({recent_momentum:.2f}%)")
        
        # DECISIÓN DE SEÑAL (4+ condiciones requeridas)
        signal_direction = None
        signal_strength = 0
        signal_reasons = []
        
        print(f"\n🔍 ANÁLISIS DE CONDICIONES:")
        print(f"📊 Condiciones LONG: {long_conditions}/5")
        for reason in long_reasons:
            print(f"   {reason}")
        
        print(f"\n📊 Condiciones SHORT: {short_conditions}/5")
        for reason in short_reasons:
            print(f"   {reason}")
        
        if long_conditions >= 4:
            signal_direction = 'LONG'
            signal_strength = min(100, long_conditions * 20)
            signal_reasons = long_reasons
            
        elif short_conditions >= 4:
            signal_direction = 'SHORT'
            signal_strength = min(100, short_conditions * 20)
            signal_reasons = short_reasons
        
        # Calcular niveles de trading
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': current_price,
            'signal': signal_direction,
            'strength': signal_strength,
            'reasons': signal_reasons,
            'indicators': indicators,
            'long_conditions': long_conditions,
            'short_conditions': short_conditions
        }
        
        if signal_direction:
            # Calcular niveles
            stop_loss_distance = atr * self.config['stop_loss_atr_multiplier']
            
            if signal_direction == 'LONG':
                stop_loss = current_price - stop_loss_distance
                take_profit = current_price + (stop_loss_distance * 2)  # R:R 1:2
            else:  # SHORT
                stop_loss = current_price + stop_loss_distance
                take_profit = current_price - (stop_loss_distance * 2)
            
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
                'risk_amount': risk_amount
            })
        
        return result
    
    def print_signal_report(self, result: dict):
        """Imprimir reporte de señal y guardar en tracking."""
        
        print(f"\n" + "="*60)
        print(f"🎯 ANÁLISIS DE SEÑAL - {result['symbol']}")
        print(f"📅 {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Precio: ${result['price']:.4f}")
        print("="*60)
        
        if result['signal']:
            print(f"\n🚨 SEÑAL DETECTADA: {result['signal']}")
            print(f"💪 Fuerza: {result['strength']}%")
            print(f"🎯 Stop Loss: ${result['stop_loss']:.4f}")
            print(f"🏆 Take Profit: ${result['take_profit']:.4f}")
            print(f"📦 Tamaño posición: {result['position_size']:.4f} {result['symbol'].replace('USDT', '')}")
            print(f"⚠️ Riesgo: ${result['risk_amount']:.2f}")
            
            print(f"\n📋 RAZONES DE LA SEÑAL:")
            for reason in result['reasons']:
                if "✅" in reason:
                    print(f"   {reason}")
            
            # 💾 GUARDAR SEÑAL EN TRACKING
            signal_id = self.signal_tracker.save_signal(result)
            print(f"\n💾 Señal guardada para tracking: {signal_id}")
                    
        else:
            print(f"\n⏳ SIN SEÑAL")
            print(f"📊 Condiciones LONG: {result['long_conditions']}/5 (necesitas 4+)")
            print(f"📊 Condiciones SHORT: {result['short_conditions']}/5 (necesitas 4+)")
            print(f"\n💡 Faltan condiciones para generar señal de calidad")
        
        print(f"\n📊 SISTEMA BASADO EN BACKTEST:")
        print(f"   ROI: 427.86% | Win Rate: 50.8% | Profit Factor: 1.46")
        print("="*60)


def main():
    """Función principal para análisis."""
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='TradeBot Real-Time Analyzer')
    parser.add_argument('--timeframe', '-t', 
                       choices=['1h', '4h'], 
                       default='4h',
                       help='Timeframe para análisis (default: 4h)')
    parser.add_argument('--symbol', '-s',
                       default='LINKUSDT',
                       help='Símbolo a analizar (default: LINKUSDT)')
    
    # Si hay argumentos de línea de comandos, usarlos
    if len(sys.argv) > 1:
        args = parser.parse_args()
        timeframe = args.timeframe
        symbol = args.symbol
        
        print(f"🚀 ANALIZADOR EN TIEMPO REAL")
        print(f"🔧 Modo línea de comandos:")
        print(f"   Timeframe: {timeframe}")
        print(f"   Símbolo: {symbol}")
        
    else:
        # Modo interactivo (mantener compatibilidad)
        print(f"🚀 ANALIZADOR EN TIEMPO REAL")
        print("⏰ Timeframes disponibles:")
        print("   1h - Análisis en 1 hora (más granular)")  
        print("   4h - Análisis en 4 horas (por defecto)")
        
        timeframe_input = input("🕐 Timeframe (1h/4h, enter para 4h): ").strip().lower()
        if timeframe_input in ['1h', '4h']:
            timeframe = timeframe_input
        else:
            timeframe = '4h'  # Default para compatibilidad
        
        symbol_input = input("💰 Símbolo (enter para LINKUSDT): ").strip().upper()
        symbol = symbol_input or "LINKUSDT"
        
        if symbol.endswith('/USDT'):
            symbol = symbol.replace('/', '')  # Convertir BTC/USDT → BTCUSDT
    
    print(f"🎯 Sistema optimizado: ROI 427.86% | Win Rate 50.8%")
    print(f"📈 Condiciones selectivas: 4+ condiciones requeridas")
    print(f"⏰ Timeframe seleccionado: {timeframe}")
    
    try:
        # Crear analyzer con timeframe especificado
        analyzer = RealTimeAnalyzer(timeframe)
        
        # Analizar señal
        result = analyzer.analyze_signal(symbol)
        
        # Mostrar reporte
        analyzer.print_signal_report(result)
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")


if __name__ == "__main__":
    main()