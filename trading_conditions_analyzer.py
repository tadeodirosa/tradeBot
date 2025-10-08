#!/usr/bin/env python3
"""
Análisis de Condiciones de Trading LINK
======================================

Analizamos las condiciones exitosas del simulador anterior para LINK
y las adaptamos al nuevo sistema verificado manteniendo la fiabilidad.
"""

import ccxt
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class TradingConditionsAnalyzer:
    """Analizador de condiciones de trading para recuperar generación de señales."""
    
    def __init__(self, symbol: str = 'LINK/USDT'):
        self.symbol = symbol
        
        self.exchange = ccxt.binance({
            'apiKey': '', 'secret': '', 'sandbox': False,
            'enableRateLimit': True, 'timeout': 30000,
        })
        
        print("🔍 ANÁLISIS DE CONDICIONES DE TRADING")
        print("=" * 50)
        print(f"📊 Símbolo: {self.symbol}")
        print(f"🎯 Objetivo: Recuperar generación de trades")
        print("=" * 50)
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Obtener datos históricos para análisis."""
        limit = days * 6  # Aproximadamente 6 barras de 4h por día
        
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '4h', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Datos históricos: {len(df)} registros")
        print(f"📅 Rango: {df.index[0]} a {df.index[-1]}")
        
        return df
    
    def calculate_corrected_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcular indicadores corregidos con método Wilder."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        # ATR corregido (método Wilder)
        tr_values = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        if len(tr_values) >= 14:
            # ATR Wilder
            first_atr = np.mean(tr_values[:14])
            atr_current = first_atr
            for i in range(14, len(tr_values)):
                atr_current = (atr_current * 13 + tr_values[i]) / 14
            
            indicators['atr'] = atr_current
            indicators['atr_percentage'] = (atr_current / close[-1]) * 100
        
        # RSI corregido (método Wilder)
        if len(close) >= 15:
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Primera media
            avg_gain = np.mean(gains[:14])
            avg_loss = np.mean(losses[:14])
            
            # Smoothing Wilder
            for i in range(14, len(gains)):
                avg_gain = (avg_gain * 13 + gains[i]) / 14
                avg_loss = (avg_loss * 13 + losses[i]) / 14
            
            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # EMAs
        close_series = df['close']
        indicators['ema_9'] = close_series.ewm(span=9).mean().iloc[-1]
        indicators['ema_21'] = close_series.ewm(span=21).mean().iloc[-1]
        indicators['ema_50'] = close_series.ewm(span=50).mean().iloc[-1]
        
        # SMAs
        indicators['sma_20'] = close_series.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close_series.rolling(50).mean().iloc[-1]
        
        # Precio actual
        indicators['current_price'] = close[-1]
        
        return indicators
    
    def analyze_trading_conditions(self, df: pd.DataFrame) -> Dict:
        """Analizar condiciones de trading históricas para encontrar patrones exitosos."""
        
        print(f"\n📊 ANALIZANDO CONDICIONES HISTÓRICAS")
        print("-" * 40)
        
        # Calcular indicadores para todo el período
        signals_generated = []
        
        for i in range(50, len(df)):  # Empezar después de 50 barras para indicadores
            # Datos hasta la barra actual
            current_data = df.iloc[:i+1]
            indicators = self.calculate_corrected_indicators(current_data)
            
            if not indicators:
                continue
            
            current_price = indicators['current_price']
            rsi = indicators.get('rsi', 50)
            atr_pct = indicators.get('atr_percentage', 2)
            ema_9 = indicators.get('ema_9', current_price)
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            
            # Condiciones RELAJADAS para generar más señales
            signal_type = None
            signal_strength = 0
            reasons = []
            
            # LONG conditions (más permisivas)
            long_score = 0
            if current_price > ema_21:
                long_score += 1
                reasons.append("Precio > EMA21")
            
            if ema_9 > ema_21:
                long_score += 1
                reasons.append("EMA9 > EMA21")
            
            if rsi < 50:  # RSI bajo la media
                long_score += 1
                reasons.append(f"RSI oversold tendency ({rsi:.1f})")
            
            if atr_pct > 1.0:  # Volatilidad mínima
                long_score += 1
                reasons.append("Volatilidad adecuada")
            
            # SHORT conditions (más permisivas)
            short_score = 0
            if current_price < ema_21:
                short_score += 1
                reasons.append("Precio < EMA21")
            
            if ema_9 < ema_21:
                short_score += 1
                reasons.append("EMA9 < EMA21")
            
            if rsi > 50:  # RSI arriba de la media
                short_score += 1
                reasons.append(f"RSI overbought tendency ({rsi:.1f})")
            
            # Generar señal con menor umbral (2 condiciones en lugar de 3)
            if long_score >= 2:
                signal_type = 'LONG'
                signal_strength = long_score * 25
            elif short_score >= 2:
                signal_type = 'SHORT'
                signal_strength = short_score * 25
            
            if signal_type:
                signals_generated.append({
                    'timestamp': current_data.index[-1],
                    'bar_index': i,
                    'signal_type': signal_type,
                    'strength': signal_strength,
                    'price': current_price,
                    'rsi': rsi,
                    'atr_pct': atr_pct,
                    'ema_9': ema_9,
                    'ema_21': ema_21,
                    'reasons': reasons.copy()
                })
                
                reasons.clear()  # Limpiar para próxima iteración
        
        print(f"✅ Señales generadas con condiciones relajadas: {len(signals_generated)}")
        
        if signals_generated:
            # Estadísticas
            long_signals = [s for s in signals_generated if s['signal_type'] == 'LONG']
            short_signals = [s for s in signals_generated if s['signal_type'] == 'SHORT']
            
            print(f"   📈 Señales LONG: {len(long_signals)}")
            print(f"   📉 Señales SHORT: {len(short_signals)}")
            
            # Mostrar últimas 5 señales
            print(f"\n🎯 ÚLTIMAS 5 SEÑALES:")
            for signal in signals_generated[-5:]:
                timestamp = signal['timestamp'].strftime('%Y-%m-%d %H:%M')
                print(f"   {signal['signal_type']} | {timestamp} | ${signal['price']:.4f} | RSI: {signal['rsi']:.1f}")
        
        return {
            'total_signals': len(signals_generated),
            'long_signals': len([s for s in signals_generated if s['signal_type'] == 'LONG']),
            'short_signals': len([s for s in signals_generated if s['signal_type'] == 'SHORT']),
            'signals': signals_generated[-10:]  # Últimas 10 señales
        }
    
    def test_current_conditions(self, df: pd.DataFrame) -> Dict:
        """Probar condiciones actuales para generar señal."""
        
        print(f"\n🎯 PROBANDO CONDICIONES ACTUALES")
        print("-" * 40)
        
        indicators = self.calculate_corrected_indicators(df)
        
        if not indicators:
            return {'signal': None, 'reason': 'Indicadores insuficientes'}
        
        current_price = indicators['current_price']
        rsi = indicators.get('rsi', 50)
        atr_pct = indicators.get('atr_percentage', 2)
        ema_9 = indicators.get('ema_9', current_price)
        ema_21 = indicators.get('ema_21', current_price)
        ema_50 = indicators.get('ema_50', current_price)
        
        print(f"💰 Precio actual: ${current_price:.4f}")
        print(f"📊 RSI: {rsi:.2f}")
        print(f"📈 ATR: {atr_pct:.2f}%")
        print(f"📊 EMA9: ${ema_9:.4f}")
        print(f"📊 EMA21: ${ema_21:.4f}")
        print(f"📊 EMA50: ${ema_50:.4f}")
        
        # Condiciones RELAJADAS
        signal_type = None
        reasons = []
        confidence = 0
        
        # LONG conditions
        long_score = 0
        if current_price > ema_21:
            long_score += 1
            reasons.append("✅ Precio > EMA21")
        else:
            reasons.append("❌ Precio < EMA21")
        
        if ema_9 > ema_21:
            long_score += 1
            reasons.append("✅ EMA9 > EMA21 (tendencia alcista)")
        else:
            reasons.append("❌ EMA9 < EMA21")
        
        if rsi < 50:
            long_score += 1
            reasons.append(f"✅ RSI undersold tendency ({rsi:.1f})")
        else:
            reasons.append(f"❌ RSI oversold tendency ({rsi:.1f})")
        
        if atr_pct > 1.0:
            long_score += 1
            reasons.append(f"✅ Volatilidad adecuada ({atr_pct:.2f}%)")
        else:
            reasons.append(f"❌ Volatilidad baja ({atr_pct:.2f}%)")
        
        print(f"\n📊 EVALUACIÓN LONG:")
        for reason in reasons:
            print(f"   {reason}")
        
        print(f"\n📊 Score LONG: {long_score}/4")
        
        if long_score >= 2:  # Umbral relajado
            signal_type = 'LONG'
            confidence = long_score * 25
            
            # Calcular niveles
            atr_value = indicators.get('atr', current_price * 0.02)
            stop_loss = current_price - (atr_value * 1.5)
            take_profit = current_price + (atr_value * 2.0)
            
            return {
                'signal': 'LONG',
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'indicators': indicators,
                'reasons': reasons
            }
        
        # Si no hay LONG, evaluar SHORT
        reasons.clear()
        short_score = 0
        
        if current_price < ema_21:
            short_score += 1
            reasons.append("✅ Precio < EMA21")
        else:
            reasons.append("❌ Precio > EMA21")
        
        if ema_9 < ema_21:
            short_score += 1
            reasons.append("✅ EMA9 < EMA21 (tendencia bajista)")
        else:
            reasons.append("❌ EMA9 > EMA21")
        
        if rsi > 50:
            short_score += 1
            reasons.append(f"✅ RSI overbought tendency ({rsi:.1f})")
        else:
            reasons.append(f"❌ RSI undersold tendency ({rsi:.1f})")
        
        print(f"\n📊 EVALUACIÓN SHORT:")
        for reason in reasons:
            print(f"   {reason}")
        
        print(f"\n📊 Score SHORT: {short_score}/3")
        
        if short_score >= 2:
            signal_type = 'SHORT'
            confidence = short_score * 25
            
            # Calcular niveles
            atr_value = indicators.get('atr', current_price * 0.02)
            stop_loss = current_price + (atr_value * 1.5)
            take_profit = current_price - (atr_value * 2.0)
            
            return {
                'signal': 'SHORT',
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'indicators': indicators,
                'reasons': reasons
            }
        
        return {
            'signal': 'HOLD',
            'confidence': 0,
            'reason': 'No se cumplen condiciones mínimas',
            'indicators': indicators,
            'long_score': long_score,
            'short_score': short_score
        }
    
    def run_analysis(self):
        """Ejecutar análisis completo."""
        # Obtener datos
        df = self.get_historical_data(30)
        
        # Analizar condiciones históricas
        historical_analysis = self.analyze_trading_conditions(df)
        
        # Probar condiciones actuales
        current_analysis = self.test_current_conditions(df)
        
        print(f"\n🎯 RESUMEN DEL ANÁLISIS")
        print("=" * 50)
        print(f"📊 Señales históricas (30 días): {historical_analysis['total_signals']}")
        print(f"📈 LONG: {historical_analysis['long_signals']}")
        print(f"📉 SHORT: {historical_analysis['short_signals']}")
        
        if current_analysis['signal'] != 'HOLD':
            print(f"\n🚀 SEÑAL ACTUAL: {current_analysis['signal']}")
            print(f"📊 Confianza: {current_analysis['confidence']}%")
            print(f"💰 Entrada: ${current_analysis['entry_price']:.4f}")
            print(f"🛡️ Stop Loss: ${current_analysis['stop_loss']:.4f}")
            print(f"🎯 Take Profit: ${current_analysis['take_profit']:.4f}")
        else:
            print(f"\n⏸️ Sin señal actual - Condiciones insuficientes")
            print(f"📊 Long score: {current_analysis['long_score']}/4")
            print(f"📊 Short score: {current_analysis['short_score']}/3")
        
        return {
            'historical': historical_analysis,
            'current': current_analysis
        }

def main():
    """Función principal."""
    try:
        analyzer = TradingConditionsAnalyzer('LINK/USDT')
        results = analyzer.run_analysis()
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_conditions_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Análisis guardado en: {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()