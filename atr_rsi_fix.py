#!/usr/bin/env python3
"""
Corrección de ATR y RSI para Máxima Precisión
============================================

Análisis y corrección de las diferencias encontradas en:
- ATR: 8.16% diferencia máxima
- RSI: 11.92 puntos diferencia máxima

Objetivo: Alinear con implementaciones estándar TA-Lib
"""

import ccxt
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class ATRRSICorrector:
    """Corrector de ATR y RSI para alinearse con estándares TA-Lib."""
    
    def __init__(self, symbol: str = 'LINK/USDT'):
        self.symbol = symbol
        
        self.exchange = ccxt.binance({
            'apiKey': '', 'secret': '', 'sandbox': False,
            'enableRateLimit': True, 'timeout': 30000,
        })
        
        print("🔧 CORRECTOR DE ATR Y RSI")
        print("=" * 50)
        print(f"📊 Símbolo: {self.symbol}")
        print(f"🎯 Objetivo: Alinear con TA-Lib estándar")
        print("=" * 50)
    
    def get_test_data(self, limit: int = 100) -> pd.DataFrame:
        """Obtener datos de prueba."""
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '4h', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Datos obtenidos: {len(df)} registros")
        return df
    
    def analyze_atr_differences(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Analizar diferencias en ATR y encontrar la implementación correcta."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        print(f"\n🔍 ANALIZANDO ATR (período {period})")
        print("-" * 40)
        
        results = {}
        
        # 1. Nuestra implementación SMA original
        tr_values = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        # ATR con SMA
        atr_sma = np.mean(tr_values[-period:])
        results['our_sma'] = atr_sma
        print(f"📊 Nuestra implementación SMA: ${atr_sma:.6f}")
        
        # 2. TA-Lib (referencia estándar)
        if TALIB_AVAILABLE:
            atr_talib = ta.ATR(high, low, close, timeperiod=period)
            atr_talib_current = atr_talib[-1]
            results['talib_standard'] = atr_talib_current
            print(f"📊 TA-Lib estándar: ${atr_talib_current:.6f}")
            
            # Diferencia
            diff_pct = abs(atr_sma - atr_talib_current) / atr_talib_current * 100
            print(f"❌ Diferencia: {diff_pct:.2f}%")
        
        # 3. Implementación Wilder (como TA-Lib)
        # TA-Lib usa smoothing de Wilder, no SMA simple
        tr_series = pd.Series(tr_values)
        
        # Primera ATR = SMA de primeros 14 valores
        first_atr = np.mean(tr_values[:period])
        
        # Luego usar fórmula de Wilder: ATR = ((ATR_prev * (n-1)) + TR_current) / n
        atr_wilder = first_atr
        for i in range(period, len(tr_values)):
            atr_wilder = (atr_wilder * (period - 1) + tr_values[i]) / period
        
        results['wilder_method'] = atr_wilder
        print(f"📊 Método Wilder (correcto): ${atr_wilder:.6f}")
        
        if TALIB_AVAILABLE:
            diff_wilder = abs(atr_wilder - atr_talib_current) / atr_talib_current * 100
            print(f"✅ Diferencia Wilder vs TA-Lib: {diff_wilder:.2f}%")
        
        # 4. Implementación EMA alternativa
        alpha = 1.0 / period  # Para Wilder smoothing
        atr_ema = tr_values[0]
        for tr in tr_values[1:]:
            atr_ema = alpha * tr + (1 - alpha) * atr_ema
        
        results['ema_alternative'] = atr_ema
        print(f"📊 EMA alternativa: ${atr_ema:.6f}")
        
        return results
    
    def get_corrected_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Obtener ATR corregido usando método Wilder estándar."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calcular True Range
        tr_values = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        if len(tr_values) < period:
            return np.mean(tr_values)
        
        # Método Wilder estándar
        first_atr = np.mean(tr_values[:period])
        
        atr_current = first_atr
        for i in range(period, len(tr_values)):
            atr_current = (atr_current * (period - 1) + tr_values[i]) / period
        
        return atr_current
    
    def analyze_rsi_differences(self, df: pd.DataFrame, period: int = 14) -> Dict:
        """Analizar diferencias en RSI y encontrar la implementación correcta."""
        close = df['close'].values
        
        print(f"\n🔍 ANALIZANDO RSI (período {period})")
        print("-" * 40)
        
        results = {}
        
        # 1. Nuestra implementación SMA original
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # SMA simple
        avg_gain_sma = np.mean(gains[-period:])
        avg_loss_sma = np.mean(losses[-period:])
        rs_sma = avg_gain_sma / avg_loss_sma if avg_loss_sma != 0 else float('inf')
        rsi_sma = 100 - (100 / (1 + rs_sma))
        
        results['our_sma'] = rsi_sma
        print(f"📊 Nuestra implementación SMA: {rsi_sma:.2f}")
        
        # 2. TA-Lib (referencia estándar)
        if TALIB_AVAILABLE:
            rsi_talib = ta.RSI(close, timeperiod=period)
            rsi_talib_current = rsi_talib[-1]
            results['talib_standard'] = rsi_talib_current
            print(f"📊 TA-Lib estándar: {rsi_talib_current:.2f}")
            
            diff = abs(rsi_sma - rsi_talib_current)
            print(f"❌ Diferencia: {diff:.2f} puntos")
        
        # 3. Implementación Wilder (como TA-Lib)
        # TA-Lib usa smoothing de Wilder para RSI también
        
        # Primera media = SMA de primeros 14 valores
        avg_gain = np.mean(gains[:period]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[:period]) if len(losses) >= period else np.mean(losses)
        
        # Luego usar smoothing de Wilder
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        rs_wilder = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi_wilder = 100 - (100 / (1 + rs_wilder))
        
        results['wilder_method'] = rsi_wilder
        print(f"📊 Método Wilder (correcto): {rsi_wilder:.2f}")
        
        if TALIB_AVAILABLE:
            diff_wilder = abs(rsi_wilder - rsi_talib_current)
            print(f"✅ Diferencia Wilder vs TA-Lib: {diff_wilder:.2f} puntos")
        
        return results
    
    def get_corrected_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Obtener RSI corregido usando método Wilder estándar."""
        close = df['close'].values
        
        if len(close) < period + 1:
            return 50.0
        
        # Calcular cambios
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) < period:
            return 50.0
        
        # Método Wilder estándar
        # Primera media = SMA de primeros valores
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Smoothing de Wilder para el resto
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Calcular RSI
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def test_corrections(self):
        """Probar las correcciones de ATR y RSI."""
        print(f"\n🧪 PROBANDO CORRECCIONES")
        print("=" * 50)
        
        df = self.get_test_data()
        
        # Probar ATR
        print(f"\n📊 COMPARACIÓN ATR:")
        atr_results = self.analyze_atr_differences(df)
        atr_corrected = self.get_corrected_atr(df)
        
        print(f"\n🔧 ATR Corregido final: ${atr_corrected:.6f}")
        
        # Probar RSI
        print(f"\n📊 COMPARACIÓN RSI:")
        rsi_results = self.analyze_rsi_differences(df)
        rsi_corrected = self.get_corrected_rsi(df)
        
        print(f"\n🔧 RSI Corregido final: {rsi_corrected:.2f}")
        
        # Validar que las correcciones funcionan
        if TALIB_AVAILABLE:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            atr_talib = ta.ATR(high, low, close, timeperiod=14)[-1]
            rsi_talib = ta.RSI(close, timeperiod=14)[-1]
            
            atr_diff = abs(atr_corrected - atr_talib) / atr_talib * 100
            rsi_diff = abs(rsi_corrected - rsi_talib)
            
            print(f"\n✅ VALIDACIÓN FINAL:")
            print(f"   ATR diferencia: {atr_diff:.3f}% (objetivo: <1%)")
            print(f"   RSI diferencia: {rsi_diff:.2f} puntos (objetivo: <1)")
            
            if atr_diff < 1.0 and rsi_diff < 1.0:
                print(f"🎯 ÉXITO: Correcciones alineadas con TA-Lib")
                return True
            else:
                print(f"⚠️ ATENCIÓN: Revisar implementaciones")
                return False
        
        return True

def main():
    """Función principal."""
    try:
        corrector = ATRRSICorrector('LINK/USDT')
        success = corrector.test_corrections()
        
        if success:
            print(f"\n✅ Correcciones de ATR y RSI completadas exitosamente")
        else:
            print(f"\n❌ Correcciones necesitan revisión")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()