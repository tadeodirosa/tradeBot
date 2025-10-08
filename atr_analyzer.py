#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔍 ANALIZADOR DE ATR Y NIVELES DE TRADING
========================================

Analiza el ATR real de LINKUSDT y verifica si los niveles de TP/SL son realistas.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ATRAnalyzer:
    def __init__(self):
        self.binance_api = "https://api.binance.com/api/v3/klines"
        
    def analyze_atr_and_levels(self, symbol: str = "LINKUSDT", timeframe: str = "4h"):
        """Analizar ATR y niveles de trading."""
        print(f"🔍 ANÁLISIS DE ATR Y NIVELES - {symbol} {timeframe}")
        print("=" * 60)
        
        # 1. Obtener datos históricos
        data = self._get_historical_data(symbol, timeframe, 50)  # 50 velas para ATR
        
        if data is None or data.empty:
            print("❌ No se pudieron obtener datos")
            return
        
        # 2. Calcular ATR
        atr_values = self._calculate_atr(data)
        current_atr = atr_values.iloc[-1]
        avg_atr = atr_values.mean()
        current_price = data['Close'].iloc[-1]
        
        print(f"📊 ANÁLISIS DEL ATR:")
        print(f"   Precio actual: ${current_price:.4f}")
        print(f"   ATR actual: ${current_atr:.4f}")
        print(f"   ATR promedio: ${avg_atr:.4f}")
        print(f"   ATR %: {(current_atr/current_price)*100:.3f}%")
        
        # 3. Simular niveles como en el futures_simulator
        stop_loss_mult = 2.2
        take_profit_mult = 2.0
        
        # Para LONG
        simulated_entry = current_price
        simulated_stop_loss = simulated_entry - (current_atr * stop_loss_mult)
        simulated_take_profit = simulated_entry + (current_atr * take_profit_mult)
        
        stop_loss_pct = ((simulated_stop_loss - simulated_entry) / simulated_entry) * 100
        take_profit_pct = ((simulated_take_profit - simulated_entry) / simulated_entry) * 100
        
        print(f"\n🎯 NIVELES SIMULADOS (LONG):")
        print(f"   Entrada: ${simulated_entry:.4f}")
        print(f"   Stop Loss: ${simulated_stop_loss:.4f} ({stop_loss_pct:.3f}%)")
        print(f"   Take Profit: ${simulated_take_profit:.4f} ({take_profit_pct:.3f}%)")
        
        # 4. Calcular P&L con apalancamiento
        leverage = 25
        position_size = 100
        
        tp_pnl = (take_profit_pct / 100) * position_size * leverage
        sl_pnl = (stop_loss_pct / 100) * position_size * leverage
        
        print(f"\n💰 P&L SIMULADO (25x leverage, $100):")
        print(f"   Si alcanza TP: ${tp_pnl:.2f} ({take_profit_pct:.3f}% × 25)")
        print(f"   Si alcanza SL: ${sl_pnl:.2f} ({stop_loss_pct:.3f}% × 25)")
        
        # 5. Verificar el trade específico problemático
        trade_entry = 22.27
        trade_exit = 22.30
        trade_pnl = 3.35
        
        print(f"\n🔍 ANÁLISIS DEL TRADE PROBLEMÁTICO:")
        print(f"   Entrada reportada: ${trade_entry:.2f}")
        print(f"   Salida reportada: ${trade_exit:.2f}")
        print(f"   P&L reportado: ${trade_pnl:.2f}")
        
        # Calcular ATR implícito
        price_move = trade_exit - trade_entry
        implied_atr = price_move / take_profit_mult
        
        print(f"   Movimiento: ${price_move:.4f}")
        print(f"   ATR implícito: ${implied_atr:.4f}")
        print(f"   ATR real actual: ${current_atr:.4f}")
        
        # Verificar coherencia
        if abs(implied_atr - current_atr) / current_atr > 0.5:  # 50% diferencia
            print(f"   ⚠️ DISCREPANCIA: ATR implícito vs real difiere {abs(implied_atr - current_atr)/current_atr*100:.1f}%")
        else:
            print(f"   ✅ COHERENTE: ATR implícito vs real similar")
        
        # 6. Análisis de volatilidad histórica
        self._analyze_historical_volatility(data, symbol)
        
        # 7. Recomendaciones
        self._generate_recommendations(current_atr, current_price, avg_atr)
    
    def _get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Obtener datos históricos."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=limit if timeframe == '1d' else limit//6 if timeframe == '4h' else limit//24)
            
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000)
            }
            
            response = requests.get(self.binance_api, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"❌ Error API: {response.status_code}")
                return None
            
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            print(f"✅ Datos obtenidos: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcular ATR (Average True Range)."""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR como media móvil del True Range
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _analyze_historical_volatility(self, df: pd.DataFrame, symbol: str):
        """Analizar volatilidad histórica."""
        print(f"\n📈 ANÁLISIS DE VOLATILIDAD HISTÓRICA:")
        
        # Calcular returns diarios
        returns = df['Close'].pct_change().dropna()
        
        # Estadísticas de volatilidad
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(365)  # Anualizada
        
        print(f"   Volatilidad diaria: {daily_vol*100:.3f}%")
        print(f"   Volatilidad anualizada: {annualized_vol*100:.1f}%")
        
        # Rangos de precios típicos
        price_ranges = ((df['High'] - df['Low']) / df['Close']) * 100
        avg_daily_range = price_ranges.mean()
        
        print(f"   Rango diario promedio: {avg_daily_range:.2f}%")
        
        # Clasificar volatilidad
        if avg_daily_range < 2:
            vol_class = "BAJA"
        elif avg_daily_range < 4:
            vol_class = "NORMAL"
        elif avg_daily_range < 6:
            vol_class = "ALTA"
        else:
            vol_class = "MUY ALTA"
        
        print(f"   Clasificación: {vol_class}")
        
        # Verificar coherencia con trade
        if avg_daily_range < 1:
            print(f"   ⚠️ ALERTA: Volatilidad excepcionalmente baja para crypto")
        elif avg_daily_range > 8:
            print(f"   ⚠️ ALERTA: Volatilidad excepcionalmente alta")
        else:
            print(f"   ✅ Volatilidad en rango normal para crypto")
    
    def _generate_recommendations(self, current_atr: float, current_price: float, avg_atr: float):
        """Generar recomendaciones."""
        print(f"\n💡 RECOMENDACIONES:")
        
        # ATR como % del precio
        atr_pct = (current_atr / current_price) * 100
        
        if atr_pct < 0.5:
            print(f"   🔴 ATR muy bajo ({atr_pct:.3f}%) - Movimientos limitados")
            print(f"      • Considerar timeframes menores (1h)")
            print(f"      • Reducir multiplicadores de TP/SL")
            print(f"      • Evaluar otros assets con más volatilidad")
        elif atr_pct > 5:
            print(f"   🟡 ATR muy alto ({atr_pct:.3f}%) - Alta volatilidad")
            print(f"      • Aumentar multiplicadores de TP/SL")
            print(f"      • Reducir apalancamiento")
            print(f"      • Gestión de riesgo más estricta")
        else:
            print(f"   ✅ ATR normal ({atr_pct:.3f}%) - Condiciones adecuadas")
        
        # Niveles recomendados
        rec_tp_mult = max(1.5, min(3.0, 2.0 / atr_pct if atr_pct > 0 else 2.0))
        rec_sl_mult = max(1.2, min(2.5, 1.8 / atr_pct if atr_pct > 0 else 2.2))
        
        print(f"\n🎯 MULTIPLICADORES RECOMENDADOS:")
        print(f"   Take Profit: {rec_tp_mult:.1f}x ATR (actual: 2.0x)")
        print(f"   Stop Loss: {rec_sl_mult:.1f}x ATR (actual: 2.2x)")
        
        # P&L esperado
        expected_tp_pct = (current_atr * rec_tp_mult / current_price) * 100
        expected_pnl = expected_tp_pct * 25  # 25x leverage
        
        print(f"   P&L esperado en TP: {expected_pnl:.2f}% (${expected_pnl:.2f} en $100)")

def main():
    """Función principal."""
    analyzer = ATRAnalyzer()
    analyzer.analyze_atr_and_levels("LINKUSDT", "4h")

if __name__ == "__main__":
    main()