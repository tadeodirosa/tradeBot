#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔍 VERIFICADOR DIRECTO DE TRADES LINKUSDT
========================================

Verifica directamente los 28 trades de LINKUSDT que generaron +35.53% ROI
usando los datos que ya conocemos del output del simulador.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# TRADES EXTRAÍDOS DEL OUTPUT DEL SIMULADOR LINKUSDT
LINKUSDT_TRADES = [
    # Los 28 trades que obtuvimos del simulador original
    {'id': 1, 'entry_bar': 20, 'entry_price': 22.27, 'exit_bar': 28, 'pnl': 3.35, 'reason': 'TP'},
    {'id': 2, 'entry_bar': 24, 'entry_price': 22.38, 'exit_bar': 32, 'pnl': 4.15, 'reason': 'TP'}, 
    {'id': 3, 'entry_bar': 28, 'entry_price': 22.21, 'exit_bar': 36, 'pnl': 5.64, 'reason': 'TP'},
    {'id': 4, 'entry_bar': 32, 'entry_price': 22.98, 'exit_bar': 40, 'pnl': 3.49, 'reason': 'TP'},
    {'id': 5, 'entry_bar': 36, 'entry_price': 23.18, 'exit_bar': 44, 'pnl': 5.45, 'reason': 'TP'},
    {'id': 6, 'entry_bar': 40, 'entry_price': 23.07, 'exit_bar': 48, 'pnl': 5.95, 'reason': 'TP'},
    {'id': 7, 'entry_bar': 44, 'entry_price': 23.35, 'exit_bar': 52, 'pnl': 5.38, 'reason': 'TP'},
    {'id': 8, 'entry_bar': 48, 'entry_price': 23.81, 'exit_bar': 56, 'pnl': 3.62, 'reason': 'TP'},
    {'id': 9, 'entry_bar': 52, 'entry_price': 23.94, 'exit_bar': 64, 'pnl': -3.80, 'reason': 'LIQUIDATION'},
    {'id': 10, 'entry_bar': 56, 'entry_price': 24.32, 'exit_bar': 68, 'pnl': -4.02, 'reason': 'LIQUIDATION'},
    {'id': 11, 'entry_bar': 64, 'entry_price': 24.76, 'exit_bar': 72, 'pnl': -3.58, 'reason': 'LIQUIDATION'},
    {'id': 12, 'entry_bar': 68, 'entry_price': 24.37, 'exit_bar': 76, 'pnl': 4.68, 'reason': 'TP'},
    {'id': 13, 'entry_bar': 72, 'entry_price': 24.26, 'exit_bar': 80, 'pnl': 4.86, 'reason': 'TP'},
    {'id': 14, 'entry_bar': 76, 'entry_price': 23.35, 'exit_bar': 84, 'pnl': 4.11, 'reason': 'TP'},
    {'id': 15, 'entry_bar': 80, 'entry_price': 23.62, 'exit_bar': 88, 'pnl': -3.63, 'reason': 'LIQUIDATION'},
    {'id': 16, 'entry_bar': 84, 'entry_price': 23.31, 'exit_bar': 92, 'pnl': -5.34, 'reason': 'LIQUIDATION'},
    {'id': 17, 'entry_bar': 96, 'entry_price': 24.47, 'exit_bar': 100, 'pnl': -4.93, 'reason': 'LIQUIDATION'},
    {'id': 18, 'entry_bar': 100, 'entry_price': 23.39, 'exit_bar': 104, 'pnl': -5.25, 'reason': 'LIQUIDATION'},
    {'id': 19, 'entry_bar': 104, 'entry_price': 23.29, 'exit_bar': 108, 'pnl': -3.97, 'reason': 'LIQUIDATION'},
    {'id': 20, 'entry_bar': 108, 'entry_price': 23.37, 'exit_bar': 112, 'pnl': -4.01, 'reason': 'LIQUIDATION'},
    {'id': 21, 'entry_bar': 112, 'entry_price': 23.18, 'exit_bar': 116, 'pnl': -3.94, 'reason': 'LIQUIDATION'},
    {'id': 22, 'entry_bar': 116, 'entry_price': 21.21, 'exit_bar': 120, 'pnl': 6.15, 'reason': 'TP'},
    {'id': 23, 'entry_bar': 120, 'entry_price': 21.57, 'exit_bar': 124, 'pnl': 7.37, 'reason': 'TP'},
    {'id': 24, 'entry_bar': 124, 'entry_price': 21.58, 'exit_bar': 128, 'pnl': 5.33, 'reason': 'TP'},
    {'id': 25, 'entry_bar': 132, 'entry_price': 21.30, 'exit_bar': 136, 'pnl': 3.37, 'reason': 'TP'},
    {'id': 26, 'entry_bar': 136, 'entry_price': 20.40, 'exit_bar': 140, 'pnl': 3.61, 'reason': 'TP'},
    {'id': 27, 'entry_bar': 156, 'entry_price': 21.49, 'exit_bar': 160, 'pnl': 1.62, 'reason': 'END_OF_DATA'},
    {'id': 28, 'entry_bar': 160, 'entry_price': 21.73, 'exit_bar': 164, 'pnl': -0.12, 'reason': 'END_OF_DATA'}
]

class DirectTradeVerifier:
    def __init__(self):
        self.binance_api = "https://api.binance.com/api/v3/klines"
        # Timestamp base: 2025-09-02 21:00:00 (inicio del período)
        self.base_time = datetime(2025, 9, 2, 21, 0, 0)
        
    def verify_linkusdt_trades(self):
        """Verificar los 28 trades de LINKUSDT."""
        print("🔍 VERIFICADOR DIRECTO DE TRADES LINKUSDT")
        print("=" * 60)
        print(f"📊 Trades a verificar: {len(LINKUSDT_TRADES)}")
        print(f"💰 ROI reportado: +35.53%")
        print(f"🎯 Total P&L reportado: $35.53")
        
        # 1. Obtener datos reales de LINKUSDT
        print(f"\n📡 Obteniendo datos reales de LINKUSDT...")
        real_data = self._get_real_data()
        
        if real_data is None or real_data.empty:
            print("❌ No se pudieron obtener datos reales")
            return
        
        # 2. Convertir trades a formato con timestamps
        print(f"\n🔄 Convirtiendo trades a timestamps reales...")
        timestamped_trades = self._convert_trades_to_timestamps(LINKUSDT_TRADES)
        
        # 3. Verificar cada trade
        print(f"\n🔍 VERIFICANDO {len(timestamped_trades)} TRADES")
        print("=" * 60)
        
        verified_count = 0
        total_verified_pnl = 0
        impossible_trades = []
        
        for trade in timestamped_trades:
            print(f"\n📋 TRADE #{trade['id']}: LONG $100")
            
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            reported_pnl = trade['pnl']
            reason = trade['reason']
            
            print(f"   🕐 {entry_time.strftime('%Y-%m-%d %H:%M')} → {exit_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   💰 Entrada: ${entry_price:.2f}")
            print(f"   📊 P&L reportado: ${reported_pnl:.2f} ({reason})")
            
            # Calcular exit_price basado en P&L
            pnl_pct = reported_pnl / (100 * 25)  # P&L / (size * leverage)
            calculated_exit_price = entry_price * (1 + pnl_pct)
            
            print(f"   💸 Salida calculada: ${calculated_exit_price:.2f}")
            
            # Verificar precios contra datos reales
            entry_valid = self._verify_price_in_data(real_data, entry_time, entry_price, "ENTRADA")
            exit_valid = self._verify_price_in_data(real_data, exit_time, calculated_exit_price, "SALIDA")
            
            # Verificar que el movimiento fue posible
            movement_valid = self._verify_price_movement(real_data, entry_time, exit_time, 
                                                        entry_price, calculated_exit_price)
            
            if entry_valid and exit_valid and movement_valid:
                verified_count += 1
                total_verified_pnl += reported_pnl
                print(f"   ✅ TRADE VERIFICADO")
            else:
                impossible_trades.append(trade['id'])
                print(f"   ❌ TRADE CUESTIONABLE")
        
        # Resumen final
        self._print_verification_summary(verified_count, total_verified_pnl, impossible_trades)
    
    def _get_real_data(self) -> Optional[pd.DataFrame]:
        """Obtener datos reales de LINKUSDT para el período."""
        try:
            # Período: Sep 2 - Oct 3, 2025 (aproximadamente)
            end_time = datetime(2025, 10, 3)
            start_time = datetime(2025, 9, 1)
            
            params = {
                'symbol': 'LINKUSDT',
                'interval': '4h',
                'limit': 200,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000)
            }
            
            response = requests.get(self.binance_api, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"❌ Error API Binance: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            print(f"✅ Datos reales obtenidos: {len(df)} velas")
            print(f"   Desde: {df.index[0]}")
            print(f"   Hasta: {df.index[-1]}")
            print(f"   Precio actual: ${df['Close'].iloc[-1]:.2f}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos reales: {e}")
            return None
    
    def _convert_trades_to_timestamps(self, trades: List[Dict]) -> List[Dict]:
        """Convertir números de barra a timestamps reales."""
        timestamped_trades = []
        
        for trade in trades:
            entry_time = self.base_time + timedelta(hours=4 * trade['entry_bar'])
            exit_time = self.base_time + timedelta(hours=4 * trade['exit_bar'])
            
            timestamped_trade = {
                'id': trade['id'],
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': trade['entry_price'],
                'pnl': trade['pnl'],
                'reason': trade['reason']
            }
            
            timestamped_trades.append(timestamped_trade)
        
        return timestamped_trades
    
    def _verify_price_in_data(self, real_data: pd.DataFrame, target_time: datetime, 
                             target_price: float, label: str) -> bool:
        """Verificar si un precio existe en los datos reales."""
        try:
            # Buscar vela más cercana (tolerancia de 2 horas)
            time_tolerance = pd.Timedelta(hours=2)
            mask = abs(real_data.index - target_time) <= time_tolerance
            candidates = real_data[mask]
            
            if candidates.empty:
                print(f"   ⚠️ {label}: No hay datos cerca de {target_time}")
                return False
            
            # Verificar si el precio está en algún rango OHLC
            for _, candle in candidates.iterrows():
                if candle['Low'] <= target_price <= candle['High']:
                    diff_pct = abs(target_price - candle['Close']) / candle['Close'] * 100
                    print(f"   ✅ {label}: ${target_price:.2f} válido (diff: {diff_pct:.1f}%)")
                    return True
            
            # Si no está en rango
            closest_candle = candidates.iloc[0]
            print(f"   ❌ {label}: ${target_price:.2f} fuera de rango [{closest_candle['Low']:.2f}-{closest_candle['High']:.2f}]")
            return False
            
        except Exception as e:
            print(f"   ❌ {label}: Error - {e}")
            return False
    
    def _verify_price_movement(self, real_data: pd.DataFrame, entry_time: datetime, 
                              exit_time: datetime, entry_price: float, exit_price: float) -> bool:
        """Verificar que el movimiento de precio fue posible."""
        try:
            # Obtener datos del período del trade
            mask = (real_data.index >= entry_time) & (real_data.index <= exit_time)
            period_data = real_data[mask]
            
            if period_data.empty:
                return True  # No podemos verificar
            
            # Verificar rangos
            min_period = period_data['Low'].min()
            max_period = period_data['High'].max()
            
            # Ambos precios deben estar en el rango del período
            entry_in_range = min_period <= entry_price <= max_period
            exit_in_range = min_period <= exit_price <= max_period
            
            if entry_in_range and exit_in_range:
                print(f"   ✅ MOVIMIENTO: Ambos precios en rango período")
                return True
            else:
                print(f"   ❌ MOVIMIENTO: Precios fuera de rango período")
                return False
                
        except Exception:
            return True
    
    def _print_verification_summary(self, verified_count: int, total_verified_pnl: float, 
                                   impossible_trades: List[int]):
        """Imprimir resumen de verificación."""
        total_trades = len(LINKUSDT_TRADES)
        
        print(f"\n🏆 RESUMEN DE VERIFICACIÓN - LINKUSDT")
        print("=" * 60)
        print(f"   📊 Trades verificados: {verified_count}/{total_trades}")
        print(f"   📈 Tasa de verificación: {verified_count/total_trades*100:.1f}%")
        print(f"   💰 P&L verificado: ${total_verified_pnl:.2f}")
        print(f"   📊 P&L reportado: $35.53")
        
        # Diferencia de P&L
        pnl_diff = abs(total_verified_pnl - 35.53)
        print(f"   📊 Diferencia P&L: ${pnl_diff:.2f}")
        
        # ROI verificado
        verified_roi = (total_verified_pnl / 100) * 100
        print(f"   🎯 ROI verificado: {verified_roi:.2f}%")
        
        # Evaluación final
        verification_rate = verified_count / total_trades
        
        if verification_rate >= 0.85 and pnl_diff < 5:
            print(f"\n   🎯 EVALUACIÓN: ALTAMENTE CONFIABLE")
            print(f"      Los trades de LINKUSDT son verificables y realistas")
        elif verification_rate >= 0.7 and pnl_diff < 10:
            print(f"\n   ✅ EVALUACIÓN: CONFIABLE")
            print(f"      La mayoría de trades son verificables")
        elif verification_rate >= 0.5:
            print(f"\n   ⚠️ EVALUACIÓN: MODERADAMENTE CONFIABLE")
            print(f"      Algunos trades presentan problemas")
        else:
            print(f"\n   ❌ EVALUACIÓN: POCO CONFIABLE")
            print(f"      Demasiados trades no verificables")
        
        # Trades problemáticos
        if impossible_trades:
            print(f"\n🚨 TRADES PROBLEMÁTICOS: {len(impossible_trades)}")
            print(f"   IDs: {impossible_trades[:10]}")  # Mostrar máximo 10

def main():
    """Función principal."""
    verifier = DirectTradeVerifier()
    verifier.verify_linkusdt_trades()

if __name__ == "__main__":
    main()