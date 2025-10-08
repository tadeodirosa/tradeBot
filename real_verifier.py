#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VERIFICADOR DE AUTENTICIDAD - BACKTEST REAL
===========================================

Este script descarga datos históricos reales de Binance,
ejecuta el backtest y verifica inmediatamente los resultados.
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import argparse

class RealBacktestVerifier:
    def __init__(self):
        self.binance_api = "https://api.binance.com/api/v3/klines"
        
    def verify_with_real_data(self, symbol: str, timeframe: str = "4h", days_back: int = 30):
        """Verificar backtest usando datos históricos reales."""
        print(f"🔍 VERIFICADOR DE AUTENTICIDAD - {symbol} {timeframe}")
        print("=" * 60)
        
        # 1. Descargar datos históricos reales
        print(f"📡 Descargando datos históricos de {symbol}...")
        real_data = self._download_real_historical_data(symbol, timeframe, days_back)
        
        if real_data is None or real_data.empty:
            print("❌ No se pudieron obtener datos históricos")
            return
        
        # 2. Ejecutar backtest simple con esos datos
        print(f"🎯 Ejecutando backtest con datos reales...")
        trades = self._run_simple_backtest(real_data, symbol)
        
        if not trades:
            print("❌ No se generaron trades")
            return
        
        # 3. Verificar cada trade
        print(f"🔍 Verificando lógica de {len(trades)} trades...")
        self._verify_trade_logic(trades, real_data, symbol)
        
    def _download_real_historical_data(self, symbol: str, timeframe: str, days_back: int) -> Optional[pd.DataFrame]:
        """Descargar datos históricos reales de Binance."""
        try:
            # Calcular fechas
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Convertir timeframe
            tf_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
            interval = tf_map.get(timeframe, '4h')
            
            # Calcular límite necesario
            if timeframe == '1h':
                limit = min(1000, days_back * 24)
            elif timeframe == '4h':
                limit = min(1000, days_back * 6)
            else:  # 1d
                limit = min(1000, days_back)
            
            print(f"📅 Período: {start_time.strftime('%Y-%m-%d')} a {end_time.strftime('%Y-%m-%d')}")
            print(f"🔢 Solicitando {limit} velas de {interval}")
            
            # API call
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000)
            }
            
            response = requests.get(self.binance_api, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"❌ Error API Binance: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data:
                print("❌ No se recibieron datos")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Procesar datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convertir precios a float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            print(f"✅ Datos descargados: {len(df)} velas")
            print(f"   Desde: {df.index[0]}")
            print(f"   Hasta: {df.index[-1]}")
            print(f"   Precio actual: ${df['Close'].iloc[-1]:.4f}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ Error descargando datos: {e}")
            return None
    
    def _run_simple_backtest(self, data: pd.DataFrame, symbol: str) -> List[Dict]:
        """Ejecutar backtest simple con datos reales."""
        try:
            print(f"🤖 Ejecutando estrategia simple...")
            
            trades = []
            balance = 100  # USD
            leverage = 25
            position_size = 100
            
            # Estrategia simple: RSI + tendencia
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['SMA_10'] = data['Close'].rolling(10).mean()
            data['SMA_30'] = data['Close'].rolling(30).mean()
            
            in_position = False
            entry_price = 0
            entry_time = None
            position_type = None
            
            for i in range(50, len(data) - 1):  # Empezar después de calcular indicadores
                current = data.iloc[i]
                current_time = data.index[i]
                current_price = current['Close']
                
                # Señales de entrada
                rsi_oversold = current['RSI'] < 30
                rsi_overbought = current['RSI'] > 70
                trend_up = current['SMA_10'] > current['SMA_30']
                trend_down = current['SMA_10'] < current['SMA_30']
                
                # Entrar en posición
                if not in_position:
                    if rsi_oversold and trend_up:  # LONG
                        in_position = True
                        entry_price = current_price
                        entry_time = current_time
                        position_type = 'LONG'
                        
                    elif rsi_overbought and trend_down:  # SHORT
                        in_position = True
                        entry_price = current_price
                        entry_time = current_time
                        position_type = 'SHORT'
                
                # Salir de posición
                elif in_position:
                    exit_trade = False
                    exit_reason = 'MANUAL'
                    exit_price = current_price
                    
                    # Condiciones de salida
                    if position_type == 'LONG':
                        profit_pct = (current_price - entry_price) / entry_price
                        
                        if profit_pct >= 0.025:  # +2.5% TP
                            exit_trade = True
                            exit_reason = 'TP'
                            exit_price = entry_price * 1.025
                        elif profit_pct <= -0.035:  # -3.5% SL (liquidación)
                            exit_trade = True
                            exit_reason = 'LIQUIDATION'
                            exit_price = entry_price * 0.965
                        elif current['RSI'] > 70:  # RSI sobrecomprado
                            exit_trade = True
                            exit_reason = 'MANUAL'
                    
                    else:  # SHORT
                        profit_pct = (entry_price - current_price) / entry_price
                        
                        if profit_pct >= 0.025:  # +2.5% TP
                            exit_trade = True
                            exit_reason = 'TP'
                            exit_price = entry_price * 0.975
                        elif profit_pct <= -0.035:  # -3.5% SL
                            exit_trade = True
                            exit_reason = 'LIQUIDATION'
                            exit_price = entry_price * 1.035
                        elif current['RSI'] < 30:  # RSI sobrevendido
                            exit_trade = True
                            exit_reason = 'MANUAL'
                    
                    # Crear trade
                    if exit_trade:
                        if position_type == 'LONG':
                            pnl_pct = (exit_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - exit_price) / entry_price
                        
                        pnl = pnl_pct * position_size * leverage
                        
                        trade = {
                            'id': f"REAL_{len(trades)+1}",
                            'symbol': symbol,
                            'side': position_type,
                            'entry_time': entry_time.isoformat(),
                            'exit_time': current_time.isoformat(),
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'size_usd': position_size,
                            'leverage': leverage,
                            'pnl': pnl,
                            'reason': exit_reason,
                            'duration_hours': (current_time - entry_time).total_seconds() / 3600
                        }
                        
                        trades.append(trade)
                        balance += pnl
                        
                        # Reset
                        in_position = False
                        entry_price = 0
                        entry_time = None
                        position_type = None
                        
                        # Límite de trades
                        if len(trades) >= 10:
                            break
            
            print(f"✅ Backtest completado:")
            print(f"   Trades generados: {len(trades)}")
            print(f"   Balance final: ${balance:.2f}")
            print(f"   ROI: {((balance/100)-1)*100:.1f}%")
            
            return trades
            
        except Exception as e:
            print(f"❌ Error en backtest: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _verify_trade_logic(self, trades: List[Dict], data: pd.DataFrame, symbol: str):
        """Verificar la lógica de los trades."""
        print(f"\n🔍 VERIFICANDO LÓGICA DE {len(trades)} TRADES")
        print("=" * 60)
        
        valid_trades = 0
        total_pnl = 0
        
        for i, trade in enumerate(trades, 1):
            print(f"\n📋 TRADE #{i}: {trade['side']} ${trade['size_usd']}")
            
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            pnl = trade['pnl']
            duration = trade['duration_hours']
            
            print(f"   🕐 {entry_time.strftime('%Y-%m-%d %H:%M')} → {exit_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   💰 ${entry_price:.4f} → ${exit_price:.4f} ({trade['reason']})")
            print(f"   📊 P&L: ${pnl:.2f} | Duración: {duration:.1f}h")
            
            # Verificar que los precios existen en los datos
            entry_valid = self._price_exists_in_data(data, entry_time, entry_price)
            exit_valid = self._price_exists_in_data(data, exit_time, exit_price)
            
            # Verificar lógica del P&L
            if trade['side'] == 'LONG':
                expected_pnl_pct = (exit_price - entry_price) / entry_price
            else:
                expected_pnl_pct = (entry_price - exit_price) / entry_price
            
            expected_pnl = expected_pnl_pct * trade['size_usd'] * trade['leverage']
            pnl_diff = abs(pnl - expected_pnl)
            
            # Evaluación
            if entry_valid and exit_valid and pnl_diff < 1.0:
                print(f"   ✅ TRADE VÁLIDO")
                valid_trades += 1
                total_pnl += pnl
            else:
                print(f"   ❌ TRADE PROBLEMÁTICO")
                if not entry_valid:
                    print(f"      - Precio de entrada no encontrado")
                if not exit_valid:
                    print(f"      - Precio de salida no encontrado")
                if pnl_diff >= 1.0:
                    print(f"      - Error en cálculo P&L: esperado ${expected_pnl:.2f}, obtenido ${pnl:.2f}")
        
        # Resumen final
        print(f"\n🏆 RESUMEN DE VERIFICACIÓN - {symbol}")
        print("=" * 60)
        print(f"   📊 Trades válidos: {valid_trades}/{len(trades)}")
        print(f"   📈 Tasa de validez: {valid_trades/len(trades)*100:.1f}%")
        print(f"   💰 P&L total verificado: ${total_pnl:.2f}")
        
        # Evaluación final
        if valid_trades == len(trades):
            print(f"\n   🎯 EVALUACIÓN: TOTALMENTE CONFIABLE")
            print(f"      Todos los trades son válidos y verificables")
        elif valid_trades >= len(trades) * 0.8:
            print(f"\n   ✅ EVALUACIÓN: ALTAMENTE CONFIABLE")
            print(f"      La mayoría de trades son válidos")
        elif valid_trades >= len(trades) * 0.5:
            print(f"\n   ⚠️ EVALUACIÓN: MODERADAMENTE CONFIABLE")
            print(f"      Algunos trades presentan problemas")
        else:
            print(f"\n   ❌ EVALUACIÓN: POCO CONFIABLE")
            print(f"      Demasiados trades problemáticos")
    
    def _price_exists_in_data(self, data: pd.DataFrame, target_time: datetime, target_price: float) -> bool:
        """Verificar si un precio era alcanzable en un momento específico."""
        try:
            # Buscar vela más cercana (tolerancia de 1 hora)
            time_tolerance = pd.Timedelta(hours=1)
            mask = abs(data.index - target_time) <= time_tolerance
            candidates = data[mask]
            
            if candidates.empty:
                return False
            
            # Verificar si el precio está en el rango OHLC de alguna vela
            for _, candle in candidates.iterrows():
                if candle['Low'] <= target_price <= candle['High']:
                    return True
            
            return False
            
        except Exception:
            return False

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Verificador de Autenticidad de Backtest")
    parser.add_argument('symbol', help='Símbolo crypto (ej: XRPUSDT)')
    parser.add_argument('--timeframe', default='4h', help='Timeframe (1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=30, help='Días hacia atrás')
    
    args = parser.parse_args()
    
    verifier = RealBacktestVerifier()
    verifier.verify_with_real_data(args.symbol, args.timeframe, args.days)

if __name__ == "__main__":
    main()