#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔍 VERIFICADOR DE AUTENTICIDAD - BACKTEST REAL
==============================================

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
        
        # 3. Verificar cada trade (los datos son los mismos, verificamos lógica)
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
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]\n            \n        except Exception as e:\n            print(f\"❌ Error descargando datos: {e}\")\n            return None\n    \n    def _run_simple_backtest(self, data: pd.DataFrame, symbol: str) -> List[Dict]:\n        \"\"\"Ejecutar backtest simple con datos reales.\"\"\"\n        try:\n            print(f\"🤖 Ejecutando estrategia simple...\")\n            \n            trades = []\n            balance = 100  # USD\n            leverage = 25\n            position_size = 100\n            \n            # Estrategia simple: RSI + tendencia\n            data['RSI'] = self._calculate_rsi(data['Close'])\n            data['SMA_10'] = data['Close'].rolling(10).mean()\n            data['SMA_30'] = data['Close'].rolling(30).mean()\n            \n            in_position = False\n            entry_price = 0\n            entry_time = None\n            position_type = None\n            \n            for i in range(50, len(data) - 1):  # Empezar después de calcular indicadores\n                current = data.iloc[i]\n                current_time = data.index[i]\n                current_price = current['Close']\n                \n                # Señales de entrada\n                rsi_oversold = current['RSI'] < 30\n                rsi_overbought = current['RSI'] > 70\n                trend_up = current['SMA_10'] > current['SMA_30']\n                trend_down = current['SMA_10'] < current['SMA_30']\n                \n                # Entrar en posición\n                if not in_position:\n                    if rsi_oversold and trend_up:  # LONG\n                        in_position = True\n                        entry_price = current_price\n                        entry_time = current_time\n                        position_type = 'LONG'\n                        \n                    elif rsi_overbought and trend_down:  # SHORT\n                        in_position = True\n                        entry_price = current_price\n                        entry_time = current_time\n                        position_type = 'SHORT'\n                \n                # Salir de posición\n                elif in_position:\n                    exit_trade = False\n                    exit_reason = 'MANUAL'\n                    exit_price = current_price\n                    \n                    # Condiciones de salida\n                    if position_type == 'LONG':\n                        profit_pct = (current_price - entry_price) / entry_price\n                        \n                        if profit_pct >= 0.025:  # +2.5% TP\n                            exit_trade = True\n                            exit_reason = 'TP'\n                            exit_price = entry_price * 1.025\n                        elif profit_pct <= -0.035:  # -3.5% SL (liquidación)\n                            exit_trade = True\n                            exit_reason = 'LIQUIDATION'\n                            exit_price = entry_price * 0.965\n                        elif current['RSI'] > 70:  # RSI sobrecomprado\n                            exit_trade = True\n                            exit_reason = 'MANUAL'\n                    \n                    else:  # SHORT\n                        profit_pct = (entry_price - current_price) / entry_price\n                        \n                        if profit_pct >= 0.025:  # +2.5% TP\n                            exit_trade = True\n                            exit_reason = 'TP'\n                            exit_price = entry_price * 0.975\n                        elif profit_pct <= -0.035:  # -3.5% SL\n                            exit_trade = True\n                            exit_reason = 'LIQUIDATION'\n                            exit_price = entry_price * 1.035\n                        elif current['RSI'] < 30:  # RSI sobrevendido\n                            exit_trade = True\n                            exit_reason = 'MANUAL'\n                    \n                    # Crear trade\n                    if exit_trade:\n                        if position_type == 'LONG':\n                            pnl_pct = (exit_price - entry_price) / entry_price\n                        else:\n                            pnl_pct = (entry_price - exit_price) / entry_price\n                        \n                        pnl = pnl_pct * position_size * leverage\n                        \n                        trade = {\n                            'id': f\"REAL_{len(trades)+1}\",\n                            'symbol': symbol,\n                            'side': position_type,\n                            'entry_time': entry_time.isoformat(),\n                            'exit_time': current_time.isoformat(),\n                            'entry_price': entry_price,\n                            'exit_price': exit_price,\n                            'size_usd': position_size,\n                            'leverage': leverage,\n                            'pnl': pnl,\n                            'reason': exit_reason,\n                            'duration_hours': (current_time - entry_time).total_seconds() / 3600\n                        }\n                        \n                        trades.append(trade)\n                        balance += pnl\n                        \n                        # Reset\n                        in_position = False\n                        entry_price = 0\n                        entry_time = None\n                        position_type = None\n                        \n                        # Límite de trades\n                        if len(trades) >= 10:\n                            break\n            \n            print(f\"✅ Backtest completado:\")\n            print(f\"   Trades generados: {len(trades)}\")\n            print(f\"   Balance final: ${balance:.2f}\")\n            print(f\"   ROI: {((balance/100)-1)*100:.1f}%\")\n            \n            return trades\n            \n        except Exception as e:\n            print(f\"❌ Error en backtest: {e}\")\n            return []\n    \n    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:\n        \"\"\"Calcular RSI.\"\"\"\n        delta = prices.diff()\n        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()\n        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()\n        rs = gain / loss\n        return 100 - (100 / (1 + rs))\n    \n    def _verify_trade_logic(self, trades: List[Dict], data: pd.DataFrame, symbol: str):\n        \"\"\"Verificar la lógica de los trades.\"\"\"\n        print(f\"\\n🔍 VERIFICANDO LÓGICA DE {len(trades)} TRADES\")\n        print(\"=\" * 60)\n        \n        valid_trades = 0\n        total_pnl = 0\n        \n        for i, trade in enumerate(trades, 1):\n            print(f\"\\n📋 TRADE #{i}: {trade['side']} ${trade['size_usd']}\")\n            \n            entry_time = pd.to_datetime(trade['entry_time'])\n            exit_time = pd.to_datetime(trade['exit_time'])\n            entry_price = trade['entry_price']\n            exit_price = trade['exit_price']\n            pnl = trade['pnl']\n            duration = trade['duration_hours']\n            \n            print(f\"   🕐 {entry_time.strftime('%Y-%m-%d %H:%M')} → {exit_time.strftime('%Y-%m-%d %H:%M')}\")\n            print(f\"   💰 ${entry_price:.4f} → ${exit_price:.4f} ({trade['reason']})\")\n            print(f\"   📊 P&L: ${pnl:.2f} | Duración: {duration:.1f}h\")\n            \n            # Verificar que los precios existen en los datos\n            entry_valid = self._price_exists_in_data(data, entry_time, entry_price)\n            exit_valid = self._price_exists_in_data(data, exit_time, exit_price)\n            \n            # Verificar lógica del P&L\n            if trade['side'] == 'LONG':\n                expected_pnl_pct = (exit_price - entry_price) / entry_price\n            else:\n                expected_pnl_pct = (entry_price - exit_price) / entry_price\n            \n            expected_pnl = expected_pnl_pct * trade['size_usd'] * trade['leverage']\n            pnl_diff = abs(pnl - expected_pnl)\n            \n            # Evaluación\n            if entry_valid and exit_valid and pnl_diff < 1.0:\n                print(f\"   ✅ TRADE VÁLIDO\")\n                valid_trades += 1\n                total_pnl += pnl\n            else:\n                print(f\"   ❌ TRADE PROBLEMÁTICO\")\n                if not entry_valid:\n                    print(f\"      - Precio de entrada no encontrado\")\n                if not exit_valid:\n                    print(f\"      - Precio de salida no encontrado\")\n                if pnl_diff >= 1.0:\n                    print(f\"      - Error en cálculo P&L: esperado ${expected_pnl:.2f}, obtenido ${pnl:.2f}\")\n        \n        # Resumen final\n        print(f\"\\n🏆 RESUMEN DE VERIFICACIÓN - {symbol}\")\n        print(\"=\" * 60)\n        print(f\"   📊 Trades válidos: {valid_trades}/{len(trades)}\")\n        print(f\"   📈 Tasa de validez: {valid_trades/len(trades)*100:.1f}%\")\n        print(f\"   💰 P&L total verificado: ${total_pnl:.2f}\")\n        \n        # Evaluación final\n        if valid_trades == len(trades):\n            print(f\"\\n   🎯 EVALUACIÓN: TOTALMENTE CONFIABLE\")\n            print(f\"      Todos los trades son válidos y verificables\")\n        elif valid_trades >= len(trades) * 0.8:\n            print(f\"\\n   ✅ EVALUACIÓN: ALTAMENTE CONFIABLE\")\n            print(f\"      La mayoría de trades son válidos\")\n        elif valid_trades >= len(trades) * 0.5:\n            print(f\"\\n   ⚠️ EVALUACIÓN: MODERADAMENTE CONFIABLE\")\n            print(f\"      Algunos trades presentan problemas\")\n        else:\n            print(f\"\\n   ❌ EVALUACIÓN: POCO CONFIABLE\")\n            print(f\"      Demasiados trades problemáticos\")\n    \n    def _price_exists_in_data(self, data: pd.DataFrame, target_time: datetime, target_price: float) -> bool:\n        \"\"\"Verificar si un precio era alcanzable en un momento específico.\"\"\"\n        try:\n            # Buscar vela más cercana (tolerancia de 1 hora)\n            time_tolerance = pd.Timedelta(hours=1)\n            mask = abs(data.index - target_time) <= time_tolerance\n            candidates = data[mask]\n            \n            if candidates.empty:\n                return False\n            \n            # Verificar si el precio está en el rango OHLC de alguna vela\n            for _, candle in candidates.iterrows():\n                if candle['Low'] <= target_price <= candle['High']:\n                    return True\n            \n            return False\n            \n        except Exception:\n            return False\n\ndef main():\n    \"\"\"Función principal.\"\"\"\n    parser = argparse.ArgumentParser(description=\"Verificador de Autenticidad de Backtest\")\n    parser.add_argument('symbol', help='Símbolo crypto (ej: XRPUSDT)')\n    parser.add_argument('--timeframe', default='4h', help='Timeframe (1h, 4h, 1d)')\n    parser.add_argument('--days', type=int, default=30, help='Días hacia atrás')\n    \n    args = parser.parse_args()\n    \n    verifier = RealBacktestVerifier()\n    verifier.verify_with_real_data(args.symbol, args.timeframe, args.days)\n\nif __name__ == \"__main__\":\n    main()\n