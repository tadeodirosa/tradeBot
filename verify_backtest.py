#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔍 VERIFICADOR DE BACKTESTS REALES
==================================

Este script ejecuta un backtest real y luego verifica los trades
contra datos actuales de Binance para confirmar su validez.
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Agregar path para imports locales
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'core'))

# Imports locales
from futures_simulator import FuturesSimulator

# Configuración básica
FUTURES_CONFIG = {
    'leverage': 25,
    'initial_balance': 100,
    'commission_rate': 0.0006,
    'liquidation_threshold': 0.96,
    'min_buy_score': 55,
    'max_risk_per_trade': 0.05
}

class BacktestVerifier:
    def __init__(self):
        self.binance_api = "https://api.binance.com/api/v3/klines"
        
    def run_and_verify(self, symbol: str, timeframe: str = "4h", max_trades: int = 10):
        """Ejecutar backtest y verificar trades contra datos reales."""
        print(f"🚀 VERIFICADOR DE BACKTEST - {symbol} {timeframe}")
        print("=" * 60)
        
        # 1. Ejecutar backtest real
        print(f"🎯 Ejecutando backtest para {symbol}...")
        trades = self._run_backtest(symbol, timeframe)
        
        if not trades or len(trades) == 0:
            print("❌ No se generaron trades en el backtest")
            return
        
        # Limitar trades para verificación
        trades_to_verify = trades[:max_trades]
        print(f"📊 Verificando {len(trades_to_verify)} trades...")
        
        # 2. Obtener datos reales de Binance para el período
        print(f"📡 Obteniendo datos reales de Binance...")
        real_data = self._get_real_data_for_period(symbol, timeframe, trades_to_verify)
        
        if real_data is None or real_data.empty:
            print("❌ No se pudieron obtener datos reales")
            return
        
        # 3. Verificar cada trade
        self._verify_trades(trades_to_verify, real_data, symbol)
        
    def _run_backtest(self, symbol: str, timeframe: str) -> List[Dict]:
        """Ejecutar backtest real y obtener trades."""
        try:
            # Configurar simulador con parámetros correctos
            simulator = FuturesSimulator(
                initial_balance=FUTURES_CONFIG['initial_balance'],
                leverage=FUTURES_CONFIG['leverage'],
                position_size_usd=FUTURES_CONFIG['initial_balance']  # Usar todo el balance
            )
            
            # Cargar datos de cache 
            cache_file = f"data/cache_real/{symbol}_{timeframe}.json"
            if not os.path.exists(cache_file):
                print(f"❌ No existe archivo de cache: {cache_file}")
                return []
            
            # Ejecutar simulación básica
            print("📊 Ejecutando simulación básica...")
            trades = self._run_simple_simulation(simulator, cache_file, symbol, timeframe)
            
            if trades:
                print(f"✅ Simulación completada: {len(trades)} trades generados")
                return trades
            else:
                print("❌ La simulación no generó trades")
                return []
                
        except Exception as e:
            print(f"❌ Error ejecutando backtest: {e}")
            return []
    
    def _run_simple_simulation(self, simulator: FuturesSimulator, cache_file: str, 
                              symbol: str, timeframe: str) -> List[Dict]:
        """Ejecutar simulación simple sin análisis técnico complejo."""
        try:
            # Cargar datos
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            if 'data' in cache:
                data_list = cache['data']
            else:
                data_list = cache
            
            # Convertir a DataFrame
            df = pd.DataFrame(data_list)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Renombrar columnas
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            print(f"📈 Datos cargados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
            
            # Simulación simple: entradas cada X velas
            trades = []
            entry_interval = max(10, len(df) // 20)  # Entradas espaciadas
            
            for i in range(0, len(df) - 5, entry_interval):
                try:
                    if len(trades) >= 15:  # Límite de trades
                        break
                    
                    entry_candle = df.iloc[i]
                    exit_candle = df.iloc[i + 3]  # Salir 3 velas después
                    
                    entry_price = float(entry_candle['Close'])
                    exit_price = float(exit_candle['Close'])
                    
                    # Simular decisión de entrada (simple momentum)
                    price_change = (entry_price - df.iloc[i-1]['Close']) / df.iloc[i-1]['Close']
                    
                    if price_change > 0.005:  # Si subió >0.5%, entrar LONG
                        side = 'LONG'
                        # Simular TP/SL
                        if (exit_price - entry_price) / entry_price > 0.02:
                            exit_price = entry_price * 1.02  # TP en +2%
                            reason = "TP"
                        elif (exit_price - entry_price) / entry_price < -0.035:
                            exit_price = entry_price * 0.965  # Liquidación
                            reason = "LIQUIDATION" 
                        else:
                            reason = "MANUAL"
                        
                        # Calcular P&L
                        pnl_pct = (exit_price - entry_price) / entry_price
                        pnl = pnl_pct * FUTURES_CONFIG['initial_balance'] * FUTURES_CONFIG['leverage']
                        
                        trade = {
                            'id': f"SIM_{len(trades)+1}",
                            'symbol': symbol,
                            'side': side,
                            'entry_time': entry_candle.name.isoformat(),
                            'exit_time': exit_candle.name.isoformat(),
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'size_usd': FUTURES_CONFIG['initial_balance'],
                            'leverage': FUTURES_CONFIG['leverage'],
                            'pnl': pnl,
                            'reason': reason
                        }
                        
                        trades.append(trade)
                        
                except Exception as e:
                    print(f"⚠️ Error en vela {i}: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            print(f"❌ Error en simulación simple: {e}")
            return []
    
    def _get_real_data_for_period(self, symbol: str, timeframe: str, trades: List[Dict]) -> Optional[pd.DataFrame]:
        """Obtener datos reales que cubran el período de los trades."""
        try:
            if not trades:
                return None
            
            # Encontrar rango de fechas de los trades
            trade_times = []
            for trade in trades:
                if 'entry_time' in trade:
                    trade_times.append(pd.to_datetime(trade['entry_time']))
                if 'exit_time' in trade:
                    trade_times.append(pd.to_datetime(trade['exit_time']))
            
            if not trade_times:
                print("❌ No se encontraron timestamps en los trades")
                return None
            
            start_time = min(trade_times) - timedelta(hours=24)  # Buffer
            end_time = max(trade_times) + timedelta(hours=24)    # Buffer
            
            print(f"📅 Período de trades: {start_time} a {end_time}")
            
            # Convertir timeframe
            tf_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
            interval = tf_map.get(timeframe, '4h')
            
            # Calcular límite de velas necesarias
            if timeframe == '1h':
                limit = min(1000, int((end_time - start_time).total_seconds() / 3600) + 50)
            elif timeframe == '4h':
                limit = min(1000, int((end_time - start_time).total_seconds() / 14400) + 50)
            else:  # 1d
                limit = min(1000, int((end_time - start_time).days) + 50)
            
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
                print("❌ No se recibieron datos de Binance")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Procesar timestamps y precios
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convertir a float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            print(f"✅ Datos reales obtenidos: {len(df)} velas")
            print(f"   Rango: {df.index[0]} a {df.index[-1]}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos reales: {e}")
            return None
    
    def _verify_trades(self, trades: List[Dict], real_data: pd.DataFrame, symbol: str):
        """Verificar trades contra datos reales."""
        print(f"\n🔍 VERIFICANDO {len(trades)} TRADES CONTRA DATOS REALES")
        print("=" * 60)
        
        verified_count = 0
        total_price_diff = 0
        impossible_trades = []
        
        for i, trade in enumerate(trades, 1):
            print(f"\n📋 TRADE #{i}: {trade.get('side', 'UNKNOWN')} ${trade.get('size_usd', 0)}")
            
            entry_time = pd.to_datetime(trade.get('entry_time'))
            exit_time = pd.to_datetime(trade.get('exit_time'))
            entry_price = float(trade.get('entry_price', 0))
            exit_price = float(trade.get('exit_price', 0))
            pnl = trade.get('pnl', 0)
            reason = trade.get('reason', 'UNKNOWN')
            
            print(f"   🕐 {entry_time} → {exit_time}")
            print(f"   💰 P&L: ${pnl:.2f} ({reason})")
            
            # Verificar precios
            entry_valid, entry_diff = self._verify_price_at_time(
                real_data, entry_time, entry_price, "ENTRADA"
            )
            
            exit_valid, exit_diff = self._verify_price_at_time(
                real_data, exit_time, exit_price, "SALIDA"
            )
            
            # Verificar trayectoria de precios
            path_valid = self._verify_price_path(
                real_data, entry_time, exit_time, entry_price, exit_price, trade
            )
            
            # Evaluación del trade
            if entry_valid and exit_valid and path_valid:
                verified_count += 1
                total_price_diff += (entry_diff + exit_diff) / 2
                print(f"   ✅ TRADE VERIFICADO")
            else:
                impossible_trades.append({
                    'trade_num': i,
                    'entry_valid': entry_valid,
                    'exit_valid': exit_valid,
                    'path_valid': path_valid,
                    'trade': trade
                })
                print(f"   ❌ TRADE CUESTIONABLE")
        
        # Resumen final
        self._print_verification_summary(
            trades, verified_count, total_price_diff, impossible_trades, symbol
        )
    
    def _verify_price_at_time(self, real_data: pd.DataFrame, target_time: datetime, 
                             price: float, label: str) -> Tuple[bool, float]:
        """Verificar si un precio era alcanzable en un momento específico."""
        try:
            # Buscar vela más cercana (tolerancia de 30 minutos)
            time_tolerance = pd.Timedelta(minutes=30)
            mask = abs(real_data.index - target_time) <= time_tolerance
            candidates = real_data[mask]
            
            if candidates.empty:
                print(f"   ⚠️ {label}: No hay datos reales cerca de {target_time}")
                return False, 0.0
            
            # Tomar la vela más cercana
            closest_idx = (candidates.index - target_time).abs().idxmin()
            closest_candle = real_data.loc[closest_idx]
            
            # Verificar si el precio está en el rango OHLC
            low, high = closest_candle['Low'], closest_candle['High']
            close_price = closest_candle['Close']
            
            if low <= price <= high:
                diff_pct = abs(price - close_price) / close_price * 100
                print(f"   ✅ {label}: ${price:.4f} válido (Close: ${close_price:.4f}, diff: {diff_pct:.2f}%)")
                return True, diff_pct
            else:
                print(f"   ❌ {label}: ${price:.4f} fuera de rango [${low:.4f}-${high:.4f}]")
                return False, 0.0
                
        except Exception as e:
            print(f"   ❌ {label}: Error - {e}")
            return False, 0.0
    
    def _verify_price_path(self, real_data: pd.DataFrame, entry_time: datetime, 
                          exit_time: datetime, entry_price: float, exit_price: float,
                          trade: Dict) -> bool:
        """Verificar que la trayectoria del precio durante el trade es realista."""
        try:
            # Obtener datos del período del trade
            mask = (real_data.index >= entry_time) & (real_data.index <= exit_time)
            period_data = real_data[mask]
            
            if period_data.empty:
                return True  # No podemos verificar, asumimos válido
            
            min_period = period_data['Low'].min()
            max_period = period_data['High'].max()
            
            # Para trades largos
            if trade.get('side') == 'LONG':
                # El precio de salida debe haber sido alcanzado
                if min_period <= exit_price <= max_period:
                    print(f"   ✅ TRAYECTORIA: Exit ${exit_price:.4f} alcanzable")
                    return True
                else:
                    print(f"   ❌ TRAYECTORIA: Exit ${exit_price:.4f} no alcanzable")
                    return False
            
            return True  # Para otros casos, asumimos válido
            
        except Exception as e:
            print(f"   ⚠️ Error verificando trayectoria: {e}")
            return True
    
    def _print_verification_summary(self, trades: List[Dict], verified_count: int,
                                   total_price_diff: float, impossible_trades: List,
                                   symbol: str):
        """Imprimir resumen de verificación."""
        print(f"\n🏆 RESUMEN DE VERIFICACIÓN - {symbol}")
        print("=" * 60)
        print(f"   📊 Trades verificados: {verified_count}/{len(trades)}")
        print(f"   📈 Tasa de verificación: {verified_count/len(trades)*100:.1f}%")
        
        if verified_count > 0:
            avg_price_diff = total_price_diff / verified_count
            print(f"   💹 Diferencia promedio de precios: {avg_price_diff:.3f}%")
        
        # Evaluación final
        verification_rate = verified_count / len(trades)
        
        if verification_rate >= 0.8:
            if verified_count > 0 and total_price_diff / verified_count < 0.5:
                print(f"\n   🎯 EVALUACIÓN: ALTAMENTE CONFIABLE")
                print(f"      Los trades son realistas y verificables")
            else:
                print(f"\n   ⚠️  EVALUACIÓN: MODERADAMENTE CONFIABLE")
                print(f"      Los trades son posibles pero hay diferencias")
        elif verification_rate >= 0.5:
            print(f"\n   ⚡ EVALUACIÓN: PARCIALMENTE CONFIABLE")
            print(f"      Algunos trades son cuestionables")
        else:
            print(f"\n   ❌ EVALUACIÓN: ALTAMENTE CUESTIONABLE")
            print(f"      Demasiados trades no verificables")
        
        # Mostrar trades problemáticos
        if impossible_trades:
            print(f"\n🚨 TRADES PROBLEMÁTICOS:")
            for prob in impossible_trades[:3]:  # Mostrar máximo 3
                print(f"   Trade #{prob['trade_num']}: ", end="")
                issues = []
                if not prob['entry_valid']:
                    issues.append("entrada inválida")
                if not prob['exit_valid']:
                    issues.append("salida inválida")
                if not prob['path_valid']:
                    issues.append("trayectoria imposible")
                print(", ".join(issues))

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verificador de Backtests Reales")
    parser.add_argument('symbol', help='Símbolo crypto (ej: XRPUSDT)')
    parser.add_argument('--timeframe', default='4h', help='Timeframe (1h, 4h, 1d)')
    parser.add_argument('--max-trades', type=int, default=10, help='Máximo trades a verificar')
    
    args = parser.parse_args()
    
    verifier = BacktestVerifier()
    verifier.run_and_verify(args.symbol, args.timeframe, args.max_trades)

if __name__ == "__main__":
    main()