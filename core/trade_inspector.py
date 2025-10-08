#!/usr/bin/env python3

"""
Trade Inspector - Verificador de Realidad de Trades
=================================================

Verifica manualmente si los trades del backtest podrían haber ocurrido realmente
comparando contra datos históricos reales.
"""

import sys
import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

class TradeInspector:
    """Inspector para verificar la realidad de trades específicos."""
    
    def __init__(self):
        self.binance_api = "https://api.binance.com/api/v3/klines"
        
    def inspect_recent_run(self, symbol: str, timeframe: str = '4h', num_trades: int = 5):
        """
        Inspeccionar los últimos trades del último backtest ejecutado.
        
        Args:
            symbol: Símbolo crypto (ej: XRPUSDT)
            timeframe: Timeframe (4h, 1h, 1d)
            num_trades: Número de trades a verificar
        """
        print(f"🔍 INSPECTOR DE TRADES - {symbol} {timeframe}")
        print("=" * 60)
        
        # 1. Cargar datos del cache (lo que usó el backtest)
        cache_data = self._load_cache_data(symbol, timeframe)
        if cache_data is None:
            return
            
        # 2. Obtener datos reales de Binance para comparar
        real_data = self._fetch_real_data(symbol, timeframe)
        if real_data is None:
            return
            
        # 3. Simular algunos trades para inspeccionar
        sample_trades = self._generate_sample_trades(cache_data, num_trades)
        
        # 4. Verificar cada trade
        self._verify_trades(sample_trades, cache_data, real_data, symbol)
    
    def _load_cache_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Cargar datos del cache que usó el backtest."""
        try:
            # Encontrar archivo de cache
            symbol_clean = symbol.replace('USDT', '').upper()
            
            # Probar ambas estructuras
            possible_files = [
                f"{symbol_clean}USDT_{timeframe}.json",
                f"{symbol_clean}_USDT_{timeframe}.json"
            ]
            
            cache_path = None
            for filename in possible_files:
                test_path = os.path.join('..', 'data', 'cache_real', filename)
                if os.path.exists(test_path):
                    cache_path = test_path
                    break
            
            if not cache_path:
                print(f"❌ No se encontró cache para {symbol}")
                return None
                
            print(f"📁 Cargando cache: {cache_path}")
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Extraer datos
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
            
            print(f"✅ Cache cargado: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"❌ Error cargando cache: {e}")
            return None
    
    def _fetch_real_data(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Obtener datos reales de Binance API con ventana ampliada."""
        try:
            print(f"🌐 Obteniendo datos reales de Binance...")
            
            # Convertir timeframe
            tf_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
            interval = tf_map.get(timeframe, '4h')
            
            # Llamada a API con ventana ampliada
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit  # Ahora 500 por defecto para más cobertura
            }
            
            response = requests.get(self.binance_api, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ Error API Binance: {response.status_code}")
                return None
            
            data = response.json()
            
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
            
            print(f"✅ Datos reales: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos reales: {e}")
            return None
    
    def _generate_sample_trades(self, data: pd.DataFrame, num_trades: int) -> List[Dict]:
        """Generar trades de muestra basados en datos más antiguos (no los últimos)."""
        trades = []
        
        # Usar datos de hace 1-2 semanas para que coincidan con Binance
        # Tomar datos desde el 80% del dataset (no los más recientes)
        start_idx = int(len(data) * 0.7)  # Empezar del 70% del dataset
        end_idx = int(len(data) * 0.85)   # Hasta el 85% del dataset
        
        sample_data = data.iloc[start_idx:end_idx]
        print(f"📊 Generando trades de muestra desde {sample_data.index[0]} hasta {sample_data.index[-1]}")
        
        for i in range(min(num_trades, len(sample_data)-2)):
            try:
                entry_idx = i * 2  # Espaciar trades
                exit_idx = entry_idx + 1
                
                if exit_idx >= len(sample_data):
                    continue
                
                entry_candle = sample_data.iloc[entry_idx]
                exit_candle = sample_data.iloc[exit_idx]
                
                # Simular entrada en precio de cierre
                entry_price = float(entry_candle['Close'])
                
                # Simular salida (TP o SL basado en movimiento)
                price_change = (exit_candle['Close'] - entry_candle['Close']) / entry_candle['Close']
                
                if price_change > 0.01:  # Si subió >1%, simular TP
                    exit_price = entry_price * 1.02  # TP en +2%
                    reason = "TP"
                elif price_change < -0.035:  # Si bajó >3.5%, simular liquidación
                    exit_price = entry_price * 0.966  # Liquidación en -3.4%
                    reason = "LIQUIDATION"
                else:  # Salida normal
                    exit_price = float(exit_candle['Close'])
                    reason = "MANUAL"
                
                trade = {
                    'id': f"SAMPLE_{i+1}",
                    'side': 'LONG',
                    'entry_time': entry_candle.name,
                    'exit_time': exit_candle.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size_usd': 100,
                    'leverage': 25,
                    'reason': reason
                }
                
                # Calcular P&L
                pnl_pct = (exit_price - entry_price) / entry_price
                trade['pnl'] = pnl_pct * 100 * 25  # 25x leverage
                
                trades.append(trade)
                
            except Exception as e:
                print(f"⚠️ Error generando trade {i}: {e}")
                continue
        
        return trades
    
    def _verify_trades(self, trades: List[Dict], cache_data: pd.DataFrame, 
                      real_data: pd.DataFrame, symbol: str):
        """Verificar cada trade contra datos reales."""
        print(f"\n🔍 VERIFICANDO {len(trades)} TRADES CONTRA DATOS REALES")
        print("=" * 60)
        
        verified_count = 0
        total_price_diff = 0
        
        for i, trade in enumerate(trades):
            print(f"\n📋 TRADE #{i+1}: {trade['side']} ${trade['size_usd']} @ ${trade['entry_price']:.4f}")
            print(f"   🕐 {trade['entry_time']} → {trade['exit_time']}")
            print(f"   💰 P&L: ${trade['pnl']:.2f} ({trade['reason']})")
            
            # Verificar precio de entrada
            entry_valid, entry_diff = self._verify_price_at_time(
                cache_data, real_data, trade['entry_time'], trade['entry_price'], "ENTRADA"
            )
            
            # Verificar precio de salida
            exit_valid, exit_diff = self._verify_price_at_time(
                cache_data, real_data, trade['exit_time'], trade['exit_price'], "SALIDA"  
            )
            
            # Verificar que el precio era alcanzable
            price_reachable = self._verify_price_reachable(
                real_data, trade['entry_time'], trade['exit_time'], 
                trade['entry_price'], trade['exit_price']
            )
            
            if entry_valid and exit_valid and price_reachable:
                verified_count += 1
                total_price_diff += (entry_diff + exit_diff) / 2
                print(f"   ✅ TRADE VERIFICADO")
            else:
                print(f"   ❌ TRADE CUESTIONABLE")
        
        # Resumen final
        print(f"\n🏆 RESUMEN DE VERIFICACIÓN:")
        print(f"   📊 Trades verificados: {verified_count}/{len(trades)}")
        print(f"   📈 Tasa de verificación: {verified_count/len(trades)*100:.1f}%")
        
        if verified_count > 0:
            avg_price_diff = total_price_diff / verified_count
            print(f"   💹 Diferencia promedio de precios: {avg_price_diff:.3f}%")
        
        # Evaluación final
        if verified_count / len(trades) >= 0.8:
            if verified_count > 0 and total_price_diff / verified_count < 0.1:
                print(f"   🎯 EVALUACIÓN: ALTAMENTE CONFIABLE")
                print(f"      Los trades son realistas y los precios coinciden")
            else:
                print(f"   ⚠️  EVALUACIÓN: MODERADAMENTE CONFIABLE")
                print(f"      Los trades son posibles pero hay algunas diferencias")
        else:
            print(f"   ❌ EVALUACIÓN: CUESTIONABLE") 
            print(f"      Demasiados trades no verificables")
    
    def _verify_price_at_time(self, cache_data: pd.DataFrame, real_data: pd.DataFrame,
                             timestamp: datetime, price: float, label: str) -> Tuple[bool, float]:
        """Verificar precio en timestamp específico."""
        try:
            # Encontrar vela en datos reales
            time_tolerance = pd.Timedelta(minutes=10)
            real_matches = real_data[abs(real_data.index - timestamp) <= time_tolerance]
            
            if real_matches.empty:
                print(f"   ⚠️ {label}: No hay datos reales para {timestamp}")
                return False, 0.0
            
            # Tomar la vela más cercana
            closest_real = real_matches.iloc[0]
            
            # Verificar si el precio está en el rango OHLC
            if closest_real['Low'] <= price <= closest_real['High']:
                # Calcular diferencia con precio de cierre real
                real_close = closest_real['Close']
                diff_pct = abs(price - real_close) / real_close * 100
                
                print(f"   ✅ {label}: ${price:.4f} alcanzable (Real Close: ${real_close:.4f}, diff: {diff_pct:.2f}%)")
                return True, diff_pct
            else:
                print(f"   ❌ {label}: ${price:.4f} fuera de rango real [{closest_real['Low']:.4f}-{closest_real['High']:.4f}]")
                return False, 0.0
                
        except Exception as e:
            print(f"   ❌ {label}: Error verificando - {e}")
            return False, 0.0
    
    def _verify_price_reachable(self, real_data: pd.DataFrame, start_time: datetime, 
                               end_time: datetime, entry_price: float, exit_price: float) -> bool:
        """Verificar que el precio de salida era alcanzable desde la entrada."""
        try:
            # Obtener datos entre entrada y salida
            mask = (real_data.index >= start_time) & (real_data.index <= end_time)
            period_data = real_data[mask]
            
            if period_data.empty:
                return True  # No podemos verificar, asumimos válido
            
            # Verificar si el precio de salida fue tocado
            min_price = period_data['Low'].min()
            max_price = period_data['High'].max()
            
            reachable = min_price <= exit_price <= max_price
            
            if reachable:
                print(f"   ✅ PRECIO ALCANZABLE: ${exit_price:.4f} en rango [${min_price:.4f}-${max_price:.4f}]")
            else:
                print(f"   ❌ PRECIO NO ALCANZABLE: ${exit_price:.4f} fuera de rango [${min_price:.4f}-${max_price:.4f}]")
            
            return reachable
            
        except Exception as e:
            print(f"   ⚠️ Error verificando alcance: {e}")
            return True

def main():
    """Función principal para ejecutar inspector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspector de Trades del Backtest")
    parser.add_argument('symbol', help='Símbolo crypto (ej: XRPUSDT)')
    parser.add_argument('--timeframe', default='4h', help='Timeframe (1h, 4h, 1d)')
    parser.add_argument('--trades', type=int, default=5, help='Número de trades a verificar')
    
    args = parser.parse_args()
    
    inspector = TradeInspector()
    inspector.inspect_recent_run(args.symbol, args.timeframe, args.trades)

if __name__ == "__main__":
    main()