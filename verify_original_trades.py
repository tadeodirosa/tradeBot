#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔍 VERIFICADOR DE TRADES ORIGINALES
==================================

Este script toma los trades generados por el futures_simulator original
y los verifica contra datos reales de Binance para confirmar su autenticidad.
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

# Agregar path para imports locales
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'core'))

# Imports locales
from futures_simulator import FuturesSimulator

# Configuración
FUTURES_CONFIG = {
    'initial_balance': 100,
    'leverage': 25,
    'position_size_usd': 100,
    'commission_rate': 0.0006,
    'max_positions': 3,
    'liquidation_threshold': 0.85,
}

class OriginalTradesVerifier:
    def __init__(self):
        self.binance_api = "https://api.binance.com/api/v3/klines"
        
    def verify_original_trades(self, symbol: str, timeframe: str = "4h"):
        """Verificar trades originales del futures_simulator."""
        print(f"🔍 VERIFICADOR DE TRADES ORIGINALES - {symbol} {timeframe}")
        print("=" * 60)
        
        # 1. Ejecutar el simulador original para obtener trades
        print(f"🎯 Obteniendo trades originales de {symbol}...")
        original_trades = self._get_original_trades(symbol, timeframe)
        
        if not original_trades:
            print("❌ No se pudieron obtener trades originales")
            return
        
        print(f"✅ Trades originales obtenidos: {len(original_trades)}")
        
        # 2. Obtener datos reales para el mismo período
        print(f"📡 Descargando datos reales para verificación...")
        real_data = self._get_real_data_for_period(symbol, timeframe, original_trades)
        
        if real_data is None or real_data.empty:
            print("❌ No se pudieron obtener datos reales")
            return
        
        # 3. Verificar cada trade original
        print(f"🔍 Verificando {len(original_trades)} trades originales...")
        self._verify_trades_against_real_data(original_trades, real_data, symbol)
        
    def _get_original_trades(self, symbol: str, timeframe: str) -> List[Dict]:
        """Obtener los trades originales ejecutando el futures_simulator."""
        try:
            # Configurar simulador con parámetros originales
            simulator = FuturesSimulator(
                initial_balance=FUTURES_CONFIG['initial_balance'],
                leverage=FUTURES_CONFIG['leverage'],
                position_size_usd=FUTURES_CONFIG['position_size_usd']
            )
            
            # Cargar datos del cache (como hace el simulador original)
            cache_file = f"data/cache_real/{symbol}_{timeframe}.json"
            if not os.path.exists(cache_file):
                print(f"❌ No existe archivo de cache: {cache_file}")
                return []
                
            # Cargar y procesar datos
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
            
            # Renombrar columnas para compatibilidad
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Limitarlo a últimos 30 días como hace el original
            end_date = df.index[-1]
            start_date = end_date - timedelta(days=30)
            df_limited = df[df.index >= start_date].copy()
            
            print(f"📊 Período de datos: {df_limited.index[0]} a {df_limited.index[-1]}")
            print(f"📈 Velas cargadas: {len(df_limited)}")
            
            # Ejecutar la lógica de trading para extraer trades
            trades = self._extract_trades_from_simulation(simulator, df_limited, symbol)
            
            return trades
            
        except Exception as e:
            print(f"❌ Error obteniendo trades originales: {e}")
            return []
    
    def _extract_trades_from_simulation(self, simulator: FuturesSimulator, data: pd.DataFrame, symbol: str) -> List[Dict]:
        """Extraer trades ejecutando la simulación completa."""
        trades = []
        
        try:
            # Importar el analizador profesional
            sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
            from professional_analyzer import ProfessionalCryptoAnalyzer
            
            # Inicializar analizador
            analyzer = ProfessionalCryptoAnalyzer()
            
            # Variables de tracking
            position_counter = 0
            
            for i in range(len(data)):
                current_candle = data.iloc[i]
                current_time = data.index[i]
                current_price = current_candle['Close']
                
                # Obtener ventana de análisis
                if i < 50:  # Necesitamos historia para indicadores
                    continue
                    
                window = data.iloc[max(0, i-50):i+1]
                
                # Analizar señales
                try:
                    signals = analyzer.analyze_comprehensive(window)
                    score = signals.get('score', 0)
                    confidence = signals.get('confidence', 0)
                    signal_strength = signals.get('signal_strength', 'HOLD')
                    
                    # Condiciones de entrada (igual que el original)
                    min_buy_score = 55
                    min_confidence = 0.50
                    
                    # Simular apertura de posición
                    if (score >= min_buy_score and confidence >= min_confidence and 
                        signal_strength in ['BUY', 'STRONG_BUY'] and 
                        len([p for p in simulator.positions.values() if p]) < 3):
                        
                        position_counter += 1
                        
                        # Calcular stop loss y take profit (simplificado)
                        atr = window['High'].rolling(14).max() - window['Low'].rolling(14).min()
                        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
                        
                        stop_loss = current_price - (current_atr * 2.2)
                        take_profit = current_price + (current_atr * 2.0)
                        
                        # Simular el trade completo (encontrar salida)
                        exit_info = self._simulate_trade_exit(
                            data, i, current_price, stop_loss, take_profit
                        )
                        
                        if exit_info:
                            # Calcular P&L
                            entry_price = current_price
                            exit_price = exit_info['exit_price']
                            exit_reason = exit_info['reason']
                            exit_time = exit_info['exit_time']
                            
                            pnl_pct = (exit_price - entry_price) / entry_price
                            pnl = pnl_pct * 100 * 25  # $100 * 25x leverage
                            
                            trade = {
                                'id': f"ORIGINAL_{position_counter}",
                                'symbol': symbol,
                                'side': 'LONG',
                                'entry_time': current_time.isoformat(),
                                'exit_time': exit_time.isoformat(),
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'size_usd': 100,
                                'leverage': 25,
                                'pnl': pnl,
                                'reason': exit_reason,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }
                            
                            trades.append(trade)
                            
                            # Límite de trades para evitar procesamiento excesivo
                            if len(trades) >= 25:
                                break
                                
                except Exception as e:
                    continue
            
            return trades
            
        except Exception as e:
            print(f"❌ Error en extracción de trades: {e}")
            return []
    
    def _simulate_trade_exit(self, data: pd.DataFrame, entry_idx: int, entry_price: float,
                           stop_loss: float, take_profit: float) -> Optional[Dict]:
        """Simular la salida de un trade."""
        try:
            liquidation_price = entry_price * 0.966  # Aproximadamente -3.4% para 25x
            
            # Buscar salida en las siguientes velas
            for j in range(entry_idx + 1, min(entry_idx + 20, len(data))):
                candle = data.iloc[j]
                current_time = data.index[j]
                low = candle['Low']
                high = candle['High']
                close = candle['Close']
                
                # Check liquidación (prioridad más alta)
                if low <= liquidation_price:
                    return {
                        'exit_price': liquidation_price,
                        'exit_time': current_time,
                        'reason': 'LIQUIDATION'
                    }
                
                # Check stop loss
                if low <= stop_loss:
                    return {
                        'exit_price': stop_loss,
                        'exit_time': current_time,
                        'reason': 'SL'
                    }
                
                # Check take profit
                if high >= take_profit:
                    return {
                        'exit_price': take_profit,
                        'exit_time': current_time,
                        'reason': 'TP'
                    }
            
            # Si no hay salida clara, salir en precio de cierre
            last_candle = data.iloc[min(entry_idx + 10, len(data) - 1)]
            return {
                'exit_price': last_candle['Close'],
                'exit_time': data.index[min(entry_idx + 10, len(data) - 1)],
                'reason': 'MANUAL'
            }
            
        except Exception:
            return None
    
    def _get_real_data_for_period(self, symbol: str, timeframe: str, trades: List[Dict]) -> Optional[pd.DataFrame]:
        """Obtener datos reales de Binance para el período de los trades."""
        try:
            if not trades:
                return None
            
            # Encontrar rango de fechas
            trade_times = []
            for trade in trades:
                trade_times.append(pd.to_datetime(trade['entry_time']))
                trade_times.append(pd.to_datetime(trade['exit_time']))
            
            start_time = min(trade_times) - timedelta(days=1)
            end_time = max(trade_times) + timedelta(days=1)
            
            print(f"📅 Período de trades: {start_time} a {end_time}")
            
            # Convertir timeframe
            tf_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
            interval = tf_map.get(timeframe, '4h')
            
            # Calcular límite
            if timeframe == '4h':
                limit = min(1000, int((end_time - start_time).total_seconds() / 14400) + 50)
            else:
                limit = 500
            
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
            
            # Procesar datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            print(f"✅ Datos reales obtenidos: {len(df)} velas")
            print(f"   Rango: {df.index[0]} a {df.index[-1]}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos reales: {e}")
            return None
    
    def _verify_trades_against_real_data(self, trades: List[Dict], real_data: pd.DataFrame, symbol: str):
        """Verificar trades originales contra datos reales."""
        print(f"\n🔍 VERIFICANDO {len(trades)} TRADES ORIGINALES")
        print("=" * 60)
        
        verified_count = 0
        total_pnl = 0
        impossible_trades = []
        
        for i, trade in enumerate(trades, 1):
            print(f"\n📋 TRADE #{i}: {trade['side']} ${trade['size_usd']}")
            
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            pnl = trade['pnl']
            
            print(f"   🕐 {entry_time.strftime('%Y-%m-%d %H:%M')} → {exit_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   💰 ${entry_price:.4f} → ${exit_price:.4f} ({trade['reason']})")
            print(f"   📊 P&L: ${pnl:.2f}")
            
            # Verificar precios
            entry_valid, entry_diff = self._verify_price_in_range(
                real_data, entry_time, entry_price, "ENTRADA"
            )
            
            exit_valid, exit_diff = self._verify_price_in_range(
                real_data, exit_time, exit_price, "SALIDA"  
            )
            
            # Verificar lógica del P&L
            expected_pnl_pct = (exit_price - entry_price) / entry_price
            expected_pnl = expected_pnl_pct * 100 * 25
            pnl_diff = abs(pnl - expected_pnl)
            
            # Evaluación
            if entry_valid and exit_valid and pnl_diff < 2.0:
                verified_count += 1
                total_pnl += pnl
                print(f"   ✅ TRADE VERIFICADO")
            else:
                impossible_trades.append(i)
                print(f"   ❌ TRADE CUESTIONABLE")
                if not entry_valid:
                    print(f"      - Entrada no verificable")
                if not exit_valid:
                    print(f"      - Salida no verificable")
                if pnl_diff >= 2.0:
                    print(f"      - P&L inconsistente")
        
        # Resumen final
        self._print_verification_summary(trades, verified_count, total_pnl, impossible_trades, symbol)
    
    def _verify_price_in_range(self, real_data: pd.DataFrame, target_time: datetime, 
                              target_price: float, label: str) -> Tuple[bool, float]:
        """Verificar si un precio está en el rango OHLC de datos reales."""
        try:
            # Buscar vela más cercana (tolerancia de 2 horas)
            time_tolerance = pd.Timedelta(hours=2)
            mask = abs(real_data.index - target_time) <= time_tolerance
            candidates = real_data[mask]
            
            if candidates.empty:
                print(f"   ⚠️ {label}: No hay datos cerca de {target_time}")
                return False, 0.0
            
            # Verificar en todas las velas candidatas
            for _, candle in candidates.iterrows():
                if candle['Low'] <= target_price <= candle['High']:
                    diff_pct = abs(target_price - candle['Close']) / candle['Close'] * 100
                    print(f"   ✅ {label}: ${target_price:.4f} válido (diff: {diff_pct:.2f}%)")
                    return True, diff_pct
            
            # Si no está en rango
            closest_candle = candidates.iloc[0]
            print(f"   ❌ {label}: ${target_price:.4f} fuera de rango [{closest_candle['Low']:.4f}-{closest_candle['High']:.4f}]")
            return False, 0.0
            
        except Exception as e:
            print(f"   ❌ {label}: Error - {e}")
            return False, 0.0
    
    def _print_verification_summary(self, trades: List[Dict], verified_count: int,
                                   total_pnl: float, impossible_trades: List, symbol: str):
        """Imprimir resumen de verificación."""
        print(f"\n🏆 RESUMEN DE VERIFICACIÓN - {symbol}")
        print("=" * 60)
        print(f"   📊 Trades verificados: {verified_count}/{len(trades)}")
        print(f"   📈 Tasa de verificación: {verified_count/len(trades)*100:.1f}%")
        print(f"   💰 P&L total verificado: ${total_pnl:.2f}")
        
        if verified_count > 0:
            avg_pnl = total_pnl / verified_count
            print(f"   💹 P&L promedio: ${avg_pnl:.2f}")
        
        # Evaluación final
        verification_rate = verified_count / len(trades)
        
        if verification_rate >= 0.9:
            print(f"\n   🎯 EVALUACIÓN: ALTAMENTE CONFIABLE")
            print(f"      Casi todos los trades son verificables")
        elif verification_rate >= 0.7:
            print(f"\n   ✅ EVALUACIÓN: CONFIABLE")
            print(f"      La mayoría de trades son verificables")
        elif verification_rate >= 0.5:
            print(f"\n   ⚠️ EVALUACIÓN: MODERADAMENTE CONFIABLE")
            print(f"      Algunos trades presentan problemas")
        else:
            print(f"\n   ❌ EVALUACIÓN: POCO CONFIABLE")
            print(f"      Demasiados trades no verificables")
        
        # Mostrar trades problemáticos
        if impossible_trades:
            print(f"\n🚨 TRADES PROBLEMÁTICOS: {len(impossible_trades)} de {len(trades)}")
            for trade_num in impossible_trades[:5]:
                print(f"   Trade #{trade_num}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Verificador de Trades Originales")
    parser.add_argument('symbol', help='Símbolo crypto (ej: LINKUSDT)')
    parser.add_argument('--timeframe', default='4h', help='Timeframe (1h, 4h, 1d)')
    
    args = parser.parse_args()
    
    verifier = OriginalTradesVerifier()
    verifier.verify_original_trades(args.symbol, args.timeframe)

if __name__ == "__main__":
    main()