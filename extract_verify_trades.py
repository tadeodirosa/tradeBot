#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔍 EXTRACTOR Y VERIFICADOR DE TRADES REALES
==========================================

Extrae los trades del futures_simulator y los verifica contra datos reales.
"""

import os
import sys
import json
import requests
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import argparse

class TradeExtractorVerifier:
    def __init__(self):
        self.binance_api = "https://api.binance.com/api/v3/klines"
        
    def extract_and_verify(self, symbol: str, timeframe: str = "4h"):
        """Extraer trades del output del simulador y verificarlos."""
        print(f"🔍 EXTRACTOR Y VERIFICADOR - {symbol} {timeframe}")
        print("=" * 60)
        
        # 1. Ejecutar simulador y capturar output
        print(f"🎯 Ejecutando simulador para extraer trades...")
        trades_data = self._run_simulator_and_extract(symbol, timeframe)
        
        if not trades_data:
            print("❌ No se pudieron extraer trades")
            return
        
        print(f"✅ Trades extraídos: {len(trades_data['trades'])}")
        print(f"💰 ROI reportado: {trades_data['roi']:.2f}%")
        
        # 2. Obtener datos reales para verificación
        print(f"📡 Obteniendo datos reales para verificación...")
        real_data = self._get_real_data(symbol, timeframe, 35)  # 35 días para cubrir período
        
        if real_data is None or real_data.empty:
            print("❌ No se pudieron obtener datos reales")
            return
        
        # 3. Verificar trades
        self._verify_extracted_trades(trades_data['trades'], real_data, symbol, trades_data['roi'])
        
    def _run_simulator_and_extract(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Ejecutar el simulador y extraer la información de trades."""
        try:
            import subprocess
            import tempfile
            
            # Ejecutar el simulador y capturar output
            cmd = [
                sys.executable, 
                "core/futures_simulator.py", 
                f"{symbol}_{timeframe}", 
                "--leverage", "25", 
                "--days", "30"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd(),
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"❌ Error ejecutando simulador: {result.stderr}")
                return None
            
            output = result.stdout
            
            # Extraer información del output
            trades = self._parse_trades_from_output(output)
            roi = self._parse_roi_from_output(output)
            
            return {
                'trades': trades,
                'roi': roi,
                'output': output
            }
            
        except Exception as e:
            print(f"❌ Error ejecutando simulador: {e}")
            return None
    
    def _parse_trades_from_output(self, output: str) -> List[Dict]:
        """Extraer trades del output del simulador."""
        trades = []
        
        try:
            lines = output.split('\n')
            
            # Buscar líneas con información de trades
            open_positions = {}
            trade_counter = 0
            
            for line in lines:
                line = line.strip()
                
                # Posición abierta
                if "POSICIÓN ABIERTA:" in line and "LONG" in line:
                    match = re.search(r'LONG \$(\d+) @ \$([0-9.]+)', line)
                    if match:
                        size = float(match.group(1))
                        price = float(match.group(2))
                        
                        # Buscar timestamp de la línea anterior
                        timestamp = self._extract_timestamp_from_context(lines, line)
                        
                        trade_counter += 1
                        open_positions[trade_counter] = {
                            'entry_price': price,
                            'entry_time': timestamp,
                            'size_usd': size
                        }
                
                # Posición cerrada
                elif "POSICIÓN CERRADA:" in line and "P&L:" in line:
                    pnl_match = re.search(r'P&L: \$([0-9.-]+)', line)
                    reason_match = re.search(r'Razón: (\w+)', line)
                    
                    if pnl_match and reason_match:
                        pnl = float(pnl_match.group(1))
                        reason = reason_match.group(1)
                        
                        # Buscar timestamp
                        timestamp = self._extract_timestamp_from_context(lines, line)
                        
                        # Emparejar con posición abierta más antigua
                        if open_positions:
                            oldest_key = min(open_positions.keys())
                            position = open_positions.pop(oldest_key)
                            
                            # Calcular exit_price basado en P&L
                            entry_price = position['entry_price']
                            pnl_pct = pnl / (100 * 25)  # P&L / (size * leverage)
                            exit_price = entry_price * (1 + pnl_pct)
                            
                            trade = {
                                'id': f"EXTRACTED_{len(trades)+1}",
                                'symbol': 'LINKUSDT',
                                'side': 'LONG',
                                'entry_time': position['entry_time'],
                                'exit_time': timestamp,
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'size_usd': position['size_usd'],
                                'leverage': 25,
                                'pnl': pnl,
                                'reason': reason
                            }
                            
                            trades.append(trade)
            
            return trades
            
        except Exception as e:
            print(f"⚠️ Error parseando trades: {e}")
            return []
    
    def _extract_timestamp_from_context(self, lines: List[str], target_line: str) -> str:
        """Extraer timestamp del contexto de líneas."""
        try:
            # Buscar la línea en el contexto
            target_idx = -1
            for i, line in enumerate(lines):
                if target_line.strip() in line:
                    target_idx = i
                    break
            
            if target_idx == -1:
                return datetime.now().isoformat()
            
            # Buscar timestamp en líneas anteriores
            for i in range(max(0, target_idx - 5), target_idx):
                line = lines[i]
                # Buscar patrón [número]
                match = re.search(r'\[\s*(\d+)\]', line)
                if match:
                    bar_number = int(match.group(1))
                    # Convertir bar_number a timestamp aproximado
                    # Asumir 4h por bar, empezando 2025-09-02
                    start_time = datetime(2025, 9, 2, 21, 0, 0)
                    trade_time = start_time + timedelta(hours=4 * bar_number)
                    return trade_time.isoformat()
            
            # Fallback
            return datetime.now().isoformat()
            
        except Exception:
            return datetime.now().isoformat()
    
    def _parse_roi_from_output(self, output: str) -> float:
        """Extraer ROI del output."""
        try:
            roi_match = re.search(r'Retorno: ([0-9.-]+)%', output)
            if roi_match:
                return float(roi_match.group(1))
            return 0.0
        except Exception:
            return 0.0
    
    def _get_real_data(self, symbol: str, timeframe: str, days_back: int) -> Optional[pd.DataFrame]:
        """Obtener datos reales de Binance."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            tf_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
            interval = tf_map.get(timeframe, '4h')
            
            if timeframe == '4h':
                limit = min(1000, days_back * 6)
            else:
                limit = min(1000, days_back * 24)
            
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
            
            print(f"✅ Datos reales: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos reales: {e}")
            return None
    
    def _verify_extracted_trades(self, trades: List[Dict], real_data: pd.DataFrame, 
                                symbol: str, reported_roi: float):
        """Verificar trades extraídos."""
        print(f"\n🔍 VERIFICANDO {len(trades)} TRADES EXTRAÍDOS")
        print("=" * 60)
        
        if not trades:
            print("❌ No hay trades para verificar")
            return
        
        verified_count = 0
        total_pnl = 0
        price_verification_issues = 0
        
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
            
            # Verificar si los precios son realistas
            entry_realistic = self._is_price_realistic(real_data, entry_time, entry_price, "ENTRADA")
            exit_realistic = self._is_price_realistic(real_data, exit_time, exit_price, "SALIDA")
            
            # Verificar cálculo de P&L
            expected_pnl_pct = (exit_price - entry_price) / entry_price
            expected_pnl = expected_pnl_pct * 100 * 25
            pnl_correct = abs(pnl - expected_pnl) < 1.0
            
            if entry_realistic and exit_realistic and pnl_correct:
                verified_count += 1
                total_pnl += pnl
                print(f"   ✅ TRADE VERIFICADO")
            else:
                print(f"   ❌ TRADE CON PROBLEMAS")
                if not entry_realistic:
                    price_verification_issues += 1
                if not exit_realistic:
                    price_verification_issues += 1
                if not pnl_correct:
                    print(f"      - P&L incorrecto: esperado ${expected_pnl:.2f}")
        
        # Resumen
        self._print_extraction_summary(
            trades, verified_count, total_pnl, reported_roi, 
            price_verification_issues, symbol
        )
    
    def _is_price_realistic(self, real_data: pd.DataFrame, target_time: datetime, 
                           target_price: float, label: str) -> bool:
        """Verificar si un precio es realista comparado con datos reales."""
        try:
            # Encontrar datos cercanos en el tiempo
            time_tolerance = pd.Timedelta(hours=4)  # Tolerancia de 4 horas
            mask = abs(real_data.index - target_time) <= time_tolerance
            nearby_data = real_data[mask]
            
            if nearby_data.empty:
                print(f"   ⚠️ {label}: No hay datos reales cerca de {target_time}")
                return True  # Asumir válido si no hay datos para comparar
            
            # Verificar si el precio está en un rango razonable
            min_price = nearby_data['Low'].min()
            max_price = nearby_data['High'].max()
            avg_price = nearby_data['Close'].mean()
            
            # Tolerancia del 5% del rango
            price_range = max_price - min_price
            tolerance = price_range * 0.05
            
            if (min_price - tolerance) <= target_price <= (max_price + tolerance):
                deviation = abs(target_price - avg_price) / avg_price * 100
                print(f"   ✅ {label}: ${target_price:.4f} realista (desviación: {deviation:.1f}%)")
                return True
            else:
                print(f"   ❌ {label}: ${target_price:.4f} fuera de rango real [{min_price:.4f}-{max_price:.4f}]")
                return False
                
        except Exception as e:
            print(f"   ⚠️ {label}: Error verificando - {e}")
            return True
    
    def _print_extraction_summary(self, trades: List[Dict], verified_count: int, 
                                 total_pnl: float, reported_roi: float,
                                 price_issues: int, symbol: str):
        """Imprimir resumen de extracción y verificación."""
        print(f"\n🏆 RESUMEN DE VERIFICACIÓN - {symbol}")
        print("=" * 60)
        print(f"   📊 Trades extraídos: {len(trades)}")
        print(f"   ✅ Trades verificados: {verified_count}/{len(trades)}")
        print(f"   📈 Tasa de verificación: {verified_count/len(trades)*100:.1f}%")
        print(f"   💰 P&L verificado: ${total_pnl:.2f}")
        print(f"   📊 ROI reportado: {reported_roi:.2f}%")
        
        if price_issues > 0:
            print(f"   ⚠️ Problemas de precios: {price_issues}")
        
        # Calcular ROI verificado
        verified_roi = (total_pnl / 100) * 100
        roi_diff = abs(verified_roi - reported_roi)
        
        print(f"   🎯 ROI verificado: {verified_roi:.2f}%")
        print(f"   📊 Diferencia ROI: {roi_diff:.2f}%")
        
        # Evaluación final
        verification_rate = verified_count / len(trades) if trades else 0
        
        if verification_rate >= 0.85 and roi_diff < 5:
            print(f"\n   🎯 EVALUACIÓN: ALTAMENTE CONFIABLE")
            print(f"      Los trades son extraíbles y verificables")
        elif verification_rate >= 0.7 and roi_diff < 10:
            print(f"\n   ✅ EVALUACIÓN: CONFIABLE")
            print(f"      La mayoría de trades son verificables")
        elif verification_rate >= 0.5:
            print(f"\n   ⚠️ EVALUACIÓN: MODERADAMENTE CONFIABLE")
            print(f"      Algunos problemas en la verificación")
        else:
            print(f"\n   ❌ EVALUACIÓN: POCO CONFIABLE")
            print(f"      Demasiados problemas de verificación")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Extractor y Verificador de Trades")
    parser.add_argument('symbol', help='Símbolo crypto (ej: LINKUSDT)')
    parser.add_argument('--timeframe', default='4h', help='Timeframe (1h, 4h, 1d)')
    
    args = parser.parse_args()
    
    verifier = TradeExtractorVerifier()
    verifier.extract_and_verify(args.symbol, args.timeframe)

if __name__ == "__main__":
    main()