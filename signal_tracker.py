"""
Sistema de Tracking de Señales - Guardar y Analizar Performance Real
Permite validar el modelo en tiempo real y mejorarlo continuamente
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List
import requests

class SignalTracker:
    def __init__(self, signals_file: str = "signal_history.json"):
        self.signals_file = signals_file
        self.signals_data = self._load_signals()
    
    def _load_signals(self) -> List[Dict]:
        """Cargar historial de señales desde archivo."""
        if os.path.exists(self.signals_file):
            try:
                with open(self.signals_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error cargando señales: {e}")
                return []
        return []
    
    def _save_signals(self):
        """Guardar señales en archivo."""
        try:
            with open(self.signals_file, 'w', encoding='utf-8') as f:
                json.dump(self.signals_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Señales guardadas en {self.signals_file}")
        except Exception as e:
            print(f"❌ Error guardando señales: {e}")
    
    def save_signal(self, signal_result: Dict):
        """Guardar nueva señal en el tracking."""
        if not signal_result.get('signal'):
            print("⏳ No hay señal para guardar")
            return
        
        # Crear registro de señal
        signal_record = {
            'id': f"{signal_result['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': signal_result['symbol'],
            'timestamp': signal_result['timestamp'].isoformat(),
            'direction': signal_result['signal'],
            'strength': signal_result['strength'],
            'entry_price': signal_result['price'],
            'stop_loss': signal_result.get('stop_loss'),
            'take_profit': signal_result.get('take_profit'),
            'position_size': signal_result.get('position_size'),
            'risk_amount': signal_result.get('risk_amount'),
            'indicators': signal_result['indicators'],
            'conditions_met': {
                'long': signal_result['long_conditions'],
                'short': signal_result['short_conditions']
            },
            'reasons': [r for r in signal_result['reasons'] if '✅' in r],
            
            # Estado de seguimiento
            'status': 'ACTIVE',  # ACTIVE, HIT_TP, HIT_SL, EXPIRED
            'current_price': None,
            'max_favorable': None,  # Máximo precio favorable
            'max_adverse': None,    # Máximo precio adverso
            'duration_hours': None,
            'pnl_percentage': None,
            'last_check': None,
            'notes': []
        }
        
        # Añadir al historial
        self.signals_data.append(signal_record)
        self._save_signals()
        
        print(f"✅ Señal guardada: {signal_record['id']}")
        print(f"📊 Total señales en tracking: {len(self.signals_data)}")
        
        return signal_record['id']
    
    def get_current_price(self, symbol: str) -> float:
        """Obtener precio actual de Binance."""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except Exception as e:
            print(f"⚠️ Error obteniendo precio {symbol}: {e}")
        return None
    
    def update_active_signals(self):
        """Actualizar estado de todas las señales activas."""
        active_signals = [s for s in self.signals_data if s['status'] == 'ACTIVE']
        
        if not active_signals:
            print("📭 No hay señales activas para actualizar")
            return
        
        print(f"🔄 Actualizando {len(active_signals)} señales activas...")
        
        updated_count = 0
        for signal in active_signals:
            if self._update_signal_status(signal):
                updated_count += 1
        
        if updated_count > 0:
            self._save_signals()
            print(f"✅ {updated_count} señales actualizadas")
    
    def _update_signal_status(self, signal: Dict) -> bool:
        """Actualizar estado de una señal específica."""
        try:
            # Obtener precio actual
            current_price = self.get_current_price(signal['symbol'])
            if current_price is None:
                return False
            
            signal['current_price'] = current_price
            signal['last_check'] = datetime.now().isoformat()
            
            # Calcular duración
            entry_time = datetime.fromisoformat(signal['timestamp'])
            duration = datetime.now() - entry_time
            signal['duration_hours'] = round(duration.total_seconds() / 3600, 1)
            
            # Actualizar máximos favorable/adverso
            entry_price = signal['entry_price']
            direction = signal['direction']
            
            if direction == 'LONG':
                # Para LONG: favorable es al alza, adverso a la baja
                if signal['max_favorable'] is None or current_price > signal['max_favorable']:
                    signal['max_favorable'] = current_price
                if signal['max_adverse'] is None or current_price < signal['max_adverse']:
                    signal['max_adverse'] = current_price
                
                # Calcular PnL
                signal['pnl_percentage'] = ((current_price - entry_price) / entry_price) * 100
                
                # Verificar niveles
                if current_price <= signal['stop_loss']:
                    signal['status'] = 'HIT_SL'
                    signal['notes'].append(f"Stop Loss alcanzado a ${current_price:.4f}")
                elif current_price >= signal['take_profit']:
                    signal['status'] = 'HIT_TP'
                    signal['notes'].append(f"Take Profit alcanzado a ${current_price:.4f}")
                    
            else:  # SHORT
                # Para SHORT: favorable es a la baja, adverso al alza
                if signal['max_favorable'] is None or current_price < signal['max_favorable']:
                    signal['max_favorable'] = current_price
                if signal['max_adverse'] is None or current_price > signal['max_adverse']:
                    signal['max_adverse'] = current_price
                
                # Calcular PnL
                signal['pnl_percentage'] = ((entry_price - current_price) / entry_price) * 100
                
                # Verificar niveles
                if current_price >= signal['stop_loss']:
                    signal['status'] = 'HIT_SL'
                    signal['notes'].append(f"Stop Loss alcanzado a ${current_price:.4f}")
                elif current_price <= signal['take_profit']:
                    signal['status'] = 'HIT_TP'
                    signal['notes'].append(f"Take Profit alcanzado a ${current_price:.4f}")
            
            # Verificar expiración (72 horas por defecto)
            if signal['duration_hours'] > 72 and signal['status'] == 'ACTIVE':
                signal['status'] = 'EXPIRED'
                signal['notes'].append(f"Señal expirada después de {signal['duration_hours']} horas")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error actualizando señal {signal['id']}: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """Calcular estadísticas de performance del modelo."""
        if not self.signals_data:
            return {"error": "No hay señales en el historial"}
        
        # Separar por estado
        completed_signals = [s for s in self.signals_data if s['status'] in ['HIT_TP', 'HIT_SL', 'EXPIRED']]
        active_signals = [s for s in self.signals_data if s['status'] == 'ACTIVE']
        
        if not completed_signals:
            return {
                "total_signals": len(self.signals_data),
                "active_signals": len(active_signals),
                "completed_signals": 0,
                "message": "No hay señales completadas para análisis"
            }
        
        # Calcular estadísticas
        winners = [s for s in completed_signals if s.get('pnl_percentage', 0) > 0]
        losers = [s for s in completed_signals if s.get('pnl_percentage', 0) < 0]
        
        total_pnl = sum(s.get('pnl_percentage', 0) for s in completed_signals)
        avg_winner = sum(s.get('pnl_percentage', 0) for s in winners) / len(winners) if winners else 0
        avg_loser = sum(s.get('pnl_percentage', 0) for s in losers) / len(losers) if losers else 0
        
        win_rate = (len(winners) / len(completed_signals)) * 100 if completed_signals else 0
        
        # Estadísticas por dirección
        long_signals = [s for s in completed_signals if s['direction'] == 'LONG']
        short_signals = [s for s in completed_signals if s['direction'] == 'SHORT']
        
        stats = {
            "total_signals": len(self.signals_data),
            "active_signals": len(active_signals),
            "completed_signals": len(completed_signals),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_winner": round(avg_winner, 2),
            "avg_loser": round(avg_loser, 2),
            "profit_factor": round(abs(avg_winner / avg_loser), 2) if avg_loser != 0 else float('inf'),
            "winners": len(winners),
            "losers": len(losers),
            "long_signals": len(long_signals),
            "short_signals": len(short_signals),
            "avg_duration": round(sum(s.get('duration_hours', 0) for s in completed_signals) / len(completed_signals), 1) if completed_signals else 0
        }
        
        return stats
    
    def print_performance_report(self):
        """Imprimir reporte de performance."""
        stats = self.get_performance_stats()
        
        print("\n" + "="*60)
        print("📊 REPORTE DE PERFORMANCE DEL MODELO")
        print("="*60)
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
        
        print(f"📈 Total de señales: {stats['total_signals']}")
        print(f"🔄 Señales activas: {stats['active_signals']}")
        print(f"✅ Señales completadas: {stats['completed_signals']}")
        
        if stats['completed_signals'] > 0:
            print(f"\n🎯 PERFORMANCE:")
            print(f"   Win Rate: {stats['win_rate']}%")
            print(f"   PnL Total: {stats['total_pnl']:.2f}%")
            print(f"   Promedio ganadora: {stats['avg_winner']:.2f}%")
            print(f"   Promedio perdedora: {stats['avg_loser']:.2f}%")
            print(f"   Profit Factor: {stats['profit_factor']:.2f}")
            print(f"   Duración promedio: {stats['avg_duration']} horas")
            
            print(f"\n📊 DISTRIBUCIÓN:")
            print(f"   Ganadoras: {stats['winners']}")
            print(f"   Perdedoras: {stats['losers']}")
            print(f"   Señales LONG: {stats['long_signals']}")
            print(f"   Señales SHORT: {stats['short_signals']}")
        
        print("="*60)
    
    def print_active_signals(self):
        """Mostrar señales activas."""
        active_signals = [s for s in self.signals_data if s['status'] == 'ACTIVE']
        
        if not active_signals:
            print("📭 No hay señales activas")
            return
        
        print(f"\n🔄 SEÑALES ACTIVAS ({len(active_signals)}):")
        print("-" * 80)
        
        for signal in active_signals:
            print(f"🎯 {signal['symbol']} {signal['direction']}")
            print(f"   Entrada: ${signal['entry_price']:.4f}")
            if signal.get('current_price'):
                print(f"   Actual: ${signal['current_price']:.4f}")
                print(f"   PnL: {signal.get('pnl_percentage', 0):.2f}%")
            print(f"   Duración: {signal.get('duration_hours', 0)} horas")
            print(f"   SL: ${signal['stop_loss']:.4f} | TP: ${signal['take_profit']:.4f}")
            print("-" * 40)


def main():
    """Función principal para testing."""
    tracker = SignalTracker()
    
    print("🎯 SISTEMA DE TRACKING DE SEÑALES")
    print("1. Actualizar señales activas")
    print("2. Ver performance")
    print("3. Ver señales activas")
    
    choice = input("\nElige opción (1-3): ").strip()
    
    if choice == "1":
        tracker.update_active_signals()
    elif choice == "2":
        tracker.print_performance_report()
    elif choice == "3":
        tracker.print_active_signals()
    else:
        print("Opción inválida")


if __name__ == "__main__":
    main()