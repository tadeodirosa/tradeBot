#!/usr/bin/env python3

"""
Simulador de Futuros Profesional
===============================

Simulador específico para trading de futuros con apalancamiento real.
No usa la librería backtesting estándar para evitar limitaciones.

Características:
- Apalancamiento real (30x, 50x, 100x, etc.)
- Posiciones en USD con margen calculado correctamente
- Soporte para LONG y SHORT
- Liquidación automática
- P&L amplificado por apalancamiento
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import argparse

# Añadir el directorio analysis al path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Importar nuestro analizador profesional
try:
    from professional_analyzer import ProfessionalCryptoAnalyzer, ADVANCED_CONFIG, SignalStrength
except ImportError:
    print("❌ Error: No se puede importar el analizador profesional")
    print("   Asegúrate de que professional_analyzer.py está en el mismo directorio")
    sys.exit(1)

# Configuración del simulador de futuros BALANCEADA
FUTURES_CONFIG = {
    'initial_balance': 5000,      # Balance inicial
    'leverage': 30,               # Apalancamiento
    'position_size_usd': 100,     # Tamaño de posición en USD
    'commission_rate': 0.0006,    # 0.06% comisión por operación
    'funding_rate': 0.0001,       # 0.01% funding rate cada 8h (simplificado)
    
    # Parámetros de estrategia OPTIMIZADOS para más trades
    'min_buy_score': 55,          # Menos restrictivo para más señales
    'max_sell_score': 45,         # También shorts
    'min_confidence': 0.50,       # Menor barrera de entrada
    
    # Gestión de riesgo OPTIMIZADA para mejor retorno
    'max_positions': 3,           # Más posiciones = más oportunidades
    'liquidation_threshold': 0.85, # Evitar liquidaciones
    'stop_loss_atr_mult': 2.2,    # Stop loss más amplio para evitar liquidaciones
    'take_profit_atr_mult': 2.0,  # Take profit más cercano para capturar más ganancias
}

@dataclass
class FuturesPosition:
    """Posición de futuros."""
    id: str
    symbol: str
    side: str  # 'LONG' o 'SHORT'
    size_usd: float
    entry_price: float
    leverage: int
    margin_used: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = None
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calcular P&L no realizado."""
        if self.side == 'LONG':
            price_change = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            price_change = (self.entry_price - current_price) / self.entry_price
        
        # P&L = tamaño_posición * cambio_precio * apalancamiento
        pnl = self.size_usd * price_change
        self.unrealized_pnl = pnl
        return pnl
    
    def is_liquidated(self, current_price: float, liquidation_threshold: float) -> bool:
        """Verificar si la posición debe ser liquidada."""
        pnl = self.calculate_pnl(current_price)
        # Liquidación si las pérdidas superan el threshold del margen
        return pnl <= -(self.margin_used * liquidation_threshold)

@dataclass
class Trade:
    """Registro de trade cerrado."""
    id: str
    symbol: str
    side: str
    size_usd: float
    entry_price: float
    exit_price: float
    leverage: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    reason: str  # 'TP', 'SL', 'MANUAL', 'LIQUIDATION'

class FuturesSimulator:
    """Simulador de trading de futuros."""
    
    def __init__(self, initial_balance: float, leverage: int, position_size_usd: float):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.position_size_usd = position_size_usd
        self.margin_per_position = position_size_usd / leverage
        
        # Estado del simulador
        self.positions: Dict[str, FuturesPosition] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[float] = []
        self.timestamp_history: List[datetime] = []
        
        # Contadores
        self.position_counter = 0
        self.total_commission = 0.0
        
        print(f"🚀 SIMULADOR DE FUTUROS INICIALIZADO")
        print(f"   Balance inicial: ${initial_balance:,.2f}")
        print(f"   Apalancamiento: {leverage}x")
        print(f"   Tamaño de posición: ${position_size_usd} USD")
        print(f"   Margen por posición: ${self.margin_per_position:.2f} USD")
    
    def get_available_margin(self) -> float:
        """Obtener margen disponible."""
        used_margin = sum(pos.margin_used for pos in self.positions.values())
        return self.balance - used_margin
    
    def can_open_position(self) -> bool:
        """Verificar si se puede abrir una nueva posición."""
        available_margin = self.get_available_margin()
        max_positions = FUTURES_CONFIG['max_positions']
        
        return (len(self.positions) < max_positions and
                available_margin >= self.margin_per_position)
    
    def open_position(self, symbol: str, side: str, price: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     timestamp: datetime = None) -> Optional[str]:
        """Abrir nueva posición."""
        if not self.can_open_position():
            return None
        
        self.position_counter += 1
        position_id = f"POS_{self.position_counter}"
        
        # Calcular comisión
        commission = self.position_size_usd * FUTURES_CONFIG['commission_rate']
        self.total_commission += commission
        
        # Crear posición
        position = FuturesPosition(
            id=position_id,
            symbol=symbol,
            side=side,
            size_usd=self.position_size_usd,
            entry_price=price,
            leverage=self.leverage,
            margin_used=self.margin_per_position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=timestamp or datetime.now()
        )
        
        self.positions[position_id] = position
        
        print(f"📈 POSICIÓN ABIERTA: {side} ${self.position_size_usd} @ ${price:.2f} | Margen: ${self.margin_per_position:.2f}")
        
        return position_id
    
    def close_position(self, position_id: str, price: float, reason: str = 'MANUAL',
                      timestamp: datetime = None) -> Optional[Trade]:
        """Cerrar posición."""
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        exit_time = timestamp or datetime.now()
        
        # Calcular P&L final
        final_pnl = position.calculate_pnl(price)
        
        # Calcular comisión de cierre
        commission = self.position_size_usd * FUTURES_CONFIG['commission_rate']
        self.total_commission += commission
        
        # P&L neto después de comisiones
        net_pnl = final_pnl - (commission * 2)  # Comisión de apertura + cierre
        
        # Actualizar balance
        self.balance += net_pnl
        
        # Crear registro de trade
        trade = Trade(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            size_usd=position.size_usd,
            entry_price=position.entry_price,
            exit_price=price,
            leverage=position.leverage,
            entry_time=position.timestamp,
            exit_time=exit_time,
            pnl=net_pnl,
            commission=commission * 2,
            reason=reason
        )
        
        self.trades.append(trade)
        
        # Remover posición
        del self.positions[position_id]
        
        print(f"📉 POSICIÓN CERRADA: {position.side} | P&L: ${net_pnl:.2f} | Razón: {reason}")
        
        return trade
    
    def update_positions(self, price: float, timestamp: datetime = None):
        """Actualizar todas las posiciones con el precio actual."""
        timestamp = timestamp or datetime.now()
        
        # Verificar liquidaciones y stops
        positions_to_close = []
        
        for pos_id, position in self.positions.items():
            # Calcular P&L actual
            current_pnl = position.calculate_pnl(price)
            
            # Verificar liquidación
            if position.is_liquidated(price, FUTURES_CONFIG['liquidation_threshold']):
                positions_to_close.append((pos_id, 'LIQUIDATION'))
                continue
            
            # Verificar stop loss
            if position.stop_loss:
                if ((position.side == 'LONG' and price <= position.stop_loss) or
                    (position.side == 'SHORT' and price >= position.stop_loss)):
                    positions_to_close.append((pos_id, 'SL'))
                    continue
            
            # Verificar take profit
            if position.take_profit:
                if ((position.side == 'LONG' and price >= position.take_profit) or
                    (position.side == 'SHORT' and price <= position.take_profit)):
                    positions_to_close.append((pos_id, 'TP'))
                    continue
        
        # Cerrar posiciones que necesitan cerrarse
        for pos_id, reason in positions_to_close:
            self.close_position(pos_id, price, reason, timestamp)
        
        # Calcular equity total
        total_unrealized_pnl = sum(pos.calculate_pnl(price) for pos in self.positions.values())
        current_equity = self.balance + total_unrealized_pnl
        
        # Guardar historial
        self.equity_history.append(current_equity)
        self.timestamp_history.append(timestamp)
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del trading."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # Estadísticas básicas
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100
        total_pnl = sum(t.pnl for t in self.trades)
        total_return_pct = (total_pnl / self.initial_balance) * 100
        
        best_trade = max(t.pnl for t in self.trades)
        worst_trade = min(t.pnl for t in self.trades)
        avg_trade = total_pnl / total_trades
        
        # Drawdown máximo
        if self.equity_history:
            equity_series = pd.Series(self.equity_history)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # Sharpe ratio simplificado
        if self.trades:
            returns = [t.pnl / self.initial_balance for t in self.trades]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'total_commission': self.total_commission,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }

class FuturesBacktester:
    """Backtester para futuros con análisis técnico."""
    
    def __init__(self, symbol: str, timeframe: str = '4h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.analyzer = None
        
        # El analizador es opcional para este simulador
        # Solo usaremos análisis técnico simple integrado
        print(f"✅ Simulador de futuros para {symbol} en timeframe {timeframe}")
    
    def load_data(self, days_limit: int = None) -> Optional[pd.DataFrame]:
        """Cargar datos históricos desde cache_real."""
        # Buscar en cache_real
        symbol_clean = self.symbol.replace('_USDT', '').replace('USDT', '').upper()
        cache_filename = f"{symbol_clean}_USDT_{self.timeframe}.json"
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache_real', cache_filename)
        
        if not os.path.exists(cache_path):
            print(f"❌ No se encontró: {cache_path}")
            return None
        
        print(f"📁 Cargando desde: {cache_path}")
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Extraer datos del cache
            if 'data' in cache:
                data_list = cache['data']
            else:
                data_list = cache
            
            # Convertir a DataFrame
            df = pd.DataFrame(data_list)
            
            # Convertir timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Renombrar columnas si es necesario
            if 'open' in df.columns:
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
            
            # Limitar a los últimos N días si se especifica
            if days_limit:
                cutoff_date = df.index[-1] - timedelta(days=days_limit)
                df = df[df.index >= cutoff_date]
                print(f"📊 LIMITADO A ÚLTIMOS {days_limit} DÍAS")
            
            print(f"✅ Datos OHLCV cargados: {len(df)} barras desde {df.index[0]} hasta {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def analyze_bar(self, data: pd.DataFrame, bar_index: int) -> Optional[Dict]:
        """Analizar una barra específica."""
        if bar_index < 20:  # Necesitamos al menos 20 barras
            return None
        
        try:
            # Obtener datos hasta el punto actual
            current_data = data.iloc[:bar_index+1].copy()
            
            # Análisis técnico simplificado
            close = current_data['Close'].values
            high = current_data['High'].values
            low = current_data['Low'].values
            volume = current_data['Volume'].values
            
            # RSI
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            if len(gain) >= 14:
                avg_gain = np.mean(gain[-14:])
                avg_loss = np.mean(loss[-14:])
                rsi = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss > 0 else 1)))
            else:
                rsi = 50
            
            # EMAs
            ema_9 = current_data['Close'].ewm(span=9).mean().iloc[-1]
            ema_21 = current_data['Close'].ewm(span=21).mean().iloc[-1]
            
            # ATR
            if len(current_data) >= 14:
                high_low = current_data['High'] - current_data['Low']
                high_close = np.abs(current_data['High'] - current_data['Close'].shift(1))
                low_close = np.abs(current_data['Low'] - current_data['Close'].shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(14).mean().iloc[-1]
            else:
                atr = close[-1] * 0.02
            
            # Score compuesto OPTIMIZADO para más trades rentables
            score = 50  # Base neutral
            
            # Tendencia principal (peso importante)
            if close[-1] > ema_9: score += 12
            if close[-1] > ema_21: score += 10
            if ema_9 > ema_21: score += 8
            
            # RSI - aprovechar reversiones
            if rsi < 35: score += 12      # Oversold = oportunidad
            elif rsi < 45: score += 6     # Ligeramente oversold
            elif rsi > 65: score -= 12    # Overbought = evitar
            elif rsi > 55: score -= 6     # Ligeramente overbought
            
            # Momentum corto plazo
            if len(close) >= 3:
                momentum_3 = (close[-1] - close[-4]) / close[-4] * 100
                if momentum_3 > 2: score -= 8    # Subida muy rápida = cuidado
                elif momentum_3 > 0.5: score += 5   # Momentum positivo moderado
                elif momentum_3 < -2: score += 8    # Caída = oportunidad
            
            # Volumen - confirmación de señales
            if len(volume) >= 5:
                vol_avg = np.mean(volume[-5:])
                vol_current = volume[-1]
                if vol_current > vol_avg * 1.15: score += 6  # Volumen creciente
                elif vol_current < vol_avg * 0.8: score -= 3  # Volumen bajo
            
            # Posición relativa en el rango reciente
            if len(close) >= 14:
                recent_high = np.max(high[-14:])
                recent_low = np.min(low[-14:])
                position_in_range = (close[-1] - recent_low) / (recent_high - recent_low)
                
                if position_in_range < 0.3: score += 8      # Cerca del mínimo = compra
                elif position_in_range < 0.4: score += 4    # Zona baja
                elif position_in_range > 0.7: score -= 8    # Cerca del máximo = evitar
                elif position_in_range > 0.6: score -= 4    # Zona alta
            
            # Penalizar volatilidad extrema
            if len(close) >= 7:
                volatility = np.std(close[-7:]) / np.mean(close[-7:])
                if volatility > 0.06: score -= 8  # Muy volátil
            
            # Generar señal con lógica AGRESIVA para mejor retorno
            if score >= FUTURES_CONFIG['min_buy_score']:
                direction = 'BUY'
                # Confianza más agresiva para scores altos
                raw_confidence = (score - 50) / 50
                confidence = min(0.95, max(0.50, raw_confidence * 1.2))
            elif score <= FUTURES_CONFIG['max_sell_score']:
                direction = 'SELL'
                # Confianza más agresiva para scores bajos
                raw_confidence = (50 - score) / 50
                confidence = min(0.95, max(0.50, raw_confidence * 1.2))
            else:
                direction = 'HOLD'
                confidence = 0.25  # Muy baja confianza en HOLD
            
            # Calcular niveles
            current_price = close[-1]
            
            if direction == 'BUY':
                stop_loss = current_price - (atr * FUTURES_CONFIG['stop_loss_atr_mult'])
                take_profit = current_price + (atr * FUTURES_CONFIG['take_profit_atr_mult'])
            elif direction == 'SELL':
                stop_loss = current_price + (atr * FUTURES_CONFIG['stop_loss_atr_mult'])
                take_profit = current_price - (atr * FUTURES_CONFIG['take_profit_atr_mult'])
            else:
                stop_loss = None
                take_profit = None
            
            return {
                'direction': direction,
                'score': score,
                'confidence': confidence,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'rsi': rsi,
                'atr': atr
            }
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            return None
    
    def run_backtest(self, days_limit: int = None) -> Dict:
        """Ejecutar backtest completo."""
        print(f"\n🔬 BACKTEST DE FUTUROS: {self.symbol}")
        print("=" * 60)
        
        # Cargar datos
        df = self.load_data(days_limit)
        if df is None or len(df) < 50:
            print("❌ Datos insuficientes")
            return {}
        
        print(f"📊 Período: {df.index[0]} a {df.index[-1]} ({len(df)} barras)")
        
        # Crear simulador
        simulator = FuturesSimulator(
            initial_balance=FUTURES_CONFIG['initial_balance'],
            leverage=FUTURES_CONFIG['leverage'],
            position_size_usd=FUTURES_CONFIG['position_size_usd']
        )
        
        # Ejecutar backtest barra por barra
        signals_checked = 0
        signals_generated = 0
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Actualizar posiciones existentes
            simulator.update_positions(current_price, current_time)
            
            # Analizar nueva señal cada 4 barras (más frecuente)
            if i % 4 == 0 and i >= 20:  # Análisis más frecuente
                signals_checked += 1
                analysis = self.analyze_bar(df, i)
                
                if analysis and analysis['confidence'] >= FUTURES_CONFIG['min_confidence']:
                    signals_generated += 1
                    direction = analysis['direction']
                    
                    print(f"[{i:3d}] {direction} | Score: {analysis['score']:.0f} | Conf: {analysis['confidence']:.2f} | Price: ${current_price:.2f}")
                    
                    # Abrir posición si hay señal y espacio disponible
                    if direction == 'BUY' and simulator.can_open_position():
                        simulator.open_position(
                            symbol=self.symbol,
                            side='LONG',
                            price=current_price,
                            stop_loss=analysis['stop_loss'],
                            take_profit=analysis['take_profit'],
                            timestamp=current_time
                        )
                    elif direction == 'SELL' and simulator.can_open_position():
                        simulator.open_position(
                            symbol=self.symbol,
                            side='SHORT',
                            price=current_price,
                            stop_loss=analysis['stop_loss'],
                            take_profit=analysis['take_profit'],
                            timestamp=current_time
                        )
        
        print(f"\n📊 ESTADÍSTICAS DE ANÁLISIS:")
        print(f"   Barras analizadas: {signals_checked} de {len(df)} barras totales")
        print(f"   Señales generadas: {signals_generated}")
        print(f"   Ratio señales: {signals_generated/signals_checked*100:.1f}% de las barras analizadas")
        
        # Cerrar posiciones abiertas al final
        final_price = df['Close'].iloc[-1]
        for pos_id in list(simulator.positions.keys()):
            simulator.close_position(pos_id, final_price, 'END_OF_DATA')
        
        # Obtener estadísticas
        stats = simulator.get_stats()
        
        # Mostrar resultados
        self._display_results(stats, simulator)
        
        return stats
    
    def _display_results(self, stats: Dict, simulator: FuturesSimulator):
        """Mostrar resultados del backtest."""
        print(f"\n📈 RESULTADOS DEL BACKTEST DE FUTUROS:")
        print("=" * 60)
        
        # Configuración
        print(f"🔧 CONFIGURACIÓN:")
        print(f"   Balance inicial: ${FUTURES_CONFIG['initial_balance']:,.2f}")
        print(f"   Apalancamiento: {FUTURES_CONFIG['leverage']}x")
        print(f"   Tamaño posición: ${FUTURES_CONFIG['position_size_usd']} USD")
        print(f"   Margen por posición: ${FUTURES_CONFIG['position_size_usd']/FUTURES_CONFIG['leverage']:.2f} USD")
        
        # Rendimiento
        print(f"\n💰 RENDIMIENTO:")
        print(f"   P&L Total: ${stats['total_pnl']:,.2f}")
        print(f"   Retorno: {stats['total_return_pct']:.2f}%")
        print(f"   Balance final: ${simulator.balance:,.2f}")
        
        # Trading
        print(f"\n📊 ESTADÍSTICAS DE TRADING:")
        print(f"   Total trades: {stats['total_trades']}")
        print(f"   Trades ganadores: {stats['winning_trades']}")
        print(f"   Trades perdedores: {stats['losing_trades']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Mejor trade: ${stats['best_trade']:.2f}")
        print(f"   Peor trade: ${stats['worst_trade']:.2f}")
        print(f"   Trade promedio: ${stats['avg_trade']:.2f}")
        
        # Métricas de riesgo
        print(f"\n🛡️ MÉTRICAS DE RIESGO:")
        print(f"   Max Drawdown: {stats['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
        
        # Costos
        print(f"\n💸 COSTOS:")
        print(f"   Comisiones totales: ${stats['total_commission']:.2f}")
        
        # Calificación
        grade = self._grade_performance(stats)
        print(f"\n🏆 EVALUACIÓN:")
        print(f"   Calificación: {grade}")
        
        print("=" * 60)
    
    def _grade_performance(self, stats: Dict) -> str:
        """Calificar el rendimiento."""
        score = 0
        
        # Retorno
        return_pct = stats['total_return_pct']
        if return_pct > 50: score += 3
        elif return_pct > 20: score += 2
        elif return_pct > 0: score += 1
        
        # Win Rate
        win_rate = stats['win_rate']
        if win_rate > 70: score += 3
        elif win_rate > 50: score += 2
        elif win_rate > 40: score += 1
        
        # Sharpe
        sharpe = stats['sharpe_ratio']
        if sharpe > 2: score += 3
        elif sharpe > 1: score += 2
        elif sharpe > 0.5: score += 1
        
        # Trades
        if stats['total_trades'] >= 10: score += 1
        
        if score >= 9: return "A+ (Excepcional)"
        elif score >= 7: return "A (Excelente)"
        elif score >= 5: return "B (Bueno)"
        elif score >= 3: return "C (Regular)"
        else: return "D (Malo)"

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Simulador de Futuros Profesional")
    parser.add_argument('symbol', help='Símbolo a analizar (ej: BTC_USDT_1h)')
    parser.add_argument('--balance', type=float, default=5000, help='Balance inicial')
    parser.add_argument('--leverage', type=int, default=30, help='Apalancamiento')
    parser.add_argument('--position_size', type=float, default=100, help='Tamaño de posición en USD')
    parser.add_argument('--days', type=int, default=None, help='Limitar a últimos N días')
    
    args = parser.parse_args()
    
    # Extraer símbolo y timeframe
    if '_' in args.symbol:
        parts = args.symbol.split('_')
        if len(parts) >= 3:
            symbol = '_'.join(parts[:-1])
            timeframe = parts[-1]
        else:
            symbol = args.symbol
            timeframe = '4h'
    else:
        symbol = args.symbol
        timeframe = '4h'
    
    # Actualizar configuración
    FUTURES_CONFIG['initial_balance'] = args.balance
    FUTURES_CONFIG['leverage'] = args.leverage
    FUTURES_CONFIG['position_size_usd'] = args.position_size
    
    try:
        # Crear backtester
        backtester = FuturesBacktester(symbol, timeframe)
        
        # Ejecutar backtest
        results = backtester.run_backtest(args.days)
        
        if results:
            print(f"\n✅ Backtest de futuros completado para {symbol}")
        else:
            print("❌ No se pudo completar el backtest")
            
    except KeyboardInterrupt:
        print("\n🛑 Backtest interrumpido")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()