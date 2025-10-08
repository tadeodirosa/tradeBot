"""
Multi-Timeframe Backtester V2 - Validación de Estrategia Optimizada
===================================================================

OBJETIVO: Validar históricamente las mejoras del V2 Analyzer
- Usar la MISMA lógica estricta del multi_timeframe_analyzer_v2.py
- Filtros 3/3 (4H) + 4/4 (1H) + control volatilidad + gap temporal
- Comparar con V1: esperar ~30-50 trades vs 593, Win Rate >50% vs 33.9%

CAMBIOS CRÍTICOS VS V1:
✅ Filtros 4H: 3/3 condiciones REQUERIDAS (vs 2/3)
✅ Filtros 1H: 4/4 confluencias REQUERIDAS (vs 2/3)  
✅ Control volatilidad: 1.5-8.0% ATR 4H
✅ Gap temporal: mínimo 4h entre señales
✅ Quality scoring: tendencia + confluencias
"""

import ccxt
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

class MultiTimeframeBacktesterV2:
    def __init__(self, symbol: str = "LINK/USDT"):
        self.symbol = symbol
        self.exchange = ccxt.binance()
        
        # Configuración V2 OPTIMIZADA (misma que analyzer V2)
        self.config = {
            'timeframe_trend': '4h',
            'timeframe_entry': '1h', 
            'initial_balance': 10000,
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'atr_period': 14,
            'stop_loss_atr_multiplier': 1.0,      # Más conservador
            'take_profit_atr_multiplier': 2.5,    # Mejor R:R
            'risk_per_trade': 0.02,               # Más conservador
            'max_position_size': 0.20,            # Más conservador
            'commission': 0.001,
            'min_signal_gap_hours': 4,            # NUEVO: Evitar over-trading
            'min_volatility_4h': 1.5,             # NUEVO: ATR mínimo 4H
            'max_volatility_4h': 8.0              # NUEVO: ATR máximo 4H
        }
        
        # Métricas de trading
        self.balance = self.config['initial_balance']
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.last_signal_time = None
        
    def fetch_historical_data(self, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtener datos históricos para backtesting."""
        print(f"📡 Descargando datos {timeframe}: {self.symbol} ({start_date} → {end_date})")
        
        try:
            # Convertir fechas
            since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
            until = self.exchange.parse8601(f"{end_date}T23:59:59Z")
            
            # Obtener datos con límite de rate
            all_data = []
            current_since = since
            
            while current_since < until:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        self.symbol, 
                        timeframe, 
                        since=current_since, 
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    current_since = ohlcv[-1][0] + (3600000 if timeframe == '1h' else 14400000)
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    print(f"⚠️ Error temporal descargando: {e}")
                    time.sleep(1)
                    continue
            
            # Convertir a DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filtrar por fechas exactas
            start_filter = pd.to_datetime(start_date)
            end_filter = pd.to_datetime(end_date)
            df = df[(df.index >= start_filter) & (df.index <= end_filter)]
            
            print(f"✅ Datos {timeframe} obtenidos: {len(df)} velas")
            print(f"📅 Rango: {df.index[0]} → {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos {timeframe}: {e}")
            raise
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos (MISMA lógica que V2)."""
        
        # EMA
        df['ema_9'] = df['close'].ewm(span=self.config['ema_fast']).mean()
        df['ema_21'] = df['close'].ewm(span=self.config['ema_slow']).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.config['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.config['rsi_period']).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.config['atr_period']).mean()
        df['atr_percentage'] = (df['atr'] / df['close']) * 100
        
        # Momentum (últimas 5 velas como V2)
        df['momentum'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        
        # EMA trend strength
        df['ema_trend_strength'] = ((df['ema_9'] - df['ema_21']) / df['ema_21']) * 100
        
        return df
    
    def analyze_trend_signal_v2(self, df_4h: pd.DataFrame, timestamp: pd.Timestamp) -> dict:
        """Analizar señal de tendencia en 4H con FILTROS V2 ESTRICTOS."""
        
        # Buscar la vela 4H correspondiente
        available_times = df_4h.index[df_4h.index <= timestamp]
        if len(available_times) == 0:
            return {'direction': None, 'strength': 0, 'quality_score': 0}
        
        idx = available_times[-1]
        row = df_4h.loc[idx]
        
        # Verificar que tenemos suficientes datos
        if pd.isna(row['ema_21']) or pd.isna(row['rsi']) or pd.isna(row['atr']):
            return {'direction': None, 'strength': 0, 'quality_score': 0}
        
        atr_pct = row['atr_percentage']
        momentum = row['momentum']
        ema_trend = row['ema_trend_strength']
        rsi = row['rsi']
        
        # FILTRO GLOBAL V2: Volatilidad adecuada
        if not (self.config['min_volatility_4h'] <= atr_pct <= self.config['max_volatility_4h']):
            return {'direction': None, 'strength': 0, 'quality_score': 0}
        
        trend_analysis = {'direction': None, 'strength': 0, 'quality_score': 0, 'conditions_met': 0}
        
        # Condiciones BULLISH V2 (3/3 REQUERIDAS)
        bullish_conditions = 0
        if row['ema_9'] > row['ema_21'] and ema_trend > 0.5:
            bullish_conditions += 1
        if 35 <= rsi <= 65:
            bullish_conditions += 1
        if momentum > 1.0:
            bullish_conditions += 1
        
        # Condiciones BEARISH V2 (3/3 REQUERIDAS)
        bearish_conditions = 0
        if row['ema_9'] < row['ema_21'] and ema_trend < -0.5:
            bearish_conditions += 1
        if 35 <= rsi <= 65:
            bearish_conditions += 1
        if momentum < -1.0:
            bearish_conditions += 1
        
        # Determinar tendencia V2 (REQUIERE 3/3)
        if bullish_conditions >= 3:
            trend_analysis['direction'] = 'BULLISH'
            trend_analysis['strength'] = bullish_conditions * 33
            trend_analysis['conditions_met'] = bullish_conditions
            trend_analysis['quality_score'] = min(100, bullish_conditions * 25 + abs(ema_trend) * 10)
        elif bearish_conditions >= 3:
            trend_analysis['direction'] = 'BEARISH'
            trend_analysis['strength'] = bearish_conditions * 33
            trend_analysis['conditions_met'] = bearish_conditions
            trend_analysis['quality_score'] = min(100, bearish_conditions * 25 + abs(ema_trend) * 10)
        
        return trend_analysis
    
    def analyze_entry_signal_v2(self, df_1h: pd.DataFrame, timestamp: pd.Timestamp, 
                                trend_direction: str, trend_quality: int) -> dict:
        """Analizar señal de entrada en 1H con CONFLUENCIAS V2."""
        
        if not trend_direction:
            return {'signal': None, 'strength': 0, 'confluence_score': 0}
        
        # Verificar gap temporal V2
        if self.last_signal_time:
            time_diff = (timestamp - self.last_signal_time).total_seconds() / 3600
            if time_diff < self.config['min_signal_gap_hours']:
                return {'signal': None, 'strength': 0, 'confluence_score': 0}
        
        # Buscar la vela 1H correspondiente
        if timestamp not in df_1h.index:
            return {'signal': None, 'strength': 0, 'confluence_score': 0}
        
        row = df_1h.loc[timestamp]
        
        # Verificar que tenemos datos suficientes
        if pd.isna(row['rsi']) or pd.isna(row['momentum']) or pd.isna(row['atr_percentage']):
            return {'signal': None, 'strength': 0, 'confluence_score': 0}
        
        rsi = row['rsi']
        momentum = row['momentum']
        atr_pct = row['atr_percentage']
        ema_trend = row['ema_trend_strength']
        
        entry_analysis = {'signal': None, 'strength': 0, 'confluence_score': 0, 'conditions_met': 0}
        
        if trend_direction == 'BULLISH':
            # Condiciones LONG V2 (4/4 REQUERIDAS)
            long_conditions = 0
            
            # 1. RSI zona entrada óptima
            if 25 <= rsi <= 45:
                long_conditions += 1
            
            # 2. Momentum controlado
            if -2.0 <= momentum <= 3.0:
                long_conditions += 1
            
            # 3. EMA 1H alineado
            if ema_trend > -1.0:
                long_conditions += 1
            
            # 4. Volatilidad adecuada
            if 0.8 <= atr_pct <= 6.0:
                long_conditions += 1
            
            # Generar señal LONG V2 (REQUIERE 4/4 + calidad alta)
            if long_conditions >= 4 and trend_quality >= 75:
                entry_analysis['signal'] = 'LONG'
                entry_analysis['strength'] = long_conditions * 25
                entry_analysis['conditions_met'] = long_conditions
                entry_analysis['confluence_score'] = long_conditions * 20
                self.last_signal_time = timestamp
        
        elif trend_direction == 'BEARISH':
            # Condiciones SHORT V2 (4/4 REQUERIDAS)
            short_conditions = 0
            
            # 1. RSI zona entrada óptima
            if 55 <= rsi <= 75:
                short_conditions += 1
            
            # 2. Momentum controlado
            if -3.0 <= momentum <= 2.0:
                short_conditions += 1
            
            # 3. EMA 1H alineado
            if ema_trend < 1.0:
                short_conditions += 1
            
            # 4. Volatilidad adecuada
            if 0.8 <= atr_pct <= 6.0:
                short_conditions += 1
            
            # Generar señal SHORT V2 (REQUIERE 4/4 + calidad alta)
            if short_conditions >= 4 and trend_quality >= 75:
                entry_analysis['signal'] = 'SHORT'
                entry_analysis['strength'] = short_conditions * 25
                entry_analysis['conditions_met'] = short_conditions
                entry_analysis['confluence_score'] = short_conditions * 20
                self.last_signal_time = timestamp
        
        return entry_analysis
    
    def calculate_position_size(self, price: float, atr: float) -> float:
        """Calcular tamaño de posición V2 (más conservador)."""
        
        risk_amount = self.balance * self.config['risk_per_trade']  # 2% vs 3%
        stop_distance = atr * self.config['stop_loss_atr_multiplier']
        
        position_size = risk_amount / stop_distance
        max_position_value = self.balance * self.config['max_position_size']  # 20% vs 25%
        max_position_size = max_position_value / price
        
        return min(position_size, max_position_size)
    
    def run_backtest_v2(self, start_date: str, end_date: str) -> dict:
        """Ejecutar backtest V2 con filtros optimizados."""
        
        print(f"🚀 INICIANDO BACKTEST MULTI-TIMEFRAME V2 (OPTIMIZADO)")
        print(f"📈 Símbolo: {self.symbol}")
        print(f"📅 Período: {start_date} → {end_date}")
        print(f"💰 Balance inicial: ${self.balance:,.2f}")
        print(f"⚖️ Estrategia: V2 - Filtros estrictos 3/3 + 4/4")
        print(f"🔧 Gap temporal: {self.config['min_signal_gap_hours']}h")
        print(f"📊 Volatilidad: {self.config['min_volatility_4h']}-{self.config['max_volatility_4h']}% ATR")
        
        # 1. Obtener datos históricos de ambos timeframes
        df_4h = self.fetch_historical_data('4h', start_date, end_date)
        df_1h = self.fetch_historical_data('1h', start_date, end_date)
        
        # 2. Calcular indicadores
        df_4h = self.calculate_indicators(df_4h)
        df_1h = self.calculate_indicators(df_1h)
        
        print(f"\\n🔧 Procesando señales V2...")
        
        # 3. Iterar por cada vela 1H buscando señales V2
        signals_checked = 0
        signals_generated = 0
        trend_confirmations = 0
        
        # Asegurar que tenemos suficientes datos
        min_period = max(self.config['atr_period'], self.config['ema_slow'])
        if len(df_1h) <= min_period:
            print(f"❌ Insuficientes datos 1H: {len(df_1h)} velas (necesarias: {min_period})")
            return {'error': 'Datos insuficientes'}
        
        for timestamp in df_1h.index[min_period:]:
            
            # Analizar tendencia 4H con filtros V2
            trend_analysis = self.analyze_trend_signal_v2(df_4h, timestamp)
            
            signals_checked += 1
            
            if not trend_analysis['direction']:
                continue
            
            trend_confirmations += 1
            
            # Analizar entrada 1H según tendencia con confluencias V2
            entry_analysis = self.analyze_entry_signal_v2(
                df_1h, timestamp, 
                trend_analysis['direction'], 
                trend_analysis['quality_score']
            )
            
            if not entry_analysis['signal']:
                continue
            
            signals_generated += 1
            current_price = df_1h.loc[timestamp, 'close']
            
            # Buscar ATR 4H correspondiente
            available_4h = df_4h.index[df_4h.index <= timestamp]
            if len(available_4h) == 0:
                continue
            atr_4h = df_4h.loc[available_4h[-1], 'atr']
            
            # Calcular posición V2
            position_size = self.calculate_position_size(current_price, atr_4h)
            
            if position_size * current_price < 50:  # Posición mínima
                continue
            
            # Calcular niveles V2
            stop_distance = atr_4h * self.config['stop_loss_atr_multiplier']
            
            if entry_analysis['signal'] == 'LONG':
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * self.config['take_profit_atr_multiplier'])
            else:  # SHORT
                stop_loss = current_price + stop_distance
                take_profit = current_price - (stop_distance * self.config['take_profit_atr_multiplier'])
            
            # Crear trade V2
            trade = {
                'id': len(self.trades) + 1,
                'timestamp': timestamp,
                'direction': entry_analysis['signal'],
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'trend_strength_4h': trend_analysis['strength'],
                'trend_quality_4h': trend_analysis['quality_score'],
                'entry_strength_1h': entry_analysis['strength'],
                'confluence_score_1h': entry_analysis['confluence_score'],
                'atr_4h': atr_4h,
                'status': 'OPEN',
                'strategy_version': 'V2_OPTIMIZED'
            }
            
            # Simular ejecución del trade
            self.simulate_trade_execution(trade, df_1h, timestamp)
        
        print(f"\\n📊 Señales procesadas: {signals_checked:,}")
        print(f"🎯 Tendencias confirmadas 4H: {trend_confirmations}")
        print(f"🚨 Señales finales generadas: {signals_generated}")
        print(f"📉 Ratio selectividad: {signals_generated/signals_checked*100:.2f}%")
        
        # 4. Calcular métricas finales
        return self.calculate_performance_metrics()
    
    def simulate_trade_execution(self, trade: dict, df_1h: pd.DataFrame, entry_time: pd.Timestamp):
        """Simular la ejecución de un trade hasta su cierre."""
        
        future_data = df_1h[df_1h.index > entry_time]
        if len(future_data) == 0:
            trade['exit_price'] = trade['entry_price']
            trade['exit_timestamp'] = entry_time
            trade['status'] = 'NO_DATA'
            self.calculate_trade_pnl(trade)
            return
        
        # Limitar a máximo 1 semana (168 horas)
        future_data = future_data.head(168)
        
        for timestamp in future_data.index:
            row = future_data.loc[timestamp]
            
            # Verificar stop loss
            if trade['direction'] == 'LONG':
                if row['low'] <= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_timestamp'] = timestamp
                    trade['status'] = 'STOP_LOSS'
                    break
                elif row['high'] >= trade['take_profit']:
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_timestamp'] = timestamp
                    trade['status'] = 'TAKE_PROFIT'
                    break
            
            else:  # SHORT
                if row['high'] >= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_timestamp'] = timestamp
                    trade['status'] = 'STOP_LOSS'
                    break
                elif row['low'] <= trade['take_profit']:
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_timestamp'] = timestamp
                    trade['status'] = 'TAKE_PROFIT'
                    break
        
        # Si no cerró, cerrar al final del período
        if trade['status'] == 'OPEN':
            if len(future_data) > 0:
                trade['exit_price'] = future_data.iloc[-1]['close']
                trade['exit_timestamp'] = future_data.index[-1]
                trade['status'] = 'TIME_EXIT'
            else:
                trade['exit_price'] = trade['entry_price']
                trade['exit_timestamp'] = entry_time
                trade['status'] = 'NO_DATA'
        
        # Calcular P&L
        self.calculate_trade_pnl(trade)
        self.trades.append(trade)
    
    def calculate_trade_pnl(self, trade: dict):
        """Calcular P&L de un trade."""
        
        position_value = trade['position_size'] * trade['entry_price']
        commission_cost = position_value * self.config['commission'] * 2
        
        if trade['direction'] == 'LONG':
            gross_pnl = trade['position_size'] * (trade['exit_price'] - trade['entry_price'])
        else:  # SHORT
            gross_pnl = trade['position_size'] * (trade['entry_price'] - trade['exit_price'])
        
        net_pnl = gross_pnl - commission_cost
        pnl_percentage = (net_pnl / position_value) * 100
        
        trade['gross_pnl'] = gross_pnl
        trade['commission'] = commission_cost
        trade['net_pnl'] = net_pnl
        trade['pnl_percentage'] = pnl_percentage
        trade['position_value'] = position_value
        
        # Actualizar balance
        self.balance += net_pnl
        
        # Agregar a equity curve
        self.equity_curve.append({
            'timestamp': trade['exit_timestamp'],
            'balance': self.balance,
            'trade_id': trade['id']
        })
    
    def calculate_performance_metrics(self) -> dict:
        """Calcular métricas de performance V2."""
        
        if not self.trades:
            return {'error': 'No hay trades para analizar'}
        
        df_trades = pd.DataFrame(self.trades)
        
        # Métricas básicas
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
        losing_trades = len(df_trades[df_trades['net_pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100
        
        # P&L
        total_pnl = df_trades['net_pnl'].sum()
        gross_profit = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # ROI
        initial_balance = self.config['initial_balance']
        final_balance = self.balance
        roi = ((final_balance - initial_balance) / initial_balance) * 100
        
        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df['cummax'] = equity_df['balance'].cummax()
            equity_df['drawdown'] = ((equity_df['balance'] - equity_df['cummax']) / equity_df['cummax']) * 100
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        # Métricas V2 específicas
        avg_trend_quality = df_trades['trend_quality_4h'].mean()
        avg_confluence_score = df_trades['confluence_score_1h'].mean()
        
        # Tiempo promedio
        df_trades['duration'] = (df_trades['exit_timestamp'] - df_trades['timestamp']).dt.total_seconds() / 3600
        avg_duration = df_trades['duration'].mean()
        
        results = {
            'symbol': self.symbol,
            'period': f"{self.trades[0]['timestamp'].date()} to {self.trades[-1]['timestamp'].date()}",
            'strategy': 'Multi-Timeframe V2 (Optimized)',
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_pnl': total_pnl,
            'roi_percentage': roi,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': max_drawdown,
            'avg_duration_hours': avg_duration,
            'avg_trend_quality_4h': avg_trend_quality,
            'avg_confluence_score_1h': avg_confluence_score,
            'commissions_paid': df_trades['commission'].sum(),
            
            # Métricas adicionales
            'long_trades': len(df_trades[df_trades['direction'] == 'LONG']),
            'short_trades': len(df_trades[df_trades['direction'] == 'SHORT']),
            'avg_win': df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0,
            'largest_win': df_trades['net_pnl'].max(),
            'largest_loss': df_trades['net_pnl'].min(),
        }
        
        return results
    
    def print_backtest_results_v2(self, results: dict):
        """Imprimir resultados del backtest V2."""
        
        print(f"\\n" + "="*90)
        print(f"🎯 RESULTADOS BACKTEST MULTI-TIMEFRAME V2 (OPTIMIZADO)")
        print(f"="*90)
        print(f"📈 Símbolo: {results['symbol']}")
        print(f"📅 Período: {results['period']}")
        print(f"⚖️ Estrategia: {results['strategy']}")
        
        print(f"\\n💰 RENDIMIENTO FINANCIERO V2:")
        print(f"   Balance inicial: ${results['initial_balance']:,.2f}")
        print(f"   Balance final: ${results['final_balance']:,.2f}")
        print(f"   P&L total: ${results['total_pnl']:,.2f}")
        print(f"   ROI: {results['roi_percentage']:.2f}%")
        print(f"   Drawdown máximo: {results['max_drawdown']:.2f}%")
        
        print(f"\\n📊 ESTADÍSTICAS DE TRADING V2:")
        print(f"   Total trades: {results['total_trades']}")
        print(f"   Trades ganadores: {results['winning_trades']}")
        print(f"   Trades perdedores: {results['losing_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\\n🔬 MÉTRICAS DE CALIDAD V2:")
        print(f"   Calidad promedio tendencia 4H: {results['avg_trend_quality_4h']:.1f}/100")
        print(f"   Score promedio confluencia 1H: {results['avg_confluence_score_1h']:.1f}/100")
        print(f"   Duración promedio: {results['avg_duration_hours']:.1f} horas")
        
        print(f"\\n🎯 ANÁLISIS DETALLADO:")
        print(f"   Ganancia bruta: ${results['gross_profit']:,.2f}")
        print(f"   Pérdida bruta: ${results['gross_loss']:,.2f}")
        print(f"   Ganancia promedio: ${results['avg_win']:,.2f}")
        print(f"   Pérdida promedio: ${results['avg_loss']:,.2f}")
        print(f"   Mayor ganancia: ${results['largest_win']:,.2f}")
        print(f"   Mayor pérdida: ${results['largest_loss']:,.2f}")
        
        print(f"\\n📊 DISTRIBUCIÓN:")
        print(f"   Trades LONG: {results['long_trades']}")
        print(f"   Trades SHORT: {results['short_trades']}")
        print(f"   Comisiones pagadas: ${results['commissions_paid']:,.2f}")
        
        # Evaluación de objetivos V2
        print(f"\\n🎯 EVALUACIÓN DE OBJETIVOS V2:")
        roi_status = "✅" if results['roi_percentage'] >= 300 else "❌"
        dd_status = "✅" if results['max_drawdown'] >= -35 else "❌"
        wr_status = "✅" if results['win_rate'] >= 50 else "❌"  # Objetivo más realista
        
        print(f"   {roi_status} ROI ≥300%: {results['roi_percentage']:.1f}%")
        print(f"   {dd_status} Drawdown ≤35%: {results['max_drawdown']:.1f}%")
        print(f"   {wr_status} Win Rate ≥50%: {results['win_rate']:.1f}%")
        
        print(f"\\n💡 COMPARACIÓN CON V1:")
        print(f"   V1: 593 trades, 33.9% WR, -21.6% ROI, -43.5% DD")
        print(f"   V2: {results['total_trades']} trades, {results['win_rate']:.1f}% WR, {results['roi_percentage']:.1f}% ROI, {results['max_drawdown']:.1f}% DD")
        
        improvement = "✅ MEJORA SIGNIFICATIVA" if (results['win_rate'] > 40 and results['total_trades'] < 100) else "⚠️ REQUIERE AJUSTES"
        print(f"\\n🎉 VEREDICTO: {improvement}")
        
        print("="*90)


def main():
    """Función principal para backtest V2."""
    
    parser = argparse.ArgumentParser(description='Multi-Timeframe Backtester V2 (Optimized)')
    parser.add_argument('--symbol', '-s', default='LINK/USDT', help='Símbolo a testear')
    parser.add_argument('--start-date', default='2024-09-03', help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-10-02', help='Fecha fin (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print(f"🚀 BACKTESTER MULTI-TIMEFRAME V2 (OPTIMIZADO)")
    print(f"🎯 Objetivo: Validar mejoras vs V1 (593 trades → ~30-50)")
    print(f"📊 Símbolo: {args.symbol}")
    print(f"📅 Período: {args.start_date} → {args.end_date}")
    print(f"🔧 Filtros V2: 3/3 (4H) + 4/4 (1H) + volatilidad + gap")
    
    try:
        # Crear backtester V2
        backtester = MultiTimeframeBacktesterV2(args.symbol)
        
        # Ejecutar backtest V2
        results = backtester.run_backtest_v2(args.start_date, args.end_date)
        
        # Mostrar resultados
        if 'error' not in results:
            backtester.print_backtest_results_v2(results)
        else:
            print(f"❌ {results['error']}")
        
    except Exception as e:
        print(f"❌ Error en backtest V2: {e}")


if __name__ == "__main__":
    main()