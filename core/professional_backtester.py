#!/usr/bin/env python3
"""
Backtester Profesional Avanzado
==============================

Integra el analizador profesional con el sistema de backtesting para crear
estrategias basadas en scoring multi-factor y análisis técnico avanzado.

Características:
- Estrategia basada en el analizador profesional
- Scoring dinámico multi-factor
- Gestión de riesgo avanzada
- Análisis de performance detallado
- Optimización automática de parámetros
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from backtesting import Backtest, Strategy
import argparse

# Añadir el directorio analysis al path para importar nuestro analizador
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Importar nuestro analizador profesional
try:
    from professional_analyzer import ProfessionalCryptoAnalyzer, ADVANCED_CONFIG, SignalStrength
except ImportError:
    print("❌ Error: No se puede importar el analizador profesional")
    print("   Asegúrate de que professional_analyzer.py está en el mismo directorio")
    sys.exit(1)

# Configuración del backtester profesional
BACKTEST_CONFIG = {
    'initial_cash': 10000,
    'commission': 0.0006,  # 0.06% para futuros
    'slippage': 0.0005,    # 0.05% para futuros
    
    # Configuración de FUTUROS
    'futures_mode': True,
    'leverage': 30,           # Apalancamiento 30x
    'position_size_usd': 100, # Posición fija de $100 USD
    'margin_requirement': 0.033,  # 3.33% margen para 30x (1/30)
    
    # Parámetros de la estrategia profesional
    'min_buy_score': 60,      # Score mínimo para BUY
    'min_strong_buy_score': 75, # Score para BUY fuerte
    'max_sell_score': 40,     # Score máximo para SELL
    'max_strong_sell_score': 25, # Score para SELL fuerte
    
    # Gestión de riesgo para futuros
    'max_risk_per_trade': 0.05,  # 5% máximo por trade (mayor por apalancamiento)
    'position_sizing': 'futures_fixed',  # Modo futuros con posición fija
    'stop_loss_atr_mult': 1.5,   # Más ajustado por apalancamiento
    'take_profit_atr_mult': 2.5,
    
    # Filtros adicionales
    'min_confidence': 0.5,        # 50% mínimo de confianza
    'min_volume_ratio': 0.7,      # Ratio de volumen mínimo
    'max_volatility_pct': 8.0,    # Máxima volatilidad ATR %
    
    # Timeframes para análisis
    'analysis_timeframe': '4h',
    'confirmation_timeframes': ['1h', '4h'],
    
    # Parámetros de optimización
    'lookback_periods': 50,       # Períodos hacia atrás para análisis
    'rebalance_frequency': 6,     # Cada cuántas barras reanalizar
}


@dataclass
class TradeSignal:
    """Señal de trading enriquecida."""
    timestamp: datetime
    symbol: str
    direction: str  # BUY, SELL, HOLD
    strength: SignalStrength
    confidence: float
    score: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    indicators: Dict[str, Any]
    confirmations: List[str]
    warnings: List[str]


class ProfessionalStrategy(Strategy):
    """
    Estrategia basada en el analizador profesional avanzado.
    
    Utiliza scoring multi-factor, análisis técnico profundo y gestión de riesgo avanzada.
    """
    
    # Parámetros configurables
    min_buy_score = BACKTEST_CONFIG['min_buy_score']
    min_strong_buy_score = BACKTEST_CONFIG['min_strong_buy_score']
    max_sell_score = BACKTEST_CONFIG['max_sell_score']
    max_strong_sell_score = BACKTEST_CONFIG['max_strong_sell_score']  # Añadido
    min_confidence = BACKTEST_CONFIG['min_confidence']
    min_volume_ratio = BACKTEST_CONFIG['min_volume_ratio']
    max_risk_per_trade = BACKTEST_CONFIG['max_risk_per_trade']
    rebalance_frequency = BACKTEST_CONFIG['rebalance_frequency']
    
    def init(self):
        """Inicializar la estrategia."""
        self.analyzer = None
        self.last_analysis_bar = -999
        self.current_signal = None
        self.trade_log = []
        self.analysis_cache = {}
        
        print(f"🔬 Inicializando Estrategia Profesional")
        print(f"   Parámetros: min_buy_score={self.min_buy_score}, min_confidence={self.min_confidence}")
        
        # Crear analizador (se configurará dinámicamente)
        try:
            # Usar el símbolo correcto pasado al backtester
            symbol = getattr(self.data, 'symbol', 'BTC/USDT')
            
            # El analizador espera el símbolo sin el namespace, así que lo limpiamos
            clean_symbol = symbol.replace('/', '').replace(':', '') + 'USDT' if not symbol.endswith('USDT') else symbol.replace('/', '').replace(':', '')
            
            print(f"🔧 Configurando analizador para símbolo: {clean_symbol}")
            
            self.analyzer = ProfessionalCryptoAnalyzer(
                symbol=clean_symbol, 
                timeframe=BACKTEST_CONFIG['analysis_timeframe']
            )
            print(f"✅ Analizador creado para {clean_symbol} en timeframe {BACKTEST_CONFIG['analysis_timeframe']}")
        except Exception as e:
            print(f"⚠️ Error creando analizador: {e}")
            print("   Usando análisis simplificado")
            self.analyzer = None
    
    def analyze_current_state(self, bar_index: int) -> Optional[TradeSignal]:
        """Analizar el estado actual del mercado."""
        try:
            # Solo reanalizar cada N barras para eficiencia
            if bar_index - self.last_analysis_bar < self.rebalance_frequency:
                return self.current_signal
            
            self.last_analysis_bar = bar_index
            
            # Obtener datos históricos hasta el punto actual
            current_data = self.data.df.iloc[:bar_index+1].copy()
            
            if len(current_data) < 20:  # Reducido de 50 a 20
                return None
            
            # Análisis técnico completo usando nuestro framework
            analysis_result = self._perform_technical_analysis(current_data)
            
            if not analysis_result:
                return None
            
            # Crear señal de trading
            signal = TradeSignal(
                timestamp=current_data.index[-1],
                symbol=getattr(self.data, 'symbol', 'UNKNOWN'),
                direction=analysis_result['signal']['direction'],
                strength=SignalStrength[analysis_result['signal']['strength']],
                confidence=analysis_result['signal']['confidence'],
                score=analysis_result['scoring']['total_score'],
                entry_price=current_data['Close'].iloc[-1],
                stop_loss=analysis_result['signal']['stop_loss'],
                take_profit=analysis_result['signal']['take_profit'],
                risk_reward_ratio=analysis_result['signal']['risk_reward_ratio'],
                indicators=analysis_result['technical_analysis'],
                confirmations=analysis_result['signal'].get('confirmations', []),
                warnings=analysis_result['signal'].get('warnings', [])
            )
            
            self.current_signal = signal
            return signal
            
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            return None
    
    def _perform_technical_analysis(self, data: pd.DataFrame) -> Optional[Dict]:
        """Realizar análisis técnico usando nuestro framework."""
        try:
            if self.analyzer is None:
                # Análisis simplificado si no hay analizador
                return self._simplified_analysis(data)
            
            # Configurar el analizador con los datos actuales
            # (Simulamos que el analizador puede trabajar con datos históricos)
            
            # Crear un DataFrame compatible con nuestro analizador
            df = data.copy()
            df.columns = df.columns.str.lower()  # close, high, low, open, volume
            
            # Calcular indicadores manualmente (simulando el analizador)
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
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
            ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
            ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            
            # ATR simplificado
            if len(df) >= 14:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift(1))
                low_close = np.abs(df['low'] - df['close'].shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(14).mean().iloc[-1]
            else:
                atr = close[-1] * 0.02
            
            # Volumen
            vol_ratio = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1
            
            # Bollinger Bands
            bb_period = 20
            if len(df) >= bb_period:
                bb_middle = df['close'].rolling(bb_period).mean().iloc[-1]
                bb_std = df['close'].rolling(bb_period).std().iloc[-1]
                bb_upper = bb_middle + (2 * bb_std)
                bb_lower = bb_middle - (2 * bb_std)
                bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            
            # Calcular score
            score = self._calculate_composite_score(
                rsi, ema_9, ema_21, ema_50, close[-1], vol_ratio, bb_position, atr
            )
            
            # Generar señal
            direction, strength, confidence = self._generate_signal_from_score(
                score, rsi, bb_position, vol_ratio
            )
            
            # Calcular niveles de stop loss y take profit
            stop_loss, take_profit = self._calculate_risk_levels(
                close[-1], atr, direction
            )
            
            # Estructurar resultado
            result = {
                'technical_analysis': {
                    'rsi': rsi,
                    'ema_9': ema_9,
                    'ema_21': ema_21,
                    'ema_50': ema_50,
                    'atr': atr,
                    'volume_ratio': vol_ratio,
                    'bb_position': bb_position
                },
                'scoring': {
                    'total_score': score
                },
                'signal': {
                    'direction': direction,
                    'strength': strength,
                    'confidence': confidence,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': abs(take_profit - close[-1]) / abs(close[-1] - stop_loss) if abs(close[-1] - stop_loss) > 0 else 0,
                    'confirmations': self._get_confirmations(rsi, vol_ratio, bb_position),
                    'warnings': self._get_warnings(vol_ratio, atr / close[-1] * 100)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error en análisis técnico: {e}")
            return None
    
    def _calculate_composite_score(self, rsi, ema_9, ema_21, ema_50, price, vol_ratio, bb_position, atr):
        """Calcular score compuesto basado en múltiples factores."""
        score = 50  # Base neutral
        
        # Tendencia (25% del peso)
        trend_score = 0
        if price > ema_9: trend_score += 8
        if price > ema_21: trend_score += 8
        if price > ema_50: trend_score += 9
        score += trend_score
        
        # Momentum (20% del peso)
        if rsi < 30: score += 15
        elif rsi < 40: score += 10
        elif rsi > 70: score -= 15
        elif rsi > 60: score -= 10
        
        # Bollinger position (15% del peso)
        if bb_position < 0.2: score += 12
        elif bb_position > 0.8: score -= 12
        
        # Volumen (15% del peso)
        if vol_ratio > 1.5: score += 12
        elif vol_ratio < 0.7: score -= 12
        
        return max(0, min(100, score))
    
    def _generate_signal_from_score(self, score, rsi, bb_position, vol_ratio):
        """Generar señal basada en el score y condiciones adicionales."""
        # Determinar dirección
        if score >= self.min_strong_buy_score or (score >= self.min_buy_score and rsi < 35):
            direction = 'BUY'
            strength = 'STRONG' if score >= self.min_strong_buy_score else 'MODERATE'
        elif score <= self.max_sell_score:
            direction = 'SELL'
            strength = 'STRONG' if score <= self.max_strong_sell_score else 'MODERATE'
        else:
            direction = 'HOLD'
            strength = 'WEAK'
        
        # Calcular confianza
        confidence = score / 100
        
        # Ajustar confianza por factores adicionales
        if direction == 'BUY' and rsi < 30: confidence += 0.15
        if direction == 'BUY' and bb_position < 0.2: confidence += 0.10
        if vol_ratio > 1.5: confidence += 0.10
        elif vol_ratio < 0.7: confidence -= 0.15
        
        confidence = max(0.1, min(0.95, confidence))
        
        return direction, strength, confidence
    
    def _calculate_risk_levels(self, price, atr, direction):
        """Calcular niveles de stop loss y take profit - MODO FUTUROS."""
        # Para futuros con apalancamiento, usar niveles más ajustados
        if BACKTEST_CONFIG.get('futures_mode', False):
            atr_mult_sl = BACKTEST_CONFIG['stop_loss_atr_mult']  # 1.5x
            atr_mult_tp = BACKTEST_CONFIG['take_profit_atr_mult']  # 2.5x
            
            # Con apalancamiento 30x, los movimientos son amplificados
            # Usar niveles más conservadores
            leverage = BACKTEST_CONFIG['leverage']
            conservative_factor = max(1.0, leverage / 30)  # Factor de conservación
            
            if direction == 'BUY':
                stop_loss = price - (atr * atr_mult_sl * conservative_factor)
                take_profit = price + (atr * atr_mult_tp * conservative_factor)
            elif direction == 'SELL':
                stop_loss = price + (atr * atr_mult_sl * conservative_factor)
                take_profit = price - (atr * atr_mult_tp * conservative_factor)
            else:  # HOLD
                stop_loss = price - (atr * 1.0)
                take_profit = price + (atr * 1.0)
        else:
            # MODO SPOT (código original)
            atr_mult_sl = BACKTEST_CONFIG['stop_loss_atr_mult']
            atr_mult_tp = BACKTEST_CONFIG['take_profit_atr_mult']
            
            if direction == 'BUY':
                stop_loss = price - (atr * atr_mult_sl)
                take_profit = price + (atr * atr_mult_tp)
            elif direction == 'SELL':
                stop_loss = price + (atr * atr_mult_sl)
                take_profit = price - (atr * atr_mult_tp)
            else:  # HOLD
                stop_loss = price - (atr * 1.5)
                take_profit = price + (atr * 1.5)
        
        return stop_loss, take_profit
    
    def _get_confirmations(self, rsi, vol_ratio, bb_position):
        """Obtener confirmaciones técnicas."""
        confirmations = []
        
        if rsi < 30:
            confirmations.append("RSI oversold - oportunidad de rebote")
        if vol_ratio > 1.5:
            confirmations.append("Volumen elevado confirma movimiento")
        if bb_position < 0.2:
            confirmations.append("Precio en banda inferior - posible rebote")
        
        return confirmations
    
    def _get_warnings(self, vol_ratio, atr_pct):
        """Obtener advertencias."""
        warnings = []
        
        if vol_ratio < 0.7:
            warnings.append("Volumen bajo - falta confirmación")
        if atr_pct > 5:
            warnings.append("Alta volatilidad - gestión de riesgo estricta")
        
        return warnings
    
    def _simplified_analysis(self, data):
        """Análisis técnico simplificado cuando no hay analizador disponible."""
        # Implementación básica usando solo pandas
        # (Similar a la función anterior pero más básica)
        pass
    
    def next(self):
        """Lógica principal de la estrategia en cada barra."""
        try:
            current_bar = len(self.data) - 1
            current_price = self.data.Close[-1]
            
            # Analizar situación actual
            signal = self.analyze_current_state(current_bar)
            
            if signal is None:
                return
            
            # Log para debugging
            self._log_analysis(current_bar, signal)
            
            # Gestión de posiciones existentes
            if self.position:
                self._manage_existing_position(signal, current_price)
            else:
                self._evaluate_entry(signal, current_price)
                
        except Exception as e:
            print(f"❌ Error en next(): {e}")
    
    def _manage_existing_position(self, signal: TradeSignal, current_price: float):
        """Gestionar posición existente."""
        # Salir si:
        # 1. Señal contraria fuerte
        # 2. Stop loss o take profit alcanzado
        # 3. Confianza muy baja
        
        if signal.direction == 'SELL' and signal.confidence > 0.6:
            print(f"[{len(self.data)}] SALIDA: Señal SELL fuerte")
            self.position.close()
        elif signal.confidence < 0.3:
            print(f"[{len(self.data)}] SALIDA: Confianza muy baja ({signal.confidence:.2f})")
            self.position.close()
    
    def _evaluate_entry(self, signal: TradeSignal, current_price: float):
        """Evaluar entrada en nueva posición - MODO FUTUROS."""
        # Filtros de entrada
        if signal.confidence < self.min_confidence:
            return
        
        # MODO FUTUROS: Considerar tanto LONG como SHORT
        if BACKTEST_CONFIG.get('futures_mode', False):
            
            if signal.direction == 'BUY' and signal.score >= self.min_buy_score:
                # LONG Position
                position_size = self._calculate_position_size(signal, current_price)
                
                if position_size > 0:
                    leverage = BACKTEST_CONFIG['leverage']
                    position_usd = BACKTEST_CONFIG['position_size_usd']
                    
                    print(f"[{len(self.data)}] FUTURES LONG: ${position_usd} @ {leverage}x | Score={signal.score:.1f} | Conf={signal.confidence:.2f}")
                    
                    # Calcular stop loss y take profit ajustados para futuros
                    sl_price = signal.stop_loss
                    tp_price = signal.take_profit
                    
                    self.buy(
                        size=position_size,
                        sl=sl_price,
                        tp=tp_price
                    )
                    
                    # Log del trade
                    self.trade_log.append({
                        'bar': len(self.data),
                        'action': 'FUTURES_LONG',
                        'price': current_price,
                        'position_usd': position_usd,
                        'leverage': leverage,
                        'signal': signal
                    })
                    
            elif signal.direction == 'SELL' and signal.score <= self.max_sell_score:
                # SHORT Position
                position_size = self._calculate_position_size(signal, current_price)
                
                if position_size > 0:
                    leverage = BACKTEST_CONFIG['leverage']
                    position_usd = BACKTEST_CONFIG['position_size_usd']
                    
                    print(f"[{len(self.data)}] FUTURES SHORT: ${position_usd} @ {leverage}x | Score={signal.score:.1f} | Conf={signal.confidence:.2f}")
                    
                    # Para SHORT, invertir los niveles
                    sl_price = signal.take_profit  # El TP del signal es nuestro SL para short
                    tp_price = signal.stop_loss    # El SL del signal es nuestro TP para short
                    
                    self.sell(
                        size=position_size,
                        sl=sl_price,
                        tp=tp_price
                    )
                    
                    # Log del trade
                    self.trade_log.append({
                        'bar': len(self.data),
                        'action': 'FUTURES_SHORT',
                        'price': current_price,
                        'position_usd': position_usd,
                        'leverage': leverage,
                        'signal': signal
                    })
        else:
            # MODO SPOT (código original)
            if signal.direction == 'BUY' and signal.score >= self.min_buy_score:
                # Calcular tamaño de posición
                position_size = self._calculate_position_size(signal, current_price)
                
                if position_size > 0:
                    print(f"[{len(self.data)}] ENTRADA BUY: Score={signal.score:.1f}, Conf={signal.confidence:.2f}, Size={position_size:.3f}")
                    
                    self.buy(
                        size=position_size,
                        sl=signal.stop_loss,
                        tp=signal.take_profit
                    )
                    
                    # Log del trade
                    self.trade_log.append({
                        'bar': len(self.data),
                        'action': 'BUY',
                        'price': current_price,
                        'signal': signal
                    })
    
    def _calculate_position_size(self, signal: TradeSignal, current_price: float):
        """Calcular tamaño de posición basado en riesgo - MODO FUTUROS."""
        try:
            current_price = float(current_price)
            
            # MODO FUTUROS: Posición fija en USD con apalancamiento
            if BACKTEST_CONFIG.get('futures_mode', False):
                position_size_usd = BACKTEST_CONFIG['position_size_usd']  # $100 USD
                leverage = BACKTEST_CONFIG['leverage']  # 30x
                
                # Obtener equity actual de forma segura
                try:
                    equity = float(self._broker._equity)
                except:
                    equity = float(BACKTEST_CONFIG['initial_cash'])
                
                # Calcular el margen requerido
                margin_required = position_size_usd / leverage  # $100 / 30 = $3.33
                
                # Verificar si hay suficiente margen
                if margin_required > equity * 0.8:  # No usar más del 80% del equity
                    print(f"⚠️ Margen insuficiente: Necesario ${margin_required:.2f}, Disponible ${equity * 0.8:.2f}")
                    return 0.001  # Posición mínima
                
                # Para simular futuros en el backtester, necesitamos un enfoque diferente
                # Vamos a usar solo el margen como "compra" pero simular el P&L amplificado
                
                # Usar solo el margen como posición en el backtester
                margin_fraction = margin_required / equity
                
                # Asegurar que esté dentro de límites razonables
                position_fraction = max(0.005, min(0.15, margin_fraction))  # Entre 0.5% y 15%
                
                print(f"💰 Futuros: ${position_size_usd} USD @ {leverage}x | Margen: ${margin_required:.2f} | Fracción: {position_fraction:.3f}")
                
                return position_fraction
            
            # MODO SPOT (código original simplificado)
            base_fraction = 0.02  # 2% base
            
            # Ajustar por precio
            if current_price > 1000:  # BTC y otros activos caros
                base_fraction = 0.005  # 0.5%
            
            # Ajustar por confianza y score de forma segura
            try:
                confidence_mult = float(signal.confidence) if signal.confidence > 0 else 0.5
                score_mult = float(signal.score) / 100 if signal.score > 0 else 0.6
                
                position_fraction = base_fraction * confidence_mult * score_mult
            except:
                position_fraction = base_fraction
            
            # Limitar entre 0.1% y 10%
            return max(0.001, min(0.1, position_fraction))
            
        except Exception as e:
            print(f"❌ Error calculando posición: {e}")
            # Fallback ultra-seguro
            if BACKTEST_CONFIG.get('futures_mode', False):
                return 0.02  # 2% para futuros
            return 0.01  # 1% para spot
    
    def _log_analysis(self, bar: int, signal: TradeSignal):
        """Log del análisis para debugging."""
        if bar % 24 == 0:  # Log cada 24 barras
            print(f"[{bar}] {signal.symbol} | {signal.direction} | Score: {signal.score:.0f} | Conf: {signal.confidence:.2f} | Price: ${signal.entry_price:.6f}")


class AdvancedBacktester:
    """Backtester profesional con análisis avanzado."""
    
    def __init__(self, symbol: str, timeframe: str = '4h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.results = {}
        
        # Actualizar configuración con el timeframe especificado
        BACKTEST_CONFIG['analysis_timeframe'] = timeframe
        
    def load_data(self, data_source: str = 'cache') -> Optional[pd.DataFrame]:
        """Cargar datos para backtesting."""
        try:
            if data_source == 'cache':
                return self._load_from_cache()
            else:
                return self._load_from_exchange()
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Cargar datos desde cache_real."""
        # Buscar en cache_real con formato nuevo
        symbol_clean = self.symbol.replace('_', '_').replace('/', '_')
        timeframe = self.timeframe if hasattr(self, 'timeframe') and self.timeframe else '1h'
        
        cache_real_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'data', 'cache_real', f'{symbol_clean}_{timeframe}.json'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'cache_real', f'{self.symbol}_{timeframe}.json')
        ]
        
        cache_path = None
        for path in cache_real_paths:
            if os.path.exists(path):
                cache_path = path
                break
        
        if not cache_path:
            print(f"⚠️ No existe cache para {self.symbol}_{timeframe} en ninguna ubicación")
            print(f"   Buscado en: {[os.path.basename(p) for p in cache_real_paths]}")
            return None
        
        print(f"📁 Cargando desde: {cache_path}")
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"❌ Error leyendo cache: {e}")
            return None
        
        # Procesar datos según formato (cache_real o formato antiguo)
        if 'data' in cache:
            # Formato cache_real nuevo
            ohlcv_data = cache['data']
        else:
            # Formato antiguo
            h = cache.get('history_data', {})
            if h.get('ohlcv'):
                ohlcv_data = h['ohlcv']
            else:
                print(f"❌ No se encontraron datos OHLCV en {cache_path}")
                return None
        
        # Convertir a DataFrame
        try:
            if isinstance(ohlcv_data[0], dict):
                # Formato dict con keys timestamp, Open, High, Low, Close, Volume
                df = pd.DataFrame(ohlcv_data)
                # Asegurar que timestamp sea datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Normalizar nombres de columnas
                df.columns = df.columns.str.capitalize()
                if 'Open' not in df.columns:
                    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            else:
                # Formato lista [timestamp, open, high, low, close, volume]
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
                # Manejar diferentes formatos de timestamp
                if pd.api.types.is_integer_dtype(df['timestamp'].iloc[0]):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                df = df.dropna(subset=['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            print(f"✅ Datos OHLCV cargados: {len(df)} barras desde {df.index[0]} hasta {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"❌ Error procesando datos OHLCV: {e}")
            return None
            ohlcv = h['ohlcv']
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Manejar diferentes formatos de timestamp
            if pd.api.types.is_integer_dtype(df['timestamp'].iloc[0]):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
            df = df.dropna(subset=['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Datos OHLCV cargados: {len(df)} barras")
            return df
            
        except Exception as e:
            print(f"❌ Error procesando datos OHLCV: {e}")
            return None
    
    def run_backtest(self, 
                    strategy_class=ProfessionalStrategy,
                    cash: float = None,
                    commission: float = None) -> Dict:
        """Ejecutar backtest con la estrategia profesional."""
        
        print(f"\n🔬 BACKTEST PROFESIONAL: {self.symbol}")
        print("=" * 60)
        
        # Cargar datos
        df = self.load_data()
        if df is None or len(df) < 50:  # Reducido de 100 a 50
            print(f"❌ Datos insuficientes para {self.symbol} (necesario: 50, disponible: {len(df) if df is not None else 0})")
            return {}
        
        print(f"📊 Datos cargados: {len(df)} barras ({df.index[0]} a {df.index[-1]})")
        
        # Configurar parámetros
        cash = cash or BACKTEST_CONFIG['initial_cash']
        commission = commission or BACKTEST_CONFIG['commission']
        
        # Agregar símbolo a los datos para la estrategia
        df.symbol = self.symbol
        
        # Ejecutar backtest
        bt = Backtest(df, strategy_class, cash=cash, commission=commission, 
                     exclusive_orders=True, finalize_trades=True)  # Añadido finalize_trades=True
        
        try:
            stats = bt.run()
            
            # Procesar y mostrar resultados
            self._display_results(stats, df)
            
            self.results = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'stats': stats,
                'data_period': f"{df.index[0]} to {df.index[-1]}",
                'total_bars': len(df),
                'advanced_metrics': self._calculate_advanced_metrics(stats, df)
            }
            
            # Guardar reporte detallado
            self._save_detailed_report(stats, df)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error ejecutando backtest: {e}")
            return {}
    
    def _display_results(self, stats, df):
        """Mostrar resultados completos del backtest - MODO FUTUROS."""
        futures_mode = BACKTEST_CONFIG.get('futures_mode', False)
        mode_text = "FUTUROS" if futures_mode else "SPOT"
        
        print(f"\n📈 RESULTADOS COMPLETOS DEL BACKTEST ({mode_text}):")
        print("=" * 60)
        
        if futures_mode:
            leverage = BACKTEST_CONFIG['leverage']
            position_usd = BACKTEST_CONFIG['position_size_usd']
            print(f"🔧 CONFIGURACIÓN FUTUROS:")
            print(f"   Apalancamiento: {leverage}x")
            print(f"   Tamaño de posición: ${position_usd} USD")
            print(f"   Margen por trade: ${position_usd/leverage:.2f} USD")
            print()
        
        # Métricas básicas
        return_pct = stats.get('Return [%]', 0)
        win_rate = stats.get('Win Rate [%]', 0)
        num_trades = stats.get('# Trades', 0)
        sharpe = stats.get('Sharpe Ratio', 0)
        max_drawdown = stats.get('Max. Drawdown [%]', 0)
        
        print(f"💰 RENDIMIENTO:")
        if futures_mode:
            # Para futuros, calcular el ROI sobre el margen usado
            total_margin_used = (position_usd / leverage) * num_trades if num_trades > 0 else position_usd / leverage
            effective_roi = return_pct * leverage if return_pct != 0 else 0
            print(f"   Retorno Total: {return_pct:.2f}% (sobre capital)")
            print(f"   ROI sobre Margen: {effective_roi:.2f}% (con {leverage}x leverage)")
        else:
            print(f"   Retorno Total: {return_pct:.2f}%")
        
        print(f"   Retorno Anualizado: {self._annualize_return(return_pct, df):.2f}%")
        
        # Métricas de riesgo
        print(f"\n🛡️ MÉTRICAS DE RIESGO:")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        if futures_mode and max_drawdown != 0:
            leveraged_dd = max_drawdown * leverage
            print(f"   Drawdown Efectivo (con leverage): {leveraged_dd:.2f}%")
        
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        
        # Calcular métricas adicionales
        sortino_ratio = self._calculate_sortino_ratio(stats, df)
        calmar_ratio = self._calculate_calmar_ratio(return_pct, max_drawdown)
        volatility = self._calculate_volatility(stats)
        
        print(f"   Sortino Ratio: {sortino_ratio:.2f}")
        print(f"   Calmar Ratio: {calmar_ratio:.2f}")
        print(f"   Volatilidad Anualizada: {volatility:.2f}%")
        
        # Métricas de trading
        print(f"\n📊 ESTADÍSTICAS DE TRADING:")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Número de Trades: {num_trades}")
        
        if num_trades > 0:
            avg_trade = return_pct / num_trades
            print(f"   Retorno promedio por trade: {avg_trade:.2f}%")
            
            if futures_mode:
                avg_trade_usd = (return_pct / 100) * BACKTEST_CONFIG['initial_cash'] / num_trades
                print(f"   P&L promedio por trade: ${avg_trade_usd:.2f} USD")
            
            # Trades ganadores vs perdedores
            best_trade = stats.get('Best Trade [%]', 0)
            worst_trade = stats.get('Worst Trade [%]', 0)
            avg_win = stats.get('Avg. Trade [%]', 0)
            
            print(f"   Mejor trade: {best_trade:.2f}%")
            print(f"   Peor trade: {worst_trade:.2f}%")
            print(f"   Trade promedio: {avg_win:.2f}%")
            
            # Profit Factor
            profit_factor = stats.get('Profit Factor', 0)
            if profit_factor and not pd.isna(profit_factor):
                print(f"   Profit Factor: {profit_factor:.2f}")
            
            # Expectancy
            expectancy = self._calculate_expectancy(stats)
            print(f"   Expectancy: {expectancy:.2f}%")
        
        # Análisis específico para futuros
        if futures_mode:
            self._analyze_futures_performance(stats, df)
        
        # Análisis de drawdown detallado
        self._analyze_drawdown_periods(stats, df)
        
        # Análisis temporal
        self._analyze_temporal_performance(stats, df)
        
        # Análisis de consistencia
        self._analyze_consistency(stats, df)
        
        # Calificación del resultado
        grade = self._grade_performance_advanced(return_pct, win_rate, sharpe, max_drawdown, sortino_ratio)
        print(f"\n🏆 EVALUACIÓN FINAL:")
        print(f"   Calificación: {grade}")
        
        # Resumen ejecutivo
        print(f"\n📋 RESUMEN EJECUTIVO ({mode_text}):")
        print(f"   • Estrategia {'RENTABLE' if return_pct > 0 else 'NO RENTABLE'}")
        if futures_mode:
            risk_level = 'ALTO' if max_drawdown > 10 else 'CONTROLADO' if max_drawdown < 5 else 'MODERADO'
            print(f"   • Riesgo {risk_level} (apalancamiento {leverage}x)")
        else:
            print(f"   • Riesgo {'CONTROLADO' if max_drawdown < 20 else 'ALTO'}")
        print(f"   • Consistencia {'ALTA' if win_rate > 60 else 'MEDIA' if win_rate > 40 else 'BAJA'}")
        print(f"   • Sharpe {'EXCELENTE' if sharpe > 2 else 'BUENO' if sharpe > 1 else 'REGULAR' if sharpe > 0.5 else 'MALO'}")
        
        print("=" * 60)
    
    def _annualize_return(self, total_return: float, df: pd.DataFrame) -> float:
        """Calcular retorno anualizado."""
        try:
            days = (df.index[-1] - df.index[0]).days
            if days > 0:
                years = days / 365.25
                return (((1 + total_return/100) ** (1/years)) - 1) * 100
            return total_return
        except:
            return total_return
    
    def _calculate_sortino_ratio(self, stats, df: pd.DataFrame) -> float:
        """Calcular Sortino Ratio (similar a Sharpe pero solo con volatilidad negativa)."""
        try:
            # Obtener retornos si están disponibles
            if hasattr(stats, '_equity_curve'):
                returns = stats._equity_curve.pct_change(fill_method=None).dropna()
                negative_returns = returns[returns < 0]
                
                if len(negative_returns) > 0:
                    downside_deviation = float(negative_returns.std() * np.sqrt(252))  # Anualizado
                    excess_return = self._annualize_return(stats.get('Return [%]', 0), df)
                    
                    if downside_deviation > 0:
                        return excess_return / downside_deviation
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calcular Calmar Ratio (retorno/drawdown máximo)."""
        try:
            if abs(max_drawdown) > 0:
                return float(total_return / abs(max_drawdown))
            return 0.0
        except:
            return 0.0
    
    def _calculate_volatility(self, stats) -> float:
        """Calcular volatilidad anualizada."""
        try:
            if hasattr(stats, '_equity_curve'):
                returns = stats._equity_curve.pct_change(fill_method=None).dropna()
                return float(returns.std() * np.sqrt(252) * 100)  # Anualizado en %
            return 0.0
        except:
            return 0.0
    
    def _calculate_expectancy(self, stats) -> float:
        """Calcular expectancy por trade."""
        try:
            win_rate = float(stats.get('Win Rate [%]', 0)) / 100
            avg_win = float(stats.get('Avg. Win [%]', 0))
            avg_loss = abs(float(stats.get('Avg. Loss [%]', 0)))
            
            if avg_loss > 0:
                return float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))
            return 0.0
        except:
            return 0.0
    
    def _analyze_futures_performance(self, stats, df: pd.DataFrame):
        """Análizar rendimiento específico para futuros."""
        print(f"\n⚡ ANÁLISIS ESPECÍFICO DE FUTUROS:")
        
        leverage = BACKTEST_CONFIG['leverage']
        position_usd = BACKTEST_CONFIG['position_size_usd']
        initial_cash = BACKTEST_CONFIG['initial_cash']
        
        num_trades = stats.get('# Trades', 0)
        return_pct = stats.get('Return [%]', 0)
        
        # Análisis de exposición
        max_exposure = position_usd * num_trades if num_trades > 0 else position_usd
        exposure_ratio = max_exposure / initial_cash
        
        print(f"   Exposición máxima: ${max_exposure:,.0f} USD ({exposure_ratio:.1f}x capital)")
        print(f"   Margen total usado: ${max_exposure/leverage:,.0f} USD")
        
        # Eficiencia del apalancamiento
        if return_pct != 0:
            leverage_efficiency = return_pct / (max_exposure / initial_cash)
            print(f"   Eficiencia del leverage: {leverage_efficiency:.2f}%")
        
        # Riesgo de liquidación (simulado)
        max_drawdown = stats.get('Max. Drawdown [%]', 0)
        liquidation_risk = (max_drawdown * leverage) / 100
        
        if liquidation_risk > 0.5:  # 50% del margen
            print(f"   ⚠️ RIESGO DE LIQUIDACIÓN: {liquidation_risk*100:.1f}% del margen")
        elif liquidation_risk > 0.3:  # 30% del margen
            print(f"   ⚠️ Riesgo moderado de liquidación: {liquidation_risk*100:.1f}% del margen")
        else:
            print(f"   ✅ Riesgo de liquidación bajo: {liquidation_risk*100:.1f}% del margen")
        
        # Análisis de trades LONG vs SHORT
        try:
            if hasattr(self, 'trade_log') and self.trade_log:
                long_trades = sum(1 for trade in self.trade_log if trade.get('action') == 'FUTURES_LONG')
                short_trades = sum(1 for trade in self.trade_log if trade.get('action') == 'FUTURES_SHORT')
                
                if long_trades > 0 or short_trades > 0:
                    print(f"   Trades LONG: {long_trades} | Trades SHORT: {short_trades}")
                    print(f"   Bias direccional: {'LONG' if long_trades > short_trades else 'SHORT' if short_trades > long_trades else 'NEUTRAL'}")
            else:
                print(f"   Información de trades no disponible")
        except Exception as e:
            print(f"   Error analizando trades: {e}")
    
    def _analyze_drawdown_periods(self, stats, df: pd.DataFrame):
        """Analizar períodos de drawdown en detalle."""
        print(f"\n� ANÁLISIS DE DRAWDOWN:")
        
        max_dd = stats.get('Max. Drawdown [%]', 0)
        max_dd_duration = stats.get('Max. Drawdown Duration', 0)
        
        print(f"   Drawdown Máximo: {max_dd:.2f}%")
        if max_dd_duration:
            print(f"   Duración DD Máximo: {max_dd_duration} períodos")
        
        # Clasificar severidad del drawdown
        if max_dd < 5:
            dd_severity = "BAJO"
        elif max_dd < 15:
            dd_severity = "MODERADO"
        elif max_dd < 30:
            dd_severity = "ALTO"
        else:
            dd_severity = "CRÍTICO"
        
        print(f"   Severidad: {dd_severity}")
        
        # Advice basado en drawdown
        if max_dd > 20:
            print(f"   ⚠️ ADVERTENCIA: Drawdown alto - revisar gestión de riesgo")
        elif max_dd < 10:
            print(f"   ✅ Drawdown controlado - buena gestión de riesgo")
    
    def _analyze_temporal_performance(self, stats, df: pd.DataFrame):
        """Analizar rendimiento temporal."""
        print(f"\n⏰ ANÁLISIS TEMPORAL:")
        
        try:
            total_days = (df.index[-1] - df.index[0]).days
            total_return = stats.get('Return [%]', 0)
            
            print(f"   Período analizado: {total_days} días")
            print(f"   Retorno por mes: {(total_return * 30 / total_days):.2f}%")
            
            # Análisis por fases del mercado
            if hasattr(stats, '_equity_curve'):
                equity = stats._equity_curve
                
                # Detectar fases alcistas y bajistas
                mid_point = len(equity) // 2
                first_half = equity.iloc[:mid_point].iloc[-1] / equity.iloc[0]
                second_half = equity.iloc[-1] / equity.iloc[mid_point]
                
                print(f"   Primera mitad: {((first_half - 1) * 100):.2f}%")
                print(f"   Segunda mitad: {((second_half - 1) * 100):.2f}%")
                
                # Consistencia
                if abs((first_half - 1) * 100) < abs((second_half - 1) * 100) * 2:
                    print(f"   📊 Performance CONSISTENTE entre períodos")
                else:
                    print(f"   ⚠️ Performance INCONSISTENTE entre períodos")
        
        except Exception as e:
            print(f"   Datos temporales no disponibles")
    
    def _analyze_consistency(self, stats, df: pd.DataFrame):
        """Analizar consistencia de la estrategia."""
        print(f"\n🎯 ANÁLISIS DE CONSISTENCIA:")
        
        num_trades = stats.get('# Trades', 0)
        win_rate = stats.get('Win Rate [%]', 0)
        
        if num_trades >= 10:
            consistency = "ALTA"
        elif num_trades >= 5:
            consistency = "MEDIA" 
        else:
            consistency = "BAJA (pocos trades)"
        
        print(f"   Muestra de trades: {consistency}")
        
        # Análisis de racha
        if win_rate > 80:
            print(f"   📈 Win rate muy alto - posible overfitting")
        elif win_rate > 60:
            print(f"   ✅ Win rate saludable")
        elif win_rate > 40:
            print(f"   📊 Win rate aceptable")
        else:
            print(f"   📉 Win rate bajo - revisar estrategia")
        
        # Recomendaciones
        if num_trades < 10:
            print(f"   💡 SUGERENCIA: Extender período de prueba para más trades")
        
        if win_rate == 100 and num_trades < 5:
            print(f"   ⚠️ CUIDADO: 100% win rate con pocos trades - posible cherry picking")
    
    def _grade_performance_advanced(self, return_pct: float, win_rate: float, sharpe: float, 
                                   max_drawdown: float, sortino: float) -> str:
        """Calificar el rendimiento con criterios avanzados."""
        score = 0
        
        # Retorno (25% peso)
        if return_pct > 50: score += 25
        elif return_pct > 30: score += 20
        elif return_pct > 15: score += 15
        elif return_pct > 5: score += 10
        elif return_pct > 0: score += 5
        
        # Sharpe Ratio (25% peso)
        if sharpe > 2.5: score += 25
        elif sharpe > 2: score += 20
        elif sharpe > 1.5: score += 15
        elif sharpe > 1: score += 10
        elif sharpe > 0.5: score += 5
        
        # Drawdown (25% peso) - menos es mejor
        if max_drawdown < 5: score += 25
        elif max_drawdown < 10: score += 20
        elif max_drawdown < 15: score += 15
        elif max_drawdown < 25: score += 10
        elif max_drawdown < 35: score += 5
        
        # Win Rate (15% peso)
        if win_rate > 70: score += 15
        elif win_rate > 60: score += 12
        elif win_rate > 50: score += 10
        elif win_rate > 40: score += 7
        elif win_rate > 30: score += 3
        
        # Sortino Ratio (10% peso)
        if sortino > 2: score += 10
        elif sortino > 1.5: score += 8
        elif sortino > 1: score += 6
        elif sortino > 0.5: score += 3
        
        # Calificación final
        if score >= 90: return "A++ (Excepcional)"
        elif score >= 80: return "A+ (Excelente)"
        elif score >= 70: return "A (Muy Bueno)"
        elif score >= 60: return "B+ (Bueno)"
        elif score >= 50: return "B (Aceptable)"
        elif score >= 40: return "C (Regular)"
        elif score >= 30: return "D (Malo)"
        else: return "F (Muy Malo)"
    
    def _calculate_advanced_metrics(self, stats, df: pd.DataFrame) -> Dict:
        """Calcular métricas avanzadas adicionales."""
        try:
            metrics = {
                'annualized_return': self._annualize_return(stats.get('Return [%]', 0), df),
                'sortino_ratio': self._calculate_sortino_ratio(stats, df),
                'calmar_ratio': self._calculate_calmar_ratio(stats.get('Return [%]', 0), stats.get('Max. Drawdown [%]', 0)),
                'volatility': self._calculate_volatility(stats),
                'expectancy': self._calculate_expectancy(stats),
                'max_consecutive_wins': self._calculate_max_consecutive_wins(stats),
                'max_consecutive_losses': self._calculate_max_consecutive_losses(stats),
                'recovery_factor': self._calculate_recovery_factor(stats),
                'ulcer_index': self._calculate_ulcer_index(stats)
            }
            return metrics
        except Exception as e:
            print(f"⚠️ Error calculando métricas avanzadas: {e}")
            return {}
    
    def _calculate_max_consecutive_wins(self, stats) -> int:
        """Calcular racha máxima de trades ganadores."""
        try:
            if hasattr(stats, '_trades') and len(stats._trades) > 0:
                trades = stats._trades
                current_streak = 0
                max_streak = 0
                
                for trade in trades:
                    if trade.PnL > 0:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
                        
                return max_streak
            return 0
        except:
            return 0
    
    def _calculate_max_consecutive_losses(self, stats) -> int:
        """Calcular racha máxima de trades perdedores."""
        try:
            if hasattr(stats, '_trades') and len(stats._trades) > 0:
                trades = stats._trades
                current_streak = 0
                max_streak = 0
                
                for trade in trades:
                    if trade.PnL < 0:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
                        
                return max_streak
            return 0
        except:
            return 0
    
    def _calculate_recovery_factor(self, stats) -> float:
        """Calcular factor de recuperación (total return / max drawdown)."""
        try:
            total_return = stats.get('Return [%]', 0)
            max_drawdown = stats.get('Max. Drawdown [%]', 0)
            
            if max_drawdown > 0:
                return total_return / max_drawdown
            return 0
        except:
            return 0
    
    def _calculate_ulcer_index(self, stats) -> float:
        """Calcular Ulcer Index (medida de drawdown ponderada por tiempo)."""
        try:
            if hasattr(stats, '_equity_curve'):
                equity = stats._equity_curve
                running_max = equity.expanding().max()
                drawdown = (equity - running_max) / running_max * 100
                
                # Ulcer Index = sqrt(mean(drawdown^2))
                ulcer = float(np.sqrt((drawdown ** 2).mean()))
                return ulcer
            return 0.0
        except:
            return 0.0
    
    def _save_detailed_report(self, stats, df: pd.DataFrame):
        """Guardar reporte detallado en archivo."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{self.symbol.replace('/', '_')}_{timestamp}.json"
            
            # Preparar datos para JSON
            report = {
                'metadata': {
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'start_date': df.index[0].isoformat(),
                    'end_date': df.index[-1].isoformat(),
                    'total_bars': len(df),
                    'generated_at': timestamp
                },
                'performance_metrics': {
                    'total_return_pct': float(stats.get('Return [%]', 0)),
                    'annualized_return_pct': float(self._annualize_return(stats.get('Return [%]', 0), df)),
                    'win_rate_pct': float(stats.get('Win Rate [%]', 0)),
                    'num_trades': int(stats.get('# Trades', 0)),
                    'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
                    'max_drawdown_pct': float(stats.get('Max. Drawdown [%]', 0)),
                    'sortino_ratio': float(self._calculate_sortino_ratio(stats, df)),
                    'calmar_ratio': float(self._calculate_calmar_ratio(stats.get('Return [%]', 0), stats.get('Max. Drawdown [%]', 0))),
                    'volatility_pct': float(self._calculate_volatility(stats)),
                    'expectancy_pct': float(self._calculate_expectancy(stats))
                },
                'risk_metrics': {
                    'max_consecutive_wins': self._calculate_max_consecutive_wins(stats),
                    'max_consecutive_losses': self._calculate_max_consecutive_losses(stats),
                    'recovery_factor': float(self._calculate_recovery_factor(stats)),
                    'ulcer_index': float(self._calculate_ulcer_index(stats))
                },
                'trade_metrics': {
                    'best_trade_pct': float(stats.get('Best Trade [%]', 0)),
                    'worst_trade_pct': float(stats.get('Worst Trade [%]', 0)),
                    'avg_trade_pct': float(stats.get('Avg. Trade [%]', 0)),
                    'profit_factor': float(stats.get('Profit Factor', 0)) if stats.get('Profit Factor') and not pd.isna(stats.get('Profit Factor')) else None
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Reporte detallado guardado: {filename}")
            
        except Exception as e:
            print(f"⚠️ Error guardando reporte: {e}")
    
    def _grade_performance(self, return_pct: float, win_rate: float, sharpe: float) -> str:
        """Calificar el rendimiento básico (mantenido para compatibilidad)."""
        return self._grade_performance_advanced(return_pct, win_rate, sharpe, 0, 0)
    
    def _grade_performance(self, return_pct, win_rate, sharpe):
        """Calificar el rendimiento de la estrategia."""
        score = 0
        
        # Retorno
        if return_pct > 50: score += 3
        elif return_pct > 20: score += 2
        elif return_pct > 0: score += 1
        
        # Win Rate
        if win_rate > 70: score += 3
        elif win_rate > 50: score += 2
        elif win_rate > 40: score += 1
        
        # Sharpe
        if sharpe > 2: score += 3
        elif sharpe > 1: score += 2
        elif sharpe > 0.5: score += 1
        
        if score >= 8: return "A+ (Excelente)"
        elif score >= 6: return "A (Muy Bueno)"
        elif score >= 4: return "B (Bueno)"
        elif score >= 2: return "C (Regular)"
        else: return "D (Malo)"
    
    def optimize_parameters(self, param_grid: Dict) -> pd.DataFrame:
        """Optimizar parámetros de la estrategia."""
        print(f"\n🔧 OPTIMIZANDO PARÁMETROS...")
        
        results = []
        df = self.load_data()
        
        if df is None:
            return pd.DataFrame()
        
        from itertools import product
        
        # Generar combinaciones de parámetros
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Crear clase de estrategia dinámica
            class OptimizedStrategy(ProfessionalStrategy):
                pass
            
            # Aplicar parámetros
            for param_name, param_value in params.items():
                setattr(OptimizedStrategy, param_name, param_value)
            
            try:
                bt = Backtest(df, OptimizedStrategy, 
                             cash=BACKTEST_CONFIG['initial_cash'], 
                             commission=BACKTEST_CONFIG['commission'])
                stats = bt.run()
                
                result = params.copy()
                result.update({
                    'return_pct': stats.get('Return [%]', 0),
                    'win_rate': stats.get('Win Rate [%]', 0),
                    'num_trades': stats.get('# Trades', 0),
                    'sharpe_ratio': stats.get('Sharpe Ratio', 0)
                })
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error con parámetros {params}: {e}")
        
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('return_pct', ascending=False)
            
            print(f"✅ Optimización completada: {len(results)} combinaciones probadas")
            print("\n🏆 MEJORES RESULTADOS:")
            print(results_df.head(5).to_string(index=False))
            
            return results_df
        
        return pd.DataFrame()


def main():
    """Función principal del backtester profesional."""
    parser = argparse.ArgumentParser(description="Backtester Profesional Avanzado")
    parser.add_argument('symbol_timeframe', help='Símbolo_timeframe (ej: BTC_USDT_1h)')
    parser.add_argument('--cash', type=float, default=10000, help='Capital inicial')
    parser.add_argument('--optimize', action='store_true', help='Ejecutar optimización de parámetros')
    parser.add_argument('--min_buy_score', type=float, default=60, help='Score mínimo para BUY')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Confianza mínima')
    
    args = parser.parse_args()
    
    try:
        # Parsear símbolo y timeframe
        parts = args.symbol_timeframe.split('_')
        if len(parts) >= 3:
            symbol = f"{parts[0]}_{parts[1]}"  # BTC_USDT
            timeframe = parts[2]  # 1h
        else:
            print(f"❌ Formato incorrecto. Use: SYMBOL_QUOTE_TIMEFRAME (ej: BTC_USDT_1h)")
            return
        
        print(f"🎯 Analizando: {symbol} en timeframe {timeframe}")
        
        # Crear backtester
        backtester = AdvancedBacktester(symbol, timeframe)
        
        # Configurar parámetros si se especifican
        if args.min_buy_score != 60:
            BACKTEST_CONFIG['min_buy_score'] = args.min_buy_score
        if args.min_confidence != 0.5:
            BACKTEST_CONFIG['min_confidence'] = args.min_confidence
        
        # Ejecutar backtest
        results = backtester.run_backtest(cash=args.cash)
        
        if not results:
            print("❌ No se pudo ejecutar el backtest")
            return
        
        # Optimización si se solicita
        if args.optimize:
            param_grid = {
                'min_buy_score': [55, 60, 65, 70],
                'min_confidence': [0.4, 0.5, 0.6, 0.7],
                'max_risk_per_trade': [0.015, 0.02, 0.025, 0.03]
            }
            
            optimization_results = backtester.optimize_parameters(param_grid)
            
            if not optimization_results.empty:
                # Guardar resultados de optimización
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_{symbol.replace('/', '_')}_{timeframe}_{timestamp}.csv"
                optimization_results.to_csv(filename, index=False)
                print(f"\n💾 Resultados de optimización guardados: {filename}")
        
        print(f"\n✅ Backtest profesional completado para {symbol}_{timeframe}")
        
    except KeyboardInterrupt:
        print("\n🛑 Backtest interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()