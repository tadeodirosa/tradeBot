#!/usr/bin/env python3
"""
Analizador Profesional Avanzado de Crypto
=========================================

Análisis técnico profesional con múltiples confirmaciones, patrones de velas,
análisis de volumen, niveles clave, y scoring avanzado para decisiones más informadas.

Características avanzadas:
- 20+ indicadores técnicos
- Análisis de patrones de velas
- Detección de soportes y resistencias
- Análisis de volumen y liquidez
- Divergencias en indicadores
- Scoring ponderado multi-factor
- Gestión de riesgo avanzada
"""

import sys
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import talib as ta
except ImportError:
    print("TA-Lib no disponible, usando cálculos manuales")
    ta = None

# Configuración avanzada del análisis
ADVANCED_CONFIG = {
    'symbol': 'BTC/USDT',  # Cambiado a símbolo real y verificado
    'timeframe': '4h',
    'limit': 200,  # Más datos para análisis profundo
    
    # Indicadores técnicos
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    
    'bb_period': 20,
    'bb_std': 2.0,
    
    'ema_periods': [9, 21, 50, 100, 200],
    'sma_periods': [20, 50, 100],
    
    # Momentum y volatilidad
    'atr_period': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'williams_r_period': 14,
    'cci_period': 20,
    'mfi_period': 14,
    
    # Volumen
    'volume_sma_periods': [10, 20, 50],
    'volume_profile_bins': 20,
    
    # Patrones y niveles
    'support_resistance_window': 20,
    'pattern_lookback': 10,
    'trend_strength_period': 50,
    
    # Scoring
    'weights': {
        'trend': 25,
        'momentum': 20,
        'volume': 15,
        'volatility': 10,
        'patterns': 15,
        'levels': 15
    }
}


class TrendDirection(Enum):
    STRONG_BULLISH = 5
    BULLISH = 4
    NEUTRAL_BULLISH = 3
    NEUTRAL = 2
    NEUTRAL_BEARISH = 1
    BEARISH = 0
    STRONG_BEARISH = -1


class SignalStrength(Enum):
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1


@dataclass
class AdvancedSignal:
    direction: str
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    holding_period_estimate: str
    key_levels: Dict[str, float]
    confirmations: List[str]
    warnings: List[str]


class ProfessionalCryptoAnalyzer:
    """Analizador profesional con análisis técnico avanzado."""
    
    def __init__(self, symbol: str = None, timeframe: str = None):
        # Configuración
        if symbol:
            ADVANCED_CONFIG['symbol'] = symbol
        if timeframe:
            ADVANCED_CONFIG['timeframe'] = timeframe
            
        # Exchange setup
        self.exchange = ccxt.binance({
            'apiKey': '', 'secret': '', 'sandbox': False,
            'enableRateLimit': True, 'timeout': 30000,
        })
        
        print(f"🔬 ANALIZADOR PROFESIONAL AVANZADO")
        print(f"📊 Símbolo: {ADVANCED_CONFIG['symbol']}")
        print(f"⏰ Timeframe: {ADVANCED_CONFIG['timeframe']}")
        print(f"📈 Indicadores: 20+ técnicos, patrones, niveles")
        print("-" * 60)
        
        self.verify_symbol()
    
    def verify_symbol(self):
        """Verificar y ajustar símbolo si es necesario."""
        try:
            markets = self.exchange.load_markets()
            symbol_ccxt = ADVANCED_CONFIG['symbol'].replace('USDT', '/USDT') if '/' not in ADVANCED_CONFIG['symbol'] else ADVANCED_CONFIG['symbol']
            
            if symbol_ccxt not in markets:
                print(f"❌ El símbolo {ADVANCED_CONFIG['symbol']} no existe")
                # Buscar similares
                base_symbol = ADVANCED_CONFIG['symbol'].replace('USDT', '').replace('/USDT', '')
                similar = [s for s in markets.keys() if base_symbol.lower() in s.lower() and 'USDT' in s]
                
                if similar:
                    suggested = similar[0]
                    print(f"🎯 Usando símbolo similar: {suggested}")
                    ADVANCED_CONFIG['symbol'] = suggested
                else:
                    raise ValueError(f"No se encontró {ADVANCED_CONFIG['symbol']}")
            else:
                ADVANCED_CONFIG['symbol'] = symbol_ccxt
                
        except Exception as e:
            print(f"❌ Error verificando símbolo: {e}")
            raise
    
    def fetch_enhanced_data(self) -> pd.DataFrame:
        """Obtener datos mejorados con información adicional."""
        try:
            print("📡 Obteniendo datos del mercado...")
            
            # Obtener más datos para análisis profundo
            ohlcv = self.exchange.fetch_ohlcv(
                ADVANCED_CONFIG['symbol'], 
                ADVANCED_CONFIG['timeframe'], 
                limit=ADVANCED_CONFIG['limit']
            )
            
            # Información adicional del mercado
            ticker = self.exchange.fetch_ticker(ADVANCED_CONFIG['symbol'])
            
            # Crear DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Agregar información del ticker
            df.attrs['ticker_info'] = {
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'spread': ticker.get('bid') and ticker.get('ask') and (ticker['ask'] - ticker['bid']),
                'volume_24h': ticker.get('quoteVolume'),
                'price_change_24h': ticker.get('change'),
                'price_change_24h_pct': ticker.get('percentage'),
                'vwap': ticker.get('vwap'),
                'last_update': ticker.get('timestamp')
            }
            
            print(f"✅ Datos obtenidos: {len(df)} registros")
            print(f"📅 Rango: {df.index[0]} a {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            raise
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores de tendencia completos."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        trend_indicators = {}
        
        # EMAs múltiples
        for period in ADVANCED_CONFIG['ema_periods']:
            if ta:
                ema = ta.EMA(close, timeperiod=period)
                trend_indicators[f'ema_{period}'] = ema[-1] if len(ema) > 0 else None
            else:
                ema = df['close'].ewm(span=period).mean()
                trend_indicators[f'ema_{period}'] = ema.iloc[-1]
        
        # SMAs múltiples
        for period in ADVANCED_CONFIG['sma_periods']:
            if ta:
                sma = ta.SMA(close, timeperiod=period)
                trend_indicators[f'sma_{period}'] = sma[-1] if len(sma) > 0 else None
            else:
                sma = df['close'].rolling(window=period).mean()
                trend_indicators[f'sma_{period}'] = sma.iloc[-1]
        
        # MACD con histograma
        if ta:
            macd_line, macd_signal, macd_hist = ta.MACD(close)
            trend_indicators['macd'] = {
                'line': macd_line[-1] if len(macd_line) > 0 else None,
                'signal': macd_signal[-1] if len(macd_signal) > 0 else None,
                'histogram': macd_hist[-1] if len(macd_hist) > 0 else None,
                'crossover': macd_line[-1] > macd_signal[-1] if len(macd_line) > 0 and len(macd_signal) > 0 else None
            }
        
        # ADX para fuerza de tendencia
        if ta:
            adx = ta.ADX(high, low, close, timeperiod=14)
            plus_di = ta.PLUS_DI(high, low, close, timeperiod=14)
            minus_di = ta.MINUS_DI(high, low, close, timeperiod=14)
            
            trend_indicators['adx'] = {
                'value': adx[-1] if len(adx) > 0 else None,
                'plus_di': plus_di[-1] if len(plus_di) > 0 else None,
                'minus_di': minus_di[-1] if len(minus_di) > 0 else None,
                'trend_strength': 'Strong' if adx[-1] > 25 else 'Weak' if len(adx) > 0 else 'Unknown'
            }
        
        # Parabolic SAR
        if ta:
            sar = ta.SAR(high, low)
            trend_indicators['parabolic_sar'] = {
                'value': sar[-1] if len(sar) > 0 else None,
                'signal': 'Bullish' if close[-1] > sar[-1] else 'Bearish' if len(sar) > 0 else 'Unknown'
            }
        
        return trend_indicators
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores de momentum avanzados."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        momentum_indicators = {}
        
        # RSI
        if ta:
            rsi = ta.RSI(close, timeperiod=ADVANCED_CONFIG['rsi_period'])
            momentum_indicators['rsi'] = {
                'value': rsi[-1] if len(rsi) > 0 else None,
                'condition': self._classify_rsi(rsi[-1] if len(rsi) > 0 else 50),
                'divergence': self._detect_rsi_divergence(df, rsi)
            }
        
        # Stochastic Oscillator
        if ta:
            slowk, slowd = ta.STOCH(high, low, close, 
                                   fastk_period=ADVANCED_CONFIG['stoch_k'],
                                   slowk_period=ADVANCED_CONFIG['stoch_d'])
            momentum_indicators['stochastic'] = {
                'k': slowk[-1] if len(slowk) > 0 else None,
                'd': slowd[-1] if len(slowd) > 0 else None,
                'signal': 'Bullish' if slowk[-1] > slowd[-1] else 'Bearish' if len(slowk) > 0 and len(slowd) > 0 else 'Unknown'
            }
        
        # Williams %R
        if ta:
            willr = ta.WILLR(high, low, close, timeperiod=ADVANCED_CONFIG['williams_r_period'])
            momentum_indicators['williams_r'] = {
                'value': willr[-1] if len(willr) > 0 else None,
                'condition': 'Oversold' if willr[-1] > -20 else 'Overbought' if willr[-1] < -80 else 'Neutral' if len(willr) > 0 else 'Unknown'
            }
        
        # CCI (Commodity Channel Index)
        if ta:
            cci = ta.CCI(high, low, close, timeperiod=ADVANCED_CONFIG['cci_period'])
            momentum_indicators['cci'] = {
                'value': cci[-1] if len(cci) > 0 else None,
                'condition': 'Overbought' if cci[-1] > 100 else 'Oversold' if cci[-1] < -100 else 'Neutral' if len(cci) > 0 else 'Unknown'
            }
        
        # MFI (Money Flow Index)
        if ta:
            mfi = ta.MFI(high, low, close, volume, timeperiod=ADVANCED_CONFIG['mfi_period'])
            momentum_indicators['mfi'] = {
                'value': mfi[-1] if len(mfi) > 0 else None,
                'condition': 'Overbought' if mfi[-1] > 80 else 'Oversold' if mfi[-1] < 20 else 'Neutral' if len(mfi) > 0 else 'Unknown'
            }
        
        return momentum_indicators
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores de volatilidad."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        volatility_indicators = {}
        
        # ATR (Average True Range)
        if ta:
            atr = ta.ATR(high, low, close, timeperiod=ADVANCED_CONFIG['atr_period'])
            atr_pct = (atr / close) * 100
            volatility_indicators['atr'] = {
                'value': atr[-1] if len(atr) > 0 else None,
                'percentage': atr_pct[-1] if len(atr_pct) > 0 else None,
                'volatility_level': self._classify_volatility(atr_pct[-1] if len(atr_pct) > 0 else 0)
            }
        
        # Bollinger Bands
        if ta:
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close, 
                                                     timeperiod=ADVANCED_CONFIG['bb_period'],
                                                     nbdevup=ADVANCED_CONFIG['bb_std'],
                                                     nbdevdn=ADVANCED_CONFIG['bb_std'])
            bb_width = ((bb_upper - bb_lower) / bb_middle) * 100
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            
            volatility_indicators['bollinger_bands'] = {
                'upper': bb_upper[-1] if len(bb_upper) > 0 else None,
                'middle': bb_middle[-1] if len(bb_middle) > 0 else None,
                'lower': bb_lower[-1] if len(bb_lower) > 0 else None,
                'width': bb_width[-1] if len(bb_width) > 0 else None,
                'position': bb_position[-1] if len(bb_position) > 0 else None,
                'squeeze': bb_width[-1] < bb_width[-20:].mean() * 0.8 if len(bb_width) >= 20 else False
            }
        
        # Keltner Channels
        if ta:
            ema = ta.EMA(close, timeperiod=20)
            atr_kc = ta.ATR(high, low, close, timeperiod=20)
            kc_upper = ema + (2 * atr_kc)
            kc_lower = ema - (2 * atr_kc)
            
            volatility_indicators['keltner_channels'] = {
                'upper': kc_upper[-1] if len(kc_upper) > 0 else None,
                'middle': ema[-1] if len(ema) > 0 else None,
                'lower': kc_lower[-1] if len(kc_lower) > 0 else None
            }
        
        return volatility_indicators
    
    def analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis avanzado del perfil de volumen."""
        volume_analysis = {}
        
        # Análisis básico de volumen
        volume = df['volume']
        close = df['close']
        
        # SMAs de volumen
        for period in ADVANCED_CONFIG['volume_sma_periods']:
            vol_sma = volume.rolling(window=period).mean()
            volume_analysis[f'volume_sma_{period}'] = vol_sma.iloc[-1]
        
        # Ratio de volumen actual vs promedio
        current_volume = volume.iloc[-1]
        avg_volume_20 = volume.rolling(window=20).mean().iloc[-1]
        volume_analysis['volume_ratio'] = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # OBV (On Balance Volume)
        if ta:
            obv = ta.OBV(close.values, volume.values)
            volume_analysis['obv'] = {
                'value': obv[-1] if len(obv) > 0 else None,
                'trend': 'Rising' if obv[-1] > obv[-5] else 'Falling' if len(obv) >= 5 else 'Neutral'
            }
        
        # Volume Price Trend (VPT)
        vpt = (close.pct_change() * volume).cumsum()
        volume_analysis['vpt'] = {
            'value': vpt.iloc[-1],
            'trend': 'Rising' if vpt.iloc[-1] > vpt.iloc[-5] else 'Falling' if len(vpt) >= 5 else 'Neutral'
        }
        
        # Acumulación/Distribución
        if ta:
            ad = ta.AD(df['high'].values, df['low'].values, close.values, volume.values)
            volume_analysis['accumulation_distribution'] = {
                'value': ad[-1] if len(ad) > 0 else None,
                'trend': 'Accumulation' if ad[-1] > ad[-10] else 'Distribution' if len(ad) >= 10 else 'Neutral'
            }
        
        # VWAP (Volume Weighted Average Price)
        vwap = (close * volume).cumsum() / volume.cumsum()
        volume_analysis['vwap'] = {
            'value': vwap.iloc[-1],
            'signal': 'Above VWAP' if close.iloc[-1] > vwap.iloc[-1] else 'Below VWAP'
        }
        
        return volume_analysis
    
    def detect_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detectar niveles de soporte y resistencia."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Detectar máximos y mínimos locales
        window = ADVANCED_CONFIG['support_resistance_window']
        
        # Resistencias (máximos locales)
        resistance_levels = []
        for i in range(window, len(high) - window):
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                resistance_levels.append({
                    'level': high.iloc[i],
                    'timestamp': high.index[i],
                    'touches': 0
                })
        
        # Soportes (mínimos locales)
        support_levels = []
        for i in range(window, len(low) - window):
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                support_levels.append({
                    'level': low.iloc[i],
                    'timestamp': low.index[i],
                    'touches': 0
                })
        
        # Contar toques en cada nivel
        current_price = close.iloc[-1]
        tolerance = current_price * 0.01  # 1% tolerance
        
        for level in resistance_levels:
            touches = sum(1 for price in high if abs(price - level['level']) <= tolerance)
            level['touches'] = touches
        
        for level in support_levels:
            touches = sum(1 for price in low if abs(price - level['level']) <= tolerance)
            level['touches'] = touches
        
        # Filtrar niveles más relevantes
        strong_resistance = [l for l in resistance_levels if l['touches'] >= 2]
        strong_support = [l for l in support_levels if l['touches'] >= 2]
        
        # Ordenar por proximidad al precio actual
        strong_resistance.sort(key=lambda x: abs(x['level'] - current_price))
        strong_support.sort(key=lambda x: abs(x['level'] - current_price))
        
        return {
            'resistance_levels': strong_resistance[:3],  # Top 3
            'support_levels': strong_support[:3],        # Top 3
            'nearest_resistance': strong_resistance[0]['level'] if strong_resistance else None,
            'nearest_support': strong_support[0]['level'] if strong_support else None,
            'distance_to_resistance': abs(current_price - strong_resistance[0]['level']) / current_price * 100 if strong_resistance else None,
            'distance_to_support': abs(current_price - strong_support[0]['level']) / current_price * 100 if strong_support else None
        }
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detectar patrones de velas japonesas."""
        if not ta:
            return {'patterns': [], 'note': 'TA-Lib no disponible para patrones'}
        
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        patterns = {}
        
        # Patrones bullish
        patterns['hammer'] = ta.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['doji'] = ta.CDLDOJI(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['bullish_engulfing'] = ta.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['morning_star'] = ta.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['piercing_pattern'] = ta.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)[-1]
        
        # Patrones bearish
        patterns['shooting_star'] = ta.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['bearish_engulfing'] = ta.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['evening_star'] = ta.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['dark_cloud'] = ta.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)[-1]
        
        # Filtrar patrones detectados
        detected_patterns = []
        for pattern_name, value in patterns.items():
            if value != 0:  # 0 = no pattern, 100 = bullish, -100 = bearish
                detected_patterns.append({
                    'pattern': pattern_name,
                    'strength': abs(value),
                    'direction': 'Bullish' if value > 0 else 'Bearish'
                })
        
        return {
            'detected_patterns': detected_patterns,
            'pattern_count': len(detected_patterns),
            'bullish_patterns': len([p for p in detected_patterns if p['direction'] == 'Bullish']),
            'bearish_patterns': len([p for p in detected_patterns if p['direction'] == 'Bearish'])
        }
    
    def calculate_advanced_score(self, 
                               trend_indicators: Dict,
                               momentum_indicators: Dict,
                               volatility_indicators: Dict,
                               volume_analysis: Dict,
                               levels: Dict,
                               patterns: Dict) -> Dict[str, Any]:
        """Calcular score avanzado multi-factor."""
        
        scores = {
            'trend': 0,
            'momentum': 0,
            'volume': 0,
            'volatility': 0,
            'patterns': 0,
            'levels': 0
        }
        
        # Score de tendencia
        trend_score = 0
        ema_signals = 0
        
        # Evaluar EMAs
        for i in range(len(ADVANCED_CONFIG['ema_periods']) - 1):
            ema_short = trend_indicators.get(f"ema_{ADVANCED_CONFIG['ema_periods'][i]}")
            ema_long = trend_indicators.get(f"ema_{ADVANCED_CONFIG['ema_periods'][i+1]}")
            
            if ema_short and ema_long:
                if ema_short > ema_long:
                    trend_score += 20
                else:
                    trend_score -= 20
                ema_signals += 1
        
        if ema_signals > 0:
            scores['trend'] = max(0, min(100, 50 + (trend_score / ema_signals)))
        
        # Score de momentum
        momentum_score = 50  # Base neutral
        
        rsi_val = momentum_indicators.get('rsi', {}).get('value')
        if rsi_val:
            if 40 <= rsi_val <= 60:
                momentum_score += 10
            elif 30 <= rsi_val <= 70:
                momentum_score += 5
            elif rsi_val < 30:
                momentum_score += 15  # Oversold bonus
            elif rsi_val > 70:
                momentum_score -= 15  # Overbought penalty
        
        stoch_signal = momentum_indicators.get('stochastic', {}).get('signal')
        if stoch_signal == 'Bullish':
            momentum_score += 10
        elif stoch_signal == 'Bearish':
            momentum_score -= 10
        
        scores['momentum'] = max(0, min(100, momentum_score))
        
        # Score de volumen
        volume_ratio = volume_analysis.get('volume_ratio', 1)
        obv_trend = volume_analysis.get('obv', {}).get('trend')
        
        volume_score = 50
        if volume_ratio > 1.5:
            volume_score += 25
        elif volume_ratio > 1.0:
            volume_score += 10
        elif volume_ratio < 0.5:
            volume_score -= 25
        
        if obv_trend == 'Rising':
            volume_score += 15
        elif obv_trend == 'Falling':
            volume_score -= 15
        
        scores['volume'] = max(0, min(100, volume_score))
        
        # Score de volatilidad
        volatility_score = 50
        atr_pct = volatility_indicators.get('atr', {}).get('percentage', 0)
        
        if 1 <= atr_pct <= 3:  # Volatilidad óptima
            volatility_score += 20
        elif atr_pct > 5:  # Muy volátil
            volatility_score -= 20
        
        bb_squeeze = volatility_indicators.get('bollinger_bands', {}).get('squeeze', False)
        if bb_squeeze:
            volatility_score += 15  # Squeeze puede preceder breakout
        
        scores['volatility'] = max(0, min(100, volatility_score))
        
        # Score de patrones
        pattern_score = 50
        bullish_patterns = patterns.get('bullish_patterns', 0)
        bearish_patterns = patterns.get('bearish_patterns', 0)
        
        pattern_score += (bullish_patterns * 15) - (bearish_patterns * 15)
        scores['patterns'] = max(0, min(100, pattern_score))
        
        # Score de niveles
        levels_score = 50
        distance_to_support = levels.get('distance_to_support')
        distance_to_resistance = levels.get('distance_to_resistance')
        
        if distance_to_support and distance_to_support < 2:  # Cerca del soporte
            levels_score += 20
        if distance_to_resistance and distance_to_resistance < 2:  # Cerca de resistencia
            levels_score -= 20
        
        scores['levels'] = max(0, min(100, levels_score))
        
        # Score total ponderado
        total_score = sum(scores[key] * ADVANCED_CONFIG['weights'][key] / 100 
                         for key in scores.keys())
        
        return {
            'individual_scores': scores,
            'total_score': total_score,
            'grade': self._score_to_grade(total_score),
            'recommendation': self._score_to_recommendation(total_score)
        }
    
    def _classify_rsi(self, rsi_value: float) -> str:
        """Clasificar valor RSI."""
        if rsi_value >= 70:
            return 'Overbought'
        elif rsi_value <= 30:
            return 'Oversold'
        elif 45 <= rsi_value <= 55:
            return 'Neutral'
        elif rsi_value > 55:
            return 'Bullish'
        else:
            return 'Bearish'
    
    def _classify_volatility(self, atr_pct: float) -> str:
        """Clasificar nivel de volatilidad."""
        if atr_pct < 1:
            return 'Very Low'
        elif atr_pct < 2:
            return 'Low'
        elif atr_pct < 4:
            return 'Normal'
        elif atr_pct < 6:
            return 'High'
        else:
            return 'Very High'
    
    def _detect_rsi_divergence(self, df: pd.DataFrame, rsi: np.ndarray) -> str:
        """Detectar divergencias en RSI."""
        if len(df) < 20 or len(rsi) < 20:
            return 'Insufficient data'
        
        # Simplificado: comparar últimos 10 períodos
        recent_prices = df['close'].iloc[-10:].values
        recent_rsi = rsi[-10:]
        
        price_trend = 'Rising' if recent_prices[-1] > recent_prices[0] else 'Falling'
        rsi_trend = 'Rising' if recent_rsi[-1] > recent_rsi[0] else 'Falling'
        
        if price_trend == 'Rising' and rsi_trend == 'Falling':
            return 'Bearish divergence'
        elif price_trend == 'Falling' and rsi_trend == 'Rising':
            return 'Bullish divergence'
        else:
            return 'No divergence'
    
    def _score_to_grade(self, score: float) -> str:
        """Convertir score a calificación."""
        if score >= 80:
            return 'A+'
        elif score >= 70:
            return 'A'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _score_to_recommendation(self, score: float) -> str:
        """Convertir score a recomendación."""
        if score >= 75:
            return 'STRONG BUY'
        elif score >= 65:
            return 'BUY'
        elif score >= 55:
            return 'WEAK BUY'
        elif score >= 45:
            return 'HOLD'
        elif score >= 35:
            return 'WEAK SELL'
        elif score >= 25:
            return 'SELL'
        else:
            return 'STRONG SELL'
    
    def generate_advanced_signal(self, 
                                df: pd.DataFrame,
                                trend_indicators: Dict,
                                momentum_indicators: Dict,
                                volatility_indicators: Dict,
                                volume_analysis: Dict,
                                levels: Dict,
                                scoring: Dict) -> AdvancedSignal:
        """Generar señal avanzada con análisis completo."""
        
        current_price = df['close'].iloc[-1]
        atr = volatility_indicators.get('atr', {}).get('value', current_price * 0.02)
        
        # Obtener niveles de soporte y resistencia
        nearest_support = levels.get('nearest_support', current_price * 0.95)
        nearest_resistance = levels.get('nearest_resistance', current_price * 1.05)
        
        # Determinar dirección basada en score y contexto técnico
        total_score = scoring['total_score']
        recommendation = scoring['recommendation']
        
        # Análisis de RSI para mejorar señales
        rsi_value = momentum_indicators.get('rsi', {}).get('value', 50)
        bb_position = volatility_indicators.get('bollinger_bands', {}).get('position', 0.5)
        
        # Lógica mejorada de señales
        if 'BUY' in recommendation or (rsi_value < 35 and bb_position < 0.3 and total_score > 40):
            direction = 'BUY'
            if total_score >= 75:
                strength = SignalStrength.VERY_STRONG
            elif total_score >= 65 or (rsi_value < 30 and bb_position < 0.2):
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
                
        elif 'SELL' in recommendation or (rsi_value > 65 and bb_position > 0.7 and total_score < 60):
            direction = 'SELL'
            if total_score <= 25:
                strength = SignalStrength.VERY_STRONG
            elif total_score <= 35 or (rsi_value > 70 and bb_position > 0.8):
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
        else:
            direction = 'HOLD'
            strength = SignalStrength.WEAK
        
        # Calcular niveles de gestión de riesgo mejorados
        entry_price = current_price
        
        if direction == 'BUY':
            # Para BUY: Stop loss debajo del soporte o ATR
            support_stop = nearest_support * 0.98 if nearest_support < current_price else current_price - (atr * 2)
            atr_stop = current_price - (atr * 1.5)
            stop_loss = max(support_stop, atr_stop)  # Usar el más conservador
            
            # Take profit hacia resistencia o múltiplo ATR
            resistance_target = nearest_resistance * 0.98 if nearest_resistance > current_price else current_price + (atr * 3)
            atr_target = current_price + (atr * 2.5)
            take_profit = min(resistance_target, atr_target)  # Usar el más realista
            
        elif direction == 'SELL':
            # Para SELL: Stop loss arriba de resistencia o ATR
            resistance_stop = nearest_resistance * 1.02 if nearest_resistance > current_price else current_price + (atr * 2)
            atr_stop = current_price + (atr * 1.5)
            stop_loss = min(resistance_stop, atr_stop)  # Usar el más conservador
            
            # Take profit hacia soporte o múltiplo ATR
            support_target = nearest_support * 1.02 if nearest_support < current_price else current_price - (atr * 3)
            atr_target = current_price - (atr * 2.5)
            take_profit = max(support_target, atr_target)  # Usar el más realista
            
        else:  # HOLD
            # Para HOLD: niveles observacionales basados en soportes/resistencias
            if rsi_value < 40:  # Preparado para posible compra
                stop_loss = nearest_support * 0.98 if nearest_support < current_price else current_price - (atr * 2)
                take_profit = nearest_resistance * 0.98 if nearest_resistance > current_price else current_price + (atr * 2)
            elif rsi_value > 60:  # Preparado para posible venta
                stop_loss = nearest_resistance * 1.02 if nearest_resistance > current_price else current_price + (atr * 2)
                take_profit = nearest_support * 1.02 if nearest_support < current_price else current_price - (atr * 2)
            else:  # Neutral
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 1.5)
        
        # Calcular ratio riesgo/recompensa
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Calcular confianza ajustada
        base_confidence = total_score / 100
        
        # Ajustar confianza basada en factores adicionales
        confidence_adjustments = 0
        
        # RSI extremo aumenta confianza
        if (direction == 'BUY' and rsi_value < 30) or (direction == 'SELL' and rsi_value > 70):
            confidence_adjustments += 0.15
        
        # Bollinger Bands extremos
        if (direction == 'BUY' and bb_position < 0.2) or (direction == 'SELL' and bb_position > 0.8):
            confidence_adjustments += 0.10
        
        # Volumen confirma el movimiento
        volume_ratio = volume_analysis.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            confidence_adjustments += 0.10
        elif volume_ratio < 0.7:
            confidence_adjustments -= 0.15
        
        # Proximidad a niveles clave
        if direction == 'BUY' and nearest_support and abs(current_price - nearest_support) / current_price < 0.03:
            confidence_adjustments += 0.10
        elif direction == 'SELL' and nearest_resistance and abs(current_price - nearest_resistance) / current_price < 0.03:
            confidence_adjustments += 0.10
        
        # Ajustar para HOLD
        if direction == 'HOLD':
            base_confidence = max(0.3, min(0.7, base_confidence))  # HOLD entre 30-70%
        
        final_confidence = max(0.1, min(0.95, base_confidence + confidence_adjustments))
        
        # Mejorar estimación de período de tenencia
        holding_period = self._estimate_holding_period_improved(
            strength, volatility_indicators, direction, rsi_value, total_score
        )
        # Recopilar confirmaciones
        confirmations = []
        warnings = []
        
        # Tendencia
        if scoring['individual_scores']['trend'] > 60:
            confirmations.append("Tendencia positiva confirmada")
        elif scoring['individual_scores']['trend'] < 40:
            warnings.append("Tendencia negativa")
        
        # Momentum
        rsi_condition = momentum_indicators.get('rsi', {}).get('condition')
        if rsi_condition == 'Oversold' and direction in ['BUY', 'HOLD']:
            confirmations.append("RSI oversold - oportunidad de rebote")
        elif rsi_condition == 'Overbought' and direction in ['SELL', 'HOLD']:
            confirmations.append("RSI overbought - presión de venta")
        
        # Volumen
        if volume_analysis.get('volume_ratio', 1) > 1.5:
            confirmations.append("Volumen elevado confirma movimiento")
        elif volume_analysis.get('volume_ratio', 1) < 0.7:
            warnings.append("Volumen bajo - falta confirmación")
        
        # Niveles
        if direction == 'BUY' and nearest_support and abs(current_price - nearest_support) / current_price < 0.05:
            confirmations.append("Precio cerca de soporte importante")
        elif direction == 'SELL' and nearest_resistance and abs(current_price - nearest_resistance) / current_price < 0.05:
            confirmations.append("Precio cerca de resistencia importante")
        elif direction == 'HOLD':
            if nearest_support and abs(current_price - nearest_support) / current_price < 0.05:
                confirmations.append("Precio testando soporte - observar reacción")
            elif nearest_resistance and abs(current_price - nearest_resistance) / current_price < 0.05:
                confirmations.append("Precio testando resistencia - observar reacción")
        
        # ATR y volatilidad
        atr_pct = volatility_indicators.get('atr', {}).get('percentage', 0)
        if atr_pct > 5:
            warnings.append("Alta volatilidad - gestión de riesgo estricta")
        elif atr_pct < 1:
            warnings.append("Baja volatilidad - movimientos limitados esperados")
        
        return AdvancedSignal(
            direction=direction,
            strength=strength,
            confidence=final_confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            holding_period_estimate=holding_period,
            key_levels={
                'support': nearest_support,
                'resistance': nearest_resistance,
                'vwap': volume_analysis.get('vwap', {}).get('value'),
                'atr': atr
            },
            confirmations=confirmations,
            warnings=warnings
        )
    
    def _estimate_holding_period(self, strength: SignalStrength, volatility_indicators: Dict) -> str:
        """Estimar período de tenencia recomendado."""
        atr_pct = volatility_indicators.get('atr', {}).get('percentage', 2)
        
        if strength in [SignalStrength.VERY_STRONG, SignalStrength.STRONG]:
            if atr_pct > 4:
                return "1-3 días (alta volatilidad)"
            else:
                return "3-7 días"
        elif strength == SignalStrength.MODERATE:
            return "1-2 días"
        else:
            return "Intraday"
    
    def _estimate_holding_period_improved(self, 
                                        strength: SignalStrength, 
                                        volatility_indicators: Dict,
                                        direction: str,
                                        rsi_value: float,
                                        score: float) -> str:
        """Estimar período de tenencia mejorado con más factores."""
        atr_pct = volatility_indicators.get('atr', {}).get('percentage', 2)
        
        # Base period por fuerza de señal
        if strength == SignalStrength.VERY_STRONG:
            base_days = 5
        elif strength == SignalStrength.STRONG:
            base_days = 3
        elif strength == SignalStrength.MODERATE:
            base_days = 2
        else:
            base_days = 1
        
        # Ajustar por volatilidad
        if atr_pct > 5:
            base_days = max(1, base_days - 1)
            volatility_note = " (alta volatilidad)"
        elif atr_pct < 1.5:
            base_days += 1
            volatility_note = " (baja volatilidad)"
        else:
            volatility_note = ""
        
        # Ajustar por condiciones de RSI
        if direction in ['BUY', 'HOLD'] and rsi_value < 25:
            base_days += 1  # Más tiempo para rebote desde oversold extremo
        elif direction in ['SELL', 'HOLD'] and rsi_value > 75:
            base_days = max(1, base_days - 1)  # Venta rápida desde overbought extremo
        
        # Ajustar por score
        if score > 80:
            base_days += 1
        elif score < 30:
            base_days = max(1, base_days - 1)
        
        # HOLD específico
        if direction == 'HOLD':
            if 25 <= rsi_value <= 35:
                return f"Observar 1-2 días (posible entrada BUY){volatility_note}"
            elif 65 <= rsi_value <= 75:
                return f"Observar 1-2 días (posible entrada SELL){volatility_note}"
            else:
                return f"Monitorear hasta señal clara{volatility_note}"
        
        # Formatear período final
        if base_days == 1:
            return f"Intraday{volatility_note}"
        elif base_days <= 3:
            return f"{base_days} días{volatility_note}"
        else:
            return f"{base_days}-{base_days+2} días{volatility_note}"
    
    def display_professional_analysis(self, 
                                    df: pd.DataFrame,
                                    trend_indicators: Dict,
                                    momentum_indicators: Dict,
                                    volatility_indicators: Dict,
                                    volume_analysis: Dict,
                                    levels: Dict,
                                    patterns: Dict,
                                    scoring: Dict,
                                    signal: AdvancedSignal):
        """Mostrar análisis profesional completo."""
        
        current_price = df['close'].iloc[-1]
        ticker_info = df.attrs.get('ticker_info', {})
        
        print("\n" + "="*80)
        print(f"🔬 ANÁLISIS TÉCNICO PROFESIONAL - {ADVANCED_CONFIG['symbol']}")
        print("="*80)
        
        # Información del mercado
        print(f"\n💹 INFORMACIÓN DEL MERCADO:")
        print(f"   Precio actual: ${current_price:.6f}")
        
        bid = ticker_info.get('bid')
        ask = ticker_info.get('ask')
        spread = ticker_info.get('spread')
        volume_24h = ticker_info.get('volume_24h')
        price_change_24h_pct = ticker_info.get('price_change_24h_pct')
        vwap_value = volume_analysis.get('vwap', {}).get('value')
        
        print(f"   Bid/Ask: ${bid:.6f} / ${ask:.6f}" if bid and ask else "   Bid/Ask: N/A")
        print(f"   Spread: ${spread:.6f}" if spread else "   Spread: N/A")
        print(f"   Volumen 24h: {volume_24h:,.0f}" if volume_24h else "   Volumen 24h: N/A")
        print(f"   Cambio 24h: {price_change_24h_pct:.2f}%" if price_change_24h_pct else "   Cambio 24h: N/A")
        print(f"   VWAP: ${vwap_value:.6f}" if vwap_value else "   VWAP: N/A")
        
        # Análisis de tendencia
        print(f"\n📈 ANÁLISIS DE TENDENCIA (Score: {scoring['individual_scores']['trend']:.0f}/100):")
        for period in [9, 21, 50]:
            ema_val = trend_indicators.get(f'ema_{period}')
            if ema_val:
                trend = "🟢" if current_price > ema_val else "🔴"
                print(f"   EMA {period}: {trend} ${ema_val:.6f}")
        
        adx_info = trend_indicators.get('adx', {})
        adx_value = adx_info.get('value')
        if adx_value:
            print(f"   ADX: {adx_value:.2f} ({adx_info.get('trend_strength', 'Unknown')})")
        
        # Análisis de momentum
        print(f"\n⚡ ANÁLISIS DE MOMENTUM (Score: {scoring['individual_scores']['momentum']:.0f}/100):")
        
        rsi_info = momentum_indicators.get('rsi', {})
        rsi_value = rsi_info.get('value')
        if rsi_value:
            print(f"   RSI: {rsi_value:.2f} ({rsi_info.get('condition', 'Unknown')})")
            divergence = rsi_info.get('divergence')
            if divergence and divergence != 'No divergence':
                print(f"   🔍 {divergence}")
        
        stoch_info = momentum_indicators.get('stochastic', {})
        stoch_k = stoch_info.get('k')
        stoch_d = stoch_info.get('d')
        if stoch_k and stoch_d:
            print(f"   Stochastic: %K={stoch_k:.2f}, %D={stoch_d:.2f} ({stoch_info.get('signal', 'Unknown')})")
        
        mfi_info = momentum_indicators.get('mfi', {})
        mfi_value = mfi_info.get('value')
        if mfi_value:
            print(f"   MFI: {mfi_value:.2f} ({mfi_info.get('condition', 'Unknown')})")
        
        # Análisis de volatilidad
        print(f"\n📊 ANÁLISIS DE VOLATILIDAD (Score: {scoring['individual_scores']['volatility']:.0f}/100):")
        
        atr_info = volatility_indicators.get('atr', {})
        atr_value = atr_info.get('value')
        atr_percentage = atr_info.get('percentage')
        atr_level = atr_info.get('volatility_level')
        if atr_value and atr_percentage and atr_level:
            print(f"   ATR: ${atr_value:.6f} ({atr_percentage:.2f}% - {atr_level})")
        
        bb_info = volatility_indicators.get('bollinger_bands', {})
        bb_position = bb_info.get('position')
        if bb_position is not None:
            bb_status = "Lower Band" if bb_position < 0.2 else "Upper Band" if bb_position > 0.8 else "Middle Range"
            print(f"   Bollinger: Posición {bb_position:.2f} ({bb_status})")
            if bb_info.get('squeeze'):
                print(f"   🔥 Bollinger Squeeze detectado - posible breakout")
        
        # Análisis de volumen
        print(f"\n📊 ANÁLISIS DE VOLUMEN (Score: {scoring['individual_scores']['volume']:.0f}/100):")
        volume_ratio = volume_analysis.get('volume_ratio', 1)
        print(f"   Ratio volumen: {volume_ratio:.2f}x")
        
        obv_info = volume_analysis.get('obv', {})
        obv_trend = obv_info.get('trend')
        if obv_trend:
            print(f"   OBV: {obv_trend}")
        
        ad_info = volume_analysis.get('accumulation_distribution', {})
        ad_trend = ad_info.get('trend')
        if ad_trend:
            print(f"   A/D Line: {ad_trend}")
        
        # Niveles clave
        print(f"\n🎯 NIVELES CLAVE (Score: {scoring['individual_scores']['levels']:.0f}/100):")
        nearest_support = levels.get('nearest_support')
        distance_support = levels.get('distance_to_support')
        if nearest_support and distance_support is not None:
            print(f"   Soporte más cercano: ${nearest_support:.6f} ({distance_support:.2f}% away)")
        
        nearest_resistance = levels.get('nearest_resistance')
        distance_resistance = levels.get('distance_to_resistance')
        if nearest_resistance and distance_resistance is not None:
            print(f"   Resistencia más cercana: ${nearest_resistance:.6f} ({distance_resistance:.2f}% away)")
        
        # Patrones de velas
        if patterns.get('detected_patterns'):
            print(f"\n🕯️ PATRONES DE VELAS (Score: {scoring['individual_scores']['patterns']:.0f}/100):")
            for pattern in patterns['detected_patterns']:
                direction_emoji = "🟢" if pattern['direction'] == 'Bullish' else "🔴"
                print(f"   {direction_emoji} {pattern['pattern'].replace('_', ' ').title()} ({pattern['direction']})")
        
        # Score total y recomendación
        print(f"\n🏆 EVALUACIÓN GENERAL:")
        print(f"   Score Total: {scoring['total_score']:.1f}/100 (Grado: {scoring['grade']})")
        print(f"   Recomendación: {scoring['recommendation']}")
        
        # Señal de trading
        print(f"\n⚡ SEÑAL DE TRADING:")
        signal_emoji = {"BUY": "🚀", "SELL": "📉", "HOLD": "⏸️"}.get(signal.direction, "❓")
        print(f"   {signal_emoji} {signal.direction} ({signal.strength.name})")
        print(f"   Confianza: {signal.confidence:.1%}")
        print(f"   Precio actual: ${signal.entry_price:.6f}")
        
        if signal.direction == 'HOLD':
            print(f"   📍 Niveles de observación:")
            print(f"      Stop observacional: ${signal.stop_loss:.6f} ({((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):+.2f}%)")
            print(f"      Target observacional: ${signal.take_profit:.6f} ({((signal.take_profit - signal.entry_price) / signal.entry_price * 100):+.2f}%)")
        else:
            print(f"   🛡️ Stop Loss: ${signal.stop_loss:.6f} ({((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):+.2f}%)")
            print(f"   🎯 Take Profit: ${signal.take_profit:.6f} ({((signal.take_profit - signal.entry_price) / signal.entry_price * 100):+.2f}%)")
        
        print(f"   📊 Ratio R/R: {signal.risk_reward_ratio:.2f}")
        print(f"   ⏱️ Período estimado: {signal.holding_period_estimate}")
        
        # Mostrar niveles clave
        if signal.key_levels.get('support') or signal.key_levels.get('resistance'):
            print(f"   🎯 Niveles clave:")
            if signal.key_levels.get('support'):
                support_dist = ((signal.key_levels['support'] - signal.entry_price) / signal.entry_price * 100)
                print(f"      Soporte: ${signal.key_levels['support']:.6f} ({support_dist:+.2f}%)")
            if signal.key_levels.get('resistance'):
                resistance_dist = ((signal.key_levels['resistance'] - signal.entry_price) / signal.entry_price * 100)
                print(f"      Resistencia: ${signal.key_levels['resistance']:.6f} ({resistance_dist:+.2f}%)")
            if signal.key_levels.get('vwap'):
                vwap_dist = ((signal.key_levels['vwap'] - signal.entry_price) / signal.entry_price * 100)
                print(f"      VWAP: ${signal.key_levels['vwap']:.6f} ({vwap_dist:+.2f}%)")
        
        # Confirmaciones
        if signal.confirmations:
            print(f"\n✅ CONFIRMACIONES:")
            for confirmation in signal.confirmations:
                print(f"   • {confirmation}")
        
        # Advertencias
        if signal.warnings:
            print(f"\n⚠️ ADVERTENCIAS:")
            for warning in signal.warnings:
                print(f"   • {warning}")
        
        print(f"\n" + "="*80)
        print(f"📋 RESUMEN EJECUTIVO:")
        print(f"   {ADVANCED_CONFIG['symbol']} | ${current_price:.6f} | {signal_emoji} {signal.direction} | Score: {scoring['total_score']:.0f}/100")
        print(f"   Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def save_professional_analysis(self, analysis_data: Dict):
        """Guardar análisis profesional completo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"professional_analysis_{ADVANCED_CONFIG['symbol'].replace('/', '_')}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Análisis profesional guardado: {filename}")
        except Exception as e:
            print(f"❌ Error guardando análisis: {e}")
    
    def run_professional_analysis(self, save_results: bool = False) -> Dict:
        """Ejecutar análisis profesional completo."""
        try:
            print("🔬 Iniciando análisis profesional avanzado...")
            
            # 1. Obtener datos mejorados
            df = self.fetch_enhanced_data()
            
            # 2. Calcular todos los indicadores
            print("📊 Calculando indicadores de tendencia...")
            trend_indicators = self.calculate_trend_indicators(df)
            
            print("⚡ Calculando indicadores de momentum...")
            momentum_indicators = self.calculate_momentum_indicators(df)
            
            print("📈 Calculando indicadores de volatilidad...")
            volatility_indicators = self.calculate_volatility_indicators(df)
            
            print("📊 Analizando perfil de volumen...")
            volume_analysis = self.analyze_volume_profile(df)
            
            print("🎯 Detectando niveles clave...")
            levels = self.detect_support_resistance(df)
            
            print("🕯️ Detectando patrones de velas...")
            patterns = self.detect_candlestick_patterns(df)
            
            print("🧮 Calculando scores avanzados...")
            scoring = self.calculate_advanced_score(
                trend_indicators, momentum_indicators, volatility_indicators,
                volume_analysis, levels, patterns
            )
            
            print("⚡ Generando señal avanzada...")
            signal = self.generate_advanced_signal(
                df, trend_indicators, momentum_indicators, volatility_indicators,
                volume_analysis, levels, scoring
            )
            
            # 3. Mostrar resultados
            self.display_professional_analysis(
                df, trend_indicators, momentum_indicators, volatility_indicators,
                volume_analysis, levels, patterns, scoring, signal
            )
            
            # 4. Preparar datos para guardado
            analysis_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': ADVANCED_CONFIG['symbol'],
                'timeframe': ADVANCED_CONFIG['timeframe'],
                'market_data': df.attrs.get('ticker_info', {}),
                'price_info': {
                    'current': float(df['close'].iloc[-1]),
                    'open': float(df['open'].iloc[-1]),
                    'high': float(df['high'].iloc[-1]),
                    'low': float(df['low'].iloc[-1]),
                    'volume': float(df['volume'].iloc[-1])
                },
                'technical_analysis': {
                    'trend_indicators': trend_indicators,
                    'momentum_indicators': momentum_indicators,
                    'volatility_indicators': volatility_indicators,
                    'volume_analysis': volume_analysis,
                    'key_levels': levels,
                    'candlestick_patterns': patterns
                },
                'scoring': scoring,
                'trading_signal': {
                    'direction': signal.direction,
                    'strength': signal.strength.name,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'risk_reward_ratio': signal.risk_reward_ratio,
                    'holding_period': signal.holding_period_estimate,
                    'key_levels': signal.key_levels,
                    'confirmations': signal.confirmations,
                    'warnings': signal.warnings
                }
            }
            
            # 5. Guardar si se solicita
            if save_results:
                self.save_professional_analysis(analysis_data)
            
            return analysis_data
            
        except Exception as e:
            print(f"❌ Error en análisis profesional: {e}")
            raise


def main():
    """Función principal del analizador profesional."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizador Profesional Avanzado de Crypto")
    parser.add_argument('--symbol', default='BTC/USDT', help='Símbolo a analizar')  # Cambiado por defecto
    parser.add_argument('--timeframe', default='4h', help='Timeframe')
    parser.add_argument('--save', action='store_true', help='Guardar análisis completo')
    
    args = parser.parse_args()
    
    try:
        analyzer = ProfessionalCryptoAnalyzer(args.symbol, args.timeframe)
        result = analyzer.run_professional_analysis(save_results=args.save)
        
        print(f"\n✅ Análisis profesional completado")
        
        return result
        
    except KeyboardInterrupt:
        print("\n🛑 Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()