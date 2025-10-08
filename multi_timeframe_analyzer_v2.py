"""
Multi-Timeframe Analyzer V2 - Versión Optimizada
=================================================

CORRECCIONES BASADAS EN BACKTEST:
❌ Problema V1: 593 trades, Win Rate 33.9%, Over-trading
✅ Solución V2: Filtros más estrictos, confluencia real, menos ruido

NUEVA ESTRATEGIA OPTIMIZADA:
1. 4H: Filtros MÁS ESTRICTOS para tendencia (3/3 condiciones mínimo)
2. 1H: Confirmación ADICIONAL con múltiples confluencias
3. FILTRO GLOBAL: No trading en volatilidad extrema o mercado lateral
4. TIMING: Solo señales cada 4+ horas (evitar over-trading)

OBJETIVO: Reducir trades a ~30-50, Win Rate >50%, ROI >300%
"""

import requests
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar componentes existentes
from signal_tracker import SignalTracker

class MultiTimeframeAnalyzerV2:
    def __init__(self):
        self.config = {
            'timeframe_trend': '4h',      # Para análisis de tendencia
            'timeframe_entry': '1h',      # Para timing de entrada
            'limit_4h': 50,               # Datos suficientes para tendencia
            'limit_1h': 100,              # Datos suficientes para entrada
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'atr_period': 14,
            'stop_loss_atr_multiplier': 1.0,  # Más conservador
            'take_profit_atr_multiplier': 2.5, # Mejor R:R
            'risk_per_trade': 0.02,       # Más conservador
            'max_position_size': 0.20,    # Más conservador
            'min_signal_gap_hours': 4,    # Evitar over-trading
            'min_volatility_4h': 1.5,     # ATR mínimo 4H
            'max_volatility_4h': 8.0      # ATR máximo 4H (evitar extremos)
        }
        
        # Mapeo de timeframes para API de Binance
        self.binance_intervals = {
            '1h': '1h',
            '4h': '4h'
        }
        
        # Inicializar tracker de señales
        self.signal_tracker = SignalTracker()
        
        # Tracking de última señal (evitar over-trading)
        self.last_signal_time = None
    
    def get_binance_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Obtener datos de Binance para un timeframe específico."""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': self.binance_intervals[timeframe],
                'limit': limit
            }
            
            print(f"📡 Obteniendo datos {timeframe}: {symbol} ({limit} velas)...")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Error API Binance: {response.status_code}")
            
            data = response.json()
            
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos y timestamp
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Datos {timeframe} obtenidos: {len(df)} velas")
            print(f"📅 Última vela {timeframe}: {df.index[-1]}")
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"❌ Error obteniendo datos {timeframe}: {e}")
            raise
    
    def calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Calcular indicadores técnicos para un timeframe específico."""
        
        # EMA usando pandas
        ema_9 = df['close'].ewm(span=self.config['ema_fast']).mean().iloc[-1]
        ema_21 = df['close'].ewm(span=self.config['ema_slow']).mean().iloc[-1]
        
        # RSI
        close_prices = df['close']
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.config['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.config['rsi_period']).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # ATR (método Wilder)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr_values = []
        for i in range(1, len(df)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        # ATR usando smoothing de Wilder
        first_atr = np.mean(tr_values[:self.config['atr_period']])
        atr_current = first_atr
        
        for i in range(self.config['atr_period'], len(tr_values)):
            atr_current = (atr_current * (self.config['atr_period'] - 1) + tr_values[i]) / self.config['atr_period']
        
        # ATR como porcentaje del precio
        current_price = df['close'].iloc[-1]
        atr_percentage = (atr_current / current_price) * 100
        
        # Momentum (últimas 5 velas para más estabilidad)
        momentum = 0
        if len(close_prices) >= 5:
            momentum = (current_price - close_prices.iloc[-5]) / close_prices.iloc[-5] * 100
        
        # EMA trend strength (diferencia porcentual)
        ema_trend_strength = ((ema_9 - ema_21) / ema_21) * 100
        
        return {
            'timeframe': timeframe,
            'price': current_price,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'rsi': current_rsi,
            'atr': atr_current,
            'atr_percentage': atr_percentage,
            'momentum': momentum,
            'ema_trend_strength': ema_trend_strength
        }
    
    def analyze_trend_4h_strict(self, indicators_4h: dict) -> dict:
        """Analizar tendencia en 4H con filtros MÁS ESTRICTOS."""
        
        trend_analysis = {
            'direction': None,
            'strength': 0,
            'reasons': [],
            'conditions_met': 0,
            'quality_score': 0
        }
        
        price = indicators_4h['price']
        ema_9 = indicators_4h['ema_9']
        ema_21 = indicators_4h['ema_21']
        rsi = indicators_4h['rsi']
        atr_pct = indicators_4h['atr_percentage']
        momentum = indicators_4h['momentum']
        ema_trend = indicators_4h['ema_trend_strength']
        
        print(f"\\n🔍 ANÁLISIS TENDENCIA 4H (MODO ESTRICTO):")
        print(f"💰 Precio: ${price:.4f}")
        print(f"📈 EMA9: ${ema_9:.4f} | EMA21: ${ema_21:.4f}")
        print(f"⚡ RSI: {rsi:.1f} | ATR: {atr_pct:.2f}%")
        print(f"🚀 Momentum 5-velas: {momentum:.2f}%")
        print(f"📊 EMA Trend Strength: {ema_trend:.2f}%")
        
        # FILTRO GLOBAL: Volatilidad adecuada
        if not (self.config['min_volatility_4h'] <= atr_pct <= self.config['max_volatility_4h']):
            trend_analysis['reasons'].append(f"❌ Volatilidad fuera de rango ({atr_pct:.2f}%)")
            print(f"⚠️ RECHAZADO: Volatilidad {atr_pct:.2f}% fuera de rango [{self.config['min_volatility_4h']}-{self.config['max_volatility_4h']}%]")
            return trend_analysis
        
        # Condiciones para TENDENCIA ALCISTA (MÁS ESTRICTAS)
        bullish_conditions = 0
        bullish_reasons = []
        
        # 1. EMA más estricto
        if ema_9 > ema_21 and ema_trend > 0.5:  # Mínimo 0.5% separación
            bullish_conditions += 1
            bullish_reasons.append(f"✅ EMA alcista fuerte (trend: {ema_trend:.2f}%)")
        else:
            bullish_reasons.append(f"❌ EMA trend débil ({ema_trend:.2f}%)")
        
        # 2. RSI en zona óptima (más restrictivo)
        if 35 <= rsi <= 65:  # Zona neutral-optimista
            bullish_conditions += 1
            bullish_reasons.append(f"✅ RSI zona óptima ({rsi:.1f})")
        else:
            bullish_reasons.append(f"❌ RSI fuera zona óptima ({rsi:.1f})")
        
        # 3. Momentum positivo consistente
        if momentum > 1.0:  # Mínimo 1% momentum positivo
            bullish_conditions += 1
            bullish_reasons.append(f"✅ Momentum positivo ({momentum:.2f}%)")
        else:
            bullish_reasons.append(f"❌ Momentum insuficiente ({momentum:.2f}%)")
        
        # Condiciones para TENDENCIA BAJISTA (MÁS ESTRICTAS)
        bearish_conditions = 0
        bearish_reasons = []
        
        # 1. EMA más estricto
        if ema_9 < ema_21 and ema_trend < -0.5:  # Mínimo -0.5% separación
            bearish_conditions += 1
            bearish_reasons.append(f"✅ EMA bajista fuerte (trend: {ema_trend:.2f}%)")
        else:
            bearish_reasons.append(f"❌ EMA trend débil ({ema_trend:.2f}%)")
        
        # 2. RSI en zona óptima (más restrictivo)
        if 35 <= rsi <= 65:  # Zona neutral-pesimista
            bearish_conditions += 1
            bearish_reasons.append(f"✅ RSI zona óptima ({rsi:.1f})")
        else:
            bearish_reasons.append(f"❌ RSI fuera zona óptima ({rsi:.1f})")
        
        # 3. Momentum negativo consistente
        if momentum < -1.0:  # Mínimo -1% momentum negativo
            bearish_conditions += 1
            bearish_reasons.append(f"✅ Momentum negativo ({momentum:.2f}%)")
        else:
            bearish_reasons.append(f"❌ Momentum insuficiente ({momentum:.2f}%)")
        
        # Determinar tendencia (REQUIERE 3/3 condiciones)
        if bullish_conditions >= 3:
            trend_analysis['direction'] = 'BULLISH'
            trend_analysis['strength'] = bullish_conditions * 33
            trend_analysis['reasons'] = bullish_reasons
            trend_analysis['conditions_met'] = bullish_conditions
            trend_analysis['quality_score'] = min(100, bullish_conditions * 25 + abs(ema_trend) * 10)
            print(f"📈 TENDENCIA 4H: ALCISTA CONFIRMADA ({bullish_conditions}/3)")
            
        elif bearish_conditions >= 3:
            trend_analysis['direction'] = 'BEARISH'
            trend_analysis['strength'] = bearish_conditions * 33
            trend_analysis['reasons'] = bearish_reasons
            trend_analysis['conditions_met'] = bearish_conditions
            trend_analysis['quality_score'] = min(100, bearish_conditions * 25 + abs(ema_trend) * 10)
            print(f"📉 TENDENCIA 4H: BAJISTA CONFIRMADA ({bearish_conditions}/3)")
            
        else:
            print(f"😐 TENDENCIA 4H: NO CONFIRMADA (Bull:{bullish_conditions}/3, Bear:{bearish_conditions}/3)")
            trend_analysis['reasons'] = bullish_reasons + bearish_reasons
        
        return trend_analysis
    
    def analyze_entry_1h_strict(self, indicators_1h: dict, trend_direction: str, trend_quality: int) -> dict:
        """Analizar condiciones de entrada en 1H con confluencias MÚLTIPLES."""
        
        entry_analysis = {
            'signal': None,
            'strength': 0,
            'reasons': [],
            'conditions_met': 0,
            'confluence_score': 0
        }
        
        if not trend_direction:
            print(f"\\n⏳ SIN ANÁLISIS 1H: Tendencia 4H no confirmada")
            return entry_analysis
        
        price = indicators_1h['price']
        ema_9 = indicators_1h['ema_9']
        ema_21 = indicators_1h['ema_21']
        rsi = indicators_1h['rsi']
        atr_pct = indicators_1h['atr_percentage']
        momentum = indicators_1h['momentum']
        ema_trend = indicators_1h['ema_trend_strength']
        
        print(f"\\n🎯 ANÁLISIS ENTRADA 1H (Tendencia {trend_direction}, Quality: {trend_quality}):")
        print(f"💰 Precio: ${price:.4f}")
        print(f"📈 EMA9: ${ema_9:.4f} | EMA21: ${ema_21:.4f}")
        print(f"⚡ RSI: {rsi:.1f} | ATR: {atr_pct:.2f}%")
        print(f"🚀 Momentum: {momentum:.2f}%")
        print(f"📊 EMA Trend 1H: {ema_trend:.2f}%")
        
        # Verificar gap de tiempo (evitar over-trading)
        if self.last_signal_time:
            time_diff = datetime.now() - self.last_signal_time
            hours_diff = time_diff.total_seconds() / 3600
            if hours_diff < self.config['min_signal_gap_hours']:
                entry_analysis['reasons'].append(f"❌ Muy pronto desde última señal ({hours_diff:.1f}h)")
                print(f"⏳ RECHAZADO: Última señal hace {hours_diff:.1f}h (mín: {self.config['min_signal_gap_hours']}h)")
                return entry_analysis
        
        if trend_direction == 'BULLISH':
            # Condiciones de entrada LONG (MÁS SELECTIVAS)
            long_conditions = 0
            long_reasons = []
            confluence_factors = []
            
            # 1. RSI en zona de entrada ÓPTIMA
            if 25 <= rsi <= 45:  # Zona más específica
                long_conditions += 1
                confluence_factors.append(f"RSI entrada ({rsi:.1f})")
                long_reasons.append(f"✅ RSI zona entrada óptima ({rsi:.1f})")
            else:
                long_reasons.append(f"❌ RSI fuera zona entrada ({rsi:.1f})")
            
            # 2. Momentum controlado Y positivo
            if -2.0 <= momentum <= 3.0:  # Rango más estricto
                long_conditions += 1
                confluence_factors.append(f"Momentum ({momentum:.2f}%)")
                long_reasons.append(f"✅ Momentum controlado ({momentum:.2f}%)")
            else:
                long_reasons.append(f"❌ Momentum extremo ({momentum:.2f}%)")
            
            # 3. EMA 1H alineado con 4H (nueva confluencia)
            if ema_trend > -1.0:  # EMA 1H no muy bearish
                long_conditions += 1
                confluence_factors.append(f"EMA alineación ({ema_trend:.2f}%)")
                long_reasons.append(f"✅ EMA 1H alineado ({ema_trend:.2f}%)")
            else:
                long_reasons.append(f"❌ EMA 1H desalineado ({ema_trend:.2f}%)")
            
            # 4. Volatilidad adecuada (nueva condición)
            if 0.8 <= atr_pct <= 6.0:  # Rango más específico
                long_conditions += 1
                confluence_factors.append(f"Volatilidad ({atr_pct:.2f}%)")
                long_reasons.append(f"✅ Volatilidad adecuada ({atr_pct:.2f}%)")
            else:
                long_reasons.append(f"❌ Volatilidad inadecuada ({atr_pct:.2f}%)")
            
            # Generar señal LONG (REQUIERE 4/4 condiciones + calidad 4H alta)
            if long_conditions >= 4 and trend_quality >= 75:
                entry_analysis['signal'] = 'LONG'
                entry_analysis['strength'] = long_conditions * 25
                entry_analysis['reasons'] = long_reasons
                entry_analysis['conditions_met'] = long_conditions
                entry_analysis['confluence_score'] = len(confluence_factors) * 20
                self.last_signal_time = datetime.now()
                print(f"🚨 SEÑAL 1H: LONG CONFIRMADA ({long_conditions}/4) [Confluencias: {', '.join(confluence_factors)}]")
            else:
                print(f"⏳ Sin señal LONG ({long_conditions}/4, Quality 4H: {trend_quality})")
                entry_analysis['reasons'] = long_reasons
        
        elif trend_direction == 'BEARISH':
            # Condiciones de entrada SHORT (MÁS SELECTIVAS)
            short_conditions = 0
            short_reasons = []
            confluence_factors = []
            
            # 1. RSI en zona de entrada ÓPTIMA
            if 55 <= rsi <= 75:  # Zona más específica
                short_conditions += 1
                confluence_factors.append(f"RSI entrada ({rsi:.1f})")
                short_reasons.append(f"✅ RSI zona entrada óptima ({rsi:.1f})")
            else:
                short_reasons.append(f"❌ RSI fuera zona entrada ({rsi:.1f})")
            
            # 2. Momentum controlado Y negativo
            if -3.0 <= momentum <= 2.0:  # Rango más estricto
                short_conditions += 1
                confluence_factors.append(f"Momentum ({momentum:.2f}%)")
                short_reasons.append(f"✅ Momentum controlado ({momentum:.2f}%)")
            else:
                short_reasons.append(f"❌ Momentum extremo ({momentum:.2f}%)")
            
            # 3. EMA 1H alineado con 4H (nueva confluencia)
            if ema_trend < 1.0:  # EMA 1H no muy bullish
                short_conditions += 1
                confluence_factors.append(f"EMA alineación ({ema_trend:.2f}%)")
                short_reasons.append(f"✅ EMA 1H alineado ({ema_trend:.2f}%)")
            else:
                short_reasons.append(f"❌ EMA 1H desalineado ({ema_trend:.2f}%)")
            
            # 4. Volatilidad adecuada (nueva condición)
            if 0.8 <= atr_pct <= 6.0:  # Rango más específico
                short_conditions += 1
                confluence_factors.append(f"Volatilidad ({atr_pct:.2f}%)")
                short_reasons.append(f"✅ Volatilidad adecuada ({atr_pct:.2f}%)")
            else:
                short_reasons.append(f"❌ Volatilidad inadecuada ({atr_pct:.2f}%)")
            
            # Generar señal SHORT (REQUIERE 4/4 condiciones + calidad 4H alta)
            if short_conditions >= 4 and trend_quality >= 75:
                entry_analysis['signal'] = 'SHORT'
                entry_analysis['strength'] = short_conditions * 25
                entry_analysis['reasons'] = short_reasons
                entry_analysis['conditions_met'] = short_conditions
                entry_analysis['confluence_score'] = len(confluence_factors) * 20
                self.last_signal_time = datetime.now()
                print(f"🚨 SEÑAL 1H: SHORT CONFIRMADA ({short_conditions}/4) [Confluencias: {', '.join(confluence_factors)}]")
            else:
                print(f"⏳ Sin señal SHORT ({short_conditions}/4, Quality 4H: {trend_quality})")
                entry_analysis['reasons'] = short_reasons
        
        return entry_analysis
    
    def analyze_signal(self, symbol: str) -> dict:
        """Análisis completo multi-timeframe optimizado."""
        
        # 1. Obtener datos de ambos timeframes
        df_4h = self.get_binance_data(symbol, '4h', self.config['limit_4h'])
        df_1h = self.get_binance_data(symbol, '1h', self.config['limit_1h'])
        
        # 2. Calcular indicadores
        indicators_4h = self.calculate_indicators(df_4h, '4h')
        indicators_1h = self.calculate_indicators(df_1h, '1h')
        
        # 3. Analizar tendencia en 4H (modo estricto)
        trend_analysis = self.analyze_trend_4h_strict(indicators_4h)
        
        # 4. Analizar entrada en 1H según tendencia 4H (modo estricto)
        entry_analysis = self.analyze_entry_1h_strict(
            indicators_1h, 
            trend_analysis['direction'],
            trend_analysis['quality_score']
        )
        
        # 5. Compilar resultado final
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': indicators_1h['price'],
            'signal': entry_analysis['signal'],
            'strength': entry_analysis['strength'],
            'reasons': entry_analysis['reasons'],
            'trend_4h': trend_analysis,
            'entry_1h': entry_analysis,
            'indicators_4h': indicators_4h,
            'indicators_1h': indicators_1h,
            'quality_metrics': {
                'trend_quality_4h': trend_analysis['quality_score'],
                'confluence_score_1h': entry_analysis['confluence_score'],
                'overall_quality': (trend_analysis['quality_score'] + entry_analysis['confluence_score']) / 2
            }
        }
        
        # 6. Calcular niveles de trading si hay señal
        if result['signal']:
            # Usar ATR 4H para niveles (más estables)
            atr_4h = indicators_4h['atr']
            current_price = indicators_1h['price']
            
            stop_loss_distance = atr_4h * self.config['stop_loss_atr_multiplier']
            
            if result['signal'] == 'LONG':
                stop_loss = current_price - stop_loss_distance
                take_profit = current_price + (stop_loss_distance * self.config['take_profit_atr_multiplier'])
            else:  # SHORT
                stop_loss = current_price + stop_loss_distance
                take_profit = current_price - (stop_loss_distance * self.config['take_profit_atr_multiplier'])
            
            # Risk management conservador
            balance = 10000  # Balance ejemplo
            risk_amount = balance * self.config['risk_per_trade']
            position_size = min(
                risk_amount / stop_loss_distance,
                balance * self.config['max_position_size'] / current_price
            )
            
            result.update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'atr_4h_used': atr_4h,
                'risk_reward_ratio': abs(take_profit - current_price) / abs(stop_loss - current_price)
            })
        
        return result
    
    def print_signal_report(self, result: dict):
        """Imprimir reporte completo multi-timeframe optimizado."""
        
        print(f"\\n" + "="*80)
        print(f"🎯 ANÁLISIS MULTI-TIMEFRAME V2 (OPTIMIZADO) - {result['symbol']}")
        print(f"📅 {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Precio actual: ${result['price']:.4f}")
        print("="*80)
        
        # Mostrar métricas de calidad
        quality = result['quality_metrics']
        print(f"\\n📊 MÉTRICAS DE CALIDAD:")
        print(f"   🎯 Calidad Tendencia 4H: {quality['trend_quality_4h']:.0f}/100")
        print(f"   🔗 Score Confluencia 1H: {quality['confluence_score_1h']:.0f}/100")
        print(f"   ⭐ Calidad General: {quality['overall_quality']:.0f}/100")
        
        # Mostrar análisis de tendencia 4H
        trend = result['trend_4h']
        print(f"\\n📊 TENDENCIA 4H (MODO ESTRICTO):")
        if trend['direction']:
            print(f"   Dirección: {trend['direction']} ({trend['strength']}%)")
            print(f"   Condiciones: {trend['conditions_met']}/3 ✅")
            print(f"   Calidad: {trend['quality_score']:.0f}/100")
        else:
            print(f"   Dirección: NO CONFIRMADA ❌")
            print(f"   Razón: Filtros estrictos no cumplidos")
        
        # Mostrar análisis de entrada 1H
        entry = result['entry_1h']
        print(f"\\n🎯 ENTRADA 1H (CONFLUENCIAS MÚLTIPLES):")
        if entry['signal']:
            print(f"   Señal: {entry['signal']} ({entry['strength']}%)")
            print(f"   Condiciones: {entry['conditions_met']}/4 ✅")
            print(f"   Confluencias: {entry['confluence_score']:.0f}/100")
        else:
            print(f"   Señal: NO CONFIRMADA ❌")
            print(f"   Razón: Confluencias insuficientes")
        
        # Resultado final
        if result['signal']:
            print(f"\\n🚨 SEÑAL FINAL: {result['signal']} ⭐")
            print(f"💪 Fuerza combinada: {result['strength']}%")
            print(f"📈 Calidad general: {quality['overall_quality']:.0f}/100")
            print(f"🎯 Stop Loss: ${result['stop_loss']:.4f}")
            print(f"🏆 Take Profit: ${result['take_profit']:.4f}")
            print(f"⚖️ Risk:Reward = 1:{result['risk_reward_ratio']:.1f}")
            print(f"📦 Tamaño posición: {result['position_size']:.4f} {result['symbol'].replace('USDT', '')}")
            print(f"⚠️ Riesgo: ${result['risk_amount']:.2f} ({result['risk_amount']/10000*100:.1f}%)")
            
            print(f"\\n📋 CONFLUENCIAS CONFIRMADAS:")
            for reason in result['reasons']:
                if "✅" in reason:
                    print(f"   {reason}")
            
            # Guardar señal
            try:
                signal_id = self.signal_tracker.save_signal(result)
                print(f"\\n💾 Señal guardada para tracking: {signal_id}")
            except Exception as e:
                print(f"\\n⚠️ Error guardando señal: {e}")
                    
        else:
            print(f"\\n⏳ SIN SEÑAL MULTI-TIMEFRAME V2")
            print(f"🔍 Esperando confluencias perfectas...")
            print(f"\\n💡 MEJORAS V2:")
            print(f"   ✅ Filtros más estrictos (3/3 + 4/4 requeridos)")
            print(f"   ✅ Control de volatilidad")
            print(f"   ✅ Gap temporal anti-overtrading")
            print(f"   ✅ Score de confluencias")
            print(f"   ✅ Risk:Reward optimizado")
        
        print("="*80)


def main():
    """Función principal para análisis multi-timeframe optimizado."""
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='TradeBot Multi-Timeframe Analyzer V2 (Optimizado)')
    parser.add_argument('--symbol', '-s',
                       default='LINKUSDT',
                       help='Símbolo a analizar (default: LINKUSDT)')
    
    # Si hay argumentos de línea de comandos, usarlos
    if len(sys.argv) > 1:
        args = parser.parse_args()
        symbol = args.symbol
        
        print(f"🚀 ANALIZADOR MULTI-TIMEFRAME V2 (OPTIMIZADO)")
        print(f"🔧 Modo línea de comandos:")
        print(f"   Símbolo: {symbol}")
        
    else:
        # Modo interactivo
        print(f"🚀 ANALIZADOR MULTI-TIMEFRAME V2 (OPTIMIZADO)")
        print(f"🎯 Estrategia: 4H confirmación + 1H confluencias múltiples")
        
        symbol_input = input("💰 Símbolo (enter para LINKUSDT): ").strip().upper()
        symbol = symbol_input or "LINKUSDT"
        
        if symbol.endswith('/USDT'):
            symbol = symbol.replace('/', '')  # Convertir BTC/USDT → BTCUSDT
    
    print(f"🎯 Objetivo V2: Menos trades, mayor precisión, mejor R:R")
    print(f"🔧 Filtros: 3/3 (4H) + 4/4 (1H) + Confluencias + Gap temporal")
    
    try:
        # Crear analyzer multi-timeframe optimizado
        analyzer = MultiTimeframeAnalyzerV2()
        
        # Analizar señal
        result = analyzer.analyze_signal(symbol)
        
        # Mostrar reporte
        analyzer.print_signal_report(result)
        
    except Exception as e:
        print(f"❌ Error en análisis multi-timeframe V2: {e}")


if __name__ == "__main__":
    main()