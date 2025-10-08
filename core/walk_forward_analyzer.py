"""
Walk-Forward Analysis System for Robust Trading Strategy Validation
==================================================================

Sistema avanzado de validación temporal que evita overfitting mediante
ventanas deslizantes y validación out-of-sample continua.

Objetivo: Validar robustez de optimizaciones paramétricas en diferentes 
condiciones de mercado para amplificar retornos de 5.27% a 10-15% anual.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Add the current directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from futures_simulator import FuturesBacktester, FUTURES_CONFIG

@dataclass
class WalkForwardPeriod:
    """Representa un período de walk-forward analysis"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_days: int
    test_days: int

@dataclass
class WalkForwardResult:
    """Resultado de un período de walk-forward"""
    period: WalkForwardPeriod
    best_config: Dict
    train_performance: Dict
    test_performance: Dict
    out_of_sample_degradation: float

class WalkForwardAnalyzer:
    """
    Sistema de análisis Walk-Forward para validación robusta
    """
    
    def __init__(self, asset: str = 'ETH_USDT_4h'):
        self.asset = asset
        self.results_history = []
        
        # Configuraciones prometedoras del grid search previo
        self.promising_configs = [
            {
                'leverage': 45,
                'position_size_usd': 100,
                'max_positions': 3,
                'stop_loss_atr_mult': 2.4,
                'take_profit_atr_mult': 2.2,
                'min_buy_score': 50,
                'min_confidence': 0.45,
                'analysis_frequency': 4,
                'risk_free_rate': 0.02
            },
            {
                'leverage': 40,
                'position_size_usd': 100,
                'max_positions': 3,
                'stop_loss_atr_mult': 2.6,
                'take_profit_atr_mult': 2.0,
                'min_buy_score': 55,
                'min_confidence': 0.50,
                'analysis_frequency': 3,
                'risk_free_rate': 0.02
            },
            {
                'leverage': 35,
                'position_size_usd': 100,
                'max_positions': 3,
                'stop_loss_atr_mult': 2.2,
                'take_profit_atr_mult': 1.8,
                'min_buy_score': 60,
                'min_confidence': 0.55,
                'analysis_frequency': 5,
                'risk_free_rate': 0.02
            }
        ]
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """Cargar datos históricos para análisis"""
        parts = self.asset.split('_')
        symbol = '_'.join(parts[:-1])  # ETH_USDT
        timeframe = parts[-1]  # 4h
        
        cache_file = f'../data/cache_real/{self.asset}.json'
        if not os.path.exists(cache_file):
            print(f"❌ No se encontraron datos para {self.asset}")
            return None
            
        with open(cache_file, 'r') as f:
            data = json.load(f)
            
        # Convertir a DataFrame
        if 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data)
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"📊 Datos cargados: {len(df)} barras desde {df['timestamp'].min()} hasta {df['timestamp'].max()}")
        return df
    
    def create_walk_forward_periods(self, df: pd.DataFrame, 
                                   train_days: int = 120, 
                                   test_days: int = 30, 
                                   step_days: int = 15) -> List[WalkForwardPeriod]:
        """
        Crear períodos de walk-forward analysis
        
        Args:
            train_days: Días para entrenamiento/optimización
            test_days: Días para validación out-of-sample  
            step_days: Paso entre períodos (ventana deslizante)
        """
        periods = []
        total_days = len(df)
        
        # Calcular períodos disponibles
        current_start = 0
        
        while current_start + train_days + test_days <= total_days:
            train_start_idx = current_start
            train_end_idx = current_start + train_days - 1
            test_start_idx = current_start + train_days
            test_end_idx = current_start + train_days + test_days - 1
            
            period = WalkForwardPeriod(
                train_start=df.iloc[train_start_idx]['timestamp'],
                train_end=df.iloc[train_end_idx]['timestamp'],
                test_start=df.iloc[test_start_idx]['timestamp'],
                test_end=df.iloc[test_end_idx]['timestamp'],
                train_days=train_days,
                test_days=test_days
            )
            
            periods.append(period)
            current_start += step_days
            
        print(f"🔄 Creados {len(periods)} períodos de walk-forward")
        print(f"   📚 Entrenamiento: {train_days} días")
        print(f"   🧪 Validación: {test_days} días")
        print(f"   ⏭️  Paso: {step_days} días")
        
        return periods
    
    def optimize_period(self, period: WalkForwardPeriod) -> Tuple[Dict, Dict]:
        """
        Optimizar configuración para un período específico de entrenamiento
        """
        best_config = None
        best_performance = None
        best_score = -float('inf')
        
        # Probar configuraciones prometedoras
        for config in self.promising_configs:
            try:
                # Actualizar configuración global
                original_config = FUTURES_CONFIG.copy()
                FUTURES_CONFIG.clear()
                FUTURES_CONFIG.update(config)
                
                # Extraer símbolo y timeframe
                parts = self.asset.split('_')
                symbol = '_'.join(parts[:-1])
                timeframe = parts[-1]
                
                # Crear backtester
                backtester = FuturesBacktester(symbol, timeframe)
                
                # Ejecutar backtest en período de entrenamiento
                results = backtester.run_backtest(period.train_days)
                
                # Restaurar configuración
                FUTURES_CONFIG.clear()
                FUTURES_CONFIG.update(original_config)
                
                if results and 'summary' in results:
                    summary = results['summary']
                    
                    # Calcular score compuesto (risk-adjusted)
                    total_return = summary.get('total_return_percentage', 0)
                    sharpe = summary.get('sharpe_ratio', 0)
                    max_dd = abs(summary.get('max_drawdown_pct', 0))
                    trades = summary.get('total_operations', 0)
                    
                    # Score que balancea retorno, sharpe y control de riesgo
                    score = (total_return * sharpe) / (max_dd + 1) if trades >= 10 else -1000
                    
                    if score > best_score:
                        best_score = score
                        best_config = config.copy()
                        best_performance = {
                            'total_return_pct': total_return,
                            'sharpe_ratio': sharpe,
                            'max_drawdown_pct': max_dd,
                            'total_trades': trades,
                            'score': score
                        }
                        
            except Exception as e:
                print(f"❌ Error optimizando config: {e}")
                continue
        
        return best_config, best_performance
    
    def validate_out_of_sample(self, config: Dict, period: WalkForwardPeriod) -> Dict:
        """
        Validar configuración en período out-of-sample
        """
        try:
            # Actualizar configuración global
            original_config = FUTURES_CONFIG.copy()
            FUTURES_CONFIG.clear()
            FUTURES_CONFIG.update(config)
            
            # Extraer símbolo y timeframe
            parts = self.asset.split('_')
            symbol = '_'.join(parts[:-1])
            timeframe = parts[-1]
            
            # Crear backtester
            backtester = FuturesBacktester(symbol, timeframe)
            
            # Ejecutar backtest en período de validación
            results = backtester.run_backtest(period.test_days)
            
            # Restaurar configuración
            FUTURES_CONFIG.clear()
            FUTURES_CONFIG.update(original_config)
            
            if results and 'summary' in results:
                summary = results['summary']
                return {
                    'total_return_pct': summary.get('total_return_percentage', 0),
                    'sharpe_ratio': summary.get('sharpe_ratio', 0),
                    'max_drawdown_pct': abs(summary.get('max_drawdown_pct', 0)),
                    'total_trades': summary.get('total_operations', 0),
                    'profitable_ops': summary.get('profitable_operations', 0)
                }
            
        except Exception as e:
            print(f"❌ Error en validación out-of-sample: {e}")
            
        return {
            'total_return_pct': -100,
            'sharpe_ratio': -10,
            'max_drawdown_pct': 100,
            'total_trades': 0,
            'profitable_ops': 0
        }
    
    def run_walk_forward_analysis(self, train_days: int = 90, test_days: int = 30, step_days: int = 15) -> List[WalkForwardResult]:
        """
        Ejecutar análisis completo de walk-forward
        """
        print(f"🚀 INICIANDO WALK-FORWARD ANALYSIS")
        print(f"🎯 Objetivo: Validar robustez para amplificación x2-3")
        print("=" * 60)
        
        # Cargar datos
        df = self.load_data()
        if df is None:
            return []
            
        # Crear períodos
        periods = self.create_walk_forward_periods(df, train_days, test_days, step_days)
        
        if not periods:
            print("❌ No se pudieron crear períodos de análisis")
            return []
        
        results = []
        
        for i, period in enumerate(periods):
            print(f"\n🔄 PERÍODO {i+1}/{len(periods)}")
            print(f"   📚 Entrenamiento: {period.train_start.date()} a {period.train_end.date()}")
            print(f"   🧪 Validación: {period.test_start.date()} a {period.test_end.date()}")
            
            # Optimizar en período de entrenamiento
            print(f"   ⚙️  Optimizando configuración...")
            best_config, train_perf = self.optimize_period(period)
            
            if best_config is None:
                print(f"   ❌ No se encontró configuración válida")
                continue
            
            print(f"   ✅ Mejor config encontrada - Score: {train_perf['score']:.2f}")
            print(f"      📈 Train Return: {train_perf['total_return_pct']:.2f}%")
            print(f"      📊 Train Sharpe: {train_perf['sharpe_ratio']:.2f}")
            
            # Validar out-of-sample
            print(f"   🧪 Validando out-of-sample...")
            test_perf = self.validate_out_of_sample(best_config, period)
            
            # Calcular degradación
            degradation = ((train_perf['total_return_pct'] - test_perf['total_return_pct']) / 
                          abs(train_perf['total_return_pct'])) if train_perf['total_return_pct'] != 0 else 100
            
            print(f"   📊 Test Return: {test_perf['total_return_pct']:.2f}%")
            print(f"   📉 Degradación: {degradation:.1f}%")
            
            # Crear resultado
            wf_result = WalkForwardResult(
                period=period,
                best_config=best_config,
                train_performance=train_perf,
                test_performance=test_perf,
                out_of_sample_degradation=degradation
            )
            
            results.append(wf_result)
            
        return results
    
    def analyze_walk_forward_results(self, results: List[WalkForwardResult]) -> Dict:
        """
        Analizar resultados de walk-forward para extraer insights
        """
        if not results:
            return {}
            
        print(f"\n📊 ANÁLISIS DE RESULTADOS WALK-FORWARD")
        print("=" * 50)
        
        # Extraer métricas
        train_returns = [r.train_performance['total_return_pct'] for r in results]
        test_returns = [r.test_performance['total_return_pct'] for r in results]
        degradations = [r.out_of_sample_degradation for r in results]
        sharpe_ratios = [r.test_performance['sharpe_ratio'] for r in results]
        
        # Estadísticas agregadas
        stats = {
            'num_periods': len(results),
            'avg_train_return': np.mean(train_returns),
            'avg_test_return': np.mean(test_returns),
            'avg_degradation': np.mean(degradations),
            'avg_sharpe': np.mean(sharpe_ratios),
            'consistency_score': len([r for r in test_returns if r > 0]) / len(results),
            'stability_score': 1 - (np.std(test_returns) / abs(np.mean(test_returns))) if np.mean(test_returns) != 0 else 0
        }
        
        print(f"🔢 ESTADÍSTICAS AGREGADAS:")
        print(f"   📊 Períodos analizados: {stats['num_periods']}")
        print(f"   📈 Retorno promedio (train): {stats['avg_train_return']:.2f}%")
        print(f"   🧪 Retorno promedio (test): {stats['avg_test_return']:.2f}%")
        print(f"   📉 Degradación promedio: {stats['avg_degradation']:.1f}%")
        print(f"   📊 Sharpe promedio: {stats['avg_sharpe']:.2f}")
        print(f"   ✅ Consistencia: {stats['consistency_score']:.1%}")
        print(f"   🎯 Estabilidad: {stats['stability_score']:.2f}")
        
        # Evaluar robustez
        if stats['avg_test_return'] > 1.5:  # Mejor que baseline 1.35%
            if stats['consistency_score'] > 0.7 and stats['avg_degradation'] < 50:
                evaluation = "🏆 EXCELENTE - Estrategia robusta y superior"
            else:
                evaluation = "✅ BUENA - Prometedora pero necesita refinamiento"
        else:
            evaluation = "⚠️ REGULAR - Necesita optimización adicional"
        
        print(f"\n🏆 EVALUACIÓN GENERAL: {evaluation}")
        
        # Encontrar mejor configuración consistente
        valid_results = [r for r in results if r.test_performance['total_return_pct'] > 0]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x.test_performance['total_return_pct'])
            
            print(f"\n🎯 MEJOR CONFIGURACIÓN ENCONTRADA:")
            print(f"   📅 Período: {best_result.period.test_start.date()} a {best_result.period.test_end.date()}")
            print(f"   📈 Test Return: {best_result.test_performance['total_return_pct']:.2f}%")
            print(f"   📊 Test Sharpe: {best_result.test_performance['sharpe_ratio']:.2f}")
            print(f"   ⚙️  Leverage: {best_result.best_config['leverage']}")
            print(f"   🛡️  SL Mult: {best_result.best_config['stop_loss_atr_mult']}")
            print(f"   🎯 TP Mult: {best_result.best_config['take_profit_atr_mult']}")
            
            stats['best_config'] = best_result.best_config
            stats['best_performance'] = best_result.test_performance
        
        return stats
    
    def save_walk_forward_results(self, results: List[WalkForwardResult], analysis: Dict):
        """
        Guardar resultados del análisis walk-forward
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convertir resultados a formato serializable
        results_data = []
        for r in results:
            results_data.append({
                'period': {
                    'train_start': r.period.train_start.isoformat(),
                    'train_end': r.period.train_end.isoformat(),
                    'test_start': r.period.test_start.isoformat(),
                    'test_end': r.period.test_end.isoformat(),
                    'train_days': r.period.train_days,
                    'test_days': r.period.test_days
                },
                'best_config': r.best_config,
                'train_performance': r.train_performance,
                'test_performance': r.test_performance,
                'out_of_sample_degradation': r.out_of_sample_degradation
            })
        
        # Guardar resultados detallados
        results_file = f"walk_forward_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'asset': self.asset,
                'analysis_timestamp': timestamp,
                'results': results_data,
                'analysis': analysis
            }, f, indent=2)
        
        print(f"💾 Resultados guardados en: {results_file}")

def main():
    """
    Ejecutar análisis walk-forward completo
    """
    # Configuración del análisis
    asset = 'ETH_USDT_4h'  # Mejor asset según análisis previo
    
    # Inicializar analizador
    analyzer = WalkForwardAnalyzer(asset)
    
    # Ejecutar walk-forward analysis
    results = analyzer.run_walk_forward_analysis(
        train_days=90,   # 3 meses para optimización
        test_days=30,    # 1 mes para validación
        step_days=15     # Ventana deslizante cada 2 semanas
    )
    
    if results:
        # Analizar resultados
        analysis = analyzer.analyze_walk_forward_results(results)
        
        # Guardar resultados
        analyzer.save_walk_forward_results(results, analysis)
        
        print(f"\n🎉 WALK-FORWARD ANALYSIS COMPLETADO!")
        print(f"📊 {len(results)} períodos analizados")
        
        if 'best_config' in analysis:
            improvement = analysis['avg_test_return'] / 1.35  # vs baseline ETH 4H
            print(f"🚀 Factor de mejora: {improvement:.2f}x")
            
            if improvement >= 1.5:
                print(f"🎯 ¡META ALCANZADA! Amplificación superior a 1.5x")
            else:
                print(f"⚡ Progreso hacia meta de amplificación x2-3")
    else:
        print("❌ No se pudieron obtener resultados de walk-forward")

if __name__ == "__main__":
    main()