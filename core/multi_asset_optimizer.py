"""
Advanced Multi-Asset Portfolio Optimizer
========================================

Sistema de optimización inteligente que combina ETH, SOL y BTC para maximizar retornos
mientras mantiene control de riesgo. Basado en los hallazgos del grid search previo.

Objetivo: Amplificar retornos de 5.27% baseline a 15-25% anual mediante diversificación 
inteligente y asignación dinámica de capital.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import itertools

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from futures_simulator import FuturesBacktester, FUTURES_CONFIG

class MultiAssetOptimizer:
    """
    Optimizador de portfolio multi-asset con asignación inteligente de capital
    """
    
    def __init__(self):
        # Assets validados con performance conocida
        self.asset_universe = {
            'ETH_USDT_1d': {'baseline_return': 2.60, 'baseline_sharpe': 6.38, 'tier': 'premium'},
            'SOL_USDT_1d': {'baseline_return': 1.77, 'baseline_sharpe': 4.03, 'tier': 'high'},
            'ETH_USDT_4h': {'baseline_return': 1.35, 'baseline_sharpe': 3.25, 'tier': 'medium'},
            'BTC_USDT_1d': {'baseline_return': 0.77, 'baseline_sharpe': 3.51, 'tier': 'stable'},
        }
        
        # Configuraciones optimizadas por leverage (de grid search previo)
        self.optimized_configs = {
            45: {
                'leverage': 45,
                'position_size_usd': 100,
                'max_positions': 3,
                'stop_loss_atr_mult': 2.2,
                'take_profit_atr_mult': 2.5,
                'min_buy_score': 50,
                'min_confidence': 0.45,
                'analysis_frequency': 4,
                'initial_balance': 5000,
                'risk_free_rate': 0.02
            },
            40: {
                'leverage': 40,
                'position_size_usd': 100,
                'max_positions': 3,
                'stop_loss_atr_mult': 2.4,
                'take_profit_atr_mult': 2.2,
                'min_buy_score': 55,
                'min_confidence': 0.50,
                'analysis_frequency': 3,
                'initial_balance': 5000,
                'risk_free_rate': 0.02
            },
            35: {
                'leverage': 35,
                'position_size_usd': 100,
                'max_positions': 3,
                'stop_loss_atr_mult': 2.6,
                'take_profit_atr_mult': 2.0,
                'min_buy_score': 60,
                'min_confidence': 0.55,
                'analysis_frequency': 5,
                'initial_balance': 5000,
                'risk_free_rate': 0.02
            }
        }
        
        # Estrategias de asignación de capital
        self.allocation_strategies = {
            'conservative': {'ETH_USDT_1d': 0.5, 'SOL_USDT_1d': 0.3, 'BTC_USDT_1d': 0.2},
            'balanced': {'ETH_USDT_1d': 0.4, 'SOL_USDT_1d': 0.4, 'ETH_USDT_4h': 0.2},
            'aggressive': {'ETH_USDT_1d': 0.6, 'SOL_USDT_1d': 0.4},
            'diversified': {'ETH_USDT_1d': 0.35, 'SOL_USDT_1d': 0.25, 'ETH_USDT_4h': 0.25, 'BTC_USDT_1d': 0.15}
        }
        
        self.results_history = []
    
    def test_single_asset_config(self, asset: str, config: Dict, days: int = 180) -> Optional[Dict]:
        """
        Probar una configuración en un asset específico
        """
        try:
            # Extract symbol and timeframe
            parts = asset.split('_')
            symbol = '_'.join(parts[:-1])
            timeframe = parts[-1]
            
            # Update global config
            original_config = FUTURES_CONFIG.copy()
            FUTURES_CONFIG.clear()
            FUTURES_CONFIG.update(config)
            
            # Create backtester
            backtester = FuturesBacktester(symbol, timeframe)
            
            # Run backtest
            results = backtester.run_backtest(days)
            
            # Restore config
            FUTURES_CONFIG.clear()
            FUTURES_CONFIG.update(original_config)
            
            if results and 'summary' in results:
                summary = results['summary']
                return {
                    'asset': asset,
                    'total_return_pct': summary.get('total_return_percentage', 0),
                    'sharpe_ratio': summary.get('sharpe_ratio', 0),
                    'max_drawdown_pct': abs(summary.get('max_drawdown_pct', 0)),
                    'total_trades': summary.get('total_operations', 0),
                    'profitable_ops': summary.get('profitable_operations', 0),
                    'profit_usd': summary.get('total_profit_usd', 0),
                    'win_rate': summary.get('profitable_operations', 0) / max(summary.get('total_operations', 1), 1)
                }
                
        except Exception as e:
            print(f"❌ Error testing {asset}: {e}")
            
        return None
    
    def optimize_portfolio_allocation(self, leverage: int = 40, days: int = 180) -> Dict:
        """
        Optimizar asignación de portfolio para un nivel de leverage específico
        """
        print(f"🎯 OPTIMIZANDO PORTFOLIO CON LEVERAGE {leverage}X")
        print(f"📊 Período de prueba: {days} días")
        print("-" * 50)
        
        config = self.optimized_configs[leverage]
        asset_results = {}
        
        # Probar cada asset individualmente
        for asset in self.asset_universe.keys():
            print(f"📈 Probando {asset}...")
            result = self.test_single_asset_config(asset, config, days)
            
            if result and result['total_trades'] >= 5:  # Mínimo trades para validez
                asset_results[asset] = result
                print(f"   ✅ Return: {result['total_return_pct']:.2f}%, Sharpe: {result['sharpe_ratio']:.2f}")
            else:
                print(f"   ❌ Insuficientes trades o error")
        
        if not asset_results:
            print("❌ No se obtuvieron resultados válidos")
            return {}
        
        # Calcular métricas de portfolio para cada estrategia
        portfolio_results = {}
        
        for strategy_name, allocation in self.allocation_strategies.items():
            portfolio_return = 0
            portfolio_sharpe = 0
            portfolio_drawdown = 0
            portfolio_trades = 0
            valid_allocation = 0
            
            print(f"\n💼 Evaluando estrategia: {strategy_name.upper()}")
            
            for asset, weight in allocation.items():
                if asset in asset_results:
                    result = asset_results[asset]
                    portfolio_return += result['total_return_pct'] * weight
                    portfolio_sharpe += result['sharpe_ratio'] * weight
                    portfolio_drawdown += result['max_drawdown_pct'] * weight
                    portfolio_trades += result['total_trades'] * weight
                    valid_allocation += weight
                    print(f"   {asset}: {weight:.0%} -> {result['total_return_pct']:.2f}% contribución")
                else:
                    print(f"   {asset}: {weight:.0%} -> NO DISPONIBLE")
            
            if valid_allocation > 0.5:  # Al menos 50% de asignación válida
                # Normalizar por asignación válida
                normalize_factor = valid_allocation
                portfolio_return /= normalize_factor
                portfolio_sharpe /= normalize_factor
                portfolio_drawdown /= normalize_factor
                
                # Calcular score compuesto
                risk_adjusted_score = (portfolio_return * portfolio_sharpe) / (portfolio_drawdown + 1)
                
                portfolio_results[strategy_name] = {
                    'total_return_pct': portfolio_return,
                    'sharpe_ratio': portfolio_sharpe,
                    'max_drawdown_pct': portfolio_drawdown,
                    'total_trades': portfolio_trades,
                    'risk_adjusted_score': risk_adjusted_score,
                    'valid_allocation': valid_allocation,
                    'allocation': allocation,
                    'asset_results': {k: v for k, v in asset_results.items() if k in allocation}
                }
                
                print(f"   📊 Portfolio Return: {portfolio_return:.2f}%")
                print(f"   📈 Portfolio Sharpe: {portfolio_sharpe:.2f}")
                print(f"   🛡️  Max Drawdown: {portfolio_drawdown:.2f}%")
                print(f"   ⭐ Risk Score: {risk_adjusted_score:.2f}")
        
        return {
            'leverage': leverage,
            'asset_results': asset_results,
            'portfolio_results': portfolio_results,
            'test_period_days': days
        }
    
    def compare_leverage_strategies(self, leverages: List[int] = [35, 40, 45], days: int = 180) -> Dict:
        """
        Comparar diferentes estrategias de leverage
        """
        print(f"🚀 COMPARACIÓN DE ESTRATEGIAS DE LEVERAGE")
        print(f"🎯 Objetivo: Amplificar baseline 5.27% -> 15-25% anual")
        print("=" * 60)
        
        all_results = {}
        
        for leverage in leverages:
            print(f"\n🔧 PROBANDO LEVERAGE {leverage}X")
            print("=" * 30)
            
            results = self.optimize_portfolio_allocation(leverage, days)
            if results:
                all_results[leverage] = results
        
        # Analizar mejores estrategias globalmente
        if all_results:
            print(f"\n🏆 ANÁLISIS COMPARATIVO DE LEVERAGES")
            print("=" * 50)
            
            best_overall = None
            best_score = -float('inf')
            
            for leverage, results in all_results.items():
                print(f"\n⚙️  LEVERAGE {leverage}X:")
                
                if 'portfolio_results' in results:
                    best_strategy = max(results['portfolio_results'].items(), 
                                      key=lambda x: x[1]['risk_adjusted_score'])
                    
                    strategy_name, strategy_data = best_strategy
                    
                    print(f"   🥇 Mejor estrategia: {strategy_name}")
                    print(f"   📈 Return: {strategy_data['total_return_pct']:.2f}%")
                    print(f"   📊 Sharpe: {strategy_data['sharpe_ratio']:.2f}")
                    print(f"   🛡️  Drawdown: {strategy_data['max_drawdown_pct']:.2f}%")
                    print(f"   ⭐ Score: {strategy_data['risk_adjusted_score']:.2f}")
                    
                    # Calcular retorno anualizado
                    annualized_return = strategy_data['total_return_pct'] * (365/days)
                    improvement_factor = annualized_return / 5.27  # vs baseline
                    
                    print(f"   🎯 Retorno anualizado: {annualized_return:.2f}%")
                    print(f"   🚀 Factor de mejora: {improvement_factor:.2f}x")
                    
                    if strategy_data['risk_adjusted_score'] > best_score:
                        best_score = strategy_data['risk_adjusted_score']
                        best_overall = {
                            'leverage': leverage,
                            'strategy': strategy_name,
                            'data': strategy_data,
                            'annualized_return': annualized_return,
                            'improvement_factor': improvement_factor
                        }
            
            # Mostrar resultado final
            if best_overall:
                print(f"\n🎉 CONFIGURACIÓN ÓPTIMA ENCONTRADA")
                print("=" * 40)
                print(f"🔧 Leverage: {best_overall['leverage']}x")
                print(f"💼 Estrategia: {best_overall['strategy']}")
                print(f"📈 Retorno anualizado: {best_overall['annualized_return']:.2f}%")
                print(f"🚀 Amplificación: {best_overall['improvement_factor']:.2f}x baseline")
                print(f"📊 Sharpe Ratio: {best_overall['data']['sharpe_ratio']:.2f}")
                print(f"🛡️  Max Drawdown: {best_overall['data']['max_drawdown_pct']:.2f}%")
                
                if best_overall['improvement_factor'] >= 2.0:
                    print(f"🎯 ¡OBJETIVO ALCANZADO! Amplificación x2+ conseguida")
                elif best_overall['improvement_factor'] >= 1.5:
                    print(f"⚡ Buen progreso hacia meta de amplificación x2-3")
                else:
                    print(f"🔄 Necesita optimización adicional para alcanzar meta")
                
                print(f"\n💼 ASIGNACIÓN ÓPTIMA:")
                for asset, weight in best_overall['data']['allocation'].items():
                    if asset in best_overall['data']['asset_results']:
                        asset_return = best_overall['data']['asset_results'][asset]['total_return_pct']
                        print(f"   {asset}: {weight:.0%} (Return: {asset_return:.2f}%)")
                
                all_results['best_overall'] = best_overall
        
        return all_results
    
    def save_optimization_results(self, results: Dict):
        """
        Guardar resultados de optimización
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_asset_optimization_{timestamp}.json"
        
        # Hacer serializable
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Resultados guardados en: {filename}")

def main():
    """
    Ejecutar optimización completa de portfolio multi-asset
    """
    optimizer = MultiAssetOptimizer()
    
    # Ejecutar comparación de leverages
    results = optimizer.compare_leverage_strategies(
        leverages=[35, 40, 45],
        days=180
    )
    
    if results:
        # Guardar resultados
        optimizer.save_optimization_results(results)
        
        print(f"\n✅ OPTIMIZACIÓN MULTI-ASSET COMPLETADA")
        
        if 'best_overall' in results:
            best = results['best_overall']
            print(f"🏆 Mejor configuración: Leverage {best['leverage']}x, {best['strategy']}")
            print(f"🎯 Amplificación conseguida: {best['improvement_factor']:.2f}x")
        
    else:
        print("❌ No se pudieron obtener resultados de optimización")

if __name__ == "__main__":
    main()