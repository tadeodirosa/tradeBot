"""
Multi-Timeframe Trading Strategy - Centralized Configuration
============================================================

Este archivo contiene todas las configuraciones centralizadas para las diferentes
versiones de la estrategia multi-timeframe. Facilita el mantenimiento y la 
calibración de parámetros.

Uso:
    from config import TradingConfig
    config = TradingConfig.get_v21_config()
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class TradingConfig:
    """Configuración centralizada para estrategias de trading."""
    
    # === CONFIGURACIÓN V2.1 PRODUCTION (BALANCEADA) ===
    @staticmethod
    def get_v21_config() -> Dict[str, Any]:
        """
        Configuración V2.1 - Balance profesional optimizado.
        
        Resultados validados:
        - LINK/USDT: 50.6% ROI, 66.7% WR, -5.0% DD
        - ADA/USDT: 7.4% ROI, 41.2% WR, -8.1% DD
        - SOL/USDT: -5.8% ROI, 28.7% WR, -16.1% DD
        """
        return {
            # === TIMEFRAMES ===
            'timeframe_trend': '4h',
            'timeframe_entry': '1h',
            
            # === CAPITAL MANAGEMENT ===
            'initial_balance': 10000,
            'risk_per_trade': 0.025,        # 2.5% account risk per trade
            'max_position_size': 0.22,      # 22% maximum capital per position
            'commission': 0.001,            # 0.1% trading commission
            
            # === TECHNICAL INDICATORS ===
            'ema_fast': 9,                  # Fast EMA period
            'ema_slow': 21,                 # Slow EMA period
            'rsi_period': 14,               # RSI calculation period
            'atr_period': 14,               # ATR calculation period
            
            # === RISK CONTROLS ===
            'stop_loss_atr_multiplier': 1.0,    # Stop loss distance (1x ATR)
            'take_profit_atr_multiplier': 2.2,  # Take profit distance (2.2x ATR)
            'min_signal_gap_hours': 2,          # Minimum hours between signals
            
            # === VOLATILITY FILTERS ===
            'min_volatility_4h': 1.2,      # Minimum 4H ATR percentage (1.2%)
            'max_volatility_4h': 10.0,     # Maximum 4H ATR percentage (10.0%)
            
            # === 4H TREND ANALYSIS (3/3 REQUIRED) ===
            'ema_trend_min_4h': 0.3,       # Minimum EMA separation (0.3%)
            'momentum_threshold_4h': 0.5,   # Minimum momentum (0.5%)
            'rsi_range_4h': [30, 70],      # RSI acceptable range
            
            # === 1H ENTRY ANALYSIS (3/4 REQUIRED) ===
            'rsi_long_range_1h': [20, 50],     # LONG RSI entry zone
            'rsi_short_range_1h': [50, 80],    # SHORT RSI entry zone
            'momentum_range_1h': [-3, 4],      # 1H momentum acceptable range
            'ema_alignment_threshold_1h': 1.5, # EMA alignment tolerance
            'volatility_range_1h': [0.6, 8.0], # 1H volatility range
            
            # === QUALITY THRESHOLDS ===
            'min_trend_quality_4h': 60,    # Minimum quality score (60/100)
            
            # === STRATEGY METADATA ===
            'version': 'V2.1',
            'status': 'PRODUCTION',
            'last_optimized': '2024-10-08',
            'validation_period': '2024-09-03 to 2024-10-02',
            'best_asset': 'LINK/USDT',
            'recommended_assets': ['LINK/USDT', 'ETH/USDT', 'BTC/USDT'],
            'avoid_assets': ['SOL/USDT', 'DOGE/USDT', 'SHIB/USDT']
        }
    
    # === CONFIGURACIÓN CONSERVADORA ===
    @staticmethod
    def get_conservative_config() -> Dict[str, Any]:
        """Configuración conservadora para menor riesgo."""
        config = TradingConfig.get_v21_config()
        config.update({
            'risk_per_trade': 0.015,            # 1.5% vs 2.5%
            'max_position_size': 0.15,          # 15% vs 22%
            'min_trend_quality_4h': 70,         # 70 vs 60
            'take_profit_atr_multiplier': 1.8,  # 1.8 vs 2.2
            'min_signal_gap_hours': 4,          # 4h vs 2h
            'max_volatility_4h': 8.0,           # 8% vs 10%
            'version': 'V2.1_CONSERVATIVE'
        })
        return config
    
    # === CONFIGURACIÓN AGRESIVA ===
    @staticmethod
    def get_aggressive_config() -> Dict[str, Any]:
        """Configuración agresiva para mayor riesgo/retorno."""
        config = TradingConfig.get_v21_config()
        config.update({
            'risk_per_trade': 0.035,            # 3.5% vs 2.5%
            'max_position_size': 0.30,          # 30% vs 22%
            'min_trend_quality_4h': 50,         # 50 vs 60
            'take_profit_atr_multiplier': 2.8,  # 2.8 vs 2.2
            'min_signal_gap_hours': 1,          # 1h vs 2h
            'max_volatility_4h': 12.0,          # 12% vs 10%
            'version': 'V2.1_AGGRESSIVE'
        })
        return config
    
    # === CONFIGURACIÓN LEGACY (BASELINE) ===
    @staticmethod
    def get_baseline_config() -> Dict[str, Any]:
        """Configuración original 4H (baseline preservado)."""
        return {
            'timeframe': '4h',
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'atr_period': 14,
            'risk_per_trade': 0.02,
            'stop_loss_atr_multiplier': 1.5,
            'take_profit_atr_multiplier': 2.0,
            'version': 'BASELINE_4H',
            'status': 'PRESERVED',
            'performance': {
                'roi': 427.86,
                'win_rate': 50.8,
                'profit_factor': 1.46
            }
        }


class AssetClassification:
    """Clasificación de assets basada en performance validada."""
    
    TIER_1_EXCELLENT = [
        'LINK/USDT',  # 50.6% ROI, 66.7% WR ⭐ BEST
        'ETH/USDT',   # Large cap stability
        'BTC/USDT',   # Market leader
    ]
    
    TIER_2_MODERATE = [
        'ADA/USDT',   # 7.4% ROI, 41.2% WR
        'DOT/USDT',   # Good trending behavior
        'ATOM/USDT',  # Decent technical response
        'AVAX/USDT',  # Moderate volatility
        'MATIC/USDT', # Consistent performer
    ]
    
    TIER_3_AVOID = [
        'SOL/USDT',   # -5.8% ROI, 28.7% WR
        'DOGE/USDT',  # Meme volatility
        'SHIB/USDT',  # Extreme volatility
        'LUNA/USDT',  # Unstable ecosystem
        'FTT/USDT',   # Exchange risk
    ]
    
    @staticmethod
    def get_asset_tier(symbol: str) -> str:
        """Determinar tier de un asset."""
        if symbol in AssetClassification.TIER_1_EXCELLENT:
            return 'TIER_1_EXCELLENT'
        elif symbol in AssetClassification.TIER_2_MODERATE:
            return 'TIER_2_MODERATE'
        elif symbol in AssetClassification.TIER_3_AVOID:
            return 'TIER_3_AVOID'
        else:
            return 'TIER_UNKNOWN'
    
    @staticmethod
    def get_recommended_symbols() -> List[str]:
        """Obtener símbolos recomendados para trading."""
        return AssetClassification.TIER_1_EXCELLENT + AssetClassification.TIER_2_MODERATE


class MarketRegimeConfig:
    """Configuraciones específicas para diferentes regímenes de mercado."""
    
    @staticmethod
    def get_bull_market_config() -> Dict[str, Any]:
        """Configuración optimizada para mercado alcista."""
        config = TradingConfig.get_v21_config()
        config.update({
            'momentum_threshold_4h': 0.3,       # Más sensible al momentum
            'rsi_long_range_1h': [25, 55],      # Zona de compra más amplia
            'take_profit_atr_multiplier': 2.5,  # Mayor profit target
            'regime': 'BULL_MARKET'
        })
        return config
    
    @staticmethod
    def get_bear_market_config() -> Dict[str, Any]:
        """Configuración optimizada para mercado bajista."""
        config = TradingConfig.get_v21_config()
        config.update({
            'momentum_threshold_4h': 0.7,       # Más restrictivo
            'rsi_short_range_1h': [45, 75],     # Zona de venta más amplia
            'min_trend_quality_4h': 70,         # Mayor calidad requerida
            'risk_per_trade': 0.02,             # Menor riesgo por trade
            'regime': 'BEAR_MARKET'
        })
        return config
    
    @staticmethod
    def get_sideways_config() -> Dict[str, Any]:
        """Configuración optimizada para mercado lateral."""
        config = TradingConfig.get_v21_config()
        config.update({
            'min_volatility_4h': 2.0,           # Mayor volatilidad mínima
            'take_profit_atr_multiplier': 1.8,  # Profit target más conservador
            'min_signal_gap_hours': 4,          # Mayor espaciado entre señales
            'regime': 'SIDEWAYS_MARKET'
        })
        return config


class ValidationResults:
    """Resultados de validación histórica documentados."""
    
    V21_RESULTS = {
        'LINK/USDT': {
            'period': '2024-09-03 to 2024-10-02',
            'trades': 84,
            'win_rate': 66.7,
            'roi': 50.6,
            'max_drawdown': -5.0,
            'profit_factor': 3.70,
            'status': 'EXCELLENT',
            'recommendation': 'PRIMARY_ASSET'
        },
        'ADA/USDT': {
            'period': '2024-09-03 to 2024-10-02',
            'trades': 85,
            'win_rate': 41.2,
            'roi': 7.4,
            'max_drawdown': -8.1,
            'profit_factor': 1.32,
            'status': 'MODERATE',
            'recommendation': 'SECONDARY_ASSET'
        },
        'SOL/USDT': {
            'period': '2024-09-03 to 2024-10-02',
            'trades': 94,
            'win_rate': 28.7,
            'roi': -5.8,
            'max_drawdown': -16.1,
            'profit_factor': 0.81,
            'status': 'SUBOPTIMAL',
            'recommendation': 'AVOID_OR_OPTIMIZE'
        }
    }
    
    @staticmethod
    def get_asset_performance(symbol: str) -> Dict[str, Any]:
        """Obtener performance histórica de un asset."""
        return ValidationResults.V21_RESULTS.get(symbol, {
            'status': 'UNKNOWN',
            'recommendation': 'BACKTEST_REQUIRED'
        })


# === CONFIGURACIONES RÁPIDAS ===

def get_production_config() -> Dict[str, Any]:
    """Configuración de producción recomendada (V2.1)."""
    return TradingConfig.get_v21_config()

def get_safe_config() -> Dict[str, Any]:
    """Configuración segura para principiantes."""
    return TradingConfig.get_conservative_config()

def get_performance_config() -> Dict[str, Any]:
    """Configuración de alta performance para expertos."""
    return TradingConfig.get_aggressive_config()

def get_config_for_asset(symbol: str) -> Dict[str, Any]:
    """Obtener configuración optimizada para asset específico."""
    tier = AssetClassification.get_asset_tier(symbol)
    
    if tier == 'TIER_1_EXCELLENT':
        return TradingConfig.get_v21_config()
    elif tier == 'TIER_2_MODERATE':
        return TradingConfig.get_conservative_config()
    else:
        # Para assets desconocidos o tier 3, usar configuración muy conservadora
        config = TradingConfig.get_conservative_config()
        config.update({
            'risk_per_trade': 0.01,  # Solo 1% riesgo
            'min_trend_quality_4h': 80,  # Calidad muy alta
            'warning': f'Asset {symbol} not validated. Use minimal risk.'
        })
        return config


# === VALIDACIÓN DE CONFIGURACIÓN ===

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validar que una configuración tenga parámetros sensatos."""
    warnings = []
    
    # Validar riesgos
    if config.get('risk_per_trade', 0) > 0.05:
        warnings.append('Risk per trade >5% is very aggressive')
    
    if config.get('max_position_size', 0) > 0.35:
        warnings.append('Max position size >35% is extremely risky')
    
    # Validar coherencia
    if config.get('take_profit_atr_multiplier', 0) < config.get('stop_loss_atr_multiplier', 0):
        warnings.append('Take profit should be larger than stop loss')
    
    # Validar thresholds
    if config.get('min_trend_quality_4h', 0) < 30:
        warnings.append('Trend quality <30 may generate too many signals')
    
    if config.get('min_signal_gap_hours', 0) < 1:
        warnings.append('Signal gap <1h may cause over-trading')
    
    return warnings


# === EJEMPLO DE USO ===

if __name__ == "__main__":
    # Configuración estándar
    config = get_production_config()
    print(f"Configuración V2.1: {config['version']}")
    print(f"Risk per trade: {config['risk_per_trade']*100}%")
    print(f"Best asset: {config['best_asset']}")
    
    # Validación
    warnings = validate_config(config)
    if warnings:
        print("\\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\\n✅ Configuration is valid")
    
    # Clasificación de asset
    test_symbol = 'LINK/USDT'
    tier = AssetClassification.get_asset_tier(test_symbol)
    performance = ValidationResults.get_asset_performance(test_symbol)
    print(f"\\n{test_symbol}: {tier}")
    print(f"Validated ROI: {performance.get('roi', 'N/A')}%")
    print(f"Recommendation: {performance.get('recommendation', 'Unknown')}")