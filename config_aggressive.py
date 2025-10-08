# 🔥 CONFIGURACIÓN AGRESIVA PARA 45X LEVERAGE
# Diseñada para generar 10-20% retornos con gestión de riesgo adecuada

FUTURES_CONFIG_AGGRESSIVE = {
    # Balance y leverage
    'initial_balance': 5000,      # $5,000 iniciales
    'leverage': 45,               # 45x leverage mantenido
    
    # ⚡ POSICIÓN AGRESIVA - INCREMENTADA 5X
    'position_size_usd': 500,     # $500 USD por posición (vs $100 conservador)
    'max_positions': 8,           # Máximo 8 posiciones simultáneas
    'total_exposure_limit': 0.8,  # Máximo 80% del balance en posiciones
    
    # 🎯 TAKE PROFIT MÁS AJUSTADO
    'take_profit_atr_mult': 1.2,  # TP más cercano para capturar rápido (vs 2.5x)
    'stop_loss_atr_mult': 1.8,    # SL ligeramente más amplio para evitar noise
    
    # 📊 SCORING MÁS AGRESIVO  
    'min_buy_score': 45,          # Umbral más bajo = más trades (vs 50)
    'min_sell_score': 45,         # Más shorts disponibles
    'min_confidence': 0.45,       # Menor confidence requerida
    
    # ⚡ CONFIGURACIÓN ALTA FRECUENCIA
    'trade_frequency_factor': 2.0, # Factor de multiplicación para más trades
    'atr_period': 10,              # ATR más reactivo (vs 14)
    'rsi_period': 12,              # RSI más sensible (vs 14)
    
    # 💰 COSTOS
    'commission_rate': 0.0012,     # 0.12% por transacción
    'funding_rate_daily': 0.0001,  # 0.01% funding diario
    
    # 🛡️ GESTIÓN DE RIESGO ADAPTATIVA
    'max_drawdown_limit': 0.15,   # Stop trading si DD > 15%
    'volatility_adjustment': True, # Ajustar posición según volatilidad
    'correlation_limit': 0.7,     # Máxima correlación entre posiciones
    
    # 🔄 REBALANCEO DINÁMICO
    'rebalance_frequency': 'daily', # Rebalancear posiciones diariamente
    'profit_lock_threshold': 0.05,  # Lock profits al 5%
    'trailing_stop_factor': 0.3,    # Trailing stop al 30% del movimiento
}

# 📈 CONFIGURACIÓN POR ASSET ESPECÍFICA
ASSET_SPECIFIC_AGGRESSIVE = {
    'ETH_USDT': {
        'position_size_multiplier': 1.2,  # 20% más para ETH (menos volátil)
        'take_profit_atr_mult': 1.1,      # TP más ajustado para ETH
        'min_buy_score': 42,              # Más agresivo con ETH
    },
    'SOL_USDT': {
        'position_size_multiplier': 0.8,  # 20% menos para SOL (más volátil)
        'take_profit_atr_mult': 1.4,      # TP más amplio para SOL
        'min_buy_score': 48,              # Más selectivo con SOL
    },
    'BTC_USDT': {
        'position_size_multiplier': 1.0,  # Base para BTC
        'take_profit_atr_mult': 1.2,      # TP estándar para BTC
        'min_buy_score': 45,              # Base para BTC
    }
}

# 🎯 TARGETS DE PERFORMANCE AGRESIVA
AGGRESSIVE_TARGETS = {
    'monthly_return_target': 0.08,    # 8% mensual = 96% anual (realista con 45x)
    'sharpe_ratio_min': 2.0,          # Mínimo Sharpe aceptable
    'max_drawdown_limit': 0.20,       # Máximo 20% drawdown
    'win_rate_target': 0.55,          # 55% win rate target
    'profit_factor_min': 1.8,         # Mínimo profit factor
}

def calculate_aggressive_position_size(balance, leverage, volatility, confidence):
    """
    Calcula tamaño de posición dinámico basado en:
    - Balance disponible
    - Leverage configurado  
    - Volatilidad del asset
    - Confidence del signal
    """
    base_size = FUTURES_CONFIG_AGGRESSIVE['position_size_usd']
    
    # Ajuste por confidence (0.5-1.0 multiplier)
    confidence_multiplier = 0.5 + confidence
    
    # Ajuste por volatilidad (menor size si más volátil)
    volatility_multiplier = max(0.3, 1.0 - volatility)
    
    # Ajuste por balance disponible
    max_size_by_balance = balance * 0.1  # Máximo 10% del balance por posición
    
    adjusted_size = base_size * confidence_multiplier * volatility_multiplier
    
    return min(adjusted_size, max_size_by_balance)

def validate_aggressive_config():
    """Valida que la configuración agresiva sea factible"""
    config = FUTURES_CONFIG_AGGRESSIVE
    
    max_margin_needed = (config['position_size_usd'] * config['max_positions']) / config['leverage']
    margin_ratio = max_margin_needed / config['initial_balance']
    
    print(f"🔍 VALIDACIÓN CONFIGURACIÓN AGRESIVA:")
    print(f"   Posición individual: ${config['position_size_usd']}")
    print(f"   Máximo margen por posición: ${config['position_size_usd'] / config['leverage']:.2f}")
    print(f"   Máximo margen total: ${max_margin_needed:.2f}")
    print(f"   Ratio margen/balance: {margin_ratio:.1%}")
    print(f"   {'✅ VIABLE' if margin_ratio < 0.8 else '❌ DEMASIADO RIESGOSO'}")
    
    return margin_ratio < 0.8

if __name__ == "__main__":
    validate_aggressive_config()