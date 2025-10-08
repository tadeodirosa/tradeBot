# 🚀 TradeBot - Sistema de Trading Automático# 🚀 Crypto Analysis & Backtesting Suite



![Trading Bot](https://img.shields.io/badge/Trading-Bot-green)Un proyecto profesional y modular para análisis de criptomonedas y backtesting de estrategias de trading.

![Python](https://img.shields.io/badge/Python-3.8+-blue)

![ROI](https://img.shields.io/badge/ROI-427.86%25-gold)## 📁 Estructura del Proyecto

![Win Rate](https://img.shields.io/badge/Win%20Rate-50.8%25-success)

```

Sistema de trading automático con **análisis técnico avanzado**, **backtesting verificado** y **tracking de performance en tiempo real**.crypto-analysis-project/

├── main.py                 # 🎯 Punto de entrada unificado (CLI)

## 📊 Performance Verificada│

├── core/                   # 🏗️ Clean Architecture - Lógica de negocio

- **ROI:** 427.86% (30 días)│   ├── domain/             # Entidades y eventos del dominio

- **Win Rate:** 50.8%│   │   ├── models.py       # Modelos de negocio (CandleData, BacktestResult)

- **Profit Factor:** 1.46│   │   └── events.py       # Eventos del dominio (DataFetched, SignalGenerated)

- **Max Drawdown:** 69.2%│   └── application/        # Casos de uso y servicios de aplicación

- **Señales generadas:** 120+ de alta calidad│       ├── services.py     # Servicios de aplicación

│       └── handlers.py     # Manejadores de eventos

## ⭐ Características Principales│

├── infrastructure/         # 🔌 Implementaciones externas

### 🎯 Análisis en Tiempo Real│   ├── events/             # Sistema de Event Bus

- **Indicadores técnicos:** EMA, RSI, ATR│   │   └── event_bus.py    # Event Bus lightweight sin dependencias

- **Condiciones selectivas:** 4+ condiciones requeridas para señal│   ├── logging/            # Sistema de logging estructurado

- **API Binance:** Datos en tiempo real│   │   └── structured_logger.py  # Logging con correlation IDs

- **Risk Management:** Stop Loss y Take Profit automáticos│   └── monitoring/         # Circuit breakers y métricas

│

### 📐 Backtesting Verificado├── data/                   # 💾 Pipeline de datos

- **Datos históricos reales** de Binance│   ├── adapters/           # Conectores a APIs (BaseAdapter)

- **Validación matemática** vs TA-Lib│   ├── repositories/       # Almacenamiento (Repository pattern)

- **Multiple timeframes:** 4h por defecto│   └── cache.py           # Sistema de caché inteligente

- **Multi-símbolo:** BTC, ETH, LINK, XRP, etc.│

├── strategies/             # 📈 Estrategias de trading (BaseStrategy)

### 💾 Sistema de Tracking├── config/                 # ⚙️ Configuración environment-aware

- **Guardado automático** de señales│   ├── base.yaml          # Configuración base

- **Seguimiento de performance** en tiempo real│   ├── dev.yaml           # Configuración desarrollo

- **Análisis de rentabilidad** por símbolo│   └── prod.yaml          # Configuración producción

- **Estadísticas completas** de trading│

├── scripts/                # 🛠️ Scripts utilitarios organizados

### 🛡️ Control de Versiones│   ├── discovery/          # Descubrimiento de nuevas oportunidades

- **Backup automático** de versiones estables│   ├── download/           # Descarga de datos de mercado

- **Restauración de emergencia**│   ├── analysis/           # Scripts de análisis batch

- **Validación continua** del sistema│   └── optimization/       # Optimización y grid searches

- **Documentación completa** de cambios│

├── reports/                # � Reportes generados (HTML, CSV)

## 🚀 Instalación y Uso├── tests/                  # ✅ Tests (unit, integration, property-based)

└── utils/                  # ⚙️ Utilidades generales

### Requisitos```

```bash

pip install -r requirements.txt## 🚀 Uso Rápido

```

### Instalación

### Dependencias Principales```bash

- `pandas` - Análisis de datospip install -r requirements.txt

- `numpy` - Cálculos matemáticos```

- `requests` - API Binance

- `ccxt` - Exchange connectivity### CLI Unificado

```bash

### Uso Rápido# Análisis de un activo específico

python main.py analyze single --symbol BTCUSDT --timeframe 4h

#### 1. Interfaz Completa (Recomendado)

```bash# Backtesting de estrategia

python trading_manager.pypython main.py backtest --symbol BTCUSDT --strategy meanreversion

```

# Descubrimiento de oportunidades

#### 2. Análisis en Tiempo Realpython main.py discover --top50 --binance

```bash

python real_time_analyzer.py# Descarga de datos

```python main.py download --symbols BTC,ETH --timeframes 1h,4h



#### 3. Backtesting# Optimización de parámetros

```bashpython main.py optimize --gridsearch --strategy meanreversion

python verified_backtester.py```

```

### Uso Directo de Módulos

#### 4. Tracking de Señales```bash

```bash# Análisis directo

python signal_tracker.pypython analysis/analyzer_v10.py BTCUSDT_4h

```

# Backtesting directo

## 📋 Opciones del Trading Managerpython backtesting/backtester.py BTCUSDT_4h

```
1. **📊 Analizar señal** (LINK por defecto)
2. **📈 Analizar otro símbolo**
3. **🔄 Actualizar señales activas**
4. **📋 Ver performance del modelo**
5. **👀 Ver señales activas**
6. **🎯 Análisis completo** (señal + update)
7. **📐 Ejecutar backtest**
8. **🔬 Backtest + análisis en tiempo real**

## 🔧 Configuración

### Parámetros del Sistema
```python
{
    'timeframe': '4h',           # Timeframe de análisis
    'ema_fast': 9,              # EMA rápida
    'ema_slow': 21,             # EMA lenta
    'rsi_period': 14,           # Período RSI
    'atr_period': 14,           # Período ATR
    'stop_loss_atr_multiplier': 1.2,  # Multiplicador Stop Loss
    'risk_per_trade': 0.03,     # 3% riesgo por trade
    'max_position_size': 0.25   # 25% máximo del capital
}
```

### Condiciones de Señal

#### Señales LONG (4+ condiciones requeridas)
- ✅ Precio > EMA21
- ✅ EMA9 > EMA21
- ✅ RSI en zona 25-45 (sobreventa controlada)
- ✅ ATR > 1.5% (volatilidad adecuada)
- ✅ Momentum > -2% (no en caída libre)

#### Señales SHORT (4+ condiciones requeridas)
- ✅ Precio < EMA21
- ✅ EMA9 < EMA21
- ✅ RSI en zona 55-75 (sobrecompra controlada)
- ✅ ATR > 1.5% (volatilidad adecuada)
- ✅ Momentum < 2% (no en subida descontrolada)

## 📊 Estructura del Proyecto

```
tradebot/
├── 📁 core/                    # Archivos principales del sistema
├── 📁 analysis/               # Análisis y reportes
├── 📁 config/                 # Configuraciones
├── 📁 data/                   # Datos históricos
├── 📁 reports/                # Reportes generados
├── 📁 tests/                  # Tests del sistema
├── 📄 real_time_analyzer.py   # Análisis en tiempo real
├── 📄 verified_backtester.py  # Backtesting verificado
├── 📄 signal_tracker.py       # Tracking de señales
├── 📄 trading_manager.py      # Interfaz principal
├── 📄 version_manager.py      # Control de versiones
├── 📄 system_validator.py     # Validación del sistema
└── 📄 requirements.txt        # Dependencias
```

## 🎯 Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `trading_manager.py` | **Interfaz principal** - Combina todas las funcionalidades |
| `real_time_analyzer.py` | **Análisis en tiempo real** - Genera señales actuales |
| `verified_backtester.py` | **Backtesting** - Prueba histórica del sistema |
| `signal_tracker.py` | **Tracking** - Seguimiento de performance |
| `version_manager.py` | **Versionado** - Backup y restauración |
| `system_validator.py` | **Validación** - Verificación del sistema |

## 📈 Resultados por Símbolo

| Símbolo | ROI | Win Rate | Profit Factor | Estado |
|---------|-----|----------|---------------|---------|
| LINK/USDT | **427.86%** | 50.8% | 1.46 | ✅ Excelente |
| XRP/USDT | -355.39% | 22.4% | 0.31 | ❌ No recomendado |
| BTC/USDT | *En prueba* | - | - | 🔄 |
| ETH/USDT | *En prueba* | - | - | 🔄 |

## 🛡️ Control de Calidad

### Validaciones Automáticas
- **Sintaxis:** Verificación de código Python
- **Matemáticas:** Comparación vs TA-Lib (precisión 100%)
- **Performance:** ROI, Win Rate, Profit Factor
- **Archivos:** Integridad de archivos críticos

### Criterios de Versión Estable
- ROI mínimo: >300%
- Win Rate mínimo: >45%
- Profit Factor mínimo: >1.3
- Max Drawdown máximo: <80%

## 🚨 Comandos de Emergencia

```bash
# Validación completa del sistema
python system_validator.py

# Crear backup de emergencia
python version_manager.py

# Restaurar versión estable
python version_manager.py interactive  # → Opción 3

# Verificar funcionamiento básico
python verified_backtester.py
```

## 📚 Documentación Técnica

- **[MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md)** - Guía de mantenimiento
- **[OPTIMIZATION_VALIDATION.md](OPTIMIZATION_VALIDATION.md)** - Validación de optimizaciones
- **[TRADING_SYSTEM_ANALYSIS.md](TRADING_SYSTEM_ANALYSIS.md)** - Análisis del sistema
- **[.copilot-instructions.md](.copilot-instructions.md)** - Instrucciones para desarrollo

## 🔄 Roadmap Futuro

### Próximas Mejoras
- **Multi-timeframe:** 4h + 1h + 15m
- **Multi-símbolo:** Trading simultáneo
- **Machine Learning:** Predicción probabilidad éxito
- **Stop Loss dinámico:** Basado en volatilidad
- **Portfolio Management:** Múltiples posiciones
- **Auto-trading:** Ejecución automática

### En Desarrollo
- Dashboard web para visualización
- Alertas automáticas (Telegram/Email)
- Integración con exchange para auto-trading
- Optimización de parámetros con ML

## ⚠️ Disclaimer

Este sistema es para **fines educativos y de investigación**. El trading conlleva riesgos significativos. Siempre:
- Prueba en **cuenta demo** primero
- **Nunca inviertas** más de lo que puedes permitirte perder
- **Diversifica** tu portafolio
- **Mantén** strict risk management

## 📞 Soporte

Para reportar bugs, sugerir mejoras o contribuir al proyecto:
- Crea un **Issue** en GitHub
- Revisa la **documentación técnica**
- Ejecuta `python system_validator.py` para diagnósticos

---

## 🏆 Stats del Proyecto

- **Líneas de código:** 2000+
- **Archivos:** 30+
- **Tests ejecutados:** 100+
- **Versiones estables:** 3+
- **Símbolos analizados:** 10+

**⭐ ¡No olvides dar una estrella al proyecto si te resulta útil!**

---

*Desarrollado con ❤️ para la comunidad de trading*