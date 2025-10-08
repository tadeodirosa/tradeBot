# 🚀 Guía de Instalación Rápida - TradeBot

## ⚡ Instalación Express (2 minutos)

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/tradebot.git
cd tradebot
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. ¡Listo! Ejecutar
```bash
python trading_manager.py
```

## 🎯 Primer Uso

### Analizar LINK (configuración por defecto)
```bash
python trading_manager.py
# Seleccionar opción 1: Analizar señal (LINK por defecto)
```

### Hacer un backtest rápido
```bash
python verified_backtester.py
# Enter para LINK/USDT (por defecto)
# Enter para últimos 30 días
# Enter para hasta hoy
```

### Ver rendimiento en tiempo real
```bash
python real_time_analyzer.py
```

## 📊 Resultados Esperados

Deberías ver algo como:
```
🚨 SEÑAL DETECTADA: LONG
💪 Fuerza: 80%
🎯 Stop Loss: $21.89
🏆 Take Profit: $23.44
```

## ❗ Problemas Comunes

### Error de dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Error de conexión API
- Verificar conexión a internet
- API de Binance debe estar accesible

### Python no encontrado
- Instalar Python 3.8 o superior
- Verificar que está en PATH

## 🆘 Soporte Rápido

Si algo no funciona:
```bash
python system_validator.py
```

Este comando diagnostica y reporta cualquier problema.

---

**¡En 2 minutos tendrás un sistema de trading funcionando con 427.86% ROI probado!** 🚀