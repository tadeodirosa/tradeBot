"""
Trading Signal Manager - Interfaz Unificada
Combina análisis en tiempo real + tracking de performance + backtesting
"""

import sys
import subprocess
from real_time_analyzer import RealTimeAnalyzer
from signal_tracker import SignalTracker

def main():
    print("🚀 TRADING SIGNAL MANAGER")
    print("=" * 50)
    print("1. 📊 Analizar señal (LINK por defecto)")
    print("2. 📈 Analizar otro símbolo")
    print("3. 🔄 Actualizar señales activas")
    print("4. 📋 Ver performance del modelo")
    print("5. 👀 Ver señales activas")
    print("6. 🎯 Análisis completo (señal + update)")
    print("7. 📐 Ejecutar backtest")
    print("8. 🔬 Backtest + análisis en tiempo real")
    print("=" * 50)
    
    choice = input("Elige opción (1-8): ").strip()
    
    if choice == "1":
        # Analizar LINK
        analyzer = RealTimeAnalyzer()
        try:
            result = analyzer.analyze_signal("LINKUSDT")
            analyzer.print_signal_report(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "2":
        # Analizar otro símbolo
        symbol = input("Símbolo (ej: BTCUSDT, ETHUSDT): ").strip().upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        
        analyzer = RealTimeAnalyzer()
        try:
            result = analyzer.analyze_signal(symbol)
            analyzer.print_signal_report(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "3":
        # Actualizar señales
        tracker = SignalTracker()
        tracker.update_active_signals()
    
    elif choice == "4":
        # Ver performance
        tracker = SignalTracker()
        tracker.print_performance_report()
    
    elif choice == "5":
        # Ver señales activas
        tracker = SignalTracker()
        tracker.print_active_signals()
    
    elif choice == "6":
        # Análisis completo
        print("\n🎯 ANÁLISIS COMPLETO")
        print("-" * 30)
        
        # 1. Actualizar señales existentes
        print("1️⃣ Actualizando señales activas...")
        tracker = SignalTracker()
        tracker.update_active_signals()
        
        # 2. Analizar nueva señal
        print("\n2️⃣ Analizando nueva señal LINK...")
        analyzer = RealTimeAnalyzer()
        try:
            result = analyzer.analyze_signal("LINKUSDT")
            analyzer.print_signal_report(result)
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
        
        # 3. Mostrar performance
        print("\n3️⃣ Performance del modelo:")
        tracker.print_performance_report()
        
        # 4. Mostrar señales activas
        print("\n4️⃣ Estado actual:")
        tracker.print_active_signals()
    
    elif choice == "7":
        # Ejecutar backtest
        print("\n📐 EJECUTANDO BACKTESTER")
        print("-" * 30)
        try:
            # Ejecutar el backtester como proceso separado para mantener la interactividad
            result = subprocess.run([sys.executable, "verified_backtester.py"], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print("\n✅ Backtest completado exitosamente")
            else:
                print("\n❌ Error en el backtest")
        except Exception as e:
            print(f"❌ Error ejecutando backtest: {e}")
    
    elif choice == "8":
        # Backtest + análisis en tiempo real
        print("\n🔬 ANÁLISIS COMPLETO: BACKTEST + TIEMPO REAL")
        print("-" * 50)
        
        # 1. Ejecutar backtest
        print("1️⃣ Ejecutando backtest histórico...")
        try:
            result = subprocess.run([sys.executable, "verified_backtester.py"], 
                                  capture_output=False, text=True)
            if result.returncode != 0:
                print("⚠️ Error en backtest, continuando con análisis en tiempo real...")
        except Exception as e:
            print(f"⚠️ Error en backtest: {e}, continuando...")
        
        print("\n" + "="*50)
        
        # 2. Análisis en tiempo real
        print("2️⃣ Análisis en tiempo real...")
        analyzer = RealTimeAnalyzer()
        try:
            result = analyzer.analyze_signal("LINKUSDT")
            analyzer.print_signal_report(result)
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
        
        # 3. Mostrar tracking
        print("\n3️⃣ Estado del tracking:")
        tracker = SignalTracker()
        tracker.update_active_signals()
        tracker.print_performance_report()
    
    else:
        print("❌ Opción inválida")

if __name__ == "__main__":
    main()