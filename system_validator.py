#!/usr/bin/env python3
"""
SISTEMA DE VALIDACIÓN CONTINUA
==============================
Verifica que el sistema de trading sigue funcionando correctamente
"""

import subprocess
import json
import os
from datetime import datetime

class SystemValidator:
    def __init__(self):
        self.project_root = "C:\\Proyectos\\trading\\python-analysis-project"
        self.expected_metrics = {
            "min_roi": 300.0,        # ROI mínimo esperado
            "min_win_rate": 45.0,    # Win rate mínimo
            "min_profit_factor": 1.3, # Profit factor mínimo
            "max_drawdown": 80.0,    # Drawdown máximo aceptable
            "min_trades": 50         # Mínimo número de trades
        }
        
    def run_full_validation(self):
        """Ejecutar validación completa del sistema."""
        
        print("🔍 VALIDACIÓN COMPLETA DEL SISTEMA")
        print("=" * 50)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "validations": {}
        }
        
        # 1. Validación de sintaxis
        print("1️⃣ Validando sintaxis...")
        syntax_ok = self._validate_syntax()
        results["validations"]["syntax"] = syntax_ok
        
        # 2. Validación matemática
        print("2️⃣ Validando precisión matemática...")
        math_ok = self._validate_mathematics()
        results["validations"]["mathematics"] = math_ok
        
        # 3. Validación de backtest
        print("3️⃣ Validando performance de backtest...")
        backtest_ok, metrics = self._validate_backtest()
        results["validations"]["backtest"] = backtest_ok
        results["metrics"] = metrics
        
        # 4. Validación de archivos críticos
        print("4️⃣ Validando archivos críticos...")
        files_ok = self._validate_critical_files()
        results["validations"]["files"] = files_ok
        
        # Resumen final
        all_ok = all(results["validations"].values())
        
        print("\n📊 RESUMEN DE VALIDACIÓN")
        print("=" * 30)
        
        for validation, status in results["validations"].items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {validation.title()}: {'OK' if status else 'FALLO'}")
        
        if all_ok:
            print("\n🎉 ¡SISTEMA COMPLETAMENTE VALIDADO!")
            print("✅ Todos los tests pasaron correctamente")
            print("🚀 Sistema listo para trading")
        else:
            print("\n⚠️ PROBLEMAS DETECTADOS")
            print("❌ Algunos tests fallaron")
            print("🔧 Revisa los errores y restaura versión estable si es necesario")
        
        # Guardar reporte
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📝 Reporte guardado en: {report_file}")
        
        return all_ok, results
    
    def _validate_syntax(self):
        """Validar que no hay errores de sintaxis."""
        
        critical_files = [
            "verified_backtester.py",
            "real_time_trading_system.py",
            "mathematical_validator.py"
        ]
        
        for file in critical_files:
            file_path = os.path.join(self.project_root, file)
            if not os.path.exists(file_path):
                print(f"❌ Archivo faltante: {file}")
                return False
            
            try:
                result = subprocess.run(
                    ["python", "-m", "py_compile", file_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"   ✅ {file} - Sintaxis OK")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {file} - Error de sintaxis:")
                print(f"      {e.stderr}")
                return False
        
        return True
    
    def _validate_mathematics(self):
        """Validar precisión matemática contra TA-Lib."""
        
        try:
            result = subprocess.run(
                ["python", "mathematical_validator.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout
            
            # Buscar indicadores de validación exitosa
            atr_ok = "0.000%" in output and "ATR" in output
            rsi_ok = "0.00 points" in output and "RSI" in output
            
            if atr_ok and rsi_ok:
                print("   ✅ ATR y RSI con precisión 100% vs TA-Lib")
                return True
            else:
                print("   ❌ Diferencias detectadas en cálculos matemáticos")
                return False
                
        except Exception as e:
            print(f"   ❌ Error ejecutando validación matemática: {e}")
            return False
    
    def _validate_backtest(self):
        """Validar performance del backtest."""
        
        try:
            result = subprocess.run(
                ["python", "verified_backtester.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            output = result.stdout
            
            # Extraer métricas del output
            metrics = self._extract_metrics_from_output(output)
            
            if not metrics:
                print("   ❌ No se pudieron extraer métricas del backtest")
                return False, {}
            
            # Validar métricas contra límites
            validations = {
                "roi": metrics.get("roi", 0) >= self.expected_metrics["min_roi"],
                "win_rate": metrics.get("win_rate", 0) >= self.expected_metrics["min_win_rate"],
                "profit_factor": metrics.get("profit_factor", 0) >= self.expected_metrics["min_profit_factor"],
                "drawdown": metrics.get("max_drawdown", 100) <= self.expected_metrics["max_drawdown"],
                "trades": metrics.get("total_trades", 0) >= self.expected_metrics["min_trades"]
            }
            
            all_metrics_ok = all(validations.values())
            
            for metric, passed in validations.items():
                status = "✅" if passed else "❌"
                value = metrics.get(metric.replace("_", " "), "N/A")
                print(f"   {status} {metric.title()}: {value}")
            
            return all_metrics_ok, metrics
            
        except Exception as e:
            print(f"   ❌ Error ejecutando backtest: {e}")
            return False, {}
    
    def _extract_metrics_from_output(self, output):
        """Extraer métricas del output del backtester."""
        
        metrics = {}
        
        try:
            lines = output.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if "ROI total:" in line:
                    roi_str = line.split(":")[-1].strip().replace("%", "")
                    metrics["roi"] = float(roi_str)
                
                elif "Trades ganadores:" in line and "(" in line:
                    # Extraer win rate del paréntesis
                    paren_content = line.split("(")[1].split(")")[0]
                    win_rate_str = paren_content.replace("%", "")
                    metrics["win_rate"] = float(win_rate_str)
                
                elif "Profit Factor:" in line:
                    pf_str = line.split(":")[-1].strip()
                    metrics["profit_factor"] = float(pf_str)
                
                elif "Max Drawdown:" in line and "(" in line:
                    # Extraer drawdown del paréntesis
                    paren_content = line.split("(")[1].split(")")[0]
                    drawdown_str = paren_content.replace("%", "")
                    metrics["max_drawdown"] = float(drawdown_str)
                
                elif "Total trades:" in line:
                    trades_str = line.split(":")[-1].strip()
                    metrics["total_trades"] = int(trades_str)
        
        except Exception as e:
            print(f"Error extrayendo métricas: {e}")
            
        return metrics
    
    def _validate_critical_files(self):
        """Validar que todos los archivos críticos existen."""
        
        critical_files = [
            "verified_backtester.py",
            "real_time_trading_system.py",
            "mathematical_validator.py",
            "trading_conditions_analyzer.py",
            ".copilot-instructions.md",
            "version_manager.py",
            "MAINTENANCE_GUIDE.md"
        ]
        
        all_exist = True
        
        for file in critical_files:
            file_path = os.path.join(self.project_root, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - FALTANTE")
                all_exist = False
        
        return all_exist

def main():
    """Función principal de validación."""
    
    validator = SystemValidator()
    success, results = validator.run_full_validation()
    
    if not success:
        print("\n🚨 ACCIÓN RECOMENDADA:")
        print("1. Revisa los errores mostrados arriba")
        print("2. Si es crítico, restaura la versión dorada:")
        print("   python version_manager.py interactive")
        print("   (Opción 3: Restaurar -> sistema_optimizado_v1_GOLDEN)")
        
    return success

if __name__ == "__main__":
    main()