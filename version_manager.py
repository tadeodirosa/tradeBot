#!/usr/bin/env python3
"""
SISTEMA DE RESPALDO Y VERSIONADO AUTOMÁTICO
==========================================
Protege las versiones exitosas del sistema de trading y permite rollback seguro
"""

import os
import shutil
import json
from datetime import datetime
import subprocess

class TradingSystemBackup:
    def __init__(self):
        self.project_root = "C:\\Proyectos\\trading\\python-analysis-project"
        self.backup_root = "C:\\Proyectos\\trading\\backups"
        self.versions_file = os.path.join(self.backup_root, "versions.json")
        
        # Crear directorio de backups si no existe
        os.makedirs(self.backup_root, exist_ok=True)
        
    def create_version_backup(self, version_name: str, description: str, performance_metrics: dict = None):
        """Crear respaldo completo de la versión actual."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{version_name}_{timestamp}"
        backup_path = os.path.join(self.backup_root, version_id)
        
        print(f"🔄 CREANDO RESPALDO: {version_name}")
        print("=" * 50)
        
        # Crear directorio de la versión
        os.makedirs(backup_path, exist_ok=True)
        
        # Archivos críticos a respaldar
        critical_files = [
            "verified_backtester.py",
            "real_time_trading_system.py", 
            "mathematical_validator.py",
            "trading_conditions_analyzer.py",
            ".copilot-instructions.md",
            "copilot_instructions.md",
            "requirements.txt"
        ]
        
        # Copiar archivos críticos
        backed_up_files = []
        for file in critical_files:
            source_path = os.path.join(self.project_root, file)
            if os.path.exists(source_path):
                dest_path = os.path.join(backup_path, file)
                shutil.copy2(source_path, dest_path)
                backed_up_files.append(file)
                print(f"✅ Respaldado: {file}")
        
        # Respaldar resultados de backtesting
        backtest_results = [f for f in os.listdir(self.project_root) 
                          if f.startswith("verified_backtest_") and f.endswith(".json")]
        
        if backtest_results:
            results_dir = os.path.join(backup_path, "backtest_results")
            os.makedirs(results_dir, exist_ok=True)
            for result_file in backtest_results:
                shutil.copy2(os.path.join(self.project_root, result_file), 
                           os.path.join(results_dir, result_file))
            print(f"✅ Respaldados {len(backtest_results)} resultados de backtest")
        
        # Crear manifiesto de la versión
        version_manifest = {
            "version_id": version_id,
            "version_name": version_name,
            "description": description,
            "timestamp": timestamp,
            "backup_path": backup_path,
            "files_backed_up": backed_up_files,
            "performance_metrics": performance_metrics or {},
            "git_commit": self._get_git_commit() if self._is_git_repo() else None
        }
        
        # Guardar manifiesto
        manifest_path = os.path.join(backup_path, "version_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(version_manifest, f, indent=2)
        
        # Actualizar registro de versiones
        self._update_versions_registry(version_manifest)
        
        print(f"✅ Respaldo completo creado en: {backup_path}")
        print(f"📊 Archivos respaldados: {len(backed_up_files)}")
        
        return version_id
    
    def list_versions(self):
        """Listar todas las versiones disponibles."""
        
        if not os.path.exists(self.versions_file):
            print("❌ No hay versiones respaldadas")
            return []
        
        with open(self.versions_file, 'r') as f:
            versions_data = json.load(f)
        
        print("📦 VERSIONES DISPONIBLES:")
        print("=" * 60)
        
        for version in versions_data.get("versions", []):
            metrics = version.get("performance_metrics", {})
            roi = metrics.get("roi", "N/A")
            win_rate = metrics.get("win_rate", "N/A")
            profit_factor = metrics.get("profit_factor", "N/A")
            
            print(f"🏷️  {version['version_name']} ({version['timestamp']})")
            print(f"   📝 {version['description']}")
            print(f"   📊 ROI: {roi}% | Win Rate: {win_rate}% | PF: {profit_factor}")
            print(f"   📁 {version['backup_path']}")
            print()
        
        return versions_data.get("versions", [])
    
    def restore_version(self, version_name: str):
        """Restaurar una versión específica."""
        
        versions = self._load_versions()
        target_version = None
        
        for version in versions:
            if version_name in version['version_name']:
                target_version = version
                break
        
        if not target_version:
            print(f"❌ Versión '{version_name}' no encontrada")
            return False
        
        print(f"🔄 RESTAURANDO VERSIÓN: {target_version['version_name']}")
        print("=" * 50)
        
        backup_path = target_version['backup_path']
        
        # Crear respaldo de la versión actual antes de restaurar
        current_backup = self.create_version_backup(
            "pre_restore_backup",
            f"Respaldo automático antes de restaurar {target_version['version_name']}"
        )
        print(f"🛡️  Versión actual respaldada como: {current_backup}")
        
        # Restaurar archivos
        restored_files = []
        for file in target_version['files_backed_up']:
            source_path = os.path.join(backup_path, file)
            dest_path = os.path.join(self.project_root, file)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                restored_files.append(file)
                print(f"✅ Restaurado: {file}")
        
        print(f"✅ Versión '{target_version['version_name']}' restaurada exitosamente")
        print(f"📊 Archivos restaurados: {len(restored_files)}")
        
        return True
    
    def _update_versions_registry(self, version_manifest):
        """Actualizar el registro de versiones."""
        
        versions_data = {"versions": []}
        
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
        
        versions_data["versions"].append(version_manifest)
        
        # Mantener solo las últimas 10 versiones
        versions_data["versions"] = versions_data["versions"][-10:]
        
        with open(self.versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2)
    
    def _load_versions(self):
        """Cargar lista de versiones."""
        if not os.path.exists(self.versions_file):
            return []
        
        with open(self.versions_file, 'r') as f:
            versions_data = json.load(f)
        
        return versions_data.get("versions", [])
    
    def _is_git_repo(self):
        """Verificar si el proyecto está en un repositorio git."""
        return os.path.exists(os.path.join(self.project_root, ".git"))
    
    def _get_git_commit(self):
        """Obtener el commit actual de git."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                cwd=self.project_root, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except:
            return None

def create_golden_version():
    """Crear automáticamente la versión dorada actual."""
    
    backup_system = TradingSystemBackup()
    
    # Métricas de la versión dorada actual
    current_metrics = {
        "roi": 427.86,
        "win_rate": 50.8,
        "profit_factor": 1.46,
        "max_drawdown": 69.2,
        "total_trades": 120,
        "description": "Sistema optimizado con condiciones selectivas y gestión de riesgo mejorada"
    }
    
    version_id = backup_system.create_version_backup(
        "sistema_optimizado_v1_GOLDEN",
        "Versión DORADA estable: ROI 427.86%, Win Rate 50.8%, PF 1.46, Drawdown 69.2%",
        current_metrics
    )
    
    print(f"\n🎉 ¡VERSIÓN DORADA CREADA!")
    print(f"📦 ID: {version_id}")
    print("🛡️  Esta versión está protegida y puede ser restaurada en cualquier momento")
    print("\n🔧 Para gestionar versiones manualmente, ejecuta:")
    print("python version_manager.py interactive")
    
    return version_id

def main():
    """Función principal para gestión de versiones."""
    
    import sys
    
    # Si se pasa 'interactive' como argumento, usar modo interactivo
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        # Por defecto, crear versión dorada
        create_golden_version()

def interactive_mode():
    """Modo interactivo para gestión de versiones."""
    
    backup_system = TradingSystemBackup()
    
    print("🔧 SISTEMA DE GESTIÓN DE VERSIONES")
    print("=" * 50)
    print("1. Crear nueva versión")
    print("2. Listar versiones")
    print("3. Restaurar versión")
    print("4. Crear versión de la configuración actual")
    
    choice = input("\nSelecciona una opción (1-4): ").strip()
    
    if choice == "1":
        version_name = input("Nombre de la versión: ").strip()
        description = input("Descripción: ").strip()
        backup_system.create_version_backup(version_name, description)
        
    elif choice == "2":
        backup_system.list_versions()
        
    elif choice == "3":
        backup_system.list_versions()
        version_name = input("\nNombre de la versión a restaurar: ").strip()
        backup_system.restore_version(version_name)
        
    elif choice == "4":
        create_golden_version()

if __name__ == "__main__":
    main()