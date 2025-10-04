"""
Script de vérification de l'installation
Vérifie que tous les composants sont correctement installés
"""
import sys
import os
import subprocess


def check_python_version():
    """Vérifie la version de Python"""
    print("\n" + "="*60)
    print("🐍 Vérification de Python")
    print("="*60)
    
    version = sys.version_info
    print(f"Version installée : Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        print("✅ Version Python OK (>= 3.9)")
        return True
    else:
        print("❌ Version Python insuffisante (nécessite >= 3.9)")
        return False


def check_dependencies():
    """Vérifie les dépendances Python"""
    print("\n" + "="*60)
    print("📦 Vérification des dépendances Python")
    print("="*60)
    
    required_packages = {
        "langchain": "0.1.20",
        "langchain_community": "0.0.38",
        "chromadb": "0.4.24",
        "sentence_transformers": "2.7.0",
        "streamlit": "1.32.2",
        "pymupdf": "1.24.1",
        "ollama": "0.1.8",
    }
    
    all_ok = True
    
    for package, expected_version in required_packages.items():
        try:
            if package == "pymupdf":
                import fitz
                module = fitz
            elif package == "langchain_community":
                import langchain_community
                module = langchain_community
            else:
                module = __import__(package)
            
            installed_version = getattr(module, "__version__", "version inconnue")
            print(f"✅ {package:<25} {installed_version}")
        except ImportError:
            print(f"❌ {package:<25} NON INSTALLÉ")
            all_ok = False
    
    return all_ok


def check_ollama():
    """Vérifie qu'Ollama est installé"""
    print("\n" + "="*60)
    print("🤖 Vérification d'Ollama")
    print("="*60)
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama installé : {version}")
            return True
        else:
            print("❌ Ollama non accessible")
            return False
    except FileNotFoundError:
        print("❌ Ollama non installé")
        print("💡 Installer depuis : https://ollama.ai")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        return False


def check_ollama_model():
    """Vérifie que le modèle Ollama est téléchargé"""
    print("\n" + "="*60)
    print("📥 Vérification du modèle Ollama")
    print("="*60)
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "gemma3:4b" in result.stdout or "gemma2" in result.stdout:
            print("✅ Modèle gemma3:4b trouvé")
            return True
        else:
            print("❌ Modèle gemma3:4b non trouvé")
            print("💡 Télécharger avec : ollama pull gemma3:4b")
            return False
    except Exception as e:
        print(f"⚠️ Impossible de vérifier les modèles : {e}")
        return False


def check_pdfs():
    """Vérifie que les PDFs sont présents"""
    print("\n" + "="*60)
    print("📄 Vérification des documents PDF")
    print("="*60)
    
    kb_dir = "./knowledge_base"
    required_pdfs = ["RGPD.pdf", "IA_ACT.pdf"]
    
    if not os.path.exists(kb_dir):
        print(f"❌ Dossier {kb_dir} non trouvé")
        return False
    
    all_ok = True
    
    for pdf in required_pdfs:
        pdf_path = os.path.join(kb_dir, pdf)
        if os.path.exists(pdf_path):
            size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
            print(f"✅ {pdf:<15} ({size:.2f} MB)")
        else:
            print(f"❌ {pdf:<15} NON TROUVÉ")
            all_ok = False
    
    if not all_ok:
        print("\n💡 Placez les PDFs dans le dossier knowledge_base/")
    
    return all_ok


def check_chroma_db():
    """Vérifie que la base vectorielle existe"""
    print("\n" + "="*60)
    print("🗄️  Vérification de la base vectorielle")
    print("="*60)
    
    chroma_dir = "./chroma_db"
    
    if not os.path.exists(chroma_dir):
        print("❌ Base vectorielle non trouvée")
        print("💡 Créer avec : python indexer.py")
        return False
    
    # Compter les fichiers
    num_files = len([f for f in os.listdir(chroma_dir) 
                    if os.path.isfile(os.path.join(chroma_dir, f))])
    
    if num_files > 0:
        print(f"✅ Base vectorielle trouvée ({num_files} fichiers)")
        return True
    else:
        print("⚠️ Base vectorielle vide")
        print("💡 Recréer avec : python indexer.py")
        return False


def check_ollama_server():
    """Vérifie qu'Ollama est en cours d'exécution"""
    print("\n" + "="*60)
    print("🔌 Vérification du serveur Ollama")
    print("="*60)
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            print("✅ Serveur Ollama en cours d'exécution")
            return True
        else:
            print("❌ Serveur Ollama non accessible")
            print("💡 Lancer avec : ollama serve")
            return False
    except ImportError:
        print("⚠️ Package 'requests' non installé, impossible de vérifier")
        return None
    except Exception as e:
        print("❌ Serveur Ollama non accessible")
        print("💡 Lancer avec : ollama serve")
        return False


def generate_report(checks):
    """Génère un rapport final"""
    print("\n" + "="*60)
    print("📊 RAPPORT FINAL")
    print("="*60)
    
    passed = sum(1 for result in checks.values() if result is True)
    failed = sum(1 for result in checks.values() if result is False)
    skipped = sum(1 for result in checks.values() if result is None)
    total = len(checks)
    
    print(f"\n✅ Réussi  : {passed}/{total}")
    print(f"❌ Échoué  : {failed}/{total}")
    if skipped > 0:
        print(f"⚠️ Ignoré   : {skipped}/{total}")
    
    print("\n" + "="*60)
    
    for check_name, result in checks.items():
        if result is True:
            status = "✅"
        elif result is False:
            status = "❌"
        else:
            status = "⚠️"
        print(f"{status} {check_name}")
    
    print("="*60)
    
    if failed == 0:
        print("\n🎉 Tous les composants sont installés !")
        print("🚀 Vous pouvez lancer l'application avec : streamlit run app.py")
        return True
    else:
        print(f"\n⚠️ {failed} composant(s) manquant(s)")
        print("💡 Consultez les messages ci-dessus pour corriger les problèmes")
        return False


def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("🔍 VÉRIFICATION DE L'INSTALLATION")
    print("Assistant RGPD + IA ACT")
    print("="*60)
    
    checks = {}
    
    # Vérifications
    checks["Python >= 3.9"] = check_python_version()
    checks["Dépendances Python"] = check_dependencies()
    checks["Ollama installé"] = check_ollama()
    checks["Modèle gemma3:4b"] = check_ollama_model()
    checks["Documents PDF"] = check_pdfs()
    checks["Base vectorielle"] = check_chroma_db()
    checks["Serveur Ollama"] = check_ollama_server()
    
    # Rapport final
    success = generate_report(checks)
    
    if success:
        print("\n" + "="*60)
        print("📝 PROCHAINES ÉTAPES :")
        print("="*60)
        print("1. Si la base vectorielle n'existe pas :")
        print("   → python indexer.py")
        print("\n2. Lancer l'application :")
        print("   → streamlit run app.py")
        print("\n3. Ou utiliser le script de lancement :")
        print("   → bash run.sh")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("🛠️ ACTIONS RECOMMANDÉES :")
        print("="*60)
        
        if not checks.get("Python >= 3.9"):
            print("• Installer Python 3.9+ depuis https://python.org")
        
        if not checks.get("Dépendances Python"):
            print("• Installer les dépendances : pip install -r requirements.txt")
        
        if not checks.get("Ollama installé"):
            print("• Installer Ollama depuis https://ollama.ai")
        
        if not checks.get("Modèle gemma3:4b"):
            print("• Télécharger le modèle : ollama pull gemma3:4b")
        
        if not checks.get("Documents PDF"):
            print("• Placer RGPD.pdf et IA_ACT.pdf dans knowledge_base/")
        
        if not checks.get("Base vectorielle"):
            print("• Créer la base vectorielle : python indexer.py")
        
        if checks.get("Serveur Ollama") is False:
            print("• Lancer Ollama : ollama serve")
        
        print("="*60)
    
    print("\n💡 Pour plus d'aide, consultez : README.md ou QUICK_START.md\n")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

