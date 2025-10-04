#!/bin/bash

# Script d'installation pour l'Assistant RGPD + IA ACT
# Ce script facilite l'installation et la configuration du projet

echo "============================================================"
echo "🛡️  Installation de l'Assistant RGPD + IA ACT"
echo "============================================================"
echo ""

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

echo "✅ Python3 détecté: $(python3 --version)"
echo ""

# Créer l'environnement virtuel
echo "📦 Création de l'environnement virtuel..."
python3 -m venv venv

# Activer l'environnement virtuel
echo "🔄 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "📥 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "✅ Installation terminée avec succès !"
echo "============================================================"
echo ""
echo "📋 Prochaines étapes :"
echo ""
echo "1. Vérifiez qu'Ollama est installé :"
echo "   👉 https://ollama.ai"
echo ""
echo "2. Téléchargez le modèle gemma3:4b :"
echo "   👉 ollama pull gemma3:4b"
echo ""
echo "3. Placez vos PDFs dans le dossier knowledge_base/ :"
echo "   - RGPD.pdf"
echo "   - IA_ACT.pdf"
echo ""
echo "4. Indexez les documents :"
echo "   👉 python indexer.py"
echo ""
echo "5. Lancez l'application :"
echo "   👉 streamlit run app.py"
echo ""
echo "============================================================"

