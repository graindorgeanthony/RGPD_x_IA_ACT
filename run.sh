#!/bin/bash

# Script de lancement rapide pour l'Assistant RGPD + IA ACT

echo "============================================================"
echo "🚀 Lancement de l'Assistant RGPD + IA ACT"
echo "============================================================"
echo ""

# Vérifier que l'environnement virtuel existe
if [ ! -d "venv" ]; then
    echo "❌ Environnement virtuel non trouvé."
    echo "💡 Exécutez d'abord : bash setup.sh"
    exit 1
fi

# Activer l'environnement virtuel
echo "🔄 Activation de l'environnement virtuel..."
source venv/bin/activate

# Vérifier que la base vectorielle existe
if [ ! -d "chroma_db" ]; then
    echo "⚠️  Base vectorielle non trouvée."
    echo "💡 Indexation des documents en cours..."
    python indexer.py
    echo ""
fi

# Vérifier qu'Ollama est en cours d'exécution
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama ne semble pas être en cours d'exécution."
    echo "💡 Lancez Ollama avec : ollama serve"
    echo ""
fi

# Lancer l'application
echo "🌐 Lancement de l'interface web..."
streamlit run app.py

