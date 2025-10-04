"""
Configuration centralisée pour l'application RGPD + IA ACT
Modifiez ce fichier pour personnaliser l'application
"""

# =============================================================================
# Configuration du Modèle LLM
# =============================================================================

# Modèle Ollama à utiliser
# Options populaires : "gemma3:4b", "gemma2:2b", "llama3:8b", "mistral:7b"
LLM_MODEL = "gemma3:4b"

# Température du modèle (0.0 = très factuel, 1.0 = très créatif)
LLM_TEMPERATURE = 0.1

# URL du serveur Ollama (modifier si Ollama est sur une autre machine)
OLLAMA_BASE_URL = "http://localhost:11434"


# =============================================================================
# Configuration de l'Embedding
# =============================================================================

# Modèle d'embedding à utiliser
# Options : "sentence-transformers/all-MiniLM-L6-v2" (rapide, léger)
#           "sentence-transformers/all-mpnet-base-v2" (meilleur, plus lourd)
#           "sentence-transformers/all-MiniLM-L12-v2" (équilibre performance/taille)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Device pour les embeddings ("cpu" ou "cuda")
EMBEDDING_DEVICE = "cpu"


# =============================================================================
# Configuration de la Base Vectorielle
# =============================================================================

# Répertoire de stockage de ChromaDB
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Nombre de chunks à récupérer pour chaque requête
RETRIEVAL_K = 5


# =============================================================================
# Configuration du Découpage de Documents
# =============================================================================

# Taille des chunks en caractères (augmenté pour préserver les listes complètes)
CHUNK_SIZE = 1500

# Chevauchement entre chunks en caractères (augmenté pour meilleure continuité)
# Un overlap élevé (30-40%) aide à préserver le contexte des énumérations
CHUNK_OVERLAP = 500

# Note: Les séparateurs sont gérés dynamiquement dans l'indexer avec regex
# pour mieux détecter les structures juridiques (énumérations a), b), c), etc.)
CHUNK_SEPARATORS = ["\n\n", "\n", ".", "!", "?", " ", ""]


# =============================================================================
# Configuration des Données
# =============================================================================

# Répertoire contenant les PDFs source
KNOWLEDGE_BASE_DIR = "./knowledge_base"

# Extensions de fichiers à indexer
SUPPORTED_EXTENSIONS = [".pdf"]


# =============================================================================
# Configuration de l'Interface Streamlit
# =============================================================================

# Titre de la page
APP_TITLE = "Assistant de Conformité RGPD + IA ACT"

# Icône de la page (emoji)
APP_ICON = "🛡️"

# Layout ("centered" ou "wide")
APP_LAYOUT = "wide"

# Nombre maximum de questions dans l'historique
MAX_HISTORY = 10

# Afficher les sources par défaut
DEFAULT_SHOW_SOURCES = True

# Nombre de sources à afficher par défaut
DEFAULT_NUM_SOURCES = 5


# =============================================================================
# Configuration du Prompt
# =============================================================================

# Template de prompt pour le LLM
PROMPT_TEMPLATE = """Tu es un assistant expert en conformité juridique, spécialisé dans le RGPD (Règlement Général sur la Protection des Données) et l'IA Act (règlement européen sur l'intelligence artificielle).

Utilise UNIQUEMENT les informations du contexte ci-dessous pour répondre à la question. Si la réponse n'est pas dans le contexte, dis clairement que tu ne peux pas répondre avec certitude.

Contexte:
{context}

Question: {question}

Instructions:
- Réponds de manière précise et structurée
- Cite les articles pertinents quand c'est possible
- Si plusieurs documents sont pertinents (RGPD et IA Act), indique clairement de quel règlement tu parles
- Reste factuel et évite les interprétations
- Si la réponse n'est pas dans le contexte, dis-le clairement

Réponse:"""


# =============================================================================
# Configuration du Logging
# =============================================================================

# Niveau de log ("DEBUG", "INFO", "WARNING", "ERROR")
LOG_LEVEL = "INFO"

# Activer les logs détaillés
VERBOSE_LOGGING = False


# =============================================================================
# Questions Suggérées
# =============================================================================

SUGGESTED_QUESTIONS = {
    "RGPD": [
        "Quels sont les droits des personnes concernées selon le RGPD ?",
        "Quel est le rôle et les responsabilités du DPO ?",
        "Quelles sont les sanctions en cas de non-conformité RGPD ?",
        "Qu'est-ce qu'une analyse d'impact relative à la protection des données ?",
        "Quels sont les principes de traitement des données personnelles ?",
    ],
    "IA Act": [
        "Quelles sont les catégories de risques selon l'IA Act ?",
        "Quelles sont les pratiques d'IA interdites ?",
        "Qu'est-ce qu'un système d'IA à haut risque ?",
        "Quelles sont les obligations pour les fournisseurs de systèmes d'IA ?",
        "Comment l'IA Act définit-il un système d'intelligence artificielle ?",
    ],
    "Transversal": [
        "Comment le RGPD et l'IA Act interagissent-ils en matière de protection des données ?",
        "Quelles sont les exigences communes entre le RGPD et l'IA Act ?",
    ]
}


# =============================================================================
# Fonctions Utilitaires
# =============================================================================

def get_config_summary():
    """Retourne un résumé de la configuration actuelle"""
    return {
        "LLM": {
            "Modèle": LLM_MODEL,
            "Température": LLM_TEMPERATURE,
        },
        "Embedding": {
            "Modèle": EMBEDDING_MODEL,
            "Device": EMBEDDING_DEVICE,
        },
        "Retrieval": {
            "K chunks": RETRIEVAL_K,
        },
        "Chunking": {
            "Taille": CHUNK_SIZE,
            "Overlap": CHUNK_OVERLAP,
        },
    }


if __name__ == "__main__":
    # Afficher la configuration actuelle
    import json
    print("Configuration actuelle :")
    print(json.dumps(get_config_summary(), indent=2, ensure_ascii=False))

