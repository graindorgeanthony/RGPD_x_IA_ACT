# 🛡️ Assistant de Conformité RGPD + IA ACT

Une application web locale permettant de poser des questions sur le **RGPD** (Règlement Général sur la Protection des Données) et l'**IA Act** (règlement européen sur l'intelligence artificielle) avec des réponses précises, sourcées et **100% confidentielles**.

## 🎯 Caractéristiques

- ✅ **Traitement 100% local** : Aucune donnée n'est envoyée vers le cloud
- ✅ **Confidentialité absolue** : Toutes les données restent sur votre machine
- ✅ **Réponses sourcées avec citations** : Chaque réponse cite précisément les sources [Source X] utilisées
- ✅ **RAG (Retrieval Augmented Generation)** : Évite les hallucinations en ancrant les réponses sur des sources vérifiées
- ✅ **Interface moderne avec streaming** : Réponses affichées en temps réel avec mise en forme élégante
- ✅ **Double expertise** : RGPD et IA Act dans une seule application
- ✅ **Chunking intelligent** : Préservation des structures juridiques (articles, énumérations, chapitres)
- ✅ **Configuration flexible** : Personnalisation facile via `config.py`

## ⚡ Démarrage Rapide

```bash
# 1. Installation automatique
bash setup.sh

# 2. Télécharger le modèle Ollama
ollama pull gemma3:4b

# 3. Placer les PDFs dans knowledge_base/
# (RGPD.pdf et IA_ACT.pdf)

# 4. Vérifier l'installation
python check_setup.py

# 5. Lancer l'application
bash run.sh
```

🎉 L'application s'ouvre automatiquement dans votre navigateur !

## 🔧 Technologies Utilisées

| Composant | Technologie | Version | Rôle |
|-----------|-------------|---------|------|
| **LLM Local** | Ollama (gemma3:4b) | 0.4.2 | Génération de réponses avec streaming |
| **Framework RAG** | LangChain | 0.3.7+ | Orchestration du pipeline RAG |
| **Base Vectorielle** | ChromaDB | 0.5.3 | Stockage des embeddings |
| **Embeddings** | Sentence Transformers | 3.1.1 | Vectorisation sémantique |
| **Interface** | Streamlit | 1.41.0 | Interface web moderne et réactive |
| **Chargement PDF** | PyMuPDF | 1.25.2+ | Extraction intelligente du texte |
| **Formatage** | Markdown | 3.7 | Rendu élégant des réponses |

## 📋 Prérequis

1. **Python 3.9+** installé sur votre système
2. **Ollama** installé (voir [ollama.ai](https://ollama.ai))
3. Les PDFs officiels du RGPD et de l'IA Act

## 🚀 Installation

### Option A : Installation automatique (recommandée)

```bash
cd RGPD_x_IA_ACT
bash setup.sh
```

Le script `setup.sh` va :
1. ✅ Créer l'environnement virtuel
2. ✅ Installer toutes les dépendances
3. ✅ Afficher les prochaines étapes

### Option B : Installation manuelle

#### 1. Cloner ou télécharger le projet

```bash
cd RGPD_x_IA_ACT
```

#### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# Sur macOS/Linux
source venv/bin/activate

# Sur Windows
venv\Scripts\activate
```

#### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 4. Installer et configurer Ollama

```bash
# Télécharger Ollama depuis https://ollama.ai
# Puis installer le modèle gemma3:4b

ollama pull gemma3:4b
```

#### 5. Préparer les documents

Placez vos PDFs dans le dossier `knowledge_base/` :
- `knowledge_base/RGPD.pdf`
- `knowledge_base/IA_ACT.pdf`

> 💡 **Sources officielles** :
> - RGPD : [https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679)
> - IA Act : [https://artificialintelligenceact.eu/](https://artificialintelligenceact.eu/)

### 🔍 Vérification de l'installation

Une fois l'installation terminée, vérifiez que tout est correctement configuré :

```bash
python check_setup.py
```

Ce script vérifie :
- ✅ Version de Python
- ✅ Dépendances installées
- ✅ Ollama et modèle gemma3:4b
- ✅ Documents PDF présents
- ✅ Serveur Ollama actif

## 📊 Utilisation

### Méthode rapide (recommandée)

Pour lancer l'application en une seule commande :

```bash
bash run.sh
```

Le script `run.sh` va :
1. ✅ Activer l'environnement virtuel
2. ✅ Vérifier la base vectorielle (et l'indexer si nécessaire)
3. ✅ Vérifier qu'Ollama est actif
4. ✅ Lancer l'application Streamlit

### Méthode manuelle

#### Étape 1 : Indexation des documents

Avant la première utilisation, vous devez indexer les documents PDF :

```bash
python indexer.py
```

Cette commande va :
1. 📄 Charger les PDFs depuis `knowledge_base/`
2. ✂️ Découper les documents en chunks intelligents (1500 chars, overlap 400)
3. 🧹 Nettoyer et structurer le texte juridique
4. 🔢 Vectoriser chaque chunk avec Sentence Transformers
5. 💾 Stocker les embeddings dans ChromaDB (`chroma_db/`)
6. 🏷️ Enrichir les métadonnées (articles, chapitres, termes clés)

**Sortie attendue :**
```
============================================================
🚀 Démarrage de l'indexation
============================================================

📚 2 fichiers PDF détectés : RGPD.pdf, IA_ACT.pdf

📄 Chargement du fichier : ./knowledge_base/RGPD.pdf
✅ 88 pages chargées
✂️  Découpage intelligent en chunks (taille=1500, overlap=400)
🧹 Application du LegalTextCleanerTransformer...
✅ X chunks créés avec métadonnées enrichies
...
✅ Indexation terminée avec succès !
```

> ⚠️ **Note** : L'indexation ne doit être effectuée qu'une seule fois, sauf si vous modifiez les PDFs source.

#### Étape 2 : Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

#### Étape 3 : Poser des questions

1. Tapez votre question dans le champ de texte
2. Cliquez sur **🔍 Rechercher** ou utilisez les questions suggérées
3. La réponse s'affiche en temps réel (streaming)
4. Consultez les citations [Source X] dans la réponse
5. Explorez les sources utilisées (surlignées si citées)

## 💡 Exemples de Questions

### RGPD
- "Quels sont les droits des personnes concernées selon le RGPD ?"
- "Quel est le rôle et les responsabilités du DPO ?"
- "Quelles sont les sanctions en cas de non-conformité RGPD ?"
- "Qu'est-ce qu'une analyse d'impact relative à la protection des données ?"

### IA Act
- "Quelles sont les catégories de risques selon l'IA Act ?"
- "Quelles sont les pratiques d'IA interdites ?"
- "Qu'est-ce qu'un système d'IA à haut risque ?"
- "Quelles sont les obligations pour les fournisseurs de systèmes d'IA ?"

### Questions transversales
- "Comment le RGPD et l'IA Act interagissent-ils en matière de protection des données ?"
- "Quelles sont les exigences communes entre le RGPD et l'IA Act ?"

## 🌟 Fonctionnalités Avancées

### 📌 Système de Citations Intelligentes

L'application génère automatiquement des citations **[Source X]** dans les réponses :
- ✅ Citations placées en fin de paragraphe (style naturel)
- ✅ Sources citées surlignées en **vert** avec badge ✅
- ✅ Sources non citées affichées pour transparence avec 📄
- ✅ Statistiques : "X/Y sources citées"
- ✅ Expansion automatique des sources citées

### 🎨 Réponses Formatées en Markdown

Les réponses utilisent une mise en forme professionnelle :
- **Titres et sous-titres** (## et ###) pour structurer
- **Listes à puces** pour énumérations claires
- **Gras et emphase** pour points importants
- **Paragraphes courts** (3-4 phrases) pour lisibilité
- **Style executive summary** avec résumé initial

### 🧹 Nettoyage Intelligent des Textes Juridiques

Le `LegalTextCleanerTransformer` préserve la structure :
- ✅ Énumérations a), b), c) gardées intactes
- ✅ Ponctuation française correcte (espaces insécables)
- ✅ Détection des fragments incomplets `[...]`
- ✅ Préservation des articles, chapitres, sections
- ✅ Score de qualité pour chaque chunk (0-1)

### 📊 Métadonnées Enrichies

Chaque source affiche :
- **Type de contenu** : Article, Chapitre, Section, Paragraphe
- **Termes clés juridiques** détectés automatiquement
- **Qualité du chunk** avec indicateur visuel (✅/⚠️)
- **Informations structurelles** (Article X, Chapitre Y)
- **Compteur de mots** et longueur

### ⚡ Interface Réactive

- **Streaming en temps réel** : Réponse affichée mot par mot
- **Loader animé** : Indicateur visuel pendant la recherche
- **Slider dynamique** : Ajustez le nombre de sources (10-50)
- **Historique** : Consultez les 5 dernières questions
- **Questions suggérées** : Boutons rapides pour démarrer

## 📁 Structure du Projet

```
RGPD_x_IA_ACT/
├── app.py                  # Interface Streamlit avec streaming
├── rag_chain.py            # Chaîne RAG avec citations inline
├── indexer.py              # Script d'indexation intelligente
├── config.py               # Configuration centralisée
├── check_setup.py          # Vérification de l'installation
├── requirements.txt        # Dépendances Python
├── setup.sh                # Script d'installation automatique
├── run.sh                  # Script de lancement rapide
├── readme.md               # Ce fichier
├── .gitignore              # Fichiers à ignorer par Git
├── knowledge_base/         # Dossier contenant les PDFs
│   ├── RGPD.pdf
│   └── IA_ACT.pdf
└── chroma_db/              # Base vectorielle (créée après indexation)
    └── [fichiers ChromaDB avec métadonnées enrichies]
```

## 🔍 Comment ça Marche ?

### Architecture RAG (Retrieval Augmented Generation)

```
┌─────────────┐
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  Vectorisation          │
│  (Sentence Transformers)│
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Recherche Sémantique   │
│  dans ChromaDB          │
│  (Top N chunks)         │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Construction du Prompt │
│  Question + Contexte    │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  LLM Local (Ollama)     │
│  Génération de réponse  │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Réponse + Sources      │
└─────────────────────────┘
```

### Processus détaillé

1. **Indexation intelligente** (une seule fois)
   - Extraction du texte des PDFs avec PyMuPDF
   - Découpage intelligent en chunks de 1500 caractères avec overlap de 400
   - Application du `LegalTextCleanerTransformer` (LangChain Document Transformer)
     - Nettoyage de la ponctuation française
     - Préservation des structures (articles, énumérations a), b), c))
     - Détection des fragments incomplets
   - Enrichissement des métadonnées :
     - Extraction des articles, chapitres, sections
     - Identification des termes juridiques clés
     - Calcul de la qualité du chunk (score 0-1)
   - Génération d'embeddings avec Sentence Transformers
   - Stockage dans ChromaDB avec métadonnées complètes

2. **Interrogation avec citations** (à chaque question)
   - Vectorisation de la question
   - Recherche sémantique des N chunks les plus similaires (configurable 10-50)
   - Numérotation des sources [Source 1], [Source 2], etc.
   - Construction d'un prompt structuré avec contexte numéroté
   - Génération de la réponse par le LLM local avec streaming en temps réel
   - Extraction automatique des citations [Source X] dans la réponse
   - Affichage avec mise en forme Markdown élégante
   - Sources citées surlignées en vert, non-citées affichées pour transparence

## ⚙️ Configuration et Personnalisation

Le fichier `config.py` permet de personnaliser facilement l'application :

### Configuration du Modèle LLM

```python
LLM_MODEL = "gemma3:4b"          # Modèle Ollama à utiliser
LLM_TEMPERATURE = 0.1            # Température (0.0 = factuel, 1.0 = créatif)
OLLAMA_BASE_URL = "http://localhost:11434"  # URL du serveur Ollama
```

**Modèles alternatifs :**
- `gemma2:2b` - Plus rapide, moins précis
- `llama3:8b` - Plus précis, plus lent
- `mistral:7b` - Bon équilibre

### Configuration du Chunking

```python
CHUNK_SIZE = 1500      # Taille des chunks (caractères)
CHUNK_OVERLAP = 500    # Chevauchement entre chunks
```

**Recommandations :**
- Documents techniques : `CHUNK_SIZE = 1500`, `CHUNK_OVERLAP = 400-500`
- Documents narratifs : `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200`

### Configuration de la Recherche

```python
RETRIEVAL_K = 5        # Nombre de chunks à récupérer par défaut
```

**Dans l'interface :**
- Slider pour ajuster entre 10-50 sources en temps réel
- Plus de sources = réponses plus complètes mais plus lentes

### Configuration des Embeddings

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # ou "cuda" si GPU disponible
```

**Modèles alternatifs :**
- `all-MiniLM-L6-v2` - Rapide et léger (défaut)
- `all-mpnet-base-v2` - Meilleure qualité, plus lourd
- `all-MiniLM-L12-v2` - Équilibre performance/taille

Pour voir la configuration actuelle :
```bash
python config.py
```

## 🐛 Dépannage

### Erreur : "Base de données vectorielle non trouvée"
➡️ Exécutez d'abord `python indexer.py` pour créer la base

### Erreur : "Ollama connection error"
➡️ Vérifiez qu'Ollama est en cours d'exécution : `ollama serve`
➡️ Vérifiez l'URL dans `config.py` : `OLLAMA_BASE_URL`

### Erreur : "Model not found"
➡️ Téléchargez le modèle : `ollama pull gemma3:4b`
➡️ Vérifiez que le modèle configuré dans `config.py` est disponible : `ollama list`

### Réponses lentes
➡️ Utilisez un modèle plus petit dans `config.py` : `LLM_MODEL = "gemma2:2b"`
➡️ Réduisez le nombre de sources dans l'interface (slider)
➡️ Réduisez `CHUNK_SIZE` dans `config.py` pour des chunks plus petits

### Réponses imprécises ou incomplètes
➡️ Augmentez le nombre de sources dans l'interface (slider vers 30-50)
➡️ Vérifiez que `LLM_TEMPERATURE` est bas (0.1) dans `config.py`
➡️ Réindexez avec un `CHUNK_SIZE` plus grand pour plus de contexte

### Les énumérations sont coupées
➡️ Le `LegalTextCleanerTransformer` devrait préserver les structures
➡️ Augmentez `CHUNK_OVERLAP` dans `config.py` (essayez 500-600)
➡️ Réindexez avec `python indexer.py`

### Diagnostic complet
➡️ Utilisez le script de vérification : `python check_setup.py`

## 🎓 Compétences Démontrées

Ce projet démontre une maîtrise technique avancée :

### Architecture et IA
- ✅ **RAG (Retrieval Augmented Generation)** avec système de citations inline
- ✅ **LLM locaux** (Ollama) pour la confidentialité totale
- ✅ **Streaming en temps réel** pour une UX réactive
- ✅ **Bases vectorielles** (ChromaDB) avec métadonnées enrichies
- ✅ **Embeddings sémantiques** (Sentence Transformers)

### Traitement de Texte Avancé
- ✅ **Document Transformers** (LangChain) pour le nettoyage intelligent
- ✅ **Chunking intelligent** avec préservation des structures juridiques
- ✅ **Regex avancés** pour la détection d'énumérations et structures
- ✅ **Métadonnées enrichies** (articles, chapitres, termes clés, qualité)
- ✅ **Ponctuation française** correcte (espaces insécables, guillemets)

### Développement et UX
- ✅ **Interface web moderne** (Streamlit) avec design élégant
- ✅ **Configuration centralisée** pour faciliter la personnalisation
- ✅ **Scripts d'automatisation** (setup.sh, run.sh, check_setup.py)
- ✅ **Gestion d'erreurs robuste** avec messages explicites
- ✅ **Markdown rendering** pour réponses structurées

### Domaine Métier
- ✅ **Compréhension du domaine juridique** (RGPD, IA Act)
- ✅ **Traitement de documents réglementaires** complexes
- ✅ **Conformité et confidentialité** par conception

## 📝 Licence

Ce projet est à usage éducatif et de démonstration. Les textes du RGPD et de l'IA Act sont des documents officiels de l'Union Européenne.

## 🤝 Contribution

Pour toute question ou amélioration, n'hésitez pas à ouvrir une issue ou une pull request.

## 👨‍💻 Auteur

**Anthony GRAINDORGE**
- Consultant & Développeur IA
- Spécialisé en RAG, LLM locaux et conformité

---

**Développé avec ❤️ pour la conformité, la confidentialité et l'innovation responsable**

