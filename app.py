"""
Application Streamlit - Assistant de Conformité RGPD + IA ACT
Interface web pour poser des questions sur le RGPD et l'IA Act
"""
import os
import warnings

# Disable telemetry and warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress all common warnings
warnings.filterwarnings("ignore", message="Failed to send telemetry event")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import streamlit as st
from rag_chain import RAGChain


# Configuration de la page
st.set_page_config(
    page_title="Assistant RGPD + IA ACT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        line-height: 1.8;
    }
    .answer-box h2 {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .answer-box h3 {
        color: #2c5aa0;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .answer-box ul, .answer-box ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    .answer-box li {
        margin-bottom: 0.5rem;
    }
    .answer-box p {
        margin-bottom: 1rem;
    }
    .answer-box strong {
        font-weight: 600;
        color: #333;
    }
    /* Style pour les citations inline */
    .answer-box [href*="Source"] {
        background-color: #ffd700;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: 600;
        color: #333;
        text-decoration: none;
        font-size: 0.85em;
        white-space: nowrap;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .loader-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 1rem 0;
    }
    .loader-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #1f77b4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 0.5rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .loader-text {
        color: #1f77b4;
        font-weight: 500;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def show_loader(message: str):
    """Affiche un loader animé avec un message"""
    loader_html = f"""
    <div class="loader-container">
        <div class="loader-spinner"></div>
        <div class="loader-text">{message}</div>
    </div>
    """
    return loader_html


def highlight_citations(text: str) -> str:
    """
    Transforme les citations [Source X] en badges HTML stylisés et convertit le markdown en HTML
    
    Args:
        text: Texte avec citations [Source X] et markdown
        
    Returns:
        Texte HTML avec citations stylisées et markdown converti
    """
    import re
    import markdown
    
    # Convertit le markdown en HTML
    html_text = markdown.markdown(text, extensions=['extra', 'nl2br'])
    
    # Remplace [Source X] par un badge HTML stylisé
    pattern = r'\[Source\s+(\d+)\]'
    replacement = r'<span style="background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85em; margin: 0 2px; white-space: nowrap;">[Source \1]</span>'
    
    highlighted_text = re.sub(pattern, replacement, html_text)
    return highlighted_text


def init_rag_chain(num_sources: int = 5):
    """Initialise la chaîne RAG et la met en cache"""
    if not os.path.exists("./chroma_db"):
        st.error("❌ Base de données vectorielle non trouvée. Veuillez d'abord exécuter `python indexer.py`")
        st.stop()
    
    try:
        rag = RAGChain(num_sources=num_sources)
        return rag
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
        st.info("💡 Assurez-vous qu'Ollama est installé et en cours d'exécution avec le modèle gemma3:4b")
        st.stop()


def main():
    """Fonction principale de l'application"""
    
    # En-tête
    st.markdown('<div class="main-header">🛡️ Assistant de Conformité RGPD + IA ACT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Posez vos questions sur la conformité juridique - 100% local et confidentiel</div>', unsafe_allow_html=True)
    
    # Barre latérale
    with st.sidebar:
        st.title("ℹ️ À propos")
        st.markdown("""
        ### 🎯 Fonctionnalités
        - ✅ Traitement 100% local
        - ✅ Confidentialité absolue
        - ✅ Réponses sourcées
        - ✅ RGPD + IA Act
        - ✅ Interface moderne
        
        ### 🔧 Technologies
        - **LLM**: Ollama (gemma3:4b)
        - **RAG**: LangChain 0.3.x
        - **Vector DB**: ChromaDB 0.5.x
        - **Embeddings**: Sentence Transformers 3.x
        
        ### 📚 Base de connaissances
        - Règlement RGPD
        - Règlement IA Act européen

        ### 🧙‍♂️ Concepteur & Développeur
        - Anthony GRAINDORGE
        """)
        
        st.divider()
        
        st.markdown("### ⚙️ Paramètres")
        num_sources = st.slider(
            "Nombre de sources à afficher",
            min_value=10,
            max_value=50,
            value=25,
            help="Nombre d'extraits de documents à afficher comme sources"
        )
        
        show_metadata = st.checkbox(
            "Afficher les métadonnées",
            value=True,
            help="Afficher les informations détaillées sur les sources"
        )
    
    # Initialisation de la session
    if "rag_chain" not in st.session_state or "num_sources" not in st.session_state or st.session_state.num_sources != num_sources:
        with st.spinner("🔄 Initialisation du système... Cela peut prendre quelques secondes."):
            try:
                st.session_state.rag_chain = init_rag_chain(num_sources)
                st.session_state.num_sources = num_sources
                st.success("✅ Système prêt !")
            except Exception as e:
                st.error(f"❌ Erreur d'initialisation: {str(e)}")
                st.stop()
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Indicateur de statut
    if st.session_state.rag_chain:
        st.sidebar.success("🟢 Système opérationnel")
    else:
        st.sidebar.error("🔴 Système non disponible")
    
    # Zone de saisie de la question
    st.markdown("### 💬 Posez votre question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Question",
            placeholder="Ex: Quels sont les droits des personnes concernées selon le RGPD ?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)
    
    # Questions suggérées
    st.markdown("#### 💡 Questions suggérées")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Droits des personnes"):
            question = "Quels sont les droits des personnes concernées selon le RGPD ?"
            ask_button = True
    
    with col2:
        if st.button("🔒 DPO - Rôle"):
            question = "Quel est le rôle et les responsabilités du DPO ?"
            ask_button = True
    
    with col3:
        if st.button("🤖 IA Act - Risques"):
            question = "Quelles sont les catégories de risques selon l'IA Act ?"
            ask_button = True
    
    # Traitement de la question
    if ask_button and question:
        # Container pour le loader
        loader_container = st.empty()
        
        try:
            # Phase 1: Recherche
            loader_container.markdown(show_loader("🔍 Recherche dans la base de connaissances..."), unsafe_allow_html=True)
            
            # Phase 2: Génération avec streaming
            loader_container.markdown(show_loader("🤖 Génération de la réponse..."), unsafe_allow_html=True)
            
            # Container pour la réponse en streaming
            answer_container = st.empty()
            
            # Utilise la méthode de streaming
            result = st.session_state.rag_chain.query_streaming(question, answer_container)
            
            # Nettoyage du loader
            loader_container.empty()
            
            # Ajoute à l'historique
            st.session_state.history.insert(0, result)
            
            # Analyse des citations dans la réponse
            citation_info = st.session_state.rag_chain.extract_citations(result["answer"])
            
            # Met à jour l'affichage de la réponse avec citations stylisées
            st.markdown("### 📝 Réponse")
            
            # Badge avec nombre de citations
            if citation_info["cited_sources"]:
                st.caption(f"🔗 {citation_info['total_citations']} citation(s) vers {len(citation_info['cited_sources'])} source(s)")
            
            # Affiche la réponse finale avec citations stylisées
            highlighted_answer = highlight_citations(result["answer"])
            answer_container.markdown(f'<div class="answer-box">{highlighted_answer}</div>', unsafe_allow_html=True)
            
            # Affichage des sources
            st.markdown("### 📚 Sources utilisées")
            
            # Info avec détails sur les sources citées vs non citées
            cited_count = len(citation_info["cited_sources"])
            total_count = len(result['source_documents'])
            if cited_count < total_count:
                st.info(f"📊 {cited_count}/{total_count} sources citées dans la réponse • {total_count} extraits pertinents trouvés")
            else:
                st.info(f"📊 Toutes les {total_count} sources ont été citées dans la réponse")
            
            sources = st.session_state.rag_chain.format_sources(result["source_documents"])
            
            for i, source in enumerate(sources[:num_sources]):
                # Vérifie si cette source a été citée dans la réponse
                source_num = source['index']
                is_cited = source_num in citation_info["cited_sources"]
                
                # Titre avec indicateur de citation
                if is_cited:
                    title = f"✅ Source {source_num} (Citée) - {source['chunk_title']}"
                    expanded = True  # Expand cited sources by default
                else:
                    title = f"📄 Source {source_num} (Non citée) - {source['chunk_title']}"
                    expanded = False
                
                with st.expander(title, expanded=expanded):
                    # Informations de contexte avec qualité
                    context_cols = st.columns([3, 1])
                    with context_cols[0]:
                        if source.get('context_info'):
                            st.info(f"ℹ️ {source['context_info']}")
                    
                    with context_cols[1]:
                        # Indicateur de qualité du chunk
                        chunk_quality = source.get('chunk_quality', 1.0)
                        is_complete = source.get('is_complete', True)
                        
                        if isinstance(chunk_quality, (int, float)):
                            quality_pct = int(chunk_quality * 100)
                            if is_complete:
                                st.success(f"✅ Qualité: {quality_pct}%")
                            else:
                                st.warning(f"⚠️ Qualité: {quality_pct}%")
                    
                    # Contenu principal
                    st.markdown("**Extrait:**")
                    # Affiche dans un format plus lisible
                    st.text_area(
                        "Contenu",
                        source['content'],
                        height=200,
                        label_visibility="collapsed"
                    )
                    
                    # Métadonnées enrichies
                    if show_metadata:
                        st.markdown("**Métadonnées détaillées:**")
                        
                        metadata_dict = {
                            "Fichier": source['source_file'],
                            "Page": source['page'],
                            "Type de contenu": source.get('content_type', 'paragraph'),
                            "Mots": source.get('word_count', 0),
                            "Longueur": len(source['content'])
                        }
                        
                        # Ajoute les informations structurelles si disponibles
                        if source.get('article_number'):
                            metadata_dict["Article"] = source['article_number']
                        if source.get('chapter_title'):
                            metadata_dict["Chapitre"] = source['chapter_title']
                        if source.get('section_title'):
                            metadata_dict["Section"] = source['section_title']
                        if source.get('key_terms'):
                            key_terms = source['key_terms']
                            if isinstance(key_terms, str):
                                metadata_dict["Termes clés"] = key_terms
                            else:
                                metadata_dict["Termes clés"] = ", ".join(key_terms)
                        
                        st.json(metadata_dict)
            
        except Exception as e:
            loader_container.empty()
            st.error(f"❌ Erreur lors du traitement: {str(e)}")
            st.info("💡 Vérifiez qu'Ollama est bien lancé avec: `ollama run gemma3:4b`")
    
    # Historique des questions
    if st.session_state.history:
        st.divider()
        st.markdown("### 📜 Historique des questions")
        
        for i, item in enumerate(st.session_state.history[:5]):  # Affiche les 5 dernières
            with st.expander(f"Question {i+1}: {item['question'][:60]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Réponse:** {item['answer'][:300]}...")
                st.caption(f"📚 {len(item['source_documents'])} sources utilisées")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🔐 Toutes les données restent sur votre machine - Aucune connexion Internet requise pour le traitement
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

