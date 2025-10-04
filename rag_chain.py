"""
Chaîne RAG pour interroger les documents RGPD et IA ACT
Utilise ChromaDB pour la recherche et Ollama pour la génération
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

from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler pour le streaming des réponses"""
    
    def __init__(self, container):
        self.container = container
        self.current_text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Appelé à chaque nouveau token généré"""
        self.current_text += token
        # Met à jour l'affichage en temps réel
        self.container.markdown(self.current_text)


class RAGChain:
    """Classe pour gérer la chaîne RAG"""
    
    def __init__(self, persist_directory: str = "./chroma_db", model: str = "gemma3:4b", num_sources: int = 5):
        """
        Initialise la chaîne RAG
        
        Args:
            persist_directory: Répertoire de la base vectorielle
            model: Nom du modèle Ollama à utiliser
            num_sources: Nombre de sources à récupérer
        """
        self.persist_directory = persist_directory
        self.model_name = model
        self.num_sources = num_sources
        
        # Initialise les embeddings (mêmes que lors de l'indexation)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Charge la base vectorielle
        self.vectorstore = self._load_vectorstore()
        
        # Initialise le LLM local
        self.llm = self._init_llm()
        
        # Crée le retriever (pas de chaîne complète, on gère manuellement pour les citations)
        self.retriever = self._create_qa_chain()
    
    def _load_vectorstore(self) -> Chroma:
        """Charge la base vectorielle existante"""
        print(f"📦 Chargement de la base vectorielle depuis {self.persist_directory}")
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print(f"✅ Base vectorielle chargée")
        return vectorstore
    
    def _init_llm(self) -> OllamaLLM:
        """Initialise le modèle LLM local via Ollama"""
        print(f"🤖 Initialisation du modèle {self.model_name} via Ollama")
        llm = OllamaLLM(
            model=self.model_name,
            temperature=0.1,  # Température basse pour plus de précision
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        print(f"✅ Modèle initialisé")
        return llm
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Crée le template de prompt pour le LLM avec citations inline"""
        template = """Tu es un assistant expert en conformité juridique, spécialisé dans le RGPD (Règlement Général sur la Protection des Données) et l'IA Act (règlement européen sur l'intelligence artificielle).

RÈGLE ABSOLUE: Réponds PRÉCISÉMENT à la question posée. Ne change pas de sujet.

Utilise UNIQUEMENT les informations du contexte ci-dessous pour répondre à la question. Si la réponse n'est pas dans le contexte, dis clairement que tu ne peux pas répondre avec certitude.

Contexte (extraits de documents juridiques):
{context}

Question: {question}

RAPPEL: Ta réponse doit porter UNIQUEMENT sur: {question}

Instructions de réponse IMPORTANTES:
- **STRUCTURE COMME UN EXECUTIVE SUMMARY** : Utilise des titres, sous-titres et une mise en forme claire
- Réponds de manière naturelle et fluide, comme un expert juridique
- **CRITICAL: Cite les sources à la FIN de chaque paragraphe, pas dans le texte**
- **NE mentionne JAMAIS les sources dans le texte** (évite "Le texte [Source X] dit que...")
- Écris de manière directe et naturelle, puis ajoute [Source X] à la fin du paragraphe
- Tu peux combiner plusieurs sources dans un même paragraphe: [Source 1] [Source 3]

STRUCTURE OBLIGATOIRE:
- Commence par un résumé en 2-3 phrases
- Utilise des **titres** (## Titre) pour organiser les sections principales
- Utilise des **sous-titres** (### Sous-titre) pour les détails
- Fais des **listes à puces** (-) quand c'est approprié
- Termine par une **conclusion** ou **recommandations** si pertinent
- Garde les paragraphes courts (3-4 phrases max)
- Cite les articles, chapitres et sections pertinents quand c'est possible
- Distingue clairement entre le RGPD et l'IA Act si les deux sont mentionnés
- Reste factuel et évite les interprétations personnelles
- Si la réponse n'est pas dans le contexte, dis-le clairement

Format des citations:
- Place [Source X] à la fin de chaque paragraphe qui utilise cette source
- Pour plusieurs sources dans un paragraphe: [Source 1] [Source 2]
- IMPORTANT: Ne cite JAMAIS les sources au milieu d'une phrase, TOUJOURS à la fin du paragraphe

Exemples de MAUVAIS style (NE PAS FAIRE):
❌ "Le texte [Source 7] stipule que..."
❌ "Selon [Source 8], le DPO doit..."
❌ "De plus, [Source 9] décrit..."

Exemples de BON style (À SUIVRE):
✅ "Le responsable du traitement doit mettre en œuvre des mesures appropriées. [Source 7]"
✅ "Il doit superviser les activités de traitement et évaluer les risques. [Source 3]"
✅ "Les obligations incluent la supervision continue et l'évaluation. [Source 1] [Source 3]"

RÉPONDS UNIQUEMENT À LA QUESTION POSÉE en utilisant le contexte fourni. Ne dévie pas du sujet. 

Réponse:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_chain(self):
        """
        Crée une chaîne RAG personnalisée avec citations numérotées
        Note: Retourne juste le retriever, on gère manuellement la génération pour avoir plus de contrôle
        """
        print(f"⛓️  Création de la chaîne RAG avec support de citations")
        
        # On va gérer manuellement la chaîne pour avoir un contrôle total sur le formatage
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.num_sources}
        )
        
        print(f"✅ Retriever créé avec support de citations numérotées")
        return self.retriever
    
    def _format_docs_with_sources(self, docs: List[Document]) -> str:
        """
        Formate les documents avec des numéros de source pour les citations
        
        Args:
            docs: Liste de documents récupérés
            
        Returns:
            String formatée avec documents numérotés
        """
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_docs.append(f"[Source {i}]\n{doc.page_content}\n")
        return "\n".join(formatted_docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Pose une question à la chaîne RAG avec citations inline
        
        Args:
            question: La question à poser
            
        Returns:
            Dictionnaire contenant la réponse et les sources
        """
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}\n")
        
        # 1. Récupère les documents pertinents (API moderne LangChain)
        print(f"🔍 Recherche de documents pertinents...")
        source_documents = self.retriever.invoke(question)
        print(f"✅ {len(source_documents)} documents trouvés")
        
        # 2. Formate les documents avec numéros de source
        formatted_context = self._format_docs_with_sources(source_documents)
        
        # 3. Crée le prompt avec contexte numéroté
        prompt_template = self._create_prompt_template()
        prompt = prompt_template.format(context=formatted_context, question=question)
        
        # 4. Génère la réponse avec le LLM
        print(f"🤖 Génération de la réponse avec citations...")
        answer = self.llm.invoke(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_documents
        }
    
    def query_streaming(self, question: str, container) -> Dict[str, Any]:
        """
        Pose une question à la chaîne RAG avec streaming en temps réel
        
        Args:
            question: La question à poser
            container: Container Streamlit pour l'affichage en temps réel
            
        Returns:
            Dictionnaire contenant la réponse et les sources
        """
        print(f"\n{'='*60}")
        print(f"❓ Question (streaming): {question}")
        print(f"{'='*60}\n")
        
        # 1. Récupère les documents pertinents
        print(f"🔍 Recherche de documents pertinents...")
        source_documents = self.retriever.invoke(question)
        print(f"✅ {len(source_documents)} documents trouvés")
        
        # 2. Formate les documents avec numéros de source
        formatted_context = self._format_docs_with_sources(source_documents)
        
        # 3. Crée le prompt avec contexte numéroté
        prompt_template = self._create_prompt_template()
        prompt = prompt_template.format(context=formatted_context, question=question)
        
        # 4. Configure le LLM avec le callback de streaming
        streaming_callback = StreamingCallbackHandler(container)
        llm_with_streaming = OllamaLLM(
            model=self.model_name,
            temperature=0.1,
            callbacks=[streaming_callback]
        )
        
        # 5. Génère la réponse avec streaming
        print(f"🤖 Génération de la réponse avec streaming...")
        answer = llm_with_streaming.invoke(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_documents
        }
    
    def extract_citations(self, answer: str) -> Dict[str, Any]:
        """
        Extrait les citations [Source X] de la réponse et crée un mapping
        
        Args:
            answer: Réponse générée par le LLM
            
        Returns:
            Dict avec la réponse et les sources citées
        """
        import re
        
        # Trouve toutes les citations [Source X] dans la réponse
        citation_pattern = r'\[Source\s+(\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        # Convertit en set pour avoir les sources uniques, puis en liste triée
        cited_sources = sorted(set(int(c) for c in citations))
        
        return {
            "answer": answer,
            "cited_sources": cited_sources,
            "total_citations": len(citations)
        }
    
    def format_sources(self, source_documents: List) -> List[Dict[str, str]]:
        """
        Formate les documents sources pour l'affichage avec métadonnées enrichies
        
        Args:
            source_documents: Liste de documents sources
            
        Returns:
            Liste de dictionnaires formatés avec informations structurelles
        """
        sources = []
        for i, doc in enumerate(source_documents):
            metadata = doc.metadata
            
            # Construit le titre du chunk basé sur la structure
            chunk_title = self._build_chunk_title(metadata)
            
            # Nettoie le contenu pour un meilleur affichage
            cleaned_content = self._clean_content(doc.page_content)
            
            # Extrait les informations de contexte
            context_info = self._extract_context_info(metadata)
            
            sources.append({
                "index": i + 1,
                "content": cleaned_content,
                "source_file": metadata.get("source_file", "Inconnu"),
                "page": metadata.get("page", "N/A"),
                "chunk_title": chunk_title,
                "content_type": metadata.get("content_type", "paragraph"),
                "article_number": metadata.get("article_number"),
                "chapter_title": metadata.get("chapter_title"),
                "section_title": metadata.get("section_title"),
                "key_terms": metadata.get("key_terms", []),
                "word_count": metadata.get("word_count", 0),
                "context_info": context_info,
                "chunk_quality": metadata.get("chunk_quality", 1.0),
                "is_complete": metadata.get("is_complete", True)
            })
        return sources
    
    def _build_chunk_title(self, metadata: dict) -> str:
        """
        Construit un titre descriptif pour le chunk
        
        Args:
            metadata: Métadonnées du chunk
            
        Returns:
            Titre formaté
        """
        title_parts = []
        
        # Ajoute le fichier source
        source_file = metadata.get("source_file", "Document")
        if source_file.endswith('.pdf'):
            source_file = source_file[:-4]  # Enlève l'extension
        
        # Ajoute les informations structurelles
        if metadata.get("has_chapter") and metadata.get("chapter_title"):
            title_parts.append(f"Chapitre {metadata['chapter_title']}")
        
        if metadata.get("has_section") and metadata.get("section_title"):
            title_parts.append(f"Section {metadata['section_title']}")
        
        if metadata.get("has_article") and metadata.get("article_number"):
            title_parts.append(f"Article {metadata['article_number']}")
        
        # Construit le titre final
        if title_parts:
            structure = " - ".join(title_parts)
            return f"{source_file} - {structure}"
        else:
            return f"{source_file} - Page {metadata.get('page', 'N/A')}"
    
    def _clean_content(self, content: str) -> str:
        """
        Nettoie le contenu pour un meilleur affichage (règles françaises)
        
        Args:
            content: Contenu brut
            
        Returns:
            Contenu nettoyé avec ponctuation française correcte
        """
        import re
        
        # 1. Préserve les retours à la ligne simples, normalise les multiples
        # Ne pas tout mettre sur une ligne pour préserver la structure
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 2. Supprime les espaces multiples sur une même ligne
        content = re.sub(r'[ \t]+', ' ', content)
        
        # 3. Gère la ponctuation française
        # Nettoie les espaces avant . et ,
        content = re.sub(r'\s+([.,])', r'\1', content)
        
        # Préserve un seul espace avant : ; ! ?
        content = re.sub(r'\s*([;:!?])', r' \1', content)
        
        # Assure un espace après toute ponctuation
        content = re.sub(r'([.,;:!?])([A-ZÀ-Úa-zà-ú0-9])', r'\1 \2', content)
        
        # 4. Gère les guillemets français
        content = re.sub(r'«\s*', '« ', content)
        content = re.sub(r'\s*»', ' »', content)
        
        # 5. Nettoie les espaces en début/fin de lignes
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(line for line in lines if line)  # Enlève les lignes vides
        
        # 6. Nettoie les espaces en début et fin globaux
        content = content.strip()
        
        return content
    
    def _extract_context_info(self, metadata: dict) -> str:
        """
        Extrait les informations de contexte pertinentes
        
        Args:
            metadata: Métadonnées du chunk
            
        Returns:
            Informations de contexte formatées
        """
        context_parts = []
        
        # Type de contenu
        content_type = metadata.get("content_type", "paragraph")
        if content_type == "article":
            context_parts.append("📋 Article réglementaire")
        elif content_type == "chapter":
            context_parts.append("📖 Chapitre")
        elif content_type == "section":
            context_parts.append("📑 Section")
        else:
            context_parts.append("📄 Paragraphe")
        
        # Termes clés (peuvent être une chaîne ou une liste)
        key_terms = metadata.get("key_terms", [])
        if key_terms:
            if isinstance(key_terms, str):
                # Si c'est une chaîne, prend les 3 premiers termes
                terms_list = [term.strip() for term in key_terms.split(",")]
                terms_str = ", ".join(terms_list[:3])
            else:
                # Si c'est une liste
                terms_str = ", ".join(key_terms[:3])
            context_parts.append(f"🔑 {terms_str}")
        
        return " • ".join(context_parts)


def main():
    """Fonction de test"""
    rag = RAGChain()
    
    # Question de test
    test_question = "Qu'est-ce que le RGPD et quels sont ses objectifs principaux ?"
    result = rag.query(test_question)
    
    print(f"\n{'='*60}")
    print(f"📝 Réponse:")
    print(f"{'='*60}")
    print(result["answer"])
    
    print(f"\n{'='*60}")
    print(f"📚 Sources utilisées:")
    print(f"{'='*60}")
    sources = rag.format_sources(result["source_documents"])
    for source in sources:
        print(f"\n[{source['index']}] {source['source_file']} - Page {source['page']}")
        print(f"Extrait: {source['content'][:200]}...")


if __name__ == "__main__":
    main()

