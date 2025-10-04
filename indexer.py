"""
Script d'indexation des documents RGPD et IA ACT
Charge les PDFs, découpe en chunks, vectorise et stocke dans ChromaDB
"""
import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO

# Disable ChromaDB telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress telemetry and torch warnings
warnings.filterwarnings("ignore", message="Failed to send telemetry event")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.classes.*")

from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import BaseDocumentTransformer, Document
import chromadb
import re


class LegalTextCleanerTransformer(BaseDocumentTransformer):
    """
    LangChain Document Transformer pour nettoyer les textes juridiques français
    Utilise les capacités natives de LangChain pour transformer les documents
    """
    
    def transform_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Transforme une liste de documents en nettoyant leur contenu
        
        Args:
            documents: Liste de documents LangChain à transformer
            
        Returns:
            Liste de documents nettoyés
        """
        return [self._clean_document(doc) for doc in documents]
    
    async def atransform_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """Version async de transform_documents"""
        return self.transform_documents(documents, **kwargs)
    
    def _clean_document(self, doc: Document) -> Document:
        """
        Nettoie un document individuel en préservant sa structure juridique
        
        Args:
            doc: Document à nettoyer
            
        Returns:
            Document nettoyé avec métadonnées préservées
        """
        content = doc.page_content
        
        # 1. Corrige les coupures de mots en fin de ligne (trait d'union)
        content = re.sub(r'-\s*\n\s*', '', content)
        
        # 2. Normalise les espaces multiples (mais préserve les retours à la ligne)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # 3. Nettoie les retours à la ligne excessifs (max 2 consécutifs)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 4. Répare les énumérations françaises cassées
        # Ex: "a ) texte" devient "a) texte"
        content = re.sub(r'([a-z0-9]+)\s+\)', r'\1)', content)
        
        # 5. Ajoute un saut de ligne avant les items d'énumération s'il manque
        content = re.sub(r'([;.])([a-z]\))', r'\1\n\2', content)
        
        # 6. Ponctuation française: espace insécable avant : ; ! ? «  et après »
        # Nettoie d'abord les espaces multiples
        content = re.sub(r'\s{2,}([;:!?])', r' \1', content)
        # Assure un espace avant
        content = re.sub(r'([^\s\n])([;:!?])', r'\1 \2', content)
        # Nettoie les espaces avant les ponctuations simples
        content = re.sub(r'\s+([.,])', r'\1', content)
        # Assure un espace après toute ponctuation
        content = re.sub(r'([.,;:!?])([A-ZÀ-Úa-zà-ú0-9])', r'\1 \2', content)
        
        # 7. Guillemets français
        content = re.sub(r'«\s*', '« ', content)
        content = re.sub(r'\s*»', ' »', content)
        
        # 8. Nettoie les espaces en début/fin de lignes
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(line for line in lines if line)
        
        # 9. Nettoie le début et la fin du document
        content = content.strip()
        
        # 10. Détecte et marque les fragments incomplets
        # Si commence au milieu d'une phrase
        if content and not self._is_complete_start(content):
            content = "[...] " + content
        
        # Si termine au milieu d'une phrase
        if content and not self._is_complete_end(content):
            content = content + " [...]"
        
        # Retourne un nouveau document avec le contenu nettoyé
        return Document(page_content=content, metadata=doc.metadata.copy())
    
    def _is_complete_start(self, content: str) -> bool:
        """Vérifie si le contenu commence de manière complète"""
        patterns = [
            r'^Article\s+\d+',
            r'^Article\s+premier',
            r'^CHAPITRE\s+[IVX]+',
            r'^SECTION\s+\d+',
            r'^TITRE\s+[IVX]+',
            r'^[IVX]+\.\s+',
            r'^\d+\.\s+',
            r'^[a-z]\)\s+',
            r'^\([a-z]\)\s+',
            r'^\[[A-Z]',
            r'^«\s',
            r'^[A-ZÀÉÈÊËÏÎÔÙÛÇ]',  # Commence par une majuscule
        ]
        return any(re.match(pattern, content) for pattern in patterns)
    
    def _is_complete_end(self, content: str) -> bool:
        """Vérifie si le contenu termine de manière complète"""
        patterns = [
            r'[.;!?»]$',  # Ponctuation de fin
            r':\s*$',      # Deux-points (début liste suivante)
        ]
        return any(re.search(pattern, content) for pattern in patterns)


class DocumentIndexer:
    """Classe pour indexer les documents dans ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialise l'indexeur
        
        Args:
            persist_directory: Répertoire de stockage de la base vectorielle
        """
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def load_pdf(self, pdf_path: str) -> List:
        """
        Charge un document PDF
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Liste de documents chargés
        """
        print(f"📄 Chargement du fichier : {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ {len(documents)} pages chargées")
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 1500, chunk_overlap: int = 400) -> List:
        """
        Découpe les documents en chunks avec une stratégie améliorée
        Utilise un chunking plus intelligent pour préserver le contexte complet
        
        Args:
            documents: Liste de documents à découper
            chunk_size: Taille des chunks (augmentée pour plus de contexte)
            chunk_overlap: Chevauchement entre chunks (augmenté pour meilleure continuité)
            
        Returns:
            Liste de chunks avec métadonnées enrichies
        """
        print(f"✂️  Découpage intelligent en chunks (taille={chunk_size}, overlap={chunk_overlap})")
        
        # Séparateurs optimisés pour les documents juridiques avec support énumérations
        # Ordre d'importance: préserver les structures complètes et listes
        separators = [
            "\n\n\n\n",  # Séparations majeures (parties, titres)
            "\n\n\n",    # Chapitres et sections
            "\n\n",      # Paragraphes complets
            # Pattern pour détecter le début d'une nouvelle énumération majeure
            r"\n(?=[a-z]\))",  # Nouvelle ligne avant a) b) c) etc.
            r"\n(?=\d+\))",    # Nouvelle ligne avant 1) 2) 3) etc.
            r"\n(?=[ivxlcdm]+\))",  # Nouvelle ligne avant i) ii) iii) etc.
            "\n",        # Articles et alinéas
            ". ",        # Fin de phrase (préféré à .\n pour éviter les coupures)
            " ; ",       # Point-virgule avec espaces (clauses, fin d'énumération)
            "; ",        # Point-virgule simple
            " : ",       # Deux-points avec espaces (début énumérations)
            ", ",        # Virgule (dernier recours pour phrases longues)
            " ",         # Espace (vraiment dernier recours)
            ""           # Caractère par caractère (à éviter)
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            keep_separator=True,  # Garde les séparateurs pour préserver la structure
            is_separator_regex=True  # Active le mode regex pour les patterns avancés
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # 🎯 Utilise le LangChain Document Transformer pour nettoyer (méthode native LangChain!)
        print("🧹 Application du LegalTextCleanerTransformer...")
        text_cleaner = LegalTextCleanerTransformer()
        cleaned_chunks = text_cleaner.transform_documents(chunks)
        
        # Post-traitement: enrichissement des métadonnées
        processed_chunks = []
        for i, chunk in enumerate(cleaned_chunks):
            # Enrichit les métadonnées (structure, articles, etc.)
            enriched_chunk = self._enrich_chunk_metadata(chunk, i)
            processed_chunks.append(enriched_chunk)
        
        print(f"✅ {len(processed_chunks)} chunks créés avec métadonnées enrichies")
        return processed_chunks
    
    def _clean_and_optimize_chunk(self, chunk):
        """
        Nettoie et optimise un chunk pour améliorer sa qualité
        
        Args:
            chunk: Chunk à nettoyer
            
        Returns:
            Chunk nettoyé
        """
        import re
        from langchain_core.documents import Document
        
        content = chunk.page_content
        
        # 1. Supprime les espaces multiples en préservant les retours à la ligne
        content = re.sub(r' +', ' ', content)
        
        # 2. Normalise les retours à la ligne (max 2 consécutifs)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 3. Supprime les espaces en début/fin de lignes
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)
        
        # 4. Supprime les tirets de coupure de mots en fin de ligne
        content = re.sub(r'-\n', '', content)
        
        # 5. Gère la ponctuation selon les règles françaises
        # En français: espace avant : ; ! ? et « », espace après tous
        
        # Nettoie les espaces multiples avant ponctuation simple (. ,)
        content = re.sub(r'\s+([.,])', r'\1', content)
        
        # Préserve l'espace avant les ponctuations doubles françaises (: ; ! ?)
        # mais nettoie les espaces multiples
        content = re.sub(r'\s{2,}([;:!?])', r' \1', content)
        
        # Ajoute un espace avant si manquant pour : ; ! ?
        content = re.sub(r'([^\s])([;:!?])', r'\1 \2', content)
        
        # 6. Ajoute des espaces après la ponctuation si manquants
        content = re.sub(r'([.,;:!?])([A-ZÀ-Úa-zà-ú])', r'\1 \2', content)
        
        # 7. Nettoie les espaces en début et fin
        content = content.strip()
        
        # 8. Si le chunk commence par un fragment, ajoute un indicateur
        if content and not self._starts_with_complete_sentence(content):
            content = "[...] " + content
        
        # 9. Si le chunk finit par un fragment, ajoute un indicateur
        if content and not self._ends_with_complete_sentence(content):
            content = content + " [...]"
        
        return Document(page_content=content, metadata=chunk.metadata.copy())
    
    def _starts_with_complete_sentence(self, content: str) -> bool:
        """
        Vérifie si le contenu commence par une phrase complète (français)
        
        Args:
            content: Contenu à vérifier
            
        Returns:
            True si le contenu commence de manière complète
        """
        import re
        
        # Le chunk commence bien s'il commence par:
        # - Un titre/numéro (Article, Chapitre, Section, etc.)
        # - Une majuscule après un numéro d'énumération
        # - Un début de paragraphe standard français
        # - Un guillemet ouvrant français
        
        patterns = [
            r'^Article\s+\d+',                    # Article 5
            r'^Article\s+premier',                # Article premier
            r'^CHAPITRE\s+[IVX]+',               # CHAPITRE III
            r'^SECTION\s+\d+',                    # SECTION 1
            r'^TITRE\s+[IVX]+',                  # TITRE II
            r'^[IVX]+\.\s+',                     # III. 
            r'^\d+\.\s+',                         # 1. 2. 3.
            r'^[a-z]\)\s+',                       # a) b) c)
            r'^\([a-z]\)\s+',                     # (a) (b) (c)
            r'^[«"]',                             # Guillemets français
            r'^[A-ZÀÂÆÇÉÈÊËÏÎÔŒÙÛÜ]',           # Majuscule (français)
            r'^Le\s+', r'^La\s+', r'^Les\s+',    # Articles définis
            r'^Un\s+', r'^Une\s+', r'^Des\s+',   # Articles indéfinis
            r'^Ce\s+', r'^Cette\s+', r'^Ces\s+', # Démonstratifs
            r'^Il\s+', r'^Elle\s+', r'^Ils\s+',  # Pronoms sujets
            r'^Dans\s+', r'^Pour\s+', r'^Par\s+', # Prépositions courantes
            r'^Aux\s+', r'^Au\s+',               # Contractions
        ]
        
        content_trimmed = content.strip()
        
        for pattern in patterns:
            if re.match(pattern, content_trimmed):
                return True
        
        return False
    
    def _ends_with_complete_sentence(self, content: str) -> bool:
        """
        Vérifie si le contenu finit par une phrase complète (règles françaises)
        
        Args:
            content: Contenu à vérifier
            
        Returns:
            True si le contenu finit de manière complète
        """
        import re
        
        # Le chunk finit bien s'il finit par:
        # - Un point, point-virgule, deux-points, point d'exclamation ou d'interrogation
        # - Un retour à la ligne après ponctuation
        # - Une fin d'énumération avec )
        
        content_stripped = content.strip()
        
        # Vérifie si finit par une ponctuation de fin (française)
        if re.search(r'[.;:!?]$', content_stripped):
            return True
        
        # Vérifie si finit par une fin d'énumération
        if re.search(r'[);]$', content_stripped):
            return True
        
        # Vérifie si finit par un retour à la ligne (nouveau paragraphe)
        if content.endswith('\n\n') or content.endswith('\n'):
            return True
        
        # Vérifie si finit par un guillemet fermant (français)
        if re.search(r'[»"]$', content_stripped):
            return True
        
        return False
    
    def _enrich_chunk_metadata(self, chunk, chunk_index: int):
        """
        Enrichit les métadonnées d'un chunk avec des informations structurelles
        
        Args:
            chunk: Chunk à enrichir
            chunk_index: Index du chunk
            
        Returns:
            Chunk avec métadonnées enrichies
        """
        content = chunk.page_content
        metadata = chunk.metadata.copy()
        
        # Extrait les informations structurelles du contenu
        structure_info = self._extract_structure_info(content)
        
        # Calcule la qualité du chunk
        chunk_quality = self._calculate_chunk_quality(content)
        
        # Ajoute les métadonnées enrichies (filtre les valeurs None)
        enriched_metadata = {
            "chunk_index": chunk_index,
            "chunk_length": len(content),
            "word_count": len(content.split()),
            "has_article": structure_info["has_article"],
            "has_chapter": structure_info["has_chapter"],
            "has_section": structure_info["has_section"],
            "content_type": structure_info["content_type"],
            "key_terms": structure_info["key_terms"],
            "chunk_quality": chunk_quality,  # Score de qualité du chunk
            "is_complete": chunk_quality >= 0.7  # Indique si le chunk semble complet
        }
        
        # Ajoute les valeurs non-None seulement
        if structure_info["article_number"] is not None:
            enriched_metadata["article_number"] = structure_info["article_number"]
        if structure_info["chapter_title"] is not None:
            enriched_metadata["chapter_title"] = structure_info["chapter_title"]
        if structure_info["section_title"] is not None:
            enriched_metadata["section_title"] = structure_info["section_title"]
        
        metadata.update(enriched_metadata)
        
        # Filtre les métadonnées pour ChromaDB (supprime les valeurs None et les listes vides)
        filtered_metadata = self._filter_metadata_for_chromadb(metadata)
        
        # Crée un nouveau chunk avec les métadonnées enrichies
        from langchain_core.documents import Document
        return Document(page_content=content, metadata=filtered_metadata)
    
    def _calculate_chunk_quality(self, content: str) -> float:
        """
        Calcule un score de qualité pour le chunk (0.0 à 1.0)
        
        Args:
            content: Contenu du chunk
            
        Returns:
            Score de qualité (0.0 = mauvais, 1.0 = excellent)
        """
        score = 1.0
        
        # Pénalité si le chunk commence par un fragment
        if content.startswith("[...] "):
            score -= 0.15
        
        # Pénalité si le chunk finit par un fragment
        if content.endswith(" [...]"):
            score -= 0.15
        
        # Bonus si le chunk contient une structure claire (Article, Chapitre, etc.)
        import re
        if re.search(r'(Article|CHAPITRE|SECTION)\s+\d+', content):
            score += 0.2
        
        # Pénalité si le chunk est très court (probablement incomplet)
        if len(content) < 300:
            score -= 0.2
        
        # Bonus si le chunk a une bonne taille (pas trop court, pas trop long)
        if 800 <= len(content) <= 1500:
            score += 0.1
        
        # S'assure que le score reste dans [0, 1]
        return max(0.0, min(1.0, score))
    
    def _filter_metadata_for_chromadb(self, metadata: dict) -> dict:
        """
        Filtre les métadonnées pour être compatibles avec ChromaDB
        
        Args:
            metadata: Métadonnées à filtrer
            
        Returns:
            Métadonnées filtrées
        """
        filtered = {}
        for key, value in metadata.items():
            # ChromaDB accepte: str, int, float, bool
            if value is None:
                continue  # Ignore les valeurs None
            elif isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, list):
                # Convertit les listes en chaînes séparées par des virgules
                if value:  # Ignore les listes vides
                    filtered[key] = ", ".join(str(item) for item in value)
            else:
                # Convertit tout le reste en chaîne
                filtered[key] = str(value)
        
        return filtered
    
    def _extract_structure_info(self, content: str) -> dict:
        """
        Extrait les informations structurelles du contenu
        
        Args:
            content: Contenu du chunk
            
        Returns:
            Dictionnaire avec les informations structurelles
        """
        import re
        
        structure_info = {
            "has_article": False,
            "article_number": None,
            "has_chapter": False,
            "chapter_title": None,
            "has_section": False,
            "section_title": None,
            "content_type": "paragraph",
            "key_terms": []
        }
        
        # Recherche des articles
        article_match = re.search(r'Article\s+(\d+)', content, re.IGNORECASE)
        if article_match:
            structure_info["has_article"] = True
            structure_info["article_number"] = article_match.group(1)
            structure_info["content_type"] = "article"
        
        # Recherche des chapitres
        chapter_match = re.search(r'CHAPITRE\s+([IVX]+)', content, re.IGNORECASE)
        if chapter_match:
            structure_info["has_chapter"] = True
            structure_info["chapter_title"] = chapter_match.group(1)
            structure_info["content_type"] = "chapter"
        
        # Recherche des sections
        section_match = re.search(r'SECTION\s+(\d+)', content, re.IGNORECASE)
        if section_match:
            structure_info["has_section"] = True
            structure_info["section_title"] = section_match.group(1)
            structure_info["content_type"] = "section"
        
        # Extrait les termes clés juridiques (en français)
        legal_terms = [
            "RGPD", "IA Act", "données personnelles", "consentement", "DPO",
            "responsable du traitement", "responsable de traitement", "sous-traitant",
            "droits des personnes", "personne concernée", "destinataire",
            "système d'IA", "système d'intelligence artificielle", "risque élevé",
            "haut risque", "conformité", "sanctions", "transparence",
            "privacy by design", "accountability", "protection des données",
            "traitement de données", "finalité", "licéité", "minimisation",
            "exactitude", "limitation de conservation", "intégrité",
            "confidentialité", "évaluation d'impact", "violation de données",
            "autorité de contrôle", "délégué à la protection", "portabilité",
            "droit d'accès", "droit de rectification", "droit à l'effacement",
            "sécurité", "mesures techniques", "mesures organisationnelles"
        ]
        
        content_lower = content.lower()
        found_terms = []
        for term in legal_terms:
            if term.lower() in content_lower:
                found_terms.append(term)
                if len(found_terms) >= 5:  # Limite à 5 termes
                    break
        
        structure_info["key_terms"] = found_terms
        
        return structure_info
    
    def create_vectorstore(self, chunks: List) -> Chroma:
        """
        Crée et persiste la base vectorielle
        
        Args:
            chunks: Liste de chunks à vectoriser
            
        Returns:
            Instance de la base vectorielle Chroma
        """
        print(f"🔢 Création de la base vectorielle dans {self.persist_directory}")
        
        # Supprime la base existante si elle existe
        if os.path.exists(self.persist_directory):
            print("⚠️  Base existante détectée, suppression...")
            import shutil
            shutil.rmtree(self.persist_directory)
        
        # Crée la nouvelle base (suppress telemetry errors)
        with redirect_stderr(StringIO()):
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        print(f"✅ Base vectorielle créée avec succès ({len(chunks)} embeddings)")
        return vectorstore
    
    def index_directory(self, pdf_directory: str = "./knowledge_base"):
        """
        Indexe tous les PDFs d'un répertoire
        
        Args:
            pdf_directory: Répertoire contenant les PDFs
        """
        print(f"\n{'='*60}")
        print(f"🚀 Démarrage de l'indexation")
        print(f"{'='*60}\n")
        
        # Liste tous les PDFs
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ Aucun fichier PDF trouvé dans {pdf_directory}")
            return
        
        print(f"📚 {len(pdf_files)} fichiers PDF détectés : {', '.join(pdf_files)}\n")
        
        # Charge tous les documents
        all_documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            documents = self.load_pdf(pdf_path)
            
            # Ajoute metadata pour identifier la source
            for doc in documents:
                doc.metadata['source_file'] = pdf_file
            
            all_documents.extend(documents)
        
        print(f"\n📊 Total : {len(all_documents)} pages chargées\n")
        
        # Découpe en chunks
        chunks = self.split_documents(all_documents)
        
        # Crée la base vectorielle
        vectorstore = self.create_vectorstore(chunks)
        
        print(f"\n{'='*60}")
        print(f"✅ Indexation terminée avec succès !")
        print(f"{'='*60}\n")
        
        return vectorstore


def main():
    """Point d'entrée principal"""
    indexer = DocumentIndexer()
    indexer.index_directory("./knowledge_base")


if __name__ == "__main__":
    main()

