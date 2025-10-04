"""
Script de test pour vérifier que le système RAG fonctionne correctement
"""
import os
import sys
from rag_chain import RAGChain


def test_initialization():
    """Test 1: Vérifier que la chaîne RAG s'initialise correctement"""
    print("\n" + "="*80)
    print("TEST 1: Initialisation de la chaîne RAG")
    print("="*80)
    
    try:
        if not os.path.exists("./chroma_db"):
            print("❌ ÉCHEC: Base vectorielle non trouvée")
            print("💡 Exécutez d'abord: python indexer.py")
            return False
        
        rag = RAGChain()
        print("✅ SUCCÈS: Chaîne RAG initialisée")
        return rag
    except Exception as e:
        print(f"❌ ÉCHEC: {str(e)}")
        return False


def test_simple_query(rag):
    """Test 2: Vérifier qu'une requête simple fonctionne"""
    print("\n" + "="*80)
    print("TEST 2: Requête simple")
    print("="*80)
    
    try:
        question = "Qu'est-ce que le RGPD ?"
        print(f"❓ Question: {question}\n")
        
        result = rag.query(question)
        
        if result["answer"] and len(result["source_documents"]) > 0:
            print(f"✅ SUCCÈS: Réponse générée avec {len(result['source_documents'])} sources")
            print(f"\n📝 Réponse (extrait): {result['answer'][:200]}...")
            return True
        else:
            print("❌ ÉCHEC: Réponse vide ou pas de sources")
            return False
            
    except Exception as e:
        print(f"❌ ÉCHEC: {str(e)}")
        return False


def test_rgpd_query(rag):
    """Test 3: Tester une question spécifique au RGPD"""
    print("\n" + "="*80)
    print("TEST 3: Question RGPD spécifique")
    print("="*80)
    
    try:
        question = "Quels sont les droits des personnes concernées selon le RGPD ?"
        print(f"❓ Question: {question}\n")
        
        result = rag.query(question)
        
        # Vérifier que la réponse mentionne le RGPD
        if "rgpd" in result["answer"].lower() or "données" in result["answer"].lower():
            print(f"✅ SUCCÈS: Réponse pertinente avec {len(result['source_documents'])} sources")
            
            # Afficher les sources
            print("\n📚 Sources utilisées:")
            for i, doc in enumerate(result["source_documents"][:3], 1):
                source_file = doc.metadata.get("source_file", "Inconnu")
                page = doc.metadata.get("page", "N/A")
                print(f"  [{i}] {source_file} - Page {page}")
            
            return True
        else:
            print("⚠️ AVERTISSEMENT: Réponse générée mais pertinence incertaine")
            print(f"Réponse: {result['answer'][:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ ÉCHEC: {str(e)}")
        return False


def test_ia_act_query(rag):
    """Test 4: Tester une question spécifique à l'IA Act"""
    print("\n" + "="*80)
    print("TEST 4: Question IA Act spécifique")
    print("="*80)
    
    try:
        question = "Quelles sont les catégories de risques selon l'IA Act ?"
        print(f"❓ Question: {question}\n")
        
        result = rag.query(question)
        
        if len(result["source_documents"]) > 0:
            print(f"✅ SUCCÈS: Réponse générée avec {len(result['source_documents'])} sources")
            
            # Vérifier si on trouve au moins une source du fichier IA Act
            ia_act_sources = [doc for doc in result["source_documents"] 
                            if "IA_ACT" in doc.metadata.get("source_file", "").upper()]
            
            if ia_act_sources:
                print(f"✅ {len(ia_act_sources)} source(s) provenant de l'IA Act")
            else:
                print("⚠️ Aucune source explicite de l'IA Act détectée")
            
            return True
        else:
            print("❌ ÉCHEC: Pas de sources trouvées")
            return False
            
    except Exception as e:
        print(f"❌ ÉCHEC: {str(e)}")
        return False


def test_source_quality(rag):
    """Test 5: Vérifier la qualité des sources retournées"""
    print("\n" + "="*80)
    print("TEST 5: Qualité des sources")
    print("="*80)
    
    try:
        question = "Quel est le rôle du DPO ?"
        print(f"❓ Question: {question}\n")
        
        result = rag.query(question)
        sources = rag.format_sources(result["source_documents"])
        
        print(f"📊 Analyse de {len(sources)} sources:\n")
        
        for source in sources[:3]:
            print(f"[{source['index']}] {source['source_file']} - Page {source['page']}")
            content_length = len(source['content'])
            print(f"    Longueur: {content_length} caractères")
            
            if content_length < 100:
                print("    ⚠️ Source courte")
            elif content_length > 2000:
                print("    ⚠️ Source très longue")
            else:
                print("    ✅ Longueur appropriée")
            print()
        
        print("✅ SUCCÈS: Sources analysées")
        return True
        
    except Exception as e:
        print(f"❌ ÉCHEC: {str(e)}")
        return False


def main():
    """Fonction principale pour exécuter tous les tests"""
    print("\n" + "="*80)
    print("🧪 SUITE DE TESTS - Assistant RGPD + IA ACT")
    print("="*80)
    
    results = []
    
    # Test 1: Initialisation
    rag = test_initialization()
    results.append(("Initialisation", bool(rag)))
    
    if not rag:
        print("\n❌ Impossible de continuer sans initialisation réussie")
        return
    
    # Test 2: Requête simple
    results.append(("Requête simple", test_simple_query(rag)))
    
    # Test 3: Question RGPD
    results.append(("Question RGPD", test_rgpd_query(rag)))
    
    # Test 4: Question IA Act
    results.append(("Question IA Act", test_ia_act_query(rag)))
    
    # Test 5: Qualité des sources
    results.append(("Qualité sources", test_source_quality(rag)))
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*80)
    print(f"Résultat: {passed}/{total} tests réussis ({passed/total*100:.0f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés ! Le système est opérationnel.")
    elif passed >= total * 0.6:
        print("\n⚠️ La plupart des tests sont passés. Vérifiez les échecs.")
    else:
        print("\n❌ Plusieurs tests ont échoué. Vérifiez la configuration.")
    
    print("\n💡 Pour lancer l'application web: streamlit run app.py")


if __name__ == "__main__":
    main()

