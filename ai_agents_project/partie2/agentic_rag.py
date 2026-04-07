"""
============================================================
Partie 2 : Chatbot basé sur un RAG Agentique
============================================================
Objectif : Construire un chatbot RAG où l'agent décide
           quand utiliser le retriever comme outil.

Architecture :
  User → Agent LLM → [si besoin] Retriever Tool → Vector DB
                   → Réponse finale
"""

import os
import logging
from typing import Annotated, Sequence, Literal
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools import tool

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("PARTIE 2 - CHATBOT RAG AGENTIQUE (LangChain)")
print("=" * 60)

# ── Configuration ─────────────────────────────────────────────────────────────
RESOURCES_PATH = "./resources/"
CHROMA_PERSIST_DIR = "./chroma_db_partie2"
COLLECTION_NAME = "documents_rag"

# ── ÉTAPE 1 : Vérification des ressources ─────────────────────────────────────
print("\n[ÉTAPE 1] Vérification des ressources")
print("-" * 40)

os.makedirs(RESOURCES_PATH, exist_ok=True)
pdf_files = [f for f in os.listdir(RESOURCES_PATH) if f.endswith(".pdf")]

if not pdf_files:
    print("⚠️  Aucun PDF trouvé dans ./resources/")
    print("   → Création d'un document de démonstration...")
    
    # Créer un fichier texte de démonstration
    demo_content = """
    Intelligence Artificielle et Agents IA
    =======================================
    
    L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer
    des systèmes capables d'effectuer des tâches nécessitant normalement l'intelligence humaine.
    
    Les Agents IA
    =============
    Un agent IA est un système autonome capable de percevoir son environnement,
    de raisonner et d'agir pour atteindre des objectifs définis.
    
    Architecture d'un Agent : MIND + BODY
    - MIND : Le cerveau de l'agent (LLM)
    - BODY : Les outils et capacités d'action (Tools, RAG, API)
    
    LangChain
    =========
    LangChain est un framework Python pour construire des applications IA.
    Il fournit des abstractions haut niveau pour chaîner des prompts,
    utiliser des outils et gérer la mémoire des agents.
    
    LangGraph
    =========
    LangGraph est un framework bas niveau pour orchestrer des agents complexes
    sous forme de graphe d'états et de transitions.
    Il permet : boucles, branches conditionnelles, human-in-the-loop,
    checkpointing et workflows multi-étapes.
    
    Le Pattern ReAct
    ================
    ReAct = Reasoning + Acting
    1. Reasoning : Le LLM analyse la situation et les outils disponibles
    2. Action    : Appel d'un outil (API, recherche, calcul)
    3. Observation : Résultat de l'action
    4. Répéter jusqu'à la réponse finale
    
    RAG - Retrieval Augmented Generation
    =====================================
    Le RAG permet d'augmenter les capacités d'un LLM avec des données privées.
    Étapes : Ingestion → Chunking → Embedding → Stockage → Retrieval → Génération
    
    Agentic RAG
    ===========
    Dans un RAG agentique, le retriever devient un outil parmi d'autres.
    L'agent décide dynamiquement quand utiliser la base de connaissances.
    Il peut faire des sous-requêtes multiples et combiner les résultats.
    """
    
    # Créer plusieurs "documents" de démo
    demo_files = {
        "ai_fundamentals.txt": demo_content,
        "langchain_guide.txt": """
LangChain Framework Guide
==========================

Installation: pip install langchain langchain-openai

Composants principaux:
- LLMs et ChatModels : Interface avec les modèles de langage
- Prompts : Templates de prompts réutilisables  
- Chains : Séquences de composants
- Agents : Systèmes autonomes avec outils
- Memory : Persistance de l'état conversationnel
- Tools : Capacités d'action des agents

Outils prédéfinis dans LangChain Community:
- DuckDuckGoSearchRun : Recherche web gratuite
- TavilySearchResults : Recherche IA avancée
- PythonREPLTool : Exécution de code Python
- WikipediaQueryRun : Recherche Wikipedia
- ArxivQueryRun : Recherche articles scientifiques
        """
    }
    
    for filename, content in demo_files.items():
        with open(os.path.join(RESOURCES_PATH, filename), 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"   ✅ Documents de démo créés dans {RESOURCES_PATH}")
    USE_DEMO = True
else:
    print(f"✅ {len(pdf_files)} PDF(s) trouvé(s) :")
    for f in pdf_files:
        print(f"   - {f}")
    USE_DEMO = False


# ── ÉTAPE 2 : Chargement et découpage des documents ───────────────────────────
print("\n[ÉTAPE 2] Chargement et découpage")
print("-" * 40)

if USE_DEMO:
    # Charger les fichiers texte de démo
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    loader = DirectoryLoader(RESOURCES_PATH, glob="*.txt", loader_cls=TextLoader)
else:
    loader = PyPDFDirectoryLoader(path=RESOURCES_PATH)

documents = loader.load()
print(f"✅ {len(documents)} document(s) chargé(s)")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunk(s) créé(s)")


# ── ÉTAPE 3 : Création du Vector Store ───────────────────────────────────────
print("\n[ÉTAPE 3] Création du Vector Store (ChromaDB)")
print("-" * 40)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST_DIR
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print(f"✅ Vector store créé avec {len(chunks)} vecteurs")


# ── ÉTAPE 4 : Définition du Retriever Tool ────────────────────────────────────
print("\n[ÉTAPE 4] Définition du Retriever Tool")
print("-" * 40)

@tool
def retriever_tool(query: str) -> str:
    """
    Recherche dans la base de connaissances privée.
    Utilise cet outil pour répondre aux questions sur les documents fournis.
    Retourne les extraits les plus pertinents.
    
    Args:
        query: Question ou requête de recherche
    Returns:
        Contexte pertinent extrait des documents
    """
    logger.info(f"Retriever appelé avec : '{query}'")
    try:
        relevant_docs = retriever.invoke(query)
        if not relevant_docs:
            return "Aucun document pertinent trouvé pour cette requête."
        
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('source', 'Document')
            page = doc.metadata.get('page', '')
            page_info = f" (page {page})" if page else ""
            context_parts.append(f"[Source {i}: {source}{page_info}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"

print("✅ Retriever tool défini")

# Test du retriever
test_result = retriever_tool.invoke("agent IA LangGraph")
print(f"\nTest retriever : {test_result[:200]}...")


# ── ÉTAPE 5 : Configuration LLM + Tools ──────────────────────────────────────
print("\n[ÉTAPE 5] Configuration LLM + Tools")
print("-" * 40)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools=tools)
tool_map = {t.name: t for t in tools}

SYSTEM_PROMPT = """Tu es un assistant intelligent spécialisé dans l'analyse de documents.

Règles :
1. Utilise TOUJOURS l'outil 'retriever_tool' pour répondre aux questions sur les documents.
2. Base tes réponses UNIQUEMENT sur le contexte fourni par le retriever.
3. Cite toujours la source des informations.
4. Si l'information n'est pas dans les documents, dis "Cette information n'est pas disponible dans mes documents."
5. Réponds en français de manière claire et structurée.
"""

print("✅ LLM configuré avec tools")


# ── ÉTAPE 6 : Agent RAG Agentique ────────────────────────────────────────────
print("\n[ÉTAPE 6] Agent RAG Agentique")
print("-" * 40)

class AgenticRAGChatbot:
    """Chatbot avec RAG Agentique et mémoire conversationnelle."""
    
    def __init__(self, llm_with_tools, tool_map: dict, system_prompt: str):
        self.llm = llm_with_tools
        self.tool_map = tool_map
        self.conversation_history: list = [SystemMessage(content=system_prompt)]
        self.query_count = 0
    
    def chat(self, user_input: str) -> str:
        self.query_count += 1
        logger.info(f"[Q{self.query_count}] User: {user_input}")
        
        # Ajouter le message utilisateur à l'historique
        self.conversation_history.append(HumanMessage(content=user_input))
        
        # Boucle ReAct
        for iteration in range(5):
            response = self.llm.invoke(self.conversation_history)
            self.conversation_history.append(response)
            
            if not response.tool_calls:
                logger.info(f"[Q{self.query_count}] Réponse finale (iter {iteration+1})")
                return response.content
            
            logger.info(f"[Q{self.query_count}] Iter {iteration+1}: {len(response.tool_calls)} tool(s) appelé(s)")
            
            # Exécuter les tools
            for tc in response.tool_calls:
                tool_name = tc["name"]
                logger.info(f"  → {tool_name}({tc['args']})")
                
                from langchain_core.messages import ToolMessage
                try:
                    result = self.tool_map[tool_name].invoke(tc["args"])
                    self.conversation_history.append(
                        ToolMessage(content=str(result), tool_call_id=tc["id"])
                    )
                    logger.info(f"  ← Résultat : {str(result)[:100]}...")
                except Exception as e:
                    error_msg = f"Erreur tool {tool_name}: {str(e)}"
                    self.conversation_history.append(
                        ToolMessage(content=error_msg, tool_call_id=tc["id"])
                    )
        
        return "Désolé, je n'ai pas pu générer une réponse complète."
    
    def clear_history(self):
        """Réinitialise l'historique (garde le system message)."""
        self.conversation_history = [self.conversation_history[0]]
        print("🗑️ Historique effacé")
    
    def show_stats(self):
        """Affiche les statistiques du chatbot."""
        print(f"\n📊 Statistiques :")
        print(f"   - Total questions : {self.query_count}")
        print(f"   - Messages en mémoire : {len(self.conversation_history)}")


# ── ÉTAPE 7 : Tests du Chatbot RAG ───────────────────────────────────────────
print("\n[ÉTAPE 7] Tests du Chatbot RAG Agentique")
print("=" * 60)

chatbot = AgenticRAGChatbot(
    llm_with_tools=llm_with_tools,
    tool_map=tool_map,
    system_prompt=SYSTEM_PROMPT
)

test_questions = [
    "Qu'est-ce qu'un agent IA ?",
    "Explique-moi le pattern ReAct",
    "Quelle est la différence entre LangChain et LangGraph ?",
    "Qu'est-ce que le RAG agentique ?",
    "Quels sont les outils prédéfinis dans LangChain ?",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'─' * 50}")
    print(f"❓ Question {i}: {question}")
    print(f"{'─' * 50}")
    answer = chatbot.chat(question)
    print(f"🤖 Réponse:\n{answer}")

chatbot.show_stats()

# ── ÉTAPE 8 : Mode interactif (optionnel) ─────────────────────────────────────
print("\n" + "=" * 60)
print("MODE INTERACTIF")
print("=" * 60)
print("Tapez 'quit' pour quitter, 'clear' pour effacer l'historique")
print("─" * 60)

try:
    while True:
        user_input = input("\n💬 Vous : ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break
        if user_input.lower() == "clear":
            chatbot.clear_history()
            continue
        if user_input.lower() == "stats":
            chatbot.show_stats()
            continue
        
        response = chatbot.chat(user_input)
        print(f"\n🤖 Assistant : {response}")

except (KeyboardInterrupt, EOFError):
    print("\n\n✅ Session terminée.")
    chatbot.show_stats()
