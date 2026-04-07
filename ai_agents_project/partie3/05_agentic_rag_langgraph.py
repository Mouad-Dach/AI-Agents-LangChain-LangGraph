"""
============================================================
Partie 3 - Fichier 05 : Agentic RAG avec LangGraph
============================================================
Objectif : Reproduire la démo Agentic RAG du cours
           en utilisant LangGraph pour l'orchestration

Architecture :
  START → agent
            ├─[end]  → END
            └─[tools]→ tools → agent (boucle ReAct)

Ressources : PDFs dans ./resources/ (IBM, GOOGL, AMZN, MSFT, NVDA)
"""

import os
import logging
from typing import Annotated, Sequence, Literal, TypedDict
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage
)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools import tool

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("PARTIE 3 - AGENTIC RAG avec LangGraph")
print("=" * 60)

RESOURCES_PATH   = "./resources/"
CHROMA_PERSIST   = "./chroma_db_partie3"
COLLECTION_NAME  = "AI_Initiatives"

# ── ÉTAPE 1 : Vérification ressources ─────────────────────────────────────────
print("\n[ÉTAPE 1] Ressources")
os.makedirs(RESOURCES_PATH, exist_ok=True)
pdf_files = [f for f in os.listdir(RESOURCES_PATH) if f.endswith(".pdf")]

if not pdf_files:
    print("⚠️  Aucun PDF trouvé. Création de documents de démo...")
    
    demo_docs = {
        "IBM.txt": """IBM AI Initiatives Report 2024
IBM is a global technology leader focusing on enterprise AI.
Key AI Projects:
1. Watson X Platform: IBM's enterprise AI platform for business applications.
   Investment: Over $500M annually. Launch: 2023.
2. Granite LLM: IBM's family of foundation models for enterprise use.
   Specialized for code generation and business analytics.
3. AI for Cybersecurity: IBM Security QRadar uses AI for threat detection.
   Processes 150 billion security events daily.
4. Hybrid Cloud AI: Integration of AI with IBM Cloud and Red Hat OpenShift.
Strategy: IBM focuses on trusted, explainable AI for regulated industries.""",

        "MSFT.txt": """Microsoft AI Initiatives Report 2024
Microsoft Corporation, founded in 1975, is a global technology leader.
Key AI Projects:
1. Copilot: AI assistant integrated across Microsoft 365 suite.
   Powers productivity tools for 300M+ Office users.
2. Azure OpenAI Service: Enterprise access to GPT-4 and DALL-E.
   Used by 18,000+ organizations globally.
3. Phi Models: Small Language Models (SLMs) for edge deployment.
   Phi-3 achieves GPT-3.5 quality at fraction of the cost.
4. GitHub Copilot: AI code completion used by 1.3M+ developers.
Investment: Microsoft invested $13B in OpenAI partnership.
Enterprise Enablement: Provides organizations with AI tools and infrastructure.""",

        "NVDA.txt": """NVIDIA AI Initiatives Report 2024
NVIDIA is the world's leading AI chip manufacturer.
Key AI Projects:
1. Project G-Assist: On-device AI assistant for GeForce RTX PCs.
   Integrates LLM for gaming and system settings optimization.
2. Deep Learning Super Sampling (DLSS) 4: AI upscaling technology.
   Million-dollar range investment as critical to product strategy.
3. NIM (NVIDIA Inference Microservices): Deploy AI models anywhere.
   Supports LLaMA, Mistral, Stable Diffusion and 100+ models.
4. CUDA Platform: Foundation of modern deep learning.
   Used by 90% of AI researchers globally.
Investment: NVIDIA R&D spending exceeds $8B annually.
Revenue from AI datacenter products: $47B in FY2024.""",

        "GOOGL.txt": """Google AI Initiatives Report 2024
Google has positioned itself as a leader in AI research and deployment.
Key AI Projects:
1. Gemini: Google DeepMind's multimodal foundation model.
   Gemini Ultra, Pro, and Nano for different deployment scenarios.
2. AI Ecosystem: Google integrates AI into Search, Gmail, Google Assistant.
   Spending several hundred million dollars annually on AI research.
3. Google Cloud Vertex AI: Enterprise ML platform.
   Supports training, deployment and monitoring of ML models.
4. AlphaFold: Revolutionary protein structure prediction AI.
   Database contains 200M+ protein structure predictions.
Investment: Google parent Alphabet spends $45B+ on R&D annually.""",

        "AMZN.txt": """Amazon AI Initiatives Report 2024
Amazon Web Services (AWS) is the world's largest cloud provider.
Key AI Projects:
1. Bedrock: AWS managed service for foundation models.
   Access to Anthropic Claude, Llama, Titan and 30+ models.
2. CodeWhisperer: AI coding assistant integrated in AWS IDE.
   Free tier available for individual developers.
3. Alexa+: Next-generation conversational AI assistant.
   Powered by large language models for complex reasoning.
4. SageMaker: Complete ML development platform.
   Trusted by 100,000+ customers including Netflix, NASA.
Investment: Amazon allocates significant budget to AWS AI services.
Robotics AI: Amazon uses AI in 750,000+ warehouse robots."""
    }
    
    for fname, content in demo_docs.items():
        fpath = os.path.join(RESOURCES_PATH, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"   ✅ {len(demo_docs)} documents de démo créés")
    USE_DEMO = True
else:
    print(f"✅ {len(pdf_files)} PDF(s) trouvé(s) : {pdf_files}")
    USE_DEMO = False


# ── ÉTAPE 2 : Chargement et découpage ─────────────────────────────────────────
print("\n[ÉTAPE 2] Chargement et découpage des documents")

if USE_DEMO:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    loader = DirectoryLoader(RESOURCES_PATH, glob="*.txt", loader_cls=TextLoader)
else:
    loader = PyPDFDirectoryLoader(path=RESOURCES_PATH)

# Reproduction exacte du cours
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=200
)

ai_initiative_chunks = loader.load_and_split(text_splitter=text_splitter)
logger.info(f"Chargé {len(ai_initiative_chunks)} chunks")


# ── ÉTAPE 3 : Vector Store ────────────────────────────────────────────────────
print("\n[ÉTAPE 3] Création du Vector Store")

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=ai_initiative_chunks,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

logger.info(f"Vector store créé avec {len(ai_initiative_chunks)} vecteurs")


# ── ÉTAPE 4 : Test manuel du retriever (Step 5 du cours) ─────────────────────
print("\n[ÉTAPE 4] Test RAG Retrieval Manuel")

test_query = "What AI projects is Microsoft working on?"
print(f"🔍 Test Query: {test_query}")
print("=" * 80)

relevant_docs = retriever.invoke(test_query)
print(f"\nRetrieved {len(relevant_docs)} relevant chunks:\n")

for i, doc in enumerate(relevant_docs[:3], 1):
    source = doc.metadata.get('source', 'Unknown')
    page   = doc.metadata.get('page',   'Unknown')
    print(f"📄 Result {i}:")
    print(f"   Source  : {source}")
    print(f"   Page    : {page}")
    print(f"   Content : {doc.page_content[:100]}...")
    print("-" * 80 + "\n")


# ── ÉTAPE 5 : Retriever Tool (Step 6 du cours) ───────────────────────────────
print("\n[ÉTAPE 5] Définition du Retriever Tool")

@tool
def retriever_tool(query: str) -> str:
    """
    Interrogez la base de données privée des rapports d'analystes et des documents
    sur les initiatives en IA.
    
    Cet outil alimenté par RAG effectue des recherches dans des documents internes
    de l'entreprise concernant les initiatives en IA, les projets de recherche,
    les domaines d'innovation et les investissements technologiques stratégiques.
    
    Utilisez cet outil lorsque vous avez besoin d'informations sur :
    - Les projets de recherche et les initiatives en IA de l'entreprise
    - Les domaines d'innovation en IA et les axes prioritaires
    - Les feuilles de route technologiques et les plans futurs
    - Les calendriers et les détails de projets d'IA spécifiques
    
    Args:
        query: Requête en langage naturel sur les initiatives en IA de l'entreprise
    Returns:
        str: Réponse détaillée fondée sur des rapports privés d'analystes
    """
    try:
        relevant_document_chunks = retriever.invoke(query)
        context_list = [d.page_content for d in relevant_document_chunks]
        context_for_query = ". ".join(context_list)
        return context_for_query
    except Exception as e:
        return f"Error querying private database: {str(e)}"


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

tools = [retriever_tool]
llm_with_tools = model.bind_tools(tools)

print("✅ Retriever tool défini et LLM configuré")


# ── ÉTAPE 6 : State et System Message (reproduction exacte) ──────────────────
print("\n[ÉTAPE 6] State et System Message")

class SimpleAgentState(TypedDict):
    """State for the financial research agent."""
    messages: Annotated[Sequence, add_messages]


agentic_rag_system_message = """ 
Vous êtes un assistant spécialisé dans l'examen des 
initiatives en IA des entreprises et dans la fourniture 
de réponses exactes fondées sur le contexte fourni.
Les données fournies par l'utilisateur incluront tout le 
contexte nécessaire pour répondre à sa question.
Le contexte contient des références à des initiatives, 
projets ou programmes d'IA spécifiques d'entreprises, 
pertinents pour la requête de l'utilisateur.
Répondez uniquement en utilisant le contexte fourni. 
N'ajoutez pas d'informations externes et ne mentionnez 
pas le contexte dans votre réponse.
Citez toujours de quelle entreprise provient 
l'information.
Si la réponse ne peut pas être trouvée dans le contexte, 
répondez par : "Je ne sais pas - cette information n'est 
pas disponible dans nos rapports d'analystes."
"""


# ── ÉTAPE 7 : Agent Node ──────────────────────────────────────────────────────
def agent_node(state: SimpleAgentState) -> dict:
    """Agent node that calls the LLM with system prompt and current state."""
    logger.info("AGENT NODE: Processing request...")
    
    system_msg = SystemMessage(content=agentic_rag_system_message)
    messages   = [system_msg] + list(state["messages"])
    
    logger.info("Calling LLM with tools...")
    response = llm_with_tools.invoke(messages)
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"✓ Agent decided to use {len(response.tool_calls)} tool(s)")
        for i, tool_call in enumerate(response.tool_calls, 1):
            logger.info(f"  {i}. {tool_call['name']}")
    else:
        logger.info("✓ Agent generated final response (no tools needed)")
    
    return {"messages": [response]}


# ── ÉTAPE 8 : Routing ─────────────────────────────────────────────────────────
def should_continue(state: SimpleAgentState) -> Literal["tools", "end"]:
    """Determines whether to continue to tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("ROUTING: Continuing to TOOLS node")
        return "tools"
    logger.info("ROUTING: Ending workflow (final response ready)")
    return "end"


# ── ÉTAPE 9 : Construire le graphe ────────────────────────────────────────────
print("\n[ÉTAPE 9] Construction du graphe LangGraph")

# Tool node avec logging
original_tool_node = ToolNode(tools)

def tool_node_with_logging(state):
    logger.info("TOOL NODE: Executing tools...")
    result = original_tool_node.invoke(state)
    logger.info("Tools executed successfully")
    return result


workflow = StateGraph(SimpleAgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node_with_logging)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end":    END
    }
)

workflow.add_edge("tools", "agent")

graph = workflow.compile()
logger.info("✅ Enhanced agent created successfully\n")

try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


# ── ÉTAPE 10 : Tests (reproduction exacte Test 5 du cours) ───────────────────
print("\n" + "=" * 80)
print("TEST 5: Enhanced Agent with RAG - AI Research Activity Check")
print("=" * 80 + "\n")

# Requête principale du cours
query = ("Provide a comprehensive investment analysis for NVIDIA (NVDA) "
         "and Google including their AI research initiatives")

print(f"Query: {query}\n")
print("-" * 80 + "\n")

result = graph.invoke(
    {"messages": [HumanMessage(content=query)]}
)

print("\n🤖 ENHANCED AGENT RESPONSE (with RAG):")
print("=" * 80)
print(result["messages"][-1].content)
print("\n" + "=" * 80)

print("\n✅ Notice: The agent now includes:")
print("  • AI research projects from private analyst reports")
print("  • Specific AI initiative details")
print("  • Integration of financial + AI research data")
print("  • Comprehensive investment recommendation")


# ── ÉTAPE 11 : Tests supplémentaires ─────────────────────────────────────────
print("\n\n[TESTS SUPPLÉMENTAIRES]")

extra_queries = [
    "What AI projects is IBM working on?",
    "Compare Amazon and Microsoft AI cloud initiatives",
    "What is NVIDIA's strategy for on-device AI?",
]

for q in extra_queries:
    print(f"\n{'─' * 60}")
    print(f"❓ Query: {q}")
    print(f"{'─' * 60}")
    r = graph.invoke({"messages": [HumanMessage(content=q)]})
    print(f"🤖 Réponse:\n{r['messages'][-1].content[:500]}...")


# ── ÉTAPE 12 : Mode Chatbot interactif ───────────────────────────────────────
print("\n\n" + "=" * 60)
print("MODE CHATBOT INTERACTIF - Agentic RAG LangGraph")
print("=" * 60)
print("Commandes : 'quit' pour quitter, 'clear' pour nouvelle session")
print("─" * 60)

conversation: list = []

try:
    while True:
        user_input = input("\n💬 Vous : ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Session terminée.")
            break
        if user_input.lower() == "clear":
            conversation = []
            print("🗑️ Nouvelle session démarrée")
            continue
        
        conversation.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": conversation})
        
        # Mettre à jour la conversation avec tous les messages
        conversation = list(result["messages"])
        
        response_content = result["messages"][-1].content
        print(f"\n🤖 Assistant : {response_content}")

except (KeyboardInterrupt, EOFError):
    print("\n\n✅ Session terminée.")

print("\n✅ Agentic RAG LangGraph OK !")
