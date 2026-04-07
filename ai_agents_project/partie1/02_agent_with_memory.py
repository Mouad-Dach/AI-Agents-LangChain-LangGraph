"""
============================================================
Partie 1 - Fichier 02 : Agent avec Mémoire
============================================================
Objectif : Ajouter une mémoire persistante à l'agent
Types    : - ConversationBufferMemory (tout l'historique)
           - ConversationBufferWindowMemory (fenêtre glissante)
           - ConversationSummaryMemory (résumé automatique)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

print("=" * 60)
print("02 - AGENT AVEC MÉMOIRE")
print("=" * 60)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── MÉTHODE 1 : Mémoire manuelle (liste de messages) ─────────────────────────
print("\n[MÉTHODE 1] Mémoire manuelle avec liste de messages")
print("-" * 50)

class SimpleMemoryAgent:
    """Agent avec mémoire manuelle basée sur une liste de messages."""
    
    def __init__(self, llm, system_prompt: str):
        self.llm = llm
        self.memory: list = [SystemMessage(content=system_prompt)]
    
    def chat(self, user_input: str) -> str:
        # Ajouter le message de l'utilisateur
        self.memory.append(HumanMessage(content=user_input))
        # Appeler le LLM avec tout l'historique
        response = self.llm.invoke(self.memory)
        # Sauvegarder la réponse dans la mémoire
        self.memory.append(AIMessage(content=response.content))
        return response.content
    
    def show_memory(self):
        print("\n📚 Historique de la conversation :")
        for i, msg in enumerate(self.memory):
            role = type(msg).__name__.replace("Message", "")
            print(f"  [{i}] {role}: {str(msg.content)[:80]}...")
    
    def clear_memory(self):
        system_msg = self.memory[0]  # Garder le system message
        self.memory = [system_msg]
        print("🗑️ Mémoire effacée (system message conservé)")


agent = SimpleMemoryAgent(
    llm=llm,
    system_prompt="Tu es un assistant expert en IA. Tu te souviens de tout ce qu'on te dit. Réponds en français."
)

# Simulation d'une conversation
print("\n💬 Conversation :")
r1 = agent.chat("Je m'appelle Mohammed et j'étudie l'IA à l'ENSET.")
print(f"User: Je m'appelle Mohammed et j'étudie l'IA à l'ENSET.")
print(f"Agent: {r1[:150]}...\n")

r2 = agent.chat("Quel est mon prénom ?")
print(f"User: Quel est mon prénom ?")
print(f"Agent: {r2}\n")

r3 = agent.chat("Et où est-ce que j'étudie ?")
print(f"User: Et où est-ce que j'étudie ?")
print(f"Agent: {r3}\n")

agent.show_memory()

# ── MÉTHODE 2 : InMemoryChatMessageHistory + RunnableWithMessageHistory ───────
print("\n\n[MÉTHODE 2] RunnableWithMessageHistory (LangChain natif)")
print("-" * 50)

# Stockage des sessions en mémoire (dict session_id -> historique)
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retourne ou crée l'historique pour une session donnée."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Prompt avec placeholder pour l'historique
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant IA. Réponds toujours en français. "
               "Tu te souviens de toute la conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Chaîne avec gestion de l'historique
chain = prompt | llm

agent_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Configuration de session
config_session_1 = {"configurable": {"session_id": "session_001"}}
config_session_2 = {"configurable": {"session_id": "session_002"}}

print("\n🔵 Session 1 (Mohammed) :")
resp = agent_with_history.invoke(
    {"input": "Bonjour, je suis Mohammed, étudiant en IA."},
    config=config_session_1
)
print(f"  Agent: {resp.content[:150]}")

resp = agent_with_history.invoke(
    {"input": "Explique-moi LangGraph en 2 phrases."},
    config=config_session_1
)
print(f"  Agent: {resp.content[:200]}")

resp = agent_with_history.invoke(
    {"input": "Rappelle-moi mon prénom."},
    config=config_session_1
)
print(f"  Agent: {resp.content}")

print("\n🔴 Session 2 (Nouvelle session - pas de mémoire partagée) :")
resp = agent_with_history.invoke(
    {"input": "Quel est mon prénom ?"},
    config=config_session_2
)
print(f"  Agent: {resp.content[:150]}")

# Afficher les sessions en mémoire
print(f"\n📦 Sessions actives : {list(store.keys())}")
for session_id, history in store.items():
    print(f"  Session '{session_id}' : {len(history.messages)} messages")

# ── MÉTHODE 3 : Fenêtre glissante manuelle ────────────────────────────────────
print("\n\n[MÉTHODE 3] Mémoire avec fenêtre glissante (Window Memory)")
print("-" * 50)

class WindowMemoryAgent:
    """Agent avec fenêtre glissante : garde uniquement les N derniers échanges."""
    
    def __init__(self, llm, system_prompt: str, window_size: int = 3):
        self.llm = llm
        self.system_prompt = system_prompt
        self.window_size = window_size  # Nombre d'échanges (paires H/A) à garder
        self.history: list = []  # Stocke les paires (HumanMessage, AIMessage)
    
    def _build_messages(self) -> list:
        messages = [SystemMessage(content=self.system_prompt)]
        # Prendre les window_size dernières paires
        recent = self.history[-self.window_size:]
        for human_msg, ai_msg in recent:
            messages.append(human_msg)
            messages.append(ai_msg)
        return messages
    
    def chat(self, user_input: str) -> str:
        human_msg = HumanMessage(content=user_input)
        messages = self._build_messages()
        messages.append(human_msg)
        response = self.llm.invoke(messages)
        ai_msg = AIMessage(content=response.content)
        self.history.append((human_msg, ai_msg))
        print(f"  [Fenêtre active: {min(len(self.history), self.window_size)}/{self.window_size} échanges]")
        return response.content


window_agent = WindowMemoryAgent(
    llm=llm,
    system_prompt="Tu es un assistant IA. Réponds en français.",
    window_size=2
)

exchanges = [
    "Je m'appelle Yassine.",
    "J'habite à Casablanca.",
    "J'étudie le machine learning.",
    "Quel est mon prénom et où j'habite ?"
]

for msg in exchanges:
    print(f"\n  User: {msg}")
    resp = window_agent.chat(msg)
    print(f"  Agent: {resp[:150]}")

print("\n✅ Agents avec mémoire OK !")
