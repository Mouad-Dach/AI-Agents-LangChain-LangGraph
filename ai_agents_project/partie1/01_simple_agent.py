"""
============================================================
Partie 1 - Fichier 01 : Agent Simple avec LangChain
============================================================
Objectif : Créer un agent de base avec un LLM OpenAI
Concept  : Agent = MIND (LLM) + BODY (Tools + Memory)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── Chargement des variables d'environnement ──────────────────────────────────
load_dotenv()

# ── 1. Initialisation du LLM ─────────────────────────────────────────────────
print("=" * 60)
print("01 - AGENT SIMPLE LANGCHAIN")
print("=" * 60)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── 2. Test basique : appel direct au LLM ────────────────────────────────────
print("\n[TEST 1] Appel direct au LLM")
print("-" * 40)

response = llm.invoke("Bonjour ! Qui es-tu ?")
print(f"Réponse : {response.content}")

# ── 3. Utilisation d'un System Message ───────────────────────────────────────
print("\n[TEST 2] Avec System Message")
print("-" * 40)

messages = [
    SystemMessage(content="Tu es un assistant expert en IA et en machine learning. "
                           "Tu réponds toujours en français de manière concise."),
    HumanMessage(content="Explique-moi ce qu'est un agent IA en 3 lignes.")
]

response = llm.invoke(messages)
print(f"Réponse : {response.content}")

# ── 4. Utilisation d'un Prompt Template ──────────────────────────────────────
print("\n[TEST 3] Avec Prompt Template")
print("-" * 40)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant spécialisé en {domain}. Réponds en français."),
    ("human", "{question}")
])

# Créer une chaîne prompt | llm
chain = prompt | llm

response = chain.invoke({
    "domain": "intelligence artificielle",
    "question": "Quelle est la différence entre LangChain et LangGraph ?"
})
print(f"Réponse : {response.content}")

# ── 5. Conversation multi-tours ───────────────────────────────────────────────
print("\n[TEST 4] Conversation multi-tours")
print("-" * 40)

conversation_history = [
    SystemMessage(content="Tu es un assistant IA amical. Réponds en français.")
]

questions = [
    "Qu'est-ce que le ReAct pattern ?",
    "Peux-tu me donner un exemple concret ?",
    "Merci, et comment l'implémenter avec LangGraph ?"
]

for question in questions:
    conversation_history.append(HumanMessage(content=question))
    response = llm.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response.content))
    print(f"\nQ: {question}")
    print(f"A: {response.content[:200]}...")

print("\n✅ Agent simple OK !")
