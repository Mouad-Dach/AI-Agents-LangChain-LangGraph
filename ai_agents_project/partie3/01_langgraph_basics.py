"""
============================================================
Partie 3 - Fichier 01 : Bases de LangGraph
============================================================
Objectif : Comprendre les concepts fondamentaux de LangGraph
           - State (TypedDict / Pydantic / Dataclass)
           - Nodes (fonctions)
           - Edges (normales et conditionnelles)
           - Compilation et exécution
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

print("=" * 60)
print("PARTIE 3 - BASES LANGGRAPH")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1 : Graphe ultra-simple (START → node1 → END)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DEMO 1] Graphe Simple : START → node1 → END")
print("-" * 50)

# 1. Définir l'état
class SimpleState(TypedDict):
    messages: list[AnyMessage]
    counter:  int

# 2. Définir un nœud
def node1(state: SimpleState) -> SimpleState:
    """Nœud simple qui ajoute un message IA."""
    messages = state["messages"]
    new_message = AIMessage(content="Bonjour depuis node1 !")
    return {"messages": messages + [new_message], "counter": 10}

# 3. Construire le graphe
workflow = StateGraph(SimpleState)
workflow.add_node("node1", node1)
workflow.add_edge(START, "node1")
workflow.add_edge("node1", END)

# 4. Compiler
graph = workflow.compile()
print("✅ Graphe compilé")

# 5. Exécuter
result = graph.invoke({
    "messages": [HumanMessage(content="Hi !")],
    "counter": 0
})

print("\nRésultat :")
for msg in result["messages"]:
    role = type(msg).__name__.replace("Message", "")
    print(f"  [{role}]: {msg.content}")
print(f"  Counter: {result['counter']}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2 : Différents types d'État
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DEMO 2] Types d'État : TypedDict vs Pydantic vs Dataclass")
print("-" * 50)

# TypedDict
from typing import TypedDict as TD
class StateTypedDict(TD):
    name:  str
    value: int

# Pydantic
from pydantic import BaseModel
class StatePydantic(BaseModel):
    name:  str
    value: int

# Dataclass
from dataclasses import dataclass
@dataclass
class StateDataclass:
    name:  str
    value: int

print("✅ TypedDict  :", StateTypedDict(name="test", value=1))
print("✅ Pydantic   :", StatePydantic(name="test", value=1))
print("✅ Dataclass  :", StateDataclass(name="test", value=1))


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3 : Graphe avec plusieurs nœuds séquentiels
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DEMO 3] Graphe Multi-Nœuds Séquentiels")
print("-" * 50)

class PipelineState(TypedDict):
    text:    str
    result1: str
    result2: str
    result3: str

def step1_uppercase(state: PipelineState) -> PipelineState:
    print("  → Nœud 1 : Uppercase")
    return {"text": state["text"], "result1": state["text"].upper(),
            "result2": "", "result3": ""}

def step2_count_words(state: PipelineState) -> PipelineState:
    print("  → Nœud 2 : Comptage mots")
    word_count = len(state["text"].split())
    return {**state, "result2": f"{word_count} mots"}

def step3_reverse(state: PipelineState) -> PipelineState:
    print("  → Nœud 3 : Inversion")
    return {**state, "result3": state["text"][::-1]}

pipeline = StateGraph(PipelineState)
pipeline.add_node("uppercase",   step1_uppercase)
pipeline.add_node("count_words", step2_count_words)
pipeline.add_node("reverse",     step3_reverse)

pipeline.add_edge(START,        "uppercase")
pipeline.add_edge("uppercase",  "count_words")
pipeline.add_edge("count_words","reverse")
pipeline.add_edge("reverse",     END)

graph3 = pipeline.compile()

result3 = graph3.invoke({
    "text":    "LangGraph est un framework puissant pour les agents IA",
    "result1": "",
    "result2": "",
    "result3": ""
})

print(f"\nTexte original : {result3['text']}")
print(f"  Uppercase    : {result3['result1'][:50]}...")
print(f"  Comptage     : {result3['result2']}")
print(f"  Inversé      : {result3['result3'][:50]}...")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4 : Graphe LLM simple
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DEMO 4] Graphe avec LLM")
print("-" * 50)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

class LLMState(TypedDict):
    messages:  list[AnyMessage]
    language:  str

def llm_node(state: LLMState) -> LLMState:
    """Nœud LLM qui appelle le modèle."""
    from langchain_core.messages import SystemMessage
    lang = state.get("language", "français")
    system = SystemMessage(content=f"Tu es un assistant utile. Réponds en {lang}.")
    all_messages = [system] + state["messages"]
    response = llm.invoke(all_messages)
    return {**state, "messages": state["messages"] + [response]}

wf4 = StateGraph(LLMState)
wf4.add_node("llm", llm_node)
wf4.add_edge(START, "llm")
wf4.add_edge("llm",  END)
graph4 = wf4.compile()

result4 = graph4.invoke({
    "messages":  [HumanMessage(content="Qu'est-ce que LangGraph en une phrase ?")],
    "language":  "français"
})

print(f"Q: {result4['messages'][0].content}")
print(f"A: {result4['messages'][-1].content}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 5 : Reducers et gestion de l'état
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DEMO 5] Reducers - add_messages")
print("-" * 50)

from typing import Annotated
from langgraph.graph.message import add_messages

class StateWithReducer(TypedDict):
    # add_messages : reducer qui AJOUTE plutôt que remplacer
    messages: Annotated[list[AnyMessage], add_messages]
    count:    int

def node_reducer(state: StateWithReducer) -> dict:
    """Avec reducer add_messages, on retourne juste le nouveau message."""
    new_msg = AIMessage(content=f"Message {state['count'] + 1}")
    return {"messages": [new_msg], "count": state["count"] + 1}

wf5 = StateGraph(StateWithReducer)
wf5.add_node("add_msg", node_reducer)
wf5.add_edge(START,      "add_msg")
wf5.add_edge("add_msg",   END)
graph5 = wf5.compile()

# Invoquer 3 fois pour voir l'accumulation
state5 = {"messages": [HumanMessage(content="Start")], "count": 0}
state5 = graph5.invoke(state5)
state5 = graph5.invoke(state5)
state5 = graph5.invoke(state5)

print(f"Nombre de messages après 3 invocations : {len(state5['messages'])}")
for m in state5["messages"]:
    role = type(m).__name__.replace("Message", "")
    print(f"  [{role}]: {m.content}")

print("\n✅ Bases LangGraph OK !")
