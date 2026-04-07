"""
============================================================
Partie 3 - Fichier 04 : ReAct Agent avec LangGraph
============================================================
Objectif : Implémenter un agent ReAct complet avec LangGraph
           Reproduction exacte de la démo du cours

Graphe :
  START → assistant
            ├─[end]      → END
            └─[continue] → tools → assistant (boucle)
"""

import os
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage,
    SystemMessage, ToolMessage
)
from langchain.tools import tool

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

load_dotenv()

print("=" * 60)
print("PARTIE 3 - REACT AGENT LANGGRAPH (Démo Cours)")
print("=" * 60)


# ── 1) État avec Reducer ──────────────────────────────────────────────────────
class AgentState(TypedDict):
    # add_messages : reducer qui accumule les messages
    messages: Annotated[List[BaseMessage], add_messages]


# ── 2) Définition des Tools (reproduction exacte du cours) ───────────────────
@tool
def add(a: float, b: float) -> float:
    """Add two float numbers"""
    print(f"  Adding 2 numbers a={a}, b={b}\n")
    return a + b

@tool
def divide(a: float, b: float) -> float:
    """Divide two float numbers"""
    print(f"  Dividing 2 numbers a={a}, b={b}\n")
    return a / b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two float numbers"""
    print(f"  Multiplying 2 numbers a={a}, b={b} \n")
    return a * b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two float numbers"""
    print(f"  Subtracting 2 numbers a={a}, b={b}\n")
    return a - b

@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent"""
    print(f"  Power {base}^{exponent}\n")
    return base ** exponent

tools = [add, multiply, divide, subtract, power]

print(f"✅ {len(tools)} tools définis : {[t.name for t in tools]}")


# ── 3) LLM avec tools ────────────────────────────────────────────────────────
llm_with_tools = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
).bind_tools(tools=tools)

print("✅ LLM gpt-4o lié aux tools")


# ── 4) Nœud Agent ─────────────────────────────────────────────────────────────
def assistant(state: AgentState) -> AgentState:
    """Nœud principal : appelle le LLM avec l'historique."""
    system_message = SystemMessage(
        content="Answer the user question using provided tools"
    )
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


# ── 5) Fonction de routage ────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Détermine si on continue (tools) ou on termine."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# ── 6) Construction du graphe ─────────────────────────────────────────────────
print("\n[Construction]")

graph_builder = StateGraph(AgentState)

graph_builder.add_node("assistant", assistant)
graph_builder.add_node("tools",     ToolNode(tools))  # ToolNode prédéfini

graph_builder.add_edge(START, "assistant")

graph_builder.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "continue": "tools",
        "end":       END
    }
)

graph_builder.add_edge("tools", "assistant")

graph = graph_builder.compile()
print("✅ Agent ReAct LangGraph compilé")

try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


# ── 7) Version avec tools_condition (prédéfini LangGraph) ─────────────────────
print("\n[Version tools_condition prédéfini]")

graph_v2_builder = StateGraph(AgentState)
graph_v2_builder.add_node("assistant", assistant)
graph_v2_builder.add_node("tools",     ToolNode(tools))
graph_v2_builder.add_edge(START, "assistant")
# tools_condition : version prédéfinie de should_continue
graph_v2_builder.add_conditional_edges("assistant", tools_condition)
graph_v2_builder.add_edge("tools", "assistant")
graph_v2 = graph_v2_builder.compile()
print("✅ Version tools_condition compilée")


# ── 8) Tests (reproduction exacte du cours) ───────────────────────────────────
print("\n" + "=" * 60)
print("TESTS - Reproduction exacte du cours")
print("=" * 60)

# --- Test 1 : Calcul multi-étapes ---
print("\n[TEST 1] Add 6 to 7 then divide by 2 and multiply by 5")
print("─" * 55)

resp = graph.invoke(
    input={
        "messages": [
            HumanMessage(
                content="""Compute this expression : 
Add 6 to 7 then divide by 2 and multiply by 5. 
Puis donne moi un proverbe"""
            )
        ]
    }
)
print("\nRésultat final :")
print(resp["messages"][-1].content)

# --- Test 2 : Add 5 to 80 then multiply by 23 ---
print("\n\n[TEST 2] Add 5 to 80 then multiply by 23")
print("─" * 55)

resp2 = graph.invoke(
    input={
        "messages": [
            HumanMessage(
                content="Add 5 to 80 then multiply by 23 et donne moi un proverbe en arabe"
            )
        ]
    }
)
print(resp2["messages"][-1].content)

# --- Test 3 : Streaming ---
print("\n\n[TEST 3] Streaming mode")
print("─" * 55)

inputs = {
    "messages": [
        HumanMessage(
            content="Add 6 to 7 then divide the result by 2. Donne moi un proverbe"
        )
    ]
}

stream = graph.stream(inputs, stream_mode="values")

for s in stream:
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()

# --- Test 4 : Questions combinées ---
print("\n\n[TEST 4] Calcul complexe")
print("─" * 55)

resp4 = graph.invoke({
    "messages": [
        HumanMessage(
            content="Calcule : (15 + 7) × 3, puis divise le résultat par 2, "
                    "puis élève à la puissance 2. Donne le résultat final."
        )
    ]
})
print("Résultat :", resp4["messages"][-1].content)

# --- Test 5 : Voir tout l'historique de messages ---
print("\n\n[TEST 5] Historique complet des messages")
print("─" * 55)

resp5 = graph.invoke({
    "messages": [HumanMessage(content="Multiplie 12 par 8 puis soustrait 16")]
})

print("\n📜 Historique complet :")
for msg in resp5["messages"]:
    msg.pretty_print()

print("\n✅ Agent ReAct LangGraph OK !")
