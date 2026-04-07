"""
============================================================
Partie 1 - Fichier 03 : Agent avec Tools Personnalisés
============================================================
Objectif : Définir et utiliser des outils custom avec @tool
Pattern  : ReAct = Reasoning → Action (Tool) → Observation → ...
"""

import os
import math
import json
import random
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.tools import tool

load_dotenv()

print("=" * 60)
print("03 - AGENT AVEC TOOLS PERSONNALISÉS (ReAct Pattern)")
print("=" * 60)

# ── 1. Définition des Tools ───────────────────────────────────────────────────
print("\n[ÉTAPE 1] Définition des outils")
print("-" * 40)

@tool
def add(a: float, b: float) -> float:
    """Additionne deux nombres flottants et retourne le résultat."""
    print(f"  🔧 Tool add appelé : {a} + {b}")
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Soustrait b de a et retourne le résultat."""
    print(f"  🔧 Tool subtract appelé : {a} - {b}")
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplie deux nombres flottants et retourne le résultat."""
    print(f"  🔧 Tool multiply appelé : {a} * {b}")
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divise a par b. Retourne une erreur si b est 0."""
    print(f"  🔧 Tool divide appelé : {a} / {b}")
    if b == 0:
        raise ValueError("Division par zéro impossible !")
    return a / b

@tool
def power(base: float, exponent: float) -> float:
    """Calcule base élevé à la puissance exponent."""
    print(f"  🔧 Tool power appelé : {base} ^ {exponent}")
    return math.pow(base, exponent)

@tool
def get_current_datetime() -> str:
    """Retourne la date et l'heure actuelles formatées."""
    now = datetime.now()
    result = now.strftime("%A %d %B %Y à %H:%M:%S")
    print(f"  🔧 Tool datetime appelé : {result}")
    return result

@tool
def get_weather_mock(city: str) -> str:
    """
    Simule la météo pour une ville donnée.
    Retourne température, conditions et humidité.
    Note: Données simulées pour démonstration.
    """
    print(f"  🔧 Tool weather appelé pour : {city}")
    # Simulation (en production : appel API OpenWeatherMap)
    conditions = ["Ensoleillé", "Nuageux", "Pluvieux", "Venteux", "Brumeux"]
    temp = random.randint(10, 35)
    humidity = random.randint(40, 90)
    condition = random.choice(conditions)
    return json.dumps({
        "ville": city,
        "temperature": f"{temp}°C",
        "condition": condition,
        "humidite": f"{humidity}%",
        "source": "Simulation (démo)"
    }, ensure_ascii=False)

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    Calcule l'IMC (Indice de Masse Corporelle).
    Args:
        weight_kg: Poids en kilogrammes
        height_m: Taille en mètres
    Returns:
        IMC avec interprétation
    """
    print(f"  🔧 Tool BMI : {weight_kg}kg / {height_m}m²")
    if height_m <= 0 or weight_kg <= 0:
        return "Valeurs invalides"
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Insuffisance pondérale"
    elif bmi < 25:
        category = "Poids normal"
    elif bmi < 30:
        category = "Surpoids"
    else:
        category = "Obésité"
    return f"IMC = {bmi:.2f} → {category}"

# Liste de tous les outils
tools = [add, subtract, multiply, divide, power,
         get_current_datetime, get_weather_mock, calculate_bmi]

print(f"✅ {len(tools)} outils définis :")
for t in tools:
    print(f"  - {t.name}: {t.description[:60]}...")


# ── 2. LLM lié aux Tools ─────────────────────────────────────────────────────
print("\n[ÉTAPE 2] Liaison LLM ↔ Tools")
print("-" * 40)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_with_tools = llm.bind_tools(tools=tools)
print("✅ LLM lié aux outils avec bind_tools()")


# ── 3. Agent ReAct Manuel ─────────────────────────────────────────────────────
print("\n[ÉTAPE 3] Agent ReAct Manuel")
print("-" * 40)

# Dictionnaire pour exécuter les tools par nom
tool_map = {t.name: t for t in tools}

def run_react_agent(user_query: str, max_iterations: int = 10) -> str:
    """
    Implémentation manuelle du pattern ReAct.
    Boucle : LLM → Tool Call → Observation → LLM → ...
    """
    print(f"\n🤖 Question : {user_query}")
    print("─" * 50)
    
    messages = [
        SystemMessage(content="Tu es un assistant IA qui utilise des outils pour répondre. "
                               "Réponds en français. Utilise les outils disponibles quand nécessaire."),
        HumanMessage(content=user_query)
    ]
    
    for iteration in range(max_iterations):
        print(f"\n  [Itération {iteration + 1}]")
        
        # 1. REASONING : Le LLM décide quoi faire
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Vérifier si le LLM veut appeler des tools
        if not response.tool_calls:
            # Plus de tools → réponse finale
            print(f"  ✅ Réponse finale prête")
            return response.content
        
        # 2. ACTION : Exécuter les tools demandés
        print(f"  🔄 Le LLM demande {len(response.tool_calls)} outil(s)")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"  ▶ Appel : {tool_name}({tool_args})")
            
            try:
                # 3. OBSERVATION : Résultat du tool
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    observation = str(result)
                else:
                    observation = f"Outil '{tool_name}' non trouvé"
                
                print(f"  ◀ Résultat : {observation}")
                
                # Ajouter l'observation comme ToolMessage
                messages.append(ToolMessage(
                    content=observation,
                    tool_call_id=tool_id
                ))
                
            except Exception as e:
                error_msg = f"Erreur : {str(e)}"
                print(f"  ❌ {error_msg}")
                messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_id
                ))
    
    return "Nombre maximum d'itérations atteint."


# ── 4. Tests de l'agent ───────────────────────────────────────────────────────
print("\n[ÉTAPE 4] Tests de l'agent ReAct")

test_queries = [
    "Calcule : (15 + 7) × 3, puis divise le résultat par 2",
    "Quelle heure est-il actuellement ?",
    "Donne-moi la météo à Casablanca",
    "Calcule l'IMC d'une personne de 75kg et 1.75m",
    "Calcule 2 à la puissance 10, puis ajoute 24"
]

for query in test_queries:
    result = run_react_agent(query)
    print(f"\n📌 RÉSULTAT FINAL : {result}")
    print("=" * 60)

print("\n✅ Agent avec tools personnalisés OK !")
