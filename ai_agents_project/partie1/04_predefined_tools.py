"""
============================================================
Partie 1 - Fichier 04 : Outils Prédéfinis LangChain
============================================================
Objectif : Utiliser les tools prédéfinis :
           - DuckDuckGoSearchRun (recherche web gratuite)
           - TavilySearchResults (recherche IA avancée)
           - PythonREPLTool     (exécution de code Python)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

print("=" * 60)
print("04 - OUTILS PRÉDÉFINIS : DuckDuckGo / Tavily / PythonREPL")
print("=" * 60)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A : DuckDuckGo Search (GRATUIT - pas de clé API requise)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("A - DuckDuckGoSearchRun (Recherche Web Gratuite)")
print("=" * 60)

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.tools import DuckDuckGoSearchResults

    # Tool simple : retourne une chaîne
    ddg_search = DuckDuckGoSearchRun()

    # Tool enrichi : retourne titre + lien + snippet
    ddg_results = DuckDuckGoSearchResults(
        num_results=3,
        output_format="list"
    )

    print(f"\n✅ DuckDuckGo configuré")
    print(f"   Nom du tool : {ddg_search.name}")
    print(f"   Description : {ddg_search.description[:80]}...")

    # Test direct du tool
    print("\n[TEST] Recherche directe")
    query = "LangGraph framework AI agents 2024"
    print(f"  Query : {query}")
    result = ddg_search.invoke(query)
    print(f"  Résultat : {result[:300]}...")

    # Intégration dans un agent
    print("\n[AGENT] LLM + DuckDuckGo")
    tools_ddg = [ddg_search]
    llm_ddg = llm.bind_tools(tools=tools_ddg)
    tool_map_ddg = {"duckduckgo_search": ddg_search}

    def agent_with_ddg(question: str) -> str:
        messages = [
            SystemMessage(content="Tu es un assistant qui fait des recherches web. Réponds en français."),
            HumanMessage(content=question)
        ]
        response = llm_ddg.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                result = tool_map_ddg[tc["name"]].invoke(tc["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            final = llm_ddg.invoke(messages)
            return final.content
        return response.content

    result = agent_with_ddg("Quelles sont les dernières nouveautés de LangGraph en 2025 ?")
    print(f"\n📌 Réponse agent : {result[:400]}...")

except ImportError as e:
    print(f"⚠️ DuckDuckGo non disponible : {e}")
    print("   Installer avec : pip install duckduckgo-search")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B : Tavily Search (Recherche IA - clé API gratuite sur tavily.com)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("B - TavilySearchResults (Recherche IA Avancée)")
print("=" * 60)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY or TAVILY_API_KEY == "tvly-VOTRE_CLE_ICI":
    print("⚠️ TAVILY_API_KEY non configurée.")
    print("   → Obtenir une clé gratuite sur https://tavily.com")
    print("   → Ajouter dans .env : TAVILY_API_KEY=tvly-...")
else:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tavily_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",      # "basic" ou "advanced"
            include_answer=True,           # Inclure une réponse directe
            include_raw_content=False,
            include_images=False,
            api_key=TAVILY_API_KEY
        )

        print(f"✅ Tavily configuré")
        print(f"   Nom : {tavily_tool.name}")

        # Test direct
        print("\n[TEST] Recherche Tavily")
        results = tavily_tool.invoke("What is LangGraph and how does it differ from LangChain?")
        if isinstance(results, list):
            for i, r in enumerate(results[:2], 1):
                print(f"  Résultat {i}: {r.get('title', 'N/A')}")
                print(f"    URL: {r.get('url', 'N/A')}")
                print(f"    Contenu: {str(r.get('content', ''))[:150]}...")

        # Intégration agent
        print("\n[AGENT] LLM + Tavily")
        tools_tavily = [tavily_tool]
        llm_tavily = llm.bind_tools(tools=tools_tavily)

        messages = [
            SystemMessage(content="Tu es un assistant de recherche IA. Réponds en français."),
            HumanMessage(content="Quelles sont les meilleures pratiques pour construire des agents IA en 2025 ?")
        ]

        response = llm_tavily.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                result = tavily_tool.invoke(tc["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            final = llm_tavily.invoke(messages)
            print(f"\n📌 Réponse : {final.content[:400]}...")

    except Exception as e:
        print(f"❌ Erreur Tavily : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C : PythonREPLTool (Exécution de code Python)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("C - PythonREPLTool (Exécution de Code Python)")
print("=" * 60)

try:
    from langchain_experimental.tools import PythonREPLTool

    python_repl = PythonREPLTool()

    print(f"✅ PythonREPL configuré")
    print(f"   ⚠️  AVERTISSEMENT : Exécute du vrai code Python - utiliser avec précaution !")

    # Test direct : exécution de code
    print("\n[TEST 1] Code simple")
    result = python_repl.invoke("print('Hello from Python REPL!')")
    print(f"  Résultat : {result}")

    print("\n[TEST 2] Calcul mathématique")
    result = python_repl.invoke("""
import math
result = math.factorial(10)
print(f"10! = {result}")
fibonacci = [0, 1]
for i in range(8):
    fibonacci.append(fibonacci[-1] + fibonacci[-2])
print(f"Fibonacci(10): {fibonacci}")
""")
    print(f"  Résultat :\n{result}")

    print("\n[TEST 3] Statistiques avec pandas")
    result = python_repl.invoke("""
try:
    import statistics
    data = [23, 45, 12, 67, 34, 89, 56, 78, 43, 21]
    print(f"Données: {data}")
    print(f"Moyenne: {statistics.mean(data):.2f}")
    print(f"Médiane: {statistics.median(data):.2f}")
    print(f"Écart-type: {statistics.stdev(data):.2f}")
    print(f"Min: {min(data)}, Max: {max(data)}")
except Exception as e:
    print(f"Erreur: {e}")
""")
    print(f"  Résultat :\n{result}")

    # Intégration dans un agent
    print("\n[AGENT] LLM + PythonREPL")
    tools_repl = [python_repl]
    llm_repl = llm.bind_tools(tools=tools_repl)
    tool_map_repl = {python_repl.name: python_repl}

    def agent_python_repl(question: str) -> str:
        messages = [
            SystemMessage(content=(
                "Tu es un assistant Python expert. "
                "Utilise le Python REPL pour effectuer des calculs et analyses. "
                "Réponds en français."
            )),
            HumanMessage(content=question)
        ]

        for _ in range(5):  # Max 5 itérations
            response = llm_repl.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                tool_name = tc["name"]
                if tool_name in tool_map_repl:
                    result = tool_map_repl[tool_name].invoke(tc["args"])
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return "Limite d'itérations atteinte"

    questions_python = [
        "Écris du code Python pour calculer les 15 premiers nombres premiers et leur somme.",
        "Génère une liste de 10 nombres aléatoires entre 1 et 100, puis calcule leur moyenne et variance."
    ]

    for q in questions_python:
        print(f"\n  Question : {q[:60]}...")
        result = agent_python_repl(q)
        print(f"  Réponse : {result[:300]}...")

except ImportError as e:
    print(f"⚠️ PythonREPLTool non disponible : {e}")
    print("   Installer avec : pip install langchain-experimental")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D : Agent combinant tous les outils
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("D - Agent Multi-Tools (DuckDuckGo + PythonREPL)")
print("=" * 60)

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_experimental.tools import PythonREPLTool
    from langchain.tools import tool

    @tool
    def get_current_date() -> str:
        """Retourne la date et l'heure actuelles."""
        from datetime import datetime
        return datetime.now().strftime("%d/%m/%Y %H:%M")

    all_tools = [DuckDuckGoSearchRun(), PythonREPLTool(), get_current_date]
    llm_all = llm.bind_tools(tools=all_tools)
    tool_map_all = {t.name: t for t in all_tools}

    def multi_tool_agent(question: str) -> str:
        messages = [
            SystemMessage(content=(
                "Tu es un assistant polyvalent avec accès à : "
                "1) Recherche web (DuckDuckGo), "
                "2) Exécution Python (REPL), "
                "3) Date/heure. "
                "Utilise les outils appropriés. Réponds en français."
            )),
            HumanMessage(content=question)
        ]

        for i in range(8):
            response = llm_all.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                tool_name = tc["name"]
                print(f"    🔧 Tool utilisé : {tool_name}")
                if tool_name in tool_map_all:
                    result = tool_map_all[tool_name].invoke(tc["args"])
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return "Limite atteinte"

    print("\n[TEST] Question nécessitant Python")
    result = multi_tool_agent(
        "Écris un code Python qui calcule la suite de Fibonacci jusqu'à 50 et retourne les valeurs."
    )
    print(f"\n📌 Réponse : {result[:400]}...")

except Exception as e:
    print(f"⚠️ Erreur agent multi-tools : {e}")

print("\n✅ Outils prédéfinis OK !")
