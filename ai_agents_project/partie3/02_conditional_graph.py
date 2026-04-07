"""
============================================================
Partie 3 - Fichier 02 : Graphe Conditionnel LangGraph
============================================================
Objectif : Implémenter des branches conditionnelles
           Reproduction exacte de la démo du cours

Graphe :
  START → analyze_employee
            ├─[standard_hr]→ standard_hr_process → END
            └─[forced_hr]  → forced_HR_validation → END
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()

print("=" * 60)
print("PARTIE 3 - GRAPHE CONDITIONNEL (Exemple RH)")
print("=" * 60)


# ── 1) Définition de l'état partagé ──────────────────────────────────────────
class EmployeeState(TypedDict):
    name:     str
    age:      int
    salary:   float
    decision: Literal["standard_hr", "forced_hr"]
    result:   str


# ── 2) Nœud de vérification ───────────────────────────────────────────────────
def analyze_employee(state: EmployeeState) -> EmployeeState:
    """Analyse le profil de l'employé et détermine le parcours RH."""
    age    = state["age"]
    salary = state["salary"]
    
    if age <= 30 and salary <= 40000:
        state["decision"] = "standard_hr"
    else:
        state["decision"] = "forced_hr"
    
    print(f"  📊 Analyse : {state['name']} | âge={age} | salaire={salary}")
    print(f"  📋 Décision : {state['decision']}")
    return state


# ── 3) Chemin 1 : Standard ────────────────────────────────────────────────────
def standard_hr_process(state: EmployeeState) -> EmployeeState:
    """Traitement RH standard."""
    state["result"] = (
        f"Demande de {state['name']} envoyée au traitement RH standard"
    )
    print(f"  ✅ {state['result']}")
    return state


# ── 4) Chemin 2 : Forcé ───────────────────────────────────────────────────────
def forced_HR_validation(state: EmployeeState) -> EmployeeState:
    """Validation RH forcée pour profils particuliers."""
    state["result"] = (
        f"Demande de {state['name']} envoyée à la validation RH forcée"
    )
    print(f"  ⚠️  {state['result']}")
    return state


# ── 5) Fonction de routage conditionnel ──────────────────────────────────────
def router(state: EmployeeState) -> str:
    """Retourne la clé de routage selon la décision."""
    return state["decision"]


# ── 6) Construction du graphe ─────────────────────────────────────────────────
print("\n[Construction du graphe]")

wf = StateGraph(EmployeeState)

wf.add_node("analyze",  analyze_employee)
wf.add_node("standard", standard_hr_process)
wf.add_node("forced",   forced_HR_validation)

wf.add_edge(START, "analyze")

wf.add_conditional_edges(
    "analyze",
    router,
    {
        "standard_hr": "standard",
        "forced_hr":   "forced"
    }
)

wf.add_edge("standard", END)
wf.add_edge("forced",   END)

graph = wf.compile()
print("✅ Graphe conditionnel compilé")

# Visualisation Mermaid (si IPython disponible)
try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
    print("✅ Graphe visualisé (Jupyter/IPython)")
except Exception:
    print("ℹ️  Pour visualiser : exécuter dans Jupyter Notebook")


# ── 7) Tests ──────────────────────────────────────────────────────────────────
print("\n[Tests]")

test_employees = [
    # (name, age, salary) → décision attendue
    {"name": "Mohamed", "age": 23, "salary": 20000,
     "decision": "standard_hr", "result": ""},   # → standard_hr
    {"name": "Yassine", "age": 55, "salary": 65000,
     "decision": "standard_hr", "result": ""},   # → forced_hr
    {"name": "Fatima",  "age": 28, "salary": 38000,
     "decision": "standard_hr", "result": ""},   # → standard_hr (28≤30, 38000≤40000)
    {"name": "Ahmed",   "age": 25, "salary": 50000,
     "decision": "standard_hr", "result": ""},   # → forced_hr (salary > 40000)
    {"name": "Sara",    "age": 35, "salary": 30000,
     "decision": "standard_hr", "result": ""},   # → forced_hr (age > 30)
]

print("\n" + "─" * 55)
for emp in test_employees:
    print(f"\n👤 Employé : {emp['name']} (âge={emp['age']}, salaire={emp['salary']})")
    resp = graph.invoke(emp)
    print(f"  Résultat : {resp['result']}")
    print("─" * 55)


# ── Exemple avancé : Graphe RH multi-niveaux ──────────────────────────────────
print("\n[BONUS] Graphe conditionnel avancé : 3 chemins")
print("-" * 50)

class SalaryRouteState(TypedDict):
    employee:  str
    salary:    float
    category:  Literal["junior", "senior", "executive"]
    bonus:     float
    message:   str

def classify_employee(state: SalaryRouteState) -> SalaryRouteState:
    s = state["salary"]
    if s < 25000:
        state["category"] = "junior"
    elif s < 70000:
        state["category"] = "senior"
    else:
        state["category"] = "executive"
    return state

def process_junior(state: SalaryRouteState) -> SalaryRouteState:
    state["bonus"]   = state["salary"] * 0.05
    state["message"] = f"Bonus junior 5% = {state['bonus']:.0f} MAD"
    return state

def process_senior(state: SalaryRouteState) -> SalaryRouteState:
    state["bonus"]   = state["salary"] * 0.10
    state["message"] = f"Bonus senior 10% = {state['bonus']:.0f} MAD"
    return state

def process_executive(state: SalaryRouteState) -> SalaryRouteState:
    state["bonus"]   = state["salary"] * 0.20
    state["message"] = f"Bonus executive 20% = {state['bonus']:.0f} MAD"
    return state

def salary_router(state: SalaryRouteState) -> str:
    return state["category"]

wf_adv = StateGraph(SalaryRouteState)
wf_adv.add_node("classify",  classify_employee)
wf_adv.add_node("junior",    process_junior)
wf_adv.add_node("senior",    process_senior)
wf_adv.add_node("executive", process_executive)

wf_adv.add_edge(START, "classify")
wf_adv.add_conditional_edges(
    "classify", salary_router,
    {"junior": "junior", "senior": "senior", "executive": "executive"}
)
for node in ["junior", "senior", "executive"]:
    wf_adv.add_edge(node, END)

graph_adv = wf_adv.compile()

employees = [
    {"employee": "Ali",    "salary": 18000, "category": "junior",  "bonus": 0.0, "message": ""},
    {"employee": "Hassan", "salary": 45000, "category": "junior",  "bonus": 0.0, "message": ""},
    {"employee": "Nadia",  "salary": 95000, "category": "junior",  "bonus": 0.0, "message": ""},
]

print("\nBonus par catégorie :")
for emp_data in employees:
    resp = graph_adv.invoke(emp_data)
    print(f"  {resp['employee']:8s} (salaire={resp['salary']:6.0f}) → {resp['message']}")

print("\n✅ Graphes conditionnels OK !")
