"""
============================================================
Partie 3 - Fichier 03 : Graphe avec Boucle (Loop)
============================================================
Objectif : Implémenter des boucles dans LangGraph
           Reproduction exacte de la démo du cours

Graphe :
  START → file_verification
            ├─[notify]  → employee_notification → file_verification (boucle)
            └─[validate]→ file_validation → END
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()

print("=" * 60)
print("PARTIE 3 - GRAPHE AVEC BOUCLE (Vérification Dossier RH)")
print("=" * 60)


# ── 1) État partagé ───────────────────────────────────────────────────────────
class AdministrativeFileState(TypedDict):
    name:                str
    remaining_documents: int
    reminders:           int
    status:              Literal["incomplete", "complete", "validated"]


# ── 2) Nœud de vérification ───────────────────────────────────────────────────
def verify_employee_file(state: AdministrativeFileState) -> AdministrativeFileState:
    """Vérifie si le dossier est complet."""
    if state["remaining_documents"] > 0:
        state["status"] = "incomplete"
    else:
        state["status"] = "complete"
    
    print("*" * 50)
    print(f"Vérification dossier de {state['name']}, statut : {state['status']}")
    return state


# ── 3) Nœud de relance ────────────────────────────────────────────────────────
def notify_employee(state: AdministrativeFileState) -> AdministrativeFileState:
    """Envoie une relance et réduit le nombre de documents manquants."""
    state["reminders"] += 1
    if state["remaining_documents"] > 0:
        state["remaining_documents"] -= 1
    
    print("-" * 50)
    print(f"Relance {state['reminders']} de l'employé {state['name']}")
    print(f"Documents restants : {state['remaining_documents']}")
    return state


# ── 4) Nœud final ─────────────────────────────────────────────────────────────
def validate_file(state: AdministrativeFileState) -> AdministrativeFileState:
    """Valide le dossier complet."""
    state["status"] = "validated"
    print("=" * 40)
    print(f"Dossier de {state['name']} validé avec succès !")
    return state


# ── 5) Fonction de routage ────────────────────────────────────────────────────
def router(state: AdministrativeFileState) -> Literal["notify", "validate"]:
    """Retourne 'notify' si dossier incomplet, 'validate' sinon."""
    if state["status"] == "incomplete":
        return "notify"
    else:
        return "validate"


# ── 6) Construction du graphe ─────────────────────────────────────────────────
print("\n[Construction]")

workflow = StateGraph(AdministrativeFileState)

workflow.add_node("file_verification",    verify_employee_file)
workflow.add_node("employee_notification", notify_employee)
workflow.add_node("file_validation",      validate_file)

workflow.set_entry_point("file_verification")

workflow.add_conditional_edges(
    "file_verification",
    router,
    {
        "notify":   "employee_notification",
        "validate": "file_validation"
    }
)

# LA BOUCLE : notification → retour vérification
workflow.add_edge("employee_notification", "file_verification")
workflow.add_edge("file_validation", END)

graph = workflow.compile()
print("✅ Graphe avec boucle compilé")

try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


# ── 7) Test principal (reproduction exacte du cours) ──────────────────────────
print("\n[TEST] Mohamed avec 4 documents manquants")
print("=" * 60)

resp = graph.invoke({
    "name":                "Mohamed",
    "remaining_documents": 4,
    "reminders":           0,
    "status":              "incomplete"
})

print(f"\n📊 État final :")
print(f"   Nom          : {resp['name']}")
print(f"   Docs restants: {resp['remaining_documents']}")
print(f"   Relances     : {resp['reminders']}")
print(f"   Statut       : {resp['status']}")


# ── 8) Tests supplémentaires ──────────────────────────────────────────────────
print("\n\n[TEST] Cas avec dossier déjà complet")
print("─" * 50)

resp2 = graph.invoke({
    "name":                "Karim",
    "remaining_documents": 0,
    "reminders":           0,
    "status":              "incomplete"
})
print(f"\n📊 Résultat : {resp2['status']} (0 relances nécessaires)")


print("\n\n[TEST] Employée avec 2 documents manquants")
print("─" * 50)

resp3 = graph.invoke({
    "name":                "Amina",
    "remaining_documents": 2,
    "reminders":           0,
    "status":              "incomplete"
})
print(f"\n📊 Résultat Amina : {resp3['reminders']} relances, statut={resp3['status']}")


# ── BONUS : Graphe avec compteur de sécurité (évite boucle infinie) ───────────
print("\n\n[BONUS] Graphe avec limite de relances")
print("-" * 50)

class SafeFileState(TypedDict):
    name:                str
    remaining_documents: int
    reminders:           int
    max_reminders:       int
    status:              Literal["incomplete", "complete", "validated", "expired"]


def safe_verify(state: SafeFileState) -> SafeFileState:
    if state["reminders"] >= state["max_reminders"]:
        state["status"] = "expired"
    elif state["remaining_documents"] > 0:
        state["status"] = "incomplete"
    else:
        state["status"] = "complete"
    print(f"  🔍 {state['name']}: docs={state['remaining_documents']}, "
          f"relances={state['reminders']}/{state['max_reminders']}, statut={state['status']}")
    return state


def safe_notify(state: SafeFileState) -> SafeFileState:
    state["reminders"] += 1
    if state["remaining_documents"] > 0:
        state["remaining_documents"] -= 1
    print(f"  📧 Relance {state['reminders']} envoyée")
    return state


def safe_validate(state: SafeFileState) -> SafeFileState:
    state["status"] = "validated"
    print(f"  ✅ Dossier validé !")
    return state


def safe_expire(state: SafeFileState) -> SafeFileState:
    print(f"  ❌ Dossier expiré (max relances atteint)")
    return state


def safe_router(state: SafeFileState) -> str:
    if state["status"] == "expired":
        return "expire"
    elif state["status"] == "incomplete":
        return "notify"
    else:
        return "validate"


safe_wf = StateGraph(SafeFileState)
safe_wf.add_node("verify",   safe_verify)
safe_wf.add_node("notify",   safe_notify)
safe_wf.add_node("validate", safe_validate)
safe_wf.add_node("expire",   safe_expire)

safe_wf.set_entry_point("verify")
safe_wf.add_conditional_edges(
    "verify", safe_router,
    {"notify": "notify", "validate": "validate", "expire": "expire"}
)
safe_wf.add_edge("notify",   "verify")
safe_wf.add_edge("validate", END)
safe_wf.add_edge("expire",   END)

safe_graph = safe_wf.compile()

print("\nTest : 10 docs manquants, max 3 relances")
result_safe = safe_graph.invoke({
    "name":                "Omar",
    "remaining_documents": 10,
    "reminders":           0,
    "max_reminders":       3,
    "status":              "incomplete"
})
print(f"  → Statut final : {result_safe['status']}")
print(f"  → Relances effectuées : {result_safe['reminders']}")

print("\n✅ Graphes avec boucle OK !")
