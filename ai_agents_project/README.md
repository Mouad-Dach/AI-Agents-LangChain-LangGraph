# 🤖 Activité Pratique N°3 : Architectures d'Agents IA avec LangChain & LangGraph

Ce projet explore la conception et l'implémentation d'agents intelligents en utilisant les frameworks **LangChain** et **LangGraph**. Il est structuré de manière pédagogique, allant des agents simples aux systèmes de RAG (Retrieval Augmented Generation) agentiques complexes.

---

## 🎯 Objectifs Pédagogiques
- Comprendre les fondements des agents autonomes et de la mémoire conversationnelle.
- Maîtriser l'utilisation d'outils (Tools) et de middlewares pour enrichir les capacités des agents.
- Apprendre l'orchestration de workflows complexes via des graphes d'états avec **LangGraph**.
- Implémenter des systèmes de RAG agentiques capables de raisonner sur des documents PDF.

---

## 📁 Structure du Projet

Le projet est divisé en trois parties progressives :

### 🟢 Partie 1 : Fondamentaux de LangChain
Introduction aux agents et à la gestion des outils.
- `01_simple_agent.py` : Premier contact avec un agent capable de raisonner.
- `02_agent_with_memory.py` : Implémentation de la persistance de l'historique.
- `03_agent_with_tools.py` : Création et intégration d'outils personnalisés.
- `04_predefined_tools.py` : Utilisation de DuckDuckGo, Tavily et PythonREPL.
- `05_middlewares.py` : Gestion avancée du flux et des logs (callbacks).

### 🔵 Partie 2 : RAG Agentique
Combinaison de la recherche documentaire et du raisonnement agentique.
- `agentic_rag.py` : Un agent capable de consulter une base de données vectorielle (ChromaDB) pour répondre à des questions complexes basées sur des documents.

### 🟣 Partie 3 : Orchestration avec LangGraph
Exploration du contrôle granulaire des agents via des graphes.
- `01_langgraph_basics.py` : Création d'un premier graphe d'états simple.
- `02_conditional_graph.py` : Introduction aux nœuds conditionnels (routage).
- `03_loop_graph.py` : Gestion des cycles et des itérations dans un graphe.
- `04_react_agent.py` : Recréation du pattern ReAct de manière explicite.
- `05_agentic_rag_langgraph.py` : Système RAG complet orchestré par LangGraph.

---

## 🛠️ Installation et Configuration

### 1. Prérequis
- Python 3.9+
- Un environnement virtuel recommandé (`venv` ou `conda`)

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Variables d'environnement
Créez un fichier `.env` à la racine du projet:
```env
OPENAI_API_KEY=votre_clef_ici
TAVILY_API_KEY=votre_clef_ici       # Optionnel : pour la recherche web
LANGCHAIN_API_KEY=votre_clef_ici    # Optionnel : pour LangSmith
LANGCHAIN_TRACING_V2=true           # Activé pour le monitoring
LANGCHAIN_PROJECT=AI-Agents-TP3
```

---

## 🚀 Guide d'Utilisation

1. **Ordre recommandé** : Suivez la numérotation des fichiers dans chaque dossier pour une progression logique.
2. **Gestion des documents** : Pour les parties 2 et 3, placez vos fichiers PDF dans le dossier `resources/`. L'agent indexera automatiquement ces documents lors de son exécution.
3. **Monitoring** : Si `LANGCHAIN_TRACING_V2` est activé, vous pouvez visualiser l'exécution de vos agents sur [LangSmith](https://smith.langchain.com/).

---

## 📚 Stack Technique
| Composant | Technologie |
|-----------|-------------|
| **LLM Orchestrator** | LangChain / LangGraph |
| **Modèles** | OpenAI GPT-4o / GPT-3.5-Turbo |
| **Vector DB** | ChromaDB |
| **Outils de recherche** | Tavily, DuckDuckGo |
| **Parsing PDF** | PyPDF, Unstructured |
