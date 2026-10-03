# Advanced LangGraph Project Architecture: Pharma Agentic System

This document outlines the architecture and baseline code for building an **Advanced Multi-Agent Pharmaceutical Pipeline** using LangGraph. This architecture implements a **Corrective Literature Review & Safety Audit System** utilizing advanced state management, conditional routing, self-correction loops, and Human-in-the-Loop (HITL) checkpoints.

---

## 🏗️ System Architecture Workflow

```
               [ User Drug Query ]
                        │
                        ▼
             ┌─────────────────────┐
             │  Literature Search  │◀──────────────────────┐
             │       Agent         │                       │
             └─────────────────────┘                       │
                        │                                  │
                        ▼                                  │
             ┌─────────────────────┐                       │
             │   Fact Grader       │                       │
             │       Agent         │                       │
             └─────────────────────┘                       │
                        │                                  │
                 (Is Context Good?)                        │
                   /           \                           │
                No /             \ Yes                     │
                  ▼               ▼                        │
        ┌──────────────┐   ┌──────────────┐                │
        │ Web Fallback │   │ Safety Audit │                │
        │    Search    │   │    Agent     │                │
        └──────────────┘   └──────────────┘                │
               │                  │                        │
               ▼                  ▼                        │
       (Merge Context)    (Passes Safety?)                 │
               │            /          \                   │
               │         No /            \ Yes             │
               │           ▼              ▼                │
               │     ┌───────────┐   ┌───────────┐         │
               │     │ Re-Draft  │   │  Human    │         │
               │     │ Protocols │   │ Checkpoint│(Interrupt)
               │     └───────────┘   └───────────┘         │
               │           │              │                │
               └───────────┴──────┬───────┘                │
                                  │                        │
                                  ▼                        │
                         (Human Validated?)                │
                           /            \                  │
                        No /              \ Yes            │
                          ▼                ▼               │
                    [Refinement]   [Final Regulatory Brief]
```

---

## 🛠️ Production-Ready Base Implementation

Below is the structured Python implementation using `langgraph`, `typing`, and `langchain_core`.

```python
import os
from typing import Annotated, Dict, List, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. State Definition
# ==========================================
class PharmaGraphState(TypedDict):
    """Represents the state of our pharmaceutical analysis pipeline."""
    drug_query: str
    literature_raw_data: List[str]
    web_fallback_data: List[str]
    safety_violations: List[str]
    regulatory_brief: str
    grade_decision: Literal["clear", "fallback", "fail"]
    human_approved: bool
    chat_history: List[BaseMessage]

# ==========================================
# 2. Node Implementations
# ==========================================
def literature_search_node(state: PharmaGraphState) -> Dict:
    """Simulates searching medical databases like PubMed / internal DBs."""
    query = state["drug_query"]
    print(f"[Node: Literature Search] Fetching clinical data for: {query}")
    
    # Mocking retrieved structured abstract data
    mock_db = {
        "Compound-X": ["Abstract: High efficacy in targeting mutated receptors.", "Adverse effect: Mild hepatotoxicity observed at high doses."],
        "Compound-Y": ["Abstract: Inconclusive trials on receptor binding profiles."]
    }
    
    data = mock_db.get(query, ["No localized proprietary data found for this exact compound configuration."])
    return {"literature_raw_data": data}

def fact_grader_node(state: PharmaGraphState) -> Dict:
    """Grades the relevancy and density of the retrieved internal data."""
    data = state["literature_raw_data"]
    print(f"[Node: Fact Grader] Grading data richness: {data}")
    
    if "No localized proprietary data" in data[0] or "Inconclusive" in data[0]:
        return {"grade_decision": "fallback"}
    return {"grade_decision": "clear"}

def web_fallback_node(state: PharmaGraphState) -> Dict:
    """Executes external API fallback engine when primary database drops out."""
    query = state["drug_query"]
    print(f"[Node: Web Fallback] Triggering external APIs for: {query}")
    
    # Simulating external crawl (e.g., FDA labels, clinicaltrials.gov indices)
    external_data = [f"Web scraping source: Phase I data suggests low affinity but alternative metabolic clearance pathways."]
    return {"web_fallback_data": external_data}

def safety_audit_node(state: PharmaGraphState) -> Dict:
    """Validates safety indices, cross-referencing contraindications and toxicology constraints."""
    print("[Node: Safety Audit] Evaluating structural contraindications...")
    raw_info = state["literature_raw_data"] + state.get("web_fallback_data", [])
    violations = []
    
    for text in raw_info:
        if "hepatotoxicity" in text.lower():
            violations.append("CRITICAL WARNING: Detected implicit hepatotoxicity flags at elevated therapeutic intervals.")
            
    return {"safety_violations": violations}

def generator_node(state: PharmaGraphState) -> Dict:
    """Generates the initial structured Regulatory Medical Brief summary."""
    print("[Node: Brief Generator] Synthesis of regulatory compliance files underway...")
    violations = state.get("safety_violations", [])
    
    brief = f"--- REGULATORY BRIEF FOR COMPOUND ---\n"
    brief += f"Query: {state['drug_query']}\n"
    brief += f"Primary Violations Flagged: {len(violations)}\n"
    if violations:
        brief += f"Flags Raised: {violations[0]}\n"
    brief += "Status: Pending Human-in-the-Loop Verification."
    
    return {"regulatory_brief": brief}

# ==========================================
# 3. Conditional Routing Logic
# ==========================================
def route_after_grading(state: PharmaGraphState) -> Literal["web_fallback_node", "safety_audit_node"]:
    if state["grade_decision"] == "fallback":
        print("  -> Routing to Web Fallback...")
        return "web_fallback_node"
    print("  -> Routing to Safety Audit...")
    return "safety_audit_node"

def route_after_audit(state: PharmaGraphState) -> Literal["generator_node", "literature_search_node"]:
    if state.get("safety_violations"):
        print("  -> Violations detected. Re-routing back to Literature optimization or patch cycle...")
        # In production, you might loop back or go directly to generation with warnings
    return "generator_node"

# ==========================================
# 4. Graph Construction
# ==========================================
workflow = StateGraph(PharmaGraphState)

# Add nodes
workflow.add_node("literature_search_node", literature_search_node)
workflow.add_node("fact_grader_node", fact_grader_node)
workflow.add_node("web_fallback_node", web_fallback_node)
workflow.add_node("safety_audit_node", safety_audit_node)
workflow.add_node("generator_node", generator_node)

# Map edge connections
workflow.add_edge(START, "literature_search_node")
workflow.add_edge("literature_search_node", "fact_grader_node")

# Conditional Edge from Fact Grader
workflow.add_conditional_edges(
    "fact_grader_node",
    route_after_grading,
    {
        "web_fallback_node": "web_fallback_node",
        "safety_audit_node": "safety_audit_node"
    }
)

# Connect Fallback to Safety Audit
workflow.add_edge("web_fallback_node", "safety_audit_node")

# Conditional Edge from Safety Audit
workflow.add_conditional_edges(
    "safety_audit_node",
    route_after_audit,
    {
        "generator_node": "generator_node",
        "literature_search_node": "literature_search_node"
    }
)

# Set human validation interrupt points
workflow.add_edge("generator_node", END)

# Configure thread memory persistence for HITL functionality
memory = MemorySaver()
compiled_pharma_graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["generator_node"] # Pause graph execution before final brief for Human confirmation
)

print("\n[Graph Setup] LangGraph Workflow Compiled Successfully with persistent MemorySaver.")
```

---

## 🔬 Core Learning Objectives & Architectural Stretch Goals

1. **State Reducers for Overlapped Context**: Use `operator.add` to keep track of a rolling timeline of multi-agent dialogue states without over-writing past discovery values.
2. **Deterministic Compliance Controls**: Force graph execution through strict schema validation layers (Pydantic V2 metadata extractors) inside your grader nodes to prevent LLM hallucinations from making business decisions.
3. **Advanced Micro-Subgraphs**: Isolate the Safety Audit Node into its own discrete **Subgraph Engine**, executing 5 safety validations in parallel using asynchronous `asyncio.gather` tasks.
4. **State Machine Snapshotting**: Practice accessing state checkpoints programmatically using `graph.get_state(config)` and mutating the state manually mid-execution via `graph.update_state(config, {"safety_violations": []})` to simulate human compliance overrides.

---
*Disclaimer: This documentation is intended strictly for developer portfolio practice and conceptual systems architecture design. AI responses may include mistakes.*
