# HTN Blue Agent for CybORG CAGE Challenge 2

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![CybORG](https://img.shields.io/badge/CybORG-CAGE%202-green)
![Planning](https://img.shields.io/badge/Planning-HTN%20(gtpyhop)-orange)

## Project Overview
This repository contains a **Hierarchical Task Network (HTN)** based Blue Agent designed for the **CybORG CAGE Challenge 2**. 

Unlike Reinforcement Learning agents that learn a policy through trial and error, this agent uses **domain-specific planning logic** to decompose high-level defense goals (e.g., "Secure Network") into primitive actions (e.g., "Restore Host 0"). It features a **Lazy Lookahead** mechanism to balance reactive speed with long-term strategic planning.

## Architecture & Interaction Flow

The system is split into three layers: the **Experiment Layer** (runners), the **Agent Layer** (HTN logic), and the **Translation Layer** (adapters).

```mermaid
graph TD
    subgraph "Experiment Layer"
        Runner[run_simplified_cage.py] -->|Starts Episode| Controller
        Runner -->|Inits| Env[SimplifiedCAGE Environment]
        Runner -->|Inits| Red[red_policy.py]
    end

    subgraph "Agent Layer (htn_agent.py)"
        Controller[run_cage_controller]
        Planner[GTPyhop Planner]
        Domain[HTN Domain Rules]
        
        Controller -- "1. Request Plan" --> Planner
        Planner -- "2. Uses" --> Domain
        Planner -- "3. Returns Sequence" --> Controller
    end

    subgraph "Translation Layer (minicage_adapter.py)"
        Adapter[Adapter Functions]
    end

    %% Interactions
    Env -- "Raw Obs (52-bit)" --> Controller
    Controller -- "Raw Obs" --> Adapter
    Adapter -- "Symbolic State" --> Planner
    
    Controller -- "Action String (e.g., 'restore_host')" --> Adapter
    Adapter -- "Action Int (e.g., 42)" --> Controller
    
    Controller -- "Step(BlueAction, RedAction)" --> Env
    Red -- "Red Action" --> Controller

## File Manifest
## 📂 File Manifest

| File Name | Role | Description |
| :--- | :--- | :--- |
| **`htn_agent.py`** | **The Brain** | Contains the core Hierarchical Task Network (HTN) logic. It defines the domain (`cage_htn`), the tasks (`secure_network`), and the `run_cage_controller` loop that manages planning execution and replanning intervals ($k$). |
| **`minicage_adapter.py`** | **The Translator** | A bridge between the numerical simulation and symbolic planning. It converts raw 52-bit CybORG observations into `Symbolic` Python objects (sets of compromised hosts, etc.) and maps action strings to integer IDs. |
| **`run_simplified_cage.py`** | **The Runner** | The main entry point for experiments. It initializes the environment and runs comparison tests between "Lazy Lookahead" (planning every $k$ steps) and "Full Lookahead" (planning every step) against different Red agents. |
| **`red_policy.py`** | **The Adversary** | Provides wrapper classes for Red agents (like `Meander` and `B_line`) to ensure their observation and action shapes match what the `SimplifiedCAGE` environment expects. |
| **`probe_blue.py`** | **The Diagnostic** | A utility script that steps through every possible Blue action (0-51) to verify how they alter the raw observation state vector. Useful for debugging the environment or adapter logic. |
| **`requirements.txt`** | **Dependencies** | Lists the required Python packages (including `gtpyhop` and the specific CAGE environment branch) needed to run the agent. |