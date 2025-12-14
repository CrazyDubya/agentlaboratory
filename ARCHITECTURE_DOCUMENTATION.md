# Agent Laboratory: Comprehensive Architecture Documentation

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Agent System](#4-agent-system)
5. [Workflow & Data Flows](#5-workflow--data-flows)
6. [LLM Integration Layer](#6-llm-integration-layer)
7. [Tools & Command System](#7-tools--command-system)
8. [Web Interface](#8-web-interface)
9. [Configuration System](#9-configuration-system)
10. [Suggested Enhancements](#10-suggested-enhancements)
11. [Optimization Opportunities](#11-optimization-opportunities)
12. [File Reference Index](#12-file-reference-index)

---

## 1. Executive Summary

**Agent Laboratory** is an end-to-end autonomous research system powered by LLM agents that collaborate to conduct scientific research from literature review through publication. The system orchestrates 6 specialized agents across 7 research phases to automate the complete research lifecycle.

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Python Code** | ~4,800 lines |
| **Core Modules** | 11 Python files |
| **Agent Types** | 6 specialized agents |
| **Research Phases** | 7 distinct phases |
| **Supported LLMs** | 10+ models (OpenAI, Anthropic, Google, DeepSeek) |
| **Documentation Languages** | 16+ |

### High-Level Architecture

```
                                 +---------------------------+
                                 |     LaboratoryWorkflow    |
                                 |    (ai_lab_repo.py:19)    |
                                 +-------------+-------------+
                                               |
              +--------------------------------+--------------------------------+
              |                                |                                |
   +----------v----------+        +------------v-----------+        +-----------v----------+
   |   Agent Ensemble    |        |    Solver Engines      |        |   External Tools     |
   |   (agents.py)       |        | (mlesolver/papersolver)|        |   (tools.py)         |
   +----------+----------+        +------------+-----------+        +-----------+----------+
              |                                |                                |
   +----------v----------+        +------------v-----------+        +-----------v----------+
   |  - PhDStudentAgent  |        |  - MLESolver           |        |  - ArxivSearch       |
   |  - PostdocAgent     |        |  - PaperSolver         |        |  - HFDataSearch      |
   |  - ProfessorAgent   |        |  - Code Execution      |        |  - SemanticScholar   |
   |  - MLEngineerAgent  |        |  - LaTeX Compilation   |        |  - Code Execution    |
   |  - SWEngineerAgent  |        +------------------------+        +----------------------+
   |  - ReviewersAgent   |
   +---------------------+
              |
   +----------v----------+
   |   LLM Inference     |
   |  (inference.py)     |
   +---------------------+
```

---

## 2. System Architecture

### 2.1 Directory Structure

```
/home/user/agentlaboratory/
├── Core Application Files
│   ├── ai_lab_repo.py      # Main workflow orchestrator (891 lines)
│   ├── agents.py           # Agent class definitions (739 lines)
│   ├── inference.py        # LLM query interface (213 lines)
│   ├── mlesolver.py        # ML experiment solver (566 lines)
│   ├── papersolver.py      # Paper generation solver (579 lines)
│   ├── tools.py            # External tool integrations (325 lines)
│   ├── utils.py            # Utility functions (480 lines)
│   └── common_imports.py   # Centralized imports (113 lines)
│
├── Demo & Web Interface
│   ├── app.py              # Flask web application (298 lines)
│   ├── demo.py             # Full-featured demo (368 lines)
│   └── simple_demo.py      # Simplified demo (235 lines)
│
├── Configuration
│   └── experiment_configs/
│       ├── MATH_agentlab.yaml
│       └── MATH_agentrxiv.yaml
│
├── Web Templates
│   └── templates/
│       ├── base.html       # Base template
│       ├── index.html      # Paper library
│       ├── search.html     # Semantic search
│       ├── upload.html     # PDF upload
│       ├── view.html       # Paper viewer
│       └── demo.html       # Interactive demo
│
├── Documentation
│   ├── README.md
│   ├── DEMO_README.md
│   ├── DEMO_GUIDE.md
│   ├── COMMERCIAL_EVALUATION.md
│   ├── ENHANCEMENT_ROADMAP.md
│   └── readme/             # 16+ language translations
│
└── Assets
    └── media/
        ├── AgentLabLogo.png
        ├── AgentLab.png
        └── AgentLabWF.png
```

### 2.2 Component Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  app.py (Flask)  │  demo.py (CLI/Web)  │  simple_demo.py  │  YAML Config   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                     LaboratoryWorkflow (ai_lab_repo.py)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ Phase       │  │ Checkpoint   │  │ Human-in-   │  │ State           │   │
│  │ Management  │  │ & Recovery   │  │ Loop        │  │ Propagation     │   │
│  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   AGENT LAYER       │ │   SOLVER LAYER      │ │   TOOL LAYER        │
├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤
│ BaseAgent           │ │ MLESolver           │ │ ArxivSearch         │
│ ├─ PhDStudentAgent  │ │ ├─ Replace Command  │ │ HFDataSearch        │
│ ├─ PostdocAgent     │ │ ├─ Edit Command     │ │ SemanticScholarSearch│
│ ├─ ProfessorAgent   │ │ └─ Code Repair      │ │ execute_code()      │
│ ├─ MLEngineerAgent  │ │                     │ │ compile_latex()     │
│ ├─ SWEngineerAgent  │ │ PaperSolver         │ └─────────────────────┘
│ └─ ReviewersAgent   │ │ ├─ PaperReplace     │
└─────────────────────┘ │ ├─ PaperEdit        │
                        │ └─ Arxiv Commands   │
                        └─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LLM INFERENCE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                        query_model() (inference.py)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ OpenAI   │  │ Anthropic│  │ Google   │  │ DeepSeek │  │ HuggingFace    │
│  │ GPT/O1/O3│  │ Claude   │  │ Gemini   │  │ V3       │  │ Qwen           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 LaboratoryWorkflow (ai_lab_repo.py:19-574)

The central orchestrator managing the entire research pipeline.

**Key Attributes:**

| Attribute | Type | Description | Line |
|-----------|------|-------------|------|
| `research_topic` | str | Research problem statement | 26 |
| `phases` | list | Phase definitions with subtasks | 60-65 |
| `phase_status` | dict | Completion tracking per phase | 66-69 |
| `human_in_loop_flag` | dict | Per-phase human approval flags | 74-81 |
| `phd/postdoc/professor/...` | Agent | Agent instances | 93-99 |
| `notes` | list | Research guidance notes | 89 |

**Key Methods:**

| Method | Purpose | Lines |
|--------|---------|-------|
| `perform_research()` | Main execution loop | 139-206 |
| `literature_review()` | ArXiv paper discovery | 465-545 |
| `plan_formulation()` | Research strategy design | 414-463 |
| `data_preparation()` | Dataset loading code | 343-412 |
| `running_experiments()` | ML experiment execution | 310-341 |
| `results_interpretation()` | Result analysis | 274-308 |
| `report_writing()` | LaTeX paper generation | 240-272 |
| `report_refinement()` | Review and iteration | 207-238 |
| `save_state()` | Checkpoint persistence | 106-113 |
| `set_agent_attr()` | State propagation | 115-126 |
| `human_in_loop()` | User feedback collection | 547-574 |

### 3.2 Inference Module (inference.py)

Unified interface for all LLM API calls with cost tracking.

**Supported Models:**

| Provider | Models | Configuration |
|----------|--------|---------------|
| OpenAI | gpt-4o-mini, gpt-4o, o1-mini, o1, o3-mini | API key via env/param |
| Anthropic | claude-3-5-sonnet-latest | API key via env/param |
| Google | gemini-2.0-pro, gemini-1.5-pro | API key via env/param |
| DeepSeek | deepseek-chat (v3) | Custom base URL |

**Cost Tracking (Lines 12-33):**

```python
# Token costs per 1M tokens (as of Dec 2024)
COSTS = {
    "gpt-4o": ($2.50, $10.00),      # input/output
    "gpt-4o-mini": ($0.15, $0.60),
    "o3-mini": ($1.10, $4.40),
    "claude-3-5-sonnet": ($3.00, $12.00),
    "deepseek-chat": ($1.00, $5.00),
}
```

### 3.3 Utility Functions (utils.py)

| Function | Purpose | Lines |
|----------|---------|-------|
| `query_deepseekv3()` | DeepSeek API wrapper | 11-26 |
| `query_gpt4omini()` | GPT-4o-mini wrapper | 53-72 |
| `query_gpt4o()` | GPT-4o wrapper | 76-94 |
| `count_tokens()` | Token counting | 163-166 |
| `clip_tokens()` | Context window management | 197-231 |
| `extract_prompt()` | Command extraction from LLM output | 235-239 |
| `compile_latex()` | LaTeX to PDF compilation | 127-160 |
| `is_equiv()` | Mathematical equivalence checking | 258-295 |
| `process_results()` | Answer extraction/validation | 296-340 |

---

## 4. Agent System

### 4.1 Agent Class Hierarchy

```
BaseAgent (agents.py:204-296)
│
├── ReviewersAgent (lines 184-201) [Standalone - no inheritance from BaseAgent]
│
├── ProfessorAgent(BaseAgent) (lines 299-362)
│   └── Phases: report writing
│   └── Commands: DIALOGUE, LATEX
│
├── PostdocAgent(BaseAgent) (lines 365-438)
│   └── Phases: plan formulation, results interpretation
│   └── Commands: DIALOGUE, PLAN, INTERPRETATION
│
├── MLEngineerAgent(BaseAgent) (lines 441-503)
│   └── Phases: data preparation, running experiments
│   └── Commands: python, DIALOGUE, SEARCH_HF
│
├── SWEngineerAgent(BaseAgent) (lines 507-560)
│   └── Phases: data preparation
│   └── Commands: DIALOGUE, SUBMIT_CODE
│
└── PhDStudentAgent(BaseAgent) (lines 563-737)
    └── Phases: ALL (central coordinator)
    └── Commands: SUMMARY, FULL_TEXT, ADD_PAPER, DIALOGUE, SUBMIT_CODE, PLAN
```

### 4.2 BaseAgent Interface (agents.py:204-296)

**Constructor Parameters:**

```python
def __init__(self,
    model="gpt-4o-mini",     # LLM backbone
    notes=None,               # Task-specific instructions
    max_steps=100,            # Max iterations per phase
    phases=[],                # Active phases for this agent
    openai_api_key=None       # API credentials
)
```

**State Attributes:**

| Attribute | Purpose | Initial Value |
|-----------|---------|---------------|
| `plan` | Research strategy | `""` |
| `report` | Generated paper | `""` |
| `history` | Conversation with expiration | `[]` |
| `exp_results` | Experiment outputs | `""` |
| `dataset_code` | Data loading code | `""` |
| `results_code` | Experiment script | `""` |
| `lit_review_sum` | Literature summary | `""` |
| `interpretation` | Results analysis | `""` |
| `second_round` | Iteration flag | `False` |

**Abstract Methods (must override):**

| Method | Purpose |
|--------|---------|
| `context(phase)` | Phase-specific context string |
| `phase_prompt(phase)` | Instructions for current phase |
| `role_description()` | Agent persona description |
| `command_descriptions(phase)` | Available commands |

### 4.3 Agent Communication Patterns

**Dialogue-Based Interaction:**

```
Phase: data_preparation()

Step 1: SWEngineerAgent.inference()
        ↓ Output: DIALOGUE or SUBMIT_CODE

Step 2: Parse DIALOGUE content → feedback to MLEngineerAgent

Step 3: MLEngineerAgent.inference(feedback)
        ↓ Output: python code or DIALOGUE

Step 4: Execute code if present
        ↓ Output: results or error

Step 5: Loop until SUBMIT_CODE successful
```

**State Synchronization (ai_lab_repo.py:115-126):**

```python
def set_agent_attr(self, attr, obj):
    """Propagate state to all agents"""
    setattr(self.phd, attr, obj)
    setattr(self.postdoc, attr, obj)
    setattr(self.professor, attr, obj)
    setattr(self.ml_engineer, attr, obj)
    setattr(self.sw_engineer, attr, obj)
```

### 4.4 Reviewer Scoring System (agents.py:36-181)

The `get_score()` function implements NeurIPS-style paper review:

**Review Metrics:**

| Metric | Description | Weight |
|--------|-------------|--------|
| Summary | Paper summary | - |
| Strengths | Key contributions | - |
| Weaknesses | Areas for improvement | - |
| Originality | Novel contribution (1-4) | 0.1 |
| Quality | Technical soundness (1-4) | 0.2 |
| Clarity | Presentation quality (1-4) | 0.1 |
| Significance | Impact potential (1-4) | 0.1 |
| Soundness | Methodology rigor (1-4) | 0.2 |
| Presentation | Writing quality (1-4) | 0.1 |
| Contribution | Field advancement (1-4) | 0.1 |
| Overall | Aggregate score (1-10) | 0.1 |
| Confidence | Reviewer certainty (1-5) | - |
| Decision | Accept/Reject | - |

---

## 5. Workflow & Data Flows

### 5.1 Complete Research Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH WORKFLOW PHASES                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PHASE 1        │    │  PHASE 2        │    │  PHASE 3        │
│  Literature     │───▶│  Plan           │───▶│  Data           │
│  Review         │    │  Formulation    │    │  Preparation    │
│                 │    │                 │    │                 │
│  Agent: PhD     │    │  Agents:        │    │  Agents:        │
│  Output:        │    │  Postdoc + PhD  │    │  SWEng + MLEng  │
│  lit_review_sum │    │  Output: plan   │    │  Output:        │
│                 │    │                 │    │  dataset_code   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
       ┌───────────────────────────────────────────────┘
       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PHASE 4        │    │  PHASE 5        │    │  PHASE 6        │
│  Running        │───▶│  Results        │───▶│  Report         │
│  Experiments    │    │  Interpretation │    │  Writing        │
│                 │    │                 │    │                 │
│  Engine:        │    │  Agents:        │    │  Engine:        │
│  MLESolver      │    │  Postdoc + PhD  │    │  PaperSolver    │
│  Output:        │    │  Output:        │    │  Output:        │
│  results_code,  │    │  interpretation │    │  report (LaTeX) │
│  exp_results    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
       ┌───────────────────────────────────────────────┘
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 7: Report Refinement                                                  │
│                                                                              │
│  ┌───────────────┐         ┌──────────────────┐         ┌─────────────────┐│
│  │ ReviewersAgent │────────▶│ Human/LLM        │────────▶│ Decision        ││
│  │ (3 reviewers)  │         │ Decision Point   │         │                 ││
│  │ get_score()    │         │                  │         │ Accept → DONE   ││
│  └───────────────┘         └──────────────────┘         │ Reject → Redo   ││
│                                                          │ Phases 2-7      ││
│                                                          └─────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow Diagram

```
research_topic
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Literature Review                                                   │
│ PhDStudentAgent.inference() → ArxivSearch → lit_review_sum                  │
└─────────────────────────────────────────────────────────────────────────────┘
      │ lit_review_sum
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Plan Formulation                                                    │
│ PostdocAgent ↔ PhDStudentAgent → plan                                       │
└─────────────────────────────────────────────────────────────────────────────┘
      │ plan + lit_review_sum
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Data Preparation                                                    │
│ SWEngineerAgent ↔ MLEngineerAgent → dataset_code                            │
│ HFDataSearch.retrieve_ds() → dataset recommendations                        │
└─────────────────────────────────────────────────────────────────────────────┘
      │ plan + dataset_code
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Running Experiments                                                 │
│ MLESolver.initial_solve() → MLESolver.solve() → results_code + exp_results  │
│ execute_code() → execution logs & metrics                                   │
└─────────────────────────────────────────────────────────────────────────────┘
      │ plan + dataset_code + results_code + exp_results
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Results Interpretation                                              │
│ PostdocAgent ↔ PhDStudentAgent → interpretation                             │
└─────────────────────────────────────────────────────────────────────────────┘
      │ All above + interpretation
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 6: Report Writing                                                      │
│ PaperSolver.initial_solve() → gen_initial_report() → solve() → report      │
│ Sections: abstract, introduction, related work, methods, results, etc.      │
└─────────────────────────────────────────────────────────────────────────────┘
      │ plan + report
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 7: Report Refinement                                                   │
│ ReviewersAgent.inference(plan, report) → scores + feedback                  │
│                                                                              │
│ IF REJECT:                                                                   │
│   ├─ second_round = True                                                    │
│   ├─ Store: prev_report, prev_exp_results, prev_interpretation              │
│   └─ Restart from Phase 2 with enhanced context                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Checkpoint & Recovery System

**Checkpointing (ai_lab_repo.py:106-113):**

```python
def save_state(self, phase):
    """Pickle entire workflow state"""
    with open(f"state_saves/Paper{self.paper_index}.pkl", "wb") as f:
        pickle.dump(self, f)
```

**Checkpoint Triggers:**
- After every subtask completion (line 200)
- After successful report refinement (line 184)

**Recovery Usage:**
```bash
python ai_lab_repo.py --load-existing True \
                      --load-existing-path "state_saves/Paper1.pkl"
```

### 5.4 Human-in-the-Loop Integration

**Integration Points per Phase:**

| Phase | Method | User Prompt | Lines |
|-------|--------|-------------|-------|
| Literature Review | `human_in_loop()` | Approve lit review summary | 510-511 |
| Plan Formulation | `human_in_loop()` | Approve research plan | 436-437 |
| Data Preparation | `human_in_loop()` | Approve dataset code | 375-376 |
| Running Experiments | `human_in_loop()` | Approve experiment results | 332-333 |
| Results Interpretation | `human_in_loop()` | Approve interpretation | 293-294 |
| Report Writing | `human_in_loop()` | Approve final report | 264-265 |
| Report Refinement | Direct input | Accept or request improvements | 214-216 |

**Feedback Loop:**
1. Display phase output to user
2. Prompt: "Are you happy with the presented content? (Y/N)"
3. If "N": Collect improvement notes → append to `self.notes` → retry phase
4. If "Y": Proceed to next phase

---

## 6. LLM Integration Layer

### 6.1 Model Configuration

**Default Model (ai_lab_repo.py:13):**
```python
DEFAULT_LLM_BACKBONE = "o3-mini"
```

**Per-Phase Model Configuration:**
```python
agent_model_backbone = {
    "literature review": "gpt-4o-mini",
    "plan formulation": "gpt-4o",
    "data preparation": "o3-mini",
    "running experiments": "gpt-4o-mini",
    "results interpretation": "gpt-4o",
    "report writing": "gpt-4o",
    "report refinement": "gpt-4o-mini"
}
```

### 6.2 API Call Pattern (inference.py:35-211)

```python
def query_model(model_str, prompt, system_prompt,
                openai_api_key=None, gemini_api_key=None,
                anthropic_api_key=None, tries=5, timeout=5.0, temp=None):
    """
    Unified LLM query interface with:
    - Retry mechanism (default: 5 attempts)
    - Timeout between retries (default: 5 seconds)
    - Token counting and cost tracking
    - Multi-provider support
    """
```

### 6.3 Prompt Construction (agents.py:247-260)

```python
prompt = f"""
{self.role_description()}

{self.phase_prompt(phase)}

Context:
{context}

Previous History:
{history_str}

Phase Notes:
{phase_notes}

Feedback:
{feedback}
"""
```

### 6.4 Response Parsing

**JSON Extraction (agents.py:7-32):**
```python
def extract_json_between_markers(llm_output):
    # Pattern 1: ```json ... ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    # Fallback: Raw JSON object
    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)
```

**Command Extraction (utils.py:235-239):**
```python
def extract_prompt(text, word):
    """Extract content from ```COMMAND ... ``` blocks"""
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    return "\n".join(code_blocks).strip()
```

---

## 7. Tools & Command System

### 7.1 Solver Command Classes

**MLESolver Commands (mlesolver.py):**

| Command | Class | Format | Purpose | Lines |
|---------|-------|--------|---------|-------|
| REPLACE | `Replace` | \`\`\`REPLACE\n{code}\n\`\`\` | Full code replacement | 55-82 |
| EDIT | `Edit` | \`\`\`EDIT N M\n{code}\n\`\`\` | Line range modification | 86-139 |

**PaperSolver Commands (papersolver.py):**

| Command | Class | Format | Purpose | Lines |
|---------|-------|--------|---------|-------|
| SUMMARY | `ArxivSearch` | \`\`\`SUMMARY\n{query}\n\`\`\` | ArXiv paper search | 47-82 |
| FULL_TEXT | `ArxivSearch` | \`\`\`FULL_TEXT\n{id}\n\`\`\` | Paper retrieval | 47-82 |
| REPLACE | `PaperReplace` | \`\`\`REPLACE\n{latex}\n\`\`\` | Full LaTeX replacement | 91-118 |
| EDIT | `PaperEdit` | \`\`\`EDIT N M\n{latex}\n\`\`\` | LaTeX line editing | 122-174 |

### 7.2 External Tool Integrations

**ArxivSearch (tools.py:200-285):**
```python
class ArxivSearch:
    def find_papers_by_str(query, N=20):
        """Search ArXiv with retry logic (3 attempts)"""

    def retrieve_full_paper_text(paper_id):
        """Download PDF, extract text (max 50,000 chars)"""
```

**HFDataSearch (tools.py:21-176):**
```python
class HFDataSearch:
    def __init__(likes_threshold=3, downloads_threshold=50):
        """Load HuggingFace dataset index with TF-IDF vectorization"""

    def retrieve_ds(query, N=10, cos_weight=1.0):
        """Weighted search: cosine similarity + likes + downloads"""
```

**SemanticScholarSearch (tools.py:179-197):**
```python
class SemanticScholarSearch:
    def search_paper(query, min_citations=3, open_access=True):
        """Search Semantic Scholar for papers"""
```

### 7.3 Code Execution (tools.py:306-325)

```python
def execute_code(code, timeout=600):
    """
    Execute Python code in isolated process
    - Uses multiprocessing.Process
    - Enforces timeout (default: 10 minutes)
    - Blocks dangerous operations (exit(), certain datasets)
    - Returns stdout/stderr with error prefix on failure
    """
```

### 7.4 Error Handling & Repair

**Code Repair Mechanism (mlesolver.py:166-200):**
```python
def code_repair(code, error, cmd_type):
    """
    Attempt to fix broken code using LLM:
    - For 'replace': Temperature 0.8 (exploratory)
    - For 'edit': Temperature 0.2 (deterministic)
    - GLOBAL_REPAIR_ATTEMPTS = 2
    """
```

---

## 8. Web Interface

### 8.1 Flask Application (app.py)

**Configuration:**
```python
SECRET_KEY = 'your-secret-key'
UPLOAD_FOLDER = 'uploads/'
SQLALCHEMY_DATABASE_URI = 'sqlite:///papers.db'
```

**Routes:**

| Route | Method | Purpose | Lines |
|-------|--------|---------|-------|
| `/` | GET | Paper library homepage | 79-83 |
| `/upload` | GET/POST | PDF upload form | 85-113 |
| `/search` | GET | Semantic search interface | 115-152 |
| `/api/search` | GET | Search JSON API | 154-199 |
| `/view/<id>` | GET | Paper viewer | 205-209 |
| `/demo` | GET | Research demo page | 211-213 |

### 8.2 Database Model

```python
class Paper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    text = db.Column(db.Text, nullable=True)
```

### 8.3 Semantic Search

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Implementation (app.py:140-149):**
```python
1. Encode query: model.encode([query])
2. Encode all paper texts: model.encode(paper_texts)
3. Calculate cosine_similarity
4. Sort by similarity score
```

**Fallback Text Search (offline mode):**
```python
# Word frequency matching
matches = sum(text.count(word) for word in query.split())
score = min(matches * 0.1, 1.0)
```

### 8.4 Demo Modes

| Mode | File | Command | Features |
|------|------|---------|----------|
| Web | demo.py | `--mode web` | Flask UI, sample papers |
| CLI | demo.py | `--mode cli` | Interactive terminal |
| Full | demo.py | `--mode full` | Actual AI research |
| Simple | simple_demo.py | (default) | Lightweight web demo |

---

## 9. Configuration System

### 9.1 YAML Configuration Structure

```yaml
# Core Settings
copilot-mode: True|False
research-topic: "Your research prompt here"
language: "English"

# API Configuration
api-key: "OPENAI-API-KEY"
# OR deepseek-api-key: "DEEPSEEK-API-KEY"
llm-backend: "o3-mini"
lit-review-backend: "o3-mini"

# Research Parameters
num-papers-lit-review: 5
num-papers-to-write: 1
parallel-labs: False
mlesolver-max-steps: 3
papersolver-max-steps: 1

# Recovery
load-existing: False
lab-index: 1

# Output
compile-latex: False
except-if-fail: False

# Phase-Specific Notes
task-notes:
  plan-formulation:
    - "Note 1..."
  data-preparation:
    - "Note 1..."
  running-experiments:
    - "Note 1..."
```

### 9.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API access |
| `DEEPSEEK_API_KEY` | DeepSeek API access |
| `ANTHROPIC_API_KEY` | Anthropic API access |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer warnings |

---

## 10. Suggested Enhancements

### 10.1 Architecture Improvements

#### A. Agent Communication Protocol

**Current State:** Agents communicate via string-based dialogue parsing.

**Enhancement:** Implement structured message passing:

```python
@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: MessageType  # DIALOGUE, COMMAND, RESULT, ERROR
    content: dict
    timestamp: datetime
    correlation_id: str  # For tracking conversation threads

class MessageBus:
    def publish(self, message: AgentMessage): ...
    def subscribe(self, agent_id: str, handler: Callable): ...
```

**Benefits:**
- Type-safe communication
- Easier debugging/logging
- Support for async message handling
- Message history and replay capability

#### B. Plugin Architecture for Tools

**Enhancement:** Create pluggable tool interface:

```python
class ToolPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

    @abstractmethod
    def validate_input(self, **kwargs) -> bool: ...

class ToolRegistry:
    def register(self, plugin: ToolPlugin): ...
    def get(self, name: str) -> ToolPlugin: ...
    def list_available(self) -> List[str]: ...
```

**Benefits:**
- Easy addition of new tools (Google Scholar, IEEE, etc.)
- Consistent error handling
- Tool capability discovery for agents

#### C. Distributed Execution Support

**Enhancement:** Add parallel lab execution with proper coordination:

```python
class LabCluster:
    def __init__(self, num_workers: int, coordinator_url: str):
        self.workers = []
        self.coordinator = Coordinator(coordinator_url)

    async def run_parallel_research(self, topics: List[str]):
        """Distribute research across workers"""
        tasks = [self.submit_to_worker(t) for t in topics]
        return await asyncio.gather(*tasks)

    def aggregate_results(self, results: List[ResearchOutput]):
        """Combine findings from multiple labs"""
```

### 10.2 Performance Optimizations

#### A. Intelligent Context Management

**Current Issue:** Fixed context window with simple truncation.

**Enhancement:** Implement semantic context compression:

```python
class ContextManager:
    def __init__(self, max_tokens: int = 100000):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def compress_context(self, messages: List[str], query: str) -> str:
        """
        Keep only semantically relevant context:
        1. Embed all messages
        2. Score relevance to current query
        3. Keep top-K most relevant + recent N messages
        """
        embeddings = self.embedder.encode(messages)
        query_emb = self.embedder.encode([query])
        scores = cosine_similarity(query_emb, embeddings)[0]

        # Keep top 50% by relevance + last 5 messages
        relevant_idx = np.argsort(scores)[-len(messages)//2:]
        recent_idx = list(range(max(0, len(messages)-5), len(messages)))
        keep_idx = sorted(set(relevant_idx) | set(recent_idx))

        return "\n".join(messages[i] for i in keep_idx)
```

#### B. Response Caching Layer

**Enhancement:** Cache LLM responses for deterministic queries:

```python
class ResponseCache:
    def __init__(self, redis_url: str = None):
        self.cache = redis.Redis.from_url(redis_url) if redis_url else {}

    def get_cache_key(self, model: str, prompt: str, system: str) -> str:
        content = f"{model}:{system}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, model: str, prompt: str, system: str) -> Optional[str]:
        key = self.get_cache_key(model, prompt, system)
        return self.cache.get(key)

    def set(self, model: str, prompt: str, system: str, response: str, ttl: int = 3600):
        key = self.get_cache_key(model, prompt, system)
        self.cache.set(key, response, ex=ttl)
```

#### C. Batch API Calls

**Enhancement:** Batch multiple inference requests:

```python
async def batch_query_model(requests: List[QueryRequest]) -> List[str]:
    """
    Send multiple queries in parallel, respecting rate limits.

    Benefits:
    - Reduced latency for multi-agent phases
    - Better utilization of API concurrency limits
    """
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

    async def bounded_query(req):
        async with semaphore:
            return await query_model_async(**req)

    return await asyncio.gather(*[bounded_query(r) for r in requests])
```

### 10.3 Reliability Improvements

#### A. Enhanced Error Recovery

**Enhancement:** Implement circuit breaker pattern:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failures = 0
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
```

#### B. Structured Logging

**Enhancement:** Add comprehensive logging:

```python
import structlog

logger = structlog.get_logger()

class LoggedWorkflow(LaboratoryWorkflow):
    def literature_review(self):
        with logger.contextualize(
            phase="literature_review",
            paper_index=self.paper_index,
            model=self.phd.model
        ):
            logger.info("phase_started")
            try:
                result = super().literature_review()
                logger.info("phase_completed", papers_found=len(self.phd.lit_review))
                return result
            except Exception as e:
                logger.error("phase_failed", error=str(e))
                raise
```

#### C. Validation Pipeline

**Enhancement:** Add input/output validation:

```python
from pydantic import BaseModel, validator

class PlanOutput(BaseModel):
    title: str
    hypothesis: str
    methodology: List[str]
    expected_outcomes: List[str]

    @validator('methodology')
    def validate_methodology(cls, v):
        if len(v) < 3:
            raise ValueError("Plan must have at least 3 methodology steps")
        return v

class OutputValidator:
    VALIDATORS = {
        "plan formulation": PlanOutput,
        "results interpretation": InterpretationOutput,
        # ...
    }

    def validate(self, phase: str, output: str) -> bool:
        if phase in self.VALIDATORS:
            try:
                self.VALIDATORS[phase].parse_raw(output)
                return True
            except ValidationError:
                return False
        return True
```

### 10.4 Feature Additions

#### A. Multi-Modal Support

**Enhancement:** Add image/figure understanding:

```python
class MultiModalAgent(BaseAgent):
    def __init__(self, vision_model: str = "gpt-4-vision-preview"):
        super().__init__()
        self.vision_model = vision_model

    def analyze_figure(self, image_path: str, context: str) -> str:
        """Analyze generated figures for insights"""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        return query_model(
            self.vision_model,
            prompt=f"Analyze this research figure. Context: {context}",
            images=[{"type": "base64", "data": image_data}]
        )
```

#### B. Interactive Research Dashboard

**Enhancement:** Real-time monitoring UI:

```python
from flask_socketio import SocketIO

socketio = SocketIO(app)

class MonitoredWorkflow(LaboratoryWorkflow):
    def emit_status(self, phase: str, status: dict):
        socketio.emit('workflow_update', {
            'phase': phase,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })

    def literature_review(self):
        self.emit_status('literature_review', {'state': 'started'})
        result = super().literature_review()
        self.emit_status('literature_review', {
            'state': 'completed',
            'papers_found': len(self.phd.lit_review)
        })
        return result
```

#### C. Research Memory System

**Enhancement:** Long-term knowledge persistence:

```python
class ResearchMemory:
    def __init__(self, vector_store: str = "chroma"):
        self.db = chromadb.Client()
        self.collection = self.db.get_or_create_collection("research_memory")

    def store_finding(self, finding: str, metadata: dict):
        embedding = self.embed(finding)
        self.collection.add(
            embeddings=[embedding],
            documents=[finding],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def recall_relevant(self, query: str, k: int = 5) -> List[str]:
        results = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=k
        )
        return results['documents'][0]
```

---

## 11. Optimization Opportunities

### 11.1 Cost Optimization

| Optimization | Estimated Savings | Implementation Effort |
|--------------|-------------------|----------------------|
| Response caching | 20-40% | Low |
| Model selection per phase | 30-50% | Low |
| Context compression | 15-25% | Medium |
| Batch API calls | 10-20% (latency) | Medium |

### 11.2 Latency Optimization

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| Parallel agent calls | High | Use asyncio for independent dialogues |
| Pre-fetching | Medium | Speculatively load papers during planning |
| Connection pooling | Low | Reuse HTTP connections to APIs |

### 11.3 Quality Optimization

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| Ensemble reviews | High | Multiple reviewer passes with different prompts |
| Iterative refinement | Medium | Automatic re-run on low scores |
| Cross-validation | Medium | Verify findings with alternative approaches |

---

## 12. File Reference Index

### Core Files

| File | Lines | Primary Classes/Functions |
|------|-------|--------------------------|
| `ai_lab_repo.py` | 891 | `LaboratoryWorkflow`, `perform_research()` |
| `agents.py` | 739 | `BaseAgent`, `PhDStudentAgent`, `PostdocAgent`, etc. |
| `inference.py` | 213 | `query_model()`, `curr_cost_est()` |
| `mlesolver.py` | 566 | `MLESolver`, `Replace`, `Edit`, `code_repair()` |
| `papersolver.py` | 579 | `PaperSolver`, `PaperReplace`, `PaperEdit` |
| `tools.py` | 325 | `ArxivSearch`, `HFDataSearch`, `execute_code()` |
| `utils.py` | 480 | `compile_latex()`, `extract_prompt()`, `is_equiv()` |
| `common_imports.py` | 113 | Centralized library imports |

### Web Interface

| File | Lines | Primary Functions |
|------|-------|-------------------|
| `app.py` | 298 | Flask routes, Paper model, semantic search |
| `demo.py` | 368 | `AgentLabDemo` class, multi-mode demos |
| `simple_demo.py` | 235 | `run_web_demo()`, `create_sample_data()` |

### Templates

| File | Purpose |
|------|---------|
| `base.html` | Base template with navigation and styling |
| `index.html` | Paper library with statistics |
| `search.html` | Semantic search interface |
| `upload.html` | PDF upload form |
| `view.html` | Paper content viewer |
| `demo.html` | Interactive research demo |

### Configuration

| File | Purpose |
|------|---------|
| `experiment_configs/MATH_agentlab.yaml` | MATH benchmark configuration |
| `experiment_configs/MATH_agentrxiv.yaml` | AgentRxiv integration config |
| `requirements.txt` | Python dependencies (139 packages) |

---

## Appendix A: Quick Start Reference

### Running a Research Workflow

```bash
# Standard execution
python ai_lab_repo.py --yaml-location "experiment_configs/MATH_agentlab.yaml"

# With copilot mode
python ai_lab_repo.py --yaml-location "config.yaml" --copilot-mode "true"

# Resume from checkpoint
python ai_lab_repo.py --yaml-location "config.yaml" \
                      --load-existing True \
                      --load-existing-path "state_saves/Paper1.pkl"
```

### Running Demos

```bash
# Web demo
python demo.py --mode web

# CLI demo
python demo.py --mode cli --topic "AI research"

# Full AI workflow (requires API key)
python demo.py --mode full --quick

# Simple demo
python simple_demo.py
```

### API Key Setup

```bash
export OPENAI_API_KEY="sk-..."
# OR
export DEEPSEEK_API_KEY="..."
```

---

*Documentation generated: December 2024*
*Agent Laboratory Version: 1.0*
