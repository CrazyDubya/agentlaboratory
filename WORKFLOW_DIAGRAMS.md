# Agent Laboratory: Workflow Diagrams & Quick Reference

## System Overview Diagram

```
+============================================================================+
|                           AGENT LABORATORY                                  |
|                    Autonomous Research System                               |
+============================================================================+

                          +-------------------+
                          |   User/Config     |
                          |   (YAML + CLI)    |
                          +--------+----------+
                                   |
                                   v
+============================================================================+
|                        LaboratoryWorkflow                                   |
|                       (ai_lab_repo.py:19)                                  |
|                                                                             |
|  +------------------+  +------------------+  +-------------------+          |
|  | Phase Manager    |  | State Manager   |  | Checkpoint System |          |
|  | perform_research |  | set_agent_attr  |  | save_state/load   |          |
|  +------------------+  +------------------+  +-------------------+          |
+============================================================================+
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
+----------------+      +------------------+      +------------------+
|   AGENTS       |      |    SOLVERS       |      |     TOOLS        |
+----------------+      +------------------+      +------------------+
| PhDStudent     |      | MLESolver        |      | ArxivSearch      |
| Postdoc        |      |  - Replace       |      | HFDataSearch     |
| Professor      |      |  - Edit          |      | SemanticScholar  |
| MLEngineer     |      |  - CodeRepair    |      | execute_code()   |
| SWEngineer     |      +------------------+      | compile_latex()  |
| Reviewers      |      | PaperSolver      |      +------------------+
+----------------+      |  - PaperReplace  |
         |              |  - PaperEdit     |
         |              |  - ArxivSearch   |
         |              +------------------+
         |                         |
         +------------+------------+
                      |
                      v
          +------------------------+
          |   INFERENCE LAYER      |
          |   (inference.py)       |
          +------------------------+
          | OpenAI  | Anthropic    |
          | Google  | DeepSeek     |
          +------------------------+
```

---

## Research Phase Flow

```
+===========================================================================+
|                      RESEARCH WORKFLOW PHASES                              |
+===========================================================================+

  START
    |
    v
+---------------------------+
|   PHASE 1: Literature     |
|   Review                  |
|   [PhDStudentAgent]       |
|                           |
|   Commands:               |
|   - SUMMARY (arXiv)       |
|   - FULL_TEXT (PDF)       |
|   - ADD_PAPER             |
|                           |
|   Output: lit_review_sum  |
+-------------+-------------+
              |
              v
+---------------------------+
|   PHASE 2: Plan           |
|   Formulation             |
|   [Postdoc <-> PhD]       |
|                           |
|   Commands:               |
|   - DIALOGUE              |
|   - PLAN                  |
|                           |
|   Output: plan            |
+-------------+-------------+
              |
              v
+---------------------------+
|   PHASE 3: Data           |
|   Preparation             |
|   [SWEng <-> MLEng]       |
|                           |
|   Commands:               |
|   - DIALOGUE              |
|   - python                |
|   - SEARCH_HF             |
|   - SUBMIT_CODE           |
|                           |
|   Output: dataset_code    |
+-------------+-------------+
              |
              v
+---------------------------+
|   PHASE 4: Running        |
|   Experiments             |
|   [MLESolver]             |
|                           |
|   Commands:               |
|   - REPLACE               |
|   - EDIT N M              |
|   + code_repair()         |
|                           |
|   Outputs:                |
|   - results_code          |
|   - exp_results           |
+-------------+-------------+
              |
              v
+---------------------------+
|   PHASE 5: Results        |
|   Interpretation          |
|   [Postdoc <-> PhD]       |
|                           |
|   Commands:               |
|   - DIALOGUE              |
|   - INTERPRETATION        |
|                           |
|   Output: interpretation  |
+-------------+-------------+
              |
              v
+---------------------------+
|   PHASE 6: Report         |
|   Writing                 |
|   [PaperSolver]           |
|                           |
|   Sections Generated:     |
|   - Abstract              |
|   - Introduction          |
|   - Related Work          |
|   - Background            |
|   - Methods               |
|   - Experimental Setup    |
|   - Results               |
|   - Discussion            |
|                           |
|   Output: report (LaTeX)  |
+-------------+-------------+
              |
              v
+---------------------------+
|   PHASE 7: Report         |
|   Refinement              |
|   [ReviewersAgent]        |
|                           |
|   Review Criteria:        |
|   - Originality (1-4)     |
|   - Quality (1-4)         |
|   - Clarity (1-4)         |
|   - Significance (1-4)    |
|   - Overall (1-10)        |
|                           |
|   Decision:               |
+------+------------+-------+
       |            |
   ACCEPT        REJECT
       |            |
       v            v
    +----+    +-------------+
    |DONE|    | second_round|
    +----+    | = True      |
              |             |
              | Reset:      |
              | Phases 2-7  |
              +------+------+
                     |
                     | (Loop back to Phase 2)
                     +-------------------------->
```

---

## Agent Hierarchy

```
+============================================================================+
|                          AGENT HIERARCHY                                    |
+============================================================================+

                          +------------------+
                          |    BaseAgent     |
                          |  (agents.py:204) |
                          +--------+---------+
                                   |
       +------------+--------------+---------------+--------------+
       |            |              |               |              |
       v            v              v               v              v
+-------------+ +-----------+ +-----------+ +-------------+ +------------+
|  Professor  | |  Postdoc  | | MLEngineer| | SWEngineer  | | PhDStudent |
| (line 299)  | | (line 365)| | (line 441)| | (line 507)  | | (line 563) |
+------+------+ +-----+-----+ +-----+-----+ +------+------+ +-----+------+
       |              |             |              |              |
       |              |             |              |              |
   Phase:         Phases:       Phases:        Phase:        All Phases
   report      plan-form,    data-prep,     data-prep      (Coordinator)
   writing     results-     running-exp
               interpret


+------------------+
| ReviewersAgent   |  (Standalone - no inheritance)
| (line 184)       |
+------------------+
```

---

## Data Flow Through Phases

```
+============================================================================+
|                          DATA FLOW DIAGRAM                                  |
+============================================================================+

research_topic ─────────────────────────────────────────────────────────────>
                                                                             |
Phase 1: Literature Review                                                   |
+-----------------------------------------------------------------------+   |
|  PhDStudent.inference()                                                |   |
|       │                                                                |   |
|       ├──> ArxivSearch.find_papers_by_str(query)                      |   |
|       │         │                                                      |   |
|       │         └──> papers[] ─────────────────────────────────────>  |   |
|       │                                                                |   |
|       ├──> ArxivSearch.retrieve_full_paper_text(id)                   |   |
|       │         │                                                      |   |
|       │         └──> full_text[] ──────────────────────────────────>  |   |
|       │                                                                |   |
|       └──> PhDStudent.format_review()                                 |   |
|                 │                                                      |   |
|                 └──> lit_review_sum ──────────────────────────────────+─> |
+-----------------------------------------------------------------------+   |
                                                                             |
Phase 2: Plan Formulation                                                    |
+-----------------------------------------------------------------------+   |
|  lit_review_sum + research_topic                                       |   |
|       │                                                                |   |
|       ├──> Postdoc.inference() ──> DIALOGUE ─┐                        |   |
|       │                                       │                        |   |
|       └──> PhD.inference() <── DIALOGUE ──────┘                        |   |
|                 │                                                      |   |
|                 └──> plan ────────────────────────────────────────────+─> |
+-----------------------------------------------------------------------+   |
                                                                             |
Phase 3: Data Preparation                                                    |
+-----------------------------------------------------------------------+   |
|  lit_review_sum + plan                                                 |   |
|       │                                                                |   |
|       ├──> SWEngineer.inference() ──> feedback ─┐                     |   |
|       │                                          │                     |   |
|       ├──> MLEngineer.inference() <─────────────┘                     |   |
|       │         │                                                      |   |
|       │         ├──> HFDataSearch.retrieve_ds()                       |   |
|       │         │                                                      |   |
|       │         └──> python code                                       |   |
|       │                   │                                            |   |
|       │                   └──> execute_code()                          |   |
|       │                             │                                  |   |
|       └──> SUBMIT_CODE ─────────────┘                                  |   |
|                 │                                                      |   |
|                 └──> dataset_code ────────────────────────────────────+─> |
+-----------------------------------------------------------------------+   |
                                                                             |
Phase 4: Running Experiments                                                 |
+-----------------------------------------------------------------------+   |
|  plan + dataset_code                                                   |   |
|       │                                                                |   |
|       ├──> MLESolver.initial_solve()                                  |   |
|       │         │                                                      |   |
|       │         └──> gen_initial_code() ──> baseline                  |   |
|       │                                                                |   |
|       └──> MLESolver.solve() x N steps                                |   |
|                 │                                                      |   |
|                 ├──> REPLACE/EDIT commands                             |   |
|                 │         │                                            |   |
|                 │         └──> execute_code()                          |   |
|                 │                   │                                  |   |
|                 │                   ├──> success ──> get_score()      |   |
|                 │                   │                                  |   |
|                 │                   └──> error ──> code_repair()      |   |
|                 │                                                      |   |
|                 └──> best_codes[0]                                     |   |
|                           │                                            |   |
|                           ├──> results_code ──────────────────────────+─> |
|                           └──> exp_results ───────────────────────────+─> |
+-----------------------------------------------------------------------+   |
                                                                             |
Phase 5: Results Interpretation                                              |
+-----------------------------------------------------------------------+   |
|  plan + dataset_code + results_code + exp_results                      |   |
|       │                                                                |   |
|       ├──> Postdoc.inference() ──> analysis ─┐                        |   |
|       │                                       │                        |   |
|       └──> PhD.inference() <─────────────────┘                        |   |
|                 │                                                      |   |
|                 └──> interpretation ──────────────────────────────────+─> |
+-----------------------------------------------------------------------+   |
                                                                             |
Phase 6: Report Writing                                                      |
+-----------------------------------------------------------------------+   |
|  All accumulated data + interpretation                                 |   |
|       │                                                                |   |
|       ├──> PaperSolver.initial_solve()                                |   |
|       │         │                                                      |   |
|       │         └──> gen_initial_report()                             |   |
|       │                   │                                            |   |
|       │                   └──> scaffold, abstract, intro, ...         |   |
|       │                                                                |   |
|       └──> PaperSolver.solve() x N steps                              |   |
|                 │                                                      |   |
|                 └──> best_report[0]                                    |   |
|                           │                                            |   |
|                           └──> report (LaTeX) ────────────────────────+─> |
+-----------------------------------------------------------------------+   |
                                                                             |
Phase 7: Report Refinement                                                   |
+-----------------------------------------------------------------------+   |
|  plan + report                                                         |   |
|       │                                                                |   |
|       └──> ReviewersAgent.inference()                                 |   |
|                 │                                                      |   |
|                 └──> get_score()                                       |   |
|                           │                                            |   |
|                           ├──> scores (1-10)                           |   |
|                           └──> decision                                |   |
|                                   │                                    |   |
|                                   ├──> ACCEPT ──> COMPLETE            |   |
|                                   │                                    |   |
|                                   └──> REJECT ──> second_round=True   |   |
|                                                         │              |   |
|                                              +----------+              |   |
|                                              v                         |   |
|                                        Store prev_*                    |   |
|                                              │                         |   |
|                                              └──> Loop to Phase 2 ─────+─> |
+-----------------------------------------------------------------------+
```

---

## Checkpoint & Recovery Flow

```
+============================================================================+
|                      CHECKPOINT & RECOVERY                                  |
+============================================================================+

                     +-----------------------+
                     |  Workflow Execution   |
                     +-----------+-----------+
                                 |
                                 v
        +------------------------+------------------------+
        |                        |                        |
        v                        v                        v
+---------------+       +----------------+       +----------------+
| Phase         |       | Phase          |       | Phase          |
| Complete      |       | Complete       |       | Complete       |
+-------+-------+       +--------+-------+       +--------+-------+
        |                        |                        |
        v                        v                        v
+-------+-------+       +--------+-------+       +--------+-------+
| save_state()  |       | save_state()   |       | save_state()   |
| Paper{N}.pkl  |       | Paper{N}.pkl   |       | Paper{N}.pkl   |
+-------+-------+       +--------+-------+       +--------+-------+
        |                        |                        |
        +------------------------+------------------------+
                                 |
                                 v
                      +----------+----------+
                      |   state_saves/      |
                      |   Paper1.pkl        |
                      |   Paper2.pkl        |
                      |   ...               |
                      +----------+----------+
                                 |
                                 | On failure/restart
                                 v
                      +----------+----------+
                      | --load-existing     |
                      | --load-existing-path|
                      +----------+----------+
                                 |
                                 v
                      +----------+----------+
                      | Resume from last    |
                      | checkpoint          |
                      +---------------------+
```

---

## Human-in-the-Loop Integration

```
+============================================================================+
|                      HUMAN-IN-THE-LOOP FLOW                                 |
+============================================================================+

                    +-------------------------+
                    |  Phase Execution        |
                    +-----------+-------------+
                                |
                                v
                    +-----------+-------------+
                    |  Generate Output        |
                    +-----------+-------------+
                                |
                                v
                    +-----------+-------------+
                    | human_in_loop_flag      |
                    | for this phase?         |
                    +-----------+-------------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
        +-----+-----+                       +-----+-----+
        | YES       |                       | NO        |
        +-----+-----+                       +-----+-----+
              |                                   |
              v                                   v
        +-----+-----+                       +-----+-----+
        | Display   |                       | Continue  |
        | Output    |                       | to next   |
        +-----+-----+                       | phase     |
              |                             +-----+-----+
              v                                   |
        +-----+-----+                             |
        | Prompt:   |                             |
        | Happy?    |                             |
        | (Y/N)     |                             |
        +-----+-----+                             |
              |                                   |
    +---------+---------+                         |
    |                   |                         |
    v                   v                         |
+---+---+           +---+---+                     |
|  Y    |           |  N    |                     |
+---+---+           +---+---+                     |
    |                   |                         |
    |                   v                         |
    |           +-------+-------+                 |
    |           | Collect       |                 |
    |           | Feedback      |                 |
    |           +-------+-------+                 |
    |                   |                         |
    |                   v                         |
    |           +-------+-------+                 |
    |           | Append to     |                 |
    |           | self.notes    |                 |
    |           +-------+-------+                 |
    |                   |                         |
    |                   v                         |
    |           +-------+-------+                 |
    |           | Reset Agent   |                 |
    |           | State         |                 |
    |           +-------+-------+                 |
    |                   |                         |
    |                   v                         |
    |           +-------+-------+                 |
    |           | Retry Phase   |                 |
    |           +-------+-------+                 |
    |                   |                         |
    +-------------------+-----------------+-------+
                        |
                        v
              +---------+---------+
              |  Continue to      |
              |  Next Phase       |
              +-------------------+
```

---

## Web Interface Architecture

```
+============================================================================+
|                      WEB INTERFACE (app.py)                                 |
+============================================================================+

                    +-------------------------+
                    |      Flask App          |
                    |      (Port 5000)        |
                    +-----------+-------------+
                                |
        +-----------------------+------------------------+
        |           |           |           |            |
        v           v           v           v            v
   +--------+  +--------+  +--------+  +---------+  +--------+
   |   /    |  |/upload |  |/search |  |/view/id |  | /demo  |
   +---+----+  +---+----+  +---+----+  +----+----+  +---+----+
       |           |           |            |           |
       v           v           v            v           v
   +--------+  +--------+  +--------+  +---------+  +--------+
   |index   |  |upload  |  |search  |  |view     |  |demo    |
   |.html   |  |.html   |  |.html   |  |.html    |  |.html   |
   +---+----+  +---+----+  +---+----+  +----+----+  +---+----+
       |           |           |            |           |
       |           |           |            |           |
       v           v           v            v           v
   +--------+  +--------+  +--------+  +---------+  +--------+
   | Paper  |  | PyPDF2 |  |Sentence|  | Paper   |  | Demo   |
   | List   |  | Extract|  |Trans-  |  | Content |  | Mode   |
   +--------+  +--------+  |former  |  +---------+  +--------+
                           +--------+


+============================================================================+
|                      DATABASE MODEL                                         |
+============================================================================+

        +----------------------------------------+
        |              Paper                      |
        +----------------------------------------+
        | id       | Integer    | Primary Key   |
        | filename | String(120)| NOT NULL      |
        | text     | Text       | NULLABLE      |
        +----------------------------------------+


+============================================================================+
|                      SEARCH MODES                                           |
+============================================================================+

        +------------------+        +------------------+
        | SEMANTIC SEARCH  |        | TEXT SEARCH      |
        | (Online)         |        | (Offline)        |
        +------------------+        +------------------+
        |                  |        |                  |
        | Model: MiniLM-L6 |        | Word frequency   |
        | cosine_similarity|        | matching         |
        |                  |        |                  |
        +------------------+        +------------------+
```

---

## LLM Provider Integration

```
+============================================================================+
|                      LLM PROVIDERS (inference.py)                           |
+============================================================================+

                    +-------------------------+
                    |    query_model()        |
                    |    (line 35-211)        |
                    +-----------+-------------+
                                |
        +--------+--------------+---------------+--------+
        |        |              |               |        |
        v        v              v               v        v
   +--------+ +--------+   +--------+    +--------+ +--------+
   | OpenAI | |Anthropic   | Google |    |DeepSeek| |HuggingFace
   +--------+ +--------+   +--------+    +--------+ +--------+
   |        | |        |   |        |    |        | |        |
   |gpt-4o  | |claude- |   |gemini- |    |deepseek| |Qwen-   |
   |gpt-4o- | |3-5-    |   |2.0-pro |    |-chat   | |32B     |
   |mini    | |sonnet  |   |gemini- |    |(v3)    | |        |
   |o1      | |        |   |1.5-pro |    |        | |        |
   |o1-mini | |        |   |        |    |        | |        |
   |o3-mini | |        |   |        |    |        | |        |
   +--------+ +--------+   +--------+    +--------+ +--------+


                    +-------------------------+
                    |   COST TRACKING         |
                    +-------------------------+
                    | TOKENS_IN  (dict)       |
                    | TOKENS_OUT (dict)       |
                    | curr_cost_est()         |
                    +-------------------------+
```

---

## Quick Command Reference

### Agent Commands by Phase

| Phase | Agent(s) | Commands |
|-------|----------|----------|
| Literature Review | PhD | `SUMMARY`, `FULL_TEXT`, `ADD_PAPER` |
| Plan Formulation | Postdoc, PhD | `DIALOGUE`, `PLAN` |
| Data Preparation | SWEng, MLEng | `DIALOGUE`, `python`, `SEARCH_HF`, `SUBMIT_CODE` |
| Running Experiments | MLESolver | `REPLACE`, `EDIT N M` |
| Results Interpretation | Postdoc, PhD | `DIALOGUE`, `INTERPRETATION` |
| Report Writing | PaperSolver | `REPLACE`, `EDIT N M` |
| Report Refinement | Reviewers | (Direct inference - no commands) |

### Command Format

```
```COMMAND_NAME
content here
```
```

### Edit Command Format
```
```EDIT 10 20
replacement lines
for lines 10-20
```
```

---

*Quick Reference Generated: December 2024*
