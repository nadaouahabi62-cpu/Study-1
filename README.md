# production_llm_drl — Stage 1 (core)
Simulator (Weibull breakdowns, automatic corrective repair), Gymnasium env, dispatching rules, PPO-MLP, HGT-PPO.

    pytest -q tests
    python main.py generate-data
    python main.py run-baselines
    python main.py train --algorithm ppo_mlp --updates 200
    python main.py train --algorithm hgt_ppo --updates 30
    python main.py evaluate --checkpoint checkpoints/hgt_ppo/best.pt

Tested with Python 3.12, torch 2.14.1 (CPU), torch-geometric 2.8.1, gymnasium 1.4.0.
Note: with as-good-as-new renewal and restart-from-scratch, an operation longer than a machine's
failure age can never finish; keep Weibull scales well above the longest processing time.

## Stage 2 (LLM, graphs, digital twin)
    pytest -q tests                                  # 30 tests
    python main.py evaluate-twin --events 300 --scenarios 8   # twin selection vs NO_CHANGE under the real hidden future

* `llm/`: Pydantic schemas (closed 8-strategy catalogue), validation (ID existence, untrusted-text sanitising, maintenance language
  rejected), client layer (MockLLM, Anthropic over HTTPS with key from env, optional local HF), retries, JSONL logging, cost/latency stats.
* `graphs/`: NetworkX production graph (all node kinds) + causal event graph (engineering-rule validation; LLM edges stay
  `hypothesis` unless supported by observed events; `established_view()` excludes them).
* `digital_twin/`: strategy operators (legal pairs only), clones with hidden future arrivals and redrawn failure/repair tables,
  common random numbers across candidates, feasibility audit, paired statistics vs NO_CHANGE.
* The mock LLM only exercises the JSON contract; real-LLM quality is untested here (no API key in the sandbox).

## Stage 3 (hierarchical agent, memory, protocols)
    python main.py train --algorithm causal_llm_hierarchical_ppo --budget 240   # resumable chunks -> checkpoints/proposed/{last.pt,memory.db}
    python main.py run-generalization    # protocol A, zero-shot on held-out combinations (cached in results/generalization.json)
    python main.py run-ablations
    python main.py run-adaptation        # protocol B (limited interaction)
    python main.py run-robustness        # protocol C (noisy text reports)
    python main.py analyze-results       # paired bootstrap CI, sign-flip permutation test, Holm correction
    python main.py plot-results

Not implemented: MLflow/Hydra tracking (plain YAML + JSON logs instead), DQN / GA / GA-tabu / CP-SAT / LLM baselines,
noisy numeric observations in protocol C. Python 3.12 / torch 2.14.1 CPU tested; the Dockerfile (3.11) was not built here.
