"""CLI. Commands: generate-data, train, evaluate, run-baselines, evaluate-twin (Stage 2)."""
import argparse, json, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(p):
    return yaml.safe_load(open(p))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-data"); g.add_argument("--config", default="configs/environment.yaml"); g.add_argument("--out", default="results/instances.json")
    t = sub.add_parser("train"); t.add_argument("--algorithm", required=True, choices=["ppo_mlp", "hgt_ppo", "causal_llm_hierarchical_ppo"])
    t.add_argument("--budget", type=int, default=1200, help="seconds (hierarchical, resumable)"); t.add_argument("--updates", type=int); t.add_argument("--n-steps", type=int); t.add_argument("--size")
    e = sub.add_parser("evaluate"); e.add_argument("--checkpoint", required=True)
    b = sub.add_parser("run-baselines")
    h = sub.add_parser("train-hier"); h.add_argument("--budget", type=int, default=1200); h.add_argument("--fresh", action="store_true")
    sub.add_parser("run-generalization"); sub.add_parser("run-ablations"); sub.add_parser("run-adaptation"); sub.add_parser("run-robustness")
    sub.add_parser("analyze-results"); sub.add_parser("plot-results")
    w = sub.add_parser("evaluate-twin"); w.add_argument("--events", type=int, default=300); w.add_argument("--scenarios", type=int, default=8)
    a = ap.parse_args()
    env_cfg = load("configs/environment.yaml")
    if getattr(a, "size", None): env_cfg["size"] = a.size
    if a.cmd == "generate-data":
        from experiments.data_generator import generate_instance
        cfg = load(a.config)
        seeds = [cfg["eval_seed0"] + i for i in range(cfg["n_eval_instances"])]
        json.dump([generate_instance(cfg["size"], s, cfg["ddt"], urgent_rate=cfg["urgent_rate"]) for s in seeds], open(a.out, "w"))
        print(f"wrote {len(seeds)} instances to {a.out}")
    elif a.cmd == "train" and a.algorithm == "causal_llm_hierarchical_ppo":
        from experiments.train_hierarchical import train_hier
        train_hier(a.budget, max_updates=a.updates)
    elif a.cmd == "train":
        from experiments.train import train
        ppo, hgt = load("configs/ppo.yaml"), load("configs/hgt.yaml")
        if a.updates: ppo["updates"] = a.updates
        if a.n_steps: ppo["n_steps"] = a.n_steps
        train(a.algorithm, env_cfg, ppo, hgt)
    elif a.cmd == "evaluate" and "proposed" in a.checkpoint:
        import numpy as np
        from experiments.run_generalization import EVAL_SEED0, eval_one, load_components
        from experiments.scenario_generator import TEST_COMBOS
        C = load_components(a.checkpoint, os.path.join(os.path.dirname(a.checkpoint), "memory.db"))
        for c in TEST_COMBOS:
            print(f"{c:26s} proposed TWT={np.mean([eval_one(C, 'proposed', c, EVAL_SEED0 + i)['total_weighted_tardiness'] for i in range(5)]):7.1f}")
    elif a.cmd == "evaluate":
        import torch
        from experiments.train import build_policy
        from experiments.evaluate import eval_policy, make_env, summarize
        ck = torch.load(a.checkpoint, weights_only=False)
        env = make_env(ck["env_cfg"]); pol = build_policy(ck["kind"], env, ck["hgt_cfg"]); pol.load_state_dict(ck["state"])
        seeds = [ck["env_cfg"]["eval_seed0"] + i for i in range(ck["env_cfg"]["n_eval_instances"])]
        print(json.dumps(summarize(eval_policy(pol, ck["kind"], ck["env_cfg"], seeds)), indent=1))
    elif a.cmd == "train-hier":
        from experiments.train_hierarchical import train_hier
        if a.fresh:
            for f in ("checkpoints/proposed/last.pt", "checkpoints/proposed/memory.db"):
                if os.path.exists(f): os.remove(f)
        train_hier(a.budget)
    elif a.cmd == "run-generalization":
        from experiments.run_generalization import run; run()
    elif a.cmd == "run-ablations":
        from experiments.run_ablations import ABL; from experiments.run_generalization import run; run(methods=ABL)
    elif a.cmd == "run-adaptation":
        from experiments.run_adaptation_robustness import run_adaptation; run_adaptation()
    elif a.cmd == "run-robustness":
        from experiments.run_adaptation_robustness import run_robustness; run_robustness()
    elif a.cmd == "analyze-results":
        from experiments.statistical_analysis import analyze, print_table; d = analyze(); print_table(d)
    elif a.cmd == "plot-results":
        from experiments.plot_results import plot; plot()
    elif a.cmd == "evaluate-twin":
        from experiments.evaluate_twin import main as twin_main
        twin_main(a.events, a.scenarios)
    elif a.cmd == "run-baselines":
        from experiments.run_baselines import run_baselines
        run_baselines(env_cfg, load("configs/baselines.yaml")["rules"])


if __name__ == "__main__":
    main()
