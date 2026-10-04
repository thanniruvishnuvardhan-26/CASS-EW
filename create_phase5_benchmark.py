import re

with open('evaluation/phase4_benchmark.py', 'r') as f:
    content = f.read()

content = content.replace('from algorithms.temporal_belief import TemporalBeliefScheduler', 
'''from algorithms.temporal_belief import TemporalBeliefScheduler
from algorithms.phase5_temporal_profile import MultiBandTemporalProfileScheduler''')

content = content.replace('print("CASS-EW PHASE 4 BENCHMARK (SEEDS 42-46)")',
'print("CASS-EW PHASE 5 BENCHMARK (SEEDS 42-46)")')

new_benchmark = '''
        phase5_sched = MultiBandTemporalProfileScheduler(
            num_bands=config.environment.num_bands,
            epsilon=config.phase4.epsilon,
            belief_weight=config.phase4.belief_weight,
            temporal_weight=config.phase4.temporal_weight,
            seed=seed
        )
        phase5_results.append(run_evaluation(phase5_sched, config, seed))
'''
content = content.replace('temporal_results.append(run_evaluation(temporal_sched, config, seed))', 'temporal_results.append(run_evaluation(temporal_sched, config, seed))' + new_benchmark)
content = content.replace('temporal_results = []', 'temporal_results = []\n    phase5_results = []')

new_print = '''
    print_result("MultiBandTemporalProfile (Phase 5)", phase5_results)
'''
content = content.replace('print_result("TemporalBelief (Phase 4)", temporal_results)', 'print_result("TemporalBelief (Phase 4)", temporal_results)' + new_print)

ablation_replace = '''    ablations = [
        (1.0, 0.0, False, False, "Pure Belief"),
        (0.0, 1.0, False, False, "Pure Temporal"),
        (0.7, 0.3, False, False, "Hybrid Default (70/30)"),
        (0.7, 0.3, True, False, "Reliability Disabled"),
        (0.7, 0.3, False, True, "Freshness Disabled")
    ]
    
    for bw, tw, dr, df, name in ablations:
        res = []
        for seed in seeds:
            sched = MultiBandTemporalProfileScheduler(
                num_bands=config.environment.num_bands,
                epsilon=config.phase4.epsilon,
                belief_weight=bw,
                temporal_weight=tw,
                disable_reliability=dr,
                disable_freshness=df,
                seed=seed
            )
            res.append(run_evaluation(sched, config, seed))
        print_result(f"Ablation: {name}", res)'''

content = re.sub(r'    ablations = \[\s*\(1\.0.*?print_result\(f"Ablation: {name}", res\)', ablation_replace, content, flags=re.DOTALL)

with open('evaluation/phase5_benchmark.py', 'w') as f:
    f.write(content)
