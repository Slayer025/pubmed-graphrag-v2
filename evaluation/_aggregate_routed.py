import json
from pathlib import Path

p = Path('evaluation/results_routed.jsonl')
recs = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
n = len(recs)
metrics = {
    'num_queries': n,
    'recall@5': round(sum(r['recall@5'] for r in recs) / n, 4),
    'recall@10': round(sum(r['recall@10'] for r in recs) / n, 4),
    'mrr@10': round(sum(r['mrr@10'] for r in recs) / n, 4),
    'avg_latency_ms': round(sum(r['latency_ms'] for r in recs) / n, 2),
}
print(json.dumps(metrics, indent=2))
