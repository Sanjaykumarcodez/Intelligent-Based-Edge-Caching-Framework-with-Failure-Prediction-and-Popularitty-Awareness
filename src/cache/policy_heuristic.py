import numpy as np

class HeuristicPolicy:
    def __init__(self, capacity_items=1000, alpha=1.0, beta=0.001, gamma=1.0):
        self.capacity = capacity_items
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cache = set()

    def score(self, demand_pred, size, fail_risk):
        return self.alpha*demand_pred - self.beta*size - self.gamma*fail_risk

    def decide(self, candidates, demand_pred, sizes, fail_risk):
        # candidates: list of item ids
        scores = {it: self.score(demand_pred.get(it,0.0), sizes.get(it,1.0), fail_risk) for it in candidates}
        # admit top until capacity; evict lowest if overflow
        current = list(self.cache)
        union = set(current) | set(candidates)
        ranked = sorted(union, key=lambda k: scores.get(k, -1e9), reverse=True)
        self.cache = set(ranked[:self.capacity])
        evicted = set(current) - self.cache
        admitted = self.cache - set(current)
        return admitted, evicted
