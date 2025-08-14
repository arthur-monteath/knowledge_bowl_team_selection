
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Iterable
import json
import math

@dataclass
class Player:
    name: str
    categories: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_point(self, category: str, points: int = 1) -> None:
        if points < 0:
            raise ValueError("points must be non-negative")
        self.categories[category] += points

    def total_points(self) -> int:
        return sum(self.categories.values())

class KnowledgeBowl:
    def __init__(self) -> None:
        self.players: Dict[str, Player] = {}
        self.category_weights: Dict[str, float] = {}

    def add_player(self, name: str) -> None:
        if name in self.players:
            raise ValueError(f"Player '{name}' already exists.")
        self.players[name] = Player(name=name)

    def add_point(self, name: str, category: str, points: int = 1) -> None:
        if name not in self.players:
            self.add_player(name)
        self.players[name].add_point(category, points)

    def set_weight(self, category: str, weight: float) -> None:
        if weight <= 0:
            raise ValueError("weight must be > 0")
        self.category_weights[category] = float(weight)

    def list_players(self) -> List[str]:
        return sorted(self.players.keys())

    def list_categories(self) -> List[str]:
        cats = set()
        for p in self.players.values():
            cats.update(p.categories.keys())
        return sorted(cats)

    def _team_category_totals(self, team_names: Iterable[str]) -> Dict[str, int]:
        totals: Dict[str, int] = defaultdict(int)
        for n in team_names:
            p = self.players[n]
            for c, v in p.categories.items():
                totals[c] += v
        return totals

    def _apply_weights(self, category_totals: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for c, v in category_totals.items():
            w = self.category_weights.get(c, 1.0)
            out[c] = v * w
        return out

    def team_score(
        self,
        team_names: Iterable[str],
        aggregator: str = "sum",
        epsilon: float = 1e-9,
    ) -> Tuple[float, Dict[str, float]]:
        totals = self._team_category_totals(team_names)
        for c in self.category_weights:
            totals.setdefault(c, 0.0)
        weighted = self._apply_weights(totals)

        if not weighted:
            return 0.0, {}

        if aggregator == "sum":
            score = float(sum(weighted.values()))
        elif aggregator == "min":
            score = float(min(weighted.values()))
        elif aggregator == "geomean":
            vals = [max(epsilon, float(v)) for v in weighted.values()]
            log_sum = sum(math.log(v) for v in vals)
            score = math.exp(log_sum / len(vals))
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        return score, dict(weighted)

    def best_combinations(
        self,
        team_size: int = 4,
        top_n: int = 10,
        aggregator: str = "sum",
    ) -> List[Dict]:
        names = self.list_players()
        if len(names) < team_size:
            raise ValueError(f"Need at least {team_size} players, have {len(names)}.")

        results = []
        for combo in combinations(names, team_size):
            score, breakdown = self.team_score(combo, aggregator=aggregator)
            results.append(
                {"team": list(combo), "score": score, "breakdown": breakdown}
            )

        results.sort(key=lambda r: (-r["score"], tuple(r["team"])))
        return results[:top_n]

    def to_json(self) -> str:
        data = {
            "players": {
                name: dict(p.categories) for name, p in self.players.items()
            },
            "category_weights": dict(self.category_weights),
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "KnowledgeBowl":
        data = json.loads(s)
        kb = cls()
        for name, cats in data.get("players", {}).items():
            kb.add_player(name)
            for c, v in cats.items():
                kb.add_point(name, c, int(v))
        for c, w in data.get("category_weights", {}).items():
            kb.set_weight(c, float(w))
        return kb
