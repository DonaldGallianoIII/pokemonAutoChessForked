"""
Behavioral metrics, flags, and formatting for replay/eval output.

Computes derived metrics from the info dict returned by the training env.
Used by replay.py (per-fight) and train_ppo.py evaluate() (per-game aggregate).
"""

from __future__ import annotations
from typing import Any

# ── Gold Pressure constants (must match training-config.ts) ──────────

AVG_DAMAGE_BY_STAGE = {
    (5, 10): 8,
    (11, 16): 12,
    (17, 22): 16,
    (23, 999): 18,
}

PRESSURE_TIERS = [
    # (min_lives, name, free_gold)
    (4, "SAFE", float("inf")),
    (3, "MINOR", 50),
    (2, "MEDIUM", 30),
    (1, "ALERT", 10),
    (0, "ALERT", 10),
]


def get_avg_damage(stage: int) -> int:
    """Lookup average damage per loss for the given stage."""
    for (lo, hi), dmg in AVG_DAMAGE_BY_STAGE.items():
        if lo <= stage <= hi:
            return dmg
    return 0


def get_lives_remaining(hp: float, stage: int) -> int:
    """Calculate lives remaining = floor(HP / avgDamage)."""
    avg = get_avg_damage(stage)
    if avg <= 0:
        return 99  # stages 1-4: effectively infinite
    return int(hp // avg)


def get_pressure_tier(hp: float, stage: int) -> str:
    """Return pressure tier name: SAFE, MINOR, MEDIUM, or ALERT."""
    lives = get_lives_remaining(hp, stage)
    if lives >= 4:
        return "SAFE"
    if lives == 3:
        return "MINOR"
    if lives == 2:
        return "MEDIUM"
    return "ALERT"


def compute_board_stats(info: dict) -> dict:
    """Compute board star/unit stats from info dict."""
    board = info.get("board", [])
    if not board:
        return {
            "total_units": 0,
            "one_star_count": 0,
            "avg_star_level": 0.0,
        }
    stars = [u.get("stars", 1) for u in board]
    one_star = sum(1 for s in stars if s == 1)
    return {
        "total_units": len(board),
        "one_star_count": one_star,
        "avg_star_level": sum(stars) / len(stars) if stars else 0.0,
    }


def compute_synergy_stats(info: dict) -> dict:
    """Compute active synergy count and average depth from info dict."""
    synergies = info.get("synergies", [])
    active_count = 0
    total_count = len(synergies)
    # We can't get tier_hit from the info directly, but we can count active
    for s in synergies:
        if s["count"] >= s["threshold"] and s["threshold"] > 0:
            active_count += 1
    return {
        "active_count": active_count,
        "total_count": total_count,
    }


# ── Per-fight snapshot ───────────────────────────────────────────────

class FightSnapshot:
    """Captures behavioral state at each fight for accumulation."""

    def __init__(self, info: dict, reward_delta: dict[str, float] | None = None):
        self.stage = info.get("stage", 0)
        self.gold = info.get("gold", info.get("money", 0))
        self.life = info.get("life", 0)
        self.rank = info.get("rank", 8)
        self.level = info.get("level", 1)
        self.board_size = info.get("boardSize", 0)
        self.max_team = info.get("maxTeamSize", self.level)

        board_stats = compute_board_stats(info)
        self.one_star_count = board_stats["one_star_count"]
        self.total_board_units = board_stats["total_units"]
        self.avg_star_level = board_stats["avg_star_level"]

        syn_stats = compute_synergy_stats(info)
        self.active_synergies = syn_stats["active_count"]

        self.bench_count = len(info.get("bench", []))
        self.bench_dead_weight = info.get("benchDeadWeightCount", 0)

        self.lives_remaining = get_lives_remaining(self.life, self.stage)
        self.pressure_tier = get_pressure_tier(self.life, self.stage)

        self.buy_count = info.get("buyCount", 0)
        self.sell_count = info.get("sellCount", 0)
        self.reroll_count = info.get("rerollCount", 0)
        self.gold_spent = info.get("goldSpent", 0)

        # Per-fight reward delta (if provided)
        self.reward_delta = reward_delta or {}


# ── Game-level behavioral accumulator ────────────────────────────────

class GameMetrics:
    """Accumulates per-fight snapshots for a full game and generates flags."""

    def __init__(self):
        self.fights: list[FightSnapshot] = []
        self.total_rerolls = 0
        self.peak_gold = 0
        self.gold_at_death: float | None = None

    def add_fight(self, snapshot: FightSnapshot):
        self.fights.append(snapshot)
        self.total_rerolls += snapshot.reroll_count
        if snapshot.gold > self.peak_gold:
            self.peak_gold = snapshot.gold

    def set_death(self, gold: float):
        self.gold_at_death = gold

    def _fights_in_stage_range(self, lo: int, hi: int) -> list[FightSnapshot]:
        return [f for f in self.fights if lo <= f.stage <= hi]

    def generate_flags(self) -> list[tuple[str, str]]:
        """Generate behavioral warning/info/good flags. Returns [(severity, message)]."""
        flags: list[tuple[str, str]] = []

        # Died rich
        if self.gold_at_death is not None and self.gold_at_death > 30:
            flags.append(("WARN", f"Died with {self.gold_at_death:.0f}g (gold_at_death)"))

        # Zoo board
        late_fights = self._fights_in_stage_range(19, 999)
        for f in late_fights:
            if f.one_star_count > 4:
                flags.append(("WARN", f"{f.one_star_count} one-star units on board at stage {f.stage}"))
                break

        # Never rolled
        if self.total_rerolls == 0 and len(self.fights) > 15:
            flags.append(("WARN", "Never rolled after stage 15"))
        elif self.total_rerolls == 0 and len(self.fights) > 10:
            pass  # still early, no warning

        # Gold pressure alert
        for f in self.fights:
            if f.pressure_tier == "ALERT":
                flags.append(("WARN", f"Gold pressure hit ALERT tier at stage {f.stage}"))
                break

        # Bench hoarding
        consecutive_dw = 0
        for f in self.fights:
            if f.bench_dead_weight > 2:
                consecutive_dw += 1
                if consecutive_dw >= 3:
                    flags.append(("WARN", f"{f.bench_dead_weight} dead-weight bench units for 3+ consecutive fights (stage {f.stage})"))
                    break
            else:
                consecutive_dw = 0

        # Synergy soup
        for f in late_fights:
            if f.active_synergies < 3 and f.stage >= 15:
                flags.append(("WARN", f"Only {f.active_synergies} active synergies at stage {f.stage}"))
                break

        # Board not full
        mid_fights = self._fights_in_stage_range(10, 999)
        for f in mid_fights:
            if f.board_size < f.max_team - 1:
                flags.append(("WARN", f"Board not full ({f.board_size}/{f.max_team}) at stage {f.stage}"))
                break

        # Gold excess chronic
        excess_streak = 0
        for f in self.fights:
            if f.gold > 55:
                excess_streak += 1
                if excess_streak >= 5:
                    flags.append(("WARN", f"Gold > 55 for 5+ consecutive fights (stage {f.stage})"))
                    break
            else:
                excess_streak = 0

        # No upgrades
        for f in late_fights:
            if f.avg_star_level < 1.5 and f.stage >= 20:
                flags.append(("INFO", f"Avg star level {f.avg_star_level:.2f} at stage {f.stage}"))
                break

        # Strong comp (good)
        for f in late_fights:
            if f.active_synergies >= 5 and f.avg_star_level >= 2.0:
                flags.append(("GOOD", f"Strong comp: {f.active_synergies} active synergies, {f.avg_star_level:.1f} avg stars at stage {f.stage}"))
                break

        # Clean econ (good)
        if self.gold_at_death is not None and self.gold_at_death < 10:
            flags.append(("GOOD", f"Clean econ: died with only {self.gold_at_death:.0f}g"))

        # Peak gold
        flags.append(("INFO", f"Peak gold: {self.peak_gold}g"))

        # Active synergies peak
        if self.fights:
            peak_syn = max(f.active_synergies for f in self.fights)
            flags.append(("INFO", f"Active synergies peaked at {peak_syn}"))

        # Rolls after stage 15
        rolls_after_15 = sum(f.reroll_count for f in self.fights if f.stage >= 15)
        if rolls_after_15 == 0 and any(f.stage >= 15 for f in self.fights):
            flags.append(("WARN", "Never rolled after stage 15"))

        return flags

    def compute_aggregate(self) -> dict[str, float]:
        """Compute aggregate behavioral metrics for TensorBoard logging."""
        if not self.fights:
            return {}

        result: dict[str, float] = {}

        # Gold by stage range
        early = self._fights_in_stage_range(1, 10)
        mid = self._fights_in_stage_range(11, 20)
        late = self._fights_in_stage_range(21, 999)

        if early:
            result["gold_at_fight_early"] = sum(f.gold for f in early) / len(early)
        if mid:
            result["gold_at_fight_mid"] = sum(f.gold for f in mid) / len(mid)
        if late:
            result["gold_at_fight_late"] = sum(f.gold for f in late) / len(late)

        result["peak_gold"] = float(self.peak_gold)
        if self.gold_at_death is not None:
            result["gold_at_death"] = self.gold_at_death

        # One-star percentage by stage range
        if early:
            pcts = [f.one_star_count / max(f.total_board_units, 1) for f in early]
            result["one_star_pct_early"] = sum(pcts) / len(pcts)
        if mid:
            pcts = [f.one_star_count / max(f.total_board_units, 1) for f in mid]
            result["one_star_pct_mid"] = sum(pcts) / len(pcts)
        if late:
            pcts = [f.one_star_count / max(f.total_board_units, 1) for f in late]
            result["one_star_pct_late"] = sum(pcts) / len(pcts)

        # Star level late
        if late:
            result["avg_star_level_late"] = sum(f.avg_star_level for f in late) / len(late)

        # Active synergies late
        if late:
            result["active_synergies_late"] = sum(f.active_synergies for f in late) / len(late)

        # Bench dead-weight late
        if late:
            result["bench_deadweight_late"] = sum(f.bench_dead_weight for f in late) / len(late)

        # Total rolls
        result["total_rolls_per_game"] = float(self.total_rerolls)
        result["rolls_after_stage_15"] = float(
            sum(f.reroll_count for f in self.fights if f.stage >= 15)
        )

        # Board full percentage
        full_pct = sum(1 for f in self.fights if f.board_size >= f.max_team) / len(self.fights)
        result["board_full_pct"] = full_pct

        # Pressure alert frequency
        alert_hit = any(f.pressure_tier == "ALERT" for f in self.fights)
        result["pressure_alert_hit"] = 1.0 if alert_hit else 0.0

        # Died rich
        if self.gold_at_death is not None:
            result["died_rich"] = 1.0 if self.gold_at_death > 30 else 0.0

        return result


# ── Formatting helpers for replay ────────────────────────────────────

def format_enhanced_fight_line(snapshot: FightSnapshot) -> str:
    """Format the enhanced fight detail lines for replay output."""
    lines = []

    # Line 1: Lives, pressure, gold details
    rd = snapshot.reward_delta
    ge = rd.get("goldExcess", 0)
    gp = rd.get("goldPressure", 0)
    lines.append(
        f"    Lives: {snapshot.lives_remaining} | Pressure: {snapshot.pressure_tier} "
        f"| Gold: {snapshot.gold:.0f}g"
        + (f" (excess: {ge:+.2f}, pressure: {gp:+.2f})" if ge != 0 or gp != 0 else "")
    )

    # Line 2: Board composition
    sd = rd.get("synergyDepth", 0)
    lines.append(
        f"    Board: {snapshot.board_size}/{snapshot.max_team} "
        f"| 1\u2605: {snapshot.one_star_count} "
        f"| AvgStar: {snapshot.avg_star_level:.2f} "
        f"| ActiveSyn: {snapshot.active_synergies}"
        + (f" (depth: {sd:+.3f})" if sd != 0 else "")
    )

    # Line 3: Bench and unit quality (only if relevant)
    uq = rd.get("unitQuality", 0)
    bdw = rd.get("benchDeadWeight", 0)
    if snapshot.bench_count > 0 or uq != 0:
        bench_part = f"    Bench: {snapshot.bench_count} units"
        if snapshot.bench_dead_weight > 0:
            bench_part += f" ({snapshot.bench_dead_weight} dead-weight, penalty: {bdw:+.2f})"
        if uq != 0:
            bench_part += f" | UnitQual: {uq:+.2f}"
        lines.append(bench_part)

    return "\n".join(lines)


def format_flags(flags: list[tuple[str, str]]) -> str:
    """Format behavioral flags for output."""
    if not flags:
        return ""
    lines = ["  --- Behavioral Flags ---"]
    for severity, msg in flags:
        tag = f"[{severity}]"
        lines.append(f"    {tag:<6} {msg}")
    return "\n".join(lines)
