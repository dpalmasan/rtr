"""Job hopping on Moving Peaks: a team of agents that should keep every peak staffed.

Each agent works one hill (its employer) by hill climbing: a Gaussian step accepted when the value
does not drop, with a per-agent step size adapted by the 1/5th success rule. Every ``cycle``
rounds an agent decides whether to stay or hop. A hop moves it to a place where no colleague works
(the farthest of ``destinations`` random points from the rest of the team, read off the forum) and
restarts its step size.

Collaboration enters through the agent's utility. An agent is *redundant* when a fitter colleague
works within ``exclusion`` of it (the same hill); otherwise it *leads* its hill. Its utility is its
value when it leads and 0 when redundant, so the team utility is roughly the summed height of the
staffed peaks and an agent is paid only for what it adds to the team. A cycle is rewarded by the
change in that utility. The context of a decision is whether the agent leads or is redundant.

Policies:
  learned     epsilon-greedy over values per (context, arm); one table shared by the team (forum)
              or one per agent (private)
  random      hop with probability ``hop_probability``
  rule        hand-coded exclusion (hop exactly when redundant), as in mQSO-style niching
  fixed:stay  never hop (independent hill climbers)
  fixed:hop   always hop
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from aumo.algorithms.cma_workers import CMAWorkers
from aumo.problems.moving_peaks import MovingPeaksWorld, PeaksConfig

STAY, HOP = 0, 1
LEAD, REDUNDANT = 0, 1


@dataclass(frozen=True, slots=True)
class PeakHoppingConfig:
    agents: int = 20
    policy: str = "learned"
    value_sharing: str = "forum"
    context: bool = True
    cycle: int = 10
    epsilon: float = 0.1
    learning_rate: float = 0.1
    hop_probability: float = 0.5
    # Radius of "the same hill"; None = half the mean peak spacing, span / (2 * peaks ** (1 / dim)).
    exclusion: float | None = None
    destinations: int = 8
    step_init: float = 2.0
    hop_step: float = 5.0
    step_up: float = 1.5
    step_down: float = 0.9
    step_bounds: tuple[float, float] = (0.01, 20.0)
    reward_scale: float = 100.0
    # "title": change in the agent's own utility over the cycle (inheriting a hill counts as earned).
    # "counterfactual": team utility now minus team utility had the agent stayed where it started
    # the cycle, colleagues held at their current positions (costs one evaluation per cycle).
    credit: str = "title"
    # Count a hill as worth max(f, 0) (a slope far below every peak is not a job).
    floor_utility: bool = False
    # Hop destination: "far" = farthest of ``destinations`` random points from the team; "uniform" = random.
    destination: str = "far"
    # Probation: a hopper that is redundant after this many rounds of its stint quits early and
    # re-decides (0 = off).
    probation: int = 0
    # Discount on the value of the next decision's context (Q-learning over lead/redundant);
    # 0 = contextual bandit, where a hop is judged only on its own cycle.
    lookahead: float = 0.0
    # Memory (lineage over time): a decision is settled once ``memory`` later cycles have closed, on
    # its own reward plus memory_decay**k times the reward k cycles later. 0 = settle immediately.
    memory: int = 0
    memory_decay: float = 0.7
    # value_sharing="kings": private tables; every ``epoch`` rounds the ``kings`` agents with the best
    # reputation (mean settled credit of their own decisions and of decisions made on their advice)
    # become kings, and every other agent follows one, chosen by reputation rank. A follower decides
    # on (1 - mentor_weight) * own + mentor_weight * mentor's effective table; its settled credit also
    # trains the mentor's table (rate learning_rate * king_learning). With meta_kings a king follows
    # the best king that out-ranks it by ``meta_margin`` (a king of kings).
    kings: int = 3
    epoch: int = 200
    mentor_weight: float = 0.5
    king_learning: float = 0.5
    reputation_prior: float = 5.0
    reputation_decay: float = 0.5
    meta_kings: bool = False
    meta_margin: float = 0.005
    # reputation="clan": a king is judged by its clan (itself and its direct followers): the mean
    # utility per member over the epoch, i.e. the peaks the clan holds. Kings persist; the weakest is
    # deposed when the best clan out-scores it by ``depose_margin`` (relative), and the best clan's
    # strongest member founds a new clan carrying that clan's blended table. Each epoch a follower
    # re-picks its king (by clan rank) with probability ``migration``. clan_bonus adds the change of
    # the agent's clan utility per member over its cycle to its reward.
    # Partial vision (None = everyone sees the whole team). Agent i sees colleagues within
    # v_i = vision + (vision_max - vision) * h_i / (h_i + vision_scale), where h_i counts its successful
    # hops (hop cycles that ended with i leading a hill of positive value, as far as i can tell): career
    # breadth. Redundancy, hop destinations, counterfactual credit and the utility an agent reports
    # use only the colleagues it sees. With clan_view (kings mode) an agent also sees whatever any
    # member of its clan tree sees: the king pools its clan's view (a meta-view).
    vision: float | None = None
    vision_max: float = 50.0
    vision_scale: float = 3.0
    clan_view: bool = False
    # Social vision (needs vision set, which then only switches partial vision on): every agent has its
    # own radius, log-uniform in vision_init at the start. At each cycle close an agent compares its
    # record of holding a hill with a random colleague's (read off the forum) and, when the colleague's
    # is better, copies the colleague's radius times exp(N(0, vision_noise)). Holding is judged with a
    # hill-valley test, so a blind agent cannot flatter itself: the agent holds when its value is
    # positive and the nearest fitter colleague within exclusion is separated from it by a valley (the
    # midpoint is lower than both), or there is none. At most one evaluation per cycle.
    vision_social: bool = False
    vision_init: tuple[float, float] = (0.5, 100.0)
    vision_noise: float = 0.2
    vision_memory: float = 0.7
    # hold_positive: holding also needs a positive value (the confirmed setting, designed for Moving Peaks).
    # False judges holding by the hill-valley test alone, so imitation also works when f <= 0 everywhere.
    hold_positive: bool = True
    # hill_valley: a redundant verdict also needs the fitter colleague (nearest one in sight within
    # exclusion) to pass the hill-valley test. At most one evaluation per decision.
    hill_valley: bool = False
    # "always": re-evaluate every agent each round (sees changes at once); "detect": re-evaluate only the
    # best agent each round and the whole team when its value has changed (the standard protocol);
    # "informed": re-evaluate the team only when the world reports a change (DMMOP protocol).
    reevaluate: str = "always"
    # Worker: "hill" (Gaussian hill climber with the 1/5th rule) or "cma": each agent runs its own
    # CMA-ES (cma_lambda offspring per generation, one generation per round), restarted with cma_sigma0
    # around its new place after a hop and with cma_sigma_change around its best point after a change
    # (so "cma" needs reevaluate="detect" or "informed"). A converged worker sleeps (spends no
    # evaluations) until restarted; when every worker sleeps, all are restarted with cma_sigma0 around
    # their best points. The agent's position is the best point of its current run.
    worker: str = "hill"
    cma_lambda: int | None = None
    cma_sigma0: float = 10.0
    cma_sigma_change: float = 2.0
    cma_tol: float = 1e-10
    # Team archive (static worlds, cma workers): a worker that converges retires its point to the
    # archive (unless an archived point on the same hill, by the hill-valley test, is at least as good)
    # and the agent moves to a new place (the far-destination rule, counting archived points as
    # occupied) with a fresh CMA-ES. Archived points act as colleagues who never hop: an agent that
    # sees a fitter archived point within exclusion is redundant. The world's reporter gets team and
    # archive. Independent of the hop policy, so "fixed:stay" with the archive is a restart CMA-ES.
    archive: bool = False
    # Restart-on-convergence control (cma workers, no archive): a worker that converges, or whose best
    # point has not improved for 10 + ceil(30 D / lambda) generations (stagnation), moves to the
    # farthest of ``destinations`` random points from all teammates and restarts with cma_sigma0.
    # Combine with policy="fixed:stay" to remove the redundancy trigger.
    restart_far: bool = False
    # With restart_far: the point a searcher leaves is kept as a reported solution until the next
    # environment change (a per-environment archive), and kept points count as occupied for destinations.
    restart_keep: bool = False
    # Radius imitation (with vision_social): False keeps every agent's initial log-uniform radius.
    imitate: bool = True
    # Stale teammate information: the redundancy test and the relocation destination see the
    # teammates' positions and fitness as of the last snapshot, refreshed every ``stale_period`` rounds
    # and after every announced change; each agent knows its own state exactly. 1 = always current.
    stale_period: int = 1
    # Peaks assumed when deriving the default exclusion radius on a world that does not declare them.
    assumed_peaks: int = 10
    reputation: str = "credit"
    # policy="culture" (needs value_sharing="kings", reputation="clan"): every king carries a heritable
    # culture (hop propensity as a leader, hop propensity when redundant, and how close a fitter
    # colleague must be to count as "on my hill", as a multiple of ``exclusion``). A member hops with
    # probability (1 - culture_weight) * clan culture + culture_weight * its own learned preference.
    # A prince founding a clan inherits the best clan's culture with Gaussian mutation (logit / log
    # scale). Clans are still scored with the standard exclusion, so a culture cannot flatter itself.
    # Clan judging. clan_score: "utility" (mean utility per member) or "held" (share of members leading
    # a hill of positive value: hills held, whatever their height). depose: "margin" (depose_margin) or
    # "bayes": every epoch / score_blocks rounds each clan's score is sampled, a king keeps its last
    # score_window samples, and the weakest king (with at least score_window // 2 samples) is deposed,
    # or a king bows to the best king (meta_kings), only when BF10 that the best clan scores higher
    # reaches bayes_threshold.
    clan_score: str = "utility"
    depose: str = "margin"
    score_blocks: int = 4
    score_window: int = 12
    bayes_threshold: float = 3.0
    culture_weight: float = 0.3
    # Fixed starting cultures (logit hop as leader, logit hop when redundant, log same-hill scale), one per
    # king in election order; None = random.
    initial_cultures: tuple[tuple[float, float, float], ...] | None = None
    culture_mutation: float = 0.5
    culture_exclusion: tuple[float, float] = (0.1, 2.0)
    depose_margin: float = 0.1
    migration: float = 0.2
    clan_bonus: float = 0.0
    radius: float = 1.0
    burn_in: int = 500
    record_cycles: bool = False


@dataclass(slots=True)
class PeakHoppingResult:
    coverage: np.ndarray
    occupancy: np.ndarray
    error: np.ndarray
    hop_share: float
    evaluations: int
    changes: int
    values: np.ndarray
    burn_in: int
    summary: dict[str, float] = field(default_factory=dict)
    # With record_cycles: (round, context, arm, redundant at end, reward) per closed cycle.
    cycles: list[tuple[int, int, int, bool, float]] = field(default_factory=list)
    # With record_cycles, per round: leads on a peak, leads off peaks, redundant on a peak,
    # redundant off peaks, distinct peaks covered by leads, by redundant agents only.
    census: np.ndarray | None = None
    # Per epoch: round, policy drift, king churn, kings, hierarchy depth, followers of the top king,
    # coverage over the epoch.
    epochs: list[dict[str, float]] = field(default_factory=list)
    summary_extra: dict[str, float] = field(default_factory=dict)


def _redundant(x: np.ndarray, f: np.ndarray, exclusion: float) -> np.ndarray:
    distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    index = np.arange(len(f))
    fitter = (f[None, :] > f[:, None]) | ((f[None, :] == f[:, None]) & (index[None, :] < index[:, None]))
    return ((distance < exclusion) & fitter).any(axis=1)


def _redundant_seen(distance: np.ndarray, f: np.ndarray, exclusion: float, sees: np.ndarray) -> np.ndarray:
    index = np.arange(len(f))
    fitter = (f[None, :] > f[:, None]) | ((f[None, :] == f[:, None]) & (index[None, :] < index[:, None]))
    return ((distance < exclusion) & fitter & sees).any(axis=1)


def _bayes_factor_better(a: np.ndarray, b: np.ndarray) -> float:
    """BF10 that mean(a) > mean(b): difference ~ N(0, tau^2) with tau^2 the pooled sample variance
    (unit information) against difference 0; 0 when the observed difference has the wrong sign."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    gap = float(np.mean(a) - np.mean(b))
    if gap <= 0:
        return 0.0
    tau2 = max(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)), 1e-8)
    v0 = tau2 / len(a) + tau2 / len(b)
    v1 = v0 + tau2
    return float(np.sqrt(v0 / v1) * np.exp(0.5 * gap * gap * (1.0 / v0 - 1.0 / v1)))


def _team_utility(x: np.ndarray, f: np.ndarray, exclusion: float) -> float:
    return float(f[~_redundant(x, f, exclusion)].sum())


def _team_utility_floor(x: np.ndarray, f: np.ndarray, exclusion: float) -> float:
    return float(np.maximum(f[~_redundant(x, f, exclusion)], 0.0).sum())


def run_peak_hopping(world_config: PeaksConfig, config: PeakHoppingConfig, seed: int, rounds: int = 0,
                     budget: int | None = None) -> PeakHoppingResult:
    """Run for ``rounds`` rounds, or, with ``budget``, until the world has counted that many evaluations
    (then coverage is averaged over the rounds after the first change, and the summary adds the
    standard offline error)."""
    cfg = config
    if isinstance(world_config, PeaksConfig):
        wc = world_config
        world = MovingPeaksWorld(wc, seed)
    else:
        # Any world with low/high/dim, evaluate (counted), values, measure, maybe_change and changes;
        # distances in the config are in units of 1/100 of the domain span (Moving Peaks units).
        world = world_config
        wc = SimpleNamespace(low=world.low, high=world.high, dim=world.dim, peaks=cfg.assumed_peaks, change_frequency=0)
        if budget is None:
            budget = world.budget
        if hasattr(world, "reporter"):
            world.reporter = lambda: np.vstack([x] + archive_x + kept_x) if (archive_x or kept_x) else x.copy()
    unit = (wc.high - wc.low) / 100.0
    if unit != 1.0:
        cfg = replace(cfg, step_init=cfg.step_init * unit, hop_step=cfg.hop_step * unit,
                      step_bounds=(cfg.step_bounds[0] * unit, cfg.step_bounds[1] * unit), radius=cfg.radius * unit,
                      vision=cfg.vision * unit if cfg.vision is not None else None, vision_max=cfg.vision_max * unit,
                      vision_init=(cfg.vision_init[0] * unit, cfg.vision_init[1] * unit),
                      exclusion=cfg.exclusion * unit if cfg.exclusion is not None else None,
                      cma_sigma0=cfg.cma_sigma0 * unit, cma_sigma_change=cfg.cma_sigma_change * unit, cma_tol=cfg.cma_tol * unit)
    if budget is not None:
        rounds = budget // cfg.agents + 2
    if cfg.policy == "culture" and not (cfg.value_sharing == "kings" and cfg.reputation == "clan"):
        raise ValueError("culture needs value_sharing='kings' and reputation='clan'")
    if cfg.policy not in {"learned", "random", "rule", "fixed:stay", "fixed:hop", "culture"}:
        raise ValueError(f"Unknown policy: {cfg.policy}")
    if cfg.credit not in {"title", "counterfactual"}:
        raise ValueError(f"Unknown credit: {cfg.credit}")
    if cfg.value_sharing not in {"forum", "private", "kings"}:
        raise ValueError(f"Unknown value sharing: {cfg.value_sharing}")
    if cfg.clan_score not in {"utility", "held"} or cfg.depose not in {"margin", "bayes"}:
        raise ValueError("clan_score must be utility|held and depose margin|bayes")
    if cfg.worker not in {"hill", "cma"} or (cfg.worker == "cma" and cfg.reevaluate == "always"):
        raise ValueError("worker must be hill|cma, and cma needs reevaluate detect|informed")
    if cfg.reevaluate not in {"always", "detect", "informed"}:
        raise ValueError(f"Unknown reevaluate: {cfg.reevaluate}")
    if cfg.reputation not in {"credit", "clan"}:
        raise ValueError(f"Unknown reputation: {cfg.reputation}")
    if cfg.value_sharing == "kings" and cfg.lookahead:
        raise ValueError("kings use memory, not lookahead")
    rng = np.random.default_rng([seed, 5])
    n, dim = cfg.agents, wc.dim
    exclusion = cfg.exclusion if cfg.exclusion is not None else (wc.high - wc.low) / (2 * wc.peaks ** (1 / dim))

    x = rng.uniform(wc.low, wc.high, (n, dim))
    step = np.full(n, cfg.step_init)
    kings_mode = cfg.value_sharing == "kings"
    tables = 1 if cfg.value_sharing == "forum" else n
    values = np.zeros((tables, 2, 2))
    mentor = np.full(n, -1)
    king_set: set[int] = set()
    credit_sum = np.zeros(n)
    credit_count = np.zeros(n)
    reputation = np.zeros(n)
    pending: list[deque] = [deque() for _ in range(n)]
    previous_policy: np.ndarray | None = None
    epochs: list[dict[str, float]] = []
    epoch_start = 0
    kings_list: list[int] = []
    clan = np.arange(n)
    clan_sum = np.zeros(n)
    clan_count = np.zeros(n)
    own_sum = np.zeros(n)
    clan_start = np.zeros(n)
    clan_mode = kings_mode and cfg.reputation == "clan"
    evidence = cfg.depose == "bayes"
    block_len = max(1, cfg.epoch // cfg.score_blocks)
    block_sum = np.zeros(n)
    block_count = np.zeros(n)
    history: dict[int, deque] = {}
    founders: list[int] = []
    partial = cfg.vision is not None
    cultural = cfg.policy == "culture"
    culture = np.zeros((n, 3))
    low_x, high_x = np.log(cfg.culture_exclusion[0]), np.log(cfg.culture_exclusion[1])
    career = np.zeros(n)
    sees = np.ones((n, n), dtype=bool)
    vision_rng = np.random.default_rng([seed, 13])
    radius = np.exp(vision_rng.uniform(*np.log(cfg.vision_init), n)) if cfg.vision_social else None
    hold_score = np.zeros(n)
    if cfg.vision_social and not partial:
        raise ValueError("vision_social needs vision (partial vision)")
    if cfg.archive and cfg.worker != "cma":
        raise ValueError("archive needs cma workers")
    archive_x: list[np.ndarray] = []
    kept_x: list[np.ndarray] = []
    archive_f: list[float] = []
    retirements = 0

    def radius_of() -> np.ndarray | None:
        if not partial:
            return None
        if cfg.vision_social:
            return radius
        return cfg.vision + (cfg.vision_max - cfg.vision) * career / (career + cfg.vision_scale)

    def nearest_fitter(i: int, f: np.ndarray, mask: np.ndarray) -> int:
        gap = np.linalg.norm(x - x[i], axis=1)
        fitter = (f > f[i]) | ((f == f[i]) & (np.arange(n) < i))
        rivals = np.flatnonzero((gap < exclusion) & fitter & mask)
        return int(rivals[np.argmin(gap[rivals])]) if len(rivals) else -1
    cycle_left = np.zeros(n, dtype=int)
    arm = np.full(n, STAY)
    ctx = np.zeros(n, dtype=int)
    u_start = np.zeros(n)
    x_start = x.copy()
    started = np.zeros(n, dtype=bool)
    hops = cycles = evaluations = 0
    # Hops taken while judged redundant, and those where no fitter colleague in range (and in sight)
    # actually stood on the same cone as the agent (a neighbouring peak was mistaken for "my hill").
    redundant_hops = false_hops = 0
    coverage, occupancy, error = (np.zeros(rounds) for _ in range(3))
    log: list[tuple[int, int, int, bool, float]] = []
    census = np.zeros((rounds, 6)) if cfg.record_cycles else None

    def choose(i: int, redundant: bool) -> int:
        if cfg.policy == "fixed:stay":
            return STAY
        if cfg.policy == "fixed:hop":
            return HOP
        if cfg.policy == "rule":
            return HOP if redundant else STAY
        if cultural:
            r = root_of(i)
            p_culture = 1.0 / (1.0 + np.exp(-culture[r, ctx[i]]))
            row = values[i, ctx[i]]
            p_own = 0.5 if row[HOP] == row[STAY] else float(row[HOP] > row[STAY])
            return int(rng.random() < (1 - cfg.culture_weight) * p_culture + cfg.culture_weight * p_own)
        if cfg.policy == "random" or rng.random() < cfg.epsilon:
            return int(rng.random() < cfg.hop_probability) if cfg.policy == "random" else int(rng.integers(2))
        row = effective(i)[ctx[i]]
        return int(rng.choice(np.flatnonzero(row == row.max())))

    def effective(i: int, depth: int = 0) -> np.ndarray:
        own = values[i if tables > 1 else 0]
        if not kings_mode or mentor[i] < 0 or depth > n:
            return own
        return (1 - cfg.mentor_weight) * own + cfg.mentor_weight * effective(int(mentor[i]), depth + 1)

    def settle(i: int, c: int, a: int, advisor: int, credit: float, next_ctx: int) -> None:
        table = i if tables > 1 else 0
        target = credit
        if cfg.lookahead:
            target += cfg.lookahead * values[table, next_ctx].max()
        values[table, c, a] += cfg.learning_rate * (target - values[table, c, a])
        credit_sum[i] += credit
        credit_count[i] += 1
        if kings_mode and advisor >= 0:
            values[advisor, c, a] += cfg.learning_rate * cfg.king_learning * (credit - values[advisor, c, a])
            credit_sum[advisor] += credit
            credit_count[advisor] += 1

    def root_of(i: int) -> int:
        d = 0
        while mentor[i] >= 0 and d <= n:
            i, d = int(mentor[i]), d + 1
        return i

    def depth_of(i: int) -> int:
        d = 0
        while mentor[i] >= 0 and d <= n:
            i, d = int(mentor[i]), d + 1
        return d

    def close_epoch(t: int) -> None:
        nonlocal king_set, previous_policy, epoch_start
        policy = np.array([[0.5 if r[STAY] == r[HOP] else float(r[HOP] > r[STAY]) for r in effective(i)] for i in range(n)])
        drift = float(np.abs(policy - previous_policy).mean()) if previous_policy is not None else float("nan")
        previous_policy = policy
        record = {"round": t, "drift": drift, "coverage": float(coverage[epoch_start:t].mean()) if t > epoch_start else float("nan"),
                  "hop_policy_lead": float(policy[:, LEAD].mean()), "hop_policy_redundant": float(policy[:, REDUNDANT].mean())}
        epoch_start = t
        if clan_mode:
            elect_clans(record)
        elif kings_mode:
            fresh = credit_sum / (credit_count + cfg.reputation_prior)
            reputation[:] = cfg.reputation_decay * reputation + (1 - cfg.reputation_decay) * fresh
            credit_sum[:] = 0.0
            credit_count[:] = 0.0
            order = np.argsort(-reputation, kind="stable")
            new_kings = [int(k) for k in order[: cfg.kings]]
            churn = 1 - len(king_set & set(new_kings)) / len(king_set | set(new_kings)) if king_set else float("nan")
            king_set = set(new_kings)
            mentor[:] = -1
            ranks = np.arange(len(new_kings), 0, -1, dtype=float)
            for i in range(n):
                if i not in king_set:
                    mentor[i] = new_kings[int(rng.choice(len(new_kings), p=ranks / ranks.sum()))]
            if cfg.meta_kings:
                for k in new_kings[1:]:
                    better = [j for j in new_kings if reputation[j] > reputation[k] + cfg.meta_margin]
                    if better:
                        mentor[k] = better[0]
            followers = [int(np.sum(mentor == k)) for k in new_kings]
            record.update(churn=churn, kings=float(len(new_kings)), depth=float(max(depth_of(i) for i in range(n))),
                          top_followers=float(followers[0]), top_king=float(new_kings[0]),
                          top_reputation=float(reputation[new_kings[0]]))
        epochs.append(record)

    def elect_clans(record: dict[str, float]) -> None:
        nonlocal kings_list, king_set
        previous = set(kings_list)
        own = own_sum / max(cfg.epoch, 1)
        deposed, bf_record = 0.0, float("nan")
        clan_raw = {k: clan_sum[k] / clan_count[k] for k in kings_list if clan_count[k] > 0}
        if not kings_list:
            kings_list = [int(k) for k in np.argsort(-own, kind="stable")[: cfg.kings]]
            for k in kings_list:
                reputation[k] = own[k]
                history[k] = deque(maxlen=cfg.score_window)
                if cultural:
                    culture[k] = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(low_x, high_x))
            if cultural and cfg.initial_cultures is not None:
                for k, traits in zip(kings_list, cfg.initial_cultures):
                    culture[k] = traits
            founders[:] = kings_list
        elif evidence:
            for k in kings_list:
                if history[k]:
                    reputation[k] = float(np.mean(history[k]))
            kings_list.sort(key=lambda k: -reputation[k])
            eligible = [k for k in kings_list if len(history[k]) >= max(2, cfg.score_window // 2)]
            if len(eligible) > 1:
                best, worst = eligible[0], eligible[-1]
                bf_record = _bayes_factor_better(np.array(history[best]), np.array(history[worst]))
                members = [i for i in range(n) if mentor[i] == best and i not in kings_list]
                if bf_record >= cfg.bayes_threshold and members:
                    prince = max(members, key=lambda i: own[i])
                    values[prince] = effective(prince)
                    if cultural:
                        culture[prince] = culture[best] + cfg.culture_mutation * rng.normal(size=3)
                        culture[prince, 2] = np.clip(culture[prince, 2], low_x, high_x)
                    reputation[prince] = reputation[worst]
                    history.pop(worst, None)
                    history[prince] = deque(maxlen=cfg.score_window)
                    kings_list[kings_list.index(worst)] = prince
                    kings_list.sort(key=lambda k: -reputation[k])
                    deposed = 1.0
        else:
            for k in kings_list:
                score = clan_sum[k] / max(clan_count[k], 1.0)
                reputation[k] = cfg.reputation_decay * reputation[k] + (1 - cfg.reputation_decay) * score
            kings_list.sort(key=lambda k: -reputation[k])
            best, worst = kings_list[0], kings_list[-1]
            members = [i for i in range(n) if mentor[i] == best and i not in kings_list]
            if len(kings_list) > 1 and members and reputation[worst] < (1 - cfg.depose_margin) * reputation[best]:
                prince = max(members, key=lambda i: own[i])
                values[prince] = effective(prince)
                if cultural:
                    culture[prince] = culture[best] + cfg.culture_mutation * rng.normal(size=3)
                    culture[prince, 2] = np.clip(culture[prince, 2], low_x, high_x)
                reputation[prince] = 0.5 * (reputation[best] + reputation[worst])
                kings_list[-1] = prince
                kings_list.sort(key=lambda k: -reputation[k])
        king_set = set(kings_list)
        ranks = np.arange(len(kings_list), 0, -1, dtype=float)
        for i in range(n):
            if i in king_set:
                mentor[i] = -1
            elif mentor[i] < 0 or mentor[i] not in king_set or rng.random() < cfg.migration:
                mentor[i] = kings_list[int(rng.choice(len(kings_list), p=ranks / ranks.sum()))]
        if cfg.meta_kings:
            top = kings_list[0]
            for k in kings_list[1:]:
                if evidence:
                    bows = len(history.get(top, ())) >= 2 and _bayes_factor_better(
                        np.array(history[top]), np.array(history.get(k, ()))) >= cfg.bayes_threshold
                else:
                    bows = reputation[top] > reputation[k] + cfg.meta_margin
                if bows:
                    mentor[k] = top
        clan[:] = [i if i in king_set else mentor[i] for i in range(n)]
        clan_sum[:] = 0.0
        clan_count[:] = 0.0
        own_sum[:] = 0.0
        churn = 1 - len(previous & king_set) / len(previous | king_set) if previous else float("nan")
        if cultural:
            top = culture[kings_list[0]]
            record.update(culture_hop_lead=float(1 / (1 + np.exp(-top[LEAD]))), culture_hop_redundant=float(1 / (1 + np.exp(-top[REDUNDANT]))),
                          culture_exclusion=float(np.exp(top[2])))
        record.update(deposed=deposed, bayes_factor=bf_record)
        record["clan_raw"] = [float(clan_raw.get(k, np.nan)) for k in founders]
        record.update(churn=churn, kings=float(len(kings_list)), depth=float(max(depth_of(i) for i in range(n))),
                      top_followers=float(np.sum(mentor == kings_list[0])), top_king=float(kings_list[0]),
                      top_reputation=float(reputation[kings_list[0]]))

    def clan_now(i: int, utility: np.ndarray) -> float:
        return float(utility[clan == clan[i]].mean() / cfg.reward_scale)

    evaluate = world.evaluate
    seen_changes = world.changes
    cma = None
    if cfg.worker == "cma":
        cma = CMAWorkers(n, dim, cfg.cma_lambda, cfg.cma_tol)
        cma.restart(np.arange(n), x, cfg.cma_sigma0)
    eval_at = np.zeros(rounds, dtype=np.int64)
    last_round = rounds
    # Redundancy accounting (Sec. 3.7 of the manuscript; read-only, no effect on the search): agent-rounds
    # spent redundant, those with an active searcher, those in a relocation stint, and episode lengths.
    red_rounds = red_active = red_recent = agent_rounds = active_rounds = 0
    red_since = np.full(n, -1)
    episodes: list[int] = []
    snap_x = snap_f = None
    last_gain = np.zeros(n, dtype=int)
    far_restarts = 0
    stall_gens = 10 + int(np.ceil(30 * dim / cma.lam)) if cma is not None else 0
    for t in range(rounds):
        if budget is not None and world.evaluations >= budget:
            last_round = t
            break
        world.maybe_change(t)
        changed_now = False
        if cfg.reevaluate == "informed" and t > 0:
            if world.changes != seen_changes:
                seen_changes = world.changes
                changed_now = True
                kept_x.clear()
                f = evaluate(x)
                if cma is not None:
                    cma.restart(np.arange(n), x, cfg.cma_sigma_change)
        elif cfg.reevaluate == "detect" and t > 0:
            best = int(np.argmax(f))
            if evaluate(x[best : best + 1])[0] != f[best]:
                changed_now = True
                kept_x.clear()
                f = evaluate(x)
                if cma is not None:
                    cma.restart(np.arange(n), x, cfg.cma_sigma_change)
        else:
            f = evaluate(x)
        evaluations += n
        if cfg.stale_period > 1 and (t % cfg.stale_period == 0 or snap_x is None or changed_now):
            snap_x, snap_f = x.copy(), f.copy()
        if partial and cfg.stale_period > 1:
            distance = np.linalg.norm(x[:, None, :] - snap_x[None, :, :], axis=2)
            np.fill_diagonal(distance, np.inf)
            sees = distance <= radius_of()[:, None]
            index = np.arange(n)
            fitter = (snap_f[None, :] > f[:, None]) | ((snap_f[None, :] == f[:, None]) & (index[None, :] < index[:, None]))
            redundant = ((distance < exclusion) & fitter & sees).any(axis=1)
        elif partial:
            distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            sees = distance <= radius_of()[:, None]
            if cfg.clan_view and kings_mode:
                roots = np.array([root_of(i) for i in range(n)])
                same = roots[:, None] == roots[None, :]
                sees = (same.astype(np.int32) @ sees.astype(np.int32)) > 0
            redundant = _redundant_seen(distance, f, exclusion, sees)
        else:
            redundant = _redundant(x, f, exclusion)
        if archive_x:
            gap_a = np.linalg.norm(x[:, None, :] - np.array(archive_x)[None, :, :], axis=2)
            seen_a = gap_a <= radius_of()[:, None] if partial else np.ones_like(gap_a, dtype=bool)
            redundant = redundant | np.any((gap_a < exclusion) & (np.array(archive_f)[None, :] > f[:, None]) & seen_a, axis=1)
        utility = np.where(redundant, 0.0, np.maximum(f, 0.0) if cfg.floor_utility else f)
        judged = redundant
        if cultural:
            if not partial:
                distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            roots = np.array([root_of(i) for i in range(n)])
            own_exclusion = exclusion * np.exp(np.where(roots >= 0, culture[roots, 2], 0.0))
            judged = _redundant_seen(distance, f, own_exclusion[:, None], sees)
        if clan_mode:
            scaled = utility / cfg.reward_scale
            sample = (utility > 0).astype(float) if cfg.clan_score == "held" else scaled
            np.add.at(clan_sum, clan, sample)
            np.add.at(clan_count, clan, 1.0)
            own_sum += scaled
            if evidence:
                np.add.at(block_sum, clan, sample)
                np.add.at(block_count, clan, 1.0)
                if t > 0 and t % block_len == 0:
                    for k in kings_list:
                        if block_count[k] > 0:
                            history[k].append(block_sum[k] / block_count[k])
                    block_sum[:] = 0.0
                    block_count[:] = 0.0

        active_now = cma.active if cma is not None else np.ones(n, dtype=bool)
        agent_rounds += n
        active_rounds += int(active_now.sum())
        red_rounds += int(judged.sum())
        red_active += int((judged & active_now).sum())
        red_recent += int((judged & (arm == HOP)).sum())
        for i in np.flatnonzero(~judged & (red_since >= 0)):
            episodes.append(t - red_since[i])
        red_since[~judged] = -1
        red_since[judged & (red_since < 0)] = t
        due = cycle_left <= 0
        if cfg.probation:
            due |= (arm == HOP) & judged & (cfg.cycle - cycle_left >= cfg.probation)
        if t > 0 and t % cfg.epoch == 0:
            close_epoch(t)
        for i in np.flatnonzero(due):
            if started[i]:
                if cfg.credit == "title":
                    gain = utility[i] - u_start[i]
                else:
                    x_cf, f_cf = x.copy(), f.copy()
                    x_cf[i] = x_start[i]
                    f_cf[i] = evaluate(x_start[i : i + 1])[0]
                    evaluations += 1
                    team = _team_utility_floor if cfg.floor_utility else _team_utility
                    if partial:
                        near_start = np.linalg.norm(x - x_start[i], axis=1) <= radius_of()[i]
                        seen = sees[i] | near_start
                        seen[i] = True
                        gain = team(x[seen], f[seen], exclusion) - team(x_cf[seen], f_cf[seen], exclusion)
                    else:
                        gain = team(x, f, exclusion) - team(x_cf, f_cf, exclusion)
                if partial and arm[i] == HOP and not redundant[i] and f[i] > 0:
                    career[i] += 1
                reward = float(np.clip(gain / cfg.reward_scale, -1.0, 1.0))
                if cfg.clan_bonus:
                    reward += cfg.clan_bonus * (clan_now(i, utility) - clan_start[i])
                next_ctx = REDUNDANT if (cfg.context and judged[i]) else LEAD
                pending[i].append([int(ctx[i]), int(arm[i]), int(mentor[i]), 0.0, 0])
                for item in pending[i]:
                    item[3] += cfg.memory_decay ** item[4] * reward
                    item[4] += 1
                while pending[i] and pending[i][0][4] > cfg.memory:
                    c, a, advisor, credit, _ = pending[i].popleft()
                    settle(i, c, a, advisor, credit, next_ctx)
                if cfg.record_cycles:
                    log.append((t, int(ctx[i]), int(arm[i]), bool(redundant[i]), reward))
            if cfg.vision_social and cfg.imitate and started[i]:
                rival = nearest_fitter(i, f, np.ones(n, dtype=bool))
                holds = f[i] > 0 or not cfg.hold_positive
                if holds and rival >= 0:
                    middle = evaluate(((x[i] + x[rival]) / 2)[None, :])[0]
                    evaluations += 1
                    holds = middle < min(f[i], f[rival])
                hold_score[i] = cfg.vision_memory * hold_score[i] + (1 - cfg.vision_memory) * float(holds)
                other = int(vision_rng.integers(n - 1))
                other += other >= i
                if hold_score[other] > hold_score[i]:
                    radius[i] = np.clip(radius[other] * np.exp(cfg.vision_noise * vision_rng.normal()), *cfg.vision_init)
            started[i] = True
            is_red = bool(judged[i])
            if cfg.hill_valley and is_red:
                rival = nearest_fitter(i, f, sees[i])
                if rival >= 0:
                    middle = evaluate(((x[i] + x[rival]) / 2)[None, :])[0]
                    evaluations += 1
                    is_red = middle >= min(f[i], f[rival])
            ctx[i] = REDUNDANT if (cfg.context and is_red) else LEAD
            arm[i] = choose(i, is_red)
            u_start[i] = utility[i]
            if cfg.clan_bonus:
                clan_start[i] = clan_now(i, utility)
            x_start[i] = x[i]
            cycle_left[i] = cfg.cycle
            cycles += 1
            if arm[i] == HOP:
                hops += 1
                if red_since[i] >= 0:
                    episodes.append(t - red_since[i] + 1)
                    red_since[i] = -1
                if is_red:
                    redundant_hops += 1
                    gap = np.linalg.norm(x - x[i], axis=1)
                    fitter = (f > f[i]) | ((f == f[i]) & (np.arange(n) < i))
                    rivals = np.flatnonzero((gap < exclusion) & fitter & sees[i])
                    cone = lambda pts: np.argmax(world.heights - world.widths * np.linalg.norm(
                        pts[:, None, :] - world.centers[None, :, :], axis=2), axis=1)
                    if hasattr(world, "centers") and len(rivals) and not np.any(cone(x[rivals]) == cone(x[i : i + 1])[0]):
                        false_hops += 1
                if partial:
                    seen = sees[i].copy()
                    seen[i] = False
                    pool = snap_x if cfg.stale_period > 1 else x
                    others = pool[seen] if seen.any() else np.empty((0, dim))
                else:
                    others = np.delete(x, i, axis=0)
                candidates = rng.uniform(wc.low, wc.high, (cfg.destinations if cfg.destination == "far" else 1, dim))
                if len(others):
                    gaps = np.linalg.norm(candidates[:, None, :] - others[None, :, :], axis=2).min(axis=1)
                    x[i] = candidates[int(np.argmax(gaps))]
                else:
                    x[i] = candidates[0]
                step[i] = cfg.hop_step
                f[i] = evaluate(x[i : i + 1])[0]
                evaluations += 1
                if cma is not None:
                    cma.restart(i, x[i], cfg.cma_sigma0)

        if cma is not None:
            if not cma.active.any():
                cma.restart(np.arange(n), x, cfg.cma_sigma0)
            agents, points, values_ = cma.generation(rng, evaluate, wc.low, wc.high, wc.high - wc.low)
            evaluations += values_.size
            if len(agents):
                top = values_.argmax(axis=1)
                best_v = values_[np.arange(len(agents)), top]
                gain = best_v >= f[agents]
                last_gain[agents[best_v > f[agents] + 1e-12]] = t
                x[agents[gain]] = points[np.arange(len(agents)), top][gain]
                f[agents[gain]] = best_v[gain]
            if cfg.restart_far and not cfg.archive:
                stalled = (cma.gen >= stall_gens) & (t - last_gain >= stall_gens)
                for i in np.flatnonzero(~cma.active | stalled):
                    if cfg.restart_keep:
                        kept_x.append(x[i].copy())
                    others = np.delete(x, i, axis=0)
                    if cfg.restart_keep and kept_x:
                        others = np.vstack([others] + kept_x)
                    candidates = rng.uniform(wc.low, wc.high, (cfg.destinations, dim))
                    gaps = np.linalg.norm(candidates[:, None, :] - others[None, :, :], axis=2).min(axis=1)
                    x[i] = candidates[int(np.argmax(gaps))]
                    f[i] = evaluate(x[i : i + 1])[0]
                    evaluations += 1
                    cma.restart(i, x[i], cfg.cma_sigma0)
                    last_gain[i] = t
                    far_restarts += 1
            if cfg.archive:
                for i in np.flatnonzero(~cma.active):
                    keep = True
                    if archive_x:
                        gaps = np.linalg.norm(np.array(archive_x) - x[i], axis=1)
                        j = int(np.argmin(gaps))
                        if gaps[j] < exclusion:
                            middle = evaluate(((x[i] + archive_x[j]) / 2)[None, :])[0]
                            evaluations += 1
                            if middle >= min(f[i], archive_f[j]):  # same hill: keep the better point
                                keep = False
                                if f[i] > archive_f[j]:
                                    archive_x[j], archive_f[j] = x[i].copy(), float(f[i])
                    if keep:
                        archive_x.append(x[i].copy())
                        archive_f.append(float(f[i]))
                    retirements += 1
                    others = np.concatenate([np.delete(x, i, axis=0)] + [np.array(archive_x)])
                    candidates = rng.uniform(wc.low, wc.high, (cfg.destinations, dim))
                    gaps = np.linalg.norm(candidates[:, None, :] - others[None, :, :], axis=2).min(axis=1)
                    x[i] = candidates[int(np.argmax(gaps))]
                    f[i] = evaluate(x[i : i + 1])[0]
                    evaluations += 1
                    cma.restart(i, x[i], cfg.cma_sigma0)
                    cycle_left[i] = 0
                    started[i] = False
        else:
            candidate = np.clip(x + step[:, None] * rng.normal(size=x.shape), wc.low, wc.high)
            fc = evaluate(candidate)
            evaluations += n
            better = fc >= f
            x[better] = candidate[better]
            f = np.where(better, fc, f)
            step = np.clip(np.where(better, step * cfg.step_up, step * cfg.step_down), *cfg.step_bounds)
        cycle_left -= 1

        m = world.measure(x, cfg.radius)
        if census is not None:
            visible = world.visible()
            near = np.linalg.norm(x[:, None, :] - world.centers[None, visible, :], axis=2) <= cfg.radius
            on = near.any(axis=1)
            red = _redundant(x, world.values(x), exclusion)
            by_lead = near[~red].any(axis=0)
            census[t] = (np.sum(on & ~red), np.sum(~on & ~red), np.sum(on & red), np.sum(~on & red),
                         by_lead.sum(), np.sum(near[red].any(axis=0) & ~by_lead))
        coverage[t], occupancy[t], error[t] = m.coverage, m.occupancy, m.error
        eval_at[t] = world.evaluations

    rounds = last_round
    if hasattr(world, "finish"):
        world.finish()
    coverage, occupancy, error, eval_at = coverage[:rounds], occupancy[:rounds], error[:rounds], eval_at[:rounds]
    if budget is not None:
        window = eval_at >= (wc.change_frequency or 0)
        window = window if window.any() else np.ones(rounds, dtype=bool)
    else:
        window = slice(min(cfg.burn_in, rounds - 1), rounds)
    result = PeakHoppingResult(coverage, occupancy, error, hops / max(cycles, 1), evaluations, world.changes,
                               values.mean(axis=0), cfg.burn_in)
    result.epochs = epochs
    if partial:
        result.summary_extra = {"vision_mean": float(radius_of().mean()), "vision_max_agent": float(radius_of().max()),
                                "career_mean": float(career.mean()), "vision_median": float(np.median(radius_of()))}
    result.cycles = log
    result.census = census
    result.summary_extra["false_hop_share"] = false_hops / redundant_hops if redundant_hops else float("nan")
    episodes += [rounds - s for s in red_since if s >= 0]
    result.summary_extra["far_restarts"] = far_restarts
    result.summary_extra.update(red_share=red_rounds / max(agent_rounds, 1), red_active_share=red_active / max(active_rounds, 1),
                                red_recent_share=red_recent / red_rounds if red_rounds else float("nan"),
                                episode_mean=float(np.mean(episodes)) if episodes else float("nan"),
                                episode_p95=float(np.percentile(episodes, 95)) if episodes else float("nan"),
                                episode_max=float(np.max(episodes)) if episodes else float("nan"))
    if cfg.archive:
        result.summary_extra["archive_size"] = len(archive_x)
        result.summary_extra["retirements"] = retirements
    result.summary = {
        "coverage": float(coverage[window].mean()),
        "occupancy": float(occupancy[window].mean()),
        "error": float(error[window].mean()),
        "hop_share": result.hop_share,
    }
    if budget is not None:
        if hasattr(world, "offline_error"):
            result.summary["offline_error"] = world.offline_error
    result.evaluations = world.evaluations
    return result
