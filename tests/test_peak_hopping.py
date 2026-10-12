import numpy as np

from aumo.algorithms.peak_hopping import PeakHoppingConfig, run_peak_hopping
from aumo.problems.moving_peaks import MovingPeaksWorld, PeaksConfig


def test_agents_on_every_visible_peak_give_full_coverage():
    world = MovingPeaksWorld(PeaksConfig(), seed=0)
    visible = world.visible()
    assert visible.any()
    m = world.measure(world.centers[visible], radius=0.5)
    assert m.coverage == 1.0 and m.occupancy == 1.0 and abs(m.error) < 1e-9


def test_world_changes_only_on_period_and_moves_by_severity():
    world = MovingPeaksWorld(PeaksConfig(change_probability=1.0, period=5), seed=1)
    before = world.centers.copy()
    assert not world.maybe_change(3)
    assert world.maybe_change(5)
    assert np.allclose(np.linalg.norm(world.centers - before, axis=1), 1.0, atol=1e-6) or world.centers.min() <= 1.0


def test_peak_hopping_is_deterministic_and_policies_behave():
    world = PeaksConfig()
    a = run_peak_hopping(world, PeakHoppingConfig(burn_in=50), seed=3, rounds=200)
    b = run_peak_hopping(world, PeakHoppingConfig(burn_in=50), seed=3, rounds=200)
    assert a.summary == b.summary
    stay = run_peak_hopping(world, PeakHoppingConfig(policy="fixed:stay", burn_in=50), seed=3, rounds=200)
    assert stay.hop_share == 0.0
    hop = run_peak_hopping(world, PeakHoppingConfig(policy="fixed:hop", burn_in=50), seed=3, rounds=200)
    assert hop.hop_share == 1.0
    assert 0.0 <= a.summary["coverage"] <= 1.0


def test_kings_mode_elects_kings_and_records_epochs():
    cfg = PeakHoppingConfig(credit="counterfactual", floor_utility=True, value_sharing="kings", memory=2, meta_kings=True,
                            epoch=50, burn_in=50)
    a = run_peak_hopping(PeaksConfig(), cfg, seed=2, rounds=300)
    b = run_peak_hopping(PeaksConfig(), cfg, seed=2, rounds=300)
    assert a.summary == b.summary
    assert len(a.epochs) == 5
    assert all(e["kings"] == cfg.kings for e in a.epochs)
    assert all(1 <= e["depth"] <= 2 for e in a.epochs)


def test_clan_reputation_keeps_king_count_and_is_deterministic():
    cfg = PeakHoppingConfig(credit="counterfactual", floor_utility=True, value_sharing="kings", reputation="clan",
                            clan_bonus=1.0, meta_kings=True, memory=2, epoch=50, burn_in=50)
    a = run_peak_hopping(PeaksConfig(), cfg, seed=4, rounds=400)
    assert a.summary == run_peak_hopping(PeaksConfig(), cfg, seed=4, rounds=400).summary
    assert all(e["kings"] == cfg.kings and e["depth"] <= 2 for e in a.epochs)


def test_partial_vision_and_culture_run_deterministically():
    cfg = PeakHoppingConfig(credit="counterfactual", floor_utility=True, value_sharing="kings", reputation="clan",
                            policy="culture", kings=4, vision=5.0, clan_view=True, memory=2, epoch=50, burn_in=50)
    a = run_peak_hopping(PeaksConfig(), cfg, seed=5, rounds=400)
    assert a.summary == run_peak_hopping(PeaksConfig(), cfg, seed=5, rounds=400).summary
    assert 5.0 <= a.summary_extra["vision_mean"] <= cfg.vision_max
    assert all(0.0 <= e["culture_hop_redundant"] <= 1.0 for e in a.epochs)
    blind = run_peak_hopping(PeaksConfig(), PeakHoppingConfig(policy="rule", vision=5.0, vision_max=5.0, burn_in=50), seed=5, rounds=200)
    assert blind.summary_extra["vision_max_agent"] == 5.0


def test_social_vision_and_hill_valley_stay_in_bounds():
    rule = PeakHoppingConfig(policy="rule", cycle=20, burn_in=50)
    a = run_peak_hopping(PeaksConfig(), PeakHoppingConfig(policy="rule", cycle=20, burn_in=50, vision=5.0, vision_social=True), 6, 400)
    assert 0.5 <= a.summary_extra["vision_median"] <= 100.0
    b = run_peak_hopping(PeaksConfig(), PeakHoppingConfig(policy="rule", cycle=20, burn_in=50, hill_valley=True), 6, 400)
    assert b.evaluations > run_peak_hopping(PeaksConfig(), rule, 6, 400).evaluations


def test_evaluation_based_world_changes_on_schedule_and_tracks_offline_error():
    from aumo.problems.moving_peaks import scenario2
    world = MovingPeaksWorld(scenario2(change_frequency=10), seed=0)
    assert np.all(world.heights == 50.0)
    world.evaluate(np.full((25, 5), 50.0))
    assert world.changes == 2 and world.evaluations == 25
    assert world.offline_error >= 0.0


def test_budget_mode_and_mqso_respect_the_budget():
    from aumo.algorithms.mqso import run_mqso
    from aumo.problems.moving_peaks import scenario2
    world = scenario2(change_frequency=1000)
    r = run_peak_hopping(world, PeakHoppingConfig(policy="rule", reevaluate="detect", vision=5.0, vision_social=True), 1, budget=5000)
    assert 5000 <= r.evaluations < 5100 and "offline_error" in r.summary
    m = run_mqso(world, 1, 5000)
    assert 5000 <= m["evaluations"] < 5300 and m["changes"] >= 4


def test_dmmop_port_optima_and_oracle_peak_ratio():
    from aumo.problems.dmmop import DMMOPWorld
    for problem in (1, 5, 15):
        world = DMMOPWorld(problem, max_env=3)
        assert np.all(world.values(world.o) >= world.of[0] - 1e-9)
        world.reporter = lambda: world.o.copy()
        for env in range(3):
            world._record()
            if env < 2:
                world._advance()
        found, total = world.peak_counts()
        assert np.all(found == total)


def test_hmso_and_mqso_run_on_both_worlds():
    from aumo.algorithms.hmso import run_hmso
    from aumo.algorithms.mqso import run_mqso
    from aumo.problems.dmmop import DMMOPWorld
    from aumo.problems.moving_peaks import scenario2
    r = run_hmso(scenario2(change_frequency=1000), 0, 4000)
    assert 4000 <= r["evaluations"] < 4300 and r["changes"] >= 3
    world = DMMOPWorld(1, max_env=2)
    run_mqso(world, 0)
    assert world.exhausted and len(world.records) == 2


def test_dmmop_replays_dumped_instances(tmp_path):
    from aumo.problems.dmmop import DMMOPWorld, export_instances

    rng = np.random.default_rng(0)
    for problem in (1, 5, 15, 16):
        export_instances(problem, tmp_path, max_env=6)
        native, replay = DMMOPWorld(problem, max_env=6), DMMOPWorld(problem, max_env=6, instances=tmp_path)
        for e in range(6):
            if e:
                native._advance()
                replay._advance()
            pts = rng.uniform(-5, 5, (20, native.dim))
            assert np.allclose(native.values(pts), replay.values(pts))
            assert np.allclose(native.o, replay.o)
