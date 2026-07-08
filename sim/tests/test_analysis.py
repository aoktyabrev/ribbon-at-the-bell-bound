"""Тесты анализа (SPEC §7): MLE восстанавливает известный p; пила отличима."""

import numpy as np

from ribbon_sim import analysis


THETAS = np.radians(np.arange(0, 180.001, 7.5))


def test_E_from_counts_matches_probs():
    # синтетика при p=2 ⇒ E(θ) ≈ −cos θ
    counts = analysis.synthetic_counts(THETAS, p=2.0, n_per_point=2_000_000, seed=0)
    E = analysis.E_from_counts(counts)
    assert np.allclose(E, -np.cos(THETAS), atol=5e-3)


def test_chord_probs_normalized_and_symmetric():
    probs = analysis.chord_probs(THETAS, 1.7)
    assert np.allclose(probs.sum(-1), 1.0, atol=1e-12)
    # ±-симметрия: P(pp)=P(mm), P(pm)=P(mp)
    assert np.allclose(probs[:, 0], probs[:, 3])
    assert np.allclose(probs[:, 1], probs[:, 2])


def test_mle_recovers_known_p():
    for p_true in [1.0, 1.5, 2.0, 3.0]:
        counts = analysis.synthetic_counts(THETAS, p=p_true, n_per_point=500_000, seed=1)
        p_hat, _ = analysis.fit_chord_p(counts, THETAS)
        assert abs(p_hat - p_true) < 0.05, f"p_true={p_true}, p_hat={p_hat}"


def test_bootstrap_ci_covers_truth():
    p_true = 1.5
    counts = analysis.synthetic_counts(THETAS, p=p_true, n_per_point=100_000, seed=2)
    p_hat, _ = analysis.fit_chord_p(counts, THETAS)
    boot = analysis.bootstrap_p(counts, THETAS, n_boot=300, seed=3)
    lo, hi = boot["ci95"]
    # Percentile-bootstrap CI охватывает оценку по данным p_hat...
    assert lo <= p_hat <= hi
    # ...а p_hat близок к истине (при таком n систематики нет)
    assert abs(p_hat - p_true) < 0.03, f"p_hat={p_hat}"
    assert (hi - lo) < 0.2  # CI разумно узок


def test_saw_distinguished_from_chord():
    # данные — пила: AIC пилы должен быть заметно лучше (меньше), чем хорда-p=2
    rng = np.random.default_rng(4)
    saw_p = analysis.saw_probs(THETAS)
    counts = np.stack([rng.multinomial(300_000, saw_p[i]) for i in range(len(THETAS))])
    cmp = analysis.compare_models(counts, THETAS)
    assert cmp["aic_saw"] < cmp["aic_chord_p2"] - 10
    # и наоборот: данные-хорда p=2 лучше описываются хордой, чем пилой
    counts2 = analysis.synthetic_counts(THETAS, p=2.0, n_per_point=300_000, seed=5)
    cmp2 = analysis.compare_models(counts2, THETAS)
    assert cmp2["aic_chord_p"] < cmp2["aic_saw"] - 10
    assert abs(cmp2["p_hat"] - 2.0) < 0.05


def test_harmonics_recovers_cos():
    # E = −cos θ ⇒ A1 ≈ −1, A3 ≈ 0
    E = -np.cos(THETAS)
    h = analysis.harmonics(THETAS, E)
    assert abs(h["A1"] + 1.0) < 1e-6
    assert abs(h["A3"]) < 1e-6
    assert abs(h["c0"]) < 1e-6


def test_reproducibility_maxstat():
    # два независимых сида из ОДНОГО распределения ⇒ контроль проходит (классика, degen OFF)
    ca = analysis.synthetic_counts(THETAS, p=2.0, n_per_point=200_000, seed=10)
    cb = analysis.synthetic_counts(THETAS, p=2.0, n_per_point=200_000, seed=11)
    rep = analysis.reproducibility(ca, cb)
    assert rep["passes"]
    assert rep["z_thresh"] > 3.0  # Бонферрони на 25 сравнениях строже 3σ
    assert rep["global_p"] > 0.05
    assert rep["n_degenerate"] == 0
    # грубо сдвинутый второй сид (другой p) — контроль обязан провалиться
    cc = analysis.synthetic_counts(THETAS, p=1.0, n_per_point=200_000, seed=12)
    rep_bad = analysis.reproducibility(ca, cc)
    assert not rep_bad["passes"]


def test_reproducibility_degenerate_classification():
    # classify_degenerate (R4b): изолированные сид-флипы (|ΔE|>0.3) → DEGENERATE, не провал
    ca = analysis.synthetic_counts(THETAS, p=2.0, n_per_point=200_000, seed=20)
    cb = ca.copy()
    # инвертируем ветви (s·t → −) в ДВУХ точках: имитируем сид-флип конкурирующих минимумов
    for j in (4, 16):
        cb[j] = ca[j][[1, 0, 3, 2]]  # своп pp↔pm, mp↔mm ⇒ E меняет знак
    rep = analysis.reproducibility(ca, cb, classify_degenerate=True)
    assert rep["n_degenerate"] == 2, rep["n_degenerate"]
    assert rep["passes"]  # вне вырожденных точек — воспроизводимо
    # без classify — те же точки валят контроль
    rep_off = analysis.reproducibility(ca, cb, classify_degenerate=False)
    assert not rep_off["passes"]


def test_marginals_half_for_symmetric():
    counts = analysis.synthetic_counts(THETAS, p=2.0, n_per_point=500_000, seed=6)
    p_s, p_t = analysis.marginals(counts)
    assert np.allclose(p_s, 0.5, atol=5e-3)
    assert np.allclose(p_t, 0.5, atol=5e-3)
