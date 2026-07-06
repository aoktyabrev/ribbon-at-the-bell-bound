"""Протокол анализа (SPEC §4): счётчики → контроли, MLE-фиты, AIC, bootstrap,
гармоники. Чистый numpy (без scipy — оптимизация p одномерная, сеткой + золотым
сечением).

Порядок ветвей всюду: [pp, pm, mp, mm]. Произведение s·t: [+1, −1, −1, +1].
"""

import math

import numpy as np

ST_SIGN = np.array([+1, -1, -1, +1])  # s*t для [pp, pm, mp, mm]
_PROB_FLOOR = 1e-12


def _two_sided_p(z):
    """Двусторонний p-value стандартной нормали: P(|Z| > z) = erfc(z/√2)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _bonferroni_z(n, alpha=0.05):
    """Порог |z| для контроля семейной ошибки на n сравнениях (max-статистика):
    решаем 1 − (1 − erfc(z/√2))^n = alpha бисекцией по z."""
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fwe = 1.0 - (1.0 - _two_sided_p(mid)) ** n
        if fwe > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
#  Наблюдаемые из счётчиков
# --------------------------------------------------------------------------- #
def E_from_counts(counts):
    """E(θ) = <s·t> из счётчиков ветвей. counts формы (..., 4)."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(-1)
    return (counts * ST_SIGN).sum(-1) / n


def marginals(counts):
    """P(s=+), P(t=+) из счётчиков (SPEC §4.2, no-signaling)."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(-1)
    p_s = (counts[..., 0] + counts[..., 1]) / n  # pp + pm
    p_t = (counts[..., 0] + counts[..., 2]) / n  # pp + mp
    return p_s, p_t


def marginal_sigma(p_hat, n):
    """Биномиальная σ доли: sqrt(p(1-p)/n)."""
    return np.sqrt(np.maximum(p_hat * (1 - p_hat), 0.0) / n)


def check_controls(counts, thetas):
    """Обязательные контроли SPEC §4.2 на одном прогоне.

    Возврат dict с полями по каждой θ: маргиналы ±3σ и ±-симметрия ветвей.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(-1)
    p_s, p_t = marginals(counts)
    sig_s, sig_t = marginal_sigma(p_s, n), marginal_sigma(p_t, n)

    # ±-симметрия: P(pp)≈P(mm), P(pm)≈P(mp) в пределах пуассоновского шума.
    n_pp, n_pm, n_mp, n_mm = counts[..., 0], counts[..., 1], counts[..., 2], counts[..., 3]
    sym_same = np.abs(n_pp - n_mm) / np.sqrt(np.maximum(n_pp + n_mm, 1.0))  # в σ
    sym_opp = np.abs(n_pm - n_mp) / np.sqrt(np.maximum(n_pm + n_mp, 1.0))

    marg_ok = (np.abs(p_s - 0.5) < 3 * sig_s) & (np.abs(p_t - 0.5) < 3 * sig_t)
    sym_ok = (sym_same < 3.0) & (sym_opp < 3.0)
    return {
        "thetas": np.asarray(thetas, dtype=np.float64),
        "n": n,
        "p_s": p_s,
        "p_t": p_t,
        "sig_s": sig_s,
        "sig_t": sig_t,
        "marg_ok": marg_ok,
        "sym_same_sigma": sym_same,
        "sym_opp_sigma": sym_opp,
        "sym_ok": sym_ok,
        "all_marg_ok": bool(np.all(marg_ok)),
        "all_sym_ok": bool(np.all(sym_ok)),
    }


def reproducibility(counts_a, counts_b, alpha=0.05):
    """Воспроизводимость на 2 сидах через max-статистику по всем точкам θ.

    Поточечный 3σ на 25 сравнениях завышает частоту ложных срабатываний
    (~1−0.997^25 ≈ 7%). Контролируем семейную ошибку: считаем глобальный
    p-value max|z| и сравниваем с порогом Бонферрони z*(n, alpha).
    """
    Ea, Eb = E_from_counts(counts_a), E_from_counts(counts_b)
    na = np.asarray(counts_a, dtype=np.float64).sum(-1)
    nb = np.asarray(counts_b, dtype=np.float64).sum(-1)
    # σ(E) ≈ sqrt((1-E²)/n); складываем дисперсии двух сидов.
    sig = np.sqrt((1 - Ea ** 2) / na + (1 - Eb ** 2) / nb)
    dE = Ea - Eb
    z = dE / np.where(sig > 0, sig, 1.0)
    n = len(dE)
    max_z = float(np.max(np.abs(z)))
    z_thresh = _bonferroni_z(n, alpha)
    global_p = 1.0 - (1.0 - _two_sided_p(max_z)) ** n
    return {
        "dE": dE,
        "z": z,
        "max_abs_dE": float(np.max(np.abs(dE))),
        "max_z": max_z,
        "n_points": n,
        "z_thresh": z_thresh,      # порог max-статистики при данном n и alpha
        "global_p": global_p,      # семейный p-value наблюдённого max|z|
        "passes": bool(max_z < z_thresh),
    }


# --------------------------------------------------------------------------- #
#  Модели-соперники (SPEC §1, §4.3)
# --------------------------------------------------------------------------- #
# Конвенция знака (директива архитектора, R1_theory_prediction §5): упругая лента
# ФЕРРО (E_sim(0)=+1), семейства SPEC §1 записаны в СИНГЛЕТНОЙ конвенции (E(0)=−1).
# Сырые счётчики хранятся в ферро-конвенции; ПЕРЕД фитом делаем точный глобальный
# флип t̃=−t (singlet_counts), затем фитим семейства «как есть». Перебор знака
# (бывший fit_chord_signed) УДАЛЁН.


def singlet_counts(counts):
    """Глобальный флип t̃=−t: ферро-счётчики → синглетная конвенция.

    t̃=−t переставляет ветви [pp,pm,mp,mm] → [pm,pp,mm,mp] ⇒ E_singlet = −E_ferro.
    """
    counts = np.asarray(counts, dtype=np.float64)
    return counts[..., [1, 0, 3, 2]]


def chord_probs(theta, p):
    """P[pp,pm,mp,mm] для закона хорды P ∝ |s·a − t·b|^p (SPEC §2.6, синглет).

    w_same = |sin(θ/2)|^p (ветви pp,mm), w_opp = |cos(θ/2)|^p (ветви pm,mp).
    """
    theta = np.asarray(theta, dtype=np.float64)
    s2 = np.sin(theta / 2) ** 2
    c2 = np.cos(theta / 2) ** 2
    ws = s2 ** (p / 2)
    wo = c2 ** (p / 2)
    Z = 2 * (ws + wo)
    return np.stack([ws / Z, wo / Z, wo / Z, ws / Z], axis=-1)


def saw_probs(theta):
    """P[pp,pm,mp,mm] для пилы E = 2θ/π − 1 (SPEC §1, синглетная конвенция)."""
    theta = np.asarray(theta, dtype=np.float64)
    E = 2 * theta / np.pi - 1
    p_same = (1 + E) / 4  # на каждую из pp, mm
    p_opp = (1 - E) / 4   # на каждую из pm, mp
    return np.stack([p_same, p_opp, p_opp, p_same], axis=-1)


def _loglik(counts, probs):
    """Мультиномиальный log-lik Σ count·log(P), суммарно по θ и ветвям."""
    probs = np.clip(probs, _PROB_FLOOR, 1.0)
    return float(np.sum(counts * np.log(probs)))


def loglik_chord(counts, thetas, p):
    return _loglik(counts, chord_probs(thetas, p))


def loglik_saw(counts, thetas):
    return _loglik(counts, saw_probs(thetas))


def _golden_max(f, lo, hi, tol=1e-5, iters=80):
    """Максимум унимодальной f на [lo, hi] золотым сечением. Возврат (x*, f*)."""
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        if abs(b - a) < tol:
            break
    x = 0.5 * (a + b)
    return float(x), f(x)


def fit_chord_p(counts, thetas, p_lo=0.2, p_hi=6.0):
    """MLE p закона хорды (синглетная конвенция): сетка + золотое сечение.

    ВНИМАНИЕ: ожидает счётчики уже в СИНГЛЕТНОЙ конвенции (см. singlet_counts).
    Возврат (p_hat, loglik).
    """
    counts = np.asarray(counts, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    grid = np.linspace(p_lo, p_hi, 60)
    lls = [loglik_chord(counts, thetas, p) for p in grid]
    i = int(np.argmax(lls))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    return _golden_max(lambda p: loglik_chord(counts, thetas, p), lo, hi)


def aic(loglik, k):
    """AIC = 2k − 2·lnL."""
    return 2 * k - 2 * loglik


def compare_models(counts, thetas):
    """Фит пила/хорда-p/хорда-p=2 + AIC (SPEC §4.3), СИНГЛЕТНАЯ конвенция.

    Ожидает счётчики уже в синглетной конвенции (вызывающий делает singlet_counts).
    """
    counts = np.asarray(counts, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)

    p_hat, ll_chord = fit_chord_p(counts, thetas)
    ll_saw = loglik_saw(counts, thetas)
    ll_p2 = loglik_chord(counts, thetas, 2.0)
    return {
        "p_hat": p_hat,
        "aic_saw": aic(ll_saw, 0),
        "aic_chord_p": aic(ll_chord, 1),
        "aic_chord_p2": aic(ll_p2, 0),
        "loglik": {"saw": ll_saw, "chord_p": ll_chord, "chord_p2": ll_p2},
    }


def bootstrap_p(counts, thetas, n_boot=1000, seed=0):
    """Bootstrap CI на p (SPEC §4.3, синглетная конвенция): ресемпл мультиномиала.

    Ожидает счётчики уже в синглетной конвенции (вызывающий делает singlet_counts).
    """
    counts = np.asarray(counts, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = counts.sum(-1)
    props = counts / n[:, None]

    ps = np.empty(n_boot)
    for b in range(n_boot):
        resampled = np.stack(
            [rng.multinomial(int(n[i]), props[i]) for i in range(len(thetas))]
        )
        ps[b], _ = fit_chord_p(resampled, thetas)
    return {
        "p_mean": float(np.mean(ps)),
        "ci95": (float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))),
        "samples": ps,
    }


def harmonics(thetas, E):
    """Гармонический анализ E(θ) = c0 + A1·cosθ + A3·cos3θ (SPEC §4.4)."""
    thetas = np.asarray(thetas, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    design = np.stack([np.ones_like(thetas), np.cos(thetas), np.cos(3 * thetas)], axis=1)
    coef, *_ = np.linalg.lstsq(design, E, rcond=None)
    return {"c0": float(coef[0]), "A1": float(coef[1]), "A3": float(coef[2])}


def synthetic_counts(thetas, p, n_per_point, seed=0):
    """Синтетические счётчики закона хорды при известном p (для тестов §7)."""
    rng = np.random.default_rng(seed)
    thetas = np.asarray(thetas, dtype=np.float64)
    probs = chord_probs(thetas, p)
    return np.stack([rng.multinomial(n_per_point, probs[i]) for i in range(len(thetas))])
