# C3-L / L3 — аналитическая нота: Йенсен-зазор ⇒ телеграф;
# Цирельсон как следствие no-signaling (без information causality).
Статус: этап 0, аналитика (урожай). Батарея-дополнение — ПОСЛЕ prereg-коммита.
Обобщает addendum2 (C3-B): там доказан ТОЧНЫЙ ноль зазора в p=2;
здесь — строгая положительность зазора для ВСЕГО неаффинного хвоста.

## 0. Обозначения (наследует C3-B / battery_c3b_hardening.py)
- Правило считывания: f(c) — вероятность исхода «+» у детектора при
  проекции c = m·b̂ ∈ [−1,1] (m — Bloch-вектор ансамбля, суб-единичный).
- Семейство хорды: f_p(c) = |1+c|^{p/2} / (|1+c|^{p/2}+|1−c|^{p/2}).
- **Нечётность (антисимметрия исхода):** f(−c) = 1 − f(c). Выполнена ∀ f_p.
  Следствие: f(0)=½, и на c-симметричном ансамбле маргинал = ½ (нет
  сигналинга ПЕРВОГО порядка — остаётся зазор Йенсена, второй порядок).
- **Внутренний стиринг (F2):** источник со смещённой мерой μ_χ порождает
  при настройке Алисы A и её исходе s условный под-ансамбль конца Боба
  с средним вектором m_s^A, весом P(s|A). Закон полной вероятности:
  Σ_s P(s|A) m_s^A = ⟨λ⟩ — ОДИН И ТОТ ЖЕ ∀A (равные средние).
- **D(χ)>0 (диверсность):** при равных средних две настройки A,A′ дают
  РАЗНЫЕ высшие моменты распределения c=m_s^A·b̂ (mean-preserving spread).
  Это и есть внутренне порождённая стиринг-диверсность (не импорт).

## 1. Лемма (Йенсен-зазор ⇒ телеграф)
**Утв.** Маргинал Боба при настройке Алисы A есть
  P_Bob(+|A) = Σ_s P(s|A) f(c_s^A),  c_s^A = m_s^A·b̂,
случайная величина c (по s) с ФИКСИРОВАННЫМ средним μ_b = ⟨λ⟩·b̂ ∀A,
но с A-зависимым распределением (D>0). Определим зазор
  J(A) = P_Bob(+|A) − f(μ_b) = E_A[f(c)] − f(E_A[c]).
No-signaling ⇔ J(A) = J(A′) ∀A,A′.

**Лемма.** f аффинна на [−1,1] ⟺ J(A) ≡ 0 на ВСЕХ внутренних ансамблях
(⟺ no-signaling ∀ конфигураций). Эквивалентно: если f НЕ аффинна, то
∃ пара внутренних настроек с D>0, для которой J(A) ≠ J(A′) ⇒ телеграф.

**Док-во.**
(⇐, аффинность ⇒ ноль) f(c)=α+βc ⇒ E_A[f(c)] = α+βE_A[c] = α+βμ_b =
f(μ_b) ∀A ⇒ J≡0. (Это ровно алгебраический ноль addendum2 при p=2:
f₂(c)=(1+c)/2 аффинна.)
(⇒, неаффинность ⇒ сигналинг) Классическая характеризация: E[f(X)]
инвариантно ко ВСЕМ mean-preserving spread ⟺ f аффинна. Если f не
аффинна, ∃ a<b и середина μ=(a+b)/2 с f(μ) ≠ ½(f(a)+f(b)) (провал
midpoint-аффинности). Возьмём A с двухточечным {a,b} (равновес, mean=μ,
дисперсия>0) и A′ с меньшим разбросом при том же μ: E_A[f]−E_{A′}[f] =
[E_A f − f(μ)] − [E_{A′} f − f(μ)] ≠ 0 в общем положении. D>0 гарантирует,
что конструкция РЕАЛИЗУЕТ две различные разбросности при равном среднем
(это и есть определение D>0) ⇒ J(A) ≠ J(A′) ⇒ P_Bob(+|A) ≠ P_Bob(+|A′). ∎

**NB (сила/оговорка):** лемма даёт СУЩЕСТВОВАНИЕ сигнализирующей пары для
любой неаффинной f. Отсутствие СЛУЧАЙНОГО зануления J(A)−J(A′) на ДВУХ
КОНКРЕТНЫХ разбросах, реализуемых канонической F2, — предмет батареи
(§3): для f_p зазор строго монотонен по |p−2| (кривизна f_p меняет знак
в p=2), численно + аналитикой рядом.

## 2. Следствие: единственность p=2 и Цирельсон из no-signaling
**2.1 Единственность p=2 (в нечётном классе).** Пересечение
{аффинна} ∩ {нечётна f(−c)=1−f(c)} ∩ {f(1)=1 нормировка} = единственная
f(c)=(1+c)/2 = f_{p=2}. ⇒ no-signaling на внутренних ансамблях с D>0
выделяет p=2 ЕДИНСТВЕННО (лемма §1 + нечётность + нормировка).

**2.2–2.3 ФИНАЛЬНЫЙ ТЕКСТ СЕКЦИИ (архитектор, дословно; в статью 3).**
Сырьё чисел: C3L_L3.json (9567ddb), C3L_L2.json (49e8c1b).

> The no-signaling argument stratifies into two layers, and stating
> them separately is what makes the claim exact. The RULE-SELECTION
> layer is mechanism-independent: whenever the source measure
> generates steering diversity (D(χ)>0), any non-affine odd readout
> produces a strictly positive Jensen gap and hence a telegraph;
> affinity, oddness and normalization intersect in the single rule
> f₂=(1+c)/2, so no-signaling selects p=2 in every layer where
> steering exists. The VALUE layer is mechanism-dependent: what S the
> selected rule attains is a property of how the pair is resolved.
> Shared-λ (factorizable) resolution is Bell-bounded — our internal
> family sits at S=√2 at χ=0.5, and its ceiling is the 1/3-cosine
> point E=cosθ/3, the maximal isotropic-cosine statistics available
> to local linear readout of a hidden vector. Joint resolution of the
> chord law attains S=2√2 exactly, and the same Jensen argument cuts
> the non-affine tail there too: the super-quantum range S∈(2√2,4],
> carried by p>2, is forbidden by no-signaling alone — the Tsirelson
> bound emerges as the upper cut of the non-affine tail in the joint
> layer, with no appeal to information causality. What no-signaling
> does not do is lift amplitude: it selects the rule in both layers
> but leaves the local layer at the 1/3-cosine point and the quantum
> value to the joint layer. The gap between them — the AMPLITUDE
> SEAM — is, we find, the same wall met by the classical relaxation
> dynamics of the preceding cycles, now seen from the reconstruction
> side. Whether an internal construction can generate its own
> steering diversity AND attain S=2√2 — closing the seam — or whether
> the seam is fundamental to outcome-definite mechanisms, is the
> program's registered central open problem (campaign C3-S).

## 3. Батарея-дополнение L3 — ВЫПОЛНЕНА (prereg 2d78c51, сырьё 9567ddb)
- Ячейка1 скан Δ(p) p∈{4,6,10,∞}: ПОДТВЕРЖДЁН — Δ>0 ∀p, монотонно,
  Δ(∞)=0.375 макс, аналит↔числ <2σ. Прогноз 0.85 — ✓.
- Ячейка2 внутренний мост S(p): S монотонен, S(∞)=4 (PR), **S(2)=√2**
  (не 2√2 — фактор-2, суб-единичное усл. среднее). Прогноз S(2)=2√2 (0.7)
  ОПРОВЕРГНУТ ⇒ расслоение §2.2 (амплитуда ≠ отбор). См. C3L_addendum1.

## 4. Связь с корпусом
Обобщает addendum2 (точный ноль p=2 → строгая положительность хвоста).
Наследует урок «Δ(2)>0=баг» и стандарт «аналитика рядом». Не трогает
границы C3-B (F1 — прямой закон; χ феноменологичен). Родня по no-signaling:
[G89,G90,P91,SBG01] (см. C3B_paper_SKELETON §6); IC-линия — §2.3 ARCH.
