# Paper 3 (C3-B) — SKELETON
## «Born from causality: internal derivation of p=2 within the ribbon's steering class»
Статус: скелет (этап 1.5, задача 3). Выводы — архитектор поверх.
Источники результатов: prereg dd7a74e + addendum1 (96a3178) + addendum2
(ccf5af6); сырьё batteries 968aaa5 (B1-B7) и c152c98 (hardening T1-T3);
JSON — phase_D/results/C3B_hardening.json, C3B_battery*.json.

---

### 0. Abstract (заполнить после выводов архитектора)
Одна фраза-клейм: no-signaling форсирует показатель p=2 (Борн)
ВНУТРЕННЕ, из асимметричного закона ленты через смещённую меру
источника μ_χ, без импорта КМ-ансамблей. Область: steering class
(семейство F2, смещённая мера). Граница: F1 (прямой корр.-закон) —
не закрывает; χ феноменологичен (долг C3-B-mech).

### 1. Введение и импорт #2
- Что такое импорт #2 (канон §3.2 / FINAL_v1:284-290): «within this
  ansatz… no claim of a general derivation of the Born rule» — партиально
  запутанные ансамбли ВВОДИЛИСЬ руками.
- Цель C3-B: вывести стиринг ВНУТРЕННЕ ⇒ снять оговорку.
- Что доказывается и что нет (клейм-дисциплина addendum1).

### 2. Конструкция F2 (steering class)
- Асимметричный закон ленты P(s,t|a,b,χ); смещённая мера источника
  μ_χ(λ) ∝ 1 + χ λ_z (§ prereg B-family).
- Внутренний стиринг = СЫРОЕ условное среднее E[λ|s] (суб-единичный
  Bloch-вектор). **Урок 968aaa5:** нормировка условной оси = баг
  (ломала закон полной вероятности; Δ(p=2)>0 сигналил баг) → фикс:
  сырой Bloch. [методология: «Δ(2)>0 = баг»-критерий сработал.]
- f_p(c) = |1+c|^{p/2} / (|1+c|^{p/2}+|1-c|^{p/2}).

### 3. Результаты батареи B1-B7 (968aaa5)
- B1/B2 PASS; B3 no-signaling PASS ∀χ при p=2 (по построению).
- B5: D(χ): 0.03→0.08 при χ>0 — стиринг ГЕНЕРИРУЕТСЯ μ_χ, не импортируется.
- Телеграф-скан: Δ(p=2)=0, нуль-множество = {2}.
- **Аналитика (Jensen-щель):** численность точно; Δ=0 ⇔ p=2 ∀χ>0.
- Замыкание ∃χ: D>0 ∧ (Δ=0⇔p=2). Прогнозы архитектора взяты
  (F2 0.6, D>0 0.5, замыкание 0.75; ловушка защиты 0.15 не сработала).

### 4. Упрочнение до теоремы (addendum1, hardening c152c98)
- **T1 (B7 цепные/порядок):** повторяемость P(+)=1.0; порядок a·b==b·a;
  no-signaling p=2 оба порядка; Δ-скан послеизмерит. ансамбля (ŷ+)
  → нуль-множество {2}, D_post=0.057. **ПРОХОД** (прогноз 0.7).
- **T2 (анти-циркулярность, несущий):** 5 деформаций меры
  (μ_κ, κ∈{0.5,2,3}; μ_cub две точки, позитивность+точки зарегистр.
  ДО скана). Все D>0 (0.03–0.09); аналит. нуль-множ.={2} на ВСЕХ;
  |аналит−числ|<2σ. **РОБАСТНО** (прогноз 0.6). Замыкание не сужается
  до линейной меры.
- **T3 (F1):** прямой корреляционный закол |s√w_a a − t√w_b b|^p —
  свободный p, НЕ стиринг-коллапс ⇒ B5 неприменим; F1 постулирует
  ансамбли, не порождает из скрытой меры. **ГРАНИЦА конструкции**
  (closes_import=false; прогноз 0.4 отработал как записан).

### 5. МЕТОДЫ: коррекция гейта (addendum2) — часть доказательства дисциплины
- Ложный СТОП прогона 19:39: гейт `analytic_zero_set(tol=1e-9)` на
  трапеции 20001 — ошибка квадратуры ~2.3e-5 ≫ tol, валил T1/T2 при
  чистой физике. Сходимость остатка 1/ngrid → 0 продемонстрирована
  (2.3e-4→2.3e-7 на ×10 сетки).
- Диагноз: гейт отсутствовал в prereg; зарегистрированный критерий
  (addendum1/A4) = числ. нуль-множ.={2} ∧ |аналит−числ|<2σ.
- Коррекция: гейт → prereg; в p=2 аналит. Δ АЛГЕБРАИЧЕСКИ (аффинность
  f₂ ⇒ зазор Йенсена≡0 тождественно ∀ меры), не квадратурой.
- **Почему в статью:** ложный СТОП и его разбор = свидетельство
  prereg-дисциплины (сырьё до правок; prereg↔gate построчно; порог не
  подгонялся под проход — верифицирован независимо от вердикта).

### 6. Атрибуция (Gisin) и контекст
- Стиринг из скрытой меры / no-signaling-ограничение на корреляции —
  линия Gisin и последователей. Наш довод — измерительно-правиловый
  аналог жизиновского (форма довода [G89,G90], объект наш); клейма
  «вывод Борна» у Жизина нет ни в одной из работ (канон-коррекция
  2026-07-15). Отличие нашего: p=2 форсируется ВНУТРЕННЕ асимметрией
  ленты, не постулатом КМ; стиринг-ансамбли порождаются геометрией
  модели, не предполагаются из КМ.

Ссылки (проверено архитектором 2026-07-21; Gisin-трио + Polchinski —
web-сверка с DOI; остальные канонические, финальная DOI-сверка при вёрстке):
- [G89] N. Gisin, "Stochastic quantum dynamics and relativity",
  Helv. Phys. Acta 62, 363–371 (1989).
- [G90] N. Gisin, "Weinberg's non-linear quantum mechanics and
  supraluminal communications", Phys. Lett. A 143(1–2), 1–2 (1990).
  DOI 10.1016/0375-9601(90)90786-N.
- [G91] N. Gisin, "Bell's inequality holds for all non-product states",
  Phys. Lett. A 154(5–6), 201–202 (1991).
- [P91] J. Polchinski, "Weinberg's nonlinear quantum mechanics and the
  Einstein–Podolsky–Rosen paradox", Phys. Rev. Lett. 66, 397 (1991).
- [S35] E. Schrödinger, "Discussion of probability relations between
  separated systems", Proc. Camb. Phil. Soc. 31, 555 (1935); 32, 446
  (1936) — steering.
- [HJW93] L. P. Hughston, R. Jozsa, W. K. Wootters, Phys. Lett. A 183,
  14–18 (1993) — классификация ансамблей данной ρ (структура, которую
  наша D(χ)-диверсность реализует геометрически).
- [SBG01] C. Simon, V. Bužek, N. Gisin, "No-signaling condition and
  quantum dynamics", Phys. Rev. Lett. 87, 170405 (2001).
- [PR94] S. Popescu, D. Rohrlich, Found. Phys. 24, 379 (1994).
- [A04] S. Aaronson, "Is Quantum Mechanics an Island in Theoryspace?",
  quant-ph/0401062 (2004) — |ψ|^p-деформации.

### 7. Границы и открытые долги
- χ феноменологичен: механическая реализация = **C3-B-mech** (долг A2,
  открывается ПОСЛЕ теоремы, не блокирует).
- F1 не закрывает импорт (§4 T3) — фиксируется как граница семейства.
- Область клейма: steering class (F2, смещённая мера), одиночные +
  цепные измерения (T1). Не общий вывод Борна вне класса.

### 8. Conclusions (архитектор, дословно)
Within a geometric hidden-connection model, we close what was an
acknowledged import from quantum formalism: partial entanglement is
realized internally as a shifted orientation measure of the shared
connection (family F2), which by itself generates steering
diversity — remotely prepared ensembles of the near end with equal
means and distinct higher moments (D(χ) > 0). Given that diversity,
the no-signaling requirement forces the readout rule to be affine
in the projection, singling out the chord exponent p = 2 — the Born
rule — uniquely. The result is verified numerically (N = 2·10⁶ per
cell), matched by the analytic Jensen-gap computation to
< 3·10⁻⁵, exact algebraically at p = 2, robust under five
deformations of the measure family, and consistent under sequential
measurements. The full battery is a single reproducible script with
preregistered kill criteria; one false stop (a quadrature-based gate
absent from the preregistration) was caught, documented, and
corrected without loosening any registered criterion.

We state the claim's boundaries as part of the claim. This is an
internal derivation within the ribbon's steering class, not a
general derivation of the Born rule: the asymmetry parameter χ is
phenomenological at this stage (its mechanical realization in the
elastic ribbon is a registered open debt); the weighted-chord
family F1 admits no steering construction and marks a genuine
boundary of the mechanism; and the kinship with the no-signaling
no-go results of Gisin and successors [G89, G90, P91, SBG01] is
acknowledged — our contribution is that the steering ensembles are
produced by the model's own geometry rather than assumed from
quantum theory, turning the causality argument into a closed loop
inside one construction. Combined with the exclusion results of the
preceding cycles — the amplitude–form–isotropy trilemma, the
measured factorization of relaxational readout, and the structural
decorrelation of the stiff-chain regime — the program's overall
picture is symmetric: classical relaxational dynamics in this class
cannot reach quantum correlations, and, in the same geometry, the
prohibition of superluminal signaling leaves the quantum rule no
freedom. The wall and the selection are two faces of the same
constraint, and both are stated with executable evidence.

---
### Счёт прогнозов архитектора (C3-B addendum1)
T1 0.7 — проход ✓ | T2 0.6 — робастно ✓ | T3 0.4 — граница (как записан) ✓.
