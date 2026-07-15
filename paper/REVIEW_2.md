# External review 2 (on DRAFT_v2) + architect resolutions

## Provenance

Verbatim as posted by the author into the architect chat; architect relayed as transport.
Integrity markers received with the text: begins "v2 стала заметно лучше", ends
"не косметическая, а структурная переработка." Both verified against the inserted text.

**Note on fidelity.** The review is inserted exactly as received, including the mathematical
formulas, which arrived broken by the chat's rendering (display math collapsed into
line-fragments, e.g. `E(θ)= π 2θ ​ −1`). Nothing was repaired: repairing a quoted external
document would substitute the executor's reading for the reviewer's words. The formulas are
readable in context; the reviewer's intent is not in doubt at any of these points.

## 1. Text of the review

=== BEGIN REVIEW 2 VERBATIM ===

v2 стала заметно лучше как оформленный черновик, но важно точно назвать её статус: это пока не исправленная после рецензии статья, а промежуточная версия с вставленными иллюстрациями, несколькими уточнениями и зафиксированным планом дальнейшей переработки.

Большинство принципиальных замечаний правильно принято в REVIEW_1.md, но ещё не перенесено в сам текст статьи.

Что стало лучше
1. Иллюстрации теперь являются частью аргумента

Добавлены сами графики и полноценные подписи:

plateau и cross-scan;
зависимость амплитуды от жёсткости;
карта анизотропии;
triangle versus cosine.

Это серьёзное улучшение. Раздел 5 теперь можно читать без постоянного обращения к репозиторию. Особенно хорошо выглядит Fig. 4: она визуально выражает главный отрицательный результат статьи.

2. Исправлено описание треугольной корреляции

Теперь аналитический результат корректно разделён на:

E(θ)=
π
2θ
    ​

−1

при полной амплитуде и динамический вариант

E(θ)=−ρ(1−
π
2θ
    ​

)

с измеренным ρ<1.

Это стало гораздо понятнее: аналитика задаёт форму предельной модели, а динамика — реальную недостаточную амплитуду.

3. Аккуратнее сформулирован provenance

Уточнение, что коммитом датируются только rediscoveries, сделанные после начала version-controlled record, правильное. Оно убирает потенциально недоказуемое заявление о календарном приоритете.

4. Проверка Gisin проведена очень хорошо

Вывод в REVIEW_1.md точный: работы Gisin 1989/1990 относятся к нелинейной динамике и сверхсветовой сигнализации, а не к выводу Born rule из no-signaling. Предложенная формулировка —

measurement-rule analogue of Gisin's no-go for nonlinear dynamics

— намного лучше текущей.

Главный вывод
Большая часть принятого ревью пока не реализована

В REVIEW_1.md правильно приняты девять основных замечаний: добавить Abstract и Model, ограничить causal claim рамками ansatz, снять Kochen–Specker attribution, доказать Bell-locality, ослабить категоричные формулировки и переработать AI-раздел.

Но DRAFT_v2.md всё ещё содержит почти все проблемные формулировки из v1.

1. Abstract всё ещё отсутствует

После frontmatter сразу начинается Introduction. Для подачи это обязательный блок.

Хороший Abstract здесь должен содержать пять элементов:

Какая модель тестируется.
Какой максимальный результат пытались получить.
Что было измерено.
Где модель упёрлась в классическую границу.
Какой итоговый вклад статьи.

Ориентировочно 170–220 слов.

2. Model section всё ещё отсутствует

Это по-прежнему главный блокер.

В тексте упоминаются:

quaternion frames;
Bishop transport;
energy;
Langevin dynamics;
clamps;
basins;
sector preparation;
parity;
axial readout;

но их математические определения отсутствуют.

До чтения результатов рецензент должен увидеть примерно такую структуру:

Configuration space
x
i
    ​

∈R
4
,q
i
    ​

∈S
3
,

с определением касательных, локального репера и дискретизации.

Energy functional
H=H
stretch
    ​

+H
bend
    ​

+H
frame
    ​

+H
clamp
    ​

.

Нужны все слагаемые и параметры.

Dynamics

Точное дискретное Langevin equation, timestep, noise normalization и критерий окончания релаксации.

Measurement

Как из конечной конфигурации получаются:

A(a,λ)∈{−1,+1},B(b,λ)∈{−1,+1}.
Correlation estimator
E(a,b)=
M
1
    ​

j=1
∑
M
    ​

A
j
    ​

B
j
    ​

.
Topological invariant

Как вычисляется Z
2
    ​

-паритет и при каких условиях он определён.

Без этого невозможно независимо оценить численный результат.

3. Сборщик пока даже не предусматривает новые разделы

В build_draft.py порядок по-прежнему начинается так:

"00_frontmatter.md",
"01_intro.md",
"02_theory.md",
"03_program.md",

Файлов Abstract и Model в ORDER нет. Значит, даже после написания они не попадут в следующий draft, пока не будет изменён сборщик.

Я бы сделал новый порядок:

"00_frontmatter.md",
"00_abstract.md",
"01_intro.md",
"02_model.md",
"03_analytical.md",
"04_methods.md",
"05_nogo.md",
"06_results.md",
"07_selfcorrection.md",
"08_discussion.md",
"90_availability.md",

Не обязательно именно с такой нумерацией, но Model должен идти до результатов и simulation history.

Теоретические замечания пока остались без изменений
4. Chord law всё ещё подан как вывод

Текущий текст:

The measurement model assigns outcome probabilities through a chord measure…

и затем:

Its quadratic exponent is not a choice.

Но в резолюции уже принято более точное понимание: исходный квадратный chord law — это геометрическая транскрипция квантового совместного распределения, а не его вывод.

Потому что:

8
∣sa−tb∣
2
    ​

=
4
1−stcosθ
    ​

.

Это уже singlet distribution.

Нужно разделить два утверждения:

Квадратный chord law точно переписывает quantum target.
Внутри специально определённого семейства p-деформаций no-signaling оставляет только p=2.

Сейчас эти два шага слиты, создавая впечатление, будто вся Born statistics выведена из геометрии.

5. Ограничение «within the ansatz» пока не внесено

Фраза:

Causality pins the Born exponent from both sides.

по-прежнему звучит универсально.

Нужна версия:

Within the specified chord-power ansatz and its stated partially entangled extension, the no-signaling condition singles out p=2.

Это сохраняет твой результат, но не выдаёт его за общую теорему о происхождении правила Борна.

6. Ошибочная атрибуция Gisin всё ещё стоит в статье

В трёх местах по-прежнему написано, что результат «reproduces Gisin (1990)», хотя собственная проверка в REVIEW_1.md уже установила, что это неточно.

Следует заменить везде согласованно:

The argument is a measurement-rule analogue of Gisin's no-go results for nonlinear quantum dynamics (Gisin 1989, 1990), but applies to a different object: the outcome measure rather than the evolution law.

Это честно и при этом показывает реальное концептуальное родство.

7. Kochen–Specker всё ещё не снят

Несмотря на принятую резолюцию, статья по-прежнему говорит:

Kochen–Specker supplies the constructive complement

и:

the constructive face of Kochen–Specker.

Это нужно удалить полностью.

Твоё «conservation law of the postulate» можно оставить как сильное авторское наблюдение:

We call this relocation of the probabilistic assumption the conservation of the postulate. This is a methodological observation within the tested model family, not a consequence of the Kochen–Specker theorem.

Так идея становится защищаемой.

Не решена Bell-locality
8. Модель названа manifestly local, но факторизации нет

Статья несколько раз исходит из того, что:

any CHSH > 2 in a manifestly local model is a protocol error.

Методологически это сработало прекрасно. Однако для научного доказательства нужно показать:

P(A,B∣a,b,λ)=P(A∣a,λ)P(B∣b,λ),

а также measurement independence:

P(λ∣a,b)=P(λ).

Необходимо прояснить, что входит в λ:

начальная геометрия;
исходная ориентация;
все noise variables;
sector;
локальные внутренние состояния;
random seeds.

Особенно опасный вопрос: если после установки обоих clamp settings весь объект совместно релаксирует, то результат A потенциально зависит от b. Геометрическая цельность объекта сама по себе не обеспечивает Bell-locality.

Возможны два честных варианта:

Вариант A: доказать factorization

Показать, что при фиксированном λ:

A=A(a,λ),B=B(b,λ).
Вариант B: не заявлять Bell-locality динамики

И ограничить Bell-анализ отдельным isotropized effective readout, для которого локальная структура действительно доказана.

Пока это крупнейшая логическая лакуна статьи после отсутствующей Model section.

Оставшиеся сверхутверждения
9. «Бинарность и спинорность следуют бесплатно»

Фраза сохранилась почти дословно:

binary character of outcomes and the 720° spinor property descend without postulates.

Топология гарантирует естественный Z
2
    ​

-класс и belt-trick structure, но не автоматически физический measurement outcome.

Лучше:

The framing topology supplies a natural Z
2
    ​

-valued structural label and the homotopy underlying the 4π belt-trick return. A physical map from this label to measurement outcomes remains an additional requirement.

Это особенно важно, потому что твой U(1) theorem затем показывает: axial readout именно этот класс не видит.

10. Exact zeros всё ещё названы запрещёнными

Во введении:

exact outcome zeros are forbidden to it by a structural theorem.

А census показывает более узкое:

каждая ветвь имеет допустимых представителей;
топология не делает ветвь пустой;
axial readout не видит инвариант.

Это не исключает basin ненулевой конфигурационной поддержки, но нулевой меры.

Точнее:

The topology does not enforce exact outcome zeros: no branch is topologically forbidden, and the framing invariant is inaccessible to axial readout.

Это сильный результат. Его не нужно усиливать до глобального утверждения, что модель принципиально не может породить нулевую вероятность никаким механизмом.

11. «No measure-zero prohibition» логически слишком сильно

Наличие хотя бы одного non-singular representative означает non-empty support, но не обязательно положительную меру.

Нужно заменить:

there is no measure-zero prohibition

на:

there is no topological empty-support prohibition.

Это важная математическая правка.

12. A
∞
    ​

 и scale-invariant преждевременны

Измерения показывают отсутствие заметного падения на:

16≤N≤96.

Но обозначение A
∞
    ​

 уже утверждает предел N→∞, который не измерен.

Лучше использовать:

A
plateau
    ​

илиA
fit
    ​

.

Вместо:

scale-invariant amplitude

писать:

no statistically resolved length dependence over 16≤N≤96.

13. «Constant wins decisively»

Для

ΔAICc=6.36

разумнее:

the constant model is moderately to strongly preferred over the tested power-law alternative.

Слово decisively обычно оставляют для ещё более сильных различий.

14. Предел k
f
    ​

→∞ пока является экстраполяцией

Статья утверждает:

k
f
    ​

→∞⇒ρ→1,S→2.

Но измерено четыре значения жёсткости, максимум k
f
    ​

×4. Это правдоподобная механическая гипотеза, но не установленный численный предел.

Лучше:

The measured family moves toward the Bell bound as stiffness increases. Its deterministic axial limit is expected to satisfy ρ→1 and S→2.

Раздел AI пока тоже не обновлён

Он по-прежнему описывает:

AI architect;
AI executor;
human owner.

Для внутренней методологии это интересно, но журнальная версия должна однозначно закреплять ответственность:

Generative AI tools were used for research planning, implementation assistance, code review, source checking, and manuscript drafting. The author made all scientific decisions, verified the reported results and references, and accepts full responsibility for the manuscript.

При этом подробную историю взаимного аудита лучше оставить в supplementary methodological note — она действительно интересная.

Отдельно о REVIEW_1.md

Файл организован хорошо, но в нём прямо отмечено, что дословный внешний отзыв отсутствует.

Это стоит исправить обязательно. Мой предыдущий отзыв нужно вставить туда полностью, иначе получается:

решения архитектора сохранены;
источник замечаний не сохранён;
provenance chain неполон.

Учитывая, что прозрачность является одной из центральных тем статьи, такой пробел особенно заметен.

Вердикт по v2
Что готово
основная исследовательская история;
численные результаты;
self-correction narrative;
графики и подписи;
provenance numerical campaigns;
карта главных необходимых исправлений.
Что ещё не готово
Abstract;
воспроизводимая Model section;
Bell-locality proof;
корректная подача chord law;
снятие Kochen–Specker;
исправление Gisin;
сужение топологических утверждений;
сужение N→∞ и k
f
    ​

→∞ claims;
журнальный AI disclosure;
References/Bibliography.
Оценка
Оформление и читаемость: 8.5/10
Вычислительная методология: 9/10
Сила центрального отрицательного результата: 8/10
Строгость теоретических формулировок: 5/10
Готовность к подаче: 6/10

То есть v2 стала лучше визуально и организационно, но основные научные блокеры пока те же. Хорошая новость в том, что REVIEW_1.md показывает: команда их поняла правильно. Теперь нужна именно v3 — не косметическая, а структурная переработка.

=== END REVIEW 2 VERBATIM ===

## 2. Resolutions — implemented in v3

Every numbered item was accepted and carried into the text. The commits below are the record;
the review's central complaint ("большая часть принятого ревью пока не реализована") is what
v3 answers.

| # | review item | resolution | commit |
|---|---|---|---|
| 1 | Abstract missing | §0a added, 221 words, five required elements | `b3d0ec7` |
| 2 | Model section missing | §2 Model (2.1–2.6) written, then checked line-by-line against the code inventory `MODEL_FACTS.md` | `620816f`, `3203e69`, `d8262b2`, `2b5a1b1` |
| 3 | Builder lacks new sections | `ORDER` rebuilt to the reviewer's proposed sequence; SI split out | `b3d0ec7` |
| 4 | Chord law presented as a derivation | §3.2 now states the transcription identity \|s·a − t·b\|²/8 = (1 − st·cosθ)/4 explicitly, then separates the ansatz-level claim | `e5f267f` |
| 5 | "within the ansatz" limit | "Causality pins p = 2 within this ansatz — we make no claim of a general derivation of the Born rule"; PR-box (p → ∞, CHSH → 4) added as the counterweight | `e5f267f` |
| 6 | Gisin misattribution | replaced in all three places with the measurement-rule-analogue formulation; dated correction added to the canonical record rather than rewriting it | `4f6f69f` |
| 7 | Kochen–Specker not removed | removed from §3.3, §8.2 **and §1.1** (the review named two places; a third carried the same attribution); conservation of the postulate reclassified as a methodological observation, Gleason named as the nearest formal relative and explicitly not invoked | `e5f267f` |
| 8 | Bell-locality not proven | **Option B taken.** §2.6 states three levels separately: (i) readout locality holds by construction; (ii) dynamical locality is *not claimed* — factorization over the relaxation would require its stationary measure, which is not derived; (iii) the isotropized protocol is a shared-λ scheme with λ = (R, initial configuration, noise realization) and independently drawn settings. The paper's Bell-side claims rest on (i) and (iii) only | `620816f` |
| 9 | "descend without postulates" | replaced in §1, §3.1 (as an amendment to the approved paragraph), and — for consistency — §5.2 and §8.1, which carried the same claim | `e5f267f`, `3203e69` |
| 10 | exact zeros "forbidden" | narrowed to the census + inaccessibility formulation | `e5f267f` |
| 11 | "no measure-zero prohibition" | → "no topological empty-support prohibition (non-empty support established by explicit representatives; measure statements are a separate, dynamical question)" | `e5f267f` |
| 12 | A∞ premature | renamed A_plateau throughout text, captions and figure scripts, with the definition at first use | `e5f267f`, `cd420cf` |
| 13 | "wins decisively" | → "is substantially preferred among the three tested models (ΔAICc = 6.36)" | `e5f267f` |
| 14 | k_f → ∞ extrapolation | → "the measured family moves toward the Bell bound as stiffness grows (\|S\| = 1.62 at k_f×4); its deterministic axial limit is expected — not measured — to reach ρ → 1, S → 2" | `e5f267f` |
| — | AI section | journal disclosure in the main text (§9.2); the mutual-audit history moved to the Supplementary methodological note | `e5f267f`, `7bff35e` |
| — | References/Bibliography | 12 entries, each verified on the web against the publisher record or an authoritative index; citation keys placed and cross-checked (no orphans, no dangling) | `f335896` |

Structural revision and camera-ready build: `b3d0ec7`, `7bff35e`.

**Still open:** the review's closing point — that REVIEW_1's verbatim text is missing and the
provenance chain is therefore incomplete — is **not yet closed**. The transport of review 1
has not reached the executor; `REVIEW_1.md §1` remains a placeholder. The reviewer is right
that this gap is conspicuous in a paper whose subject is transparency.
