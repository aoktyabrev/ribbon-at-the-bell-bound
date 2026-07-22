# C4-GHZ / G-T2 — Schedule-invisibility v2 (клейм исправлен) + ВТОРОЙ адверс-проход.
# γ авторизован при чистом проходе (архитектор). Наследует C4GT2_T2N_adversarial (v1 дыры).

## Клейм v2 (архитектор, дословно)
> Schedule-invisibility (class M, N events) — corrected claim v2: let ≺ be the
> precedence structure of a mechanism in M realizing statistics P. (a) Chain-rule
> identity: every linear extension (resolution schedule) of ≺ yields the same P.
> (b) No functional of the operational statistics P recovers WHICH schedule was
> used. (c) Scope: the EXISTENCE and block structure of ≺ are NOT claimed
> invisible — they are certified by achieved tiers (Mermin/Svetlichny values);
> invisible is the schedule within the structure. All claims relative to
> operational P-statistics (L2c-caveat). Slogan: the tier reveals the structure;
> nothing reveals the schedule.

## ВТОРОЙ АДВЕРСАРИАЛЬНЫЙ ПРОХОД (по v2)
### Закрытие дыр v1:
- **ДЫРА-T2N-2** (невидимо расширение, не существование ≺): закрыта (c) —
  существование/блок-структура certified by tiers, НЕ claimed invisible. ✓
- **ДЫРА-T2N-3** (внутри одного ≺): закрыта (a) явным «linear extension OF ≺»
  (одного) + (c) не заявляет межструктурную инвариантность. ✓
- **ДЫРА-T2N-1** (P-скоуп): закрыта «All claims relative to operational P». ✓

### (a) — доказательство + премиса
Механизм РЕАЛИЗУЕТ P (посылка): исходы — консистентные условные P(next|preceding)
для единого P. Любое линейное расширение ≺ = порядок факторизации того же P;
цепное правило: Π P(next|prev) не зависит от порядка факторизации ⇒ то же P.
Батарея d612214 (N=4, 4 расширения): 4-корр разброс 0.000000. ✓ АНАЛИТИЧНО.
**Премиса «realizing P» несущая:** механизм с schedule-ЗАВИСИМЫМ joint (L1
collapse, N=2, 0.0418) НЕ реализует ЕДИНОЕ P ⇒ не удовлетворяет посылке ⇒ НЕ
контрпример (его ≺ обязан упорядочить пару, тогда одно расширение; при пустом ≺
и schedule-зависимом joint — нарушение M2/единого P). Назвал явно.

### (b) — королларий (a)
P тождественно ∀ расширений (a) ⇒ ЛЮБОЙ функционал P тождествен ⇒ schedule
невосстановим ИЗ P. Прямое следствие, не отдельная посылка. ✓

### (c) — согласованность с ярусами
Ярус (M>бонд/Свеличны/блок-размер) = P-функционал, ЗАВИСИТ от структуры ≺
(m=4→4.000 vs пара→2.000). ⇒ структура certified (снизу: min блок-размер), НЕ
невидима. Слоган «tier reveals structure; nothing reveals schedule» — точен:
schedule полностью невидим (P идентично), структура частично видима (ярусы).

### Контрпримеры (заново):
- смешанные расписания = разные линейные расширения одного ≺ → то же P (a). Не КП.
- частичный транскрипт → M3-нарушение ИЛИ неявный порядок (⊂≺). Классифицирован.
- w-кондиционирование → корреляции, не маргиналы; which-schedule не P-функционал. Чисто.
- schedule-коррелированная случайность ω → M4′/M1 требуют λ приготовительной
  (⊂λ, до настроек); resolution-time ω, зависящая от расписания, запрещена. Чисто.
- восстановление БЛОК-СТРУКТУРЫ функционалом P → разрешено (c) (не claimed invisible).
  Не КП (согласовано с ярусами).
- tier→structure не инъективно (разные ≺ = тот же ярус) → (c) говорит «certified»
  = НИЖНЯЯ граница структуры, не точная ≺; переусиления нет. Чисто.

## ВЕРДИКТ ВТОРОГО ПРОХОДА: ЧИСТ.
Дыры v1 закрыты; премиса «realizing P» несущая и названа; (b) королларий; (c)
согласован с ярусами; контрпримеры исчерпаны. **СТАТУС: ТЕОРЕМА** (schedule-
invisibility в классе M) — присвоена (α клейм v2 + β батарея d612214 + γ условная
авторизация архитектора при чистом проходе). Компаньон теоремы предшествования
класса M (c8e1bf3). Точная формулировка: schedule (линейное расширение ≺)
статистически невидим; структура ≺ certified ярусами, НЕ невидима.
