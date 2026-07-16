# External review 3 (on paper-v1.1 / camera-ready) + architect resolutions

## 1. Текст внешнего ревью

> **[НЕ ПОЛУЧЕН ИСПОЛНИТЕЛЕМ.]**
>
> Задание указывало «Артем передаст verbatim или возьми из этого задания приложением —
> пришлю следом, если не приложено». Приложения в задании не было; текста нет ни в нём,
> ни в предыдущих, ни в репозитории.
>
> Исполнитель НЕ реконструирует его по резолюциям. Ровно этот пробел рецензент №2 назвал
> единственной дырой провенанса («решения архитектора сохранены; источник замечаний не
> сохранён»), и по ревью 1 и 2 он закрыт — оба лежат дословно, с маркерами целостности.
> Здесь он открыт снова.
>
> Вставить дословно сюда (с маркерами начала/конца), затем закоммитить.

## 2. Резолюции архитектора

Записаны со слов архитектора (задание к `paper-v1.2`), ДО получения дословного текста.
Формулировки претензий — в пересказе; сверить с оригиналом, когда он придёт.

### 2.1 Принято

| # | претензия (в пересказе) | резолюция | где |
|---|---|---|---|
| 1 | «manifestly local model» — заявляет больше, чем доказано | заменено везде на «a model whose readout is local by construction (Section 2.6)»; в §7.2 обоснование kill-триггера переведено с «конструкция манифестно локальна» на операциональное: «the isotropized protocol is an operational shared-λ scheme with settings independent of λ (Section 2.6(iii))» | §4.1, §7 преамбула, §7.2 |
| 2 | relaxation-model wording — утверждения звучат как о ленте, а не об исследованной модели | «the classical dynamics of such an object» → «the studied classical relaxation model of such an object» | §1 (ставка), §8.4 (что опровергнуто) |
| 3 | novelty framing | §6.4: добавлено «The isotropized correlation is statistically consistent with the standard triangular LHV form; we do not claim Bell-locality of the underlying global relaxation dynamics, for which factorization has not been established (Section 2.6(ii))» | §6.4 |
| 5 | «binarity and spinority descending from its framing topology» — пережило правку по ревью 2 п.9 (искали по другой формулировке, «descend without postulates») | → «whose framing topology supplies the ℤ₂ label behind binarity and spinority (Section 3.1)». Противоречие с §1/§3.1/§5.2/§8.1 снято: во всех пяти местах теперь одно — топология даёт label, физическое отображение к исходам требует отдельного обоснования | §8.4 |

### 2.3 Оставлено намеренно (решение архитектора)

| # | место | решение |
|---|---|---|
| 6 | §3.1: «What this paper tests, to its limit, is whether the classical dynamics of such an object can also produce the quantitative statistics of the singlet state» — формулировка не сужена до «studied classical relaxation model», в отличие от §1 и §8.4 | **НЕ менять.** Вопрос ставки намеренно шире исследованной модели; ответ ограничен в §8.4/§8.5. Асимметрия сознательная: §3.1 ставит вопрос об онтологии, §8.4 и §8.5 отвечают в границах исследованного класса моделей и измеренного диапазона. |

### 2.2 Отклонено с объяснением

| # | претензия (в пересказе) | почему отклонено |
|---|---|---|
| 4 | «путаница с DOI: в статье два разных DOI» | **Оба легитимны и различны по назначению.** `10.5281/zenodo.21383967` — *version* DOI: указывает на конкретный снапшот `paper-v1.1` (camera-ready) и не меняется при выходе новых версий; на него ссылается цитата архива. `10.5281/zenodo.21383667` — *concept* DOI: всегда резолвится на последнюю версию записи; на него стоит бейдж README, чтобы тот не устаревал. §9.1 их уже различает явно («The concept DOI … resolves to the latest archived version»). По ревью уточнён README: у бейджа — «(concept DOI — all versions)», у цитаты — «(version DOI — camera-ready snapshot)». Оба DOI проверены исполнителем по вебу: version резолвится на запись «paper-v1.1 — camera-ready snapshot», concept — на неё же как на последнюю. |

## 3. Открытое

- Текст ревью (§1).
- DOI-строка §9.1 будет обновлена на Zenodo v3 после тега `paper-v1.2` и релиза —
  отдельным коммитом поверх тега, той же механикой, что и для v1.1. Формулировка §9.1 это
  покрывает: архив = camera-ready состояние, DOI-строка не может архивировать саму себя.
