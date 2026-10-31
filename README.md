# time-lab

Досліди про час: внутрішній час систем і моделей, ціна годинника, порядок подій у протоколах. Лабораторія починається з висновків архівованої `lab` ([STARTER.md](STARTER.md)).

Метод: попередня реєстрація як записи Warrant 3 ([METHOD.md](METHOD.md)). Гіпотеза — це лінія `prediction@1`. Її закриває лише attest від `time-lab-decider` з кодом аналізу, названим у реєстрації. Seeds даних виводяться з id реєстрації та раунду маяка drand, який настав після push.

| Каталог | Питання | Стан |
|---|---|---|
| [`q1-weber/`](q1-weber/) | Чи йде оцінка «скільки слів я прочитала» мовної моделі за подіями (словами) чи за власною тривалістю (токенами)? | вирішено: gemma3 — за власною тривалістю (H1 supported), gemma4 — проти неї, qwen3.8 — майже за подіями: [REPORT](q1-weber/REPORT.md) |
| [`q2-crystal/`](q2-crystal/) | Чи тримає солітонний кристал `lab` (ненавчений, під тягою) ритм світу, і чи це колективна властивість: тугіше з розміром і зв'язком, стійкіше до шуму? | реєстрація |

Стан гіпотез:

```bash
.venv/bin/python tools/tl.py assess
```

Контролі аналізу Q1:

```bash
.venv/bin/python -m unittest q1-weber/tests/test_analyze.py && .venv/bin/python -I q1-weber/tests/mutants.py
```

Перевірки самого `tl.py` (те, чого ядро Warrant не перевіряє, METHOD §4):

```bash
.venv/bin/python -m unittest tools/tests/test_tl.py && .venv/bin/python -I tools/tests/mutants.py
```
