# GpuCodeforcesProblems

Банк задач GPU Codeforces. Задача добавляется или меняется pull request'ом; backend платформы
забирает ветку `master` и импортирует каждый каталог `problems/<id>/`. Имя каталога — постоянный ID
задачи (`[A-Za-z0-9._-]`), его нельзя менять после публикации.

```
problems/vector-addition/
  statement.md     условие с YAML-шапкой
  problem.json     описание задачи для судьи (без генератора), в том числе процессоры
  problem.py       генератор тестов и эталон
  starter.cu       шаблон решения на CUDA (необязательно)
  starter.rtl.sv   шаблоны других процессоров: starter.<процессор><расширение языка> (необязательно)
```

Полное описание формата — [docs/backend/problem-package.md](https://github.com/lottery7/GpuCodeforces/blob/master/docs/backend/problem-package.md)
и [judge/docs/problem-format.md](https://github.com/lottery7/GpuCodeforces/blob/master/judge/docs/problem-format.md)
в репозитории платформы. Кратко:

- `statement.md`: `code` (номер в банке), `title`, `difficulty` (`easy`/`medium`/`hard`), `tags`,
  `authors` (ники на платформе; черновик `draft: true` видят только они), затем условие в Markdown.
- `problem.json`: сигнатура `solve` (тип, роль и размер каждого массива, где он лежит — `DEVICE`/`HOST`),
  допуски сравнения, лимиты времени и памяти, флаги компиляции и запреты, а также `processors` — на каких
  процессорах и языках сдаётся задача и лимит score на каждом: `gpu-cuda` (мс, по умолчанию
  `limits.timeLimitMs`), `rtl` (такты × глубина логики × транзисторы схемы, у `rtl` ещё `readPorts` и `dataMode`
  `MEMORY`/`STREAM`), `riscv-rvv` (инструкции + 10 × промахи L1 + 100 × промахи L2), `cuda-sim` (такты GPGPU-Sim), `coral-npu`
  (инструкции ядра Coral NPU).
  Без `processors` задача сдаётся только на `gpu-cuda`.
- `problem.py`: `tests()` перечисляет тесты (открытый пример — `sample: True` с рукописными входами и
  ответом), `generate()` создаёт входы, `reference()` считает эталонный ответ (Python + PyTorch). У теста
  можно указать `"processors": [...]` — на каких процессорах он идёт (без поля — на всех процессорах задачи), и
  `"measure": False` — только проверка ответа, без score. Эмуляторы медленные, поэтому для них тесты маленькие
  (тысячи элементов для `rtl`, сотни тысяч для `riscv-rvv` и `cuda-sim`), а большие — только для `gpu-cuda`. У
  `coral-npu` все массивы теста лежат в 32 КБ памяти данных ядра вместе с данными решения: тест больше примерно
  28 КБ на нём не пройдёт никто. У каждого
  процессора нужен хотя бы один замерный тест.
- Без шаблона шаблоном станет пустое решение с подсказкой, построенное по сигнатуре.

Задача с ошибкой в пакете не попадёт в банк, причина видна в логе backend. Проверка из корня репозитория
платформы (нужны поднятый судья, `docker compose --profile judge up -d --wait`, и torch для числа тестов по
процессорам):

```sh
python3 judge/tools/judge_cli.py validate ../GpuCodeforcesProblems/problems/prefix-sum
python3 judge/tools/judge_cli.py submit ../GpuCodeforcesProblems/problems/prefix-sum \
    solution.sv --processor rtl --watch
```

Генератор отдельно (`--processor` — только тесты этого процессора):

```sh
PYTHONPATH=judge/pytools python3 -m judge_problem.prepare --spec problem.json --source problem.py --out /tmp/out --processor rtl
```
