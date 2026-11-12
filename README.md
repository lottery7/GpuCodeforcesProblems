<div align="center">

<img src=".github/logo.svg" width="56" height="56" alt="">

# Банк задач GPU Forces

**Двадцать задач для пяти процессоров: видеокарта, её симулятор, векторный RISC-V, Coral NPU и схемы на SystemVerilog.**

[Платформа](https://github.com/lottery7/GpuCodeforces) · [Задачи](#задачи) · [Формат пакета](#формат-пакета)

</div>

Задачи платформы [GPU Forces](https://github.com/lottery7/GpuCodeforces) — соревнований, где мало решить задачу:
её надо решить быстро на заданном процессоре. Задача добавляется или меняется pull request'ом; backend платформы
забирает ветку `master` и импортирует каждый каталог `problems/<id>/`. Имя каталога — постоянный ID
задачи (`[A-Za-z0-9._-]`), его нельзя менять после публикации.

## Задачи

| № | Задача | Формат | Сложность | `gpu-cuda` | `cuda-sim` | `rtl` | `riscv-rvv` | `coral-npu` |
|---|---|---|---|:-:|:-:|:-:|:-:|:-:|
| 1 | [Vector Addition](problems/vector-addition/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | [Matrix Multiplication](problems/matrix-multiplication/statement.md) | обычная | medium | ✓ |  | ✓ | ✓ | ✓ |
| 3 | [Prefix Sum](problems/prefix-sum/statement.md) | обычная | easy |  |  | ✓ |  |  |
| 4 | [Dot Product](problems/dot-product/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5 | [Hamming Distance](problems/hamming-distance/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6 | [Byte Histogram](problems/byte-histogram/statement.md) | обычная | medium | ✓ | ✓ |  | ✓ | ✓ |
| 7 | [Matrix Transpose](problems/matrix-transpose/statement.md) | обычная | medium | ✓ | ✓ |  | ✓ | ✓ |
| 8 | [Greatest Common Divisor](problems/gcd/statement.md) | обычная | easy |  |  | ✓ | ✓ | ✓ |
| 9 | [Speed Up the Transpose](problems/transpose-optimize/statement.md) | оптимизация | medium |  |  |  | ✓ |  |
| 10 | [Speed Up the Bit Count](problems/hamming-optimize/statement.md) | оптимизация | easy |  |  |  |  | ✓ |
| 11 | [Rows or Columns](problems/guess-matrix-walk/statement.md) | угадывание | easy |  |  |  | ✓ |  |
| 12 | [One Instruction or Thirty-Two Lanes](problems/guess-bit-count/statement.md) | угадывание | medium |  |  |  |  | ✓ |
| 13 | [Scale and Add](problems/axpy/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 14 | [Clamp](problems/clamp/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 | [Mirror the Bits](problems/bit-reverse/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 16 | [Maximum](problems/array-maximum/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 17 | [Shades of Gray](problems/rgb-to-gray/statement.md) | обычная | easy | ✓ |  | ✓ | ✓ | ✓ |
| 18 | [Three-Point Stencil](problems/three-point-sum/statement.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 19 | [Collatz Steps](problems/collatz-steps/statement.md) | обычная | medium | ✓ | ✓ | ✓ | ✓ | ✓ |
| 20 | [Matrix Times Vector](problems/matrix-vector/statement.md) | обычная | medium | ✓ | ✓ |  | ✓ | ✓ |
| | **Всего задач** | | | **14** | **12** | **13** | **17** | **17** |

## Формат пакета

```
problems/vector-addition/
  statement.md     условие с YAML-шапкой
  problem.json     описание задачи для судьи (без генератора), в том числе процессоры
  problem.py       генератор тестов и эталон
  starter.cu       шаблон решения на CUDA (необязательно)
  starter.rtl.sv   шаблоны других процессоров: starter.<процессор><расширение языка> (необязательно)
  format.json      формат задачи: optimization или guess (необязательно, без него задача обычная)
  first.c second.c два кода задачи формата guess, расширение - по её языку
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
- Форматы ([docs/backend/formats.md](https://github.com/lottery7/GpuCodeforces/blob/master/docs/backend/formats.md)):
  в `optimization` шаблон каждого процессора - рабочее исходное решение, которое участник ускоряет
  (`transpose-optimize`, `hamming-optimize`); в `guess` у задачи один процессор и один язык, а в пакете лежат два
  кода `first` и `second` и объяснение в `format.json` (`guess-matrix-walk`, `guess-bit-count`). Исходное решение и
  оба варианта платформа измеряет сама; объяснение должно совпадать с тем, что она намерит, а замерные тесты -
  быть достаточно большими: у крошечного теста ответ случаен. Тест с `"measure": False` в результат не входит.
- Взломы ([docs/backend/participation.md](https://github.com/lottery7/GpuCodeforces/blob/master/docs/backend/participation.md)):
  `"allowCustomTests": true` в `problem.json` и функция `validate_input(test, inputs)` в `problem.py`, которая
  возвращает `True` только для теста в ограничениях условия. Включены у обычных задач банка, кроме
  `matrix-multiplication` и `prefix-sum`.

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
