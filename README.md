<div align="center">

<img src=".github/logo.svg" width="56" height="56" alt="">

# Банк задач GPU Forces

**Тридцать три задачи для пяти процессоров: видеокарта, её симулятор, векторный RISC-V, Coral NPU и схемы на
SystemVerilog. Условия — на русском и английском.**

[Платформа](https://github.com/lottery7/GpuCodeforces) · [Задачи](#задачи) · [Формат пакета](#формат-пакета)

</div>

Задачи платформы [GPU Forces](https://github.com/lottery7/GpuCodeforces) — соревнований, где мало решить задачу:
её надо решить быстро на заданном процессоре. Задача добавляется или меняется pull request'ом; backend платформы
забирает ветку `master` и импортирует каждый каталог `problems/<id>/`. Имя каталога — постоянный ID
задачи (`[A-Za-z0-9._-]`), его нельзя менять после публикации.

## Задачи

| № | Задача | Problem | Формат | Сложность | `gpu-cuda` | `cuda-sim` | `rtl` | `riscv-rvv` | `coral-npu` |
|---|---|---|---|---|:-:|:-:|:-:|:-:|:-:|
| 1 | [Сложение векторов](problems/vector-addition/statement.md) | [Vector Addition](problems/vector-addition/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | [Умножение матриц](problems/matrix-multiplication/statement.md) | [Matrix Multiplication](problems/matrix-multiplication/statement.en.md) | обычная | medium | ✓ |   | ✓ | ✓ | ✓ |
| 3 | [Префиксные суммы](problems/prefix-sum/statement.md) | [Prefix Sum](problems/prefix-sum/statement.en.md) | обычная | easy |   |   | ✓ |   |   |
| 4 | [Скалярное произведение](problems/dot-product/statement.md) | [Dot Product](problems/dot-product/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5 | [Расстояние Хэмминга](problems/hamming-distance/statement.md) | [Hamming Distance](problems/hamming-distance/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6 | [Гистограмма байтов](problems/byte-histogram/statement.md) | [Byte Histogram](problems/byte-histogram/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 7 | [Транспонирование матрицы](problems/matrix-transpose/statement.md) | [Matrix Transpose](problems/matrix-transpose/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 8 | [Наибольший общий делитель](problems/gcd/statement.md) | [Greatest Common Divisor](problems/gcd/statement.en.md) | обычная | easy |   |   | ✓ | ✓ | ✓ |
| 9 | [Ускорьте транспонирование](problems/transpose-optimize/statement.md) | [Speed Up the Transpose](problems/transpose-optimize/statement.en.md) | оптимизация | medium |   |   |   | ✓ |   |
| 10 | [Ускорьте подсчёт битов](problems/hamming-optimize/statement.md) | [Speed Up the Bit Count](problems/hamming-optimize/statement.en.md) | оптимизация | easy |   |   |   |   | ✓ |
| 11 | [Строки или столбцы](problems/guess-matrix-walk/statement.md) | [Rows or Columns](problems/guess-matrix-walk/statement.en.md) | угадывание | easy |   |   |   | ✓ |   |
| 12 | [Одна инструкция или тридцать две полосы](problems/guess-bit-count/statement.md) | [One Instruction or Thirty-Two Lanes](problems/guess-bit-count/statement.en.md) | угадывание | medium |   |   |   |   | ✓ |
| 13 | [Умножить и сложить](problems/axpy/statement.md) | [Scale and Add](problems/axpy/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 14 | [В границах](problems/clamp/statement.md) | [Clamp](problems/clamp/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 | [Зеркало битов](problems/bit-reverse/statement.md) | [Mirror the Bits](problems/bit-reverse/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 16 | [Максимум](problems/array-maximum/statement.md) | [Maximum](problems/array-maximum/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 17 | [Оттенки серого](problems/rgb-to-gray/statement.md) | [Shades of Gray](problems/rgb-to-gray/statement.en.md) | обычная | easy | ✓ |   | ✓ | ✓ | ✓ |
| 18 | [Трёхточечный шаблон](problems/three-point-sum/statement.md) | [Three-Point Stencil](problems/three-point-sum/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 19 | [Шаги Коллатца](problems/collatz-steps/statement.md) | [Collatz Steps](problems/collatz-steps/statement.en.md) | обычная | medium | ✓ | ✓ | ✓ | ✓ | ✓ |
| 20 | [Матрица на вектор](problems/matrix-vector/statement.md) | [Matrix Times Vector](problems/matrix-vector/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 21 | [Рекорды тепла](problems/heat-records/statement.md) | [Heat Records](problems/heat-records/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 22 | [Диск энкодера](problems/gray-code/statement.md) | [Encoder Disc](problems/gray-code/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 23 | [Квадрат из дронов](problems/integer-sqrt/statement.md) | [Drone Square](problems/integer-sqrt/statement.en.md) | обычная | medium | ✓ | ✓ | ✓ | ✓ | ✓ |
| 24 | [Турнирная таблица](problems/lower-bound/statement.md) | [Tournament Table](problems/lower-bound/statement.en.md) | обычная | medium | ✓ | ✓ | ✓ | ✓ | ✓ |
| 25 | [Одно поколение](problems/game-of-life/statement.md) | [One Generation](problems/game-of-life/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 26 | [С одной пересадкой](problems/min-plus-product/statement.md) | [One Change](problems/min-plus-product/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 27 | [Удачная полоса](problems/max-subarray/statement.md) | [Best Stretch](problems/max-subarray/statement.en.md) | обычная | medium | ✓ | ✓ | ✓ | ✓ | ✓ |
| 28 | [Чихотная трава](problems/fibonacci/statement.md) | [Sneezewort](problems/fibonacci/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 29 | [Плитки карты](problems/morton-code/statement.md) | [Map Tiles](problems/morton-code/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 30 | [Таблица простых](problems/sieve/statement.md) | [Table of Primes](problems/sieve/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 31 | [Двойная экспозиция](problems/alpha-blend/statement.md) | [Double Exposure](problems/alpha-blend/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| 32 | [Перепись в Клетчатом](problems/summed-area/statement.md) | [Census in Gridville](problems/summed-area/statement.en.md) | обычная | medium | ✓ | ✓ |   | ✓ | ✓ |
| 33 | [Таблица замены](problems/substitution-cipher/statement.md) | [Substitution Table](problems/substitution-cipher/statement.en.md) | обычная | easy | ✓ | ✓ | ✓ | ✓ | ✓ |
| | **Всего задач** | | | | **27** | **25** | **21** | **30** | **30** |

## Формат пакета

```
problems/vector-addition/
  statement.md     условие на русском с YAML-шапкой
  statement.en.md  условие на английском, в шапке только название
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
  `authors` (ники на платформе; черновик `draft: true` видят только они), затем условие в Markdown на русском.
- `statement.en.md`: то же условие на английском; в шапке только `title`. Оба файла обязательны: сайт показывает
  название и условие на языке интерфейса. Два текста должны говорить одно и то же — те же разделы, числа и пример.
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
  Тексты `format.json` тоже на двух языках: `explanation` и `benchmark.inputDescription` — русские, `explanationEn`
  и `benchmark.inputDescriptionEn` — английские.
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
