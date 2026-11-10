# GpuCodeforcesProblems

Банк задач GPU Codeforces. Задача добавляется или меняется pull request'ом; backend платформы
забирает ветку `master` и импортирует каждый каталог `problems/<id>/`. Имя каталога — постоянный ID
задачи (`[A-Za-z0-9._-]`), его нельзя менять после публикации.

```
problems/vector-addition/
  statement.md   условие с YAML-шапкой
  problem.json   описание задачи для судьи (без генератора)
  problem.py     генератор тестов и эталон
  starter.cu     шаблон решения (необязательно)
```

Полное описание формата — [docs/backend/problem-package.md](https://github.com/lottery7/GpuCodeforces/blob/master/docs/backend/problem-package.md)
и [judge/docs/problem-format.md](https://github.com/lottery7/GpuCodeforces/blob/master/judge/docs/problem-format.md)
в репозитории платформы. Кратко:

- `statement.md`: `code` (номер в банке), `title`, `difficulty` (`easy`/`medium`/`hard`), `tags`,
  `authors` (ники на платформе; черновик `draft: true` видят только они), затем условие в Markdown.
- `problem.json`: сигнатура `solve` (тип, роль и размер каждого массива, где он лежит — `DEVICE`/`HOST`),
  допуски сравнения, лимиты времени и памяти, флаги компиляции и запреты.
- `problem.py`: `tests()` перечисляет тесты (открытый пример — `sample: True` с рукописными входами и
  ответом), `generate()` создаёт входы, `reference()` считает эталонный ответ (Python + PyTorch).
- Без `starter.cu` шаблоном станет пустая `solve` с подсказкой.

Задача с ошибкой в пакете не попадёт в банк, причина видна в логе backend. Генератор можно проверить
локально (нужен torch), из корня репозитория платформы:

```sh
PYTHONPATH=judge/pytools python3 -m judge_problem.prepare --spec spec.json --source problem.py --out /tmp/out
```

где `spec.json` — `{"signature": ...}` из `problem.json`.
