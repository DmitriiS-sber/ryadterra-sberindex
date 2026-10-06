# Запуск исправлений и следующей проверки

Linux, Python 3.12. Команды выполняются из корня проекта. Полный архив содержит входные данные и сохранённые прогнозы. Архив исходников не содержит панель и прогнозы; для расчётов нужен полный комплект либо загрузка и полный исторический расчёт из RUN.md.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python src/detect.py --config configs/validation.yaml
python src/future_warning.py
python src/validation_diagnostics.py
python src/news_detector.py
python src/verify.py
python src/verify_validation.py
python src/verify_release.py
python src/build_materials.py
```

Эта последовательность пересчитывает остаточные Ridge-прогнозы и исправленные детекторы. Метрики прогнозных моделей вычисляются по сохранённым predictions.parquet: повторного подбора и полного обучения девяти моделей нет. Повторение полного forecast/Chronos-расчёта описано в RUN.md. Повторная интернет-проверка первичных новостей необязательна для численного пересчёта и обновляет время проверки:

```bash
python src/recheck_sources.py
```

Дополнительные параметры и seed находятся в `configs/validation.yaml`. Стресс-тесты используют исходные 48-месячные синтетические пороги, а реальные детекторы используют отдельные 12-месячные пороги `real_thresholds.json`. При изменении конфигурации результаты являются новым исследовательским экспериментом; он не должен перезаписывать опубликованный будущий lock/plan.

## Данные, доступные на конкретную дату

Заполнить `data/templates/observation_vintages.csv` настоящими публикациями и версиями. Формат времени ISO 8601 с часовым поясом. `retrieved_at` относится к фактическому сохранению конкретного снимка, `published_at` к публикации именно этой версии значения (пересмотра). Дата месяца наблюдения не подставляется вместо публикации. Реестр текущей истории с неизвестным published_at намеренно не проходит этот режим.

```bash
python src/asof.py --input my_observation_vintages.csv --cutoff 2026-11-15T12:00:00+03:00 --output data/processed/asof_2026_11.parquet
python src/predict.py --input data/processed/asof_2026_11.parquet --origin 2026-10 --model prophet_auto --output results/issued_2026_11.csv
```

Числа дат выше иллюстрируют формат, а не наличие таких данных. Если месяц origin ещё не опубликован, выбрать последний доступный месяц согласно заранее зарегистрированному правилу. Панель для predict.py должна иметь не менее 12 последовательных месячных колонок. Прогнозный файл содержит origin, target_month, horizon, prediction, issued_at и hash входного файла. Для ex-ante доказательства сохранить его во внешнем неизменяемом журнале сразу при выдаче.

## Независимый будущий тест

Код уже зафиксирован в `configs/prospective_lock.json`. Это фиксация после просмотра 2024, а не доказательство независимости исторического теста. Не запускать freeze повторно и не перезаписывать lock.

До прогнозов заполнить шаблон фиксированных тестовых ключей и зарегистрировать его. Все целевые месяцы должны находиться в будущем относительно регистрации. Затем собирать полный набор прогнозов и фактических целей с published_at/vintage_id.

```bash
python src/prospective.py register --plan my_prospective_test_keys.csv
python src/prospective.py evaluate --predictions my_issued_predictions.csv --actuals my_released_actuals.csv --output results/prospective_score.json
```

Оценка запрещает смену кода/модели, несовпадение origin+horizon с target_month, выдачу после начала целевого месяца, неполную популяцию и ещё не опубликованные цели. При незавершённом тесте вывод остаётся «ожидает данных». Новый эксперимент требует нового заранее зарегистрированного проекта; существующий lock и результаты сохраняются.

## Проверка независимо размеченных событий

Заполнить пустые шаблоны independent_events.csv, issued_warnings.csv, surveillance_population.csv. direction=-1/+1, independent_review=true. Полноту и независимость реестра устанавливает внешний reviewer: код не может доказать их по флагу. Запуск с пустыми событиями останавливается, а не выдаёт нулевую частоту шоков.

```bash
python src/evaluate_events.py --events my_events.csv --warnings my_warnings.csv --population my_population.csv --window-start 2027-01-01T00:00:00Z --window-end 2029-01-01T00:00:00Z --adjudicated-at 2029-03-01T00:00:00Z --output results/independent_event_metrics.json
```

Правило: earliest unused warning того же направления в пределах трёх месяцев до onset. Сигналы после onset не являются ранними предупреждениями. Первые три месяца событий и последние три месяца предупреждений цензурируются отдельно; отчёт показывает обе численности и условную популяцию оценки. Неполный внешний реестр не допускает вывода о реальной точности.

### Версия 05.10.2026

Пересмотр выбирается по `published_at`, затем `retrieved_at`. Позднее скачивание старой версии не делает её новой. Разные `vintage_id` с одинаковым временем публикации отклоняются: для них нужен подтверждённый порядок версий. Старый lock сохранён в configs/archive; новый lock относится только к этой копии. Прогнозный тест не зарегистрирован и не выполнен.
