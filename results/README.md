# Снимки агрегированных результатов

CSV и JSON перенесены из проверенного комплекта от 05.10.2026. Числа в
сохранённых таблицах не менялись. Это исторические результаты и манифесты,
а не новый полный расчёт при публикации GitHub.

Основные таблицы:

- `validation/forecast_by_horizon.csv`: MAE, RMSE, R², within R² и процентные ошибки девяти конфигураций;
- `validation/forecast_by_origin_horizon.csv`: метрики по датам отсечения;
- `validation/equal_territory_metrics.csv`: одинаковый вес территорий;
- `validation/detector_stress_grid.csv`: синтетические стресс-сценарии;
- `detection/synthetic_metrics.csv`: оценки детекторов на искусственных рядах;
- `future_warning/metrics.csv`, `future_warning/direction_metrics.csv`: предупреждения по proxy, включая направление;
- `validation/news_detector_ablation.csv`: сравнение остаточных детекторов с новостями и без них.

Сохранённые `verification.json` и `validation/verification.json` относятся
к предыдущим проверкам. `release_verification.json` пересоздаётся командой
`make verify-release`; проверки публикации описаны в `docs/CHECKS_20261006.md`.

Таблицы индивидуальных наблюдений, сигналов и прогнозов, тяжёлые Parquet
прогнозов/траекторий, gzip по территориям и старый `latest_forecast.csv`
не включены. Для `make validation` используйте полный комплект; для короткого
нового расчёта скопируйте подготовленные данные и выполните `make smoke-lite`.
Папка `results/smoke-lite/` исключена из Git.
