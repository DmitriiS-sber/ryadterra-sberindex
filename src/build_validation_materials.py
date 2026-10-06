"""Create Russian methodology report and reproducible data for the updated slides."""

from pathlib import Path
import json, html
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from config import load_config
from forecast import past_fill, prophet_job
from validation_diagnostics import wilson

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "prophet_auto": "Prophet (5 точек)",
    "prophet_default": "Prophet default",
    "prophet_monthly": "Prophet сезонный",
    "damped_trend": "Затухающий тренд",
    "last_value": "Последнее значение",
    "seasonal_naive": "Прошлый год",
    "pooled_ridge": "Ridge",
    "pooled_ridge_news": "Ridge + новости",
    "chronos_bolt_tiny": "Chronos-Bolt Tiny",
}
METHODS = {
    "cusum": "CUSUM",
    "page_hinkley": "Page-Hinkley",
    "ewma": "EWMA",
    "shewhart": "Shewhart",
}


def f(x, d=0):
    if x is None or pd.isna(x):
        return "не определён"
    return f"{float(x):,.{d}f}".replace(",", " ").replace(".", ",")


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main():
    out = ROOT / "report"
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    val = ROOT / "results/validation"
    met = pd.read_csv(val / "forecast_by_horizon.csv")
    eq = pd.read_csv(val / "equal_territory_metrics.csv")
    origins = pd.read_csv(val / "forecast_by_origin_horizon.csv")
    ci = pd.read_csv(val / "conditional_cluster_intervals.csv")
    compare = pd.read_csv(val / "territory_comparison.csv")
    stress = pd.read_csv(val / "detector_stress_grid.csv", keep_default_na=False)
    syn = pd.read_csv(
        ROOT / "results/detection/synthetic_metrics.csv", keep_default_na=False
    )
    null = pd.read_csv(ROOT / "results/detection/real_protocol_null_test.csv")
    news = pd.read_csv(val / "news_detector_ablation.csv")
    directions = pd.read_csv(ROOT / "results/future_warning/direction_metrics.csv")
    alerts = pd.read_csv(ROOT / "results/detection/real_alerts.csv")
    real = pd.read_parquet(ROOT / "results/detection/real_innovations.parquet")
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    cfg = load_config(ROOT / "configs/validation.yaml")
    for table, ncol in [(null, "n"), (syn, "n")]:
        bounds = [
            wilson(round(r.any_alarm_pct * r.n / 100), int(r.n))
            for r in table.itertuples()
        ]
        table["alarm_wilson_lower_pct"] = [b[0] for b in bounds]
        table["alarm_wilson_upper_pct"] = [b[1] for b in bounds]
    null.to_csv(val / "real_null_uncertainty.csv", index=False)
    syn.to_csv(val / "synthetic_uncertainty.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    # Re-fit two illustrative one-step Prophet paths only. Main comparison is untouched.
    cases = []
    case_rows = []
    original = (
        pd.read_csv(
            ROOT.parent / "recovered/sberindex_task2/results/detection/real_alerts.csv"
        )
        if (
            ROOT.parent / "recovered/sberindex_task2/results/detection/real_alerts.csv"
        ).exists()
        else pd.DataFrame()
    )
    if not original.empty:
        original.to_csv(val / "original_protocol_real_alerts.csv", index=False)
    else:
        original = pd.read_csv(val / "original_protocol_real_alerts.csv")
    for tid in [322, 319]:
        actual = panel.loc[tid].to_numpy(float)
        ridge = (
            real[real.territory_id.eq(tid)]
            .set_index("date")
            .reindex(panel.columns[12:])
            .prediction.to_numpy()
        )
        prop = []
        for t in range(12, 24):
            _, res, errors = prophet_job(
                (
                    tid,
                    past_fill(actual[None, :t])[0],
                    pd.to_datetime(panel.columns[:t]),
                    1,
                    cfg,
                )
            )
            if errors:
                raise RuntimeError(errors)
            prop.append(float(res["prophet_auto"][0]))
        last = actual[11:23]
        months = list(panel.columns[12:])
        frame = pd.DataFrame(
            dict(
                territory_id=tid,
                date=months,
                actual=actual[12:],
                ridge=ridge,
                last_value=last,
                prophet=prop,
            )
        )
        frame.to_csv(val / f"case_{tid}_baselines.csv", index=False)
        case_rows.extend(records(frame))
        target = "2024-06" if tid == 322 else "2024-07"
        row = frame[frame.date.eq(target)].iloc[0]
        cases.append(
            dict(
                territory_id=tid,
                illustrative_month=target,
                actual_rub=float(row.actual),
                ridge_rub=float(row.ridge),
                last_value_rub=float(row.last_value),
                prophet_rub=float(row.prophet),
                ridge_deviation_pct=float(100 * (row.actual / row.ridge - 1)),
                last_value_deviation_pct=float(100 * (row.actual / row.last_value - 1)),
                prophet_deviation_pct=float(100 * (row.actual / row.prophet - 1)),
                economic_onset=None,
                published_at=None,
                original_alerts=records(original[original.territory_id.eq(tid)]),
                corrected_alerts=records(alerts[alerts.territory_id.eq(tid)]),
            )
        )
        chart, ax = plt.subplots(figsize=(8.2, 3.3))
        x = np.arange(12)
        ax.plot(x, actual[12:], "-o", color="#11394b", label="Факт", lw=2)
        ax.plot(x, ridge, "--", color="#ad7146", label="Ridge, 1 месяц")
        ax.plot(x, last, ":", color="#888888", label="Последнее значение")
        ax.plot(x, prop, "-.", color="#00896b", label="Prophet, 1 месяц")
        old = original[original.territory_id.eq(tid)]
        new = alerts[alerts.territory_id.eq(tid)]
        for source, label, style, c in [
            (old, "Старые сигналы", "--", "#ab83b5"),
            (new, "Исправленный протокол", ":", "#df9a36"),
        ]:
            for k, date in enumerate(sorted(source.first_alarm.unique())):
                ax.axvline(
                    months.index(date),
                    color=c,
                    ls=style,
                    alpha=0.6,
                    label=label if k == 0 else None,
                )
        ax.set(
            xticks=x,
            xticklabels=[m[5:] for m in months],
            xlabel="Месяц наблюдения 2024, дата публикации неизвестна",
            ylabel="Средние расходы, руб.",
        )
        ax.legend(loc="upper left", fontsize=7, ncol=3)
        ax.grid(alpha=0.13)
        chart.tight_layout()
        chart.savefig(fig / f"case_{tid}.png", dpi=175)
        plt.close(chart)
    selected = stress[
        (stress.kind == "null")
        & (stress.missing_fraction == 0)
        & (stress.threshold_multiplier == 1)
    ]
    chart, ax = plt.subplots(figsize=(8, 3))
    for method in METHODS:
        s = selected[selected.method.eq(method)].sort_values("phi")
        ax.plot(s.phi, s.any_alarm_pct, "o-", label=METHODS[method])
    ax.set(
        xlabel="Автокорреляция phi",
        ylabel="Ряды с хотя бы одной тревогой, %",
        xticks=[0, 0.3, 0.6],
    )
    ax.axhline(5, color="#999999", ls=":", label="Цель 5%")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.13)
    chart.tight_layout()
    chart.savefig(fig / "validation_stress.png", dpi=175)
    plt.close(chart)
    # Report content uses explicit page breaks to retain readable scientific tables.
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("DV", font))
    pdfmetrics.registerFont(TTFont("DVB", bold))
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="RU",
            fontName="DV",
            fontSize=9.3,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor("#203748"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H",
            fontName="DVB",
            fontSize=19,
            leading=25,
            spaceAfter=18,
            textColor=colors.HexColor("#123b4a"),
        )
    )
    styles.add(ParagraphStyle(name="T", fontName="DV", fontSize=7.7, leading=10))
    story = []
    md = []

    def para(text):
        story.append(Paragraph(text, styles["RU"]))
        md.append(text.replace("<b>", "").replace("</b>", ""))

    def heading(title, new=True):
        if new and story:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["H"]))
        md.append("\n## " + title + "\n")

    def table(rows, widths=None):
        data = [
            [Paragraph(html.escape(str(c)), styles["T"]) for c in row] for row in rows
        ]
        t = Table(
            data,
            colWidths=widths or [505 / len(rows[0])] * len(rows[0]),
            repeatRows=1,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5efed")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#a4beb4")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f4f7f7")],
                    ),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])
        md.extend(["| " + " | ".join(map(str, row)) + " |" for row in rows])

    heading("СберИндекс: прогноз расходов и сигналы изменений", False)
    para(
        "<b>Задача №2. Версия от 5 октября 2026.</b> Исследование прогнозирует средние расходы на горизонтах 1, 3, 6 и 12 месяцев и отдельно исследует предупреждение и обнаружение изменений. Код, настройки, таблицы и ограничения входят в полный комплект."
    )
    para(
        "Восемь конфигураций без Chronos повторно обучены при независимой проверке: 194 520 прогнозов совпали с сохранёнными результатами. Chronos здесь проверен по сохранённым прогнозам. В этой версии исправлены выбор пересмотра в asof.py и зависимости сборки. Исторические прогнозы не переоптимизировались; они не используют asof.py. Публикационные задержки исторической базы неизвестны, поэтому реальная доступность данных на origins не доказана."
    )
    para(
        "<b>Вывод по прогнозу.</b> В ретроспективной выборке Prophet имеет наименьшую MAE на горизонтах 1/3/6 месяцев. По отдельным датам он проигрывает последнему значению. Различие с Prophet default практически отсутствует. На 12 месяцах выигрывает затухающий тренд, но доступна одна дата."
    )
    para(
        "<b>Вывод по сигналам.</b> Согласование длины калибровки и параметра PH, пропуск обновлений при NaN и добавление Shewhart дают 108 первых сигналов метод-территория в 54 территориях. Независимых меток реальных шоков нет, поэтому их precision/recall неизвестны."
    )
    para(
        "<b>Независимость теста.</b> Модель выбиралась с учётом результатов 2024. Этот период уже просмотрен и остаётся ретроспективным. Замена названия периода не создаёт независимый holdout. Код будущего теста фиксируется после исправлений; для проверки нужны новые наблюдения и доказуемо заранее выданные прогнозы."
    )
    para(
        "История заканчивается декабрём 2024. Старый экспорт 2025 не является текущим прогнозом на октябрь 2026. Публикация в GitHub пока не выполнена. Полный статус 14 групп замечаний: docs/VALIDATION_TASKS.md."
    )
    heading("Данные, время и география")
    table(
        [
            ["Характеристика", "Значение"],
            ["Исходные строки / территории / категории", "303 126 / 2 190 / 6"],
            ["Период / категория", "2023-01 .. 2024-12 / Все категории"],
            ["Отбор по полному 2023", "2 075 территорий"],
            ["Хотя бы одна тестовая цель", "2 032 территории"],
            ["Пропуски 2024", "575 ячеек в 59 территориях"],
            [
                "Географический справочник",
                "ID источника, соответствие OKTMO неизвестно",
            ],
        ],
        [220, 285],
    )
    para(
        "Показатель: модельная оценка средних номинальных месячных безналичных расходов жителей, а не суммарный оборот территории. Охват, изменения измерения и причины пропусков требуют первичной документации. Отбор зависит только от 2023; отсутствующие тестовые цели исключаются одинаково для всех моделей."
    )
    table(
        [
            ["Поле реестра", "Статус текущей истории"],
            ["observation_month", "Известен месяц расходов"],
            ["published_at", "Неизвестен, оставлен пустым"],
            ["retrieved_at", "2026-10-03, точность до дня по исходному manifest"],
            ["vintage_id", "ID текущего снимка из SHA-256"],
            ["Исторические пересмотры", "Не предоставлены"],
        ],
        [220, 285],
    )
    para(
        "asof.py допускает только версии с published_at и retrieved_at не позднее cutoff и выбирает последнюю по времени публикации пересмотра. Поздно скачанная старая версия не вытесняет новую. Неизвестные даты, повторные ключи и неоднозначный порядок пересмотров отклоняются. Текущий temporal_registry намеренно не проходит реальный as-of тест. Сценарий задержки на месяц остаётся анализом чувствительности, а не доказательством доступности."
    )
    para(
        "При повторной проверке 4 октября архив и справочник СберИндекса вернули HTTP 502. Полная таблица попыток: source_recheck.json. Названия территорий и даты публикаций не восстановлены по предположениям."
    )
    heading("Как строится прогноз")
    para("<b>1. История и разбиения.</b> Для каждой территории используются только месяцы до origin включительно. Origins: декабрь 2023, март, июнь и сентябрь 2024. Горизонт h — отдельный месяц t+h, не сумма за h месяцев. Пропуски внутри истории заполняются только предыдущим значением; отсутствующие тестовые цели исключаются одинаково для моделей.")
    para("<b>2. Основной Prophet.</b> Модель обучается отдельно на каждой территории. В проверенных окнах истории меньше двух лет: автоматическая годовая сезонность выключена; недельная и дневная также выключены. Нет инфляции, погоды, новостей или праздников. Поэтому prophet_auto фактически продолжает сглаженный кусочно-линейный тренд средних номинальных расходов.")
    para("Тренд можно записать как g(t) = (k + Σ aⱼ(t)δⱼ)t + m − Σ aⱼ(t)sⱼδⱼ, где sⱼ — пять возможных точек смены наклона, aⱼ(t)=1 после sⱼ, δⱼ — изменение наклона. Непрерывность линии обеспечивается поправкой −sⱼδⱼ. Возможные точки не являются подтверждёнными экономическими событиями.")
    para("Параметры оцениваются методом MAP в Stan: нормальная модель ошибки приближает линию к наблюдениям, а распределение Лапласа для δⱼ штрафует большие изменения наклона. При фиксированных масштабе ошибки и масштабировании данных это соответствует сумме квадратов ошибок плюс L1-штраф λΣ|δⱼ| и остальным априорным членам. В коде задаётся changepoint_prior_scale=0.05; его нельзя отождествлять с λ в рублях. Будущий прогноз продолжает последний наклон; новые будущие разрывы заранее не определяются. Значение снизу ограничено одним рублём.")
    para("<b>3. Сравнения.</b> Последнее значение повторяет y(t), сезонный наивный — месяц прошлого года. Затухающий тренд оценивает линейный наклон логарифма за шесть месяцев, ограничивает его ±0.03 и уменьшает дальнейшие приращения с коэффициентом 0.85. Prophet monthly добавляет три гармоники Фурье годовой сезонности. Это другая конфигурация, её результаты показаны отдельно.")
    para("<b>4. Ridge.</b> Общая для территорий регрессия прогнозирует месячный прирост логарифма. Признаки: лаги 1/2/3/6, среднее и стандартное отклонение шести месяцев, разность лагов 1 и 3, четыре календарных sin/cos. Масштаб территории — геометрическое среднее первых шести наблюдений. StandardScaler и Ridge обучаются внутри origin. Цель: сумма квадратов ошибок плюс 20·||β||². Прогноз рекурсивный; прирост логарифма ограничен ±0.25. В новостной версии добавлены четыре признака из прошлых сообщений ЦБ, на будущие шаги замороженные на origin.")
    para("<b>5. Сигналы.</b> CUSUM, Page–Hinkley, EWMA и Shewhart обновляются по последовательно получаемым остаткам Ridge. Это обнаружение начавшегося изменения. Отдельный ранний индикатор использует будущую прогнозную траекторию и proxy 2 из 3 месяцев. Он не является независимо размеченным экономическим шоком. Подробные параметры: configs/experiment.yaml и configs/validation.yaml; реализация: src/forecast.py, src/detect.py, src/future_warning.py.")
    heading("Прогнозные модели и метрики")
    wide = met.pivot(index="model", columns="horizon", values="mae_rub")
    table(
        [["Модель", "MAE h1", "MAE h3", "MAE h6", "MAE h12"]]
        + [[NAMES[m]] + [f(wide.loc[m, h]) for h in [1, 3, 6, 12]] for m in NAMES],
        [173, 83, 83, 83, 83],
    )
    para(
        "MAE измеряется в рублях, равный вес имеют доступные пары территория-origin-target. Число пар на модель: 8 111 / 8 106 / 6 075 / 2 023. Число origins: 4 / 4 / 3 / 1. Пропуски дают разный вклад отдельных территорий."
    )
    table(
        [["h", "MAE пар", "MAE территорий", "WAPE, %", "MASE lag1", "Within R²"]]
        + [
            [
                h,
                f(met.query('model=="prophet_auto" and horizon==@h').mae_rub.iloc[0]),
                f(
                    eq.query(
                        'model=="prophet_auto" and horizon==@h'
                    ).equal_territory_mae_rub.iloc[0]
                ),
                f(
                    met.query('model=="prophet_auto" and horizon==@h').wape_pct.iloc[0],
                    2,
                ),
                f(
                    eq.query(
                        'model=="prophet_auto" and horizon==@h'
                    ).mase_lag1_equal_territories.iloc[0],
                    3,
                ),
                f(
                    met.query(
                        'model=="prophet_auto" and horizon==@h'
                    ).within_territory_r2.iloc[0],
                    3,
                ),
            ]
            for h in [1, 3, 6, 12]
        ],
        [30, 82, 98, 74, 84, 137],
    )
    para(
        "MAE территорий = среднее территориальных MAE. WAPE = сумма абсолютных ошибок / сумма фактических значений. MASE использует масштаб средних абсолютных месячных изменений каждой территории только в 2023, лаг 1. Seasonal MASE с лагом 12 не определён на первой 12-месячной истории."
    )
    para(
        "Pooled R² Prophet составляет 0,92-0,94 и в значительной мере отражает различия уровней между территориями. Within R² использует центрирование фактов внутри территории; отрицательное значение h6 показывает слабость прогноза динамики. h12 не определён: одна цель на территорию."
    )
    heading("Неоднородность ошибок")
    rows = [["Origin, h1", "Prophet MAE", "Последнее значение MAE"]]
    for origin in sorted(origins.origin.unique()):
        s = origins[(origins.origin == origin) & (origins.horizon == 1)].set_index(
            "model"
        )
        rows.append(
            [
                origin,
                f(s.loc["prophet_auto", "mae_rub"]),
                f(s.loc["last_value", "mae_rub"]),
            ]
        )
    table(rows, [165, 170, 170])
    para(
        "Апрельская и июльская цели h1 хуже последнего значения для Prophet. Общая победа не означает превосходства в каждом периоде или каждой территории. Квартили расходов определены только по 2023."
    )
    table(
        [["h", "С чем сравниваем", "Территорий, где Prophet хуже, %"]]
        + [
            [r.horizon, NAMES[r.reference], f(r.territories_prophet_worse_pct, 2)]
            for r in compare.itertuples()
        ],
        [50, 205, 250],
    )
    table(
        [["h", "Median MAE территории", "p90 MAE", "p99 MAE"]]
        + [
            [
                r.horizon,
                f(r.median_territory_mae),
                f(r.p90_territory_mae),
                f(r.p99_territory_mae),
            ]
            for r in eq[eq.model.eq("prophet_auto")].itertuples()
        ],
        [40, 200, 130, 135],
    )
    para(
        "Полные разрезы: forecast_by_origin_horizon.csv, forecast_by_territory.csv.gz, forecast_by_quartile.csv. Структура пропусков: missing_cells.csv. Причины отсутствия фактов неизвестны; их нельзя трактовать как случайные без внешней проверки."
    )
    heading("Неопределённость сравнений")
    selected_ci = ci[
        ci.horizon.lt(12) & ci.reference.isin(["last_value", "prophet_default"])
    ]
    table(
        [["h", "Сравнение", "Δ MAE, руб.", "Условный 95% интервал"]]
        + [
            [
                r.horizon,
                "Prophet - " + NAMES[r.reference],
                f(r.difference_mae_rub, 2),
                f(r.conditional_lower_95, 1) + " .. " + f(r.conditional_upper_95, 1),
            ]
            for r in selected_ci.itertuples()
        ],
        [30, 210, 90, 175],
    )
    para(
        "Двухфакторный bootstrap: территории и origins выбираются независимо с возвращением, 2000 повторов, фиксированный seed. Интервалы описывают сохранённую выборку. При трёх-четырёх временных кластерах нельзя заявлять надёжную временную статистическую значимость или переносимость на другой режим."
    )
    para(
        "Интервалы различий Prophet - последнее значение на h1/h3/h6 включают ноль. Различия с Prophet default также включают ноль. h12 имеет только один origin, поэтому временной интервал не рассчитывается."
    )
    para(
        "Ни bootstrap ошибки средней MAE, ни разброс по территориям не являются прогнозным интервалом индивидуального расхода. В историческом Prophet uncertainty_samples=0. Квантили Chronos не проходили оценку покрытия. Калибровка прогнозных интервалов требует отдельной новой истории."
    )
    para(
        "Chronos-Bolt выпущен 26 ноября 2024, позже всех origins (последний 2024-09). Его результат можно использовать как современный ретроспективный benchmark. Историческая исполнимость на этих датах и отсутствие пересечения предобучения с тестом не доказаны. Источник даты выпуска: github.com/amazon-science/chronos-forecasting, раздел News, проверен 4 октября 2026."
    )
    heading("Последовательные детекторы")
    table(
        [["Метод", "Recall step 2σ, %", "Delay, мес.", "Null 48 мес., % (95% Wilson)"]]
        + [
            [
                METHODS[m],
                f(
                    syn[(syn.model == m) & (syn.kind == "step")].recall_12m_pct.iloc[0],
                    1,
                ),
                f(
                    syn[
                        (syn.model == m) & (syn.kind == "step")
                    ].median_delay_months.iloc[0]
                ),
                f(syn[(syn.model == m) & (syn.kind == "null")].any_alarm_pct.iloc[0], 1)
                + " ("
                + f(
                    syn[
                        (syn.model == m) & (syn.kind == "null")
                    ].alarm_wilson_lower_pct.iloc[0],
                    1,
                )
                + " .. "
                + f(
                    syn[
                        (syn.model == m) & (syn.kind == "null")
                    ].alarm_wilson_upper_pct.iloc[0],
                    1,
                )
                + ")",
            ]
            for m in METHODS
        ],
        [100, 105, 75, 225],
    )
    para(
        "Старый синтетический опыт сохранён: 300 null для порогов, 150 новых рядов на тип, 12 warmup и 48 контролируемых месяцев. Для реального применения выполнена отдельная калибровка на 12 месяцев с таким же PH reference_count=12."
    )
    table(
        [["Метод", "Null 12 мес., % (95% Wilson)", "Первых сигналов в реальных рядах"]]
        + [
            [
                METHODS[r.model],
                f(r.any_alarm_pct, 1)
                + " ("
                + f(r.alarm_wilson_lower_pct, 1)
                + " .. "
                + f(r.alarm_wilson_upper_pct, 1)
                + ")",
                int(alerts.model.eq(r.model).sum()),
            ]
            for r in null.itertuples()
        ],
        [110, 220, 175],
    )
    para(
        "Проверка 12-месячных порогов использует 500 новых null-рядов, отдельный seed. Частота относится к синтетическому AR(1), а не к реальным экономическим шокам. Для отсутствующего факта NaN не обновляет состояние и счётчик PH, тревога запрещена."
    )
    para(
        "Масштаб реальных инноваций по-прежнему оценён по пяти остаткам августа-декабря 2023 с общим нижним пределом. Эта короткая оценка остаётся ограничением. Новый протокол даёт 108 первых сигналов в 54 территориях; исходный протокол давал восемь в четырёх. Менялись длина калибровки, PH reference_count и состав методов, поэтому это не эффект только исправления NaN."
    )
    heading("Стресс-проверки синтетической устойчивости")
    story.append(Image(str(fig / "validation_stress.png"), width=505, height=190))
    para(
        "500 рядов на сценарий, phi 0/0,3/0,6, пропуски 0/10%, сдвиги 0,5/1/2 sigma, сезонность, выбросы, порог 0,8/1/1,2. Эти сценарии показывают зависимость от предположений. Порог не выбирался по лучшему результату 2024."
    )
    step = stress[
        (stress.kind == "step")
        & (stress.phi == 0.3)
        & (stress.missing_fraction == 0)
        & (stress.threshold_multiplier == 1)
    ]
    table(
        [["Метод", "Recall 0,5σ, %", "Recall 1σ, %", "Recall 2σ, %"]]
        + [
            [METHODS[m]]
            + [
                f(
                    step[
                        (step.method == m) & (step.magnitude_sigma == mag)
                    ].recall_12m_pct.iloc[0],
                    1,
                )
                for mag in [0.5, 1, 2]
            ]
            for m in METHODS
        ],
        [145, 120, 120, 120],
    )
    para(
        "Wilson-интервалы оценивают Monte Carlo неопределённость доли рядов с хотя бы одной тревогой в фиксированном синтетическом сценарии. Они не устраняют ошибку спецификации шума. Полная сетка из 432 строк: detector_stress_grid.csv."
    )
    heading("Ранние предупреждения и направление")
    d = directions[
        (directions.threshold_pct == 20) & directions.status.eq("calculated")
    ]
    table(
        [["Модель", "Precision", "Recall", "F1", "Неверное направление"]]
        + [
            [
                NAMES[r.model],
                f(r.precision, 3),
                f(r.recall, 3),
                f(r.f1, 3),
                int(r.wrong_direction),
            ]
            for r in d.itertuples()
        ],
        [200, 73, 73, 73, 86],
    )
    para(
        "Proxy: минимум два из трёх будущих месяцев отличаются от известного прошлогоднего уровня на порог в одном направлении. Дополнительная оценка требует совпадения фактического и прогнозного направления. Неверное направление даёт FP и FN. Макрофлаг не прогнозирует направление и исключён из этой таблицы."
    )
    table(
        [["Порог, %", "Событий proxy", "Precision Prophet", "Recall", "F1"]]
        + [
            [
                r.threshold_pct,
                int(r.tp_correct_direction + r.fn_including_wrong_direction),
                f(r.precision, 3),
                f(r.recall, 3),
                f(r.f1, 3),
            ]
            for r in directions[directions.model.eq("prophet_auto")].itertuples()
        ],
        [65, 110, 140, 95, 95],
    )
    para(
        "Пороги 10/20/30% задают разные задачи: инфляция, низкая база и изменения измерения могут влиять на proxy. Ни одна из таблиц не доказывает прогнозирование независимых экономических шоков. Для них нужны реальные onset_at, published_at, direction и внешняя независимая проверка."
    )
    heading("Новости: первичные источники и абляция")
    table(
        [["Метод", "Без новостей: кандидатов", "С новостями: кандидатов"]]
        + [
            [
                METHODS[m],
                int(
                    news[
                        (news.method == m) & (~news.use_news)
                    ].territories_with_candidate_alert.iloc[0]
                ),
                int(
                    news[
                        (news.method == m) & news.use_news
                    ].territories_with_candidate_alert.iloc[0]
                ),
            ]
            for m in METHODS
        ],
        [145, 180, 180],
    )
    para(
        "Обе ветви используют expanding Ridge, пять остаточных значений для центра/MAD, PH reference_count=12, одинаковые 12-месячные пороги и политику пропусков. Меняются только четыре датированных новостных признака. Сравниваются количества кандидатов, а не подтверждённая точность. Большое число Shewhart-кандидатов в новостной ветви показывает нестабильность остаточной модели."
    )
    para(
        "17 сообщений ЦБ являются 17 независимыми макросигналами, повторяемыми по территориям. Увеличение числа муниципальных строк не увеличивает число независимых публикаций. В исходном прогнозном сравнении новости ухудшили Ridge. Удалять отрицательный результат нельзя."
    )
    recheck = json.loads((val / "source_recheck.json").read_text())
    para(
        f'Повторно получено {sum("sha256"in r for r in recheck[:17])} из 17 страниц ЦБ; все текущие HTML-хеши и два проверяемых словарных флага совпадают с сохранёнными метаданными (флаги forward_tightening и demand_pressure). Это проверка текущей версии, историческая неизменность страниц неизвестна. Полные тексты не распространяются в комплекте.'
    )
    para(
        "Следующая задача: длинный корпус региональных публикаций с датами, версиями и независимыми событиями. Необходимо заранее определить географическую привязку и split по времени. Текущий корпус не подтверждает муниципальный эффект или причинную связь."
    )
    heading("Реальные случаи и зависимость от базовой модели")
    for case in cases:
        tid = case["territory_id"]
        story.append(Image(str(fig / f"case_{tid}.png"), width=505, height=203))
        para(
            f'<b>ID {tid}, {case["illustrative_month"]}.</b> Факт {f(case["actual_rub"])} руб. Отклонение от Ridge {f(case["ridge_deviation_pct"],2)}%, последнего значения {f(case["last_value_deviation_pct"],2)}%, Prophet {f(case["prophet_deviation_pct"],2)}%. Величина необычности зависит от базового прогноза.'
        )
    para(
        "Вертикальные линии показывают месяцы наблюдения первого сигнала каждого протокола. Опубликованная дата расхода и экономическое onset неизвестны. Это кандидаты, а не доказанные причинные шоки. Накопленный CUSUM может сигнализировать в месяце с малой текущей инновацией. Прогнозные интервалы для этих линий отсутствуют."
    )
    heading("Будущий тест и критерии реальной проверки")
    para(
        "prospective_lock.json фиксирует code/YAML hash, модель Prophet, горизонты и метрики после исправлений. До первых прогнозов требуется register с полным списком будущих ключей. evaluate требует тот же код, модель, все зарегистрированные цели, выдачу до целевого месяца и публикацию факта после прогноза. Повторная оценка блокируется общим файлом prospective_scored.json, включая попытку записать другое имя результата."
    )
    para(
        "Локальные часы и SHA-256 помогают аудиту, но не доказывают ex-ante выдачу. Нужен внешний неизменяемый журнал прогнозов, настоящий release/vintage реестр и независимое подтверждение, что новые цели не использовались для настройки. Старый 2024 тест остаётся просмотренным."
    )
    para(
        "evaluate_events.py требует внешний полный реестр independently reviewed событий и фиксированную популяцию территорий. Оно сопоставляет самое раннее ещё не использованное предупреждение того же направления в пределах трёх месяцев до onset. Публикация события хранится отдельно. Предупреждение после onset не может быть ранним."
    )
    para(
        "Первые три месяца событий и последние три месяца предупреждений цензурируются отдельно. Метрики условны относительно объявленного окна и входной популяции. Код проверяет схемы и границы, но полноту и независимость устанавливает внешний reviewer. Пустой реестр останавливает оценку: реальная precision/recall остаётся неизвестной."
    )
    para(
        "Блокирующие данные: независимые будущие месяцы; публикации/версии расходов; onset/publication/direction реальных событий; валидированный справочник географии; длинный региональный новостной корпус. Прогнозные интервалы и стабильность во времени требуют новых периодов. Публикация GitHub требует подключения владельца."
    )
    para(
        "Все 14 групп замечаний, статусы и критерии завершения: docs/VALIDATION_TASKS.md. Инструкции и шаблоны: docs/VALIDATION_RUN.md, data/templates/. Исходные и исправленные численные проверки: results/verification.json и results/validation/verification.json."
    )
    heading("Воспроизводимость и источники")
    para(
        "Запуск исправлений из полного комплекта: установить requirements-core.txt, выполнить make validation. validation_diagnostics использует сохранённые прогнозы без refit/tuning; detect/news_detector заново рассчитывают остаточные прогнозы. build_materials создаёт этот отчёт. Параметры исправлений: validation.yaml. Полный исходный расчёт девяти моделей описан в docs/RUN.md."
    )
    table(
        [
            ["Артефакт", "Назначение"],
            [
                "forecast_by_horizon / equal_territory_metrics",
                "Агрегация, WAPE, MASE, within R²",
            ],
            [
                "forecast_by_origin_horizon / forecast_by_territory",
                "Неоднородность и ошибки",
            ],
            ["conditional_cluster_intervals", "Описательная условная неопределённость"],
            [
                "detector_stress_grid / real_null_uncertainty",
                "Синтетическая устойчивость и Monte Carlo интервалы",
            ],
            [
                "direction_metrics / news_detector_ablation",
                "Направление proxy и парная новостная абляция",
            ],
            [
                "temporal_registry / source_recheck",
                "Статус времени данных и первичных источников",
            ],
            [
                "VALIDATION_TASKS / VALIDATION_RUN",
                "Замечания, незакрытые задачи и запуск",
            ],
        ],
        [240, 265],
    )
    para(
        "Источники: официальное положение конкурса (docs/Official_Contest_Rules.pdf, задача №2 в приложении); официальный архив СберИндекса и лицензия CC BY-SA 4.0 в data/raw/data_license.pdf; первичные пресс-релизы ЦБ (17 индивидуальных URL в news_events.csv); github.com/amazon-science/chronos-forecasting; facebook.github.io/prophet/. Код MIT. Полные новости и веса foundation model не включены."
    )
    para(
        "GitHub не опубликован, конкурсная форма не отправлена. Входные прогнозы, диагностическая среда и SHA-256 сохранены в results/validation/run_manifest.json; хеши всех файлов архива в deliverable_manifest.json. Все результаты исследовательские; экономические причины сигналов не подтверждены."
    )

    def footer(canvas, doc):
        canvas.setFont("DV", 7)
        canvas.setFillColor(colors.HexColor("#627980"))
        canvas.drawString(44, 24, "СберИндекс · задача №2 · 05.10.2026")
        canvas.drawRightString(550, 24, str(doc.page))

    doc = SimpleDocTemplate(
        str(out / "SberIndex_Task2_Report.pdf"),
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=44,
        bottomMargin=44,
        title="СберИндекс: прогноз расходов и сигналы изменений",
        author="Исследовательский проект",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (out / "methodology.md").write_text("\n\n".join(md), encoding="utf-8")
    payload = dict(
        date="2026-10-05",
        forecast=records(met),
        equal=records(eq),
        origins=records(origins),
        conditional_intervals=records(ci),
        territory_comparison=records(compare),
        detectors=records(syn),
        null12=records(null),
        stress=records(stress),
        warnings=records(directions),
        news=records(news),
        cases=cases,
        case_paths=case_rows,
        real_alert_count=len(alerts),
        real_candidate_territories=alerts.territory_id.nunique(),
    )
    (ROOT / "presentation/slide_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    print("Report and slide data updated")


if __name__ == "__main__":
    main()
