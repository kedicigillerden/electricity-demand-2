from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TARGET_YEARS = [2022, 2023, 2024]
ALL_CONTINUOUS_COLUMNS = ["HDD", "CDD", "PMI", "IR", "CUR"]
BINARY_COLUMNS = ["weekend", "night"]

CONSUMPTION_FILES = {
    2022: RAW_DATA_ROOT / "consumption" / "tuketim_2022.xlsx",
    2023: RAW_DATA_ROOT / "consumption" / "tuketim_2023.xlsx",
    2024: RAW_DATA_ROOT / "consumption" / "tuketim_2024.xlsx",
}

TEMPERATURE_FILES = {
    2022: RAW_DATA_ROOT / "temperature" / "sicaklik_2022.xlsx",
    2023: RAW_DATA_ROOT / "temperature" / "sicaklik_2023.xlsx",
    2024: RAW_DATA_ROOT / "temperature" / "sicaklik_2024.xlsx",
}

ECONOMIC_FILES = {
    "PMI": RAW_DATA_ROOT / "economic" / "pmi_degerleri.xlsx",
    "IR": RAW_DATA_ROOT / "economic" / "faiz_oranlari.xlsx",
    "CUR": RAW_DATA_ROOT / "economic" / "kapasite_kullanim_orani.xlsx",
}

TURKISH_CHAR_MAP = str.maketrans(
    {
        "\u00e7": "c",
        "\u00c7": "C",
        "\u011f": "g",
        "\u011e": "G",
        "\u0131": "i",
        "\u0130": "I",
        "\u00f6": "o",
        "\u00d6": "O",
        "\u015f": "s",
        "\u015e": "S",
        "\u00fc": "u",
        "\u00dc": "U",
    }
)


@dataclass(frozen=True)
class ZScoreScaler:
    means: dict[str, float]
    stds: dict[str, float]
    columns: tuple[str, ...]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.columns:
            out[column] = (
                pd.to_numeric(out[column], errors="coerce") - self.means[column]
            ) / self.stds[column]
        return out


def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _normalize_label(value: object) -> str:
    text = str(value).translate(TURKISH_CHAR_MAP)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _rename_by_alias(
    df: pd.DataFrame,
    alias_map: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    normalized = {_normalize_label(column): column for column in df.columns}
    rename_map: dict[str, str] = {}
    for target, aliases in alias_map.items():
        for alias in aliases:
            original = normalized.get(alias)
            if original is not None:
                rename_map[original] = target
                break
    return df.rename(columns=rename_map)


def _excel_date_to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    as_object = series.astype("object")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    python_date_mask = as_object.map(
        lambda value: isinstance(value, (datetime, date, pd.Timestamp))
    )
    if python_date_mask.any():
        result.loc[python_date_mask] = pd.to_datetime(
            as_object.loc[python_date_mask], errors="coerce"
        )

    remaining = ~python_date_mask
    if remaining.any():
        numeric = pd.to_numeric(as_object.loc[remaining], errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            result.loc[numeric_mask.index[numeric_mask]] = (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(numeric.loc[numeric_mask], unit="D")
            )

        text_mask = ~numeric_mask
        if text_mask.any():
            result.loc[text_mask.index[text_mask]] = pd.to_datetime(
                as_object.loc[text_mask.index[text_mask]],
                errors="coerce",
                dayfirst=True,
            )
    return result


def _excel_time_to_string(series: pd.Series) -> pd.Series:
    as_object = series.astype("object")
    result = pd.Series("", index=series.index, dtype="object")

    datetime_mask = as_object.map(
        lambda value: isinstance(value, (datetime, pd.Timestamp))
    )
    if datetime_mask.any():
        parsed = pd.to_datetime(as_object.loc[datetime_mask], errors="coerce")
        result.loc[datetime_mask] = parsed.dt.strftime("%H:%M:%S")

    pure_time_mask = as_object.map(lambda value: isinstance(value, time))
    if pure_time_mask.any():
        result.loc[pure_time_mask] = as_object.loc[pure_time_mask].map(
            lambda value: value.strftime("%H:%M:%S")
        )

    remaining = ~(datetime_mask | pure_time_mask)
    if remaining.any():
        numeric = pd.to_numeric(as_object.loc[remaining], errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            result.loc[numeric_mask.index[numeric_mask]] = (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(numeric.loc[numeric_mask], unit="D")
            ).strftime("%H:%M:%S")

        text_index = numeric_mask.index[~numeric_mask]
        if len(text_index) > 0:
            text = as_object.loc[text_index].astype(str).str.strip()
            text = text.replace({"nan": "", "NaT": "", "None": ""})

            hours_only = text.str.match(r"^\d{1,2}$", na=False)
            text = text.where(~hours_only, text.str.zfill(2) + ":00:00")

            hhmm = text.str.match(r"^\d{1,2}:\d{2}$", na=False)
            text = text.where(
                ~hhmm,
                text.str.replace(
                    r"^(\d{1,2}:\d{2})$",
                    lambda match: match.group(1).zfill(5) + ":00",
                    regex=True,
                ),
            )

            hhmmss = text.str.match(r"^\d{1,2}:\d{2}:\d{2}$", na=False)
            text = text.where(
                ~hhmmss,
                text.str.replace(
                    r"^(\d{1,2}):",
                    lambda match: match.group(1).zfill(2) + ":",
                    regex=True,
                ),
            )
            result.loc[text_index] = text
    return result


def _build_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = _excel_date_to_datetime(date_series)
    times = _excel_time_to_string(time_series)
    return pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + times,
        errors="coerce",
    )


def _read_excel(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", **kwargs)


def load_consumption(year: int) -> pd.DataFrame:
    df = _read_excel(CONSUMPTION_FILES[year])
    df = _rename_by_alias(
        df,
        {
            "date": ("tarih",),
            "time": ("saat",),
            "consumption": ("tuketimmiktarimwh", "consumption"),
        },
    )
    if {"date", "time", "consumption"} - set(df.columns):
        df = df.rename(
            columns={
                df.columns[0]: "date",
                df.columns[1]: "time",
                df.columns[2]: "consumption",
            }
        )
    df["datetime"] = _build_datetime(df["date"], df["time"])
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    return (
        df[["datetime", "consumption"]]
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def load_temperature(year: int) -> pd.DataFrame:
    df = _read_excel(TEMPERATURE_FILES[year])
    df = _rename_by_alias(
        df,
        {
            "city": ("istasyonadi", "city"),
            "date": ("tarih", "date"),
            "time": ("saat", "time"),
            "temperature": ("sicaklik", "temperature"),
        },
    )
    if {"city", "date", "time", "temperature"} - set(df.columns):
        df = df.rename(
            columns={
                df.columns[1]: "city",
                df.columns[3]: "date",
                df.columns[4]: "time",
                df.columns[5]: "temperature",
            }
        )
    df["datetime"] = _build_datetime(df["date"], df["time"])
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    return (
        df[["datetime", "city", "temperature"]]
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def load_economic_data() -> pd.DataFrame:
    pmi = _read_excel(ECONOMIC_FILES["PMI"])
    pmi = _rename_by_alias(pmi, {"date": ("date", "tarih"), "PMI": ("pmi",)})
    pmi["date"] = _excel_date_to_datetime(pmi["date"])
    pmi["PMI"] = pd.to_numeric(pmi["PMI"], errors="coerce")
    pmi["year_month"] = pmi["date"].dt.to_period("M")
    pmi = pmi[["year_month", "PMI"]]

    interest = _read_excel(ECONOMIC_FILES["IR"], header=None, names=["date", "IR"])
    interest["date"] = _excel_date_to_datetime(interest["date"])
    interest["IR"] = pd.to_numeric(interest["IR"], errors="coerce")
    interest["year_month"] = interest["date"].dt.to_period("M")
    interest = interest[["year_month", "IR"]]

    capacity = _read_excel(ECONOMIC_FILES["CUR"])
    capacity = _rename_by_alias(
        capacity,
        {
            "date": ("date", "tarih"),
            "CUR": ("cur", "kko"),
        },
    )
    capacity["date"] = _excel_date_to_datetime(capacity["date"])
    capacity["CUR"] = pd.to_numeric(
        capacity["CUR"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    capacity["year_month"] = capacity["date"].dt.to_period("M")
    capacity = capacity[["year_month", "CUR"]]

    econ = pmi.merge(interest, on="year_month", how="outer")
    econ = econ.merge(capacity, on="year_month", how="outer")
    econ = econ.sort_values("year_month").drop_duplicates("year_month")
    return econ.reset_index(drop=True)


def build_analysis_frame() -> pd.DataFrame:
    consumption = pd.concat(
        [load_consumption(year) for year in (2022, 2023, 2024)],
        ignore_index=True,
    )
    temperature = pd.concat(
        [load_temperature(year) for year in (2022, 2023, 2024)],
        ignore_index=True,
    )
    temperature_hourly = (
        temperature.groupby("datetime", as_index=False)["temperature"].mean()
    )

    df = consumption.merge(temperature_hourly, on="datetime", how="inner")
    df = df.sort_values("datetime").reset_index(drop=True)

    df["HDD"] = np.maximum(18 - df["temperature"], 0)
    df["CDD"] = np.maximum(df["temperature"] - 24, 0)
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["year_month"] = df["datetime"].dt.to_period("M")
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["night"] = df["hour"].between(0, 6).astype(int)
    df["log_consumption"] = np.log(df["consumption"])
    df["sqrt_consumption"] = np.sqrt(df["consumption"])

    econ = load_economic_data()
    df = df.merge(econ, on="year_month", how="left")
    return df


def create_month_dummies(df: pd.DataFrame) -> pd.DataFrame:
    columns = [f"month_{month}" for month in range(1, 13) if month != 9]
    dummies = pd.get_dummies(df["month"], prefix="month", dtype=int)
    return dummies.reindex(columns=columns, fill_value=0)


def fit_scaler(frame: pd.DataFrame, columns: list[str]) -> ZScoreScaler:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        means[column] = float(numeric.mean())
        std = float(numeric.std(ddof=0))
        stds[column] = std if not np.isnan(std) and std != 0.0 else 1.0
    return ZScoreScaler(means=means, stds=stds, columns=tuple(columns))


def build_model_specs() -> list[dict[str, object]]:
    return [
        {
            "model_id": "model_1",
            "prompt": "TABLE 1 - consumption ~ HDD + CDD",
            "y_col": "consumption",
            "features": ["HDD", "CDD"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_2",
            "prompt": "TABLE 2 - consumption ~ HDD + CDD + night dummy",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_3",
            "prompt": "TABLE 3 - consumption ~ HDD + CDD + night dummy + weekend dummy",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_4",
            "prompt": "TABLE 4 - consumption ~ HDD + CDD + night dummy + weekend dummy + month dummies",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend"],
            "use_month_dummies": True,
        },
        {
            "model_id": "model_5",
            "prompt": "TABLE 5 - consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_6",
            "prompt": "TABLE 6 - consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR + CUR",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR", "CUR"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_7",
            "prompt": "TABLE 7 - log_consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR",
            "y_col": "log_consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_8",
            "prompt": "TABLE 8 - sqrt_consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR",
            "y_col": "sqrt_consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR"],
            "use_month_dummies": False,
        },
    ]


def fit_single_model(
    df: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[
    RegressionResultsWrapper,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    feature_columns = list(spec["features"])
    features_raw = df[feature_columns].copy()
    if bool(spec["use_month_dummies"]):
        features_raw = pd.concat([features_raw, create_month_dummies(df)], axis=1)

    continuous_columns = [
        column for column in features_raw.columns if column in ALL_CONTINUOUS_COLUMNS
    ]
    scaler = fit_scaler(features_raw, continuous_columns)
    features_scaled = scaler.transform(features_raw)

    X_raw = sm.add_constant(features_raw.astype(float), has_constant="add")
    X_scaled = sm.add_constant(features_scaled.astype(float), has_constant="add")
    y_col = str(spec["y_col"])
    y = df[y_col].astype(float).rename(y_col)

    model_data = pd.concat(
        [y, X_raw.add_prefix("raw__"), X_scaled.add_prefix("scaled__")],
        axis=1,
    ).dropna()

    X_raw_clean = model_data[[f"raw__{col}" for col in X_raw.columns]].rename(
        columns={f"raw__{col}": col for col in X_raw.columns}
    )
    X_scaled_clean = model_data[[f"scaled__{col}" for col in X_scaled.columns]].rename(
        columns={f"scaled__{col}": col for col in X_scaled.columns}
    )
    y_clean = model_data[y_col]

    model = sm.OLS(y_clean, X_scaled_clean).fit(cov_type="HC1")

    fitted = df.loc[model_data.index].copy()
    fitted[f"fitted_{y_col}"] = model.predict(X_scaled_clean)

    vif_input = X_scaled_clean.drop(columns=["const"], errors="ignore")
    vif_table = pd.DataFrame(
        {
            "model_id": str(spec["model_id"]),
            "prompt": str(spec["prompt"]),
            "variable": vif_input.columns,
            "vif": [
                variance_inflation_factor(vif_input.values, i)
                for i in range(vif_input.shape[1])
            ],
        }
    ).sort_values("vif", ascending=False)

    ols_table = pd.DataFrame(
        {
            "model_id": str(spec["model_id"]),
            "prompt": str(spec["prompt"]),
            "variable": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
        }
    )

    diagnostics = pd.DataFrame(
        [
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "dependent_variable", "value": y_col},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "plot_year", "value": int(df["year"].iloc[0])},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "month_dummy_in_model", "value": int(bool(spec["use_month_dummies"]))},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "n_obs", "value": len(model_data)},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "durbin_watson", "value": durbin_watson(model.resid)},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "raw_condition_number", "value": np.linalg.cond(X_raw_clean)},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "scaled_condition_number", "value": np.linalg.cond(X_scaled_clean)},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "r_squared", "value": model.rsquared},
            {"model_id": str(spec["model_id"]), "prompt": str(spec["prompt"]), "metric": "adj_r_squared", "value": model.rsquared_adj},
        ]
    )

    scaler_table = pd.DataFrame(
        {
            "model_id": str(spec["model_id"]),
            "prompt": str(spec["prompt"]),
            "variable": continuous_columns,
            "mean": [scaler.means[column] for column in continuous_columns],
            "std": [scaler.stds[column] for column in continuous_columns],
        }
    )

    return model, fitted, ols_table, diagnostics, scaler_table, vif_table


def save_year_plots(year_df: pd.DataFrame, year: int, fitted_column: str) -> list[Path]:
    plot_paths: list[Path] = []

    if year_df.empty:
        return plot_paths

    daily = (
        year_df.set_index("datetime")[["consumption", fitted_column]]
        .resample("D")
        .mean()
    )
    ax = daily.plot(figsize=(14, 5), linewidth=1.2)
    ax.set_title(f"{year} Gunluk Ortalama Tuketim: Gercek vs OLS")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Tuketim (MWh)")
    plt.tight_layout()
    path_daily = FIGURES_DIR / f"{year}_gunluk_gercek_vs_ols.png"
    plt.savefig(path_daily, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path_daily)

    hourly = year_df.groupby("hour", as_index=True)["consumption"].mean()
    ax = hourly.plot(figsize=(10, 4), marker="o", color="darkgreen")
    ax.set_title(f"{year} Saatlik Ortalama Tuketim Profili")
    ax.set_xlabel("Saat")
    ax.set_ylabel("Tuketim (MWh)")
    plt.tight_layout()
    path_hourly = FIGURES_DIR / f"{year}_saatlik_tuketim_profili.png"
    plt.savefig(path_hourly, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path_hourly)

    monthly = year_df.groupby("month", as_index=True)["consumption"].mean()
    ax = monthly.plot(kind="bar", figsize=(10, 4), color="steelblue")
    ax.set_title(f"{year} Aylik Ortalama Tuketim")
    ax.set_xlabel("Ay")
    ax.set_ylabel("Tuketim (MWh)")
    plt.tight_layout()
    path_monthly = FIGURES_DIR / f"{year}_aylik_ortalama_tuketim.png"
    plt.savefig(path_monthly, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path_monthly)

    return plot_paths


def write_vif_txt(path: Path, title: str, vif_table: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        for _, row in vif_table.iterrows():
            handle.write(f"{row['variable']:<20} {row['vif']:.4f}\n")


def write_grouped_vif_txt(path: Path, title: str, vif_table: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        for prompt, group in vif_table.groupby("prompt", sort=False):
            handle.write(f"{prompt}\n")
            handle.write("-" * len(prompt) + "\n")
            for _, row in group.iterrows():
                handle.write(f"{row['variable']:<20} {row['vif']:.4f}\n")
            handle.write("\n")


def save_r_squared_comparison_plot(all_diagnostics: pd.DataFrame) -> list[Path]:
    plot_paths: list[Path] = []

    r2_data = all_diagnostics[all_diagnostics["metric"] == "r_squared"].copy()
    if r2_data.empty:
        return plot_paths

    r2_data["table_no"] = (
        r2_data["model_id"].astype(str).str.extract(r"model_(\d+)").astype(int)
    )
    r2_data["label"] = (
        "Table "
        + r2_data["table_no"].astype(str)
        + "\n"
        + r2_data["prompt"].str.replace(r"^TABLE \d+ - ", "", regex=True)
    )
    r2_data["value"] = pd.to_numeric(r2_data["value"], errors="coerce")

    pivot = (
        r2_data.sort_values(["table_no", "year"])
        .pivot(index="label", columns="year", values="value")
        .sort_index()
    )
    pivot.to_csv(REPORTS_DIR / "r_squared_karsilastirma_tablosu.csv", encoding="utf-8-sig")

    years = [year for year in TARGET_YEARS if year in pivot.columns]
    positions = np.arange(len(pivot.index))
    width = 0.24
    colors = {2022: "#1f77b4", 2023: "#ff7f0e", 2024: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(22, 9))
    for idx, year in enumerate(years):
        offset = (idx - (len(years) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            pivot[year].values,
            width=width,
            label=f"{year} R^2",
            color=colors.get(year, None),
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("2022-2023-2024 OLS Modelleri R^2 Karsilastirmasi", fontsize=16, pad=16)
    ax.set_ylabel("R^2", fontsize=12)
    ax.set_xlabel("OLS Modeli", fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(pivot.index, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, max(1.0, float(np.nanmax(pivot.values)) + 0.12))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Yil", frameon=True)
    plt.tight_layout()

    grouped_path = FIGURES_DIR / "r_squared_karsilastirma_grafigi.png"
    plt.savefig(grouped_path, dpi=180, bbox_inches="tight")
    plt.close()
    plot_paths.append(grouped_path)

    fig, ax = plt.subplots(figsize=(22, 9))
    heatmap_data = pivot.values
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1.0, np.nanmax(heatmap_data)))
    ax.set_title("R^2 Isı Haritasi - Hangi Model Hangi Yilda Daha Guclu", fontsize=16, pad=16)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], fontsize=11)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Yil", fontsize=12)
    ax.set_ylabel("OLS Modeli", fontsize=12)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R^2", rotation=90)
    plt.tight_layout()

    heatmap_path = FIGURES_DIR / "r_squared_karsilastirma_isiharitasi.png"
    plt.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close()
    plot_paths.append(heatmap_path)

    return plot_paths


def main() -> None:
    ensure_directories()

    df = build_analysis_frame()
    all_ols: list[pd.DataFrame] = []
    all_diag: list[pd.DataFrame] = []
    all_scaler: list[pd.DataFrame] = []
    all_vif: list[pd.DataFrame] = []
    all_plot_paths: list[Path] = []

    for year in TARGET_YEARS:
        year_df = df[df["year"] == year].copy()
        year_df.to_csv(PROCESSED_DIR / f"analiz_verisi_{year}.csv", index=False)

        combined_ols: list[pd.DataFrame] = []
        combined_diag: list[pd.DataFrame] = []
        combined_scaler: list[pd.DataFrame] = []
        combined_vif: list[pd.DataFrame] = []
        summary_lines: list[str] = []

        for spec in build_model_specs():
            model, fitted_df, ols_table, diagnostics, scaler_table, vif_table = fit_single_model(
                year_df,
                spec,
            )

            model_id = str(spec["model_id"])
            prompt = str(spec["prompt"])
            slug = model_id.replace("model_", "table_")
            prefix = f"{year}_{slug}"

            fitted_df.to_csv(PROCESSED_DIR / f"{prefix}_model_verisi_tahminli.csv", index=False)
            ols_table.to_csv(REPORTS_DIR / f"{prefix}_ols_regresyon_tablosu.csv", index=False)
            diagnostics.to_csv(REPORTS_DIR / f"{prefix}_model_diyagnostik.csv", index=False)
            scaler_table.to_csv(REPORTS_DIR / f"{prefix}_standardizasyon_bazlari.csv", index=False)
            vif_table.to_csv(REPORTS_DIR / f"{prefix}_vif_tablosu.csv", index=False)
            write_vif_txt(
                REPORTS_DIR / f"{prefix}_vif_tablosu.txt",
                f"{year} - {prompt} - VIF",
                vif_table,
            )

            with open(REPORTS_DIR / f"{prefix}_ols_ozeti.txt", "w", encoding="utf-8") as handle:
                handle.write(f"{year} - {prompt}\n\n")
                handle.write(model.summary().as_text())

            combined_ols.append(ols_table.assign(year=year))
            combined_diag.append(diagnostics.assign(year=year))
            combined_scaler.append(scaler_table.assign(year=year))
            combined_vif.append(vif_table.assign(year=year))

            summary_lines.append(f"{year} - {prompt}")
            summary_lines.append(f"OLS table: outputs/reports/{prefix}_ols_regresyon_tablosu.csv")
            summary_lines.append(f"VIF table: outputs/reports/{prefix}_vif_tablosu.csv")
            summary_lines.append(f"OLS summary: outputs/reports/{prefix}_ols_ozeti.txt")
            summary_lines.append("")

            if model_id == "model_6":
                all_plot_paths.extend(save_year_plots(fitted_df, year, "fitted_consumption"))

        year_ols = pd.concat(combined_ols, ignore_index=True)
        year_diag = pd.concat(combined_diag, ignore_index=True)
        year_scaler = pd.concat(combined_scaler, ignore_index=True)
        year_vif = pd.concat(combined_vif, ignore_index=True)

        year_ols.to_csv(REPORTS_DIR / f"{year}_tum_ols_regresyon_tablolari.csv", index=False)
        year_diag.to_csv(REPORTS_DIR / f"{year}_tum_model_diyagnostikleri.csv", index=False)
        year_scaler.to_csv(REPORTS_DIR / f"{year}_tum_standardizasyon_bazlari.csv", index=False)
        year_vif.to_csv(REPORTS_DIR / f"{year}_tum_vif_tablolari.csv", index=False)
        write_grouped_vif_txt(
            REPORTS_DIR / f"{year}_tum_vif_tablolari.txt",
            f"{year} - Tum VIF Tablolari",
            year_vif,
        )

        with open(REPORTS_DIR / f"{year}_model_rehberi.txt", "w", encoding="utf-8") as handle:
            handle.write(f"{year} OLS MODELLERI - DOSYA REHBERI\n\n")
            handle.write("\n".join(summary_lines))

        all_ols.append(year_ols)
        all_diag.append(year_diag)
        all_scaler.append(year_scaler)
        all_vif.append(year_vif)

    pd.concat(all_ols, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_ols_regresyon_tablolari.csv",
        index=False,
    )
    pd.concat(all_diag, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_model_diyagnostikleri.csv",
        index=False,
    )
    pd.concat(all_scaler, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_standardizasyon_bazlari.csv",
        index=False,
    )
    pd.concat(all_vif, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_vif_tablolari.csv",
        index=False,
    )
    write_grouped_vif_txt(
        REPORTS_DIR / "tum_yillar_vif_tablolari.txt",
        "Tum Yillar - Tum VIF Tablolari",
        pd.concat(all_vif, ignore_index=True),
    )

    all_diagnostics_df = pd.concat(all_diag, ignore_index=True)
    r2_plot_paths = save_r_squared_comparison_plot(all_diagnostics_df)

    print("2022, 2023 ve 2024 OLS analizleri tamamlandi.")
    for year in TARGET_YEARS:
        print(f"Model rehberi ({year}): {REPORTS_DIR / f'{year}_model_rehberi.txt'}")
        print(f"Toplu OLS tablo ({year}): {REPORTS_DIR / f'{year}_tum_ols_regresyon_tablolari.csv'}")
        print(f"Toplu VIF tablo ({year}): {REPORTS_DIR / f'{year}_tum_vif_tablolari.csv'}")
    for path in all_plot_paths:
        print(f"Figur: {path}")
    for path in r2_plot_paths:
        print(f"R^2 Figur: {path}")


if __name__ == "__main__":
    main()
