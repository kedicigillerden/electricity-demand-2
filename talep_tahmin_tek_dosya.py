from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from itertools import combinations
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
MACRO_COLUMNS = ["PMI", "IR", "CUR"]
CONTINUOUS_COLUMNS = ["HDD", "CDD", "PMI", "IR", "CUR", "hour_z"]
POLYNOMIAL_SOURCE_COLUMNS = {"HDD", "CDD", "PMI", "IR", "CUR", "hour_z"}
NEAR_ZERO_VARIANCE_TOL = 1e-10

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
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    description: str
    features: list[str]
    interactions: list[tuple[str, str]]
    use_month_dummies: bool


@dataclass(frozen=True)
class ZScoreScaler:
    means: dict[str, float]
    stds: dict[str, float]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column, mean in self.means.items():
            out[column] = (pd.to_numeric(out[column], errors="coerce") - mean) / self.stds[column]
        return out


def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _normalize_label(value: object) -> str:
    text = str(value).translate(TURKISH_CHAR_MAP)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _rename_by_alias(df: pd.DataFrame, alias_map: dict[str, tuple[str, ...]]) -> pd.DataFrame:
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
    date_mask = as_object.map(lambda value: isinstance(value, (datetime, date, pd.Timestamp)))
    if date_mask.any():
        result.loc[date_mask] = pd.to_datetime(as_object.loc[date_mask], errors="coerce")
    remaining = ~date_mask
    if remaining.any():
        numeric = pd.to_numeric(as_object.loc[remaining], errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            result.loc[numeric_mask.index[numeric_mask]] = pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric.loc[numeric_mask], unit="D")
        text_index = numeric_mask.index[~numeric_mask]
        if len(text_index) > 0:
            result.loc[text_index] = pd.to_datetime(as_object.loc[text_index], errors="coerce", dayfirst=True)
    return result


def _excel_time_to_string(series: pd.Series) -> pd.Series:
    as_object = series.astype("object")
    result = pd.Series("", index=series.index, dtype="object")
    datetime_mask = as_object.map(lambda value: isinstance(value, (datetime, pd.Timestamp)))
    if datetime_mask.any():
        result.loc[datetime_mask] = pd.to_datetime(as_object.loc[datetime_mask], errors="coerce").dt.strftime("%H:%M:%S")
    pure_time_mask = as_object.map(lambda value: isinstance(value, time))
    if pure_time_mask.any():
        result.loc[pure_time_mask] = as_object.loc[pure_time_mask].map(lambda value: value.strftime("%H:%M:%S"))
    remaining = ~(datetime_mask | pure_time_mask)
    if remaining.any():
        numeric = pd.to_numeric(as_object.loc[remaining], errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            result.loc[numeric_mask.index[numeric_mask]] = (pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric.loc[numeric_mask], unit="D")).dt.strftime("%H:%M:%S")
        text_index = numeric_mask.index[~numeric_mask]
        if len(text_index) > 0:
            text = as_object.loc[text_index].astype(str).str.strip().replace({"nan": "", "NaT": "", "None": ""})
            text = text.where(~text.str.match(r"^\d{1,2}$", na=False), text.str.zfill(2) + ":00:00")
            text = text.where(~text.str.match(r"^\d{1,2}:\d{2}$", na=False), text + ":00")
            text = text.str.replace(r"^(\d{1,2}):", lambda match: match.group(1).zfill(2) + ":", regex=True)
            result.loc[text_index] = text
    return result


def _build_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = _excel_date_to_datetime(date_series)
    times = _excel_time_to_string(time_series)
    return pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + times, errors="coerce")


def _read_excel(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", **kwargs)


def load_consumption(year: int) -> pd.DataFrame:
    df = _read_excel(CONSUMPTION_FILES[year])
    df = _rename_by_alias(df, {"date": ("tarih", "date"), "time": ("saat", "time"), "consumption": ("tuketimmiktarimwh", "consumption")})
    if {"date", "time", "consumption"} - set(df.columns):
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "time", df.columns[2]: "consumption"})
    df["datetime"] = _build_datetime(df["date"], df["time"])
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    return df[["datetime", "consumption"]].dropna().sort_values("datetime").reset_index(drop=True)


def load_temperature(year: int) -> pd.DataFrame:
    df = _read_excel(TEMPERATURE_FILES[year])
    df = _rename_by_alias(df, {"city": ("istasyonadi", "city"), "date": ("tarih", "date"), "time": ("saat", "time"), "temperature": ("sicaklik", "temperature")})
    if {"city", "date", "time", "temperature"} - set(df.columns):
        df = df.rename(columns={df.columns[1]: "city", df.columns[3]: "date", df.columns[4]: "time", df.columns[5]: "temperature"})
    df["datetime"] = _build_datetime(df["date"], df["time"])
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    return df[["datetime", "city", "temperature"]].dropna().sort_values("datetime").reset_index(drop=True)


def load_economic_data() -> pd.DataFrame:
    pmi = _read_excel(ECONOMIC_FILES["PMI"])
    pmi = _rename_by_alias(pmi, {"date": ("date", "tarih"), "PMI": ("pmi",)})
    pmi["date"] = _excel_date_to_datetime(pmi["date"])
    pmi["PMI"] = pd.to_numeric(pmi["PMI"], errors="coerce")
    pmi["year_month"] = pmi["date"].dt.to_period("M")

    interest = _read_excel(ECONOMIC_FILES["IR"], header=None, names=["date", "IR"])
    interest["date"] = _excel_date_to_datetime(interest["date"])
    interest["IR"] = pd.to_numeric(interest["IR"], errors="coerce")
    interest["year_month"] = interest["date"].dt.to_period("M")

    capacity = _read_excel(ECONOMIC_FILES["CUR"])
    capacity = _rename_by_alias(capacity, {"date": ("date", "tarih"), "CUR": ("cur", "kko")})
    capacity["date"] = _excel_date_to_datetime(capacity["date"])
    capacity["CUR"] = pd.to_numeric(capacity["CUR"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    capacity["year_month"] = capacity["date"].dt.to_period("M")

    econ = pmi[["year_month", "PMI"]].merge(interest[["year_month", "IR"]], on="year_month", how="outer")
    econ = econ.merge(capacity[["year_month", "CUR"]], on="year_month", how="outer")
    return econ.sort_values("year_month").drop_duplicates("year_month").reset_index(drop=True)


def build_analysis_frame() -> pd.DataFrame:
    consumption = pd.concat([load_consumption(year) for year in TARGET_YEARS], ignore_index=True)
    temperature = pd.concat([load_temperature(year) for year in TARGET_YEARS], ignore_index=True)
    temperature_hourly = temperature.groupby("datetime", as_index=False)["temperature"].mean()
    df = consumption.merge(temperature_hourly, on="datetime", how="inner").sort_values("datetime").reset_index(drop=True)

    df["HDD"] = np.maximum(18 - pd.to_numeric(df["temperature"], errors="coerce"), 0)
    df["CDD"] = np.maximum(pd.to_numeric(df["temperature"], errors="coerce") - 24, 0)
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    hour_std = float(df["hour"].std(ddof=0))
    df["hour_z"] = (df["hour"] - float(df["hour"].mean())) / (hour_std if hour_std else 1.0)
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["year_month"] = df["datetime"].dt.to_period("M")
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["night"] = df["hour"].between(0, 6).astype(int)
    df["holiday"] = 0
    df["log_consumption"] = np.log(df["consumption"])
    df = df.merge(load_economic_data(), on="year_month", how="left")
    return df


def create_month_dummies(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["month"], prefix="month", dtype=int)
    columns = [f"month_{month}" for month in range(1, 13) if month != 1]
    return dummies.reindex(columns=columns, fill_value=0)


def fit_scaler(frame: pd.DataFrame, columns: list[str]) -> ZScoreScaler:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        means[column] = float(numeric.mean())
        std = float(numeric.std(ddof=0))
        stds[column] = std if not np.isnan(std) and std != 0.0 else 1.0
    return ZScoreScaler(means, stds)


def _is_hdd_cdd_pair(left: str, right: str) -> bool:
    left_n = _normalize_label(left)
    right_n = _normalize_label(right)
    return ("hdd" in left_n and "cdd" in right_n) or ("cdd" in left_n and "hdd" in right_n)


def _is_macro_pair(left: str, right: str) -> bool:
    return left in MACRO_COLUMNS and right in MACRO_COLUMNS


def should_skip_interaction(left: str, right: str, use_month_dummies: bool) -> bool:
    # HDD=max(18-temperature,0) ve CDD=max(temperature-24,0) ayni anda pozitif olamaz.
    # Bu nedenle HDD x CDD ailesindeki carpimlar fiziksel olarak anlamsizdir.
    if _is_hdd_cdd_pair(left, right):
        return True
    # Full month dummy varken saf macro x macro terimleri aylik sabit kaldigi icin ay
    # kuklalari tarafindan absorbe edilir; identifiye edilemez.
    if use_month_dummies and _is_macro_pair(left, right):
        return True
    return False


def is_near_zero_variance(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return True
    std = float(values.std(ddof=0))
    return bool(np.isnan(std) or std <= NEAR_ZERO_VARIANCE_TOL)


def build_model_specs() -> list[ModelSpec]:
    high_frequency_macro_interactions = [
        ("CDD", "PMI"),
        ("CDD", "CUR"),
        ("HDD", "IR"),
        ("weekend", "PMI"),
        ("night", "IR"),
        ("hour_z", "PMI"),
        ("hour_z", "IR"),
    ]
    return [
        ModelSpec(
            "model_A",
            "Seasonal baseline: month dummies var, macro ana etki yok",
            ["HDD", "CDD", "night", "weekend", "holiday"],
            [],
            True,
        ),
        ModelSpec(
            "model_B",
            "Ana model: month dummies + macro x high-frequency interaction",
            ["HDD", "CDD", "night", "weekend", "holiday"],
            high_frequency_macro_interactions,
            True,
        ),
        ModelSpec(
            "model_C",
            "Robustness: month dummy yok, macro ana etkiler var",
            ["HDD", "CDD", "night", "weekend", "holiday", "PMI", "IR", "CUR"],
            [],
            False,
        ),
        ModelSpec(
            "model_D",
            "Genisletilmis robustness: macro ana etkiler + high-frequency interaction",
            ["HDD", "CDD", "night", "weekend", "holiday", "PMI", "IR", "CUR"],
            high_frequency_macro_interactions,
            False,
        ),
    ]


def prepare_feature_frame(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for column in spec.features:
        if column in df.columns:
            features[column] = pd.to_numeric(df[column], errors="coerce")
    if spec.use_month_dummies:
        features = pd.concat([features, create_month_dummies(df)], axis=1)
    for left, right in spec.interactions:
        if left not in df.columns or right not in df.columns:
            continue
        if should_skip_interaction(left, right, spec.use_month_dummies):
            continue
        features[f"{left}_x_{right}"] = pd.to_numeric(df[left], errors="coerce") * pd.to_numeric(df[right], errors="coerce")
    return features


def build_pairwise_interaction_candidates(df: pd.DataFrame, columns: list[str], use_month_dummies: bool) -> pd.DataFrame:
    # Ham hour (0-23) ve month burada yoktur. Saat icin hour_z, ay icin month dummy kullanilir.
    candidate_columns: dict[str, pd.Series] = {}
    valid_columns = [column for column in columns if column in df.columns and column != "hour" and column != "month"]
    for left, right in combinations(valid_columns, 2):
        if should_skip_interaction(left, right, use_month_dummies):
            continue
        values = pd.to_numeric(df[left], errors="coerce") * pd.to_numeric(df[right], errors="coerce")
        if not is_near_zero_variance(values):
            candidate_columns[f"{left}_x_{right}"] = values
    return pd.DataFrame(candidate_columns, index=df.index)


def build_polynomial_candidates(df: pd.DataFrame) -> pd.DataFrame:
    polynomial_columns: dict[str, pd.Series] = {}
    for column in POLYNOMIAL_SOURCE_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce") ** 2
        if not is_near_zero_variance(values):
            polynomial_columns[f"{column}_sq"] = values
    return pd.DataFrame(polynomial_columns, index=df.index)


def fit_ols(df: pd.DataFrame, y_col: str, features: pd.DataFrame) -> tuple[RegressionResultsWrapper, pd.DataFrame, pd.DataFrame]:
    continuous = [column for column in features.columns if column in CONTINUOUS_COLUMNS or "_x_" in column or column.endswith("_sq")]
    scaler = fit_scaler(features, continuous)
    scaled_features = scaler.transform(features)
    y = pd.to_numeric(df[y_col], errors="coerce").rename(y_col)
    X = sm.add_constant(scaled_features.astype(float), has_constant="add")
    model_data = pd.concat([y, X], axis=1).dropna()
    model = sm.OLS(model_data[y_col], model_data.drop(columns=[y_col])).fit(cov_type="HC1")
    scaler_table = pd.DataFrame({"variable": list(scaler.means), "mean": list(scaler.means.values()), "std": [scaler.stds[col] for col in scaler.means]})
    return model, model_data, scaler_table


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    vif_input = X.drop(columns=["const"], errors="ignore")
    if vif_input.empty:
        return pd.DataFrame(columns=["variable", "vif"])
    values = []
    for idx, column in enumerate(vif_input.columns):
        try:
            vif = variance_inflation_factor(vif_input.values, idx)
        except Exception:
            vif = np.nan
        values.append({"variable": column, "vif": vif})
    return pd.DataFrame(values).sort_values("vif", ascending=False)


def identification_diagnostics(df: pd.DataFrame, spec: ModelSpec, features: pd.DataFrame) -> list[str]:
    lines = [f"Identification diagnostic - {spec.model_id}", "=" * 48]
    for macro in MACRO_COLUMNS:
        if macro in df.columns:
            monthly_nunique = df.groupby("year_month")[macro].nunique(dropna=True)
            lines.append(f"{macro} ay icinde sabit mi: {bool((monthly_nunique <= 1).all())}")
    macro_main = [column for column in MACRO_COLUMNS if column in features.columns]
    pure_macro_interactions = []
    for column in features.columns:
        if "_x_" not in column:
            continue
        left, right = column.split("_x_", 1)
        if _is_macro_pair(left, right):
            pure_macro_interactions.append(column)
    lines.append(f"Month dummy kullaniliyor mu: {spec.use_month_dummies}")
    lines.append(f"Month dummy ile macro ana etki sayisi: {len(macro_main) if spec.use_month_dummies else 0}")
    lines.append(f"Month dummy ile saf macro interaction sayisi: {len(pure_macro_interactions) if spec.use_month_dummies else 0}")
    X = sm.add_constant(features.apply(pd.to_numeric, errors="coerce"), has_constant="add").dropna()
    if not X.empty:
        lines.append(f"Matrix rank: {np.linalg.matrix_rank(X.values)}")
        lines.append(f"Column count with const: {X.shape[1]}")
        lines.append(f"Condition number: {np.linalg.cond(X.values):.4f}")
        vif = vif_table(X)
        if not vif.empty:
            lines.append(f"Max VIF: {pd.to_numeric(vif['vif'], errors='coerce').max():.4f}")
    if spec.use_month_dummies and macro_main:
        lines.append("UYARI: Full month dummy ile PMI/IR/CUR ana etkileri identifiye edilemez.")
    if spec.use_month_dummies and pure_macro_interactions:
        lines.append("UYARI: Full month dummy ile saf macro x macro interactionlar identifiye edilemez.")
    if not spec.use_month_dummies and macro_main:
        lines.append("NOT: Macro ana etki katsayilari saf makro etki degildir; aylik mevsimsellik ve gozlenmeyen ay etkilerini de tasiyabilir.")
    return lines


def hdd_cdd_diagnostics(df: pd.DataFrame) -> str:
    product = pd.to_numeric(df["HDD"], errors="coerce") * pd.to_numeric(df["CDD"], errors="coerce")
    scaled = fit_scaler(df[["HDD", "CDD"]], ["HDD", "CDD"]).transform(df[["HDD", "CDD"]])
    scaled_product = scaled["HDD"] * scaled["CDD"]
    lines = ["HDD/CDD diagnostic", "=" * 32]
    lines.append(f"Temperature numeric mi: {pd.api.types.is_numeric_dtype(df['temperature'])}")
    lines.append(f"Timestamp merge gozlem sayisi: {len(df)}")
    lines.append(f"HDD>0 ve CDD>0 ayni anda gozlem sayisi: {int(((df['HDD'] > 0) & (df['CDD'] > 0)).sum())}")
    lines.append(f"Ham HDD x CDD near-zero variance mi: {is_near_zero_variance(product)}")
    lines.append(f"Standardize HDD x CDD near-zero variance mi: {is_near_zero_variance(scaled_product)}")
    lines.append("\nHDD descriptive statistics:\n" + pd.to_numeric(df["HDD"], errors="coerce").describe().to_string())
    lines.append("\nCDD descriptive statistics:\n" + pd.to_numeric(df["CDD"], errors="coerce").describe().to_string())
    lines.append("\nNot: Ham HDD x CDD yapisal olarak sifirdir. Standardizasyon once yapilirsa HDD_z x CDD_z yapay olarak sifirdan farkli olabilir; bu nedenle interactionlar ham degiskenlerden uretilir, sonra olceklenir.")
    return "\n".join(lines)


def write_model_outputs(year: int, spec: ModelSpec, model: RegressionResultsWrapper, model_data: pd.DataFrame, scaler_table: pd.DataFrame, features: pd.DataFrame) -> dict[str, float]:
    prefix = f"{year}_{spec.model_id}"
    summary_path = REPORTS_DIR / f"{prefix}_ols_ozeti.txt"
    summary_path.write_text(f"{year} - {spec.description}\n\n{model.summary().as_text()}\n", encoding="utf-8")
    coef_table = pd.DataFrame({"variable": model.params.index, "coefficient": model.params.values, "std_error": model.bse.values, "t_stat": model.tvalues.values, "p_value": model.pvalues.values})
    coef_table.to_string(REPORTS_DIR / f"{prefix}_katsayi_tablosu.txt", index=False)
    scaler_table.to_string(REPORTS_DIR / f"{prefix}_standardizasyon.txt", index=False)
    X = model_data.drop(columns=["consumption"], errors="ignore")
    vif_table(X).to_string(REPORTS_DIR / f"{prefix}_vif.txt", index=False)
    (REPORTS_DIR / f"{prefix}_identification_diagnostic.txt").write_text("\n".join(identification_diagnostics(model_data.assign(year_month=df_year_month(model_data.index)), spec, features.loc[model_data.index])) + "\n", encoding="utf-8")
    metrics = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "durbin_watson": float(durbin_watson(model.resid)),
        "condition_number": float(np.linalg.cond(model.model.exog)),
    }
    lines = [f"{key}: {value:.6f}" for key, value in metrics.items()]
    (REPORTS_DIR / f"{prefix}_metrikler.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def df_year_month(index: pd.Index) -> pd.Series:
    return pd.Series(index=index, dtype="object")


def save_plots(df: pd.DataFrame, year: int) -> None:
    daily = df.set_index("datetime")["consumption"].resample("D").mean()
    ax = daily.plot(figsize=(14, 5), linewidth=1.1)
    ax.set_title(f"{year} Gunluk Ortalama Tuketim")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Tuketim")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{year}_gunluk_tuketim.png", dpi=150, bbox_inches="tight")
    plt.close()

    hourly = df.groupby("hour")["consumption"].mean()
    ax = hourly.plot(figsize=(10, 4), marker="o")
    ax.set_title(f"{year} Saatlik Ortalama Tuketim")
    ax.set_xlabel("Saat")
    ax.set_ylabel("Tuketim")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{year}_saatlik_tuketim.png", dpi=150, bbox_inches="tight")
    plt.close()


def missing_inputs() -> list[Path]:
    required = list(CONSUMPTION_FILES.values()) + list(TEMPERATURE_FILES.values()) + list(ECONOMIC_FILES.values())
    return [path for path in required if not path.exists()]


def main() -> None:
    ensure_directories()
    missing = missing_inputs()
    if missing:
        report = ["Ham veri dosyalari eksik oldugu icin model calistirilmadi.", "", "Beklenen dosyalar:"]
        report.extend(f"- {path.relative_to(PROJECT_ROOT)}" for path in missing)
        report.append("\nVerileri ekledikten sonra script ayni komutla tam OLS raporlarini ve grafikleri uretir.")
        (REPORTS_DIR / "veri_eksik_uyarisi.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        print("Ham veri eksik; uyarı raporu yazıldı.")
        return

    df = build_analysis_frame()
    df.to_csv(PROCESSED_DIR / "analiz_verisi.csv", index=False)
    (REPORTS_DIR / "hdd_cdd_diyagnostik.txt").write_text(hdd_cdd_diagnostics(df), encoding="utf-8")

    all_metrics: list[dict[str, object]] = []
    for year in TARGET_YEARS:
        year_df = df[df["year"] == year].copy()
        save_plots(year_df, year)
        for spec in build_model_specs():
            features = prepare_feature_frame(year_df, spec)
            model, model_data, scaler_table = fit_ols(year_df, "consumption", features)
            metrics = write_model_outputs(year, spec, model, model_data, scaler_table, features)
            all_metrics.append({"year": year, "model_id": spec.model_id, "description": spec.description, **metrics})

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(REPORTS_DIR / "model_A_B_C_D_ozet_metrikleri.csv", index=False)
    metrics_df.to_string(REPORTS_DIR / "model_A_B_C_D_ozet_metrikleri.txt", index=False)
    print("Model A/B/C/D analizleri tamamlandi.")


if __name__ == "__main__":
    main()
