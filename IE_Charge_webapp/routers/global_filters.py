import pandas as pd

from db import query_df
from routers.warning_utils import parse_warning_codes

VALID_TEMP_COLUMNS = {
    "DCBM_Temp_H_Min", "DCBM_Temp_H_Max", "DCBM_Temp_H_Moy",
    "DCBM_Temp_L_Min", "DCBM_Temp_L_Max", "DCBM_Temp_L_Moy",
    "PISTOLET_Temp_1_Min", "PISTOLET_Temp_1_Max", "PISTOLET_Temp_1_Moy",
    "PISTOLET_Temp_2_Min", "PISTOLET_Temp_2_Max", "PISTOLET_Temp_2_Moy",
    "DCBM_Temp_Moy_Diff", "PISTOLET_Temp_Moy_Diff", "DCBM_PISTOLET_Moy_Diff",
}


def apply_energy_filter(df: pd.DataFrame, energy_mode: str = "all", energy_min: str = "", energy_max: str = "") -> pd.DataFrame:
    if df.empty or "Energy (Kwh)" not in df.columns:
        return df

    mode = (energy_mode or "all").strip().lower()
    if mode == "all":
        return df

    energy = pd.to_numeric(df["Energy (Kwh)"], errors="coerce")
    mask = energy.notna()

    try:
        if mode == "lt" and energy_max:
            mask &= energy <= float(energy_max)
        elif mode == "gt" and energy_min:
            mask &= energy >= float(energy_min)
        elif mode == "between":
            if energy_min:
                mask &= energy >= float(energy_min)
            if energy_max:
                mask &= energy <= float(energy_max)
    except (TypeError, ValueError):
        return df

    return df[mask].copy()


def apply_temperature_filter(
    df: pd.DataFrame,
    temp_column: str = "",
    temp_mode: str = "all",
    temp_min: str = "",
    temp_max: str = "",
) -> pd.DataFrame:
    if df.empty or "ID" not in df.columns:
        return df

    mode = (temp_mode or "all").strip().lower()
    column = (temp_column or "").strip()
    if mode == "all" or not column or column not in VALID_TEMP_COLUMNS:
        return df

    conds = [f"`{column}` IS NOT NULL"]
    params = {}

    try:
        if mode == "lt" and temp_max:
            params["temp_max_val"] = float(temp_max)
            conds.append(f"`{column}` <= :temp_max_val")
        elif mode == "gt" and temp_min:
            params["temp_min_val"] = float(temp_min)
            conds.append(f"`{column}` >= :temp_min_val")
        elif mode == "between":
            if temp_min:
                params["temp_min_val"] = float(temp_min)
                conds.append(f"`{column}` >= :temp_min_val")
            if temp_max:
                params["temp_max_val"] = float(temp_max)
                conds.append(f"`{column}` <= :temp_max_val")
    except (TypeError, ValueError):
        return df

    if len(conds) <= 1:
        return df

    temp_df = query_df(f"SELECT ID FROM kpi_temperature WHERE {' AND '.join(conds)}", params)
    if temp_df.empty:
        return df.iloc[0:0].copy()

    valid_ids = set(temp_df["ID"].astype(str))
    return df[df["ID"].astype(str).isin(valid_ids)].copy()


def apply_warning_binary_filter(df: pd.DataFrame, warning_filter: str = "") -> pd.DataFrame:
    if df.empty or "warning" not in df.columns:
        return df

    mode = (warning_filter or "").strip().lower()
    if mode in {"", "tous", "all"}:
        return df

    warning_codes = df["warning"].apply(parse_warning_codes)
    has_warning = warning_codes.apply(lambda codes: len(codes) > 0)

    if mode == "avec":
        return df[has_warning].copy()
    if mode == "sans":
        return df[~has_warning].copy()
    return df


def enrich_warning_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "warning" not in df.columns:
        df["warning"] = ""
    df["warning_codes_list"] = df["warning"].apply(parse_warning_codes)
    df["warning_details_list"] = df["warning"].apply(
        lambda v: [
            "Fin de charge" if code == 1 else {
                3: "DCBM_Temp_H_Moy > 65",
                5: "DCBM_Temp_L_Moy > 65",
                7: "PISTOLET_Temp_1_Moy > 65",
                9: "PISTOLET_Temp_2_Moy > 65",
                10: "PISTOLET_Temp_Moy_Diff > 10",
                11: "DCBM_Temp_Moy_Diff > 10",
                12: "DCBM_PISTOLET_Moy_Diff > 25",
            }.get(code, f"Code {code}")
            for code in parse_warning_codes(v)
        ]
    )
    df["has_warning"] = df["warning_codes_list"].apply(bool)
    return df
