from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode, unquote
import pandas as pd
import numpy as np
import logging

from db import query_df, LOGV0_BASE_URL
from routers.filters import MOMENT_ORDER

EVI_MOMENT = "EVI Status during error"
EVI_CODE = "EVI Error Code"
DS_PC = "Downstream Code PC"

PHASE_MAP = {
    "Avant charge": {"Init", "Lock Connector", "CableCheck"},
    "Charge": {"Charge"},
    "Fin de charge": {"Fin de charge"},
    "Unknown": {"Unknown"},
}

SITE_COLOR_PALETTE = [
    "#f97316",
    "#a855f7",
    "#eab308",
    "#ef4444",
    "#6366f1",
    "#3b82f6",
    "#06b6d4",
    "#10b981",
    "#0ea5e9",
    "#f43f5e",
]

templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)

_vehicle_strategy_cache = None


def _get_vehicle_strategy():
    global _vehicle_strategy_cache

    if _vehicle_strategy_cache is not None:
        return _vehicle_strategy_cache

    _vehicle_strategy_cache = ("k.Vehicle", "")

    return _vehicle_strategy_cache


def _calculate_period_dates(period_type: str, date_debut: date | None = None, date_fin: date | None = None):
    today = date.today()

    if period_type == "focus_jours":
        return today, today
    elif period_type == "mois":
        return date(today.year, today.month, 1), today
    elif period_type == "j-1":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    elif period_type == "semaine-1":
        today_weekday = today.weekday()
        days_since_monday = today_weekday
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday, last_sunday
    elif period_type == "toute_periode":
        return None, None
    elif period_type == "manuel":
        return date_debut, date_fin
    else:
        today_weekday = today.weekday()
        days_since_monday = today_weekday
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday, last_sunday


VALID_TEMP_COLUMNS = {
    "DCBM_Temp_H_Min", "DCBM_Temp_H_Max", "DCBM_Temp_H_Moy",
    "DCBM_Temp_L_Min", "DCBM_Temp_L_Max", "DCBM_Temp_L_Moy",
    "PISTOLET_Temp_1_Min", "PISTOLET_Temp_1_Max", "PISTOLET_Temp_1_Moy",
    "PISTOLET_Temp_2_Min", "PISTOLET_Temp_2_Max", "PISTOLET_Temp_2_Moy",
    "DCBM_Temp_Moy_Diff", "PISTOLET_Temp_Moy_Diff", "DCBM_PISTOLET_Moy_Diff",
}


def _build_conditions(
    sites: str,
    date_debut: date | None,
    date_fin: date | None,
    table_alias: str | None = None,
    error_types: list[str] | None = None,
    moments: list[str] | None = None,
    energy_mode: str = "all",
    energy_min: str = "",
    energy_max: str = "",
    temp_column: str = "",
    temp_mode: str = "all",
    temp_min: str = "",
    temp_max: str = "",
    warnings: list[int] | None = None,
):
    conditions = ["1=1"]
    params = {}

    alias = f"{table_alias}." if table_alias else ""
    datetime_col = f"{alias}`Datetime start`"
    site_col = f"{alias}Site"

    if date_debut:
        conditions.append(f"{datetime_col} >= :date_debut")
        params["date_debut"] = str(date_debut)
    if date_fin:
        conditions.append(f"{datetime_col} < DATE_ADD(:date_fin, INTERVAL 1 DAY)")
        params["date_fin"] = str(date_fin)
    if sites:
        site_list = [s.strip() for s in sites.split(",") if s.strip()]
        if site_list:
            placeholders = ",".join([f":site_{i}" for i in range(len(site_list))])
            conditions.append(f"{site_col} IN ({placeholders})")
            for i, s in enumerate(site_list):
                params[f"site_{i}"] = s

    if error_types:
        error_placeholders = ",".join([f":error_type_{i}" for i in range(len(error_types))])
        conditions.append(f"{alias}type_erreur IN ({error_placeholders})")
        for i, et in enumerate(error_types):
            params[f"error_type_{i}"] = et

    if moments:
        moment_placeholders = ",".join([f":moment_{i}" for i in range(len(moments))])
        conditions.append(f"{alias}moment IN ({moment_placeholders})")
        for i, m in enumerate(moments):
            params[f"moment_{i}"] = m

    # Energy filter (SQL-level)
    energy_mode_norm = (energy_mode or "all").strip().lower()
    if energy_mode_norm != "all":
        energy_col = f"{alias}`Energy (Kwh)`"
        if energy_mode_norm == "lt" and energy_max and energy_max.strip():
            try:
                params["energy_max_val"] = float(energy_max.strip())
                conditions.append(f"{energy_col} IS NOT NULL AND {energy_col} <= :energy_max_val")
            except (ValueError, TypeError):
                pass
        elif energy_mode_norm == "gt" and energy_min and energy_min.strip():
            try:
                params["energy_min_val"] = float(energy_min.strip())
                conditions.append(f"{energy_col} IS NOT NULL AND {energy_col} >= :energy_min_val")
            except (ValueError, TypeError):
                pass
        elif energy_mode_norm == "between":
            try:
                if energy_min and energy_min.strip():
                    params["energy_min_val"] = float(energy_min.strip())
                    conditions.append(f"{energy_col} IS NOT NULL AND {energy_col} >= :energy_min_val")
                if energy_max and energy_max.strip():
                    params["energy_max_val"] = float(energy_max.strip())
                    conditions.append(f"{energy_col} <= :energy_max_val")
            except (ValueError, TypeError):
                pass

    # Temperature filter via subquery on kpi_temperature
    temp_mode_norm = (temp_mode or "all").strip().lower()
    if temp_column and temp_column.strip() in VALID_TEMP_COLUMNS and temp_mode_norm != "all":
        safe_col = f"`{temp_column.strip()}`"
        id_col = f"{alias}ID"
        if temp_mode_norm == "lt" and temp_max and temp_max.strip():
            try:
                params["temp_max_val"] = float(temp_max.strip())
                conditions.append(
                    f"{id_col} IN (SELECT t.ID FROM kpi_temperature t WHERE t.{safe_col} IS NOT NULL AND t.{safe_col} <= :temp_max_val)"
                )
            except (ValueError, TypeError):
                pass
        elif temp_mode_norm == "gt" and temp_min and temp_min.strip():
            try:
                params["temp_min_val"] = float(temp_min.strip())
                conditions.append(
                    f"{id_col} IN (SELECT t.ID FROM kpi_temperature t WHERE t.{safe_col} IS NOT NULL AND t.{safe_col} >= :temp_min_val)"
                )
            except (ValueError, TypeError):
                pass
        elif temp_mode_norm == "between":
            sub_conds = [f"t.{safe_col} IS NOT NULL"]
            try:
                if temp_min and temp_min.strip():
                    params["temp_min_val"] = float(temp_min.strip())
                    sub_conds.append(f"t.{safe_col} >= :temp_min_val")
                if temp_max and temp_max.strip():
                    params["temp_max_val"] = float(temp_max.strip())
                    sub_conds.append(f"t.{safe_col} <= :temp_max_val")
                if len(sub_conds) > 1:
                    conditions.append(
                        f"{id_col} IN (SELECT t.ID FROM kpi_temperature t WHERE {' AND '.join(sub_conds)})"
                    )
            except (ValueError, TypeError):
                pass

    # Warning filter
    if warnings:
        warning_col = f"{alias}warning"
        warning_conditions = []
        for i, w in enumerate(warnings):
            param_name = f"warning_{i}"
            params[param_name] = int(w)
            warning_conditions.append(
                f"FIND_IN_SET(:{param_name}, REPLACE(CAST({warning_col} AS CHAR), ' ', '')) > 0"
            )
        if warning_conditions:
            conditions.append("(" + " OR ".join(warning_conditions) + ")")

    return " AND ".join(conditions), params


def _apply_energy_filter(df: pd.DataFrame, energy_mode: str = "all", energy_min: str = "", energy_max: str = "") -> pd.DataFrame:
    """Apply energy filter on a DataFrame that has 'Energy (Kwh)' column."""
    energy_mode_norm = (energy_mode or "all").strip().lower()
    if energy_mode_norm == "all" or "Energy (Kwh)" not in df.columns or df.empty:
        return df

    energy_col = pd.to_numeric(df["Energy (Kwh)"], errors="coerce")
    energy_mask = energy_col.notna()

    try:
        if energy_mode_norm == "lt" and energy_max and energy_max.strip():
            energy_mask &= energy_col <= float(energy_max.strip())
        elif energy_mode_norm == "gt" and energy_min and energy_min.strip():
            energy_mask &= energy_col >= float(energy_min.strip())
        elif energy_mode_norm == "between":
            if energy_min and energy_min.strip():
                energy_mask &= energy_col >= float(energy_min.strip())
            if energy_max and energy_max.strip():
                energy_mask &= energy_col <= float(energy_max.strip())
    except (ValueError, TypeError):
        return df

    return df[energy_mask].copy()


def _apply_temperature_filter(df: pd.DataFrame, temp_column: str = "", temp_mode: str = "all", temp_min: str = "", temp_max: str = "") -> pd.DataFrame:
    """Filter DataFrame by temperature: queries kpi_temperature for matching IDs, then filters."""
    temp_mode_norm = (temp_mode or "all").strip().lower()
    if temp_mode_norm == "all" or not temp_column or temp_column.strip() not in VALID_TEMP_COLUMNS:
        return df
    if df.empty or "ID" not in df.columns:
        return df

    safe_col = f"`{temp_column.strip()}`"
    sub_conds = [f"{safe_col} IS NOT NULL"]
    sub_params = {}

    try:
        if temp_mode_norm == "lt" and temp_max and temp_max.strip():
            sub_params["temp_max_val"] = float(temp_max.strip())
            sub_conds.append(f"{safe_col} <= :temp_max_val")
        elif temp_mode_norm == "gt" and temp_min and temp_min.strip():
            sub_params["temp_min_val"] = float(temp_min.strip())
            sub_conds.append(f"{safe_col} >= :temp_min_val")
        elif temp_mode_norm == "between":
            if temp_min and temp_min.strip():
                sub_params["temp_min_val"] = float(temp_min.strip())
                sub_conds.append(f"{safe_col} >= :temp_min_val")
            if temp_max and temp_max.strip():
                sub_params["temp_max_val"] = float(temp_max.strip())
                sub_conds.append(f"{safe_col} <= :temp_max_val")
    except (ValueError, TypeError):
        return df

    if len(sub_conds) <= 1:
        return df

    sql = f"SELECT ID FROM kpi_temperature WHERE {' AND '.join(sub_conds)}"
    temp_df = query_df(sql, sub_params)
    if temp_df.empty:
        return df.iloc[0:0].copy()

    valid_ids = set(temp_df["ID"].astype(str).tolist())
    return df[df["ID"].astype(str).isin(valid_ids)].copy()


def _apply_warning_filter(df: pd.DataFrame, warnings: list[int] | None) -> pd.DataFrame:
    """Filter DataFrame to keep only rows whose warning code is in the given list."""
    if not warnings or "warning" not in df.columns or df.empty:
        return df
    from routers.warning_utils import parse_warning_codes
    warning_set = set(warnings)
    mask = df["warning"].apply(lambda v: bool(set(parse_warning_codes(v)) & warning_set))
    return df[mask].copy()


def _apply_status_filters(df: pd.DataFrame, error_type_list, moment_list) -> pd.DataFrame:
    """
    error_type_list=None  → pas de filtre type (tous sélectionnés)
    error_type_list=[]    → filtre vide (rien sélectionné → 100% réussite)
    error_type_list=[...] → filtre sur les types listés
    Idem pour moment_list.
    """
    df["is_ok"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).eq(0)
    if "warning" in df.columns:
        from routers.warning_utils import parse_warning_codes
        df["is_warning"] = df["warning"].apply(lambda v: len(parse_warning_codes(v)) > 0)
        df["is_ok"] = df["is_ok"] | df["is_warning"]
    else:
        df["is_warning"] = False
    mask_nok = ~df["is_ok"]
    mask_type = (
        df["type_erreur"].isin(error_type_list)
        if error_type_list is not None and "type_erreur" in df.columns
        else pd.Series(True, index=df.index)
    )
    mask_moment = (
        df["moment"].isin(moment_list)
        if moment_list is not None and "moment" in df.columns
        else pd.Series(True, index=df.index)
    )
    df["is_ok_filt"] = np.where(mask_nok & mask_type & mask_moment, False, True)
    return df


def _apply_super_filters(
    df: pd.DataFrame,
    charge_type: str = "Toutes",
    energy_max: str = "",
    energy_operator: str = "<=",
    vehicle_type: str = "Tous",
    voltage_800v: str = "Tous",
    pdc_filter: str = "",
    mac_address_filter: str = "",
    pmax_min: str = "",
    pmax_operator: str = ">=",
    duration: str = "",
    duration_operator: str = ">=",
    error_moment: str = "",
) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    if charge_type != "Toutes":
        if "is_ok" in df.columns:
            if charge_type == "OK":
                if df["is_ok"].dtype == bool:
                    mask = mask & (df["is_ok"] == True)
                else:
                    mask = mask & (df["is_ok"] == 1)
            elif charge_type == "Non OK":
                if df["is_ok"].dtype == bool:
                    mask = mask & (df["is_ok"] == False)
                else:
                    mask = mask & (df["is_ok"] == 0)
        elif "state" in df.columns:
            state_num = pd.to_numeric(df["state"], errors="coerce").fillna(1)
            if charge_type == "OK":
                mask = mask & (state_num == 0)
            elif charge_type == "Non OK":
                mask = mask & (state_num == 1)

    if energy_max and energy_max.strip():
        try:
            energy_val = float(energy_max.strip())
            op = str(energy_operator).strip() if energy_operator else "<="

            op_decoded = unquote(op)

            if "Energy (Kwh)" in df.columns:
                energy_col = pd.to_numeric(df["Energy (Kwh)"], errors="coerce")

                if op_decoded == ">=" or op == ">=" or op == "%3E%3D":
                    mask = mask & (energy_col >= energy_val) & energy_col.notna()
                elif op_decoded == "<=" or op == "<=" or op == "%3C%3D":
                    mask = mask & (energy_col <= energy_val) & energy_col.notna()
                else:
                    mask = mask & (energy_col <= energy_val) & energy_col.notna()
        except (ValueError, TypeError):
            pass

    if vehicle_type != "Tous" and vehicle_type.strip():
        if "Vehicle" in df.columns:
            mask = mask & (df["Vehicle"] == vehicle_type.strip())

    if voltage_800v != "Tous":
        voltage_cols = ["charge_900V", "Is800V", "800V", "VoltageClass", "EV_800V", "Voltage"]
        voltage_col = None
        for col in voltage_cols:
            if col in df.columns:
                voltage_col = col
                break

        if voltage_col:
            if voltage_800v == "800V":
                if voltage_col == "charge_900V":
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num == 1)
                elif df[voltage_col].dtype == bool:
                    mask = mask & (df[voltage_col] == True)
                else:
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num >= 650)
            elif voltage_800v == "Non 800V":
                if voltage_col == "charge_900V":
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num == 0)
                elif df[voltage_col].dtype == bool:
                    mask = mask & (df[voltage_col] == False)
                else:
                    voltage_num = pd.to_numeric(df[voltage_col], errors="coerce")
                    mask = mask & (voltage_num < 650)

    if pdc_filter and pdc_filter.strip():
        try:
            pdc_list = [p.strip() for p in pdc_filter.split(",") if p.strip()]
            if pdc_list and "PDC" in df.columns:
                pdc_str = df["PDC"].astype(str)
                mask = mask & pdc_str.isin(pdc_list)
        except Exception:
            pass

    if mac_address_filter and mac_address_filter.strip():
        mac_search = mac_address_filter.strip()
        if "MAC Address" in df.columns:
            try:
                mac_series = df["MAC Address"].fillna("")
                mac_str = mac_series.astype(str)
                mac_str_normalized = mac_str.str.replace(":", "", regex=False).str.replace("-", "", regex=False).str.replace(" ", "", regex=False).str.upper()
                mac_search_normalized = mac_search.replace(":", "").replace("-", "").replace(" ", "").upper()

                mac_mask = mac_str_normalized.str.contains(mac_search_normalized, na=False, regex=False)

                mask = mask & mac_mask
            except Exception as e:
                pass

    if pmax_min and pmax_min.strip():
        try:
            pmax_val = float(pmax_min.strip())
            if "Max Power (Kw)" in df.columns:
                pmax_col = pd.to_numeric(df["Max Power (Kw)"], errors="coerce")
                if pmax_operator == ">=":
                    mask = mask & (pmax_col >= pmax_val) & pmax_col.notna()
                else:
                    mask = mask & (pmax_col <= pmax_val) & pmax_col.notna()
        except (ValueError, TypeError):
            pass

    if duration and duration.strip():
        try:
            duration_val = float(duration.strip())
            duration_cols = ["duration", "Duration", "duration_min", "Duration (min)"]
            duration_col = None
            for col in duration_cols:
                if col in df.columns:
                    duration_col = col
                    break

            if duration_col:
                duration_num = pd.to_numeric(df[duration_col], errors="coerce")
                if duration_operator == ">=":
                    mask = mask & (duration_num >= duration_val) & duration_num.notna()
                else:
                    mask = mask & (duration_num <= duration_val) & duration_num.notna()
        except (ValueError, TypeError):
            pass

    if error_moment and error_moment.strip():
        moment_mapping = {
            "Avant charge": ["Init", "Lock Connector", "CableCheck"],
            "Charge": ["Charge"],
            "Fin de charge": ["Fin de charge"]
        }

        if error_moment in moment_mapping:
            moment_values = moment_mapping[error_moment]
            if "moment" in df.columns:
                mask = mask & df["moment"].isin(moment_values)

    return df[mask].copy()


def _map_moment_label(val: int) -> str:
    try:
        v = int(val)
    except Exception:
        return "Unknown"

    if v == 0:
        return "Fin de charge"
    if 1 <= v <= 2:
        return "Init"
    if 4 <= v <= 6:
        return "Lock Connector"
    if v == 7:
        return "CableCheck"
    if v == 8:
        return "Charge"
    if v > 8:
        return "Fin de charge"
    return "Unknown"


def _map_phase_label(moment: str | int | float | None) -> str:
    if pd.isna(moment):
        return "Unknown"

    if isinstance(moment, (list, tuple, set)):
        for value in moment:
            mapped = _map_phase_label(value)
            if mapped != "Unknown":
                return mapped
        return "Unknown"

    moment_str = str(moment)

    for phase, moments in PHASE_MAP.items():
        if moment_str in moments:
            return phase

    return "Unknown"


def _build_pivot_table(detail_df: pd.DataFrame, by_site: pd.DataFrame) -> dict[str, Any]:
    if detail_df.empty:
        return {"columns": [], "rows": []}

    if "Site" not in detail_df.columns:
        return {"columns": [], "rows": []}

    pivot_df = detail_df.assign(
        _site=detail_df["Site"],
        _type=detail_df.get("type", ""),
        _moment=detail_df["moment_label"],
        _step=detail_df["step"],
        _code=detail_df["code"],
    )

    pivot_table = pd.pivot_table(
        pivot_df,
        index="_site",
        columns=["_type", "_moment", "_step", "_code"],
        aggfunc="size",
        fill_value=0,
    ).sort_index(axis=1)

    pivot_table = pivot_table.reset_index()

    if isinstance(pivot_table.columns, pd.MultiIndex):
        pivot_table.columns = [
            col[0] if col[0] in ["_site", "Site"] else " | ".join(str(c) for c in col if c).strip()
            for col in pivot_table.columns
        ]

    if "_site" in pivot_table.columns:
        pivot_table = pivot_table.rename(columns={"_site": "Site"})

    new_columns = []
    for col in pivot_table.columns:
        if col == "Site":
            new_columns.append("Site")
        elif isinstance(col, tuple):
            new_columns.append(" | ".join(map(str, col)).strip())
        else:
            new_columns.append(str(col))
    pivot_table.columns = new_columns

    if "Site" not in pivot_table.columns:
        return {"columns": [], "rows": []}

    if "Site" not in by_site.columns:
        by_site = by_site.reset_index()

    pivot_table = pivot_table.merge(
        by_site[["Site", "Total_Charges"]].rename(columns={"Total_Charges": "Total Charges"}),
        on="Site",
        how="left",
    )

    ordered_columns = ["Site", "Total Charges"] + [
        col for col in pivot_table.columns if col not in {"Site", "Total Charges"}
    ]
    pivot_table = pivot_table[ordered_columns].fillna(0)

    numeric_cols = [col for col in pivot_table.columns if col != "Site"]
    pivot_table[numeric_cols] = pivot_table[numeric_cols].astype(int)

    return {
        "columns": pivot_table.columns.tolist(),
        "rows": pivot_table.to_dict("records"),
    }


def _comparaison_base_context(
    request: Request,
    filters: dict,
    site_focus: str = "",
    month_focus: str = "",
    error_message: str | None = None,
):
    return {
        "request": request,
        "site_rows": [],
        "count_bars": [],
        "percent_bars": [],
        "max_total": 0,
        "peak_rows": [],
        "heatmap_rows": [],
        "heatmap_hours": [],
        "heatmap_max": 0,
        "site_options": [],
        "site_focus": site_focus,
        "month_options": [],
        "month_focus": month_focus,
        "monthly_rows": [],
        "daily_rows": [],
        "filters": filters,
        "error_message": error_message,
    }
