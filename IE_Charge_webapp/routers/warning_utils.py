WARNING_DETAILS = {
    1: "Fin de charge",
    3: "DCBM_Temp_H_Moy > 65",
    5: "DCBM_Temp_L_Moy > 65",
    7: "PISTOLET_Temp_1_Moy > 65",
    9: "PISTOLET_Temp_2_Moy > 65",
    10: "PISTOLET_Temp_Moy_Diff > 10",
    11: "DCBM_Temp_Moy_Diff > 10",
    12: "DCBM_PISTOLET_Moy_Diff > 25",
}


def parse_warning_codes(value) -> list[int]:
    if value is None:
        return []

    if isinstance(value, (int, float)):
        try:
            code = int(value)
            return [code] if code > 0 else []
        except (TypeError, ValueError):
            return []

    raw = str(value).strip()
    if not raw:
        return []

    codes: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            code = int(float(token))
        except (TypeError, ValueError):
            continue
        if code > 0:
            codes.append(code)
    return sorted(set(codes))


def get_warning_details(value) -> list[str]:
    return [WARNING_DETAILS.get(code, f"Code {code}") for code in parse_warning_codes(value)]
