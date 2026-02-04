# recommendations.py — weather-aware, farmer-friendly
from typing import Dict, Optional

def recommend_actions(
    crop: str,
    predicted_yield: float,
    pesticides: float = None,
    stage: str = "vegetative",
    avg_temp: float = 25.0,
    rainfall: float = 1000.0,
    forecast_next_48h_mm: Optional[float] = None
) -> Dict:
    actions = {}

    # Irrigation (mm)
    base_mm = 18
    if avg_temp >= 32: base_mm += 4
    if predicted_yield < 2500: base_mm += 4
    if stage in ("flowering",): base_mm -= 2
    if forecast_next_48h_mm is not None and forecast_next_48h_mm >= 15:
        base_mm -= 6
        cut_reason = f"Rain expected (~{forecast_next_48h_mm:.0f} mm in 48h)"
    else:
        cut_reason = None
    base_mm = max(8, min(base_mm, 30))

    irr_reason = "Night irrigation saves water; 2 cycles reduce runoff."
    if cut_reason: irr_reason += f" | {cut_reason} → reduce water."
    actions["irrigation"] = {"label": f"Apply {base_mm} mm tonight (split into 2 cycles)",
                             "reason": irr_reason}

    # Fertilizer (kg/acre)
    if stage in ("sowing","vegetative"):
        fert = "Urea 20 kg now + 20 kg in 10 days; DAP 25 kg now"
    elif stage in ("tillering",):
        fert = "Urea 25 kg now + 25 kg at 10 days; MOP 15 kg now"
    elif stage in ("flowering","maturation"):
        fert = "Light top-dress: Urea 15 kg; avoid excess N; add MOP 10 kg if K is low"
    else:
        fert = "Maintain previous schedule; avoid over-application"
    actions["fertilizer"] = {"label": fert,
                             "reason": "Split doses improve uptake; reduces loss."}

    # Pest control
    if pesticides is not None and pesticides > 5:
        pest = "Reduce chemical use; 8 pheromone traps/acre; spot-spray only above threshold"
        why = "High pesticide trend—adopt IPM to cut cost/resistance."
    else:
        pest = "Install 4–6 pheromone traps/acre; scout field edges every 3 days"
        why = "Early detection reduces sprays; safer & cheaper."
    actions["pest_control"] = {"label": pest, "reason": why}

    return actions
