"""
HydroLoad-AI | modeling/cargo.py

Deterministische Umrechnung: Pegelstand -> Abladetiefe -> Zuladung ->
Fehlmenge -> Ersatzkapazitaet.

WICHTIG: Alle Konstanten unten sind ANNAHMEN und muessen vor der Abgabe
gegen offizielle Quellen (WSV, ELWIS, Reederei-Abladetabelle) geprueft werden.
Jede traegt den Status "TO VERIFY" oder eine Quelle.
"""
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class Assumptions:
    # --- Gewaesser / Pegel Kaub -------------------------------------------
    # Quelle: PEGELONLINE Stammdaten, Messstelle KAUB (Nr. 25700100,
    # UUID 1d26e504-7f9e-480a-b52c-5932be6549ab), WSA Rhein / Standort Bingen,
    # Rhein-km 546,230. Pegelnullpunkt 67,67 m ue. NHN (DE_DHHN2016_NH_RP),
    # gueltig ab 01.11.2019. Abgerufen 27.09.2026.
    glw_cm: float = 77.0                # GlW, gueltig ab 01.01.2023   [VERIFIED]
    fairway_depth_at_glw_m: float = 1.90  # TuGlW 190 cm               [VERIFIED]
    under_keel_clearance_m: float = 0.30  # Sicherheitsabstand unter Kiel [ANNAHME]

    # --- Referenzschiff: Grossmotorgueterschiff 110 m ---
    vessel_name: str = "GMS 110 m (reference)"
    max_draft_m: float = 2.80           # maximaler Tiefgang           [TO VERIFY]
    light_draft_m: float = 0.50         # Leertiefgang                 [TO VERIFY]
    max_payload_t: float = 2500.0       # Zuladung bei max. Tiefgang   [TO VERIFY]

    # --- Ersatzverkehr ---
    truck_payload_t: float = 25.0       # Nutzlast Sattelzug           [ANNAHME]
    train_payload_t: float = 1200.0     # Nutzlast Ganzzug             [ANNAHME]

    def as_dict(self) -> dict:
        return asdict(self)


# Charakteristische Wasserstaende am Pegel Kaub laut PEGELONLINE-Stammdaten.
# Nicht Teil der Rechnung, aber nuetzlich zur Einordnung eines Messwerts.
CHARACTERISTIC_LEVELS_CM = {
    "HHW": 825,    # hoechster bekannter Wasserstand, 05.01.1883
    "HSW": 640,    # hoechster Schifffahrtswasserstand, ab 01.01.1950
    "MHW": 544,    # mittlerer Hochwasserstand, 2010-2020
    "MW": 208,     # Mittelwasser, 2010-2020
    "TuGlW": 190,  # Tiefe unter GlW = unterhaltene Fahrrinnentiefe
    "GlW": 77,     # gleichwertiger Wasserstand, ab 01.01.2023
    "MNW": 65,     # mittlerer Niedrigwasserstand, 2010-2020
    "NNW": 25,     # niedrigster bekannter Wasserstand, 22.10.2018
}

DEFAULTS = Assumptions()


# --------------------------------------------------------------------------
def available_depth_m(level_cm: float, a: Assumptions = DEFAULTS) -> float:
    """Verfuegbare Fahrrinnentiefe aus dem Pegelstand."""
    return a.fairway_depth_at_glw_m + (level_cm - a.glw_cm) / 100.0


def safe_draft_m(level_cm: float, a: Assumptions = DEFAULTS) -> float:
    """Sicher fahrbarer Tiefgang: verfuegbare Tiefe minus Kielfreiheit."""
    return max(0.0, available_depth_m(level_cm, a) - a.under_keel_clearance_m)


def payload_t(level_cm: float, a: Assumptions = DEFAULTS) -> float:
    """Zuladung aus dem sicheren Tiefgang (lineare Naeherung der Abladekurve)."""
    d = min(safe_draft_m(level_cm, a), a.max_draft_m)
    if d <= a.light_draft_m:
        return 0.0
    span = a.max_draft_m - a.light_draft_m
    return round(a.max_payload_t * (d - a.light_draft_m) / span, 1)


def normal_payload_t(a: Assumptions = DEFAULTS) -> float:
    """Zuladung bei unbeschraenktem Wasserstand = Referenzwert fuer die Fehlmenge."""
    return a.max_payload_t


def shortfall(level_cm: float, a: Assumptions = DEFAULTS) -> dict:
    """Vollstaendige Kette fuer EINEN Pegelwert."""
    p = payload_t(level_cm, a)
    deficit = max(0.0, normal_payload_t(a) - p)
    return {
        "level_cm": round(level_cm, 1),
        "available_depth_m": round(available_depth_m(level_cm, a), 2),
        "safe_draft_m": round(safe_draft_m(level_cm, a), 2),
        "payload_t": p,
        "utilisation_pct": round(100 * p / a.max_payload_t, 1),
        "deficit_t": round(deficit, 1),
        "trucks": int(-(-deficit // a.truck_payload_t)),   # aufrunden
        "trains": int(-(-deficit // a.train_payload_t)),
    }


def shortfall_range(lo_cm: float, mid_cm: float, hi_cm: float,
                    a: Assumptions = DEFAULTS) -> dict:
    """
    Unsicherheit durchreichen: ein niedriger Pegel bedeutet MEHR Ersatzbedarf,
    die Reihenfolge dreht sich also um.
    """
    low, mid, high = shortfall(lo_cm, a), shortfall(mid_cm, a), shortfall(hi_cm, a)
    return {
        "level": (low["level_cm"], mid["level_cm"], high["level_cm"]),
        "safe_draft_m": (low["safe_draft_m"], mid["safe_draft_m"], high["safe_draft_m"]),
        "payload_t": (low["payload_t"], mid["payload_t"], high["payload_t"]),
        "deficit_t": (high["deficit_t"], mid["deficit_t"], low["deficit_t"]),
        "trucks": (high["trucks"], mid["trucks"], low["trucks"]),
        "trains": (high["trains"], mid["trains"], low["trains"]),
    }


def recommended_payload_t(lo_cm: float, a: Assumptions = DEFAULTS) -> float:
    """
    Beladeempfehlung asymmetrisch: nach dem UNTEREN Quantil, nicht nach dem Median.
    Ueberladung bedeutet Aufsetzen, Unterladung nur entgangenen Umsatz.
    """
    return payload_t(lo_cm, a)


if __name__ == "__main__":
    for lv in [200, 120, 78, 50, 30, 10]:
        print(shortfall(lv))
    print()
    print(shortfall_range(60, 100, 140))
