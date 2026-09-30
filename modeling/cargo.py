"""
HydroLoad-AI | modeling/cargo.py

Deterministic conversion: gauge level -> fairway depth -> safe draft ->
deadweight -> shortfall -> replacement capacity.

Every constant is listed in docs/cargo_sources.md with its source and the date
it was checked. Constants marked [ASSUMPTION] are not sourced.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Assumptions:
    # ---------------------------------------------------------------- gauge
    # Source: PEGELONLINE station master data, KAUB (no. 25700100),
    # WSA Rhein / Standort Bingen, Rhine-km 546.230. Gauge datum 67.67 m
    # above NHN (DE_DHHN2016_NH_RP), valid from 01.11.2019. Checked 27.09.2026.
    glw_cm: float = 77.0                   # GlW, valid from 01.01.2023  [SOURCED]
    fairway_depth_at_glw_m: float = 1.90   # TuGlW 190 cm                [SOURCED]
    under_keel_clearance_m: float = 0.30   # Flottwasser                 [ASSUMPTION]

    # ------------------------------------------------------- reference vessel
    # GMS 110, CEMT class Va. Source: PLATINA3, Deliverable D2.2 "Options for
    # shallow-water / climate resilient vessels" (2022), which restates the
    # ECCONET figures (Beuthe et al., 2013) and defines T_min as the draught of
    # the empty, stern-trimmed vessel. Both deadweight endpoints are given
    # there, so the loading curve is interpolated between two sourced points
    # rather than assuming zero cargo at T_min.
    vessel_name: str = "GMS 110 (CEMT Va)"
    min_draft_m: float = 1.35              # T_min                       [SOURCED]
    max_draft_m: float = 3.50              # T_max                       [SOURCED]
    min_deadweight_t: float = 400.0        # DWT at T_min                [SOURCED]
    max_deadweight_t: float = 2900.0       # DWT at T_max                [SOURCED]

    # ---------------------------------------------------- replacement transport
    truck_payload_t: float = 25.0          # articulated lorry           [ASSUMPTION]
    train_payload_t: float = 1200.0        # block train                 [ASSUMPTION]

    # ----- backwards compatibility with the earlier parameter names -----
    @property
    def light_draft_m(self) -> float:
        return self.min_draft_m

    @property
    def max_payload_t(self) -> float:
        return self.max_deadweight_t

    def as_dict(self) -> dict:
        d = asdict(self)
        d["light_draft_m"] = self.min_draft_m
        d["max_payload_t"] = self.max_deadweight_t
        return d


# Characteristic levels at Kaub, from the PEGELONLINE station master data.
# Not used in the calculation; useful for putting a reading in context.
CHARACTERISTIC_LEVELS_CM = {
    "HHW": 825, "HSW": 640, "MHW": 544, "MW": 208,
    "TuGlW": 190, "GlW": 77, "MNW": 65, "NNW": 25,
}

DEFAULTS = Assumptions()


# ---------------------------------------------------------------- geometry
def available_depth_m(level_cm: float, a: Assumptions = DEFAULTS) -> float:
    """Fairway depth available at this gauge reading."""
    return a.fairway_depth_at_glw_m + (level_cm - a.glw_cm) / 100.0


def safe_draft_m(level_cm: float, a: Assumptions = DEFAULTS) -> float:
    """Draft a vessel may safely take: available depth minus under-keel clearance."""
    return max(0.0, available_depth_m(level_cm, a) - a.under_keel_clearance_m)


def is_navigable(level_cm: float, a: Assumptions = DEFAULTS) -> bool:
    """
    False when the safe draft falls below the vessel's minimum draft: the
    reference vessel cannot pass at all, however little it carries.
    """
    # 1 mm tolerance: floating point makes an exact boundary read as "below"
    return safe_draft_m(level_cm, a) >= a.min_draft_m - 1e-3


def min_navigable_level_cm(a: Assumptions = DEFAULTS) -> float:
    """Lowest gauge reading at which the reference vessel can still pass."""
    return a.glw_cm + 100 * (a.min_draft_m + a.under_keel_clearance_m
                             - a.fairway_depth_at_glw_m)


def full_load_level_cm(a: Assumptions = DEFAULTS) -> float:
    """Gauge reading from which the vessel can be loaded to maximum deadweight."""
    return a.glw_cm + 100 * (a.max_draft_m + a.under_keel_clearance_m
                             - a.fairway_depth_at_glw_m)


# ---------------------------------------------------------------- loading
def payload_t(level_cm: float, a: Assumptions = DEFAULTS) -> float:
    """
    Deadweight the vessel can carry at this gauge reading.

    Linear between the two sourced endpoints: 400 t at 1.35 m draft and
    2,900 t at 3.50 m. Below the minimum draft the vessel cannot navigate,
    which is reported as 0 t - use is_navigable() to tell the two apart.

    Deadweight includes bunkers and stores, so it slightly overstates the
    commercial cargo, most noticeably near the minimum draft.
    """
    if not is_navigable(level_cm, a):
        return 0.0
    d = min(safe_draft_m(level_cm, a), a.max_draft_m)
    span = a.max_draft_m - a.min_draft_m
    share = (d - a.min_draft_m) / span
    return round(a.min_deadweight_t
                 + share * (a.max_deadweight_t - a.min_deadweight_t), 1)


def normal_payload_t(a: Assumptions = DEFAULTS) -> float:
    """Deadweight under unrestricted water levels - the reference for shortfall."""
    return a.max_deadweight_t


def shortfall(level_cm: float, a: Assumptions = DEFAULTS) -> dict:
    """The whole chain for ONE gauge reading."""
    p = payload_t(level_cm, a)
    deficit = max(0.0, normal_payload_t(a) - p)
    return {
        "level_cm": round(level_cm, 1),
        "available_depth_m": round(available_depth_m(level_cm, a), 2),
        "safe_draft_m": round(safe_draft_m(level_cm, a), 2),
        "navigable": is_navigable(level_cm, a),
        "payload_t": p,
        "utilisation_pct": round(100 * p / a.max_deadweight_t, 1),
        "deficit_t": round(deficit, 1),
        "trucks": int(-(-deficit // a.truck_payload_t)),
        "trains": int(-(-deficit // a.train_payload_t)),
    }


def shortfall_range(lo_cm: float, mid_cm: float, hi_cm: float,
                    a: Assumptions = DEFAULTS) -> dict:
    """
    Carry the forecast interval through the chain. A LOWER level means a
    LARGER replacement requirement, so those figures run the other way.
    """
    low, mid, high = shortfall(lo_cm, a), shortfall(mid_cm, a), shortfall(hi_cm, a)
    return {
        "level": (low["level_cm"], mid["level_cm"], high["level_cm"]),
        "safe_draft_m": (low["safe_draft_m"], mid["safe_draft_m"], high["safe_draft_m"]),
        "navigable": (low["navigable"], mid["navigable"], high["navigable"]),
        "payload_t": (low["payload_t"], mid["payload_t"], high["payload_t"]),
        "deficit_t": (high["deficit_t"], mid["deficit_t"], low["deficit_t"]),
        "trucks": (high["trucks"], mid["trucks"], low["trucks"]),
        "trains": (high["trains"], mid["trains"], low["trains"]),
    }


def recommended_payload_t(lo_cm: float, a: Assumptions = DEFAULTS) -> float:
    """
    Loading follows the LOWER bound of the forecast interval, not the median.
    Overloading risks grounding; underloading only costs revenue, so the two
    errors are not equivalent.
    """
    return payload_t(lo_cm, a)


if __name__ == "__main__":
    A = Assumptions()
    print(f"Reference vessel: {A.vessel_name}")
    print(f"Navigable from   {min_navigable_level_cm(A):.0f} cm")
    print(f"Full load from   {full_load_level_cm(A):.0f} cm\n")
    print(f"{'level':>7} {'fairway':>9} {'draft':>8} {'navig.':>7} {'deadweight':>12} {'trucks':>8}")
    for lv in [-1, 20, 40, 52, 77, 120, 180, 267, 350]:
        r = shortfall(lv, A)
        print(f"{lv:7d} {r['available_depth_m']:7.2f} m {r['safe_draft_m']:6.2f} m "
              f"{('yes' if r['navigable'] else 'NO'):>7} {r['payload_t']:10,.0f} t "
              f"{r['trucks']:8d}")
