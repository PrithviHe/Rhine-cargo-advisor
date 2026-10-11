"""Estimate vessel cargo and truckloads from forecast gauge level."""

import math


def estimate_cargo(
    forecast_cm,
    reference_gauge_cm,
    reference_depth_m,
    clearance_m,
    empty_draft_m,
    full_draft_m,
    max_payload_t,
    requested_cargo_t,
    truck_capacity_t,
):
    # Validate inputs.
    values = (
        forecast_cm,
        reference_gauge_cm,
        reference_depth_m,
        clearance_m,
        empty_draft_m,
        full_draft_m,
        max_payload_t,
        requested_cargo_t,
        truck_capacity_t,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("All inputs must be finite numbers.")

    if reference_depth_m <= 0 or clearance_m <= 0 or empty_draft_m <= 0:
        raise ValueError("Depth, clearance, and empty draft must be positive.")

    if full_draft_m <= empty_draft_m:
        raise ValueError("Full-load draft must exceed empty draft.")

    if max_payload_t <= 0 or truck_capacity_t <= 0:
        raise ValueError("Vessel and truck payloads must be positive.")

    if requested_cargo_t < 0:
        raise ValueError("Requested cargo cannot be negative.")

    # Approximation: depth changes one-to-one with gauge level.
    depth_m = (
        reference_depth_m
        + (forecast_cm - reference_gauge_cm) / 100
    )

    # Leave clearance beneath the vessel.
    available_draft_m = depth_m - clearance_m

    # Approximation: payload increases linearly with draft.
    loading_fraction = (
        (available_draft_m - empty_draft_m)
        / (full_draft_m - empty_draft_m)
    )
    loading_fraction = max(0.0, min(1.0, loading_fraction))

    # Capacity and cargo carried in one vessel load.
    capacity_t = max_payload_t * loading_fraction
    vessel_cargo_t = min(requested_cargo_t, capacity_t)
    remaining_t = max(0.0, requested_cargo_t - vessel_cargo_t)

    # Round away floating-point noise before rounding up.
    truckloads = math.ceil(round(remaining_t / truck_capacity_t, 12))

    return {
        "depth_m": depth_m,
        "available_draft_m": available_draft_m,
        "capacity_t": capacity_t,
        "vessel_cargo_t": vessel_cargo_t,
        "remaining_t": remaining_t,
        "truckloads": truckloads,
        "insufficient_empty_draft": available_draft_m < empty_draft_m,
    }