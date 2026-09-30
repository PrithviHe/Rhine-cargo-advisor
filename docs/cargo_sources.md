# Cargo parameters — verification log

Every constant in `modeling/cargo.py` is listed here. None of them may stay
marked *unverified* in the final submission: either a source is recorded, or the
value is labelled an assumption on the slide and in the app.

**How to use this file.** Fill in the three right-hand columns as you find the
figures, then update the default in `modeling/cargo.py` and change the inline
comment from `[TO VERIFY]` to the document name.

---

## 1 · Gauge and fairway, Kaub

| Constant | Current value | Source found | Document / URL | Checked by, date |
|---|---|---|---|---|
| `glw_cm` — equivalent low-water level (GlW) | **77 cm** | ☑ | PEGELONLINE station master data, KAUB (no. 25700100). Listed under *Kennzeichnende Wasserstände* as **GlW 77 cm, valid from 01.01.2023**. | 27.09.2026 |
| `fairway_depth_at_glw_m` — maintained fairway depth at GlW | **1.90 m** | ☑ | Same page, listed as **TuGlW 190 cm, valid from 01.01.2004**. TuGlW = depth below the equivalent water level, i.e. the maintained fairway depth. Worth confirming the abbreviation once in a WSV glossary. | 27.09.2026 |
| `under_keel_clearance_m` — safety margin under the keel | 0.30 m | ☐ | still an assumption | |

**Station identity, for the record**

| Field | Value |
|---|---|
| Station | KAUB, no. 25700100 |
| UUID | `1d26e504-7f9e-480a-b52c-5932be6549ab` |
| Operator | WSA Rhein, Standort Bingen |
| River kilometre | 546.230 |
| Gauge datum (PNP) | 67.67 m above NHN (DE_DHHN2016_NH_RP), valid from 01.11.2019 |

**Other characteristic levels on the same page** — useful for putting a reading in
context, not used in the calculation:

| Code | Value | Meaning |
|---|---|---|
| HHW | 825 cm | highest known level, 05.01.1883 |
| HSW | 640 cm | highest navigable level |
| MHW | 544 cm | mean high water, 2010–2020 |
| MW | 208 cm | mean water, 2010–2020 |
| MNW | 65 cm | mean low water, 2010–2020 |
| NNW | 25 cm | lowest known level, 22.10.2018 |

**Note on NNW.** The lowest level on record for the 2010–2020 reference period is
25 cm. The readings in August and September 2026 fall well below that, including
negative values. A negative reading means the water surface is below the gauge
datum; it is not a water depth. The fairway still carried roughly 1.1 m at a
reading of −3 cm.

**Where to look**

- WSV — Wasserstraßen- und Schifffahrtsverwaltung des Bundes
- ELWIS — Elektronisches Wasserstraßen-Informationssystem, notices to skippers
- Zentralkommission für die Rheinschifffahrt (ZKR), market observation reports

**What matters**

The GlW is periodically re-determined; note *which* GlW edition the figure comes
from (for example GlW 2012 or GlW 2022), because the value changes between them.
The fairway depth is a maintenance target, not a guarantee — state that in a
footnote if you cite it.

---

## 2 · Reference vessel — GMS 110, CEMT class Va

| Constant | Value | Source found | Document | Checked |
|---|---|---|---|---|
| `vessel_name` | GMS 110 (CEMT Va) | ☑ | PLATINA3 D2.2, restating ECCONET | 29.09.2026 |
| `min_draft_m` — T_min | **1.35 m** | ☑ | PLATINA3 D2.2, table of vessel characteristics | 29.09.2026 |
| `max_draft_m` — T_max | **3.50 m** | ☑ | same table | 29.09.2026 |
| `min_deadweight_t` at T_min | **400 t** | ☑ | same table | 29.09.2026 |
| `max_deadweight_t` at T_max | **2,900 t** | ☑ | same table | 29.09.2026 |

**Sources**

- PLATINA3 (2022). *Deliverable D2.2 — Options for shallow-water / climate
  resilient vessels.* Restates the ECCONET figures and defines T_min in a
  footnote.
- Beuthe, M. et al. (2013). *ECCONET: Climate change and navigation — vessel
  characteristics and navigation parameters.* UNECE. Original source of the
  1.35 m / 3.50 m / 2,873 t figures for the GMS 110.
- UNECE (2019). *Classification of European inland waterways.* Gives the class
  Va ranges: 95–110 m length, 11.4 m beam, 2.50–2.80 m draught,
  1,500–3,000 t capacity.
- Busch, D., Bergen, A., Wosniok, W. & Krause, K. (2021). *Analyse des
  Schiffsverkehrs auf dem nordrhein-westfälischen Niederrhein 2018–2020*
  (LANUV Fachbericht 122). Gives roughly 2,800 t for the Großes Rheinschiff on
  the Lower Rhine — consistent order of magnitude.

**Two distinctions that matter**

*Class versus vessel.* CEMT class Va describes a range (2.50–2.80 m draught),
not one ship. The GMS 110 is a specific reference configuration within that
class with a 3.50 m maximum draught. Always write "reference vessel GMS 110,
CEMT class Va" — never "CEMT Va has a draught of 3.50 m".

*T_min is not zero cargo.* The PLATINA3 footnote defines T_min as the draught
of the empty, stern-trimmed vessel measured from its lowest point, and states
that stores or a small amount of cargo may still be on board. The same table
gives a minimum deadweight of 400 t at that draught.

Because both endpoints are given, `payload_t()` interpolates between two
sourced points — 400 t at 1.35 m and 2,900 t at 3.50 m — instead of assuming
zero cargo at T_min, which the source does not support.

**One caveat to state in the presentation.** Deadweight includes bunkers,
water and stores, so it slightly overstates the commercial cargo, most
noticeably near the minimum draught. We model deadweight because that is what
the source gives.

**Operational consequence.** Below a safe draught of 1.35 m the reference
vessel cannot pass at all. At Kaub that corresponds to a gauge reading of
**52 cm**. Full deadweight is reached from **267 cm**.

## 3 · Replacement transport

| Constant | Current value | Source found | Document / URL | Checked by, date |
|---|---|---|---|---|
| `truck_payload_t` — usable payload per articulated lorry | 25 t | ☐ | | |
| `train_payload_t` — usable payload per block train | 1,200 t | ☐ | | |

**What matters**

For dense bulk cargo the weight limit binds before the volume limit, so 24–26 t
per lorry is plausible. Block-train capacity varies widely by commodity and by
the operator's train length limit — state which assumption you used.

---

## 4 · Not implemented

**Low-water surcharge (Kleinwasserzuschlag, KWZ).** No surcharge schedule exists
in the project data, so the app shows no cost figures at all rather than
estimated ones. If a schedule is obtained, it belongs in `cargo.py` as a lookup
keyed on the gauge level, and the Transport Options page has a placeholder ready
for it.

---

## 5 · Sensitivity instead of certainty

Until the constants are sourced, the honest presentation is a range rather than a
point. Two mechanisms are already built in:

- **Cargo Planning page** — varies `max_payload_t` by ±20 % and shows how the
  truck count moves.
- **Settings sidebar** — the three most influential constants can be changed live
  during a demo.

Use these instead of claiming precision the data does not support.
