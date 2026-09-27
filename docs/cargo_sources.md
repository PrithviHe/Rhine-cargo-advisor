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

## 2 · Reference vessel

| Constant | Current value | Source found | Document / URL | Checked by, date |
|---|---|---|---|---|
| `vessel_name` | GMS 110 m | ☐ | | |
| `max_draft_m` | 2.80 m | ☐ | | |
| `light_draft_m` | 0.50 m | ☐ | | |
| `max_payload_t` | 2,500 t | ☐ | | |

**Where to look**

- A shipping company's loading table (Abladetabelle) for a Großmotorgüterschiff
- CCNR / ZKR vessel classification (CEMT class Va covers the 110 m motor vessel)
- Any published draft-to-payload curve for the Middle Rhine

**What matters**

`payload_t()` currently interpolates linearly between the light draft and the
maximum draft. Real loading curves are close to linear in the working range but
not exactly. If you obtain a real table, replace the linear formula with a lookup
and note the change here.

---

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
