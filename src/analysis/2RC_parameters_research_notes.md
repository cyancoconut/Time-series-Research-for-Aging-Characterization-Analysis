# 2RC ECM parameters (R₀, R₁/τ₁, R₂/τ₂): physical meaning and aging behaviour

**Research notes.** Physical interpretation of the pulse-fit parameters, how they
behave with SOC and state-of-health, and the measurement caveats specific to our
pulse protocol. Empirical numbers are from the Sony/Murata US18650VTC6 check-up
pulses (cell `…_007`, 9-point SOC×C-rate×direction grid, ~77–96 % SOH), fit by
`fit_2rc_pulse.py`. Literature is cited to anchor the interpretation; verify exact
page/DOI against the originals before quoting.

---

## 1. The model

A single CC current pulse (≈20 s) followed by a rest is described by a Thevenin
2RC network: an instantaneous ohmic drop in series with two parallel RC branches.

```
V(t) = OCV(t) − I·R₀ − I·R₁(1 − e^(−t/τ₁)) − I·R₂(1 − e^(−t/τ₂))     (during pulse)
```

with τ_k = R_k·C_k. R₀ is the instantaneous jump; the two RC branches add
overpotential on separated timescales. This is the standard second-order
equivalent-circuit model (ECM); see Hu, Li & Peng (2012) for a comparison of ECM
orders and Plett (2015) for the derivation and the diffusion→RC-ladder rationale.

The mapping of ECM elements to physical processes below follows the EIS→ECM
identification of Andre et al. (2011) and the impedance-vs-condition study of
Waag, Käbitz & Sauer (2013).

---

## 2. Physical interpretation

### R₀ — ohmic resistance (instantaneous, < ~1 s)
Electrolyte ionic resistance, separator, current collectors, contacts, and the
electronic resistance of the electrodes. Nearly independent of SOC in principle
(bulk conductivity), weakly temperature-dependent. In EIS it is the high-frequency
real-axis intercept (Andre et al. 2011, Part I; Barsoukov & Macdonald 2018).

### R₁, τ₁ — charge transfer + double layer (fast, ~1–5 s here)
The kinetic resistance of the electrochemical reaction at the electrode/electrolyte
interface (charge-transfer resistance R_ct) in parallel with the double-layer
capacitance C_dl, convolved with the surface/SEI film response. In EIS this is the
mid-frequency semicircle(s). R_ct is **strongly** SOC- and temperature-dependent and
is governed by Butler–Volmer kinetics, so it is also **current-dependent** — the
apparent resistance falls as pulse current rises (Bard & Faulkner 2001; Waag et al.
2013). τ₁ ≈ R_ct·C_dl.

### R₂, τ₂ — diffusion / mass transport (slow, ~20–500 s here)
Solid-state Li diffusion into the active-material particles plus electrolyte
concentration relaxation. In EIS this is the low-frequency Warburg branch. Physically
this is a *distributed* process (a continuum of timescales), which the single RC
branch only approximates — see §3.4. The very large fitted C₂ (10²–10⁶ F) is not a
real capacitance but a lumped stand-in for the diffusion charge reservoir.

### 2.4 Timescale separation and the DRT view
The three-element split (ohmic / kinetic / diffusion) is a coarse discretisation of
the cell's full relaxation-time spectrum. The Distribution of Relaxation Times (DRT)
formalism makes this explicit: real cells show a *continuum* of τ, and a 2RC model
lumps it into two peaks (Ivers-Tiffée & Weber 2017; Wan et al. 2015). Consequences:

- τ₁, τ₂ are **effective** time constants, not first-principles material properties.
  Don't convert τ₂ directly to a diffusion coefficient without a physics model.
- The R₀/R₁/R₂ assignment is only trustworthy when **τ₁ ≪ τ₂**. When they approach
  each other (aged, low-SOC pulses), the branches become unidentifiable — this is the
  root of the extraction instability in §3.

---

## 3. Extraction notes (protocol-specific)

### 3.1 R₀: R_DC,Δt vs pure-ohmic extrapolation — two different quantities
- **`R0_ohm` = R_DC,0.5 s** — voltage drop over a *fixed* 0.5 s window from onset
  (ΔU/ΔI), following the DC-pulse-resistance convention (Ludwig et al. 2021; the
  method traces to HPPC-style DCIR, e.g. the Idaho/PNGV pulse protocols, Belt 2010).
  Reproducible (a *read*, not a fit), but it is ohmic **plus** the fast-RC
  overpotential grown within 0.5 s, so it **overstates** pure ohmic by an
  SOC-/current-dependent amount.
- **`R0_extrap_onset` / `R0_staged`** — the true t=0 ohmic intercept from a coupled
  onset fit. Pure ohmic, but one parameter of a 5-parameter fit → it trades off
  against R₁ and is noisy ("rumbling" vs SOH), worst at high SOC.
- Empirically **R_DC,0.5 s ≈ R₀** (the fast branch is only ~10–20 % developed by
  0.5 s, contributing ~1–2 mΩ). So R_DC is essentially the near-instantaneous
  resistance; its SOC/aging variation lives in the ohmic+sub-second term, not R₁.

### 3.2 R₂ from the relaxation tail, and the `dev2` extrapolation
τ₂ and the slow amplitude A₂ are fit on the **full 30-min post-pulse pause**
(fast branch already decayed), so they are cleanly measured. The reported **R₂ is
the asymptotic value**, back-solved from the fraction the *pulse* actually excited:

```
A₂ = I·R₂·(1 − e^(−t_p/τ₂)) ;   R₂ = A₂ / (I·dev2),   dev2 = 1 − e^(−t_p/τ₂)
```

Because t_p ≈ 20 s ≪ τ₂, `dev2` is small (0.04–0.52 across our pulses), so R₂ is a
**1/dev2 extrapolation** and amplifies any τ₂ variation — e.g. the long-τ₂ DCH@10 %
pulses (τ₂≈470 s, dev2≈0.04) inflate a ~6 mΩ *realized* overpotential into a ~130 mΩ
"R₂". **The clean, low-noise observable is the realized slow overpotential A₂/I**;
the asymptotic R₂ is noisy by construction, not because of a bad tail fit.

> Design implication: a 20 s pulse is intrinsically weak at resolving the *magnitude*
> of a >100 s diffusion process. Long rests fix τ₂ but not R₂. Dedicated
> long-timescale probes (slow pulses, or EIS/DRT at low frequency) are the proper
> tool for the diffusion resistance.

### 3.3 Pulse test structure (data provenance)
Per SOC point: test pulse (`<BM>_<n>`, 20 s) → 30-min pause (`n+1`) → **restore
pulse** (`n+2`, C/2, returns SoC — *filtered out* in the export) → 30-min pause →
next test. The fit pairs each pulse with the single `n+1` pause **by ID**, so the
relaxation fit sees exactly one 30-min tail and stops before the restore pulse.

### 3.4 Identifiability / degeneracy
When τ₁→τ₂ the joint fit collapses one branch (railed τ₂, R₂→0); these rows are
flagged `degenerate` and excluded from trends. The high-SOC (@90 %) pulses are the
worst case (see §4.2) — the R₀↔R₁ split becomes ambiguous and R₁ absorbs part of R₀.

---

## 4. Empirical behaviour (VTC6, cell 007)

### 4.1 Aging trends (fresh → aged, non-degenerate)
| parameter | trend with aging | interpretation |
|---|---|---|
| **R₀ (ohmic)** | **↑ ~25–35 %**, clean & monotone | SEI/CEI film growth, electrolyte conductivity loss, contact resistance |
| **R₁ (charge transfer)** | **↑ ~20–25 %** (clean @10/50 % pulses) | interfacial degradation: SEI thickening, loss of active surface area |
| **R₂ (diffusion, asymptotic)** | ~+15 % at @50 % (least-amplified); noisy elsewhere | mild transport-resistance rise; obscured by 1/dev2 amplification at @10 % |
| **A₂/I (realized slow)** | mild ↑, low-noise | the honest diffusion-aging observable |
| **τ₁** | roughly stable / slightly ↓ | (@90 % "collapse" is artefact) |
| **τ₂** | roughly stable | diffusion timescale changes little over this SOH range |

R₀ growth as the dominant, cleanest aging signature is consistent with the resistance-
increase degradation mode catalogued by Vetter et al. (2005) and Birkl et al. (2017),
and with the impedance-aging trends of Waag et al. (2013) and Schmalstieg et al.
(2014) for NMC 18650 cells.

### 4.2 SOC dependence
R_DC,0.5 s (≈R₀) is **~1.5–1.7× higher at 90 % SOC** (≈40 mΩ) than mid-SOC (≈24 mΩ at
50 %), with 10 % intermediate — a U-shaped R(SOC), lowest mid-SOC, consistent with
Waag et al. (2013). Key observations:
- The 90 % elevation is **symmetric in charge/discharge** (CHA 40 ≈ DCH 38 mΩ) → an
  SOC-*state* property (kinetics near the fully lithiated/delithiated electrode), not
  a directional "charging-into-the-wall" effect.
- It is **not** an OCV-slope artefact: the in-pulse ΔOCV is *smallest* at 90 % (~2 mV,
  flat plateau) and largest at 10 % (~24–39 mV).
- The extra resistance sits in the **near-instantaneous** term (R_DC / extrapolated
  R₀), not in the fitted R₁/R₂ branches.

### 4.3 Near-V_max (CV-throttling) regime
At 90 % SOC the rested OCV (~4.09 V) leaves only ~0.11 V to V_max (4.20 V). A charge
pulse's sustainable current ≈ 0.11 V / R falls as R grows, so the nominal-3 A charge
throttles into constant-voltage: 3.0 A (fresh) → 2.9 A → 1.7 A (aged, V pinned at
4.20 V). The "CHA 3.0 A@90 %" bin therefore empties with age — itself an aging
readout (the SOH at which 3 A can no longer be sustained at 90 %). These CV-clamped
segments are **not valid pulses** (ΔU≈0, decaying I) and must be gated out, not
re-binned by measured current (which would also break the current-independence of a
fixed-condition R comparison — Butler–Volmer, §2.2).

---

## 5. Recommendations for aging tracking
1. **R₀ and R₁** are the reliable aging markers here (ohmic film growth + interfacial
   kinetics). Use the mid-SOC (@50 %) pulses — cleanest, least identifiability trouble.
2. For **diffusion**, track **A₂/I (realized) and τ₂**, not the extrapolated R₂.
3. **Exclude** `degenerate` rows and CV-clamped (near-V_max) segments; treat the @90 %
   R₀-vs-R₁ split as low-confidence.
4. To resolve ohmic-vs-fast-Rct at high SOC, or the diffusion magnitude, the pulse fit
   is insufficient — use **EIS + DRT** (Andre et al. 2011; Ivers-Tiffée & Weber 2017).

---

## 6. References
*(bibliographic details to be verified against the originals)*

- Andre, D., Meiler, M., Steiner, K., Wimmer, C., Soczka-Guth, T., Sauer, D.U. (2011).
  *Characterization of high-power lithium-ion batteries by electrochemical impedance
  spectroscopy. I. Experimental investigation; II. Modelling.* J. Power Sources 196.
- Barsoukov, E., Macdonald, J.R. (2018). *Impedance Spectroscopy: Theory, Experiment,
  and Applications*, 3rd ed. Wiley.
- Bard, A.J., Faulkner, L.R. (2001). *Electrochemical Methods: Fundamentals and
  Applications*, 2nd ed. Wiley. (Butler–Volmer, charge-transfer kinetics.)
- Belt, J.R. (2010). *Battery Test Manual for Plug-In Hybrid Electric Vehicles* (HPPC
  / DCIR pulse protocol). Idaho National Laboratory, INL/EXT.
- Birkl, C.R., Roberts, M.R., McTurk, E., Bruce, P.G., Howey, D.A. (2017).
  *Degradation diagnostics for lithium ion cells.* J. Power Sources 341, 373–386.
- Hu, X., Li, S., Peng, H. (2012). *A comparative study of equivalent circuit models
  for Li-ion batteries.* J. Power Sources 198, 359–367.
- Ivers-Tiffée, E., Weber, A. (2017). *Evaluation of electrochemical impedance spectra
  by the distribution of relaxation times.* J. Ceramic Society of Japan 125(4).
- Ludwig, S., et al. (2021). *Pulse-resistance (R_DC,Δt) based method* — as cited in
  the code header of `fit_2rc_pulse.py`; confirm the exact reference.
- Plett, G.L. (2015). *Battery Management Systems, Vol. 1: Battery Modeling.* Artech
  House. (Thevenin/RC ECM, diffusion RC ladder.)
- Schmalstieg, J., Käbitz, S., Ecker, M., Sauer, D.U. (2014). *A holistic aging model
  for Li(NiMnCo)O₂ based 18650 lithium-ion batteries.* J. Power Sources 257, 325–334.
- Vetter, J., et al. (2005). *Ageing mechanisms in lithium-ion batteries.* J. Power
  Sources 147, 269–281.
- Waag, W., Käbitz, S., Sauer, D.U. (2013). *Experimental investigation of the
  lithium-ion battery impedance characteristic at various conditions and aging states
  and its influence on the application.* Applied Energy 102, 885–897.
- Wan, T.H., Saccoccio, M., Chen, C., Ciucci, F. (2015). *Influence of the
  discretization methods on the DRT deconvolution.* Electrochimica Acta 184, 483–499.
