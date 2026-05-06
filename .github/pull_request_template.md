## Summary
<!-- What changed and why? 1-3 bullets. -->
- 

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Config / parameters
- [ ] Docs / comments

## Checklist
- [ ] Ran `python main.py <config>` without errors
- [ ] GOLD output looks correct (spot-checked targets: CAP, PUL, qOCV, PAU)
- [ ] No unintended row drops or NaN leakage in SILVER/GOLD
- [ ] PAU stubs present in GOLD with correct `Duration_minutes`
- [ ] Restore pulses labelled `PUL*RES`, test pulses labelled `PUL`

## Branch naming
`feat/`, `fix/`, `refactor/`, `config/` — e.g. `feat/pulse-groupby`, `fix/dtype-coercion`
