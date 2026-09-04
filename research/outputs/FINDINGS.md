# Harness Comparison — SROIE pilot

| variant | field exact | field F1 | doc full | calls/doc | ECE |
|---|---|---|---|---|---|
| A | 0.833 | 0.983 | 0.667 | 4.3 | nan |
| B | 0.833 | 0.983 | 0.667 | 4.3 | 0.277 |
| C | 0.833 | 0.983 | 0.667 | 4.3 | 0.277 |

## Error analysis (variant C)
- `date`: 1 docs wrong
- `address`: 1 docs wrong

## Self-correction effect
- docs where correction fixed a field: **0**
- docs where correction hurt a field: **0**
- flagged docs where correction helped: **0**
