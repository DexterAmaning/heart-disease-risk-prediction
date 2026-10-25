# Data

## `raw/heart_disease_uci45.zip`

| | |
|---|---|
| Dataset | Heart Disease (Cleveland subset used) |
| Creators | Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano |
| Source | UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/45/heart+disease |
| Direct file | https://archive.ics.uci.edu/static/public/45/heart+disease.zip |
| DOI | https://doi.org/10.24432/C52P4X |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0). Redistributed unchanged with attribution. |
| Downloaded | 2026-10-24 |
| SHA-256 | `b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc` |
| File used | `processed.cleveland.data`: 303 patients, 14 attributes; `?` marks missing values (`ca`: 4, `thal`: 2) |

**Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Coding follows `heart-disease.names` in the archive. The outcome `num` is the angiographic diagnosis (0 = < 50% diameter narrowing in all major vessels; 1–4 = > 50% narrowing). The analysis uses `num > 0`.

`heart-disease.names` states that patient names and social security numbers were removed from the database and replaced with dummy values. The processed file used here contains only the 14 clinical attributes.

### Why not the earlier `heart.csv`?

The project previously used a Kaggle `heart.csv`. Matching its 303 rows against the UCI file showed:
- its `target` column is **inverted**: `target = 1` corresponds to UCI `num = 0` (no disease) for all 165 such patients, and `target = 0` to disease for all 138;
- chest-pain type is renumbered (Kaggle 0 = asymptomatic, 3 = typical angina);
- missing `thal` values are coded as a category (0).

So the earlier model's output "the Person has heart disease" meant the opposite.
