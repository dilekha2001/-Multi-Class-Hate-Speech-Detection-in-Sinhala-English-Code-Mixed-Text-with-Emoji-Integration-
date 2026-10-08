# Data card - YOUTUBE_FACEBOOK_DATA.xlsx

**Languages:** romanised Sinhala mixed with English (a few Sinhala-script comments remain).
**Platforms:** Facebook (comments copied manually from public pages) and YouTube (collected with the YouTube Data API v3 via a collection script).
**Labels** (original column `Lable`): 0 = neutral, 1 = offensive, 2 = hate (Milestone 2 taxonomy, after Davidson et al.).

## Cleaning applied (`cleaning_report.json`)
| Step | Rows |
|---|---|
| Raw rows in xlsx | 2,419 |
| Missing comment or label | -3 |
| Empty after URL/@mention removal and normalisation | -15 |
| Same text with conflicting labels (44 texts) | -87 |
| Exact duplicate texts (case/punctuation-insensitive) | -105 |
| **Cleaned dataset (`annotated_final.csv`)** | **2,210** |
| After minimum length filter (>= 3 tokens) in BOTH variants | **1,947** |

URLs and @mentions were removed. Check manually that no personal names remain in Facebook comments before publishing any data.

## Final modelling set (`processed/`) - 1,947 comments
| Platform | hate | neutral | offensive | total |
|---|---|---|---|---|
| Facebook | 379 | 527 | 349 | 1,255 |
| YouTube | 159 | 319 | 214 | 692 |
| **All** | **538** | **846** | **563** | **1,947** |

About 36% of raw comments contain emoji. Folds: stratified, grouped by comment text, ~390 test comments each, 0 duplicate leakage.

## Items to state in the paper (fill in accurately)
- Who annotated the comments, how many annotators, the guideline used, and agreement (kappa) - not recorded in the file.
- How the Facebook comments were collected (manual copy, dates, which pages), instead of the Facepager method planned in Milestone 2.
- Any use of AI tools in labelling, drafting or coding must be disclosed and the integrity declaration must match it.
- Dataset size (1,947 after filtering) is below the 2,500 target in Milestone 2.
