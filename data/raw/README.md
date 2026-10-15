Place the cleaned, UTF-8 event table here as `new_data.csv`. The event data are not included in the upload package. No redistribution permission is inferred from local access.

The manuscript describes 1,046 verified lightning-fire events from 2005-05-23 to 2024-08-20, including 576 in Inner Mongolia and 470 in Heilongjiang. Obtain the cleaned table through the manuscript's authors/data-access procedure. Keep its row order unchanged when comparing historical assignments.

Required columns:

| Group | Exact column names |
| --- | --- |
| Temperature | Mean Temperature, Maximum Temperature, Minimum Temperature, Mean Temperature over the Past 10 Days, Mean Temperature over the Past 30 Days |
| Precipitation | Precipitation, Observed Precipitation over the Past 24 Hours, Observed Precipitation over the Past 3 Days, Observed Precipitation over the Past 10 Days, Observed Precipitation over the Past 30 Days |
| Splits | time (ISO date), province (Inner Mongolia or Heilongjiang) |
| Interpretation/descriptive analysis | drought_index |

The loader automatically converts original Chinese headers and province values to the English schema above. It also accepts the legacy misspelling `provience` as an alias for `province`, and the full province names `Inner Mongolia Autonomous Region` and `Heilongjiang Province`. Supplying both an alias and its English column is rejected as ambiguous. Source data files are never rewritten.

Temperature is supplied in the units used in the cleaned manuscript table and precipitation in its original nonnegative measurement units. Drought index is an explanatory variable, excluded from functional clustering. Missing or non-finite values are rejected. A sequential R export row-number column is removed only if its name and values match the export convention.

Station matching, manual geocoding corrections, interpolation and drought-index construction are upstream of this cleaned-table repository. These operations cannot be reproduced from the supplied analysis scripts alone. Their station inputs, coefficients and historical processing environment have not been supplied.
