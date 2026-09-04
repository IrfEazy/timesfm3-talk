Primary target:
  prezzo marketplace giornaliero, non consiglio di investimento.

Primary horizons:
  h=28 e h=56.

Primary metric:
  skill = 1 - MAE_model / MAE_naive.

Primary comparison:
  TimesFM-3 univariato vs naive.

Secondary comparisons:
  multivariato vs univariato;
  covariata lecita vs nessuna;
  raw vs log1p;
  make_positive True vs False.

Uncertainty:
  paired moving-block bootstrap;
  block length >= horizon;
  correzione BH per p-value multipli.

Claim rule:
  "migliora" solo se CI del delta è coerente e preregistrata.
