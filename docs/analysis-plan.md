Primary target:
  prezzo marketplace giornaliero, non consiglio di investimento.

Primary horizons:
  h=28 e h=56 (dentro la griglia completa h in {1, 7, 28, 56, 64} -- vedi
  configs/benchmark_preregistered.example.json). h=28/56 sono i primary
  perche' coprono l'orizzonte "trading window" tipico per una carta
  collezionabile (mese/bimestre), non il one-step banale ne' il tail 64
  dominato dallo stitching multi-pass di TimesFM-3 oltre OUTPUT_PATCH_LENGTH.

Primary metric:
  skill = 1 - MAE_model / MAE_naive (relative MAE = MAE_model / MAE_naive;
  skill ne e' il complemento a 1 -- skill > 0 vuol dire il modello batte la
  naive, skill < 0 il contrario).

  Aggregazione fra carte: mediana (robusta a una carta outlier) E media
  pesata per numero di osservazioni per carta (n) -- mai una media ingenua
  di rapporti (una carta con 5 osservazioni peserebbe come una con 5000).
  Vedi tfm3lab.summarize.aggregate_leaderboard.

Primary comparison:
  TimesFM-3 univariato vs naive.

Secondary comparisons (leaderboard multi-baseline, non solo naive):
  vs seasonal_naive, drift, ets (quando converge) --
    tfm3lab.summarize.summarize_leaderboard;
  multivariato vs univariato;
  multivariato panel vero vs panel placebo (carte random dal pool piu'
    ampio, seed configurabile) -- tfm3lab.benchmark.select_placebo_panel;
  covariata lecita vs nessuna;
  raw vs log1p;
  make_positive True vs False;
  TimesFM-2.5 (zero-shot storico) vs TimesFM-3, quando l'adapter e'
    disponibile (tfm3lab.model_2p5) -- nessun run GPU reale eseguito in
    questo branch.

Ablation grid:
  context_lengths in {64, 128, 256, 512};
  horizons in {1, 7, 28, 56, 64};
  origin set condiviso -- calcolato UNA volta su
  (max(context_lengths)=512, max(horizons)=64), thinnato da origin_stride,
  poi riusato identico per ogni cella della griglia (vedi
  tfm3lab.benchmark.common_origin_set) -- requisito esplicito: "tutte le
  configurazioni devono essere confrontate sulle stesse origini".

Card pool:
  showcase (7 carte, DEFAULT_CARDS) per il talk;
  benchmark manifest piu' ampio (30+) OPZIONALE -- formato + criteri
  documentati in configs/benchmark_cards.example.csv, nessuna selezione
  reale di 30+ carte inventata in questo branch (richiederebbe un crawl
  TCGCSV live, fuori scope).

Exclusion rules:
  righe con observed=False escluse da ogni metrica (forward-fill, non
    un'osservazione reale -- gia' applicato in summarize.py, non toccato
    da questo lavoro);
  ETS escluso dal confronto quando non converge sul contesto dato
    (eccezione soppressa per QUELLA riga in backtest._baseline_forecasts,
    non l'intero run);
  combinazione multivariate_placebo esclusa quando il pool di carte
    disponibili e' piu' piccolo o uguale a placebo_panel_size (una
    partizione placebo richiede piu' carte del pool assegnato al panel
    reale, altrimenti campiona l'intero pool -- skip esplicito, contato
    nel dry-run report, mai un gap silenzioso);
  Diebold-Mariano non calcolato sotto MIN_OBSERVATIONS_FOR_DM_TEST
    osservazioni per gruppo (dm_stat/dm_pvalue = NaN, non un numero
    inventato).

Uncertainty:
  paired moving-block bootstrap;
  block length >= horizon;
  correzione BH per p-value multipli.

Claim rule:
  "migliora" solo se CI del delta e' coerente e preregistrata.
