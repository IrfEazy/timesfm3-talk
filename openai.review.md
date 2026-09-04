## Valutazione sintetica

La direzione è **molto buona**: il tuo progetto non è più “provo un modello nuovo su Magic”, ma può diventare un talk molto più forte:

> **“Zero-shot non è una garanzia di generalizzazione: come si red-teamma un foundation model per serie temporali?”**

È un *hot topic* perché TimesFM-3 è stato annunciato da Google Research il **31 agosto 2026** — quindi, al 3 settembre 2026, è letteralmente freschissimo. Google dichiara 330M parametri, pretraining su oltre 1 trilione di punti temporali, forecasting nativamente multivariato, covariate passate e note nel futuro, e output quantilici. ([research.google](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/))

### Correzione importante sul “paper”

Nelle fonti ufficiali che ho verificato, per TimesFM-3 vedo un **post di lancio**, il repository e il checkpoint/model card, ma non un paper V3 chiaramente indicato. Il “paper” collegato nel repository ufficiale è quello del TimesFM originale, ICML 2024. Quindi nelle slide, per ora, parlerei di:

- **technical release / model card / codice open source**
- non di “paper di TimesFM-3”, salvo trovi un preprint V3 ufficiale.

Inoltre: il codice è Apache-2.0, ma i pesi V3 sono sotto licenza **non-commerciale e non-production**. Per una demo/ricerca va bene come contesto, ma non va presentato come candidato immediato per produzione aziendale. ([github.com](https://github.com/google-research/timesfm?ref=aisecret.us))

---

# Il giudizio sul repository

Ho fatto una review statica della versione pubblica: README, scaletta talk, moduli core, ingestion MTG/mercati, backtest, metriche, grafici e script degli esperimenti principali. Non ho eseguito il checkpoint né riprodotto i numeri su GPU.

## Cose fatte molto bene

Hai già risolto parecchi problemi che rendono deboli la maggior parte dei talk sui modelli di forecasting:

1. **Hai una domanda scientifica, non una demo promozionale.**  
   La distinzione tra “il modello ha generalizzato?” e “il modello ha forse visto dati simili nel pretraining?” è eccellente.

2. **Hai una semantica rolling-origin centralizzata.**  
   `windows.py` definisce bene che `origin` è il primo indice predetto e non l’ultimo osservato; è il genere di dettaglio che evita errori off-by-one devastanti nei backtest. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/windows.py))

3. **Hai separato dati osservati e forward-filled.**  
   `SeriesData.observed` e il filtro nei riepiloghi sono una scelta molto corretta: non stai facendo finta che una forward-fill sia un’osservazione reale. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/backtest.py))

4. **Hai baseline sensate.**  
   Non stai usando il fuorviante “BeatNaive_%” come metrica primaria; hai MAE relativa, MASE, DM test e baseline naive/drift/ETS/stagionale. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/metrics.py))

5. **Hai una narrativa riproducibile.**  
   Risultati in parquet, demo offline, figure generate da codice e numeri non digitati manualmente nelle slide: ottimo. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/README.md))

6. **Hai già scritto le limitazioni esplicitamente.**  
   La sezione “Domande scomode” è probabilmente il materiale più convincente del talk. Non nascondere il fatto che una contaminazione del pretraining non è dimostrabile: è esattamente ciò che un pubblico tecnico apprezzerà.

---

# Le priorità da correggere prima di usare i risultati in slide

## P0 — bug metodologico nel grafico PIT / quantili

Questo è il punto più urgente.

In `pit_values`, i valori sotto `q10` e sopra `q90` vengono troncati rispettivamente a `0.1` e `0.9`. Poi `pit_histogram()` usa come bin proprio `[0.1, 0.2, ..., 0.9]`, ma etichetta il primo bin come `≤ q10` e l’ultimo come `≥ q90`.

Non è corretto:

- il primo bin contiene sia i casi sotto `q10`, **sia** quelli tra `q10` e `q20`;
- l’ultimo contiene sia i casi sopra `q90`, **sia** quelli tra `q80` e `q90`.

Quindi l’interpretazione “massa nelle code” del grafico C4 va invalidata e rigenerata. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/metrics.py))

### Fix corretto

Non chiamarlo PIT histogram. Chiamalo ad esempio:

> **Discrete quantile-bin calibration**

Costruisci 10 bin espliciti:

1. `actual <= q10`
2. `q10 < actual <= q20`
3. …
9. `q80 < actual <= q90`
10. `actual > q90`

Se i quantili sono calibrati, ogni bin dovrebbe avere circa il **10%** della massa.

---

## P0 — la demo può mostrare dati imputati come se fossero reali

Le metriche aggregate filtrano correttamente `observed == True`. Però `build_forecast_slice()` usa tutte le righe della previsione nel grafico, senza rimuovere o segnalare i target forward-filled; inoltre `rank_windows()` usa quei punti per scegliere finestre “eroe”. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/summarize.py))

### Fix

- Per selezionare finestre demo: usa solo finestre con `observed=True` per tutti i target, oppure mostra la quota osservata.
- Nel grafico:
  - reale osservato: nero;
  - valore forward-filled: grigio chiaro o marker vuoto;
  - mai etichettare un imputato come “reale”.
- Aggiungi una sensitivity analysis:
  - `max_ffill_days = 0`
  - `max_ffill_days = 1`
  - `max_ffill_days = 3`

Se la headline cambia, il risultato è fragile; se non cambia, hai una bella slide in appendice.

---

## P0 — associazione output/input: non affidarti all’ordine

`forecast_batch()` conserva i `ts_id` restituiti dal modello, ma `run_univariate_backtest()` e `run_multivariate_backtest()` associano output e input per posizione nell’array. Se un’implementazione futura del valutatore riordina l’output, potresti assegnare una previsione alla carta sbagliata. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/model.py))

### Fix

- Costruisci una mappa `ts_id -> metadata input`.
- Verifica:
  - nessun `ts_id` duplicato;
  - stesso insieme input/output;
  - ordine non assunto.
- Aggiungi un test con un fake forecaster che restituisce le serie in ordine invertito.

### Test extra fondamentale

Nel tuo univariato passi ID del tipo:

```python
f"{s.name}::{origin}"
```

Quindi fai un test di invarianza:

- stessi tensori;
- `ts_ids` originali;
- `ts_ids` anonimi, es. `series_0001`;
- `ts_ids` randomizzati.

Se la previsione cambia, i nomi delle carte entrano in qualche modo nel comportamento dell’API e diventano una possibile via di contaminazione o confondimento. Se non cambia, puoi dichiarare con più sicurezza che il test usa solo valori temporali e covariate.

---

## P0 — contratto dati MTG troppo implicito

La tua ingestion è già seria, ma ci sono tre rischi concreti:

1. `fetch_daily_prices()` salva un solo prezzo per `productId`; se l’archive contiene più righe per lo stesso prodotto con `subTypeName` differente, l’ultima riga vince silenziosamente. Il tuo schema stesso documenta `subTypeName`, ma non lo conserva nella chiave. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/data/mtg.py))

2. Tutti gli `HTTPError` vengono trattati come “giorno mancante”. Un 404 può essere una lacuna legittima; un 429 o un 500 invece può diventare un falso missing value e poi una forward-fill. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/data/mtg.py))

3. `end=None` usa la data corrente, quindi una rerun non è veramente riproducibile. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/data/mtg.py))

### Fix

Per ogni osservazione salva almeno:

```text
series_id
group_id
product_id
printing / finish / subtype
price_field_used          # marketPrice o midPrice
observed
source_archive_date
archive_sha256
fetched_at_utc
```

E per gli esperimenti imponi:

```text
--as-of YYYY-MM-DD
```

Mai “today” come default di un esperimento che finirà in slide.

---

## P1 — il test “long horizon” oggi è ancora corto per raccontare la tesi

Il default dell’esperimento MTG è:

- context: 64 giorni;
- horizon massimo: 28 giorni.

Dato che la patch temporale è 32, un contesto di 64 equivale sostanzialmente a due patch temporali. È un test utile, ma non sfrutta davvero il potenziale di contesto lungo che vuoi raccontare. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/config.py))

### Proposta

Usa come griglia primaria:

| Dimensione | Valori |
|---|---:|
| Context length | 64, 128, 256, 512 |
| Horizon | 1, 7, 28, 56, 64 |
| Trasformazione | raw, log1p |
| Modalità | univariato, multivariato |
| Positività | `make_positive=True/False` |
| Baseline | naive, seasonal naive, drift, ETS |

Scegli **prima** di guardare il risultato:

- headline business: `h=28`;
- headline “longer horizon”: `h=56`;
- `h=1` solo come diagnostica e sanity check;
- `h=64` per mostrare il confine dell’output patch;
- oltre 64, separa chiaramente il caso “stitching / multiple calls”.

Nota: nel codice discuti `make_positive`, ma l’esperimento MTG non fa davvero l’ablation `True/False`; oggi confronta solo raw e log1p. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/src/tfm3lab/model.py))

---

## P1 — sette carte vanno bene per la demo, non per una conclusione generale

Le sette carte sono ottime come:

- casi visuali;
- confronto fra liquidità/volatilità;
- storytelling.

Ma non bastano per dire:

> “TimesFM-3 funziona/non funziona sul mercato Magic.”

Hai due opzioni oneste:

### Opzione A — deadline stretta

Dichiara apertamente:

> “È un case study su sette serie intenzionalmente selezionate, non una stima della performance sull’intero mercato MTG.”

### Opzione B — benchmark serio

Crea:

- **7 carte showcase**, quelle attuali;
- **30–100 carte benchmark**, selezionate con regola congelata prima del run.

La selezione deve essere stratificata almeno per:

- fascia di prezzo;
- volatilità;
- copertura/missingness;
- set;
- ristampa/non ristampa;
- liquidità, se riesci a ottenerla.

---

## P1 — non dire “feature invisibili”: rendilo falsificabile

La frase:

> “Voglio vedere se estrae feature invisibili ad occhio nudo”

è intuitiva, ma non è misurabile direttamente.

Un forecast migliore non dimostra quali feature il transformer abbia scoperto; e l’attention non è automaticamente una spiegazione.

Trasformala in quattro ipotesi verificabili:

1. **H1 — Skill incrementale**  
   TimesFM-3 riduce l’errore rispetto alla persistence/naive su orizzonti 28 e 56?

2. **H2 — Valore del multivariato**  
   TimesFM-3 multivariato migliora rispetto allo stesso modello univariato sulle stesse origini?

3. **H3 — Valore di una covariata lecita**  
   Una covariata nota al tempo dell’origine migliora rispetto a:
   - nessuna covariata;
   - covariata temporalmente shuffleata;
   - covariata volutamente leaked?

4. **H4 — Incertezza utile**  
   I quantili hanno copertura corretta e ampiezza utile, soprattutto a `h=28` e `h=56`?

Se H2 non migliora, non è un fallimento: significa che **mettere sette carte arbitrarie nello stesso tensore non crea automaticamente informazione predittiva**.

---

## P1 — lo shock pre/post-cutoff è interessante, ma non deve essere il cuore del talk

La tua sezione shock è intelligente e onesta, ma come evidenza causale è fragile:

- solo 5 eventi;
- gruppi `n=3` vs `n=2`;
- eventi molto eterogenei;
- il detector non seleziona davvero gli eventi: verifica se eventi scelti a priori corrispondono a outlier;
- l’offset è in indici della serie intersecata, quindi parla di **sedute/osservazioni**, non di “giorni” di calendario;
- se una lag non rientra nella finestra, `None -> NaN`, e la media Pandas può ignorare proprio gli eventi peggiori. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/scripts/03_exp_shock.py))

Inoltre, il tuo “market_calm” non è davvero un campione di mercato calmo: è il resto delle finestre attorno agli eventi, cioè spesso post-shock. ([raw.githubusercontent.com](https://raw.githubusercontent.com/IrfEazy/timesfm3-talk/main/scripts/04_exp_calibration.py))

### Mia raccomandazione

Metti Experiment B in **appendice / red-team section**:

> “Un tentativo esplorativo di capire se la data di cutoff possa lasciare tracce nel comportamento del modello. Non una prova di contaminazione.”

Il protagonista deve restare Magic, altrimenti il talk diventa:

1. Magic;
2. S&P500;
3. VIX;
4. CPI;
5. covariate;
6. licenza.

Troppa narrativa per 20 minuti.

---

# La struttura di talk che ti consiglierei

## Titolo

> **TimesFM-3 su Magic: un foundation model da 330M parametri batte davvero `prezzo_domani = prezzo_oggi`?**

Oppure, più provocatorio:

> **Zero-shot, ma non zero-verifica: red-teaming di TimesFM-3 sui prezzi Magic**

## Scaletta da 20 minuti

### 1. Hook — 2 min

Mostra una carta, un forecast a 28 giorni e due linee non etichettate.

> “Una linea viene da un transformer da 330 milioni di parametri.  
> L’altra da una riga di Python: `repeat(last_price)`.”

Fai votare mentalmente il pubblico.

Poi riveli la naive.

### 2. Che cosa promette V3 — 3 min

Una sola slide tecnica:

- patch temporali;
- attention nel tempo;
- attention fra variate;
- covariate note nel futuro;
- quantili.

Non fare 20 layer, RoPE e RevIN nel corpo del talk: mettili in appendice.

### 3. Protocollo — 3 min

Mostra una timeline:

```text
Nov 2023  | cutoff conservativo
Feb 2024  | inizio archive MTG
...       | rolling-origin forecast
h=28/h=56 | evaluation
```

Messaggio:

> “Post-cutoff non prova assenza totale di contaminazione.  
> Ma è un holdout temporale molto più onesto di un benchmark pubblico storico.”

### 4. Risultato primario — 4 min

Mostra:

- skill vs naive per `h=7, 28, 56, 64`;
- intervalli bootstrap;
- dot plot per carta;
- mediana e distribuzione, non solo una carta “eroe”.

Se TimesFM perde alla naive, non nasconderlo:

> “Il risultato più interessante non è che il modello è magico.  
> È che un modello enorme può perdere contro una baseline banale quando il target è un prezzo rumoroso.”

### 5. Cosa aggiunge davvero V3? — 3 min

Ablation:

```text
univariato
multivariato con carte correlate
multivariato con carte casuali/placebo
```

Poi:

```text
senza covariata
covariata lecita
covariata shuffleata
covariata leaked
```

Questa è la slide che dimostra il valore reale dell’architettura.

### 6. I quantili sono onesti? — 3 min

Mostra:

- copertura P10–P90 per horizon;
- discrete quantile-bin calibration corretta;
- ampiezza media dell’intervallo;
- eventualmente raw vs conformalized intervals.

Messaggio:

> “Un intervallo largo può essere calibrato ma inutile.  
> Un intervallo stretto può essere utile ma pericolosamente under-confident.”

### 7. Leakage demo — 1.5 min

La demo volutamente “sporca” è eccellente.

> “Ecco cosa succede se passo accidentalmente il futuro come covariata.”

Mostra metriche implausibilmente belle in rosso.

### 8. Chiusura — 0.5 min

> “Zero-shot è una proprietà dell’addestramento.  
> Generalizzazione è una proprietà che devi ancora dimostrare.”

---

# Piano di coding consigliato

| Fase | Obiettivo | Output da portare nelle slide |
|---|---|---|
| 0 | Freeze della versione attuale | tag Git, lockfile, manifest |
| 1 | Correzioni P0 | test, schema risultati nuovo, PIT corretto |
| 2 | Data card MTG | qualità dati, missingness, fallback market/mid |
| 3 | Pilot GPU | 7 carte, `h=1/7/28/56/64`, context 64/256 |
| 4 | Pre-registrazione | metriche, horizon primario, exclusion rules |
| 5 | Full run | benchmark, CI, ablation multivariata/covariate |
| 6 | Figure | 5–6 grafici, tutti generati da artifact |
| 7 | Talk offline | notebook demo senza rete né checkpoint |

## Regole di decisione da definire prima del full run

Scrivile in `docs/analysis-plan.md`:

```text
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
```

---

# Prompt per Claude Code

Usali **uno alla volta**, non tutti insieme. Dopo ogni prompt: fai review del diff, esegui test, poi fai commit.

## Prefisso comune da aggiungere a ogni prompt

```text
Sei nella root del repository tfm3lab/timesfm3-talk.

Tratta questo repository come un progetto di ricerca riproducibile, non come
una demo di prodotto.

Vincoli obbligatori:
- usa esclusivamente uv: mai pip o uv pip;
- non modificare manualmente risultati parquet o numeri nelle slide;
- non inventare risultati, metriche, fonti o dati;
- test offline per default; chiamate live, download HF e GPU devono restare opt-in;
- non introdurre leakage: ogni feature futura deve avere una disponibilità storica
  esplicita al tempo dell'origine;
- tutte le metriche devono rispettare observed=True;
- preserva la convenzione di windows.py: origin è il primo indice predetto;
- aggiungi test unitari per ogni bug corretto;
- prima di implementare, leggi README.md, docs/talk-outline.md, pyproject.toml,
  src/tfm3lab e i test rilevanti;
- alla fine scrivi un breve report: file cambiati, test eseguiti, rischi residui,
  migrazioni necessarie per i vecchi risultati.
```

---

## Prompt 1 — Correzioni metodologiche P0

```text
Esegui un audit e una correzione P0 della pipeline di forecasting e figure.

Obiettivi obbligatori:

1. Correggi il diagnostic dei quantili in figdata.py:
   - l'attuale pit_histogram usa bin [0.1, ..., 0.9] ma etichetta impropriamente
     i bin estremi come code <=q10 e >=q90;
   - sostituiscilo con una funzione esplicita di discrete quantile-bin calibration;
   - crea 10 bin: <=q10, (q10,q20], ..., (q80,q90], >q90;
   - ogni bin deve avere probabilità nominale 0.1;
   - rinomina funzioni, figure e testi per non chiamarlo PIT se non è PIT continuo;
   - mantieni una migrazione/backward compatibility ragionevole se serve.

2. Impedisci che figure e selezione di finestre trattino valori forward-filled
   come osservazioni reali:
   - rank_windows deve usare solo target observed;
   - build_forecast_slice deve esporre observed_mask;
   - il plotting deve distinguere graficamente osservato e imputato;
   - definisci policy esplicita per finestre con target non osservati.

3. Correggi l'associazione input/output:
   - forecast_batch deve verificare unicità e completezza dei ts_id;
   - i backtest devono associare outputs ai metadati tramite ts_id, non per posizione;
   - aggiungi test con fake forecaster che restituisce output in ordine invertito.

4. Aggiungi un test opt-in di invariance ai ts_id:
   - stessi context tensor, ts_id originali, anonimi e randomizzati;
   - se il modello reale non è disponibile, crea la struttura del test e marca il
     test come opt-in;
   - documenta perché questo test serve a escludere metadata leakage.

5. Aggiungi validazioni a SeriesData:
   - date ordinate e uniche;
   - shape coerenti;
   - input al modello finiti;
   - messaggi di errore chiari per NaN residui.

Non eseguire fetch live né il checkpoint TimesFM reale.
Aggiorna documentazione e test. Alla fine esegui ruff e pytest.
```

---

## Prompt 2 — Data contract e riproducibilità dei dati MTG

```text
Implementa un data contract robusto per l'ingestion MTG e un manifest
riproducibile per ogni run.

Requisiti:

1. Introduci una policy --as-of YYYY-MM-DD:
   - gli esperimenti completi non devono usare implicitamente date.today();
   - la data finale deve essere registrata nel manifest;
   - aggiungi una modalità esplicita --allow-live-end solo per sviluppo.

2. Migliora la gestione HTTP:
   - 404 può diventare "archive non disponibile";
   - 429 e 5xx devono fare retry con backoff e poi fallire in modo esplicito;
   - download atomico: file .part, hash, rename finale;
   - nessun errore transitorio deve trasformarsi silenziosamente in missingness.

3. Risolvi l'ambiguità productId/subTypeName:
   - ispeziona l'attuale schema;
   - se un productId ha più record nello stesso giorno, non scegliere l'ultimo
     silenziosamente;
   - rendi esplicita una price-selection policy configurabile;
   - conserva price_field_used, subtype/finish quando disponibile;
   - fallisci con messaggio utile se la policy è ambigua.

4. Crea un manifest JSON o parquet per fetch e run:
   - git SHA;
   - data range;
   - as_of;
   - hash degli archive usati;
   - card specs risolti;
   - percentuale marketPrice vs midPrice;
   - missingness e forward-fill;
   - versione package/checkpoint se disponibile;
   - hardware e flags di inference per i run modello.

5. Aggiungi una data-quality table e una figura riusabile:
   - observed rate per carta;
   - fallback rate;
   - gap massimi;
   - count di glitch;
   - range e volatilità base.

Usa test con fixture locali; non richiedere rete per pytest standard.
```

---

## Prompt 3 — Benchmark MTG primario, lungo orizzonte e ablation pulite

```text
Rendi l'esperimento MTG configurabile e adatto a una valutazione primaria
preregistrata.

Non eseguire il full run GPU: implementa CLI, dry-run, test e documentazione.

Requisiti:

1. Crea una configurazione dichiarativa per benchmark:
   - context lengths: 64, 128, 256, 512;
   - horizons: 1, 7, 28, 56, 64;
   - origin stride configurabile;
   - origin set comune calcolato rispetto a max context e max horizon;
   - tutte le configurazioni devono essere confrontate sulle stesse origini.

2. Separa:
   - 7 carte showcase;
   - benchmark card manifest più ampio, opzionale e riproducibile.
   Se non è possibile selezionare automaticamente 30+ carte senza assunzioni,
   implementa il formato manifest e documenta i criteri richiesti, senza inventare
   una selezione.

3. Esegui e salva esplicitamente tutte le baseline già presenti:
   - naive;
   - seasonal naive;
   - drift;
   - ETS quando converge.
   Non usare solo naive nel summary finale: produci una leaderboard per metodo.

4. Aggiungi l'ablation:
   - raw vs log1p;
   - make_positive True vs False;
   - univariato vs multivariato;
   - multivariato con panel vero;
   - panel placebo con carte random/non correlate, se configurabile.

5. Salva risultati con colonne:
   run_id, config_id, context_len, requested_horizon, mode, transform,
   make_positive, series, origin, target_date, observed.

6. Introduci metriche leggibili per slide:
   - relative MAE;
   - skill = 1 - relative_MAE;
   - mediana e media pesata;
   - metriche per carta e aggregate.
   Non fare medie ingenue di rapporti senza documentarne la pesatura.

7. Se possibile, prepara un adapter separato per TimesFM-2.5 come baseline
   zero-shot storica; non forzare dipendenze incompatibili. Se non è fattibile,
   documenta chiaramente il blocco e lascia il design pronto.

Aggiorna docs/analysis-plan.md con ipotesi, metriche primarie, exclusion rules
e definizione degli horizon primari.
```

---

## Prompt 4 — Inferenza statistica, calibrazione e intervalli conformal

```text
Rafforza l'inferenza statistica e la valutazione probabilistica.

Requisiti:

1. Implementa un paired moving-block bootstrap per il delta di errore
   TimesFM vs baseline:
   - blocchi almeno pari all'horizon;
   - pairing preservato per origine;
   - output CI e seed;
   - supporto panel per serie.

2. Mantieni Diebold-Mariano ma:
   - applica correzione Benjamini-Hochberg ai p-value delle molteplici
     combinazioni carta x horizon;
   - non usare p-value come headline principale;
   - rendi disponibili effect size e CI.

3. Migliora calibrazione:
   - curva nominale vs empirica con CI binomiali/bootstrap;
   - P10-P90 coverage;
   - ampiezza dell'intervallo normalizzata;
   - weighted interval score o una variante documentata;
   - discrete quantile-bin calibration corretta.

4. Implementa opzionalmente una post-elaborazione conformal causalmente valida:
   - usa solo errori di origini precedenti;
   - nessun target futuro o stesso-origin leakage;
   - separa esplicitamente risultati raw TimesFM e conformalized TimesFM;
   - valuta coverage e ampiezza;
   - documenta che il secondo non è più "zero-shot puro", ma un wrapper online.

5. Genera artifact parquet e figure da usare nelle slide, tutti con run_id
   e manifest reference.

Aggiungi test sintetici, inclusi casi di quantili perfetti, quantili undercovered
e sequenze autocorrelate.
```

---

## Prompt 5 — Covariate lecite e leakage demo rigorosa

```text
Riprogetta l'esperimento covariate con un esplicito availability contract.

Obiettivo: distinguere una covariata realmente nota al tempo dell'origine da una
covariata "oracle" o leaked.

Requisiti:

1. Definisci una struttura dati per covariate con:
   - feature_name;
   - value_time;
   - available_at;
   - source/provenance;
   - future_known boolean;
   - policy di imputazione.

2. Prima di passare una covariata past_future al modello, verifica per ogni
   origine che ogni valore futuro usato fosse disponibile entro origin_date.
   In caso contrario, fallisci.

3. Per MTG:
   - non trattare publishedOn da solo come prova che una release date fosse nota;
   - implementa due modalità:
     a) verified historical availability, se l'utente fornisce announcement dates;
     b) oracle calendar, etichettata esplicitamente come scenario ideale/non deployable.

4. Implementa quattro bracci ablation:
   - no covariate;
   - covariata lecita;
   - covariata temporalmente shuffleata/placebo;
   - future target leaked, esclusivamente come negative control.

5. La leakage demo deve essere impossibile da confondere con un risultato reale:
   - artifact e figure separati;
   - watermark "LEAKED / INVALID";
   - test che impedisca di usare quel braccio nei summary ufficiali.

6. Aggiungi sensitivity test per covariate sin/cos e make_positive:
   verifica che una feature che può essere negativa non venga accidentalmente
   distorta dalla pipeline.

Non inventare fonti storiche per release announcement: crea l'interfaccia,
la documentazione e un esempio fixture locale.
```

---

## Prompt 6 — Trasforma lo shock study in una red-team appendix

```text
Rivedi Experiment B come analisi esplorativa, non come prova di contaminazione.

Requisiti:

1. Aggiorna nomi e documentazione:
   - "adaptation lag" -> "error recovery lag";
   - "days" -> "trading sessions" o "observations", se l'offset deriva da indici;
   - "pre/post cutoff difference" -> evidenza descrittiva/esplorativa.

2. Correggi right censoring:
   - se una serie non rientra sotto soglia entro la finestra, non convertirla in NaN
     ignorata dalla media;
   - registra censored=True e la finestra osservata;
   - mostra punti individuali, non solo medie per braccio.

3. Confronta modello contro naive:
   - usa excess error o delta error rispetto alla naive;
   - non interpretare il rientro della volatilità di mercato come adattamento del modello.

4. Rinomina market_calm:
   - le righe lontane <=3 giorni ma interne alle finestre evento non sono un campione
     di mercato genericamente calmo;
   - usa "event_window_nonshock" oppure costruisci controlli matched fuori dagli eventi.

5. Mantieni questo esperimento in appendice e genera una slide che dichiari:
   n piccolo, eventi eterogenei, cutoff non certificato, nessuna causal claim.

Aggiungi test per censura, offset e aggregazioni.
```

---

## Prompt 7 — Benchmark reale di latenza e slide evidence ledger

```text
Implementa due cose: microbenchmark di inferenza e evidence ledger per la presentazione.

A. Microbenchmark:
- warm-up;
- ripetizioni;
- CPU e GPU quando disponibili;
- torch.cuda.synchronize prima/dopo il timing GPU;
- horizons 1, 7, 28, 56, 64, 65, 90;
- symmetric averaging True/False;
- batch sizes configurabili;
- mediana, p95, throughput e memoria se disponibile;
- non mescolare timing del download/caricamento modello con timing inferenza.

B. Evidence ledger:
- crea docs/evidence-ledger.md o un artifact generato;
- ogni claim in slides/talk.md deve puntare a:
  source type, run_id, parquet, figura, metrica, caveat;
- nessun numero può essere hardcoded;
- le figure devono avere titolo, run_id, intervallo dati e nota metodologica;
- aggiorna la scaletta: Magic è il corpo del talk, shock study è appendice;
- aggiungi speaker notes per:
  "cosa possiamo concludere" e "cosa non possiamo concludere".

La demo notebook deve restare offline e non deve richiedere rete, GPU o token HF.
```

---

## Prompt 8 — Review ostile finale

```text
Agisci come un reviewer ostile di un paper/technical talk.

Leggi l'intero repository e il diff corrente. Cerca in particolare:
- leakage temporale;
- leakage attraverso ts_id;
- leakage nei dati forward-filled;
- mismatch origin/target;
- contaminazione da risultati vecchi;
- metriche aggregate fuorvianti;
- multiple comparisons;
- claim di causalità non supportati;
- misuse di PIT/quantili;
- figure selezionate dopo aver visto il risultato;
- risultati non riproducibili;
- licenze incompatibili con le claim di utilizzo.

Non implementare subito grandi refactor:
1. scrivi prima docs/reviewer-report.md;
2. classifica ogni finding come blocker / high / medium / low;
3. cita file e riga;
4. proponi fix minimi;
5. aggiungi test solo per finding ad alta confidenza;
6. esegui ruff e pytest;
7. riporta esattamente cosa non hai potuto verificare senza rete/GPU/checkpoint.
```

---

# La frase chiave da portare a casa

Se dopo i fix il modello continua a perdere contro la naive, **hai comunque un talk eccellente**.

Il messaggio non sarà:

> “Google ha fatto un modello che predice Magic.”

Sarà:

> “Ho preso un modello appena uscito, ho costruito un test che poteva smentirlo, l’ho confrontato con una baseline onesta, ho verificato leakage e calibrazione, e vi mostro dove funziona, dove fallisce e perché non bisogna fidarsi dei claim del vendor senza un protocollo.”

Per un pubblico tecnico, questo è molto più interessante di una curva blu che sembra bella.

# 0. confirm TCGCSV schema still matches
uv run scripts/00_probe_tcgcsv.py

# 1. fetch fresh data, gets real manifest provenance (missing on current results/)
uv run scripts/01_fetch_data.py --as-of 2026-09-04

# 2-5. experiments (need real GPU checkpoint — run on Colab per README, or local if GPU present)
uv run scripts/02_exp_mtg.py
uv run scripts/03_exp_shock.py
uv run scripts/04_exp_calibration.py   # pure re-analysis, no GPU, reuses 02+03 cache
uv run scripts/05_exp_covariates.py

# 6-7. figures + slide numbers, local, no GPU
uv run scripts/06_make_figures.py
uv run scripts/07_build_slides.py
