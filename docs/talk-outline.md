# TimesFM-3: oracolo o aggiornatore probabilistico? — scaletta talk (20-25 min)

Pubblico: tecnico. Formato: slide introduttive + demo pratica dal vivo. Obiettivo: far
venire voglia ai colleghi di andare ad approfondire.

Ogni numero citato qui sotto deve essere rintracciabile in `results/*.parquet` — vedi
README.md per come rigenerarli. Finché gli esperimenti non sono stati eseguiti per davvero
(serve GPU + licenza HF accettata, vedi README), questa scaletta usa `[NUMERO]` come
segnaposto: non inventare un numero plausibile, lascialo vuoto finché non è calcolato.

---

## 1. Hook (2')

> "Esiste un modello fondazionale per le serie temporali: zero training, zero fine-tuning,
> gli dai una serie e lui prevede. Prima di crederci gli ho fatto lo sgambetto: gli ho dato
> dati che quasi certamente non ha mai visto, e dati su cui poteva aver barato."

Non aprire con "ecco un nuovo modello di Google" — quello è esattamente il tipo di talk che
non si vuole fare. Aprire con la domanda di fiducia: *come faccio a sapere se questo modello
generalizza o sta solo ricordando?*

## 2. TimesFM-3 in 5 minuti

Fatti verificati (non dal blog di lancio — dal sorgente installato e dalla model card):

- 330M parametri, patch da 32 step, **context massimo 15.360** (non "16k" arrotondato).
- **Multivariato nativo**: attention alternata — causale nel tempo (dentro una serie) e
  "full variate attention" fra serie diverse nello stesso passo. È la vera novità della v3.
- **Decoding non autoregressivo**: un intero orizzonte di 64 step esce da un solo forward
  pass. Conseguenza pratica: chiedere una previsione a 7 giorni o a 28 giorni costa
  *esattamente* lo stesso calcolo — un fatto sorprendente da mostrare a un pubblico tecnico.
- Output: 9 quantili (0.1→0.9). Il "punto" previsto **è definito come il quantile 0.5**, non
  una media stimata separatamente — verificato leggendo `timesfm3_forecaster.py`, non assunto.
- `make_positive=True` (default del valutatore ufficiale) clippa a zero le serie il cui
  contesto era già non-negativo — nessun trucco manuale serve per evitare prezzi negativi.
- `use_symmetric_averaging=True` (default) raddoppia il calcolo: ogni contesto viene passato
  al modello sia normale sia negato, poi i risultati mediati. Costo reale, mostralo.

## 3. Perché non mi fido dei benchmark pubblicati (3')

- Google riporta solo il **rank medio** su GIFT-Eval / FEV-Bench / TIME — non una tabella con
  i numeri per singolo task. Il rank medio nasconde l'entità dell'effetto.
- Il pretraining include **GiftEvalPretrain** (con l'overlap verso FEV-Bench rimosso, ma non
  verso GIFT-Eval stesso) più **Wikipedia Pageviews (cutoff nov 2023)** e **Google Trends
  (cutoff fine 2022)**. Significa che serie pubbliche molto note — indici di mercato in testa
  — potrebbero essere state viste in una forma o nell'altra durante il pretraining.
- Conseguenza diretta: un buon risultato di TimesFM-3 su S&P 500/VIX/oro storici **non prova
  zero-shot**. Va progettato un esperimento che lo verifichi, non assunto.

Qui si presenta `config.PRETRAIN_CUTOFF` (fine nov 2023) come confine conservativo, e si
introduce la struttura in due esperimenti che segue.

## 4. Esperimento A — la camera pulita: Magic: The Gathering (4')

- Dominio quasi certamente assente dal pretraining. Dati **interamente post-cutoff**
  (TCGCSV, storico dal 2024-02-08, 7 carte, 846 origini ciascuna).
- **Risultato reale: il modello perde.** Relative MAE 1.153 a h=1, minimo 1.044 a h=11,
  1.076 a h=28 — mai sotto 1 a nessun orizzonte (`exp_mtg_horizon_profile.png`). Per carta:
  solo Sheoldred sotto 1 (0.971), The One Ring il peggio (1.151, DM p=0.0033).
- MASE **non** in slide: dominato da Mishra's Factory (quasi piatta nei primi 64 giorni,
  scala in-sample implicita 0.00032) — vedi commento in `scripts/07_build_slides.py`.
  Diebold-Mariano corretto per orizzonte (non più `horizon=1` fisso su ogni h): modello
  significativamente peggiore nel 14% delle celle (carta, orizzonte), meglio nello 0%.
- Multivariato vs univariato: relative MAE 1.060 vs 1.061 — differenza indistinguibile,
  il multivariato non aiuta né peggiora su MTG.
- raw vs log1p: log1p leggermente peggiore (1.062-1.065 vs 1.060-1.061) — nota per sé: la
  differenza qui NON è "punto=mediana vs media" (il punto è sempre la mediana, sopravvive a
  trasformazioni monotone) — è, se c'è, un effetto di stabilizzazione della varianza.

**Il modello perde contro la naive qui, e va detto così.** "Anche i foundation model perdono
contro tomorrow=today sui prezzi rumorosi" è la slide reale, non un fallback ipotetico.

## 5. Esperimento B — shock, pre-cutoff vs post-cutoff (6') — DEMO LIVE QUI

Il cuore del talk. Stesso identico protocollo su:

- **Eventi pre-cutoff** (dentro la finestra di pretraining): crollo Covid (mar 2020),
  invasione Ucraina (feb 2022), stretta inflazionistica (giu 2022).
- **Eventi post-cutoff** (fuori): unwind carry trade yen (ago 2024), shock dazi (apr 2025).

Le date degli eventi sono confermate — non assunte — da un rilevatore automatico su z-score
dei rendimenti log di SP500 (`detect_shock_days`). **Nota onesta da dire ad alta voce**: nel
run di verifica, il rilevatore a soglia z=4 ha confermato solo Covid e lo shock dazi 2025 fra
i 5 eventi noti — gli altri non producono un salto giornaliero di quella grandezza puro
sull'SP500 (l'Ucraina e l'inflazione sono stati più shock di regime che shock di un giorno,
il carry-trade yen è stato più uno shock di VIX che di SP500). Questo *non* invalida
l'esperimento: si usa comunque la data nota, e lo si dichiara.

**Sequenza demo live** (dal notebook `notebooks/demo.ipynb`, su risultati pre-calcolati,
niente rete il giorno del talk) — vedi anche "Come leggere i grafici", C1/C2/C6 più sotto:

1. lo spaccato su una carta MTG (C1) insegna la grammatica dei colori;
2. forecast one-step su un evento pre-cutoff (Crollo Covid), naive sovrapposto (C2);
3. stesso protocollo su un evento post-cutoff (Shock dazi);
4. **adaptation lag** (C6): giorni perché l'errore torni sotto 1.5× l'errore pre-evento, per
   3 giorni di fila. **Pre-cutoff = 1.7 giorni** (Covid 1, Ucraina 0, Stretta 4) — **post-cutoff
   = 9.5 giorni** (Dazi 15, Yen 4). Detto con la tabella delle soglie subito accanto (C6): la
   soglia di Covid è 7.55%, quella di Dazi 0.99% — varia 7x, va dichiarato prima del numero.

**Il risultato che conta è il delta fra i due bracci**, non il numero assoluto:

- Il modello si comporta **meglio sugli eventi pre-cutoff** (1.7 vs 9.5 giorni) → prova
  (indiretta, non definitiva) di contaminazione: forse ha visto quei pattern durante il
  training. **Ma**: n=3 vs n=2 eventi, e Dazi contiene un secondo shock dentro la finestra
  (offset +4, pausa tariffe, +9.5% reale) che gonfia il suo lag da solo — dirlo se chiesto.

Messaggio di chiusura della sezione:

> "I foundation model per serie temporali non sono oracoli. Sono sistemi di aggiornamento
> probabilistico: riconoscono pattern osservati, ma uno shock veramente nuovo diventa
> prevedibile solo dopo che ha iniziato a lasciare una traccia nei dati — e prima di
> crederci, bisogna verificare che non lo stia semplicemente ricordando."

## 6. Esperimento C — calibrazione: oracolo o probabilità onesta? (3')

- 9 quantili, non solo P10-P90. Curva di calibrazione (copertura empirica vs nominale) e
  PIT, calcolati **allo stesso orizzonte (h=1)** su tre regimi: mercato calmo, mercato
  shock (±3 giorni da un evento noto), MTG come riferimento cross-dominio.
- **Ipotesi confermata**: sotto-copertura nelle code proprio durante lo shock, cioè quando
  servirebbe di più. Copertura P10-P90: **calmo 0.827, shock 0.664** (SP500 da solo: 0.839 →
  0.571), MTG 0.822. Da dire con cura: la prima versione di questo esperimento mischiava
  MTG multi-orizzonte in "calmo" contro mercato h=1 in "shock" e sembrava dire il contrario
  (0.635 vs 0.664) — corretto tenendo l'orizzonte fisso ovunque (`04_exp_calibration.py`).
- `exp_mtg_horizon_profile.png` (C3) mostra il meccanismo: la copertura degrada da 0.82
  (h=1) a 0.54 (h=28) — le bande sono tarate bene a un passo e non si allargano abbastanza.

## 7. Licenza (1')

- Pesi sotto `timesfm-non-commercial-license-v1.0`: non-commercial, non-production. Il
  codice resta Apache-2.0.
- In azienda: va bene per demo, ricerca, valutazione interna. Non va inserito in produzione
  senza chiarire la licenza con Google — punto pratico per il pubblico aziendale.

## 8. Come approfondire (1')

- Repo `google-research/timesfm`, model card su Hugging Face, paper (alphaXiv).
- Concorrenti diretti da guardare: Chronos-2, Toto 2.0, TimesFM-2.5 (quest'ultimo — verificare
  se resta sotto licenza permissiva, possibile alternativa utilizzabile in produzione).
- Questo repository: `tfm3lab`, riproducibile con `uv`.

---

## Come leggere i grafici

Contratto di colore, valido su tutte le figure (demo + slide): **nero** = reale osservato,
**blu** = mediana/previsione del modello, **banda blu chiara** = P10-P90, **grigio
tratteggiato** = naive, **arancione** = braccio pre-cutoff, **rosso** = evento/soglia/allarme.
Imparata una volta su C1, si riusa identica ovunque.

**C1 — `exp_mtg_forecast_slice` (l'eroe, demo passo 0-1).** Storico reale fino al taglio
(linea rossa verticale = `origin_index`, definito in `windows.py` come il primo indice
*predetto*, non l'ultimo osservato). Dopo il taglio: reale (nero), mediana (blu), banda
P10-P90 (blu chiaro), naive = ultimo prezzo osservato ripetuto (grigio tratteggiato).
**Origine 238 su The One Ring è scelta apposta come caso estremo** (~2.6° percentile di
copertura su 5.845 finestre pulite, mediana 0.714) — dirlo ad alta voce, e tenere pronto
`exp_mtg_data_glitch.png` per la domanda "come sai che i dati sono puliti".

**C2 — `exp_shock_reaction_<evento>` (demo passo 2-3).** Asse x = giorni dall'evento
(`offset`). Il punto a x=0 è la previsione fatta con tutto ciò che il modello sapeva fino a
x=-1 — non "il giorno dello shock", ma "la previsione fatta la sera prima". Il naive
sovrapposto (grigio) è `baseline_naive`, cioè `actual` di x-1: il fatto che il blu gli
somigli quasi ovunque (rapporto 0.28-0.38, corr 0.99+, verificato) è il punto centrale del
talk — senza questa riga il grafico non lo dimostra.

**C3 — `exp_mtg_horizon_profile` (slide Esperimento A).** Due pannelli sullo stesso asse
`horizon_step`. Sopra: relative MAE medio ± banda min-max fra le 7 carte, riga a 1.0 (mai
sotto). Sotto: copertura P10-P90 media, riga nominale a 0.80 — degrada da 0.82 (h=1) a 0.54
(h=28). Il pannello sotto è il *meccanismo* dietro la curva di calibrazione: bande
dimensionate bene a un passo, che non si allargano abbastanza in fretta.

**C4 — `exp_mtg_pit_histogram` (slide Esperimento C).** Tre pannelli, h=1/7/28. Bin agli
stessi 9 livelli dei quantili — i bin estremi (`≤ q10`, `≥ q90`) sono conteggi corretti di
"oltre quel quantile", non la vera forma della coda (`pit_values` taglia, non estrapola).
A h=28 il 63.7% della massa è nei due bin estremi contro il 25% atteso.

**C5 — `exp_calibration_curve` (slide Esperimento C).** Tre curve, **tutte a h=1**:
mercato-calmo, mercato-shock, MTG come riferimento cross-dominio. Prima di questo fix,
`04_exp_calibration.py` mischiava MTG multi-orizzonte (1-28) in "calm" contro mercato solo
h=1 in "shock" — la copertura degrada con l'orizzonte (vedi C3), quindi il bucket calmo
sembrava peggiore dello shock: il contrario della verità. A parità di orizzonte, calmo 0.827
vs shock 0.664 (SP500 h=1: 0.839 vs 0.571) — l'ipotesi del piano originale è confermata.

**C6 — `exp_shock_adaptation_dots` (demo passo finale, slide Esperimento B).** Un punto per
evento, non una media per braccio — con n=3 (pre) e n=2 (post) in etichetta. Pannello sotto:
la soglia (mediana errore pre-evento × 1.5) dietro ogni punto, che varia 7x fra eventi
(Covid 7.55%, Dazi 0.99%) perché la finestra pre-evento di Covid conteneva già la rampa del
crollo. Dire questo prima del numero "1.7 vs 9.5 giorni", non dopo.

**C7 — `exp_mtg_data_glitch` (appendice/backup).** The One Ring e Urza's Saga, entrambi
2024-11-15, entrambi ±40-130% in un giorno poi rientrati — due carte diverse sullo stesso
giorno è la prova che è un artefatto TCGCSV, non un evento di mercato. 74 righe su 165.816,
headline invariato con o senza (relative MAE 1.0961). Tenerlo pronto, non presentarlo a meno
che non venga chiesto.

---

## Domande scomode attese (e risposte pronte)

**"Ma quindi è contaminato o no?"**
Non lo sappiamo con certezza — nessuno lo sa, Google non pubblica la lista esatta delle serie
di pretraining. Quello che si può fare è misurare il *comportamento* su eventi che
strutturalmente non potevano essere nel training (post-cutoff) e confrontarlo con eventi che
potevano esserlo. È evidenza indiretta, non una prova.

**"Perché non hai fatto fine-tuning?"**
Il punto del talk è lo zero-shot: è quello che rende un foundation model per serie temporali
interessante rispetto a un ARIMA/ETS addestrato ad hoc. Fine-tuning è un esperimento diverso,
fuori scope qui.

**"La naive è così forte, allora perché usare TimesFM-3?"**
Sui prezzi giornalieri rumorosi sì, la naive è durissima da battere — è proprio per questo che
va sempre riportata come confronto. Il valore di TimesFM-3 non è "batte sempre la naive", è:
zero training, multivariato nativo, quantili calibrati (da verificare) e — se il talk lo
conferma — comportamento robusto agli shock quando genuinamente non ha visto l'evento prima.

**"Si può usare in azienda?"**
Non i pesi attuali, per via della licenza non-commercial. Vale la pena guardare TimesFM-2.5
(verificarne la licenza) o aspettare un rilascio commerciale.

**"Quanto costa in inferenza?"**
Un forward pass copre sempre una patch di 64 step (orizzonte 7 o 28 costano uguale).
`use_symmetric_averaging=True` (default per i numeri "ufficiali") raddoppia il calcolo.
Numeri di latenza reali: `[NUMERO]` ms/forecast su CPU, `[NUMERO]` ms su GPU (da
`BatchForecast.latency_seconds`, raccolto negli esperimenti).

**"Come si collega questo al tuo lavoro (agentic claims)?"**
Il filo conduttore è lo stesso: prima di fidarsi di un output di un modello — che sia una
previsione numerica o un giudizio su un documento — bisogna costruire l'esperimento che lo
mette alla prova, non limitarsi a leggere il benchmark che il vendor pubblica.

---

## Domande scomode sull'architettura (per un pubblico molto tecnico)

Verificato leggendo il sorgente installato (`timesfm3/transformer.py`, `model.py`, `dense.py`,
`cpm_revin_refine.py`) + il blog Google Research + la model card HF — non dal sentito dire.

**"È un transformer? Encoder-decoder o cosa?"**
Decoder-only, non encoder-decoder. 330M parametri, 20 layer, model_dims=1280, 16 teste
(head_dim=80). Nome esatto nella model card: "Stacked Mixing Transformer con Variate
Attention e CPM Iterative RevIN".

**"Come tokenizza l'input? Come un LLM?"**
No. Niente vocabolario discreto, niente embedding lookup. La serie è spezzata in patch
contigue da 32 step (`input_patch_len`), ogni patch è un vettore continuo proiettato a 1280
dimensioni da un `ResidualBlock` (due layer lineari + skip). Context massimo 15.360 = 480
patch — l'attention è quadratica nel numero di *patch*, non di osservazioni.

**"L'attention alternata cosa significa esattamente?"**
Non "metà layer temporali, metà cross-serie". *Ogni* layer (tutti e 20), in sequenza:
1. sequence attention — causale, RoPE, dentro la stessa serie (un token vede solo il passato);
2. variate attention — piena, non causale, **senza RoPE**, fra tutte le serie/covariate allo
   stesso istante (permutation-invariant: nessun ordine imposto fra variate);
3. feedforward (ReLU, RMSNorm, niente bias).
Il canale di variate attention è la vera novità v3 — TimesFM-2.5 era "strictly limited to
univariate forecasting" (citazione blog).

**"Come normalizza dati con scale diverse?"**
RevIN causale e cumulativo per variata: media/std calcolate patch dopo patch, mai su tutta
la finestra in un colpo. Per le patch future mascherate, il "CPM Iterative RevIN Refine"
raffina quelle statistiche incorporando le stime del modello stesso per le patch precedenti
nello stesso blocco — non usa ciecamente l'ultima statistica pre-taglio per tutto l'orizzonte.
Effetto collaterale verificato: una covariata con **passato a varianza zero** fa collassare
la sua scala a un valore arbitrario (guard `_make_safe_for_division`) — la causa del bug che
ho trovato e corretto in `scripts/05_exp_covariates.py` (`leaked_flat_past` vs `leaked`).

**"Autoregressivo o no? Come genera l'intero orizzonte?"**
Non autoregressivo. Non c'è un ciclo genera-un-passo/riattacca-in-input. Vengono appesi token
mascherati per tutto l'orizzonte richiesto e **un solo forward pass** predice tutte le patch
future insieme ("Contiguous Patch Masking"). Oltre i 64 step (`output_patch_len`) entra lo
"stitching": più forward pass con finestre sovrapposte, concatenati — non un ciclo
autoregressivo, ma più chiamate se l'orizzonte supera 64. Sotto i 64 step, orizzonte 7 e
orizzonte 28 costano **esattamente lo stesso** forward pass.

**"Come dà l'incertezza? Assume una gaussiana?"**
No. Testa di output lineare che produce 9 quantili (0.1→0.9) via pinball loss (quantile
regression), non media+varianza. Il punto previsto è **definito come la mediana** (quantile
0.5), quindi sopravvive esattamente a trasformazioni monotone come log1p — non c'è un
mismatch mediana/media da correggere.

**"Le covariate passano per un encoder separato, come in DeepAR/TFT?"**
No. Una covariata è un'altra riga nello stesso tensore `(batch, variata, tempo, dim)`: le
`past_only` sono mascherate nel futuro, le `past_future` (note in anticipo, es. giorno della
settimana o data di uscita di un set) usano una tecnica "lookahead" — il token concatena la
patch corrente con le patch future già note, restando causale solo sul target.

**"In cosa è diverso da un ARIMA?"**
ARIMA fitta coefficienti lineari AR/MA per singola serie, ogni volta, richiede
stazionarietà/differenziazione, e per il multivariato serve un VAR (matrice di covarianza
lineare che esplode con tante serie). TimesFM-3 è zero-shot (pesi fissi, mai un fit sulla tua
serie), cattura pattern non lineari appresi su >1 trilione di punti in pretraining, e il
multivariato è attention appresa — non lineare, scala meglio a molte variate. Contropartita:
ARIMA è interpretabile (coefficienti leggibili), TimesFM-3 è una rete neurale black-box, e
solo TimesFM-3 porta il rischio di contaminazione da pretraining (concetto che per ARIMA non
esiste). **Dato scomodo da dire subito se te lo chiedono**: su MTG il naive puro batte
TimesFM-3 (relative MAE 1.061 medio, mai sotto 1 a nessun orizzonte) — il valore del
foundation model non è "batte sempre un naive ben scelto", è zero-shot + multivariato nativo
quando non hai tempo/dati per fittare un modello per ogni serie.

**"E rispetto a Chronos, Moirai, PatchTST?"**
Chronos (Amazon) discretizza i valori in bin e riusa un vero language model (T5) sui token
discreti — un LLM applicato a serie temporali via tokenizzazione. TimesFM non discretizza
mai: patch continue. Moirai (Salesforce) è encoder-only mascherato con patching multi-scala
(lunghezze diverse per frequenze diverse); TimesFM-3 resta decoder-only, patch fissa a 32.
PatchTST ha introdotto il patching ma è tipicamente fine-tuned per dataset specifico, non
pensato come foundation model zero-shot nello stesso senso. Sui benchmark pubblici (GIFT-Eval,
FEV-Bench, TIME) Google dichiara che TimesFM-3 univariato già eguaglia/supera Chronos-2 e
Toto 2.0, e il multivariato migliora ulteriormente — ma, come già detto, solo rank medio
pubblicato, non tabella per singolo task.
