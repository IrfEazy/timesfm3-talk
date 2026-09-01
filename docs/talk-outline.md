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
  (TCGCSV, storico dal 2024-02-08).
- Risultato: TimesFM-3 batte la naive? `MASE_model = [NUMERO]` vs `MASE_naive = [NUMERO]`,
  test di Diebold-Mariano `p = [NUMERO]`.
- Multivariato vs univariato sulle carte: `relative_MAE = [NUMERO]`.
- raw vs log1p: `[NUMERO]` — nota per sé: la differenza qui NON è "punto=mediana vs media"
  (il punto è sempre la mediana, sopravvive a trasformazioni monotone) — è, se c'è, un
  effetto di stabilizzazione della varianza su prezzi con scale molto diverse.

**Se il modello perde contro la naive qui**, dillo lo stesso: è un risultato, non un
fallimento del talk. "Anche i foundation model perdono contro tomorrow=today sui prezzi
rumorosi" è una slide legittima.

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
niente rete il giorno del talk):

1. forecast one-step emesso il giorno prima dello shock;
2. si aggiunge il primo dato reale post-shock;
3. nuovo forecast;
4. confronto con il percorso effettivo, grafico errore;
5. **adaptation lag**: quanti giorni servono perché l'errore torni sotto 1.5× l'errore
   pre-evento, per 3 giorni di fila. `pre-cutoff = [NUMERO] giorni`, `post-cutoff = [NUMERO]
   giorni`.
6. Ripetere affiancato su un evento pre-cutoff e uno post-cutoff.

**Il risultato che conta è il delta fra i due bracci**, non il numero assoluto:

- Se il modello si comporta **meglio sugli eventi pre-cutoff** → prova (indiretta, non
  definitiva) di contaminazione: forse ha visto quei pattern durante il training.
- Se **non c'è differenza** → prova a favore della generalizzazione, e più credibile
  proprio perché si è cercato attivamente di falsificarla.

Messaggio di chiusura della sezione:

> "I foundation model per serie temporali non sono oracoli. Sono sistemi di aggiornamento
> probabilistico: riconoscono pattern osservati, ma uno shock veramente nuovo diventa
> prevedibile solo dopo che ha iniziato a lasciare una traccia nei dati — e prima di
> crederci, bisogna verificare che non lo stia semplicemente ricordando."

## 6. Esperimento C — calibrazione: oracolo o probabilità onesta? (3')

- 9 quantili, non solo P10-P90. Curva di calibrazione (copertura empirica vs nominale) e
  PIT, calcolati separatamente in regime calmo (MTG) e in regime di shock (±3 giorni da un
  evento noto).
- Ipotesi da verificare, non da assumere: sotto-copertura nelle code proprio durante lo
  shock, cioè quando servirebbe di più. `coverage_calm = [NUMERO]`, `coverage_shock =
  [NUMERO]`.

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
