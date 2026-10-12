<!-- generated from docs/talk-outline.md.template by scripts/11_render_docs.py; do not edit -->

# TimesFM-3 su prezzi post-cutoff: scaletta del talk

Pubblico tecnico. Formato: slide (`slides/talk.md`) più, se c'è tempo, una demo su `notebooks/demo.ipynb`. Durata prevista: circa ventiquattro minuti, tempi per sezione indicati sotto.

Regole di questo documento: ogni numero è letto dal registro delle evidenze (`results/evidence_ledger.json`) tramite segnaposto e verificato da `scripts/11_render_docs.py --check`; i verdetti sono citati alla lettera, nella lingua della regola preregistrata; i risultati descrittivi o esplorativi portano sempre la loro etichetta. Il report completo è `docs/report.md`.

Parole da evitare: "dominio mai visto" (diciamo **dati post-cutoff**), "il cuore del talk" per lo studio sugli shock (è un'appendice esplorativa), qualunque raccomandazione di deploy o di investimento.

---

## Apertura e domanda (tre minuti)

Slide: titolo, domanda e perimetro, cosa abbiamo misurato.

**Messaggio.** Prima di fidarsi di un foundation model per serie temporali bisogna chiedersi come si fa a sapere se generalizza o se sta solo ricordando.

**Note per chi parla.**

- Domanda precisa: TimesFM-3, zero-shot, batte il naive (ultimo valore) sul prezzo giornaliero delle carte Magic?
- Il cutoff di pretraining è 2023-11-30: una lettura prudente della model card, non una data certificata. Per questo si dice sempre "dati post-cutoff".
- Si parla di serie di prezzi, non di consigli di investimento.
- Spiegare la skill in una frase: uno meno il rapporto fra gli errori assoluti medi di modello e baseline; positiva vuol dire che il modello vince.

## Disegno e preregistrazione (quattro minuti)

Slide: preregistrazione e regola più severa.

**Messaggio.** Il disegno è fissato prima di vedere qualunque output del modello, e la regola di verdetto è stata resa più severa prima di vedere i risultati.

**Note per chi parla.**

- Pool di 40 carte estratte per regola e congelate in un manifest; dati TCGCSV da 2024-02-08 alla data di riferimento 2026-09-04; checkpoint `google/timesfm-3.0-pytorch`.
- Intervalli bootstrap al livello 0.95 con B = 2000: uno che ricampiona carte intere, uno a blocchi di calendario.
- La regola: "beats naive" solo se tutti e quattro gli intervalli stanno sopra zero, "loses to naive" solo se tutti stanno sotto, altrimenti "inconclusive". La regola originale, con il solo intervallo primario, è riportata come sensibilità.
- Gli emendamenti sono datati in `docs/analysis-plan.md`; citare quello sullo split di calibrazione di MTG, che serve più avanti.
- Se c'è una domanda sull'auto-controllo statistico: simulazioni su pannelli a skill nota; l'esito PASS vale solo per lo scenario di base con 40 cluster, e le costanti sono state scelte dopo un'anteprima (emendamento B). Lo scenario di stress sotto-copre con il solo intervallo per cluster (copertura 0.755): è una sensibilità, non una verifica della regola, ed è il motivo per cui serve anche l'intervallo a blocchi.

## Esperimento A: il claim primario (sei minuti)

Slide: verdetti, sensibilità e sottoinsiemi, risultati secondari, test per carta e forward-fill, figura.

**Messaggio.** Citare i verdetti così come li produce la regola, senza addolcirli.

**Note per chi parla.**

- A h = 28: "inconclusive". A h = 56: "loses to naive".
- Glossa: "inconclusive" vuol dire che almeno un intervallo non sta tutto da un lato dello zero, quindi non si afferma né vittoria né sconfitta. "loses to naive" vuol dire che il modello perde, e lo diciamo.
- Sensibilità con la regola originale: "inconclusive" e "loses to naive". Pool senza showcase: "inconclusive" e "loses to naive".
- Le carte showcase sono 7 e sono **descrittive**: la colonna verdetto riporta "descriptive (no verdict)".
- I risultati secondari (altri baseline, contesti, trasformazione log1p, multivariato, TimesFM-2.5) sono **descrittivi**: nessun verdetto, nessuna correzione per molteplicità. TimesFM-2.5 è un confronto fra modelli, non un controllo sul cutoff.
- Test di Diebold-Mariano per carta: descrittivi, bilaterali; non scavalcano mai gli intervalli aggregati.
- Il forward-fill è non informativo: l'unico giorno non osservato è 2026-09-04 e sta fuori da ogni finestra di contesto. Non dire mai "robusto".

## Esperimento C: calibrazione (quattro minuti)

Slide: calibrazione P10-P90, figura di copertura, conforme non valutabile, split post-hoc (due slide).

**Messaggio.** L'intervallo P10-P90 ha copertura nominale 0.80; si guarda dove sotto-copre e si distingue con cura ciò che era preregistrato da ciò che è esplorativo.

**Note per chi parla.**

- Un regime sotto-copre se il limite superiore dell'IC a blocchi sta sotto il nominale. Sono valutazioni empiriche di sottogruppo, non garanzie conformi.
- **Conforme su MTG: non valutabile come preregistrato.** Lo split preregistrato (prima di 2025-06-01) ha 0 righe di calibrazione per costruzione, stato `no_calibration_rows`.
- Conforme sul mercato (preregistrato): un IC che contiene il nominale mostra compatibilità, non ripristino della calibrazione; in alcune celle l'intervallo ricalibrato è largo.
- La slide "Esplorativo - split post-hoc" va presentata come tale: regola fissata prima di calcolare i numeri ricalibrati ma dopo aver visto copertura non calibrata e verdetti, quindi nessuna cecità sugli esiti. Non è il verdetto preregistrato.
- Dove "fixes it" è false, l'intervallo ricalibrato non raggiunge il nominale; a h = 1 lo split non sotto-copriva già prima.

## Appendice - Esperimento B: red-team sugli shock (due minuti)

Slide: appendice red-team, figura sull'errore attorno allo shock.

**Messaggio.** È un'appendice esplorativa, non un risultato del claim principale.

**Note per chi parla.**

- La regola congelata ha prodotto 8 eventi pre-cutoff e 7 post-cutoff.
- Verdetto di promozione: "appendix". Motivi: fewer than 8 events per arm (pre=8, post=7); primary CI does not exclude 0.
- Pochi eventi per braccio: nessuna affermazione su adattamento più veloce o più lento dopo uno shock.

## Covariate e contrasto LEAKED (due minuti)

Slide: covariate, LEAKED.

**Messaggio.** Una covariata aiuta solo se batte sia l'assenza di covariata sia il placebo; il braccio LEAKED è solo didattico.

**Note per chi parla.**

- Verdetti sul primario di MTG: "inconclusive" contro nessuna covariata e "inconclusive" contro il placebo.
- Mercato (descrittivo, 2 cluster, esplorativo): contro il placebo il verdetto è "inconclusive".
- Contrasto LEAKED: un braccio che vede valori futuri reali mostra perché una covariata deve essere nota al momento della previsione. Non è un risultato e sulla slide non si mostra nessun numero.

## Latenza (un minuto)

Slide: latenza, figura.

**Messaggio.** Dato descrittivo, una GPU e un ambiente.

**Note per chi parla.**

- Misure su `Tesla T4`, 20 ripetizioni dopo 3 riscaldamenti.
- Non è una raccomandazione di deploy: la licenza del checkpoint è non commerciale e non si fornisce alcun costo.

## Lezioni, limiti, riproducibilità e chiusura (due minuti)

Slide: cosa abbiamo imparato, limiti, riproducibilità, messaggio di chiusura.

**Messaggio.** Il valore sta nel protocollo: regole fissate prima, verdetti alla lettera, esplorativo dichiarato.

**Note per chi parla.**

- Limiti da dire ad alta voce: cutoff come lettura prudente e nessun cutoff documentato per TimesFM-2.5; una sola fonte di quotazioni; shock comuni fra carte; pochi eventi e pochi cluster nelle parti di mercato; conforme su MTG non eseguibile come preregistrato.
- Riproducibilità: ledger con sha256 verificato contro i manifest, `scripts/11_render_docs.py --check` rifiuta numeri senza claim; pool congelato in `configs/benchmark_pool_v2.csv`.
- Chiudere senza promesse: nessun consiglio di investimento, nessuna raccomandazione di deploy.

---

## Domande probabili

- **Perché non dite che il modello perde a h = 28?** Perché la regola dà "inconclusive": non tutti e quattro gli intervalli stanno sotto zero.
- **Il cutoff è certo?** No: è una lettura prudente della model card, per questo parliamo di dati post-cutoff.
- **L'analisi conforme su MTG risolve la sotto-copertura?** Non c'è un verdetto preregistrato; l'analisi post-hoc è esplorativa e non raggiunge il nominale in tutte le celle.
- **Si può usare in produzione?** Il talk non dà raccomandazioni di deploy; la licenza del checkpoint è non commerciale.
