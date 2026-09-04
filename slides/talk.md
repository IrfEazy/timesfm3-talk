---
marp: true
theme: default
paginate: true
backgroundColor: #0f172a
color: #f1f5f9
style: |
  section { font-size: 26px; }
  h1, h2 { color: #38bdf8; }
  code { color: #fbbf24; }
  .small { font-size: 18px; color: #94a3b8; }
---

<!-- _paginate: false -->
# TimesFM-3: oracolo o aggiornatore probabilistico?

Mettere alla prova un foundation model per serie temporali su dati che
(quasi) certamente non ha mai visto.

---

## Esiste un modello fondazionale per le serie temporali

- Zero training, zero fine-tuning: gli dai una serie, previene.
- 330M parametri, **multivariato nativo**, context massimo **15.360** step.
- Decoding non autoregressivo: un intero orizzonte di **64 step** esce da un
  solo forward pass — orizzonte 7 e orizzonte 28 costano lo stesso.

<span class="small">timesfm3_forecaster.py, model card google/timesfm-3.0-pytorch</span>

---

## Perché non mi fido dei benchmark pubblicati

- Google riporta solo il **rank medio** su GIFT-Eval/FEV-Bench/TIME.
- Il pretraining include Wikipedia Pageviews (**cutoff nov 2023**) e Google
  Trends (**cutoff fine 2022**), più GiftEvalPretrain.
- Un buon risultato su serie di mercato note **non prova zero-shot**.

---

## Esperimento A — la camera pulita: Magic: The Gathering

Dominio quasi certamente assente dal pretraining. Dati **post-cutoff**
(TCGCSV, storico dal 2024-02-08).

- Relative MAE (< 1 = batte il naive): h=1 **1.048**, h=28
  **1.188** — mai sotto 1
- Diebold-Mariano: modello **peggio** nel **21%** delle
  celle (p<0.05), **meglio** nel 0%
- Multivariato vs univariato: relative MAE **1.113**

---

## Esperimento B — shock, pre-cutoff vs post-cutoff

**Il cuore del talk.** Stesso protocollo su eventi dentro e fuori la
finestra di pretraining.

- Adaptation lag medio, **pre-cutoff**: **1.7** giorni
- Adaptation lag medio, **post-cutoff**: **9.5** giorni

<span class="small">Demo live qui — notebooks/demo.ipynb</span>

---

## Esperimento C — calibrazione

- Copertura P10-P90, mercato calmo (h=1): **0.827**
- Copertura P10-P90, mercato shock (h=1): **0.664**
- (nominale: 0.80 — stesso orizzonte in entrambi i regimi, mai mischiato con MTG)

---

## Esperimento D — covariate: fatte bene, e fatte male apposta

- Covariata lecita (giorno della settimana, uscita set): miglioramento reale
  ma modesto.
- **Controllo negativo**: si passa il prezzo futuro reale come covariata —
  controllo negativo confermato: il MAE crolla al **15%** del pulito quando il prezzo futuro reale filtra nella covariata.
  <span class="small">MAE pulito 0.2097 — MAE con leak 0.0322</span>

---

## Licenza

- Pesi: `timesfm-non-commercial-license-v1.0` — non-commercial, non-production.
- Codice: Apache-2.0.
- In azienda: demo/ricerca sì, produzione no senza chiarire con Google.

---

## Messaggio di chiusura

> I foundation model per serie temporali non sono oracoli. Sono sistemi di
> aggiornamento probabilistico: riconoscono pattern osservati, ma uno shock
> veramente nuovo diventa prevedibile solo dopo aver iniziato a lasciare
> una traccia nei dati — e prima di crederci, bisogna verificare che non lo
> stia semplicemente ricordando.

<span class="small">Repo: tfm3lab — riproducibile con uv</span>
