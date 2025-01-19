# <h1> 1 - Experiment
Interne Bezeichnung: `3. Testlauf`

#### <h2> Kurzbeschreibung 
Einer der ersten Versuche, eine LSTM mittels von ChatGPT vorgeschlagener Modellarchitektur bzw. Hyperparametern. <br>

#### <h2> Daten
`10Daten 2020 - 2023.xlsx` <br>
4 Jahre Daten <br>
Data Splitting: <br>
2020 + 2021 = Training <br>
2022 = Validierung <br>
2023 = Test
#### <h2> Features
- Day-Ahead
- Wochentag (1-7)
- Pumpenspeicher
- Handel
- Netzlast
- erneuerbare Energien
- nicht erneuerbare Energien
- Kernenergie 
- Wetterdaten (zu allen 30 Standorten)
  - Temperatur _(°C)_
  - Luftfeuchtigkeit _(%)_
  - Regen _(mm)_
  - Luftdruck _(hPa)_
  - Bewölkung _(%)_
  - Windgeschwindigkeit 10 m _(km/h)_
  - Windgeschwindigkeit 100 m _(km/h)_
  - Tag _(boolean)_
  - Kurzwellenstrahlung _(W/mÂ²)_
  - Direkte Strahlung _(W/mÂ²)_
  - Diffuser Strahlung _(W/mÂ²)_
#### <h2> Ziele
- Vorhersage des Intraday Wert 
- Verbesserung der MSE und MAE
- Early-Stopping für bessere Leistung
#### <h2> Architektur
LSTM Modell verfügt über zwei gestapelte LSTM Layer:
- **Erster LSTM Layer:** _<br> Hidden Size 128 <br> Dropout 0,3_
- **Zweiter LSTM Layer:** _<br> Hidden Size 64 <br> Dropout 0,3_
- Hyperparameter 
- EPOCHS=50 <br>
- LEARNING_RATE=0.0001 
- PATIENCE=5
- BATCH_SIZE=512

#### <h2> Leistungskriterien
- MSE (Mean Squared Error) <br>  
- MAE (Mean Absolut Error) <br> 
- Early Stopping
#### <h2> Baseline
#### <h2> Ergebnisse![result.png](result.png)
| **Modell**    | MSE <br>(Mean Squared Error) | MAE <br> (Mean Absolut Error) |
|---------------|------------------------------|-------------------------------|
| LSTM          | 0,0115                       | 0,0921                        |


