# <h1> 4 - Experiment
Interne Bezeichnung: `8. Testlauf`

#### <h2> Kurzbeschreibung 
Anpassung der Hyperparameter durch Optuna Test´s. <br>
Erweiterung der Daten um das Jahr 2024. <br>
Data Splitting wurde angepasst. <br>
Zusätzliche Permutation Feature Importance wurde eingebaut. <br>
Features wurden wieder mit aufgenommen.


#### <h2> Daten

`1Daten 2020 - 2024.xlsx` <br>
5 Jahre Daten <br>
Data Splitting: <br>
2020 + 2021 + 2022= Training <br>
2023 = Validierung <br>
2024 = Test
#### <h2> Features 
- Day-Ahead
- Wochentag (1-7)
- ist_Feiertag (boolean)
- wie_viel_Feiertag (%)
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
  - is_turbine_spinning _(boolen)_
  - Tag _(boolean)_
  - Kurzwellenstrahlung _(W/mÂ²)_
  - Direkte Strahlung _(W/mÂ²)_
  - Diffuser Strahlung _(W/mÂ²)_
#### <h2> Ziele
- Veruch ohne Day-Ahead Preis und andere Auslastungsdaten zu Trainieren. 
- Verbesserung der MSE und MAE

#### <h2> Architektur
Hidden Size=511 <br> Num Layers=1 <br> Dropout=0,5 <br>
- Hyperparameter 
- EPOCHS=50 <br>
- LEARNING_RATE=0.00830 
- PATIENCE=10
- BATCH_SIZE=256

#### <h2> Leistungskriterien
- MSE (Mean Squared Error) <br>  
- MAE (Mean Absolut Error) <br> 
- Early Stopping
#### <h2> Baseline
#### <h2> Ergebnisse![result.png](result.png)
| **Modell**    | MSE <br>(Mean Squared Error) | MAE <br> (Mean Absolut Error) |
|---------------|------------------------------|-------------------------------|
| LSTM          | 0,0035                       | 0,0322                        |


