# <h1> Energy Price Prediction 

#### <h2> Gruppenmitglieder:

Jan Schmutz [->](https://www.linkedin.com/in/jan-schmutz-618195280/) <br>
Robin Pankotsch <br>
Andrin Blöchlinger [->](https://www.linkedin.com/in/andrin-bl%C3%B6chlinger-33807623b/)<br>
Leon Hobelmann [->](https://www.linkedin.com/in/leon-hobelmann-04a10a122/)

#### <h2> Projektbeschreibung und Zielsetzung
Im Rahmen der Vorlesung `B5.3 Unternehmenssoftware` im Bachelorstudiengang Wirtschaftsinformatik 
bei [Prof. Dr. Axel Hochstein](https://www.linkedin.com/in/axel-hochstein-ph-d-832b6314/) an der HTW Berlin, wurde ein Projekt zur Thematik `Algorithmic Trading` im Wintersemester 2024/25 durchgeführt.

In diesem Projekt haben wir ein Modell entwickelt, um den `Intraday-Strompreis` für die nächsten
24 Stunden präzise vorherzusagen. Ziel war es, mithilfe unserer Vorhersagen fundierte
Entscheidungen im Energiehandel zu treffen und potenzielle Gewinne zu erzielen.

Für die Umsetzung haben wir ein `Long Short-Term Memory (LSTM)-Modell` in Python
programmiert. Dieses Modell wurde mit historischen Daten zu Wetterbedingungen (Temperatur,
Wind, Sonnenstrahlung etc.), der Einspeisung erneuerbarer Energien sowie der Netzauslastung trainiert.
Wir haben ausschließlich Daten ab dem Jahr 2020 verwendet, da der deutsche Atomausstieg, sowie der 
Krieg in der Ukraine signifikante Veränderungen auf den Strommarkt bewirkten und ältere Daten
daher nicht repräsentativ genug gewesen wären. 

Die initiale Phase des Projekts war geprägt von der Herausforderung, einen strukturierten Rahmen
zu entwickeln und geeignete Datenquellen zu identifizieren. Dieser Schritt war von zentraler
Bedeutung, um eine fundierte Basis für die Modellentwicklung zu gewährleisten.

Zur Unterstützung in dieser kritischen Phase wurde die Expertise von [Jan-Lukas Pflaum](https://www.pflaum.biz/), einem
Fachmann des Berliner Unternehmens [Terra One](https://www.terra.one/), herangezogen. In Besprechungen lieferte er wertvolle
methodische Impulse und praxisorientierte Lösungsansätze. Diese Beiträge waren entscheidend,
um die Projektziele klar zu definieren und einen effektiven Ansatz für die Datenakquisition und
-aufbereitung zu etablieren.

##### <h2> Datensammlung 
Es wurden Daten vom 01.01.2020 bis zum 31.12.2024 im Stundentakt herangezogen. <br>
Bei`gesondert gekennzeichnet Daten`, handelt es sich um selbst entwickelte Datenfeatures.

#### <h3> Strompreisdaten [Quelle](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DE&interval=year&year=2020&legendItems=fy6)
- Intraday-Preis _(€ kW/h)_ (Zielvariable)
- Day-Ahead-Preis _(€ kW/h)_
#### <h3> Wetterdaten [Quelle](https://open-meteo.com/en/docs/historical-weather-api)
Wetterdaten von [30 Standorten in Deutschland](https://earth.google.com/earth/d/1T-NfWtXHGBtu0GtYv641oyPhqwm4X9Ur?usp=sharing) (insgesamt ca. 6200MV Nennleistung) an denen Energie aus Wind oder Solar gewonnen wird. <br> [Quelle - Solar](https://de.wikipedia.org/wiki/Liste_von_Solarkraftwerken_in_Deutschland) <br> [Quelle - Wind - Onshore](https://de.wikipedia.org/wiki/Liste_der_gr%C3%B6%C3%9Ften_deutschen_Onshore-Windparks) <br> [Quelle - Wind - Offshore](https://de.wikipedia.org/wiki/Liste_der_deutschen_Offshore-Windparks)
- Temperatur _(°C)_
- Luftfeuchtigkeit _(%)_
- Regen _(mm)_
- Luftdruck _(hPa)_
- Bewölkung _(%)_
- Windgeschwindigkeit 10 m _(km/h)_
- Windgeschwindigkeit 100 m _(km/h)_
- `turbine_an (boolean - Windgeschwindkeit zwisch 5 & 25 Meter pro Sekunde)` [Quelle](https://de.wikipedia.org/wiki/Siemens_D7-Plattform)
- Tag _(boolean)_
- Kurzwellenstrahlung _(W/mÂ²)_
- Direkte Strahlung _(W/mÂ²)_
- Diffuser Strahlung _(W/mÂ²)_
#### <h3> Kalenderdaten 
- Wochentag _(boolean)_
- Feiertag _(boolean)_
- `wie_viel_Feiertag (% - Bevölkerungsanteil)` [Quelle](https://de.wikipedia.org/wiki/Feiertag_(Deutschland))
#### <h3> Energieversorgungsdaten [Quelle](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=de&c=DE&interval=year&year=2020&legendItems=fy2y5)
- Erneuerbaren Energien _(MV)_
- nicht Erneuerbare Energien _(MV)_
- Netzauslastung _(MV)_
- Kernkraft _(MV)_

#### <h3> Data Splitting 
Aufgrund der starken Saisonalität der Daten, musste das Splitting der Daten in ganzen Jahren erfolgen, um die Chance zu vergrößern, wiederkehrende Muster in den Daten zu finden. <br><br>
Trainingsdaten  = 2020 + 2021 + 2022   <br>
Validierungsdaten = 2023 <br>
Testdaten = 2024

# Zusammenfassung und Ergebnisse
Das LSTM wurde so konzipiert, dass das `Lookback Window` 24 Stunden beträgt und ebenfalls 24 Stunden im `Forecast Horizon` vorhersagt.
Dies ist sinnvoll, da das stärkstes Feature `Day-Ahead-Preis` 24 Stunden im Voraus festgelegt wird. 
Ebenfalls lassen sich relativ präzise Wetterdaten, sowie die Netzauslastungsprognose, 24 Stunden im Voraus vorhersagen. <br><br>
Um das entwickelte LSTM mit anderen Modellen vergleichen zu können, wurde zusätzlich ein `SARIMA (Seasonal Autoregressive Integrated Moving Average)` und ein `Random Forest` Modell entwickelt. <br> <br>
Das Ergebnis unterstreicht unsere anfangs aufgestellte These: Wetterdaten, Auslastungsdaten und Kalenderdaten liefern nutzbare Features für ein `LSTM Strompreis Prediction Modell`. 
#### <h3> **Finales Ergebnis des LSTM vs. SARIMA vs. Random Forest**
![Ergebnisse LSTM.png](Ergebnisse%20LSTM.png) 
# Experimente
Lediglich sechs Experimente wurden veröffentlicht, da bei einigen keine signifikanten Abweichungen vorlagen, die sie als eigenständige Experimente rechtfertigen würden. <br>
<br>
Das sechste Experiment repräsentiert das Ergebnis unseres Projekts und fasst die erzielten Arbeitsergebnisse zusammen.

#### <h3> **Liste der durchgeführten Experimente und deren Ergebnisse**

![Experimente_Übersicht.png](Experimente_%C3%9Cbersicht.png)
_Es wurden alle Versuche/Experimente dokumentiert. (n.d = "nicht dokumentiert")_