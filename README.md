# Solardach Erkennung
In diesem Projekt geht es um die Erkennung von Dächern die neu eingedeckt wurden, auf denen jedoch keine Solaranlage installiert wurde.
## Prerequisites
### UV
Zur Verwendung des Projekts ist UV erforderlich. Die Installation für UV auf verschiedenen Betriebssystemen ist hier zu finden:

https://docs.astral.sh/uv/getting-started/installation/

Anschließend kann folgender Command ausgeführt werden:
```
uv sync
```
### Rebuild Database
Um die Datenbank (DuckDB) zu erstellen und zu befüllen muss folgendes ausgeführt werden:
```
uv run init_database
```
### Scoring
Um bei den Datenmengen eine Vorauswahl zu treffen wurde bereits ein Scoring durchgeführt und mit dem `uv run init_database` Command in der Datenbank hinterlegt. Das Scoring kann überschrieben werden, indem man das Skript in [zero-shot-classification](src\sovia\data_preparation\zero-shot-classification.py) anpasst und ausführt.
### Labeling
Das Labeling der Daten wurde mit Labelstudio durchgeführt und die Ergebnisse des ersten Labelings wurden in `data/input/labeled_data/first_5000_labels.csv` exportiert
### Trainingsdaten Transformation
mit dem Skript `export_data_from_labelingstudio` werden die exportierten Daten aus der CSV in die DuckDb geladen und eine tabelle erstellt, die aus den gelabelten Tasks eine Tabelle macht, die alle Trainingstasks mit Labels und weiteren Informationen versieht.
### Project Structure
```
data-preparation/
├───data
│   ├───database                # Ort an dem die Datenbank abgelegt wird
│   └───input                   # Alle Input Daten, die zum befüllen der DB nötig sind
│       ├───all_shapes          # Runtergeladene Shape Files werden hier abgelegt
│       ├───dachumbau           # Weitere Daten aus Recherchen zu Dachumbauten
│       └───rvr_gebiet          # Manuell erzeugte Shapefiles für das betrachtete RVR Gebiet
├───labelstudio                 # Konfiguration für das Labeling Interface von Labelstudio
└───src                         # Source Code
    └───sovia                   # Code für Datenvorbereitung und das Training des Modells
        ├───data_preparation    # Datenvorverarbeitung
        └───model               # Entwicklung und Training des Modells
```