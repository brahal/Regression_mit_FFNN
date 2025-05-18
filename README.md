# 📘 EA2 – Regression mit Feed-Forward Neural Network (FFNN)

**Modul:** Deep Learning (BHT MIM 20 S25)  
**Studentin:** Basma Rahal  
**Technologien:** TensorFlow.js, Plotly.js, HTML5, JavaScript, CSS3

---

## 🧠 Aufgabenstellung

Ziel ist die Approximation einer unbekannten Funktion durch ein neuronales Netz. Die Ground-Truth-Funktion lautet:

```
y(x) = 0.5*(x+0.8)*(x+1.8)*(x-0.2)*(x-0.3)*(x-1.9) + 1
```

Diese wird auf dem Intervall [-2, 2] mit verrauschten und unverrauschten Daten erlernt.

---

## 📂 Projektstruktur

```
ea2-ffnn-regression/
│
├── index.html                  # Hauptseite mit allen Visualisierungen & Dokumentation
├── README.md                   # Diese Datei
│
├── assets/
│   ├── main.js                 # Logik für Daten, Modell, Training, Plots
│   └── style.css              # Layout und Gestaltung
```

---

## 🛠️ Verwendete Technologien

| Framework / Bibliothek | Verwendung                    |
|------------------------|-------------------------------|
| TensorFlow.js          | FFNN Modell, Training, MSE    |
| Plotly.js              | Interaktive Visualisierungen  |
| HTML/CSS/JS            | Webanwendung & Layout         |

---

## 🔬 Funktionsweise

- **Datengenerierung**
    - 100 Punkte aus [-2, 2]
    - zur Hälfte Trainings- und Testdaten
    - Optional: Additives Label-Rauschen (Gaussian Noise, Varianz 0.05)

- **Modelle**
    - 3 Modelle werden trainiert:
        - Ohne Rauschen
        - Mit Rauschen (Best-Fit)
        - Mit Rauschen (Overfitting)

- **Netzarchitektur**
    - 2 Hidden-Layer à 100 Neuronen
    - ReLU-Aktivierung
    - Linearer Output
    - Optimizer: Adam (LR=0.01), MSE als Loss

- **Visualisierung**
    - Scatter-Plots: echte Daten
    - Linien: Modellvorhersagen
    - MSE-Werte für Train/Test

---

## 🔎 Ergebnisse & Diskussion

- Modell ohne Rauschen generalisiert nahezu perfekt
- Best-Fit-Modell lernt trotz Rauschen stabil
- Overfit-Modell zeigt klares Overfitting (Train-Loss << Test-Loss)
- Modellverhalten ist visuell nachvollziehbar

---

## 🚀 Lokale Ausführung

1. Projekt herunterladen oder klonen
2. Öffne `index.html` lokal im Browser (empfohlen: Chrome)  
   oder verwende [Live Server für VS Code](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer)

---

## 🌐 Deployment (optional)

### GitHub Pages

1. Repository erstellen
2. Code pushen
3. Einstellungen → Pages → Quelle: `main`, Ordner: `/ (root)`
4. Seite unter `https://deinname.github.io/ea2-ffnn-regression` erreichbar

### Netlify

1. Netlify Account erstellen
2. Repository verbinden oder `.zip` hochladen
3. Deploy → eigene URL erhalten

---

## 🧾 Lizenz

Dieses Projekt wurde im Rahmen einer benoteten Einsendeaufgabe entwickelt und unterliegt den universitären Prüfungsbedingungen.