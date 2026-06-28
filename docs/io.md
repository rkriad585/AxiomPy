# Data I/O (`Axiom.io`)

Import and export data in multiple formats without external libraries.

## Quick Start

```python
from axiompy import Axiom

# Save / load
Axiom.io.save([1, 2, 3], "data.axi")
data = Axiom.io.load("data.axi")
```

## Supported Formats

| Format | Extension | Save | Load |
|--------|-----------|------|------|
| AxiomPy custom | `.axi` | `save_axi()` | `load_axi()` |
| JSON | `.json` | `save_json()` | `load_json()` |
| CSV | `.csv` | `save_csv()` | `load_csv()` |
| Text | `.txt` | `save_txt()` | `load_txt()` |

## Auto-detect

```python
Axiom.io.save(data, "file.axi")   # auto-detect from extension
data = Axiom.io.load("file.json") # auto-detect from extension
```

## CSV

```python
Axiom.io.save_csv(rows, "data.csv", headers=["x", "y"])
headers, rows = Axiom.io.load_csv("data.csv", has_headers=True)
```

## JSON

```python
Axiom.io.save_json({"key": [1, 2, 3]}, "data.json")
data = Axiom.io.load_json("data.json")
```

## TXT (pretty-print tables)

```python
Axiom.io.save_txt(rows, "table.txt", headers=["Name", "Value"])
lines = Axiom.io.load_txt("table.txt")
```
