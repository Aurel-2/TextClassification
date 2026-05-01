# Classification de texte

Le but de ce projet est de travailler sur la classification de sentiments à partir du dataset IMDb, ainsi que sur la détection de spam.

## Prérequis
- Python 3.10 ou supérieur
- uv

## Installation de uv

Si vous n'avez pas encore installé uv :

**Sur Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**Sur Windows**
```bash
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation du projet

Cloner le dépôt :
```bash
git clone https://github.com/Aurel-2/TextClassification.git
cd TextClassification
```

### Environnement virtuel avec uv

Création de l'environnement virtuel :
```bash
uv venv
```
Activer l'environnement virtuel :

**Sur Linux**
```bash
source .venv/bin/activate
```

**Sur Windows**
```bash
.venv\Scripts\activate
```

### Installer les dépendances 
```bash
uv sync
```
