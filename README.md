# Classification de texte

Le but de ce projet est de travailler sur la classification de sentiments à partir du dataset IMDb et la détection de spam.
## Prérequis

- Python 3.10+

## Installation de uv

Si vous n'avez pas encore `uv` installé, voici comment le faire :

### Sur Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sur Windows

```bash
powershell -ExecutionPolicy BypassUser -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation du projet

1. Cloner le repository :
2. 
```bash
git clone https://github.com/Aurel-2/TextClassification.git
cd textclassification
```
Créer un environnement virtuel avec uv :

```bash
uv venv
```

2. Activer l'environnement virtuel :

Sur Linux :
````bash
source .venv/bin/activate
````

Sur Windows :
````bash
.venv\Scripts\activate
````

3. Installer les dépendances avec `uv` :
```bash
uv sync
```


