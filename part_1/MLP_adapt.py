import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

# Chargement des données
USE_HF = False  # mettre True si datasets installé

if USE_HF:
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    X_train = dataset["train"]["text"][:5000]
    y_train = dataset["train"]["label"][:5000]
else:
    import pandas as pd
    df1 = pd.read_csv('../data/imdb_train.csv')
    X_train = df1['text']
    y_train = df1['label']
    df2 = pd.read_csv('../data/imdb_test.csv')
    X_test = df2['text']
    y_test = df2['label']

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents='unicode',
    lowercase=True,
    stop_words='english'
)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Tenseurs
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)


# Modèle
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2) # Deux classes

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

model = MLP()

criterion = nn.CrossEntropyLoss() # BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Entraînement
def train():
    model.train()
    for epoch in range(4):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Évaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    print("Accuracy:", correct / total)


train()
evaluate()