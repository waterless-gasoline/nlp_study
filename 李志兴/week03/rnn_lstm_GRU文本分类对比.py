
"""
RNN / LSTM / GRU 文本分类对比实验（字符级）
"""

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == y).sum().item()
    return correct / max(1, y.numel())



class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len: int):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(ch, 0) for ch in text[: self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class RNNFamilyClassifier(nn.Module):
    """
    rnn_type: "rnn" | "lstm" | "gru"
    """

    def __init__(self, rnn_type: str, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        rnn_type = rnn_type.lower().strip()
        assert rnn_type in {"rnn", "lstm", "gru"}

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.rnn_type = rnn_type
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, E)

        out, h = self.rnn(emb)

        # h 的结构在 LSTM vs RNN/GRU 不同
        if self.rnn_type == "lstm":
            h_n, c_n = h  # (num_layers, B, H)
            last_h = h_n[-1]
        else:
            h_n = h  # (num_layers, B, H)
            last_h = h_n[-1]

        logits = self.fc(last_h)  # (B, C)
        return logits



@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == y).sum().item()
        total += y.numel()
    return total_correct / max(1, total)


def train_one_model(model, train_loader, test_loader, device, epochs: int, lr: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {ep:02d}/{epochs} | loss={running_loss / len(train_loader):.4f} | test_acc={test_acc:.4f}")

    return evaluate(model, test_loader, device)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../Week01/dataset.csv", help="TSV 文件，两列：text \\t label")
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset_df = pd.read_csv(args.data, sep="\t", header=None)
    texts = dataset_df[0].astype(str).tolist()
    string_labels = dataset_df[1].astype(str).tolist()

    # label 映射（用 sorted 保证可复现）
    labels_sorted = sorted(set(string_labels))
    label_to_index = {lab: i for i, lab in enumerate(labels_sorted)}
    numerical_labels = [label_to_index[lab] for lab in string_labels]
    index_to_label = {i: lab for lab, i in label_to_index.items()}

    # char vocab
    char_to_index = {"<pad>": 0}
    for t in texts:
        for ch in t:
            if ch not in char_to_index:
                char_to_index[ch] = len(char_to_index)

    vocab_size = len(char_to_index)
    output_dim = len(label_to_index)

    full_ds = CharDataset(texts, numerical_labels, char_to_index, args.max_len)

    # train/test split
    n_total = len(full_ds)
    n_test = int(n_total * args.test_ratio)
    n_train = n_total - n_test
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    results = []
    for rnn_type in ["rnn", "lstm", "gru"]:
        print("\n" + "=" * 60)
        print(f"Model: {rnn_type.upper()}")
        model = RNNFamilyClassifier(
            rnn_type=rnn_type,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
        ).to(device)

        acc = train_one_model(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr)
        results.append((rnn_type.upper(), acc))

    print("\n" + "=" * 60)
    print("Final Test Accuracy:")
    for name, acc in results:
        print(f"{name:>4}: {acc:.4f}")

    # demo predict（可选）
    def predict(text: str, model_):
        indices = [char_to_index.get(ch, 0) for ch in text[: args.max_len]]
        indices += [0] * (args.max_len - len(indices))
        x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        model_.eval()
        with torch.no_grad():
            logits = model_(x)
        pred = torch.argmax(logits, dim=-1).item()
        return index_to_label[pred]

    # 用最后训练的模型做个示例预测（你也可以改成指定用 GRU/LSTM 等）
    print("\nExample prediction with last model:", results[-1][0])
    sample = "帮我导航到北京"
    print(sample, "->", predict(sample, model))

if __name__ == "__main__":
    main()
# 基础使用： --data ./dataset.csv --epochs 4
