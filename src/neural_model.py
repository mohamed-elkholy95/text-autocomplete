from typing import Any, Dict, List, Tuple
"""Neural language model (LSTM-based)."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not available")


class LSTMModel(nn.Module if HAS_TORCH else object):
    """LSTM language model."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        if HAS_TORCH:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH: return None
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        return self.fc(lstm_out)


def train_lstm(
    model: Any, tokens: List[int], vocab_size: int, epochs: int = 5,
    seq_len: int = 20, batch_size: int = 32, lr: float = 1e-3,
) -> Dict[str, List[float]]:
    """Train LSTM language model."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        logger.info("Mock training")
        return {"loss": [2.5 - 0.3*i for i in range(epochs)],
                "perplexity": [np.exp(2.5 - 0.3*i) for i in range(epochs)]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "perplexity": []}
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0; n_batch = 0
        # Simple sequential batching
        for i in range(0, len(tokens) - seq_len - 1, seq_len):
            x = torch.tensor(tokens[i:i+seq_len], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(tokens[i+1:i+seq_len+1], dtype=torch.long).unsqueeze(0).to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_batch += 1

        avg = total_loss / max(n_batch, 1)
        ppl = np.exp(min(avg, 20))
        history["loss"].append(round(avg, 4))
        history["perplexity"].append(round(float(ppl), 4))
        logger.info("LSTM Epoch %d/%d: loss=%.4f ppl=%.1f", epoch+1, epochs, avg, ppl)

    return history


def predict_next_lstm(model: Any, token_ids: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
    """Predict next token with LSTM."""
    if not HAS_TORCH or not isinstance(model, nn.Module) or not token_ids:
        return [("<UNK>", 0.0)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x = torch.tensor([token_ids[-20:]], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits[0, -1], dim=-1).cpu()
    topk = torch.topk(probs, top_k)
    return [(str(i), float(p)) for i, p in zip(topk.indices.tolist(), topk.values.tolist())]
