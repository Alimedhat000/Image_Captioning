import torch.nn as nn
import torch


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
            if num_layers > 1
            else 0,  # don't dropout of we only have one layer
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions, lengths):
        """
        Args:
            features: Image features from encoder [batch_size, embed_size]
            captions: Caption word indices [batch_size, max_length]
            lengthes: Actual lengthes of captions
        """

        # Embed Captions
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)

        # Concatenate Image Features
        features = features.unsqueeze(1)

        embeddings = torch.cat([features, embeddings], dim=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths + 1,  # +1 for image feature
            batch_first=True,
            enforce_sorted=True,
        )

        # LSTM forward pass
        hiddens, _ = self.lstm(packed)

        # Unpack sequences
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        # Project to vocabulary size: [batch_size, max_length+1, vocab_size]
        outputs = self.fc(hiddens)

        return outputs
