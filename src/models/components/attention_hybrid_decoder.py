import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.attention import Attention


class HybridLSTMGRUDecoder(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        attention_dim=512,
        num_lstm_layers=1,
        num_gru_layers=1,
        dropout=0.5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_gru_layers = num_gru_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size, attention_dim)

        # LSTM processes word + context first
        self.lstm = nn.LSTM(
            input_size=embed_size + embed_size,  # word embedding + context
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # GRU refines LSTM output
        self.gru = nn.GRU(
            input_size=hidden_size,  # takes LSTM output
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize hidden states
        self.init_h_lstm = nn.Linear(embed_size, hidden_size)
        self.init_c_lstm = nn.Linear(embed_size, hidden_size)
        self.init_h_gru = nn.Linear(embed_size, hidden_size)

        self.vocab_size = vocab_size

    def init_hidden_state(self, features):
        """Initialize LSTM and GRU hidden states.

        Args:
            features: [batch, embed_size] or [batch, num_pixels, embed_size]

        Returns:
            h_lstm, c_lstm, h_gru
        """
        # Handle both encoder types
        if features.ndim == 2:
            mean_features = features
        else:
            mean_features = features.mean(dim=1)

        # LSTM needs (h, c)
        h_lstm = (
            self.init_h_lstm(mean_features)
            .unsqueeze(0)
            .repeat(self.num_lstm_layers, 1, 1)
        )
        c_lstm = (
            self.init_c_lstm(mean_features)
            .unsqueeze(0)
            .repeat(self.num_lstm_layers, 1, 1)
        )

        # GRU needs h
        h_gru = (
            self.init_h_gru(mean_features)
            .unsqueeze(0)
            .repeat(self.num_gru_layers, 1, 1)
        )

        return h_lstm, c_lstm, h_gru

    def forward(self, features, captions, lengths):
        batch_size = features.size(0)
        max_length = captions.size(1)

        embeddings = self.dropout(self.embedding(captions))
        h_lstm, c_lstm, h_gru = self.init_hidden_state(features)

        # Handle both encoder types for attention
        if features.ndim == 2:
            features = features.unsqueeze(1)

        outputs = []
        alphas = []

        for t in range(max_length):
            # Attention using GRU's top layer state
            context, alpha = self.attention(features, h_gru[-1])

            # LSTM processes word + context
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(1)
            lstm_out, (h_lstm, c_lstm) = self.lstm(lstm_input, (h_lstm, c_lstm))

            # GRU refines LSTM output
            gru_out, h_gru = self.gru(lstm_out, h_gru)

            # Predict next word
            preds = self.fc(self.dropout(gru_out.squeeze(1)))

            outputs.append(preds)
            alphas.append(alpha)

        return torch.stack(outputs, dim=1), torch.stack(alphas, dim=1)

    def generate_caption(self, features, vocab, max_length=20, device="cpu"):
        result = []
        attention_weights = []

        h_lstm, c_lstm, h_gru = self.init_hidden_state(features)

        # Handle both encoder types
        if features.ndim == 2:
            features = features.unsqueeze(1)

        word = torch.tensor([vocab.start_idx]).to(device)

        for _ in range(max_length):
            word_embed = self.embedding(word)
            context, alpha = self.attention(features, h_gru[-1])
            attention_weights.append(alpha.squeeze(0).cpu().detach().numpy())

            # LSTM forward
            lstm_input = torch.cat([word_embed, context], dim=1).unsqueeze(1)
            lstm_out, (h_lstm, c_lstm) = self.lstm(lstm_input, (h_lstm, c_lstm))

            # GRU forward
            gru_out, h_gru = self.gru(lstm_out, h_gru)

            # Predict
            output = self.fc(gru_out.squeeze(1))
            predicted = output.argmax(1)

            result.append(predicted.item())
            if predicted.item() == vocab.end_idx:
                break

            word = predicted

        return result, attention_weights

    def beam_search(self, features, vocab, beam_width=5, max_length=20, device="cpu"):
        k = beam_width
        h_lstm, c_lstm, h_gru = self.init_hidden_state(features)

        # Handle both encoder types
        if features.ndim == 2:
            features = features.unsqueeze(1)

        sequences = [[vocab.start_idx]]
        scores = [0.0]
        lstm_states = [(h_lstm, c_lstm)]
        gru_states = [h_gru]
        attention_history = [[]]

        for step in range(max_length):
            all_candidates = []

            for i, seq in enumerate(sequences):
                if seq[-1] == vocab.end_idx:
                    all_candidates.append(
                        (
                            seq,
                            scores[i],
                            lstm_states[i],
                            gru_states[i],
                            attention_history[i],
                        )
                    )
                    continue

                h_lstm_current, c_lstm_current = lstm_states[i]
                h_gru_current = gru_states[i]

                word = torch.tensor([seq[-1]]).to(device)
                word_embed = self.embedding(word)

                context, alpha = self.attention(features, h_gru_current[-1])

                # LSTM forward
                lstm_input = torch.cat([word_embed, context], dim=1).unsqueeze(1)
                lstm_out, (h_lstm_new, c_lstm_new) = self.lstm(
                    lstm_input, (h_lstm_current, c_lstm_current)
                )

                # GRU forward
                gru_out, h_gru_new = self.gru(lstm_out, h_gru_current)

                output = self.fc(gru_out.squeeze(1))
                log_probs = F.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(k, dim=1)

                for j in range(k):
                    candidate_seq = seq + [top_indices[0, j].item()]
                    candidate_score = scores[i] + top_log_probs[0, j].item()
                    candidate_attention = attention_history[i] + [
                        alpha.squeeze(0).cpu().detach().numpy()
                    ]

                    all_candidates.append(
                        (
                            candidate_seq,
                            candidate_score,
                            (h_lstm_new.clone(), c_lstm_new.clone()),
                            h_gru_new.clone(),
                            candidate_attention,
                        )
                    )

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [c[0] for c in ordered[:k]]
            scores = [c[1] for c in ordered[:k]]
            lstm_states = [c[2] for c in ordered[:k]]
            gru_states = [c[3] for c in ordered[:k]]
            attention_history = [c[4] for c in ordered[:k]]

            if all(seq[-1] == vocab.end_idx for seq in sequences):
                break

        best_idx = scores.index(max(scores))
        return sequences[best_idx], attention_history[best_idx]
