import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGRUDecoder(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        attention_dim=512,
        num_layers=1,
        dropout=0.5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(
            embed_size, hidden_size, attention_dim
        )  # encoder_dim = embed_size

        self.gru = nn.GRU(
            input_size=embed_size + embed_size,  # word embedding + context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize hidden state from MEAN of spatial features
        self.init_h = nn.Linear(embed_size, hidden_size)

        self.vocab_size = vocab_size

    def init_hidden_state(self, features):
        # Handle both shapes:
        # - Old encoder: [batch, embed_size]
        # - New encoder: [batch, num_pixels, embed_size]

        if features.ndim == 2:
            # Old encoder: [batch, embed_size]
            mean_features = features
        else:
            # New encoder: [batch, num_pixels, embed_size]
            mean_features = features.mean(dim=1)

        h = self.init_h(mean_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h

    def forward(self, features, captions, lengths):
        batch_size = features.size(0)
        max_length = captions.size(1)

        embeddings = self.dropout(self.embedding(captions))
        h = self.init_hidden_state(features)

        # Handle both encoder types for attention
        if features.ndim == 2:
            # Old encoder: add sequence dimension
            features = features.unsqueeze(1)  # [batch, 1, embed_size]

        outputs = []
        alphas = []

        for t in range(max_length):
            context, alpha = self.attention(features, h[-1])
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(1)
            output, h = self.gru(lstm_input, h)
            preds = self.fc(self.dropout(output.squeeze(1)))

            outputs.append(preds)
            alphas.append(alpha)

        return torch.stack(outputs, dim=1), torch.stack(alphas, dim=1)

    def generate_caption(self, features, vocab, max_length=20, device="cpu"):
        result = []
        attention_weights = []

        h = self.init_hidden_state(features)

        # Handle both encoder types
        if features.ndim == 2:
            features = features.unsqueeze(1)  # [batch, 1, embed_size]

        word = torch.tensor([vocab.start_idx]).to(device)

        for _ in range(max_length):
            word_embed = self.embedding(word)
            context, alpha = self.attention(features, h[-1])
            attention_weights.append(alpha.squeeze(0).cpu().detach().numpy())

            lstm_input = torch.cat([word_embed, context], dim=1).unsqueeze(1)
            output, h = self.gru(lstm_input, h)
            output = self.fc(output.squeeze(1))
            predicted = output.argmax(1)

            result.append(predicted.item())
            if predicted.item() == vocab.end_idx:
                break

            word = predicted

        return result, attention_weights

    def beam_search(self, features, vocab, beam_width=5, max_length=20, device="cpu"):
        # features: [1, 49, embed_size]
        k = beam_width
        h = self.init_hidden_state(features)

        sequences = [[vocab.start_idx]]
        scores = [0.0]
        states = [h]
        attention_history = [[]]

        for step in range(max_length):
            all_candidates = []

            for i, seq in enumerate(sequences):
                if seq[-1] == vocab.end_idx:
                    all_candidates.append(
                        (seq, scores[i], states[i], attention_history[i])
                    )
                    continue

                h_current = states[i]
                word = torch.tensor([seq[-1]]).to(device)
                word_embed = self.embedding(word)

                # Attention over spatial features
                context, alpha = self.attention(features, h_current[-1])

                lstm_input = torch.cat([word_embed, context], dim=1).unsqueeze(1)
                output, h_new = self.gru(lstm_input, h_current)
                output = self.fc(output.squeeze(1))
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
                            h_new.clone(),
                            candidate_attention,
                        )
                    )

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [c[0] for c in ordered[:k]]
            scores = [c[1] for c in ordered[:k]]
            states = [c[2] for c in ordered[:k]]
            attention_history = [c[3] for c in ordered[:k]]

            if all(seq[-1] == vocab.end_idx for seq in sequences):
                break

        best_idx = scores.index(max(scores))
        return sequences[best_idx], attention_history[best_idx]


class Attention(nn.Module):
    """Soft Attention mechanism.

    Computes attention weights over encoder features based on decoder state.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """Initialize attention.

        Args:
            encoder_dim: Encoder feature dimension
            decoder_dim: Decoder hidden state dimension
            attention_dim: Attention hidden dimension
        """
        super().__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        """Compute attention.

        Args:
            encoder_out: Encoder features [batch_size, num_pixels, encoder_dim]
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]

        Returns:
            context: Attended features [batch_size, encoder_dim]
            alpha: Attention weights [batch_size, num_pixels]
        """
        # Transform encoder features
        att1 = self.encoder_att(encoder_out)  # [batch_size, num_pixels, attention_dim]

        # Transform decoder state
        att2 = self.decoder_att(decoder_hidden)  # [batch_size, decoder_dim]
        att2 = att2.unsqueeze(1)  # [batch_size, 1, attention_dim]

        # Compute attention scores
        att = self.full_att(torch.tanh(att1 + att2))  # [batch_size, num_pixels, 1]
        att = att.squeeze(2)  # [batch_size, num_pixels]

        # Softmax to get attention weights
        alpha = F.softmax(att, dim=1)  # [batch_size, num_pixels]

        # Compute weighted sum of encoder features
        context = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # [batch_size, encoder_dim]

        return context, alpha
