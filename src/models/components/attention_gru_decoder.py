import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.attention import Attention


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
        """
        GRU-based decoder with attention mechanism for image captioning.

        Args:
            embed_size (int): Size of word embeddings.
            hidden_size (int): Number of hidden units in GRU.
            vocab_size (int): Size of the output vocabulary.
            attention_dim (int): Size of attention intermediate layer. Default: 512
            num_layers (int): Number of GRU layers. Default: 1
            dropout (float): Dropout rate for embeddings and GRU. Default: 0.5
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(
            encoder_dim=embed_size, decoder_dim=hidden_size, attention_dim=attention_dim
        )

        self.gru = nn.GRU(
            input_size=embed_size + embed_size,  # word embedding + context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.vocab_size = vocab_size

    def init_hidden_state(self, features):
        """
        Initialize the GRU hidden state from encoder features.

        Args:
            features (torch.Tensor): Encoder output of shape
                [batch, embed_size] or [batch, num_pixels, embed_size].

        Returns:
            torch.Tensor: Initialized hidden state of shape
                [num_layers, batch, hidden_size].
        """
        # Handle both shapes:
        # - [batch, embed_size]
        # - [batch, num_pixels, embed_size]

        if features.ndim == 2:
            # [batch, embed_size]
            mean_features = features
        else:
            # [batch, num_pixels, embed_size]
            # Take the mean of all features
            mean_features = features.mean(dim=1)

        # init the hidden state for all layers
        h = self.init_h(mean_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h

    def forward(self, features, captions, lengths):
        """
        Forward pass for training.

        Args:
            features (torch.Tensor): Encoder output. [Batch, num_pixels, embed_size] or [batch, embed_size]
            captions (torch.Tensor): Ground truth captions. [batch, max_length]
            lengths (list[int]): Lengths of each caption.

        Returns:
            tuple:
                - outputs (torch.Tensor): Predicted token scores, shape [batch, max_length, vocab_size].
                - alphas (torch.Tensor): Attention weights, shape [batch, max_length, num_pixels].
        """

        batch_size = features.size(0)  # B
        max_length = captions.size(1)  # T (time steps)

        # Embed captions [B, T] -> [B, T, embed-size]
        embeddings = self.dropout(self.embedding(captions))
        # Init hidden state [num_layers, B, hidden_size]
        hidden = self.init_hidden_state(features)

        # Handle both encoder types for attention
        if features.ndim == 2:
            # Old encoder: add sequence dimension
            features = features.unsqueeze(1)  # [batch, 1, embed_size]
        # Now features is always: [B, L, embed_size] where L = num_pixels

        outputs = []
        alphas = []

        for t in range(max_length):
            # Compute attention over encoder features
            # hidden[-1]: [B, hidden_size] output of the last GRU layer
            # context: [B, embed_size], alpha: [batch_size, num_pixels]
            context, alpha = self.attention(features, hidden[-1])

            # Concatenate word embedding with context
            # get the specific time step in the embeddings embeddings[:, t, :]: [B, embed_size]
            # context: [B, embed_size]
            # after the cat: [B, 2*embed_size] -> [B, 1, 2*embed_size]
            input = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(1)

            # GRU forward
            # output: [B, 1, hidden_size]
            output, hidden = self.gru(input, hidden)

            # Project to vocabulary
            # preds: [B, vocab_size]
            preds = self.fc(self.dropout(output.squeeze(1)))

            outputs.append(preds)  # [B, vocab_size]
            alphas.append(alpha)  # [B, L]

        # Stack predictions and attention weights across time
        # outputs: [B, T, vocab_size]
        # alphas: [B, T, L]
        return torch.stack(outputs, dim=1), torch.stack(alphas, dim=1)

    def generate_caption(self, features, vocab, max_length=200, device="cpu"):
        """
        Generate caption using greedy decoding.

        Args:
            features: [b, L, embed_size] or [b, embed_size]
            vocab: Vocabulary object
            max_length: maximum caption length
            device: device to run on

        Returns:
            result: list of token indices
            attention_weights: list of attention weight arrays
        """

        result = []
        attention_weights = []

        h = self.init_hidden_state(features)

        # Handle both encoder types
        if features.ndim == 2:
            features = features.unsqueeze(1)  # [batch, 1, embed_size]

        # start with <sos>
        word = torch.tensor([vocab.start_idx]).to(device)

        for _ in range(max_length):
            word_embed = self.embedding(word)
            context, alpha = self.attention(features, h[-1])
            attention_weights.append(alpha.squeeze(0).cpu().detach().numpy())

            lstm_input = torch.cat([word_embed, context], dim=1).unsqueeze(1)
            output, h = self.gru(lstm_input, h)
            output = self.fc(output.squeeze(1))

            # get most likely word
            predicted = output.argmax(1)
            result.append(predicted.item())

            if predicted.item() == vocab.end_idx:
                break

            word = predicted

        return result, attention_weights

    def beam_search(self, features, vocab, beam_width=5, max_length=200, device="cpu"):
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
