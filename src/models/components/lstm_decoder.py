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
            batch_first=True,
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
            (lengths + 1).cpu(),  # +1 for image feature
            batch_first=True,
            enforce_sorted=False,
        )

        # LSTM forward pass
        hiddens, _ = self.lstm(packed)

        # Unpack sequences
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        # Project to vocabulary size: [batch_size, max_length+1, vocab_size]
        outputs = self.fc(hiddens)

        return outputs

    def generate_caption(self, features, vocab, max_length=500, device="cpu"):
        """Generate caption using greedy search (for inference).

        Args:
            features: Image features [1, embed_size]
            vocab: Vocabulary object with start_idx and end_idx
            max_length: Maximum caption length
            device: Device to run on

        Returns:
            caption: List of word indices
        """
        result = []
        states = None

        # Start with image features
        inputs = features.unsqueeze(1)  # [1, 1, embed_size]

        for _ in range(max_length):
            # LSTM forward
            hiddens, states = self.lstm(inputs, states)

            # Get predictions
            outputs = self.fc(hiddens.squeeze(1))  # [1, vocab_size]

            # Get predicted word (greedy)
            predicted = outputs.argmax(1)  # [1]

            result.append(predicted.item())

            # Stop if end token is generated
            if predicted.item() == vocab.end_idx:
                break

            # Prepare next input (embed predicted word)
            inputs = self.embedding(predicted).unsqueeze(1)  # [1, 1, embed_size]

        return result

    def beam_search(self, features, vocab, beam_width=5, max_length=20, device="cpu"):
        """Generate caption using beam search (higher quality).

        Based on the paper's beam search implementation.

        Args:
            features: Image features [1, embed_size]
            vocab: Vocabulary object
            beam_width: Number of beams (paper uses k=3 or k=5)
            max_length: Maximum caption length
            device: Device to run on

        Returns:
            best_caption: List of word indices for best caption
        """
        # Start with image features
        k = beam_width

        # Initial beam: [(sequence, score, hidden_state, cell_state)]
        sequences = [[vocab.start_idx]]
        scores = [0.0]

        # Initialize LSTM states
        h = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)

        # Process image features through LSTM first
        inputs = features.unsqueeze(1)  # [1, 1, embed_size]
        _, (h, c) = self.lstm(inputs, (h, c))

        # Replicate states for beam
        states = [(h, c)]

        # Generate sequences
        for step in range(max_length):
            all_candidates = []

            for i, seq in enumerate(sequences):
                # Stop if sequence ends with end token
                if seq[-1] == vocab.end_idx:
                    all_candidates.append((seq, scores[i], states[i]))
                    continue

                # Get last word embedding
                last_word = torch.tensor([seq[-1]]).to(device)
                inputs = self.embedding(last_word).unsqueeze(1)  # [1, 1, embed_size]

                # LSTM forward
                hiddens, new_state = self.lstm(inputs, states[i])

                # Get predictions
                outputs = self.fc(hiddens.squeeze(1))  # [1, vocab_size]
                log_probs = torch.log_softmax(outputs, dim=1)

                # Get top k predictions
                top_log_probs, top_indices = log_probs.topk(k, dim=1)

                # Create new candidates
                for j in range(k):
                    candidate_seq = seq + [top_indices[0, j].item()]
                    candidate_score = scores[i] + top_log_probs[0, j].item()
                    all_candidates.append((candidate_seq, candidate_score, new_state))

            # Select top k candidates
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [c[0] for c in ordered[:k]]
            scores = [c[1] for c in ordered[:k]]
            states = [c[2] for c in ordered[:k]]

            # Stop if all sequences end with end token
            if all(seq[-1] == vocab.end_idx for seq in sequences):
                break

        # Return best sequence
        best_idx = scores.index(max(scores))
        return sequences[best_idx]
