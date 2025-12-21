"""Vocabulary for caption encoding/decoding."""

from collections import Counter


class Vocabulary:
    """Vocabulary class for mapping words to indices and vice versa."""

    # Special tokens
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<sos>"
    END_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, freq_threshold=5):
        """Initialize vocabulary.

        Args:
            freq_threshold: Minimum frequency for word to be included in vocab
        """
        self.freq_threshold = freq_threshold

        # Initialize mappings with special tokens
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.START_TOKEN: 1,
            self.END_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_count = Counter()

    def build_vocabulary(self, captions):
        """Build vocabulary from list of captions.

        Args:
            captions: List of caption strings
        """
        # Count word frequencies
        for caption in captions:
            self.word_count.update(caption.split())

        # Add words that meet frequency threshold
        idx = len(self.word2idx)  # Start after special tokens
        for word, count in self.word_count.items():
            if count >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text):
        """Convert text to list of indices."""
        return [
            self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
            for word in text.split()
        ]

    def decode(self, indices, skip_special_tokens=True):
        """Convert indices back to text."""
        # Convert tensor to list if needed
        if hasattr(indices, "tolist"):
            indices = indices.tolist()

        words = []
        special_token_indices = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.START_TOKEN],
            self.word2idx[self.END_TOKEN],
        }

        for idx in indices:
            if skip_special_tokens and idx in special_token_indices:
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))

        return " ".join(words)

    def __len__(self):
        """Return vocabulary size."""
        return len(self.word2idx)

    def __getitem__(self, item):
        """Allow dictionary-style access."""
        if isinstance(item, str):
            return self.word2idx.get(item, self.word2idx[self.UNK_TOKEN])
        elif isinstance(item, int):
            return self.idx2word.get(item, self.UNK_TOKEN)
        else:
            raise TypeError(f"Key must be str or int, not {type(item)}")

    @property
    def pad_idx(self):
        """Return padding token index."""
        return self.word2idx[self.PAD_TOKEN]

    @property
    def start_idx(self):
        """Return start token index."""
        return self.word2idx[self.START_TOKEN]

    @property
    def end_idx(self):
        """Return end token index."""
        return self.word2idx[self.END_TOKEN]

    @property
    def unk_idx(self):
        """Return unknown token index."""
        return self.word2idx[self.UNK_TOKEN]
