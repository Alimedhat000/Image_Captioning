"""Vocabulary for caption encoding/decoding with spaCy tokenizer."""

from collections import Counter
import pickle
import os
import spacy


class Vocabulary:
    """Vocabulary class for mapping words to indices and vice versa."""

    # Special tokens
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<sos>"
    END_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, freq_threshold=5, spacy_model="en_core_web_sm", use_spacy=True):
        """Initialize vocabulary.

        Args:
            freq_threshold: Minimum frequency for word to be included in vocab
            spacy_model: spaCy model to use for tokenization
            use_spacy: Whether to use spaCy tokenizer (False for legacy whitespace splitting)
        """
        self.freq_threshold = freq_threshold
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model

        # Load spaCy tokenizer only if needed
        if use_spacy:
            try:
                self.nlp = spacy.load(
                    spacy_model, disable=["parser", "ner", "lemmatizer"]
                )
            except OSError:
                print(f"Downloading spaCy model '{spacy_model}'...")
                spacy.cli.download(spacy_model)
                self.nlp = spacy.load(
                    spacy_model, disable=["parser", "ner", "lemmatizer"]
                )
        else:
            self.nlp = None

        # Initialize mappings with special tokens
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.START_TOKEN: 1,
            self.END_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_count = Counter()

    def tokenize(self, text):
        """Tokenize text using spaCy or whitespace splitting.

        Args:
            text: Input text string

        Returns:
            List of tokens (as strings)
        """
        if self.use_spacy:
            doc = self.nlp(text)
            return [token.text.lower() for token in doc]
        else:
            # Legacy whitespace tokenization
            return text.lower().split()

    def build_vocabulary(self, captions):
        """Build vocabulary from list of captions.

        Args:
            captions: List of caption strings
        """
        # Count word frequencies using spaCy tokenization
        for caption in captions:
            # Remove special tokens before tokenizing
            clean_caption = caption.replace("<sos>", "").replace("<eos>", "").strip()
            tokens = self.tokenize(clean_caption)
            self.word_count.update(tokens)

        # Add words that meet frequency threshold
        idx = len(self.word2idx)  # Start after special tokens
        for word, count in self.word_count.items():
            if count >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def save(self, filepath):
        """Save vocabulary to a pickle file.

        Args:
            filepath: Path to save the pickle file
        """
        # Don't pickle the spaCy model object, just save the necessary data
        save_dict = {
            "freq_threshold": self.freq_threshold,
            "use_spacy": self.use_spacy,
            "spacy_model": self.spacy_model,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "word_count": self.word_count,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(
            filepath
        ) else None

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load vocabulary from a pickle file.

        Args:
            filepath: Path to the pickle file

        Returns:
            Vocabulary object
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        # Create a new vocabulary instance
        vocab = cls(
            freq_threshold=save_dict["freq_threshold"],
            spacy_model=save_dict["spacy_model"],
            use_spacy=save_dict["use_spacy"],
        )

        # Restore the saved state
        vocab.word2idx = save_dict["word2idx"]
        vocab.idx2word = save_dict["idx2word"]
        vocab.word_count = save_dict["word_count"]

        print(f"Vocabulary loaded from {filepath}")
        return vocab

    def encode(self, text):
        """Convert text to list of indices.

        Args:
            text: Input text (may contain <sos> and <eos> tokens)

        Returns:
            List of token indices
        """
        tokens = []

        result = []
        if text.startswith(self.START_TOKEN):
            result.append(self.word2idx[self.START_TOKEN])
            text = text[len(self.START_TOKEN) :].strip()  # Remove unwanted whitespaces

        if text.endswith(self.END_TOKEN):
            end_token = True
            text = text[: -len(self.END_TOKEN)].strip()
        else:
            end_token = False

        # Tokenize the main text
        tokens = self.tokenize(text)
        result.extend(
            [
                self.word2idx.get(
                    token, self.word2idx[self.UNK_TOKEN]
                )  # fall back to <unk>
                for token in tokens
            ]
        )

        if end_token:
            result.append(self.word2idx[self.END_TOKEN])

        return result

    def decode(self, indices, skip_special_tokens=True):
        """Convert indices back to text."""
        # Force conversion to list
        if hasattr(indices, "tolist"):
            indices = indices.tolist()

        # If indices is a nested list (e.g., [[1, 2, 3]]), flatten it
        if len(indices) > 0 and isinstance(indices[0], list):
            indices = indices[0]

        words = []
        special_token_indices = {self.start_idx, self.end_idx, self.pad_idx}

        for idx in indices:
            # Handle potential tensor/list items
            curr_idx = int(idx)
            if skip_special_tokens and curr_idx in special_token_indices:
                continue
            words.append(self.idx2word.get(curr_idx, self.UNK_TOKEN))

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
