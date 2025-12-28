"""Flickr8K DataModule."""

from typing import Optional
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import re

from src.data.datamodule import DataModule
from src.data.vocabulary import Vocabulary


class Flickr8kDataset(Dataset):
    """Flickr8k Dataset."""

    def __init__(self, df, images_dir, vocab, transform=None):
        """Initialize dataset.

        Args:
            df: DataFrame with 'image' and 'clean_caption' columns
            images_dir: Directory containing images
            vocab: Vocabulary object
            transform: Image transforms
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load and transform image
        img_path = os.path.join(self.images_dir, row["image"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Encode caption
        caption = row["clean_captions"]
        caption_encoded = self.vocab.encode(caption)

        return img, torch.tensor(caption_encoded, dtype=torch.long)


class Flickr8kDataModule(DataModule):
    """DataModule for Flickr8k dataset."""

    def __init__(
        self,
        data_dir: str = "data/flickr8k",
        batch_size: int = 32,
        num_workers: int = 12,
        pin_memory: bool = True,
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
        freq_threshold: int = 0,
        img_size: int = 224,
        use_spacy: bool = True,
        **kwargs,
    ):
        """Initialize MNIST DataModule.

        Args:
            data_dir: Root directory for data
            batch_size: Batch size
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory
            train_val_test_split: Train/val/test split (must sum to 1)
            freq_threshold: Vocabulary minimum frequency
            img_size: Image size to feed into the model
            use_spacey: use the spaCy when building the vocab for better tokenization
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Make sure split sums up to 1
        assert sum(train_val_test_split) == 1.0, "Split Ratios Must sum to 1.0"

        self.train_val_test_split = train_val_test_split
        self.freq_threshold = freq_threshold
        self.img_size = img_size

        self.images_dir = os.path.join(data_dir, "Images")
        self.captions_file = os.path.join(data_dir, "captions.txt")

        self.vocab = None
        self.train_dataset = None
        self.test_dataset = None

        self.use_spacy = use_spacy

        # Image transforms
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size + 32, self.img_size + 32)
                ),  # resize to a bit bigger to crop allow getting the random crops better
                transforms.RandomCrop((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self):
        # Check if datasets are downloaded
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Images directory not found at {self.images_dir}. "
                f"download dataset and place it in {self.data_dir}"
            )

        if not os.path.exists(self.captions_file):
            raise FileNotFoundError(
                f"captions file not found at {self.captions_file}. "
                f"ensure captions.txt is in {self.data_dir}"
            )

    def _preprocess_captions(self, df):
        def clean_caption(text):
            text = text.lower()

            if not self.use_spacy:
                text = re.sub(r"[^a-z\s]", " ", text)
            text = " ".join(text.split())  # Normalize multiple spaces to single space
            return text

        df["clean_captions"] = df["caption"].apply(clean_caption)
        df["clean_captions"] = "<sos> " + df["clean_captions"] + " <eos>"
        return df

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""

        df = pd.read_csv(self.captions_file)

        df = self._preprocess_captions(df)

        unique_images = df["image"].unique()

        train_ratio, val_ratio, test_ratio = self.train_val_test_split

        train_imgs, temp_imgs = train_test_split(
            unique_images,
            test_size=(val_ratio + test_ratio),
            random_state=42,
        )

        # Second split: val vs test
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42,
        )

        # Create a Dataframe for each split
        train_df = df[df["image"].isin(train_imgs)].reset_index(drop=True)
        val_df = df[df["image"].isin(val_imgs)].reset_index(drop=True)
        test_df = df[df["image"].isin(test_imgs)].reset_index(drop=True)

        # Build Vocab from training captions to ensure realistic evaluation
        vocab_path = "vocab.pkl"

        if os.path.exists(vocab_path):
            print("Loading existing vocabulary...")
            self.vocab = Vocabulary.load(vocab_path)
        else:
            print("Building vocabulary from scratch...")
            self.vocab = Vocabulary(self.freq_threshold, use_spacy=self.use_spacy)
            self.vocab.build_vocabulary(train_df["clean_captions"].to_list())
            self.vocab.save(vocab_path)

        # Finaly Create Datasets
        if stage == "fit" or stage is None:
            self.train_dataset = Flickr8kDataset(
                df=train_df,
                images_dir=self.images_dir,
                vocab=self.vocab,
                transform=self.train_transform,
            )

            self.val_dataset = Flickr8kDataset(
                df=val_df,
                images_dir=self.images_dir,
                vocab=self.vocab,
                transform=self.test_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = Flickr8kDataset(
                df=test_df,
                images_dir=self.images_dir,
                vocab=self.vocab,
                transform=self.test_transform,
            )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to pad captions to same length.

        Args:
            batch: List of (image, caption) tuples

        Returns:
            Tuple of (images, padded_captions, lengths)
        """
        # Sort batch by caption length (descending) for packed sequences
        batch.sort(key=lambda x: len(x[1]), reverse=True)

        images, captions = zip(*batch)

        # Stack images
        images = torch.stack(images, 0)

        # Get lengths
        lengths = torch.tensor([len(cap) for cap in captions])

        # Pad captions
        padded_captions = torch.zeros(len(captions), max(lengths), dtype=torch.long)
        for i, cap in enumerate(captions):
            end = lengths[i]
            padded_captions[i, :end] = cap

        return images, padded_captions, lengths

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
