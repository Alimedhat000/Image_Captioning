"""Image Captioner model."""

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.models.base_module import BaseModule


class ImageCaptioner(BaseModule):
    """Image Captioner using encoder-decoder architecture.

    This combines a CNN encoder (e.g., ResNet50) with an LSTM decoder
    for generating image captions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: Optimizer,
        scheduler: Any = None,
        pad_idx: int = 0,
        name: str = "image_captioner",
        **kwargs,
    ):
        """Initialize Image Captioner.

        Args:
            encoder: CNN encoder network (ResNet50, etc.)
            decoder: LSTM/GRU decoder network
            optimizer: Optimizer (will be instantiated by Hydra)
            scheduler: Optional learning rate scheduler
            pad_idx: Padding token index (default: 0)
            name: Model name for logging
        """
        super().__init__()

        # Store components
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.model_name = name

        # Create a net property for compatibility with your trainer's model summary
        self.net = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})

        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    @property
    def vocab_size(self):
        """Get vocabulary size from decoder."""
        return self.decoder.vocab_size

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            images: Input images [batch_size, 3, 224, 224]
            captions: Caption indices [batch_size, max_length] (optional, for training)
            lengths: Caption lengths [batch_size] (optional, for training)

        Returns:
            If training: predictions [batch_size, max_length, vocab_size]
            If inference: image features [batch_size, embed_size]
        """
        # Extract image features
        features = self.encoder(images)

        # If captions provided, use teacher forcing (training mode)
        if captions is not None and lengths is not None:
            outputs = self.decoder(features, captions, lengths)

            if isinstance(outputs, tuple):
                outputs, alphas = outputs  # Attention decoder
                return outputs, alphas
            else:
                return outputs  # Regular decoder

        # Otherwise, return features for generation (inference mode)
        return features

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training step.

        Args:
            batch: Tuple of (images, captions, lengths)
            batch_idx: Batch index

        Returns:
            Tuple of (loss, metrics_dict)
        """
        images, captions, lengths = batch

        # Forward pass
        # Input: captions without <end>, Target: captions without <start>
        outputs = self(images, captions[:, :-1], lengths - 1)

        # handle attention
        if isinstance(outputs, tuple):
            outputs, alphas = outputs
        else:
            outputs = outputs
        # Reshape for loss calculation
        # outputs: [batch_size, max_length, vocab_size]
        # targets: [batch_size, max_length]
        targets = captions[:, 1:]  # Remove <start> token

        # Flatten for CrossEntropyLoss
        # outputs: [batch_size * max_length, vocab_size]
        # targets: [batch_size * max_length]
        outputs_flat = outputs[:, : targets.shape[1], :].reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        # Calculate loss
        loss = self.criterion(outputs_flat, targets_flat)

        # Calculate perplexity
        perplexity = torch.exp(loss)

        metrics = {
            "train/loss": loss.detach(),
            "train/perplexity": perplexity.detach(),
        }

        return loss, metrics

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Tuple of (images, captions, lengths)
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        images, captions, lengths = batch

        # Forward pass
        outputs = self(images, captions[:, :-1], lengths - 1)

        if isinstance(outputs, tuple):
            outputs, alphas = outputs
        else:
            outputs = outputs

        # Prepare targets
        targets = captions[:, 1:]

        # Flatten for loss
        outputs_flat = outputs[:, : targets.shape[1], :].reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        # Calculate loss
        loss = self.criterion(outputs_flat, targets_flat)

        # Calculate perplexity exp of cross entropy
        perplexity = torch.exp(loss)

        return {
            "val/loss": loss,
            "val/perplexity": perplexity,
        }

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step.

        Args:
            batch: Tuple of (images, captions, lengths)
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        images, captions, lengths = batch

        # Forward pass
        outputs = self(images, captions[:, :-1], lengths - 1)

        if isinstance(outputs, tuple):
            outputs, alphas = outputs  # Attention decoder

        # Prepare targets
        targets = captions[:, 1:]

        # Flatten for loss
        outputs_flat = outputs[:, : targets.shape[1], :].reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        # Calculate loss
        loss = self.criterion(outputs_flat, targets_flat)

        # Calculate perplexity
        perplexity = torch.exp(loss)

        return {
            "test/loss": loss,
            "test/perplexity": perplexity,
        }

    def generate_caption(
        self, images: torch.Tensor, vocab, max_length: int = 200, method: str = "greedy"
    ):
        """Generate captions for images (inference).

        Args:
            images: Input images [batch_size, 3, 224, 224]
            vocab: Vocabulary object
            max_length: Maximum caption length
            method: "greedy" or "beam" search

        Returns:
            List of generated captions (as word indices)
        """
        self.eval()
        with torch.no_grad():
            # Extract features
            features = self.encoder(images)

            # Generate captions
            if method == "greedy":
                captions = [
                    self.decoder.generate_caption(
                        feat.unsqueeze(0), vocab, max_length, device=images.device
                    )
                    for feat in features
                ]
            elif method == "beam":
                captions = [
                    self.decoder.beam_search(
                        feat.unsqueeze(0),
                        vocab,
                        beam_width=5,
                        max_length=max_length,
                        device=images.device,
                    )
                    for feat in features
                ]
            else:
                raise ValueError(f"Unknown method: {method}")

            return captions

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Instantiate optimizer
        optimizer = self.optimizer_config(params=self.parameters())

        # Optionally instantiate scheduler
        if self.scheduler_config is not None:
            scheduler = self.scheduler_config(optimizer=optimizer)
            return optimizer, scheduler

        return optimizer
