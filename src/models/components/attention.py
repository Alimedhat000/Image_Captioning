import torch
from torch import nn
import torch.nn.functional as F


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

        # Add one dim to be able to add with the encoder_att
        att2 = att2.unsqueeze(1)  # [batch_size, 1, attention_dim]

        # Compute attention scores using tanh
        att = self.full_att(torch.tanh(att1 + att2))  # [batch_size, num_pixels, 1]
        att = att.squeeze(2)  # [batch_size, num_pixels]

        # Softmax to get attention weights
        alpha = F.softmax(att, dim=1)  # [batch_size, num_pixels]

        # Compute weighted sum of encoder features
        context = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # [batch_size, encoder_dim]

        return context, alpha
