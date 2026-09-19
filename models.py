import torch
import torch.nn as nn


class DeepPreferenceClassifier(nn.Module):
    """
    Classification head that takes pooled encoder representation and
    predicts which response is preferred:
        - 0 response_0 preferred
        - 1 response_1 preferred
        - 2 tie
    """

    def __init__(
            self,
            input_dim: int,
            hidden_state: int = 768,
            num_classes: int = 3,
            num_blocks: int = 2,
            dropout: float = 0.1,
    ) -> None:
        """
        Initializes the classifier.

        Args:
            input_dim: The dimension of the pooled input vector.
            hidden_state: The hidden dimension for the residual block.
            num_classes: The number of classes.
            num_blocks: The number of residual blocks.
            dropout: Dropout probability used throughout the head.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_state)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_state, hidden_state),
                nn.LayerNorm(hidden_state),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_blocks)
        ])

        self.fc2 = nn.Linear(hidden_state, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass to generate classification logits.

        Args:
            x: A pooled representation, shape (batch_size, input_dim).

        Returns:
            Tensor: Classification logits.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = x + block(x)

        logits = self.fc2(x)

        return logits


class PreferenceModel(nn.Module):
    """
    A model that takes the encoder model and the classifier
    model to create the full end-to-end model. The model first encodes
    prompt/response_0/response_1 through the encoder model in one
    pass. The output is fed through the classifier model to predict the
    user preference:
        - 0: response_0 is preferred
        - 1: response_1 is preferred
        - 2: tie
    """

    def __init__(
            self,
            model_encoder: nn.Module,
            model_classifier: DeepPreferenceClassifier,
    ) -> None:
        """
        Initializes the model with the model encoder and model
        classifier.

        Args:
            model_encoder (nn.Module): A pretrained encoder model
                (e.g. DebertaV2Model) that outputs last_hidden_state
                given input_ids and attention_mask.
            model_classifier (DeepPreferenceClassifier): The
                classification head that takes encoder outputs and
                predicts preference.
        """
        super().__init__()
        self.model_encoder = model_encoder
        self.model_classifier = model_classifier

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes the forward pass of the model.

        Args:
            input_ids: The input_ids.
            attention_mask: The attention mask for the input_ids.

        Returns:
            Tensor: Classifier logits.
        """
        output = self.model_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:,0,:]
        logits = self.model_classifier(output)

        return logits