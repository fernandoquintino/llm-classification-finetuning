import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class CrossEncoderDataset(Dataset):
    """
    A custom Dataset for the cross-encoder. Encodes prompt, response_a,
    and response_b as one input (and swaps the responses with the given
    probability): [CLS] prompt [SEP] response_0 [SEP] response_1 [SEP]
    """
    def __init__(
            self,
            prompts: list[str],
            responses_a: list[str],
            responses_b: list[str],
            labels: np.ndarray,
            tokenizer: PreTrainedTokenizerBase,
            prompt_max_len: int = 30,
            response_max_len: int = 239,
            prob: float = 0.5,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            prompts: A list of strings containing the prompts.
            responses_a: A list of strings containing responses to
                model_a.
            responses_b: A list of strings containing responses to
                model_b.
            labels: A 1-d array of integers containing the preferred
                response:
                    - 0: response_a preferred
                    - 1: response_b preferred
                    - 2: tie
            tokenizer: A tokenizer used to convert text into token ids.
            prompt_max_len: Max tokens reserved for the prompt segment.
            response_max_len: Max tokens reserved for each response
                segment.
            prob: The probability (between 0 and 1) of permuting the
                order of responses.
        """
        self.prompts = prompts
        self.responses_a = responses_a
        self.responses_b = responses_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.response_max_len = response_max_len
        self.prob = prob

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        """
        Builds one cross-encoder sequence: [CLS] prompt [SEP] response_0
        [SEP] response_1 [SEP].

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - input_ids (Tensor): Token ids for the full sequence.
                - attention_mask (Tensor): All 1s - no padding yet.
                - label (Tensor): Integer class label 0/1/2, adjusted if
                    swapped.
                - idx (int): The original index.
        """
        prompt = self.prompts[idx]
        label = self.labels[idx]
        response_a = self.responses_a[idx]
        response_b = self.responses_b[idx]

        if random.random() < self.prob:
            response_a, response_b = response_b, response_a
            if label == 1:
                label = 0
            elif label == 0:
                label = 1
        response_0, response_1 = response_a, response_b

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.prompt_max_len,
        )["input_ids"]
        response_0_ids = self.tokenizer(
            response_0,
            add_special_tokens=False,
            truncation=True,
            max_length=self.response_max_len,
        )["input_ids"]
        response_1_ids = self.tokenizer(
            response_1,
            add_special_tokens=False,
            truncation=True,
            max_length=self.response_max_len
        )["input_ids"]

        input_ids = (
            [self.tokenizer.cls_token_id]
            + prompt_ids
            + [self.tokenizer.sep_token_id]
            + response_0_ids
            + [self.tokenizer.sep_token_id]
            + response_1_ids
            + [self.tokenizer.sep_token_id]
        )

        attention_mask = [1] * len(input_ids)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": label,
            "idx": idx,
        }

def collate_cross_encoder(
        batch: list[dict],
        tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """
    Pads a batch of CrossEncoderDataset samples to the batch's own
    longest sequence, instead of a fixed length.

    Args:
        batch: A list of samples, each from CrossEncoderDataset.
        tokenizer: The tokenizer used to pad input_ids/attention_mask.

    Returns:
        dict: A dictionary containing:
            - input_ids (Tensor): Padded token ids, shape
                (batch_size, max_len).
            - attention_mask (Tensor): Padded attention mask, shape
                (batch_size, max_len).
            - label (Tensor): Class labels, shape (batch_size,).
            - idx (Tensor): Original dataset indices, shape
                (batch_size,).
    """
    labels = torch.stack([sample["label"] for sample in batch])
    idx = torch.tensor(
        [sample["idx"] for sample in batch], dtype=torch.long
    )

    to_pad = [
        {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"]
        }
        for sample in batch
    ]
    padded = tokenizer.pad(to_pad, padding=True, return_tensors="pt")

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "label": labels,
        "idx": idx,
    }

class EncodedPreferenceDataset(Dataset):
    """
    A custom Dataset for classifier-only training. Handles pooled
    encoded input prompts/response_0/response_1, and preference labels.
    """
    def __init__(
            self,
            encoded: torch.Tensor,
            labels: torch.Tensor,
            indices: np.ndarray,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            encoded: The pooled [CLS] embedding from the encoder's last
                hidden state, summarizing the encoded:
                [CLS] prompt [SEP] response_0 [SEP] response_1.
            labels: A tensor of integers containing the preferred
                response.
                - 0: response_0 preferred
                - 1: response_1 preferred
                - 2: tie
            indices: Array of integer row positions into the full
                precomputed tensors, e.g. train_df.index.values.
        """
        self.indices = indices
        self.encoded= encoded
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves one sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - encoded (Tensor): The pooled [CLS] embedding from the
                    encoder's last hidden state.
                - label (Tensor): Integer class label 0/1/2.
                - idx (int): The index of the sample.
        """
        row = self.indices[idx]
        encoded = self.encoded[row]
        label = self.labels[row]

        return {
            "encoded": encoded,
            "idx": idx,
            "label": label,
        }