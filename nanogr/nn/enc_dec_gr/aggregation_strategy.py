from abc import ABC, abstractmethod
from typing import Optional

import torch


def create_last_k_mask(
    sequence_length: int, last_item_index: torch.Tensor, last_k: Optional[int] = None
) -> torch.tensor:
    """
    Creates a mask to select the last K items of sequences.
    If a sequence has less than K items, all items are considered for the row.
    If last_k is None, all items are considered for all rows.

    Args:
        sequence_length (int): The length of the sequences.
        last_item_index (torch.Tensor) of shape (batch_size,).
            The tensor containing the indices of the last items in the each row
        last_k (Optional[int]): The number of last K items to consider.
            If None, all items are considered.
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, sequence_length) with
            True for the last K items in each row and False for the rest.
    """

    if last_k is None:
        start_index = torch.zeros_like(last_item_index)
    else:
        if last_k < 1:
            raise ValueError("last_k must be None or greater than or equal to 1")
        start_index = torch.clamp(
            last_item_index - last_k + 1, min=0
        )  # Shape (batch_size,)

    indices = (
        torch.arange(sequence_length, device=last_item_index.device)
        .unsqueeze(0)
        .expand(last_item_index.size(0), -1)
    )  # shape (batch_size, sequence_length)

    mask = (indices >= start_index.unsqueeze(1)) & (
        indices <= last_item_index.unsqueeze(1)
    )  # Shape (batch_size, sequence_length)
    return mask


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        pass


class MeanAggregation(AggregationStrategy):
    """
    Aggregates the embeddings by computing their mean. If last_k is specified, only the last K embeddings are considered.
    """

    def __init__(self, last_k: Optional[int] = None):
        """
        Initializes the MeanAggregation class with the specified number of last embeddings to consider.

        Args:
            last_k Optional[int] = None
                The number of last K embeddings to consider for aggregation. If None, all embeddings are considered.
        """
        self.last_k = last_k

    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregates the last K embeddings for each row by computing their mean.

        Args:
            embeddings (torch.Tensor): Shape (batch_size, sequence_length, embedding_dim).
                The tensor containing embeddings for each row.
            row_ids (torch.Tensor): Shape (return_size,).
                The tensor containing row ids for which the aggregated embedding has to be returned.
            last_item_index (torch.Tensor): Shape (return_size,).
                The tensor containing the indices of the last items in emdeddings for each row in row_ids.

        Returns:
            torch.Tensor: The aggregated embeddings of shape (return_size, embedding_dim).
        """
        # Select the embeddings for the specified row ids
        embeddings = embeddings[
            row_ids
        ]  # Shape (return_size, sequence_length, embedding_dim)
        # Create a mask to select the last K items of sequences
        mask = create_last_k_mask(embeddings.size(1), last_item_index, self.last_k)
        mask = mask.to(dtype=embeddings.dtype, device=embeddings.device)

        # Apply the mask to the embeddings
        masked_embeddings = embeddings * mask.unsqueeze(
            2
        )  # Shape (return_size, sequence_length, embedding_dim)

        # Sum the masked embeddings and divide by the count of non-zero elements in the mask
        sum_embeddings = torch.sum(
            masked_embeddings, dim=1
        )  # Shape (return_size, embedding_dim)
        count = (
            torch.sum(mask, dim=1).clamp(min=1).unsqueeze(1)
        )  # Shape (return_size, embedding_dim)
        return sum_embeddings / count  # Shape (return_size, embedding_dim)


class LastAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        return embeddings[row_ids, last_item_index]


class FirstAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        # Return the first item in each sequence, assuming sequences are right-padded
        # TODO(liam): allow all aggregation strategies to handle left padding
        return embeddings[row_ids, 0]
