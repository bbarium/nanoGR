from typing import Optional, List, Union, Any, Dict

import torchmetrics
import torch
import torch.nn as nn
from torchmetrics.utilities.distributed import gather_all_tensors

from aggregation_strategy import AggregationStrategy


class EmbeddingAggregator(nn.Module):
    """Embedding aggregator function. this determins how user embeddings are aggregated to form the final user embedding.

    Parameters
    ----------
    aggregation_type: str
        aggregation function type
    """

    def __init__(
        self,
        aggregation_strategy: AggregationStrategy,
    ):
        super(EmbeddingAggregator, self).__init__()
        self.aggregation_strategy = aggregation_strategy

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # embeddings: (batch_size, sequence_length, embedding_dim)
        # attention_mask: (batch_size, sequence_length)

        # we -1 here because the token index starts from 0
        last_item_index = attention_mask.sum(dim=1) - 1

        # The following 3 steps are equivalent to
        # row_ids = torch.arange(embeddings.size(0))
        # but in a way that is traceable with Fx.

        # 1. Create a dummy tensor with the same batch shape as attention_mask
        dummy_tensor_for_batch_shape = attention_mask[:, 0]  # Shape (batch_size,)

        # 2. Use torch.ones_like to create a tensor of ones with that shape.
        # Note that torch.ones is not traceable in Fx, so we use torch.ones_like.
        ones_tensor = torch.ones_like(dummy_tensor_for_batch_shape, dtype=torch.long)

        # 3. Use cumsum to get the 0 to batch_size - 1 sequence
        row_ids = torch.cumsum(ones_tensor, dim=0) - 1

        return self.aggregation_strategy.aggregate(embeddings, row_ids, last_item_index)
    

class MLP(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim_list: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): The dimensionality of the input tensor.
            output_dim (int): The dimensionality of the output tensor.
            hidden_dim_list Optional(List[int]): A list of the dimensions of each hidden
                layer output. The number of layers in the MLP is the length of this list
                plus one.
            activation (nn.Module): The activation function to use between layers.
            bias (bool): Whether to include bias terms in the linear layers.
            dropout (float): The dropout rate to apply after each layer.
        """
        super().__init__()

        if hidden_dim_list is None:
            hidden_dim_list = []
        hidden_dim_list.append(output_dim)
        layers = [nn.Linear(input_dim, hidden_dim_list[0], bias=bias)]
        for i in range(1, len(hidden_dim_list)):
            layers.append(activation())
            layers.append(
                nn.Linear(hidden_dim_list[i - 1], hidden_dim_list[i], bias=bias)
            )
            layers.append(nn.Dropout(dropout))
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CustomRetrievalMetric(torchmetrics.Metric):
    """
    Custom retrieval metric class to calculate ranking metrics.
    """

    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric_values = 0
        self.total_values = 0
        self.top_k = top_k
    
    def compute(self) -> torch.Tensor:
        # Aggregates the metric accross workers and returns the final value
        metric_values_tensor = torch.tensor(self.metric_values).to(self.device)
        total_values_tensor = torch.tensor(self.total_values).to(self.device)
        # Compute final metric
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.device)
        # Checks if using more than one GPU
        # If so, gather all metric values and total values from all GPUs. Else, return the current
        # worker's metric value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather all metric values and total values from all GPUs

            metric_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(metric_values_tensor)
            ]
            metric_values_tensor = torch.cat(metric_values_tensor_list).sum()

            total_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(total_values_tensor)
            ]

            total_values_tensor = torch.cat(total_values_tensor_list).sum()

        return metric_values_tensor / total_values_tensor

    def reset(self) -> None:
        self.metric_values = 0
        self.total_values = 0

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:

        batch_size = int(len(indexes) / (indexes == 0).sum().item())
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1).int()

        metric = self._metric(preds, target)
        self.metric_values += metric.sum().item()
        self.total_values += batch_size

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RetrievalEvaluator:
    """
    Wrapper for retrieval evaluation metrics.
    It takes model outputs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        should_sample_negatives_from_vocab: bool = True,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        if self.should_sample_negatives_from_vocab:
            inbatch_negatives = self.sample_negative_ids_from_vocab(
                num_of_samples=num_of_samples,
                num_of_candidates=num_of_candidates,
                num_negatives=self.num_negatives,
            )
            # we +1 here because we need to include the positive sample
            num_of_candidates = self.num_negatives + 1
            pos_embeddings = key_embeddings[labels]
            key_embeddings = key_embeddings[inbatch_negatives]
            # key_embeddings shape: (bsz, num_negatives+1, emb_dim)
            key_embeddings = torch.cat(
                [pos_embeddings.unsqueeze(1), key_embeddings], dim=1
            )
            # the positive index will always be 0 because the pos embedding will always be the first one.
            labels = torch.zeros(num_of_samples).long()

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    # this method samples random negative samples from the whole vocab
    def sample_negative_ids_from_vocab(
        self,
        num_of_samples: int,
        num_of_candidates: int,
        num_negatives: int,
    ) -> torch.Tensor:
        # num_of_samples: batch size
        # num_of_candidates: number of total vocabs
        # num_negatives: number of negative samples

        # we do randint to accelerate the negative sampling
        # this could have collision with positive pairs but the chance is very low

        # TODO (Clark): in the future we might need to have non-collision negative sampling
        # when K in top-k is very small (e.g., hits@1) and num_negatives is very large
        negative_candidates = torch.randint(
            self.placeholder_token_buffer,
            num_of_candidates,
            (num_of_samples, num_negatives),
        )

        return negative_candidates


class ModelOutput:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def list_of_row_format(self):
        """
        Function used to convert the predictions into a format that can be written to BigQuery.
        """
        raise NotImplementedError

    def _convert_to_list(self, prediction: Union[torch.Tensor, List]) -> List:
        """
        Convert the prediction to a list so it can be serialized.
        """
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().tolist()

        return prediction


class SharedKeyAcrossPredictionsOutput(ModelOutput):
    """
    A class to represent the output of a model with a single key for all predictions in a batch.

        Attributes:
        key: The single key associated with all the predictions.
        predictions: The predictions made by the model.
        key_name (str): The name of the key attribute. Default is "idx".
        prediction_name (str): The name of the prediction attribute. Default is "prediction".
    """

    def __init__(
        self,
        key,
        predictions,
        key_name: str = "idx",
        prediction_name: str = "prediction",
    ):
        self.key = key
        self.predictions = predictions
        self.key_name = key_name
        self.prediction_name = prediction_name

    @property
    def list_of_row_format(self):
        return [
            {self.key_name: self.key, self.prediction_name: pred}
            for pred in self._convert_to_list(self.predictions)
        ]


class OneKeyPerPredictionOutput(ModelOutput):
    """
    A class used to represent the output of a model where each prediction is associated with a unique key.
    Attributes
    ----------
    keys : Any
        The keys associated with each prediction.
    predictions : Any
        The predictions made by the model.
    key_name : str, optional
        The name to be used for the key in the output dictionary (default is "idx").
    prediction_name : str, optional
        The name to be used for the prediction in the output dictionary (default is "prediction").
    """

    def __init__(
        self,
        keys,
        predictions,
        key_name: str = "idx",
        prediction_name: str = "prediction",
    ):
        self.keys = keys
        self.predictions = predictions
        self.key_name = key_name
        self.prediction_name = prediction_name

    @property
    def list_of_row_format(self):
        return [
            {self.key_name: key, self.prediction_name: pred}
            for key, pred in zip(
                self._convert_to_list(self.keys),
                self._convert_to_list(self.predictions),
            )
        ]