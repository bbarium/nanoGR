from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.configuration_t5 import T5Config

from nanogr.utils import Config
from .plain_decoder import TransformerDecoder, PlainDecoderLayer


def init_decoder(backbone: str, cfg: Config):
    """Build the decoder backbone used by ``SemanticIDEncoderDecoder``.

    Args:
        backbone: Name of the backbone implementation. Currently only ``'t5'``
            is supported.
        config: A mapping (``dict`` or attribute-style namespace) carrying the
            following fields: ``vocab_size``, ``d_model``, ``d_kv``,
            ``num_heads``, ``dropout``, ``dim_feedforward``, ``num_layers``.
    """
    if backbone == 't5':
        t5_decoder_config = T5Config(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            d_kv=cfg.d_kv,
            d_ff=cfg.dim_feedforward,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout,
            num_layers=cfg.num_layers,
            is_decoder=True,
            is_encoder_decoder=False,
        )
        decoder = T5Stack(t5_decoder_config)
    elif backbone == 'plain':
        decoder_layer = PlainDecoderLayer(
            d_model=cfg.d_model, 
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward, 
            dropout=cfg.dropout,
            batch_first=True,
        )
        decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=cfg.num_layers,
        )
    else:
        raise NotImplementedError()

    return decoder