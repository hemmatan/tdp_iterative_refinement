import datetime
from transformers.models.bert import BertModel
from tdp import constants


def _current_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def _initialize_encoder(encoder_type: str):
    if encoder_type != 'BERT':
        raise ValueError("The paper configurations require BERT-base-cased.")
    encoder = BertModel.from_pretrained("bert-base-cased")
    constants.LOGGER.info("Using BERT encoder")
    return encoder
