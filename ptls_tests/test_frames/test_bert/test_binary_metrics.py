import pytest
import torch
import torchmetrics

from ptls.frames.bert import RtdModule, SopNspModule


class DummySeqEncoder(torch.nn.Module):
    embedding_size = 4
    is_reduce_sequence = True


def make_module(module_type, **kwargs):
    if module_type is SopNspModule:
        kwargs.update(hidden_size=4, drop_p=0.0)
    return module_type(seq_encoder=DummySeqEncoder(), **kwargs)


@pytest.mark.parametrize("module_type", [RtdModule, SopNspModule])
def test_default_auroc_scores_binary_predictions(module_type):
    metric = make_module(module_type)._validation_metric
    predictions = torch.tensor([0.1, 0.4, 0.35, 0.8])
    targets = torch.tensor([0, 0, 1, 1])
    assert metric(predictions, targets).item() == pytest.approx(0.75)


@pytest.mark.parametrize("module_type", [RtdModule, SopNspModule])
def test_custom_metric_is_preserved(module_type):
    try:
        metric = torchmetrics.AUROC(task="binary")
    except TypeError:
        metric = torchmetrics.AUROC(num_classes=2)
    assert make_module(module_type, validation_metric=metric)._validation_metric is metric
