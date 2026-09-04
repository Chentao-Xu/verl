import torch
import torch.nn as nn

from verl.utils.hif8_train import linear as hif8_linear


def test_hif8_linear_three_gemm_backward(monkeypatch):
    calls = []

    def fake_qdq(x, dim):
        calls.append((tuple(x.shape), dim))
        return x

    monkeypatch.setattr(hif8_linear, "_quant_dequant", fake_qdq)
    layer = hif8_linear.HIF8TrainLinear(4, 3, bias=True)
    x = torch.randn(2, 5, 4, requires_grad=True)
    expected = nn.Linear(4, 3, bias=True)
    expected.load_state_dict(layer.state_dict())
    expected_x = x.detach().clone().requires_grad_(True)

    layer(x).sum().backward()
    expected(expected_x).sum().backward()

    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(layer.weight.grad, expected.weight.grad)
    torch.testing.assert_close(layer.bias.grad, expected.bias.grad)
    assert [dim for _, dim in calls] == [-1, 0, -1, 0, -1, 0]


class _TinyPangu(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4)) for _ in range(3)])
        self.visual = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)


def test_apply_hif8_train_sim_preserves_parameters_and_scope():
    model = _TinyPangu()
    original_weight = model.model.language_model.layers[1][0].weight
    original_keys = list(model.state_dict())

    converted = hif8_linear.apply_hif8_train_sim(model, 1, 2)

    assert converted == 2
    assert type(model.model.language_model.layers[0][0]) is nn.Linear
    assert isinstance(model.model.language_model.layers[1][0], hif8_linear.HIF8TrainLinear)
    assert isinstance(model.model.language_model.layers[2][0], hif8_linear.HIF8TrainLinear)
    assert model.model.language_model.layers[1][0].weight is original_weight
    assert type(model.visual) is nn.Linear
    assert type(model.lm_head) is nn.Linear
    assert list(model.state_dict()) == original_keys
