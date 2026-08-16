import torch
from fl_code.fed_core.params import state_dict_to_tensors, tensors_to_state_dict


def test_roundtrip():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    state = model.state_dict()
    keys = list(state.keys())
    tensors = state_dict_to_tensors(state)
    assert len(tensors) == len(keys)
    restored = tensors_to_state_dict(tensors, keys)
    assert list(restored.keys()) == keys
    for k in keys:
        assert torch.equal(restored[k], state[k])


def test_dtype_is_float32():
    model = torch.nn.Linear(2, 2)
    t = state_dict_to_tensors(model.state_dict())[0]
    assert t.dtype.name == "float32"
