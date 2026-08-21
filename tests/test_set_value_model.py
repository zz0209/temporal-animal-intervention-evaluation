import torch

from animal_intervention.models import DeepSetValueModel


def test_deep_set_value_model_is_permutation_invariant() -> None:
    torch.manual_seed(7)
    model = DeepSetValueModel(member_features=3, context_features=2, hidden_features=8)
    members = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    context = torch.tensor([[0.2, 0.8]])

    original = model(members, mask, context)
    permuted = model(members[:, [1, 0, 2]], mask[:, [1, 0, 2]], context)

    torch.testing.assert_close(original, permuted)


def test_deep_set_value_model_ignores_padding_values() -> None:
    torch.manual_seed(11)
    model = DeepSetValueModel(member_features=2, context_features=1, hidden_features=4)
    members = torch.tensor([[[1.0, 2.0], [99.0, -99.0]]])
    mask = torch.tensor([[True, False]])
    context = torch.tensor([[0.5]])
    changed_padding = members.clone()
    changed_padding[0, 1] = torch.tensor([-4.0, 8.0])

    torch.testing.assert_close(
        model(members, mask, context),
        model(changed_padding, mask, context),
    )
