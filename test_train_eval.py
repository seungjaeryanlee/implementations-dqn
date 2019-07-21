"""Test train_eval.get_linear_anneal_func."""
import types

from train_eval import get_linear_anneal_func


class TestGetLinearAnnealFunc:
    def test_return_type(self):
        """Test the return type of get_linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(1, 0, 100)
        assert isinstance(linear_anneal_func, types.FunctionType)
