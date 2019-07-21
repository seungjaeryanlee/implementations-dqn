"""Test train_eval.get_linear_anneal_func."""
import types

from train_eval import get_linear_anneal_func


class TestGetLinearAnnealFunc:
    def test_return_type(self):
        """Test the return type of get_linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(1, 0, 100)
        assert isinstance(linear_anneal_func, types.FunctionType)

    def test_return_func_return_type(self):
        """Test the return type of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(1, 0, 100)
        value = linear_anneal_func(0)
        assert isinstance(value, float)

    def test_return_func_start_value(self):
        """Test the start value of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(1, 0, 100)
        value = linear_anneal_func(0)
        assert value == 1

    def test_return_func_mid_value(self):
        """Test the middle value of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(1, 0, 100)
        value = linear_anneal_func(50)
        assert value == 0.5

    def test_return_func_end_value(self):
        """Test the end value of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(1, 0, 100)
        value = linear_anneal_func(100)
        assert value == 0

    def test_return_func_after_end_value(self):
        """Test the value of linear_anneal_func after end_steps."""
        linear_anneal_func = get_linear_anneal_func(1, 0, 100)
        value = linear_anneal_func(101)
        assert value == 0
