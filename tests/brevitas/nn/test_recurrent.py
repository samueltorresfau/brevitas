# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from hypothesis import given
import hypothesis.strategies as st
import pytest
import torch

from brevitas.nn import QuantLSTM
from brevitas.nn import QuantRNN
from brevitas.nn import QuantGRU
from brevitas.quant_tensor import QuantTensor
from tests.brevitas.hyp_helper import float_tensor_random_size_st

ATOL = 1e-6

SEQ, BATCH, FEAT, HIDDEN = 5, 3, 8, 11

# -----------------------------------------------------------
# helper to build random input matching batch_first toggle
# -----------------------------------------------------------
def _make_input(seq_len, batch_size, input_size, *, batch_first):
    shape = (batch_size, seq_len, input_size) if batch_first else (seq_len, batch_size, input_size)
    return torch.randn(*shape)
# Simmilar helper for the test cases
def _inp(batch_first: bool):
    shape = (BATCH, SEQ, FEAT) if batch_first else (SEQ, BATCH, FEAT)
    return torch.randn(*shape)

# ---------------------------------------------------------------------------
# helper to copy the PyTorch weights & biases into the three-gate layout
# ---------------------------------------------------------------------------
def _copy_pt_to_quant(pt_gru: torch.nn.GRU, q_gru: QuantGRU):
    """
    Map torch.nn.GRU → QuantGRU parameters (handles any #layers and directions).

    * Every QuantGRU layer is stored as q_gru.layers[layer_idx][direction_idx]
      (even when bidirectional=False, the inner ModuleList has length 1).  So
      we can index it deterministically without introspection tricks.
    """
    dir_mult = 2 if pt_gru.bidirectional else 1
    hs = pt_gru.hidden_size
    s0, s1, s2 = slice(0, hs), slice(hs, 2 * hs), slice(2 * hs, 3 * hs)

    for layer in range(pt_gru.num_layers):
        for direction in range(dir_mult):
            suf = "_reverse" if direction else ""

            # ---------- reference tensors from torch.nn.GRU -----------------
            w_ih = getattr(pt_gru, f"weight_ih_l{layer}{suf}")
            w_hh = getattr(pt_gru, f"weight_hh_l{layer}{suf}")
            b_ih = getattr(pt_gru, f"bias_ih_l{layer}{suf}")
            b_hh = getattr(pt_gru, f"bias_hh_l{layer}{suf}")
            
            # ---------- target QuantGRU layer -------------------------------
            q_layer_container = q_gru.layers[layer]                   # always ModuleList
            q_layer = q_layer_container[direction]

            # input-to-hidden weights
            q_layer.reset_gate_params .input_weight .weight.data.copy_(w_ih[s0])
            q_layer.update_gate_params.input_weight.weight.data.copy_(w_ih[s1])
            q_layer.new_gate_params   .input_weight.weight.data.copy_(w_ih[s2])

            # hidden-to-hidden weights
            q_layer.reset_gate_params .hidden_weight.weight.data.copy_(w_hh[s0])
            q_layer.update_gate_params.hidden_weight.weight.data.copy_(w_hh[s1])
            q_layer.new_gate_params   .hidden_weight.weight.data.copy_(w_hh[s2])

            # input bias per gate
            q_layer.reset_gate_params.bias.data.copy_(b_ih[s0])
            q_layer.update_gate_params.bias.data.copy_(b_ih[s1])
            q_layer.new_gate_params.bias.data.copy_(b_ih[s2])

            # hidden bias per gate
            q_layer.reset_gate_params.hidden_bias.data.copy_(b_hh[s0])
            q_layer.update_gate_params.hidden_bias.data.copy_(b_hh[s1])
            q_layer.new_gate_params.hidden_bias.data.copy_(b_hh[s2])

class TestRecurrent:

    @given(
        inp=float_tensor_random_size_st(dims=3, max_size=3),
        hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_rnn_quant_disabled_fwd(
            self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        m = torch.nn.RNN(
            inp_size,
            hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm = QuantRNN(
            inp_size,
            hidden_size,
            weight_quant=None,
            bias_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm.load_state_dict(m.state_dict())
        ref_out = m(inp)
        out = qm(inp)
        assert torch.isclose(out[0], ref_out[0], atol=ATOL).all()
        assert torch.isclose(out[1], ref_out[1], atol=ATOL).all()

    @given(
        inp=float_tensor_random_size_st(dims=3, max_size=3),
        hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_rnn_quant_disabled_fwd_state_dict(
            self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        qm = QuantRNN(
            inp_size,
            hidden_size,
            weight_quant=None,
            bias_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        # Test that the brevitas model can be saved/loaded without warning
        with warnings.catch_warnings(record=True) as wlist:
            qm.load_state_dict(qm.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)

    @given(
        inp=float_tensor_random_size_st(dims=3, max_size=3),
        hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_lstm_quant_disabled_fwd(
            self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        m = torch.nn.LSTM(
            inp_size,
            hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm = QuantLSTM(
            inp_size,
            hidden_size,
            weight_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            cell_state_quant=None,
            bias_quant=None,
            sigmoid_quant=None,
            tanh_quant=None,
            cat_output_cell_states=True,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm.load_state_dict(m.state_dict())
        ref_out = m(inp)
        out = qm(inp)
        # output values
        assert torch.isclose(out[0], ref_out[0], atol=ATOL).all()
        # hidden states
        assert torch.isclose(out[1][0], ref_out[1][0], atol=ATOL).all()
        # cell states
        assert torch.isclose(out[1][1], ref_out[1][1], atol=ATOL).all()

    @given(
        inp=float_tensor_random_size_st(dims=3, max_size=3),
        hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_lstm_quant_disabled_fwd_state_dict(
            self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        qm = QuantLSTM(
            inp_size,
            hidden_size,
            weight_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            cell_state_quant=None,
            bias_quant=None,
            sigmoid_quant=None,
            tanh_quant=None,
            cat_output_cell_states=True,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        # Test that the brevitas model can be saved/loaded without warning
        with warnings.catch_warnings(record=True) as wlist:
            qm.load_state_dict(qm.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    def test_quant_rnn_fwd_call(
            self, batch_first, bidirectional, num_layers, bias, return_quant_tensor):
        inp_size = 4
        hidden_size = 5
        inp = torch.randn(2, 3, inp_size)
        m = QuantRNN(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            shared_input_hidden_weights=bidirectional == 'shared_input_hidden_weights',
            return_quant_tensor=return_quant_tensor)
        assert m(inp) is not None

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.skip("FIXME: many warnings due to __torch_function__")
    def test_quant_rnn_state_dict(
            self, batch_first, bidirectional, num_layers, bias, return_quant_tensor):
        inp_size = 4
        hidden_size = 5
        m = QuantRNN(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            shared_input_hidden_weights=bidirectional == 'shared_input_hidden_weights',
            return_quant_tensor=return_quant_tensor)
        with warnings.catch_warnings(record=True) as wlist:
            m.load_state_dict(m.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("coupled_input_forget_gates", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_weight_quant", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_gate_acc_quant", [True, False])
    @pytest.mark.parametrize("shared_cell_state_quant", [True, False])
    def test_quant_lstm_fwd_call(
            self,
            batch_first,
            bidirectional,
            num_layers,
            bias,
            coupled_input_forget_gates,
            return_quant_tensor,
            shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant):
        inp_size = 4
        hidden_size = 5
        inp = torch.randn(2, 3, inp_size)
        m = QuantLSTM(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            coupled_input_forget_gates=coupled_input_forget_gates,
            shared_input_hidden_weights=bidirectional == 'shared_input_hidden_weights',
            shared_intra_layer_weight_quant=shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant=shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant=shared_cell_state_quant,
            cat_output_cell_states=shared_cell_state_quant,
            return_quant_tensor=return_quant_tensor)
        assert m(inp) is not None

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("coupled_input_forget_gates", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_weight_quant", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_gate_acc_quant", [True, False])
    @pytest.mark.parametrize("shared_cell_state_quant", [True, False])
    @pytest.mark.skip("FIXME: many warnings due to __torch_function__")
    def test_quant_lstm_fwd_state_dict(
            self,
            batch_first,
            bidirectional,
            num_layers,
            bias,
            coupled_input_forget_gates,
            return_quant_tensor,
            shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant):
        inp_size = 4
        hidden_size = 5
        qm = QuantLSTM(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            coupled_input_forget_gates=coupled_input_forget_gates,
            shared_input_hidden_weights=bidirectional == 'shared_input_hidden_weights',
            shared_intra_layer_weight_quant=shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant=shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant=shared_cell_state_quant,
            cat_output_cell_states=shared_cell_state_quant,
            return_quant_tensor=return_quant_tensor)
        # Test that the brevitas model can be saved/loaded without warning
        with warnings.catch_warnings(record=True) as wlist:
            qm.load_state_dict(qm.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)

    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("cat_output_cell_states", [True, False])
    @pytest.mark.parametrize("shared_cell_state_quant", [True, False])
    @pytest.mark.xfail(
        reason=
        'TODO: Fix inconsistent return datatypes with num_layers=2, cat_output_cell_states=False, and return_quant_tensor=False'
    )
    def test_quant_lstm_cell_state(
            self,
            return_quant_tensor,
            num_layers,
            bidirectional,
            cat_output_cell_states,
            shared_cell_state_quant):

        if cat_output_cell_states and not shared_cell_state_quant:
            pytest.skip("Concatenating cell states requires shared cell quantizers.")

        inp = torch.randn(1, 3, 5)
        qm = QuantLSTM(
            inp.size(-1),
            8,
            bidirectional=bidirectional,
            num_layers=num_layers,
            return_quant_tensor=return_quant_tensor,
            cat_output_cell_states=cat_output_cell_states,
            shared_cell_state_quant=shared_cell_state_quant)
        out, (h, c) = qm(inp)

        if cat_output_cell_states and (return_quant_tensor or num_layers == 2):
            assert isinstance(c, QuantTensor)
        elif cat_output_cell_states:
            assert isinstance(c, torch.Tensor)
        else:
            assert isinstance(c, list)
            if (return_quant_tensor):
                assert all([isinstance(el, QuantTensor) for el in c])
            else:
                assert all([isinstance(el, torch.Tensor) for el in c])
    
    # QuantGRU tests
    # -----------------------------------------------------------
    # shape & type -- single pass
    # -----------------------------------------------------------
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.parametrize("batch_first",         [True, False])
    @pytest.mark.parametrize("bidirectional",       [True, False])
    @pytest.mark.parametrize("num_layers",          [1, 2])
    def test_quant_gru_shape(self, return_quant_tensor, batch_first, bidirectional, num_layers):
        torch.manual_seed(0)

        seq_len, batch_size, input_size, hidden_size = 7, 4, 8, 16
        model = QuantGRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            return_quant_tensor=return_quant_tensor,   # most training loops work with plain tensors
        )

        x = _make_input(seq_len, batch_size, input_size, batch_first=batch_first)
        out, h_n = model(x)

        dir_mult = 2 if bidirectional else 1

        # output shape --------------------------------------------------------
        if batch_first:
            assert out.shape == (batch_size, seq_len, hidden_size * dir_mult)
        else:
            assert out.shape == (seq_len, batch_size, hidden_size * dir_mult)

        # hidden shape --------------------------------------------------------
        assert h_n.shape == (num_layers * dir_mult, batch_size, hidden_size)


    # -----------------------------------------------------------
    # forward + backward  (basic autograd check)
    # -----------------------------------------------------------
    def test_quant_gru_backward(self):
        torch.manual_seed(1)

        model = QuantGRU(10, 20, num_layers=2)
        x = torch.randn(5, 3, 10, requires_grad=True)   # (T,B,F)

        out, _ = model(x)
        loss = out.sum()
        loss.backward()

        # gradients should be populated and finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


    # -----------------------------------------------------------
    # JIT / TorchScript smoke test
    # (only run when JIT is enabled at build time)
    # -----------------------------------------------------------
    @pytest.mark.skipif(
        not getattr(torch._C, '_jit_enabled', lambda: True)(),
        reason="PyTorch JIT disabled in this build",
    )
    def test_quant_gru_jit_trace(self):
        torch.manual_seed(2)

        model = QuantGRU(6, 12, batch_first=True).eval()
        x = torch.randn(4, 9, 6)    # (B,T,F)

        ref_out, ref_hn = model(x)

        traced = torch.jit.trace(model, (x,))
        out, hn = traced(x)

        assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-4)
        assert torch.allclose(hn,  ref_hn,  atol=1e-5, rtol=1e-4)
        
    # ---------------------------------------------------------------------------
    # 1) forward equivalence (QuantGRU ↔ torch.nn.GRU)
    # ---------------------------------------------------------------------------
    @pytest.mark.parametrize("batch_first",   [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers",    [1, 2])
    def test_gru_quant_matches_base(self, batch_first, bidirectional, num_layers):
        torch.manual_seed(0)

        # --- reference -------------------------------------------------------
        pt = torch.nn.GRU(
            FEAT, HIDDEN,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first
        )

        # --- QuantGRU with identity quantisers ------------------------------
        q = QuantGRU(
            FEAT, HIDDEN,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
            weight_quant=None,
            bias_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            sigmoid_quant=None,
            tanh_quant=None,
            return_quant_tensor=False,
        )

        _copy_pt_to_quant(pt, q)

        x = _inp(batch_first)
        out_pt, h_pt = pt(x)
        out_q,  h_q  = q(x)

        assert torch.allclose(out_pt, out_q, atol=1e-6, rtol=1e-6)
        assert torch.allclose(h_pt,  h_q,  atol=1e-6, rtol=1e-6)


    def test_gru_quant_backward(self):
        torch.manual_seed(1)

        pt = torch.nn.GRU(FEAT, HIDDEN, batch_first=True)
        q  = QuantGRU(
            FEAT, HIDDEN,
            batch_first=True,
            weight_quant=None,
            bias_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            sigmoid_quant=None,
            tanh_quant=None,
            return_quant_tensor=False,
        )
        _copy_pt_to_quant(pt, q)

        x = _inp(batch_first=True).requires_grad_()
        out, _ = q(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        for p in q.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            
    @pytest.mark.parametrize("flag", [
        "shared_input_hidden_weights",
        "shared_intra_layer_weight_quant",
        "shared_intra_layer_gate_acc_quant",
    ])
    def test_gru_sharing_flags(self, flag):
        kwargs = {flag: True, "return_quant_tensor": False}
        if flag == "shared_input_hidden_weights":
            kwargs["bidirectional"] = True
        model = QuantGRU(FEAT, HIDDEN, **kwargs)
        layer = model.layers[0][0]   # first (forward) layer

        if flag == "shared_input_hidden_weights":
            # input → hidden **and** hidden → hidden reuse the *same* tensor
            assert id(layer.reset_gate_params.input_weight.weight) == \
                id(layer.reset_gate_params.hidden_weight.weight)
        elif flag == "shared_intra_layer_weight_quant":
            # the three gates share the SAME tensor_quant object
            tq = layer.reset_gate_params.input_weight.weight_quant.tensor_quant
            assert tq is layer.update_gate_params.input_weight.weight_quant.tensor_quant
            assert tq is layer.new_gate_params   .input_weight.weight_quant.tensor_quant
        else:  # shared_intra_layer_gate_acc_quant
            # accumulator quantisers inside the cell are a single instance
            cell = layer.cell
            assert cell.reset_acc_quant is cell.update_acc_quant is cell.new_acc_quant


    def test_gru_return_quant_tensor(self):
        model = QuantGRU(FEAT, HIDDEN, return_quant_tensor=True)
        out, _ = model(_inp(batch_first=False))
        assert isinstance(out, QuantTensor), f"Not found a quantized tensor, but a {type(out).__name__}"