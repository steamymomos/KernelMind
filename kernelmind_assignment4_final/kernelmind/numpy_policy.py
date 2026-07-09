from __future__ import annotations

import math

import numpy as np
import torch

from .network import DirectSchedulerNet


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64)


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class NumpyPolicy:
    """Numpy inference copy of DirectSchedulerNet for deterministic, lightweight evaluation."""

    def __init__(self, net: DirectSchedulerNet):
        self.max_queue_size = net.max_queue_size
        self.embed_dim = net.embed_dim
        self.num_heads = net.num_heads
        self.head_dim = net.head_dim
        self.pos = _to_np(net.position_embedding)[0]
        self.ip_w, self.ip_b = _to_np(net.input_projection.weight), _to_np(net.input_projection.bias)
        self.q_w, self.q_b = _to_np(net.q_proj.weight), _to_np(net.q_proj.bias)
        self.k_w, self.k_b = _to_np(net.k_proj.weight), _to_np(net.k_proj.bias)
        self.v_w, self.v_b = _to_np(net.v_proj.weight), _to_np(net.v_proj.bias)
        self.o_w, self.o_b = _to_np(net.out_proj.weight), _to_np(net.out_proj.bias)
        self.n1_w, self.n1_b = _to_np(net.norm1.weight), _to_np(net.norm1.bias)
        self.ff1_w, self.ff1_b = _to_np(net.ff[0].weight), _to_np(net.ff[0].bias)
        self.ff2_w, self.ff2_b = _to_np(net.ff[2].weight), _to_np(net.ff[2].bias)
        self.n2_w, self.n2_b = _to_np(net.norm2.weight), _to_np(net.norm2.bias)
        self.h1_w, self.h1_b = _to_np(net.q_head[0].weight), _to_np(net.q_head[0].bias)
        self.h2_w, self.h2_b = _to_np(net.q_head[2].weight), _to_np(net.q_head[2].bias)

    def q_values(self, state: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        state = state.astype(np.float64)
        original_mask = valid_mask.astype(bool).copy()
        attn_mask = original_mask.copy()
        if not attn_mask.any():
            attn_mask[0] = True
        x = _linear(state, self.ip_w, self.ip_b) + self.pos[: state.shape[0]]
        q = _linear(x, self.q_w, self.q_b)
        k = _linear(x, self.k_w, self.k_b)
        v = _linear(x, self.v_w, self.v_b)
        n = x.shape[0]
        q = q.reshape(n, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(n, self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(n, self.num_heads, self.head_dim).transpose(1, 0, 2)
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(self.head_dim)
        scores[:, :, ~attn_mask] = -1e9
        scores = np.minimum(scores, 50.0)
        weights = _softmax(scores, axis=-1)
        out = np.matmul(weights, v).transpose(1, 0, 2).reshape(n, self.embed_dim)
        out = _linear(out, self.o_w, self.o_b)
        x = _layer_norm(x + out, self.n1_w, self.n1_b)
        ff = _linear(_relu(_linear(x, self.ff1_w, self.ff1_b)), self.ff2_w, self.ff2_b)
        x = _layer_norm(x + ff, self.n2_w, self.n2_b)
        qv = _linear(_relu(_linear(x, self.h1_w, self.h1_b)), self.h2_w, self.h2_b).squeeze(-1)
        qv[~original_mask] = -1e9
        return qv.astype(np.float32)
