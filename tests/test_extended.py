import torch
import torch.nn as nn
from transformers import PretrainedConfig
from qwen3.model import (
    RMSNorm,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3ForCausalLM,
    Qwen3Config,
    Qwen3Tokenizer,
)


def tiny_config(**kwargs):
    defaults = dict(
        vocab_size=32,
        context_len=16,
        n_layer=2,
        n_head=2,
        n_embd=8,
        intermediate_size=16,
    )
    defaults.update(kwargs)
    return Qwen3Config(**defaults)


def test_module_shapes():
    cfg = tiny_config()
    batch, seq_len, hidden = 2, 4, cfg.n_embd
    x = torch.randn(batch, seq_len, hidden)
    attn = Qwen3Attention(cfg)
    mlp = Qwen3MLP(cfg)
    norm = RMSNorm(hidden)
    out_a, _, _ = attn(x)
    out_m = mlp(x)
    out_n = norm(x)
    assert out_a.shape == (batch, seq_len, hidden)
    assert out_m.shape == (batch, seq_len, hidden)
    assert out_n.shape == (batch, seq_len, hidden)


def test_attention_mask_scores():
    batch, q_len, k_len, d = 1, 2, 3, 4
    q = torch.randn(batch, q_len, d)
    k = torch.randn(batch, k_len, d)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
    masked_scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
    assert torch.all(masked_scores[:, :, -1] == float('-inf'))


def test_rmsnorm_mean_var():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, unbiased=False, keepdim=True)
    norm = RMSNorm(8)
    y = norm(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(2, 3), atol=1e-5)


def test_config_inheritance():
    cfg = Qwen3Config()
    assert isinstance(cfg, PretrainedConfig)
    assert cfg.n_layer == 32
    assert cfg.n_head == 32
    assert cfg.n_embd == 4096


def test_embedding_lookup():
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    ids = torch.tensor([[1, 2, 3]])
    emb = model.get_input_embeddings().weight
    out = model.get_input_embeddings()(ids)
    assert torch.equal(emb[ids], out)


def test_mlp_manual():
    cfg = tiny_config()
    mlp = Qwen3MLP(cfg)
    x = torch.randn(1, 2, cfg.n_embd)
    W1 = mlp.gate_proj.weight.T
    W2 = mlp.up_proj.weight.T
    W3 = mlp.down_proj.weight.T
    b1 = torch.zeros(cfg.intermediate_size)
    b2 = torch.zeros(cfg.intermediate_size)
    b3 = torch.zeros(cfg.n_embd)
    manual = (torch.nn.functional.silu(x @ W1 + b1) * (x @ W2 + b2)) @ W3 + b3
    assert torch.allclose(mlp(x), manual, atol=1e-6)


def test_save_load_state_dict(tmp_path):
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    path = tmp_path / "tmp.pth"
    torch.save(model.state_dict(), path)
    model2 = Qwen3ForCausalLM(cfg)
    model2.load_state_dict(torch.load(path))
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_forward_api():
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    outputs = model(input_ids=input_ids)
    assert "logits" in outputs.keys()
    assert outputs.logits.shape == (2, 5, cfg.vocab_size)
    assert "past_key_values" in outputs.keys()


def test_dataparallel(tmp_path):
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    dp = nn.DataParallel(model)
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    out_dp = dp(ids)
    out = model(ids)
    assert torch.allclose(out_dp.logits, out.logits, atol=1e-6)


def test_tokenizer_integration():
    import os
    import pytest
    if not os.path.exists("qwen.tiktoken"):
        pytest.skip("tokenizer file missing")
    pytest.importorskip("blobfile")
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    tokenizer = Qwen3Tokenizer()
    tok_out = tokenizer("Ola mundo", return_tensors="pt")
    model(**tok_out)


def test_weight_tying():
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()

