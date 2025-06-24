import torch

from qwen3.model import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_emb,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3Block,
    Qwen3ForCausalLM,
    Qwen3Config,
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


def test_config_defaults():
    cfg = tiny_config()
    assert cfg.use_cache is True
    assert cfg.use_return_dict is True
    assert cfg.output_attentions is False
    assert cfg.output_hidden_states is False


def test_rmsnorm_basic():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    norm = RMSNorm(8)
    norm.weight.data.fill_(1.0)
    out = norm(x)
    rms = x.pow(2).mean(-1, keepdim=True)
    expected = x / torch.sqrt(rms + norm.eps)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rotary_embedding_shape():
    x = torch.randn(1, 4, 2, 4)
    rope = RotaryEmbedding(dim=4, max_seq_len=10)
    freqs = rope(x)
    out = apply_rotary_emb(x, freqs)
    assert out.shape == x.shape


def test_attention_cache():
    cfg = tiny_config()
    attn = Qwen3Attention(cfg)
    x = torch.randn(1, 3, cfg.n_embd)
    out, _, pkv = attn(x, use_cache=True)
    assert out.shape == (1, 3, cfg.n_embd)
    assert pkv[0].shape[:3] == (1, 3, cfg.n_head)
    x2 = torch.randn(1, 2, cfg.n_embd)
    _, _, pkv2 = attn(x2, past_key_value=pkv, use_cache=True)
    assert pkv2[0].shape[1] == 5


def test_mlp_shape():
    cfg = tiny_config()
    mlp = Qwen3MLP(cfg)
    x = torch.randn(2, 4, cfg.n_embd)
    out = mlp(x)
    assert out.shape == x.shape


def test_block_shape():
    cfg = tiny_config()
    block = Qwen3Block(cfg)
    x = torch.randn(1, 5, cfg.n_embd)
    out, _, _ = block(x)
    assert out.shape == x.shape


def test_model_forward():
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    outputs = model(input_ids)
    assert outputs.logits.shape == (1, 4, cfg.vocab_size)


def test_model_loss():
    cfg = tiny_config()
    model = Qwen3ForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    outputs = model(input_ids, labels=input_ids)
    assert outputs.loss is not None
