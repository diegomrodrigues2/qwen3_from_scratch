import torch
from model import Qwen3ForCausalLM


def convert_and_load_weights(model: Qwen3ForCausalLM, original_weights_path: str):
    """
    Loads weights from the original standalone .pth file and maps them to the
    Hugging Face model's state_dict.
    """
    try:
        original_state_dict = torch.load(original_weights_path, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: Weight file not found at '{original_weights_path}'")
        print(
            "Please download the original weights and place them in the correct directory."
        )
        return

    hf_state_dict = {}

    # Weight mapping logic. This requires inspecting the names of the weights
    # in both the original and new models.
    #
    # Original Name                         | HF Name
    # --------------------------------------|---------------------------------------
    # tok_embeddings.weight                 | embed_tokens.weight
    # norm.weight                           | norm.weight
    # output.weight                         | lm_head.weight
    # layers.{i}.attention_norm.weight      | layers.{i}.input_layernorm.weight
    # layers.{i}.ffn_norm.weight            | layers.{i}.post_attention_layernorm.weight
    # layers.{i}.attention.wq.weight        | layers.{i}.self_attn.q_proj.weight (and bias)
    # layers.{i}.attention.wk.weight        | layers.{i}.self_attn.k_proj.weight (and bias)
    # layers.{i}.attention.wv.weight        | layers.{i}.self_attn.v_proj.weight (and bias)
    # layers.{i}.attention.wo.weight        | layers.{i}.self_attn.o_proj.weight
    # layers.{i}.feed_forward.w1.weight     | layers.{i}.mlp.gate_proj.weight
    # layers.{i}.feed_forward.w3.weight     | layers.{i}.mlp.up_proj.weight
    # layers.{i}.feed_forward.w2.weight     | layers.{i}.mlp.down_proj.weight

    # Basic mappings
    hf_state_dict["embed_tokens.weight"] = original_state_dict["tok_embeddings.weight"]
    hf_state_dict["norm.weight"] = original_state_dict["norm.weight"]
    hf_state_dict["lm_head.weight"] = original_state_dict["output.weight"]

    # Per-layer mappings
    for i in range(model.config.n_layer):
        # Attention
        hf_state_dict[f"layers.{i}.input_layernorm.weight"] = original_state_dict[
            f"layers.{i}.attention_norm.weight"
        ]
        hf_state_dict[f"layers.{i}.self_attn.q_proj.weight"] = original_state_dict[
            f"layers.{i}.attention.wq.weight"
        ]
        hf_state_dict[f"layers.{i}.self_attn.q_proj.bias"] = original_state_dict[
            f"layers.{i}.attention.wq.bias"
        ]
        hf_state_dict[f"layers.{i}.self_attn.k_proj.weight"] = original_state_dict[
            f"layers.{i}.attention.wk.weight"
        ]
        hf_state_dict[f"layers.{i}.self_attn.k_proj.bias"] = original_state_dict[
            f"layers.{i}.attention.wk.bias"
        ]
        hf_state_dict[f"layers.{i}.self_attn.v_proj.weight"] = original_state_dict[
            f"layers.{i}.attention.wv.weight"
        ]
        hf_state_dict[f"layers.{i}.self_attn.v_proj.bias"] = original_state_dict[
            f"layers.{i}.attention.wv.bias"
        ]
        hf_state_dict[f"layers.{i}.self_attn.o_proj.weight"] = original_state_dict[
            f"layers.{i}.attention.wo.weight"
        ]

        # MLP
        hf_state_dict[f"layers.{i}.post_attention_layernorm.weight"] = (
            original_state_dict[f"layers.{i}.ffn_norm.weight"]
        )
        hf_state_dict[f"layers.{i}.mlp.gate_proj.weight"] = original_state_dict[
            f"layers.{i}.feed_forward.w1.weight"
        ]
        hf_state_dict[f"layers.{i}.mlp.up_proj.weight"] = original_state_dict[
            f"layers.{i}.feed_forward.w3.weight"
        ]
        hf_state_dict[f"layers.{i}.mlp.down_proj.weight"] = original_state_dict[
            f"layers.{i}.feed_forward.w2.weight"
        ]

    # Load the mapped weights into our model
    model.load_state_dict(hf_state_dict)
    print("Successfully loaded and converted weights.")
