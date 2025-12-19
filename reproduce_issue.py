
import torch
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

from octo_embedding_model.train_phase1 import ChromaMoEForPretraining
from octo_embedding_model.model_architecture import ChromaConfig

def test_forward_with_cu_seqlens():
    print("Testing ChromaMoEForPretraining forward with cu_seqlens...")
    
    # Create minimal config
    config = ChromaConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        moe_intermediate_size=64,
        num_routed_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2
    )

    model = ChromaMoEForPretraining(config)
    
    # Dummy data
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Simulate packed sequences args
    cu_seqlens = torch.tensor([0, seq_len, 2*seq_len], dtype=torch.int32)
    max_seqlen = seq_len
    
    try:
        # This call mimics what happens in train_epoch
        output = model(
            input_ids=input_ids,
            attention_mask=None, # In packed mode, attention_mask is often None or unused
            labels=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        print("Success! Forward pass completed.")
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
    except Exception as e:
        print(f"Caught unexpected Exception: {e}")

if __name__ == "__main__":
    test_forward_with_cu_seqlens()
