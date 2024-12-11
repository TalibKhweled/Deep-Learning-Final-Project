import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from sae_lens import (
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)

# Set up device (MPS for Mac, CUDA for NVIDIA, CPU as fallback)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer from local path
MODEL_PATH = "/app/models/llama-3.2-3B-Instruct"

if MODEL_PATH:
    tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_PATH)
    hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH)

    # Initialize HookedSAETransformer with default settings
    llama_model = HookedSAETransformer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        hf_model=hf_model,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer
    )

# Training configuration
t.set_grad_enabled(True)
total_training_steps = 50_000  # More steps for OpenWebText due to dataset complexity
batch_size = 4096  # Larger batch size for more diverse data
total_training_tokens = total_training_steps * batch_size

# Training schedule parameters
lr_warm_up_steps = l1_warm_up_steps = total_training_steps // 10  # 10% warmup
lr_decay_steps = total_training_steps // 5  # 20% decay
llama_layer_start = (llama_model.cfg.n_layers // 2) - 1  # Start from middle layer

# Train SAEs for each layer sequentially
for layer in range(llama_layer_start, llama_model.cfg.n_layers):
    cfg = LanguageModelSAERunnerConfig(
        # Model and data settings
        model_name="meta-llama/Llama-3.2-3B-Instruct",  
        hook_name=f"blocks.{layer}.hook_mlp_out",
        hook_layer=layer,
        d_in=llama_model.cfg.d_model,
        dataset_path="openwebtext",  # Using OpenWebText instead of TinyStories
        context_size=256,
        train_batch_size_tokens=batch_size,

        # SAE architecture settings
        architecture="topk",
        expansion_factor=16,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,

        # Buffer and storage settings
        n_batches_in_buffer=32,  # Reduced due to larger batch size
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,

        # Optimizer settings
        lr=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,

        # Feature detection settings
        l1_coefficient=4,
        l1_warm_up_steps=l1_warm_up_steps,
        feature_sampling_window=1000,
        dead_feature_window=500,
        dead_feature_threshold=1e-4,

        # Logging configuration
        log_to_wandb=True,
        wandb_project="SAE_Lens_llama3-3B-Instruct",
        run_name=f"llama3-3B-Instruct_topk-SAE_Layer-{layer}",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,

        # Environment settings
        device=str(device),
        seed=42,
        n_checkpoints=5,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    print(cfg.run_name)
    # Train the SAE for current layer
    sae = SAETrainingRunner(cfg, override_model=llama_model).run()

    # Save trained SAE to HuggingFace
    hf_repo_id = "talibk/SAE_Lens_llama3-3B-Instruct"
    sae_id = cfg.hook_name
    upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

print('Training complete!')