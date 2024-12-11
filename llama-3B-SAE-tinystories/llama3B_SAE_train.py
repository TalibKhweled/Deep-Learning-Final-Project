import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from sae_lens import (
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)


device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

print(f"Using device: {device}")

MODEL_PATH = "/app/models/llama-3.2-3B-Instruct"

if MODEL_PATH:
    tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_PATH)
    hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH)

    llama_model = HookedSAETransformer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        hf_model=hf_model,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer
    )



t.set_grad_enabled(True)
total_training_steps = 50_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

# limit warmup steps to 10% of training
lr_warm_up_steps = l1_warm_up_steps = total_training_steps // 10  
# limit decay steps to 20% of training
lr_decay_steps = total_training_steps // 5 
# Only train the second half of the layers
llama_layer_start = (llama_model.cfg.n_layers // 2) - 1



for layer in range(llama_layer_start, llama_model.cfg.n_layers):
    cfg = LanguageModelSAERunnerConfig(
        #
        # Data generation
        model_name="meta-llama/Llama-3.2-3B-Instruct",  
        hook_name=f"blocks.{layer}.hook_mlp_out",
        hook_layer=layer,
        d_in=llama_model.cfg.d_model,
        dataset_path="roneneldan/TinyStories",
        is_dataset_tokenized=False,
        prepend_bos=True, 
        streaming=True,  
        train_batch_size_tokens=batch_size,
        context_size=256,  
        #
        # SAE architecture
        architecture="topk",
        expansion_factor=16,
        b_dec_init_method="zeros",
        apply_b_dec_to_input=True,
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        #
        # Activations store
        n_batches_in_buffer=32,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,
        #
        # Training hyperparameters (standard)
        lr=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # controls how the LR warmup / decay works
        lr_warm_up_steps=lr_warm_up_steps,  # avoids large number of initial dead features
        lr_decay_steps=lr_decay_steps,  # helps avoid overfitting
        #
        # Training hyperparameters (SAE-specific)
        l1_coefficient=4,
        l1_warm_up_steps=l1_warm_up_steps,
        use_ghost_grads=False,  # we don't use ghost grads anymore
        feature_sampling_window=1000,  # how often we resample dead features
        dead_feature_window=500,  # size of window to assess whether a feature is dead
        dead_feature_threshold=1e-4,  # threshold for classifying feature as dead, over window
        #
        # Logging / evals
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="SAE_Lens_llama3-3B-Instruct",
        run_name=f"llama3-3B-Instruct_topk-SAE_Layer-{layer}",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        #
        # Misc.
        device=str(device),
        seed=42,
        n_checkpoints=5,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    print(cfg.run_name)
    # Kick off training
    sae = SAETrainingRunner(cfg, override_model=llama_model).run()

    # Upload the SAE to Hugging Face
    hf_repo_id = "talibk/SAE_Lens_llama3-3B-Instruct"
    sae_id = cfg.hook_name
    upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

print('Training complete!')