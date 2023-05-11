import deepspeed
from transformers import XGLMForCausalLM, TrainingArguments


def main():
    deepspeed.init_distributed()
    ds_config = {
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 500,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 1e6,
            "stage3_prefetch_bucket_size": 0.94e6,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }
    with deepspeed.zero.Init(
        config_dict_or_path=ds_config,
    ):
        model_name = "facebook/xglm-7.5B"
        training_args = TrainingArguments(deepspeed=ds_config, output_dir="./models/")
        model = XGLMForCausalLM.from_pretrained(model_name)


if __name__ == "__main__":
    main()
