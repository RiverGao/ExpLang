# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
set -e
export HF_HUB_OFFLINE=1

wandb offline

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/OpenR1-Math-Converted4VeRL/default-train.parquet \
    data.val_files=data/OpenR1-Math-Converted4VeRL/default-test.parquet \
    data.train_max_samples=51200 \
    data.val_max_samples=500 \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=model/qwen3-4b-langselect-openr1-math-lora-no-add-baseline-v2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.use_reward_loop=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='verl_grpo_example_openr1-math' \
    trainer.experiment_name='qwen3_4b_full_openr1-math_baseline_v2' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_grpo_example_openr1-math/qwen3_4b_full_openr1-math_baseline_v2/global_step_199/actor \
    --target_dir checkpoints/qwen3_4b_full_openr1-math_baseline_v2

# evaluation
python infer_eval_data_multi.py \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2 \

python infer_eval_data_en.py \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2 \

python infer_eval_data_self.py \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2

python infer_eval_data_multi.py \
    --dataset olymmath-easy \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2 \

python infer_eval_data_en.py \
    --dataset olymmath-easy \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2 \

python infer_eval_data_self.py \
    --dataset olymmath-easy \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2

python infer_eval_data_multi.py \
    --dataset aime25 \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2 \

python infer_eval_data_en.py \
    --dataset aime25 \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2 \

python infer_eval_data_self.py \
    --dataset aime25 \
    --model_path checkpoints/qwen3_4b_full_openr1-math_baseline_v2 \
    --output_dir infer_eval/qwen3_4b_full_openr1-math_baseline_v2