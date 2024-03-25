#!/bin/bash
set -x 
set -e

# augmented GCG

save_dir="./evluatation_results"
split="test"
ppl=false
augmented_GCG=llama2_lowest
victim_model=llama2-chat

python evaluate_augmentedGCG.py s_p_t_dir=$save_dir target_lm=$victim_model force_append=true prompt_own_list_name=$augmented_GCG prompt_way=own data_args.split=${split} ppl=${ppl} target_lm.generation_configs.max_new_tokens=100





# If you have obtained the AmpleGCG based on augmented GCG, you could directly apply the scripts below to evlaute its effectiveness.


# save_dir="./evluatation_results"
# split="test"
# ppl=false
# augmented_GCG=llama2_lowest
# victim_model=llama2-chat
# AmpleGCG_path="./path/to/AmpleGCG"
# show_name="My_learned_AmpleGCG"

# python evaluate_augmentedGCG.py s_p_t_dir=$save_dir target_lm=$victim_model "prompter_lm.model_name=$AmpleGCG_path" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=group_beam_search prompter_lm.generation_configs.num_return_sequences=200 prompter_lm.generation_configs.num_beams=200 prompter_lm.generation_configs.num_beam_groups=200 target_lm.batch_size=64 prompter_lm.batch_size=2

