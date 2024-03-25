#!/bin/bash

set -e
set -x

reward_dirs=(
"evluatation_results"
)

reward_mode=$1
force_replace=${2:-false}

if [ "$reward_mode" = "beaver" ]; then
  for top_dir in "${reward_dirs[@]}"
  do    
    find $top_dir -type f -name "*.jsonl" | while IFS= read -r file; do
      python add_reward.py "path='$file'" reward_lm=beaver-7b only_prev_harms=[] "force_replace=$force_replace"
      echo $file
    done

  done

elif [ "$reward_mode" = "harmbench" ]; then

  for top_dir in "${reward_dirs[@]}"
  do    
    find $top_dir -type f -name "*.jsonl" | while IFS= read -r file; do
      python add_reward.py "path='$file'" reward_lm=harmbench-13b only_prev_harms=["beaver"] "force_replace=$force_replace" batch_size=64
      echo $file
    done
  done


elif [ "$reward_mode" = "gpt4" ]; then

  for top_dir in "${reward_dirs[@]}"
  do    
    find $top_dir -type f -name "*.jsonl" | while IFS= read -r file; do
      python add_reward.py "path='$file'" reward_lm=gpt4-0613 only_prev_harms=["beaver","harmbench"] "force_replace=$force_replace"
      echo $file
    done
  done

elif [ "$reward_mode" = "sequence" ]; then

  for top_dir in "${reward_dirs[@]}"
  do    
    find $top_dir -type f -name "*.jsonl" | while IFS= read -r file; do
      python add_reward.py "path='$file'" reward_lm=beaver-7b only_prev_harms=[] "force_replace=$force_replace"
      python add_reward.py "path='$file'" reward_lm=harmbench-13b only_prev_harms=["beaver"] "force_replace=$force_replace" batch_size=64
      # python add_reward.py "path='$file'" reward_lm=gpt4-0613 only_prev_harms=["beaver","harmbench"] "force_replace=$force_replace"
      echo $file
    done
  done



else
  echo "input is not beaver, harmbench, gpt4 or sequence"
fi








