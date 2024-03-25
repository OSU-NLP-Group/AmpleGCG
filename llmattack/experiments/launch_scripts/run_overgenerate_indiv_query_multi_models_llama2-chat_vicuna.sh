set -x
set -e


n_steps=500
for offset in $(seq 0 10 200)
do
    bash run_overgenerate_indiv_query_multi_models.sh multi_models_llama2-chat_vicuna 2 $offset $n_steps
done
