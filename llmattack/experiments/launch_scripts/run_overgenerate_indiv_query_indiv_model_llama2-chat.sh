set -x
set -e


n_steps=1000
for offset in $(seq 0 10 318)
do
    bash run_overgenerate_indiv_query_indiv_model.sh indiv_model_llama2-chat $offset $n_steps
done
