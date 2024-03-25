set -x
set -e


n_steps=500
for offset in $(seq 0 10 318)
do
    bash run_overgenerate_indiv_query_indiv_model.sh indiv_model_vicuna $offset $n_steps
done
