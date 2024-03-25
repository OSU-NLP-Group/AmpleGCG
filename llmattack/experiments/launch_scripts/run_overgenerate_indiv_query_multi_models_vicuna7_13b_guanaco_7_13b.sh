set -x
set -e


n_steps=500
for offset in $(seq 0 10 200)
do
    bash run_overgenerate_indiv_query_multi_models.sh multi_models_vicuna7_13b_guanaco_7_13b 4 $offset $n_steps
done


