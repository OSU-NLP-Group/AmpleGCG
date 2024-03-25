import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizer
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from lm_components import create_targetlm,create_prompterlm
from torch.utils.data import DataLoader
import jsonlines
import os
from tqdm import tqdm
from pathlib import Path
from accelerate.utils import set_seed
from print_color import print
import json
import time
from torch.utils.data import Dataset
from itertools import chain, repeat
import random
import gc
from datetime import datetime

PROMPT_DICT = {
    "q_p": (
       "### Query:{q} ### Prompt:"
    ),
}

def unique_random_concat_combinations(lst, concat_times, sample_times):
    # 生成所有可能的唯一组合
    unique_combinations = [tuple(lst[i] for i in indices) for indices in itertools.combinations(range(len(lst)), concat_times)]

    # 如果请求的组合数超过了总组合数，返回 None
    if sample_times > len(unique_combinations):
        raise NotImplementedError("False value for concat times and sample times")

    # 随机选择 sample_times 个不同的组合
    selected_combinations = random.sample(unique_combinations, sample_times)

    # 连接选中的组合
    concatenated_combinations = [' '.join(combination) for combination in selected_combinations]
    return concatenated_combinations
import itertools


def repeat_texts_l(l,repeat_times=1):
    return list(chain.from_iterable(repeat(item, repeat_times) for item in l))


def get_data(data_args):
    if data_args.prompt_type!= "q_p":
        raise NotImplementedError()
    split_path = data_args.split_path
    with open(split_path) as f:
        splits = json.load(f)
    list_qs = []
    q_s = splits[data_args.split]
    q_s = sorted(q_s)
    for q in q_s:
        list_qs.append(dict(q = q))
    print("****************")
    print(list_qs[0])
    print("****************")
    return list_qs

def attack_collate_fn(batch):
    collated_batch = {}
    for item in batch:
        for key, value in item.items():
            if key in collated_batch:
                collated_batch[key].append(value)
            else:
                collated_batch[key] = [value]
    return collated_batch  

def set_pad_token(t):
    if t.pad_token is None:
        t.pad_token = t.eos_token
    
    return t

def get_batch(l,bs):
    for i in range(0,len(l),bs):
        yield l[i: i+bs]

def do_reps(
    source_texts, 
    num_reps
):
    source_reps = []
    for text in source_texts: 
        for _ in range(num_reps): 
            source_reps.append(text)
    return source_reps


@hydra.main(config_path="./myconfig", config_name="config_evaluate")
def main(config: "DictConfig"):
    set_seed(config.seed)
    
    start_time = time.time()
    Path(config.s_p_t_dir).mkdir(exist_ok= True, parents= True)
    s_p_t_dir = config.s_p_t_dir
    if config.prompt_way == "prompter":
        promptway_name = config.prompt_way + "_" + config.data_args.prompt_type

        decode_way = f"decode_{config.prompter_lm.generation_configs.name}"
        if config.prompter_lm.generation_configs.name == "top_p":
            if config.prompter_lm.generation_configs.top_p != 0.9:
                decode_way += f"_{config.prompter_lm.generation_configs.top_p}"

        if config.prompter_lm.generation_configs.name == "top_k":
            if config.prompter_lm.generation_configs.top_k != 50:
                decode_way += f"_{config.prompter_lm.generation_configs.top_k}"


        if config.prompter_lm.generation_configs.name == "group_beam_search":
            # config.prompter_lm.generation_configs.num_beam_groups = config.prompter_lm.generation_configs.num_beams

            decode_way += f"_group_{config.prompter_lm.generation_configs.num_beam_groups}"
            if config.prompter_lm.generation_configs.diversity_penalty != 1.0:
                decode_way += f"_diverse_pen_{config.prompter_lm.generation_configs.diversity_penalty}"

        if config.prompter_lm.generation_configs.num_return_sequences > 1:
            decode_way += f"_numreturn_{config.prompter_lm.generation_configs.num_return_sequences}"
            if config.prompt_concat > 1:
                decode_way += f"_p_concat_{config.prompt_concat}_group_{config.num_prompt_group}"
        s_p_t_dir = os.path.join(s_p_t_dir,f"{config.data_args.split}|prompter_{config.prompter_lm.show_name}|{decode_way}|promptway_{promptway_name}")

        if config.seed != 42:
            s_p_t_dir += f"_seed_{config.seed}"

        if config.q_rep!=1:
            s_p_t_dir += "|" + f"q_rep_{config.q_rep}"
        if config.q_prefix.choice in ["long","short","medium"]:
            s_p_t_dir += "|" + f"q_prefix_{config.q_prefix.choice}"
        if config.q_s_position in ["prompter_lm=processed|target_lm=processed","prompter_lm=raw|target_lm=processed"]:
            if config.q_s_position == "prompter_lm=raw|target_lm=processed":
                s_p_t_dir += "|" + f"q_s_position_{config.q_s_position}"
        else:
            raise ValueError("The q_s_position is not defined")
        
        if config.w_affirm_suffix:
            s_p_t_dir += "|" + f"w_affirm_suffix"

        if config.prompter_w_system:
            s_p_t_dir += "|" + f"prompter_w_system"
        
        if config.sys_msg.choice is not None:
            if config.sys_msg.choice in ["no_persuasive"]:
                s_p_t_dir += "|" + f"sys_msg_{config.sys_msg.choice}"
            else:
                raise ValueError(f"The {config.sys_msg.choice} is not defined")
            

    elif config.prompt_way == "own":
        assert config.prompt_own_list is not None
        assert config.prompt_own_list_name is not None
        promptway_name = config.prompt_way + "_" + config.data_args.prompt_type
        s_p_t_dir = os.path.join(s_p_t_dir,f"{config.data_args.split}|prompter_None|promptway_{promptway_name}|prompt_own_list_name_{config.prompt_own_list_name}")
        
        if config.sys_msg.choice is not None:
            if config.sys_msg.choice in ["no_persuasive"]:
                s_p_t_dir += "|" + f"sys_msg_{config.sys_msg.choice}"
            else:
                raise ValueError(f"The {config.sys_msg.choice} is not defined")
            
    if not config.target_lm.model_name.startswith("gpt-"):
        s_p_t_dir = os.path.join(s_p_t_dir,f"{config.target_lm.show_name}|max_new_tokens_{config.target_lm.generation_configs.max_new_tokens}")
    else:
        s_p_t_dir = os.path.join(s_p_t_dir,f"{config.target_lm.show_name}|temp={config.target_lm.temperature}_topp={config.target_lm.top_p}")
    Path(s_p_t_dir).mkdir(exist_ok= True, parents= True)

    save_path = os.path.join(s_p_t_dir,f"targetlm.jsonl")
    exist_lens = 0
    if os.path.exists(save_path):
        exist_lens = len(open(save_path).readlines())
        # if exist_lens != 0:
        #     print("file exists, so skip")
        #     return
            
    fp = jsonlines.open(save_path,mode = "a",flush= True)

    processed_data = get_data(config.data_args)
    print(len(processed_data),'len(processed_data)')
    exist_lens = exist_lens // config.prompter_lm.generation_configs.num_return_sequences
    if config.force_append:
        exist_lens = 0
    processed_data = processed_data[exist_lens:]
    print(len(processed_data),'len(processed_data)')
    if len(processed_data) == 0:
        print("Already get all data, skip")
        return

    print(OmegaConf.to_yaml(config), color='red')
    

    if config.target_lm.model_name.startswith("gpt-"):
        from utility import OpenaiModel
        target_lm_fn = OpenaiModel(model_name=config.target_lm.model_name,system_message = config.target_lm.system_message,template = config.target_lm.template, top_p = config.target_lm.top_p, temperature = config.target_lm.temperature)
    else:        
        target_lm_fn = create_targetlm(config)
    if config.sys_msg.choice is not None:
        target_lm_fn.replace_sys_msg(config.sys_msg[config.sys_msg.choice])

    prompter_lm_fn = None
    if config.prompt_way == "prompter":
        prompter_lm_fn = create_prompterlm(config)
        evaluate_fn(target_lm_fn,prompter_lm_fn,processed_data,config,fp)
    elif config.prompt_way == "own":
        with open(config.prompt_own_list) as f:
            prompt_own_list = json.load(f)

        print('len(prompt_own_list[config.prompt_own_list_name])',len(prompt_own_list[config.prompt_own_list_name]))
        for own_prompt in prompt_own_list[config.prompt_own_list_name]:
            print('own_prompt:\n',own_prompt)
            evaluate_fn(target_lm_fn,prompter_lm_fn,processed_data,config,fp,own_prompt)
    fp.close()
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time

    print(f"elapsed_time: {elapsed_time}秒")



@torch.no_grad()
def evaluate_fn(target_lm_fn,prompter_lm_fn,processed_data,config,fp,own_prompt = None):
    prompt_template = PROMPT_DICT[config.data_args.prompt_type]
    progress_keys = tqdm(processed_data, total=len(processed_data),desc="keys iteration")
    
    for batch in get_batch(processed_data,config.batch_size):
        batch = attack_collate_fn(batch)
        raw_q_s = batch["q"]
        processed_q_s = [" ".join([_] * config.q_rep) for _ in batch["q"]]
        if config.q_prefix.choice in ["long","short","medium"]:
            processed_q_s = [config.q_prefix[config.q_prefix.choice] + " " + _ for _ in processed_q_s]
        print("*"*50)
        print("This is prompter lm")
        print("Current Time:", datetime.now())

        # if config.q_s_position == "prompter_lm=processed|target_lm=processed":
        if config.q_s_position == "prompter_lm=raw|target_lm=processed":
            for_promptlm_q_s = raw_q_s
            for_targetlm_q_s = processed_q_s
        elif config.q_s_position == "prompter_lm=processed|target_lm=processed":
            for_promptlm_q_s = processed_q_s
            for_targetlm_q_s = processed_q_s
        else:
            raise ValueError("The q_s_position is not defined")
        
        if config.prompter_w_system:
            for_promptlm_q_s = [target_lm_fn.system_message + " " + _ for _ in for_promptlm_q_s]
            print("for_promptlm_q_s[0]:",for_promptlm_q_s[0])
        
        
        if config.prompt_way == "prompter":
            if config.data_args.prompt_type == "q_p":
                prompter_lm_inputs = [prompt_template.format(q = for_promptlm_q_s[index]) for index in range(len(for_promptlm_q_s))]
            prompt_lm_start_time = time.time()
            p_s = prompter_lm_fn.get_prompter_lm_generation(prompter_lm_inputs)
            prompt_lm_end_time = time.time()
            print('prompt_lm_time for one query',round((prompt_lm_end_time - prompt_lm_start_time)/len(for_promptlm_q_s),2))

            assert len(for_promptlm_q_s)*config.prompter_lm.generation_configs.num_return_sequences == len(p_s)
            print("prompter lm num_returns",config.prompter_lm.generation_configs.num_return_sequences)
            if config.prompt_concat > 1:
                _p_s = []
                for k in range(0,len(p_s),config.prompter_lm.generation_configs.num_return_sequences):
                    tmp_l = unique_random_concat_combinations(p_s[k:k+config.prompter_lm.generation_configs.num_return_sequences],concat_times=config.prompt_concat,sample_times=config.num_prompt_group)
                    _p_s.extend(tmp_l)
                p_s = _p_s
                repeat_for_targetlm_q_s = repeat_texts_l(for_targetlm_q_s,config.num_prompt_group)
                repeat_prompter_lm_inputs = repeat_texts_l(prompter_lm_inputs,config.num_prompt_group)
            else:
                repeat_for_targetlm_q_s = repeat_texts_l(for_targetlm_q_s,config.prompter_lm.generation_configs.num_return_sequences)
                repeat_prompter_lm_inputs = repeat_texts_l(prompter_lm_inputs,config.prompter_lm.generation_configs.num_return_sequences)
            assert len(repeat_for_targetlm_q_s) == len(p_s)


        elif config.prompt_way == "own":
            prompter_lm_inputs = [None] * len(for_promptlm_q_s)
            assert own_prompt is not None
            p_s = [own_prompt] * len(for_promptlm_q_s)
            print('config.prompt_way == "own"')
            print(p_s[0])
            repeat_prompter_lm_inputs = prompter_lm_inputs
            repeat_for_targetlm_q_s = for_targetlm_q_s
        else:
            raise NotImplementedError()
        
        gc.collect()
        torch.cuda.empty_cache() 

        if config.w_affirm_suffix:
            p_s = [_ + " " + config.affirm_suffix for _ in p_s]

        print("*"*50)
        print("This is target lm")
        print("Current Time:", datetime.now())
        target_lm_generations = target_lm_fn.get_target_lm_generation(repeat_for_targetlm_q_s,p_s)
        gc.collect()
        torch.cuda.empty_cache() 

    
        ppl_q_p = [None for _ in range(len(repeat_for_targetlm_q_s))]
        if config.ppl == True:
            print("This is ppl run")
            ppl_q_p = target_lm_fn.ppl_run(repeat_for_targetlm_q_s,p_s)


        for i in range(len(target_lm_generations)):
            save_d = dict(q = repeat_for_targetlm_q_s[i],p = p_s[i],target_lm_generation = target_lm_generations[i],ppl_q_p = ppl_q_p[i],prompter_lm_inputs = repeat_prompter_lm_inputs[i])
            fp.write(save_d)

        progress_keys.update(config.batch_size)
        gc.collect()
        torch.cuda.empty_cache() 

if __name__ == "__main__":
    main()

