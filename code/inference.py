import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

import sys
import torch
import logging
import argparse
from prettytable import PrettyTable
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set Paths
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

from settings import *

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    assert backbone in ['opt-6.7b', 'llama-7b', 'llama2-7b', 'llama3-8b', 'mistral-7b']

    if backbone == 'opt-6.7b':
        model_path = '../../models/opt-6.7b'
        lora_weight = f'./csesfp-{backbone}-lora-sth-sum/checkpoint-2550'
    elif backbone == 'llama2-7b':
        model_path = '../../models/llama-2-7b-hf'
        lora_weight = f'./csesfp-{backbone}-lora-eol-sum/checkpoint-3900'
    elif backbone == 'llama3-8b':
        model_path = '../../models/llama3-8b'
        lora_weight = f'./csesfp-{backbone}-lora-eol-sum/checkpoint-3900'
    elif backbone == 'mistral-7b':
        model_path = '../../models/mistral-7b-v0.1'
        lora_weight = f'./csesfp-{backbone}-lora-sth-sum/checkpoint-1800'
    else:
        raise ValueError('Unknown backbone model')
    
    args_list = ['--model_name_or_path', model_path,
                 '--mode', 'test',
                 '--mask_embedding_sentence',
                 '--mask_embedding_sentence_template', manual_template,
                 '--lora_weight', lora_weight,
                 '--load_kbit', '16']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--load_kbit', type=int,
                        choices=[4, 8, 16],
                        default=8,
                        help="Load model in kbit")
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--lora_weight', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--tensor_parallel', action='store_true')

    args = parser.parse_args(args_list)

    if args.tensor_parallel:
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    elif args.load_kbit == 4:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=True, 
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.float16,
            device_map='auto',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     trust_remote_code=True,
                                                     load_in_8bit=args.load_kbit == 8,)

    if args.lora_weight is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            args.lora_weight,
            torch_dtype=torch.float16,
            device_map={'': 0},
        )
        if args.load_kbit == 4:
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.float16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if ('lm_head' in name or 'embed_tokens' in name) and hasattr(module, 'weight'):
                    if 'opt' in args.model_name_or_path:
                        module = module.to(torch.float32)
                    else:
                        module = module.to(torch.float16)
    
    if 'llama' in args.model_name_or_path and 'llama3' not in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = '</s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32, 'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':16}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        if args.mask_embedding_sentence and args.mask_embedding_sentence_template is not None:
            template = args.mask_embedding_sentence_template
            template = template.replace('_', ' ').replace('*sep+*', '').replace('*cls*', '')
            
            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': 
                    s += '.'
                s = s.replace('"', '\'')
                if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
                sentences[i] = template.replace('*sent 0*', s).strip()

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings
        with torch.no_grad():
            hidden_states = model(output_hidden_states=True, return_dict=True, **batch).hidden_states

            if args.avg:
                last_layer = hidden_states[-1]
                attention_mask = batch['attention_mask'].unsqueeze(-1).expand(last_layer.shape)
                outputs = (last_layer * attention_mask).mean(1)
            else:
                outputs = hidden_states[-1][:, -1, :]

            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()

            return outputs.cpu()

    results = {}
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        scores = []
        task_names = []
        for task in ['STSBenchmark']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)
    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        scores = []
        task_names = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        
        # write results and template to file
        if args.mask_embedding_sentence_template is not None and args.task_set != 'transfer':
            with open('./sts-org-results', 'a') as f:
                bits = f'{args.load_kbit}bit'
                model_name = args.model_name_or_path.split('/')[-1] + f'({bits})'
                f.write(args.mask_embedding_sentence_template + ' ' + model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')


if __name__ == "__main__":
    main()
