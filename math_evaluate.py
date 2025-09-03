import copy
import json
import os
import re
import sys
import argparse
import fire
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_CACHE'] = './model'
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from Direc_LoRA import *

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def main():
    """主函数，移除了未使用的参数"""
    args = parse_args()# 调用 parse_args 函数解析命令行参数，存储在 args 变量中。

    def evaluate(
        instructions,#这是要输入evaluate函数的参数，没有输入的就采用默认值
        input=None,#test测试集本身就没有input，可以理解为input放入到了instruction中
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        #对输入的参数进行操作，进一步整理然后进行生成和评估
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        #配置生成器参数
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,# 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        with torch.no_grad():
            #利用上面处理过的参数进行推理生成结果
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        s = generation_output.sequences# 获取生成的 token 序列。
        #.sequences 属性包含模型生成的所有token序列（每个序列对应一个生成的回答）
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)#使用tokenizer的batch_decode方法批量将token序列解码为可读文本
        #skip_special_tokens=True 表示跳过特殊token（如<pad>, <eos>等标记）
        # return [output for output in outputs]
        return [output.lower().split("### response:")[1].strip() for output in outputs]
        #split("### Response:") 以"### Response:"为分隔符分割字符串
        #[1] 取分割后的第二部分（索引1的内容），即模型生成的回答部分
        #strip() 去除首尾空白字符

    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)#加载数据
    batches = create_batch(dataset, args.batch_size)#将数据划分为batch
    tokenizer, model = load_model(args)#加载模型和分词器

    # 移动权重合并逻辑到模型加载后，只执行一次
    if args.adapter == "LoRA" or args.adapter == "DoRA":#修改需要把Dislora添加
        merge_adapter_weights(model, args)

    total = len(batches)# 获取批次总数。
    correct = 0# 初始化正确预测的计数器。
    current = 0# 初始化当前处理的样本计数器。
    output_data = []# 初始化输出数据列表，用于存储预测结果。
    pbar = tqdm(total=total, desc="Processing batches")#创建进度条
    #使用两个for循环遍历所有样本数据
    for idx, batch in enumerate(batches):# 遍历所有批次
        try:
            current += len(batch)#len(batch) 计算当前批次(batch)中的样本数量
            instructions = [data.get('instruction') for data in batch]#instruction形成一个列表

            outputs = evaluate(instructions)#输入只有instruction，其他参数使用默认值，最终得到输出结果

            for data, output in zip(batch, outputs):#对每条数据的结果进行判断
                label = data.get('answer')
                flag = False
                
                if args.dataset.lower() in ['aqua']:
                    predict = extract_answer_letter(args, output)
                    if label == predict:
                        correct += 1
                        flag = True
                else:
                    if isinstance(label, str):
                        try:
                            label = float(label)
                        except ValueError:
                            print(f"Warning: Cannot convert label '{label}' to float")
                            continue
                    
                    predict = extract_answer_number(args, output)
                    if predict != float('inf') and abs(label - predict) <= 0.001:
                        correct += 1
                        flag = True
                
                new_data = copy.deepcopy(data)
                new_data['output_pred'] = output
                new_data['pred'] = predict
                new_data['flag'] = flag
                output_data.append(new_data)
                
                print(' ')
                print('---------------')
                print(f"Instruction: {data['instruction']}")
                print(f"Output: {output}")
                print(f"Prediction: {predict}")
                print(f"Label: {label}")
                print('---------------')

            print(f'\rtest:{idx + 1}/{total} | accuracy {correct}/{current} = {correct / current:.4f}')
            
            # 定期保存结果
            with open(save_file, 'w+') as f:
                json.dump(output_data, f, indent=4)
            
            pbar.update(1)
            
        except Exception as e:
            print(f"Error processing batch {idx}: {e}")
            continue
            
    pbar.close()
    print('\n')
    print('Test finished')
    print(f'Final accuracy: {correct}/{current} = {correct / current:.4f}')

def merge_adapter_weights(model, args):#修改，需要考虑到Dislora，可以让ai写
    """合并适配器权重到原始权重中"""
    print("Merge LoRA/DoRA weights into the original weights")
    key_list = [(key, module) for key, module in model.model.named_modules()]
    
    for key, module in key_list:
        if isinstance(model.peft_config.target_modules, str):
            target_module_found = re.fullmatch(model.peft_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.target_modules)

        if args.adapter == "DoRA":
            if model.peft_config.Wdecompose_target_modules != None:
                if isinstance(model.peft_config.Wdecompose_target_modules, str):
                    wdecompose_target_module_found = re.fullmatch(model.peft_config.Wdecompose_target_modules, key)
                else:
                    wdecompose_target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.Wdecompose_target_modules)
            else:
                wdecompose_target_module_found = False
        else:
            wdecompose_target_module_found = False

        if target_module_found or wdecompose_target_module_found:
            print(f"found {key}")
            module.merge_weights = True
            module.train(mode=False)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)  # 使用makedirs支持递归创建
    return

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501

def load_data(args) -> list:
    file_path = f'dataset/{args.dataset}/test.json'#数据集加载逻辑，规定了加载数据集的路径
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

def create_batch(dataset, batch_size):
    """创建批次数据"""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    batches = []
    num_batch = len(dataset) // batch_size if len(dataset) % batch_size == 0 else len(dataset) // batch_size + 1
    
    for i in range(num_batch):
        batch = dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    
    print(f"Created {len(batches)} batches with batch_size={batch_size}")
    return batches

def parse_args():
    parser = argparse.ArgumentParser(description='Math reasoning evaluation with batch processing')
    parser.add_argument('--dataset', 
                        choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True,
                        help='Dataset to evaluate on')
    parser.add_argument('--model', 
                        type=str, 
                        required=True,
                        help='Base model to use')
    parser.add_argument('--adapter', 
                        choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix', 'mylora'],
                        required=True,
                        help='Adapter type')
    parser.add_argument('--base_model', 
                        required=True,
                        help='Path to base model')
    parser.add_argument('--lora_weights', 
                        required=True,
                        help='Path to LoRA weights')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--load_8bit', 
                        action='store_true', 
                        default=False,
                        help='Load model in 8bit mode')
    return parser.parse_args()

def load_model(args) -> tuple:
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'Cannot find base model name by the value: {args.model}')
    
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'Cannot find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    
    # 加载tokenizer
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置left padding
    tokenizer.padding_side = 'left'
    
    # 针对 mylora 特殊处理
    if args.adapter == "mylora":
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},  # 明确指定使用GPU 0 修改
                trust_remote_code=True,
            )
            config = Direc_config(
                r=16,#评估的时候修改
                target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],  # Qwen2.5 的注意力模块
                lora_alpha=24,#评估的时候修改
                lora_dropout=0.05,
                fan_in_fan_out=False,
                bias="none",
                task_type="CAUSAL_LM",
                warmup_steps=10,
                s_tsd=8,#评估的时候修改
                ortho_lambda=1.0,
                prefer_small_sigma=True
            )
            model = Direc_Model(model, config)
            model.load_module(lora_weights)
            print(model)
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
    else:
        # 加载模型
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map={"": 0},  # 明确指定使用GPU 0  修改device_map="auto"
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"": 0}
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, 
                device_map={"": device}, 
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
        
        # 设置token IDs
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            model.config.bos_token_id = tokenizer.bos_token_id
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            model.config.eos_token_id = tokenizer.eos_token_id
        
        if device == "cpu":
            if not load_8bit:
                model.half()
            model.eval()
            if torch.__version__ >= "2" and sys.platform != "win32":
                model = torch.compile(model)
    
    return tokenizer, model

def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        try:
            pred_answer = float(pred[-1])
        except ValueError:
            return float('inf')
    else:
        raise NotImplementedError('Not support dataset: {}'.format(dataset))
    
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')    
    return pred_answer

def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        return pred_answers[0]
    else:
        return ''

if __name__ == "__main__":
    fire.Fire(main)