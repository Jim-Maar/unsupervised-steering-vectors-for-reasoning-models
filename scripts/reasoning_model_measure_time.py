# %%
import argparse
import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation import StoppingCriteriaList

def parse_args():
    parser = argparse.ArgumentParser(description='Run reasoning model and measure time')
    
    # Model selection
    parser.add_argument('--model_size', type=int, default=1, choices=[1, 7, 14],
                        help='Model size in billions (1, 7, or 14)')
    
    # Other parameters
    parser.add_argument('--source_layer', type=int, default=None, help='Source layer')
    parser.add_argument('--target_layer', type=int, default=None, help='Target layer')
    parser.add_argument('--normalization', type=int, default=12, help='Normalization value')
    parser.add_argument('--orthogonal_vectors', type=bool, default=True, help='Use orthogonal vectors')
    parser.add_argument('--num_vectors', type=int, default=4, help='Number of vectors')
    parser.add_argument('--token_idxs_start', type=int, default=-2, help='Start index for token slice')
    parser.add_argument('--token_idxs_end', type=int, default=None, help='End index for token slice')
    parser.add_argument('--num_steps', type=int, default=400, help='Number of steps')
    parser.add_argument('--power', type=int, default=4, help='Power value')
    parser.add_argument('--torch_seed', type=int, default=325, help='Torch seed')
    parser.add_argument('--save_dir', type=str, default="/home", help='Directory to save results')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum new tokens to generate')
    parser.add_argument('--question', type=str, default="Whats 2 + 3?", help='Question to answer')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for generation')
    parser.add_argument('--do_sample', action='store_true', help='Whether to use sampling for generation')
    
    return parser.parse_args()

def get_model_name(model_size):
    if model_size == 1:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    return f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{model_size}B"

def get_full_prompt(prompt):
    full_prompt = prompt + "\nPlease Reason step by step, and put your final awnser within \\boxed{}.<think>\n"
    return full_prompt

def get_response(question, model, tokenizer, max_new_tokens=1500, temperature=0.5, do_sample=False):
    full_prompt = get_full_prompt(question)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([
            lambda input_ids, scores, **kwargs: input_ids[0][-6:].tolist() == tokenizer.encode("<|im_end|>")[-6:]
        ])
    )
    end_time = time.time()
    generation_time = end_time - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reasoning = response[len(full_prompt):]
    # awnser = x with \boxed{x} in reasoning
    try:
        awnser = re.search(r'\\boxed\{(.*?)\}', reasoning).group(1)
    except:
        awnser = "No awnser found"
    return full_prompt, response, reasoning, awnser, generation_time

def main():
    args = parse_args()
    
    # Set parameters from arguments
    MODEL_NAME = get_model_name(args.model_size)
    TOKENIZER_NAME = MODEL_NAME
    
    SOURCE_LAYER = args.source_layer
    TARGET_LAYER = args.target_layer
    NORMALIZATION = args.normalization
    ORTHOGONAL_VECTORS = args.orthogonal_vectors
    NUM_VECTORS = args.num_vectors
    TOKEN_IDXS = slice(args.token_idxs_start, args.token_idxs_end)
    NUM_STEPS = args.num_steps
    POWER = args.power
    POWER_Q = POWER
    
    TORCH_SEED = args.torch_seed
    if TORCH_SEED is not None:
        torch.manual_seed(TORCH_SEED)
    
    SAVE_DIR = args.save_dir
    MAX_NEW_TOKENS = args.max_new_tokens
    
    question = args.question
    
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto", 
        trust_remote_code=True,
    )
    
    full_prompt, response, reasoning, awnser, generation_time = get_response(
        question, 
        model, 
        tokenizer, 
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=args.temperature,
        do_sample=args.do_sample
    )
    
    print("=== Question ===")
    print(question)
    print("=== Reasoning ===")
    print(reasoning)
    print("=== Awnser ===")
    print(awnser)
    print("=== Generation Time ===")
    print(f"{generation_time:.2f} seconds")

if __name__ == "__main__":
    main()