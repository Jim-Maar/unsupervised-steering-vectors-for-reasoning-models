# %%
import argparse
import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from transformers.generation import StoppingCriteriaList
import sys
import os
import pickle
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import steering model
from src.unsupervised_steering import SteeredModel

import torch
import tqdm
torch.manual_seed(325)

def parse_args():
    parser = argparse.ArgumentParser(description='Run reasoning model and measure time')
    
    # Model selection
    parser.add_argument('--model_size', type=int, default=1, choices=[1, 7, 14],
                        help='Model size in billions (1, 7, or 14)')
    
    # Other parameters
    parser.add_argument('--source_layer', type=int, default=None, help='Source layer')
    parser.add_argument('--target_layer', type=int, default=None, help='Target layer')
    parser.add_argument('--normalization', type=float, default=12, help='Normalization value')
    parser.add_argument('--orthogonal_vectors', type=bool, default=True, help='Use orthogonal vectors')
    parser.add_argument('--num_vectors', type=int, default=4, help='Number of vectors')
    parser.add_argument('--token_idxs_start', type=int, default=-2, help='Start index for token slice')
    parser.add_argument('--token_idxs_end', type=int, default=None, help='End index for token slice')
    parser.add_argument('--num_steps', type=int, default=400, help='Number of steps')
    parser.add_argument('--power', type=int, default=4, help='Power value')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training steering vectors')
    parser.add_argument('--torch_seed', type=int, default=325, help='Torch seed')
    parser.add_argument('--save_dir', type=str, default="/home", help='Directory to save results')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum new tokens to generate')
    parser.add_argument('--question', type=str, default="Whats 2 + 3?", help='Question to answer')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for generation')
    parser.add_argument('--do_sample', action='store_true', help='Whether to use sampling for generation')
    parser.add_argument('--use_8bit', action='store_true', help='Whether to use 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true', help='Whether to use 4-bit quantization')
    parser.add_argument('--offload_buffers', action='store_true', help='Whether to offload buffers to CPU')
    
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

def train_steering_vectors(
    model,
    tokenizer,
    response,
    max_new_tokens,
    source_layer=4,
    target_layer=7,
    normalization=12,
    orthogonal_vectors=True,
    num_vectors=4,
    num_steps=400,
    power=4,
    power_q=4,
    learning_rate=0.01
):
    token_idxs = slice(-max_new_tokens, None)
    steered_model = SteeredModel(
        model,
        tokenizer,
        source_layer_idx=source_layer,
        target_layer_idx=target_layer,
        target_token_idxs=token_idxs,
        normalization=normalization,
        orthogonal_vectors=orthogonal_vectors,
        num_steps=num_steps,
        power=power,
        q=power_q,
        lr=learning_rate,
    )
    steered_model.train([response], num_vectors, more_tqdm=True)
    return steered_model

def log_losses(steered_model):
    print("="*100)
    print("Losses:")
    for i in range(steered_model.num_vectors):
        print(f"Vector {i}: {steered_model.losses_all[i][-1]}")
    print("="*100)

def evaluate_steering_vectors(model, tokenizer, steered_model, full_prompt, max_new_tokens=512, do_sample=False):
    # prompt = EXAMPLES[0]
    prompt = full_prompt
    # prompt = get_full_prompt("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
    completions_steered = []
    for i in tqdm.tqdm(range(steered_model.num_vectors)):
        steered_model.set_steering_vector(i)
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        completion = tokenizer.batch_decode(generated_ids, 
                            skip_special_tokens=True)[0]
        completions_steered.append(completion)
    for i, completion in enumerate(completions_steered):
        print("====Vector %d :=========\n" % i)
        print(completion)
    return completions_steered

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
    LEARNING_RATE = args.learning_rate
    
    TORCH_SEED = args.torch_seed
    if TORCH_SEED is not None:
        torch.manual_seed(TORCH_SEED)
    
    SAVE_DIR = args.save_dir
    MAX_NEW_TOKENS = args.max_new_tokens
    
    question = args.question
    
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    
    # Configure quantization and offloading options
    quantization_config = None
    
    # Only apply memory-saving options for larger models (7B and 14B)
    if args.model_size >= 7:
        if args.use_8bit:
            print("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        elif args.use_4bit:
            print("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Configure device map with offloading if needed
        device_map = "auto"
        if args.offload_buffers:
            print("Using offload buffers")
            device_map = {"": "cuda:0"}
            
        # Load the model with memory-saving configuration for larger models
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            offload_buffers=args.offload_buffers,
            torch_dtype=torch.float16,  # Use half precision
        )
    else:
        # For smaller models, use standard loading without memory optimizations
        print("Loading smaller model without memory optimizations")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Print memory usage information
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    full_prompt, response, reasoning, awnser, generation_time = get_response(
        question, 
        model, 
        tokenizer, 
        max_new_tokens=MAX_NEW_TOKENS,
        # temperature=args.temperature,
        do_sample=False
    )
    
    print(f"Training steering vectors...")
    steered_model = train_steering_vectors(
        model,
        tokenizer,
        response,
        MAX_NEW_TOKENS,
        source_layer=SOURCE_LAYER,
        target_layer=TARGET_LAYER,
        normalization=NORMALIZATION,
        orthogonal_vectors=ORTHOGONAL_VECTORS,
        num_vectors=NUM_VECTORS,
        num_steps=NUM_STEPS,
        power=POWER,
        power_q=POWER_Q,
        learning_rate=LEARNING_RATE
    )
    log_losses(steered_model)
    
    print(f"Evaluating steering vectors...")
    completions = evaluate_steering_vectors(
        model, 
        tokenizer, 
        steered_model, 
        full_prompt, 
        max_new_tokens=MAX_NEW_TOKENS * 3,
        do_sample=False
    )

    # save steered model using pickle
    with open(os.path.join(SAVE_DIR, "steered_model.pkl"), "wb") as f:
        pickle.dump(steered_model, f)

if __name__ == "__main__":
    main()