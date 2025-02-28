import pickle
from transformers.generation import StoppingCriteriaList
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys
import os
sys.path.append('..')  # Adds parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.unsupervised_steering import SteeredModel
import json
import plotly.graph_objects as go
import numpy as np
TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
steered_model = pickle.load(open("results/steering_vectors/model_1.5B/norm_7.5_src_4_tgt_7_lr_0.01/steered_model.pkl", "rb"))


model = steered_model.model


# load questions.json
questions = json.load(open("questions.json", "r"))
questions.keys()


question = questions['cube_probability_3']


def get_full_prompt(prompt):
    full_prompt = prompt + "\nPlease Reason step by step, and put your final awnser within \\boxed{}.<think>\n"
    return full_prompt

def get_response(question, max_new_tokens=1500, temperature=0.5, do_sample=False):
    full_prompt = get_full_prompt(question)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=[StoppingCriteriaList([
            lambda input_ids, scores, **kwargs: input_ids[0][-6:].tolist() == tokenizer.encode("<|im_end|>")[-6:]
        ])]
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reasoning = response[len(full_prompt):]
    # awnser = x with \boxed{x} in reasoning
    try:
        awnser = re.search(r'\\boxed\{(.*?)\}', reasoning).group(1)
    except:
        awnser = "No awnser found"
    return full_prompt, response, reasoning, awnser


steered_model.zero_steering_vector()
try:
    with open("response.txt", "r") as f:
        response = f.read()
except FileNotFoundError:
    _, response, _, _ = get_response(question)
    with open("response.txt", "w") as f:
        f.write(response)
print(response)

# Interesting Vectors
# - 1: lots of but wait, 34
# - 2: output a list 1. 2. 3. etc
# - 3, 11: a lot of wait (I think some vectors want to keep on thinking)
# - 4: Backtracking to the last step (e.g. repeat last step?)
# - 6: A lot of: Do I interpret this correctly?
# - 8, 9, 10, 16: confident
# - 12: a lot of equations
# - 13, 14, 15: confident + lists
# - 23, 59: Confusion (Let me try to think of it as a series of steps / equations)
# - 24: python code
# - 25: so ...
# - 32: recap
# - 33: Alright, so, let me try to figure this out. Okay, so, let me try to work through this.
# - 37: chinese (correct solution)
# - 56: So the awnser is ...
# # Explore Backtracking Vector (32)

TARGET_LAYER = 20

VECTOR_INDEX = 4

import torch as t


model_inputs = tokenizer([response], return_tensors="pt", padding=False).to(model.device)

# Get unsteered target
steered_model.zero_steering_vector()
unsteered_target = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states[TARGET_LAYER].detach().to("cpu")

# Clear cache
t.cuda.empty_cache()

# Set steering vector and get steered target
steered_model.set_steering_vector(VECTOR_INDEX)
with t.no_grad():
    target = steered_model.model(model_inputs["input_ids"], output_hidden_states=True).hidden_states[TARGET_LAYER].detach().to("cpu")

# Calculate loss
loss = (unsteered_target-target).pow(4).sum(dim=-1).pow(1/4).squeeze()


# Create the simplest possible heatmap visualization
import matplotlib.pyplot as plt

# Convert tensor to numpy
loss_np = loss.detach().cpu().numpy()

# Create a simple heatmap with matplotlib
plt.figure(figsize=(12, 3))
plt.imshow([loss_np], aspect='auto', cmap='RdYlGn')
plt.colorbar(label='Loss Value')
plt.title('Loss Barcode Visualization')
plt.xlabel('Token Position')
plt.yticks([])  # Hide y-axis ticks
plt.tight_layout()
# plt.show()

# Alternative: save as image file
plt.savefig('loss_barcode.png')
plt.close()


# If matplotlib doesn't work, try the absolute simplest plotly version
import plotly.express as px

# Create the simplest possible plotly heatmap
fig = px.imshow([loss_np], 
                color_continuous_scale='RdYlGn',
                labels=dict(color="Loss Value"))

fig.update_layout(
    title="Loss Barcode Visualization",
    xaxis_title="Token Position",
    height=300,
    width=1200,
)

# fig.show()

# Create an HTML file with colored text based on loss values
import matplotlib.colors as mcolors
import html

def create_colored_text_html():
    # Tokenize the response to get individual tokens
    tokens = tokenizer.tokenize(response)
    
    # Make sure the number of tokens matches the loss vector length
    # If not, we'll use the minimum length to avoid index errors
    min_length = min(len(tokens), len(loss_np))
    
    # Create a custom colormap (red for low loss, green for high loss)
    cmap = mcolors.LinearSegmentedColormap.from_list("RdYlGn", ["red", "yellow", "green"])
    
    # Normalize loss values to range 0-1 for color mapping
    if min_length > 0:
        loss_min = np.min(loss_np[:min_length])
        loss_max = np.max(loss_np[:min_length])
        normalized_loss = (loss_np[:min_length] - loss_min) / (loss_max - loss_min) if loss_max > loss_min else np.zeros(min_length)
    else:
        normalized_loss = []
    
    # Start building the HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Response with Loss Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            .text-content {
                white-space: pre-wrap;
                margin: 20px 0;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
                overflow-wrap: break-word;
            }
            .legend {
                height: 30px;
                width: 100%;
                margin: 20px 0;
                position: relative;
                background: linear-gradient(to right, red, yellow, green);
                border-radius: 3px;
            }
            .legend-label {
                position: absolute;
                font-size: 12px;
            }
            .legend-label.left {
                left: 5px;
            }
            .legend-label.right {
                right: 5px;
            }
            .explanation {
                font-size: 14px;
                color: #666;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Response with Loss Visualization</h1>
            <div class="explanation">
                Text is colored based on loss values: <span style="color: red;">Red = Low Loss</span>, 
                <span style="color: yellow;">Yellow = Medium Loss</span>, 
                <span style="color: green;">Green = High Loss</span>
            </div>
            <div class="text-content">
    """
    
    # Add each token with its color
    for i in range(min_length):
        # Get the color for this token based on its normalized loss
        color = mcolors.to_hex(cmap(normalized_loss[i]))
        
        # Get the decoded token text and escape any HTML special characters
        token_text = tokenizer.convert_tokens_to_string([tokens[i]])
        escaped_text = html.escape(token_text)
        
        # Add the colored token to the HTML
        html_content += f'<span style="color: {color};">{escaped_text}</span>'
    
    # Complete the HTML
    html_content += """
            </div>
            <div class="legend">
                <span class="legend-label left">Low Loss</span>
                <span class="legend-label right">High Loss</span>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    output_file = f"response_visualization_{VECTOR_INDEX}.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML visualization created: {output_file}")
    
    # Try to open the HTML file in the default browser
    try:
        import webbrowser
        webbrowser.open(output_file)
    except:
        print("Could not automatically open the HTML file. Please open it manually.")

# Call the function to create the HTML file
create_colored_text_html()