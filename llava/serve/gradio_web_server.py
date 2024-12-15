import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, masked_image, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:3], image_process_mode)
    state.skip_next = False
    
    state.messages[-2] = [
        state.messages[-2][0], 
        (state.messages[-2][1][0],masked_image, state.messages[-2][1][2], state.messages[-2][1][3])  # Create a new tuple with the updated image
    ]

    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5



def add_text_wCLS(state, text, masked_image, image_process_mode, imagebox, request: gr.Request):
    logger.info(f"add_text_withcls. ip: {request.client.host}. len: {len(text)}")
    
    if len(text) <= 0 and masked_image is None and imagebox is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]
    if imagebox is not None:
        text = text[:1200]
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, masked_image, imagebox, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    state.cls=True
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def add_text(state, text, masked_image, image_process_mode, imagebox, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    
    if len(text) <= 0 and masked_image is None and imagebox is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]
    if imagebox is not None:
        text = text[:1200]
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, masked_image, imagebox, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    state.cls=False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, raw_tokens, request: gr.Request):
    cls_flag = state.cls
    print(f">>>>>>>>CLS_FLAG_{cls_flag}")
    select_tokens = raw_tokens.strip('[]')
    select_tokens = list(map(int, select_tokens.split()))
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
        "select_tokens":select_tokens,
        "cls_flag":cls_flag,
    }
    logger.info(f"==== request ====\n{pload}")
    state.cls=cls_flag
    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=20)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# VisionZip: Longer is Better but Not Necessary in Vision Language Models
[[Code](https://github.com/dvlab-research/VisionZip)] [[Demo-Visualizer](http://202.104.135.156:11030)] [[Usage-Video](https://youtu.be/9GNIJy4U6-k?si=jcWIJ2O0IjB4aamm)] [[Intro-Video](https://youtu.be/sytaAzmxxpo?si=IieArmQ7YNf2dVyM)]

This demo allows users to manually select which visual tokens to send to the LLM to observe how different visual tokens impact the final response.

### Instructions:
1. Upload an image.
2. Select the visual tokens.
3. Generate the answer.

For a step-by-step guide, refer to the [Usage Video](https://youtu.be/9GNIJy4U6-k?si=jcWIJ2O0IjB4aamm).
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the [License](https://github.com/dvlab-research/VisionZip/blob/main/LICENSE) of VisionZip, model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""
import gradio as gr
import numpy as np
# Function to capture coordinates of the drawing on the image
import numpy as np
from PIL import Image, ImageDraw


def create_mask(image, grid_vet):
    if image is None:
        return None
    # Resize the image to 336x336
    image = image.resize((336, 336))  
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    grid_size = 14
    grid_count = 24

    for i in range(grid_count):
        for j in range(grid_count):
            # Calculate the bounding box of each grid cell
            left = j * grid_size
            top = i * grid_size
            right = left + grid_size
            bottom = top + grid_size
            
            # If the value in grid_vet is 0, draw a white mask with 70% transparency
            if grid_vet[i][j] == 0:
                draw.rectangle([left, top, right, bottom], fill=(255, 255, 255, 178))  # 70% transparency

    # Composite the image with the overlay
    final_image = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    # Convert back to RGB if needed (remove alpha channel)
    return final_image.convert('RGB')

def capture_coordinates(image, drawing):
    outputs = drawing['layers'][0][:, :, -1]  # Alpha channel (transparency)
    
    non_zero_pixels = np.argwhere(outputs > 0)  # Non-transparent pixels

    grid_size = 14
    grid_count = 24

    grid_vector = np.zeros((grid_count, grid_count), dtype=int)

    for y, x in non_zero_pixels:
        grid_x = x // grid_size  
        grid_y = y // grid_size 
        grid_vector[grid_y, grid_x] = 1 
        
    grid_vector_flat = grid_vector.flatten()
    index = np.where(grid_vector_flat==1)[0]
    final_image = create_mask(image,grid_vector)


    return str(index),final_image

def calculate_dominant_tokens_192(image, model_selector,state):
    token_num=192
    model_name = model_selector

    controller_url = args.controller_url

    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    
    pload = {
        "images": [state.process_image(image,  "Default")],
        "token_num":token_num,
    }

    response = requests.post(worker_addr + "/worker_get_visonzip",json=pload, timeout=20)
    
    select_idx = response.json()['token_idx'][0]
    grid_count=24
    grid_vector = np.zeros((grid_count, grid_count), dtype=int)
    for idx in select_idx:
        row = idx // grid_count 
        col = idx % grid_count  
        grid_vector[row, col] = 1 

    final_image = create_mask(image,grid_vector)
    select_idx = np.array(select_idx)
    
    return str(select_idx), final_image

def calculate_dominant_tokens_128(image, model_selector,state):
    ## Call the Model to get the visionzip
    ## use the index to get the grid vector
    token_num=128
    model_name = model_selector

    controller_url = args.controller_url

    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    
    pload = {
        "images": [state.process_image(image,  "Default")],
        "token_num":token_num,
    }

    response = requests.post(worker_addr + "/worker_get_visonzip",json=pload, timeout=20)
    
    select_idx = response.json()['token_idx'][0]
    grid_count=24
    grid_vector = np.zeros((grid_count, grid_count), dtype=int)
    for idx in select_idx:
        row = idx // grid_count 
        col = idx % grid_count  
        grid_vector[row, col] = 1 

    final_image = create_mask(image,grid_vector)
    select_idx = np.array(select_idx)
    
    return str(select_idx), final_image

def calculate_dominant_tokens_64(image, model_selector,state):
    ## Call the Model to get the visionzip
    ## use the index to get the grid vector
    token_num=64
    model_name = model_selector

    controller_url = args.controller_url

    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    
    pload = {
        "images": [state.process_image(image,  "Default")],
        "token_num":token_num,
    }

    response = requests.post(worker_addr + "/worker_get_visonzip",json=pload, timeout=20)
    
    select_idx = response.json()['token_idx'][0]
    grid_count=24
    grid_vector = np.zeros((grid_count, grid_count), dtype=int)
    for idx in select_idx:
        row = idx // grid_count 
        col = idx % grid_count  
        grid_vector[row, col] = 1 

    final_image = create_mask(image,grid_vector)
    select_idx = np.array(select_idx)
    
    return str(select_idx), final_image

from PIL import Image

# Function to resize the image to 336x336 and return it
def resize_image(image):
    if image is None:
        return None
    return image.resize((336, 336))

def default_img(image):
    grid_count = 24
    grid_vector = np.zeros((grid_count, grid_count), dtype=int)
    default_image = create_mask(image,grid_vector)
    return default_image

def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER (No CLS)", container=False)
    
    with gr.Blocks(title="VisionZip", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil", label="Upload Image", interactive=True)
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)
                
                
                sketchbox = gr.Sketchpad(
                    label="Select on the Image", 
                    height=250, 
                    brush=gr.Brush(
                        colors=["#FF0000", "#0000FF", "#00FF00", "#FFFF00"],  # Red, Blue, Green, Yellow, Black
                        default_color="#FF0000",
                        color_mode="defaults"  # Fixed color mode (can also be "dynamic" for multiple colors)
                    )
                )

                get_coordinates_btn = gr.Button(value="Get the Selected Tokens")
                with gr.Row():  # Add this new row to hold both buttons side by side
                    get_dominant64_btn = gr.Button(value="Get 64 Dominant Tokens")
                    get_dominant128_btn = gr.Button(value="Get 128 Dominant Tokens")
                    get_dominant192_btn = gr.Button(value="Get 192 Dominant Tokens")

                coordinates_output = gr.Textbox(label="Select Tokens Index", interactive=False)

                # Add the new image output area
                masked_image_output = gr.Image(type="pil", label="Selected Visual Tokens", interactive=False)

                get_coordinates_btn.click(
                    capture_coordinates, 
                    [imagebox, sketchbox], 
                    [coordinates_output,masked_image_output]
                )
                get_dominant64_btn.click(
                    calculate_dominant_tokens_64,
                    [imagebox,model_selector,state],
                    [coordinates_output,masked_image_output]

                )
                get_dominant128_btn.click(
                    calculate_dominant_tokens_128,
                    [imagebox,model_selector,state],
                    [coordinates_output,masked_image_output]

                )
                get_dominant192_btn.click(
                    calculate_dominant_tokens_192,
                    [imagebox,model_selector,state],
                    [coordinates_output,masked_image_output]

                )
                # Link the uploaded image to the sketchbox with resizing
                imagebox.change(fn=lambda img: resize_image(img), inputs=imagebox, outputs=sketchbox)
                # imagebox.change(fn=lambda img: default_img(img), inputs=imagebox, outputs=masked_image_output)

                imagebox.change(
                    fn=lambda img: [default_img(img), ""] ,  # Reset coordinates_output to empty string
                    inputs=imagebox, 
                    outputs=[masked_image_output, coordinates_output]  # Include coordinates_output in outputs
                )
                
                # Example input examples
                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                    [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens")

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="LLaVA Chatbot",
                    height=650,
                    layout="panel",
                )
                with gr.Row():
                    with gr.Column(scale=7):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        CLS_btn = gr.Button(value="Add CLS", variant="primary")
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="No CLS", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )

        regenerate_btn.click(
            regenerate,
            [state, masked_image_output, image_process_mode],  # No need for imagebox here, you already have masked_image_output
            [state, chatbot, textbox] + btn_list  # Use masked_image_output in the outputs
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, coordinates_output],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, masked_image_output, image_process_mode, imagebox],
            [state, chatbot, textbox] + btn_list, 
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, coordinates_output],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        submit_btn.click(
            add_text,
            [state, textbox, masked_image_output, image_process_mode, imagebox], 
            [state, chatbot, textbox] + btn_list  
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, coordinates_output],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )
        CLS_btn.click(
            add_text_wCLS,
            [state, textbox, masked_image_output, image_process_mode, imagebox], 
            [state, chatbot, textbox] + btn_list  
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, coordinates_output],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
