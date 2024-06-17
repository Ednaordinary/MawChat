import time
import torch
import gc
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
import threading

past_chat = []
current_chat = None
current_response = None
model_loaded = 0 # 0 is unloaded, 1 is a request to load, 2 is loaded, 3 is a request to unload

def text_streamer(streamer):
    """
    For streaming text while model is generating.
    """
    for word in streamer:
        #TODO: add styletts when a period is printed here
        print(word, flush=True, end='')
    print("\n")

def model_runner():
    """
    This will be a thread for running the llama3 model. Should dynamically offload if we need vram
    """
    global past_chat
    global current_chat
    global current_response
    global model_loaded
    model = None
    tokenizer = None
    while True:
        time.sleep(0.01)
        if model_loaded == 1:
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    truncation_side="left",
            )
            model_loaded = 2
        if model_loaded == 3:
            model, tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
        if current_chat:
            if model_loaded != 2:
                model_loaded = 1
            else:
                sys_prompt= [ {"role": "system", "content": "You are Maw, an intelligence model that answers questions to the best of my knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna."}]
                this_chat = {"role": "user", "content": str(current_chat)}
                past_chat.append(this_chat)
                init_prompt = tokenizer.apply_chat_template(conversation=sys_prompt, tokenize=True, return_tensors='pt', add_generation_prompt=False)
                past_chat.append({"role": "user", "content": str(current_chat)})
                input_ids = tokenizer.apply_chat_template(conversation=past_chat, tokenize=True, return_tensors='pt', add_generation_prompt=True, max_length=7200 - init_prompt.size()[1], truncation=True)
                input_ids = torch.cat((init_prompt, input_ids), 1).to("cuda")
                output_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                stream_thread = threading.Thread(target=text_streamer, args=[output_streamer])
                model_kwargs = dict(input_ids=input_ids, max_new_tokens=768, use_cache=True,  do_sample=True, pad_token_id=tokenizer.eos_token_id) #max_matching_ngram_size=2, prompt_lookup_num_tokens=15,
                #TODO: Add prompt_lookup_num_tokens once the eot_id pr is merged
                stop_token = tokenizer.encode("<|eot_id|>")
                stream_thread.start()
                current_response = model.generate(**model_kwargs, streamer=output_streamer, eos_token_id=stop_token)
                current_chat = None

def text_input():
    global current_chat
    while True:
        current_chat = input("User>")
        while current_chat:
            time.sleep(0.01)

llama_thread = threading.Thread(target=model_runner)
text_thread = threading.Thread(target=text_input)
llama_thread.start()
text_thread.start()
text_thread.join()
