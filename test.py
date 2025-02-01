import os
import time
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

load_dir = "models"
# gguf_file = "Llama-3.2-3B-Instruct-IQ3_M.gguf"

tokenizer = AutoTokenizer.from_pretrained(load_dir)

model = AutoModelForCausalLM.from_pretrained(
    load_dir,
    torch_dtype='auto',
    device_map="mps",
    num_return_sequences=1,
)


# Input text
messages = [
    {'role': 'system', 'content': 'You are a professional chatbot in AI domain, your name is Mr Beast'},
    {'role': 'user', 'content': "hello, what is deep learning ?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("mps")

generated_text = "" 

# Tạo streamer để xử lý token khi được sinh
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

# Thiết lập giới hạn độ dài tối đa
max_output_length = 2048
count = 0
start_time = time.time()
tokens_per_second = 0


# Tạo luồng riêng để gọi `generate`
def generate_tokens():
    model.generate(
        **input_ids,
        max_new_tokens=max_output_length,
        do_sample=True,
        temperature=0.01,
        top_k=50,
        repetition_penalty=1.25,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=False,
        streamer=streamer,
    )


# Chạy generate trong một luồng riêng
thread = threading.Thread(target=generate_tokens)
thread.start()

# Đọc token từ streamer, in ra và tính tốc độ
last_time = start_time
for token in streamer:
    current_time = time.time()
    elapsed_time = current_time - last_time
    generated_text += token
    count += 1
    if elapsed_time > 0:
        tokens_per_second = 1 / elapsed_time
    # print(token, end="", flush=True)  # Hiển thị từng token ngay lập tức
    for digit in token:
        print(digit, end="", flush=True)
        time.sleep(0.01)
    last_time = current_time

# Tổng kết
end_time = time.time()
print(f"Total tokens generated: {count}")
print(f"Generation time: {end_time - start_time:.2f} seconds")
print(f"Average tokens/sec: {count / (end_time - start_time):.2f}")