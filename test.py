import os
import time
import torch
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

load_dir = "model"

tokenizer = AutoTokenizer.from_pretrained(load_dir)

model = AutoModelForCausalLM.from_pretrained(
    load_dir,
    torch_dtype=torch.bfloat16,
    device_map="mps",
    num_return_sequences=1,
)


# Input text
input_text = (
    "Hi, what is deep learning, compare deep learning and machine learning in table ?"
)

input_ids = tokenizer(input_text, return_tensors="pt").to("mps")

generated_text = "" 

# Dọn bộ nhớ GPU
torch.cuda.empty_cache()

# Tạo streamer để xử lý token khi được sinh
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Thiết lập giới hạn độ dài tối đa
max_output_length = 512
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
        repetition_penalty=1.2,
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
    print(token, end="", flush=True)  # Hiển thị từng token ngay lập tức
    last_time = current_time

# Tổng kết
end_time = time.time()
print(f"Total tokens generated: {count}")
print(f"Generation time: {end_time - start_time:.2f} seconds")
print(f"Average tokens/sec: {count / (end_time - start_time):.2f}")