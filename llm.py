import os
import time
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

load_dir = "models"

class CustomLLM:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(CustomLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.generated_text = ""
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir, cache_dir=load_dir)
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            cache_dir=load_dir,
            skip_special_tokens=True,
            skip_prompt=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            load_dir,
            torch_dtype="auto",
            device_map="mps",
            num_return_sequences=1,
            cache_dir=load_dir,
        )

    def predict(self, input_text: str, limit: int | None):
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("mps")

        def generate_tokens():
            self.model.generate(
                **input_ids,
                max_new_tokens=512 if limit is None else limit,
                do_sample=True,
                temperature=0.01,
                output_scores=False,
                repetition_penalty=1.25,
                top_p=0.9,
                top_k=20,
                streamer=self.streamer,
            )

        thread = threading.Thread(target=generate_tokens)
        thread.start()

        for token in self.streamer:
            self.generated_text += token
            #print(token, end="", flush=True)
            for digit in token:
                print(digit, end="", flush=True)
                time.sleep(0.005)

        return self.generated_text
