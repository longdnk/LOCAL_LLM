from llm import CustomLLM

model = CustomLLM()

input = "Describe short and concise what is deep learning ?"

model.predict(input, 128)