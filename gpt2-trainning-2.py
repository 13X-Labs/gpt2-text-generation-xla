import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
	"gpt2", padding_side="left", pad_token="</s>"
)
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

inputs_text_1 = ["Vietnam is a"]

tokenize_text_1 = tokenizer(inputs_text_1, return_tensors="tf") # Length = 11
print(f"`tokenize_inputs_text_1` shape = {tokenize_text_1.input_ids.shape}")


# One line to create a XLA generation function
xla_generate = tf.function(model.generate, jit_compile=True)

start = time.time_ns()
generated_input_1 = xla_generate(**tokenize_text_1)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
print(f"{tokenizer.decode(generated_input_1[0])}\n") 


# Calls XLA generation without padding
padding_kwargs = {"pad_to_multiple_of": 8, "padding": True}
tokenized_input_1_with_padding = tokenizer(
    inputs_text_1, return_tensors="tf", **padding_kwargs
) # Length = 8

print(
    "`tokenized_input_1_with_padding` shape = ",
    f"{tokenized_input_1_with_padding.input_ids.shape}"
)

start = time.time_ns()
generated_input_1_with_padding = xla_generate(**tokenized_input_1_with_padding)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
print(f"{tokenizer.decode(generated_input_1_with_padding[0])}\n") 

