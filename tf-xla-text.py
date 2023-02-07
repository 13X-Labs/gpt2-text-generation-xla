# Note: execution times are deeply dependent on hardware -- a 3090 was used here.
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# Notice the new argument, `padding_side="left"` -- decoder-only models, which can
# be instantiated with TFAutoModelForCausalLM, should be left-padded, as they
# continue generating from the input prompt.
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", padding_side="left", pad_token="</s>"
)
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
input_1 = ["TensorFlow is"]
input_2 = ["TensorFlow is a"]

# One line to create a XLA generation function
xla_generate = tf.function(model.generate, jit_compile=True)

# Calls XLA generation without padding
tokenized_input_1 = tokenizer(input_1, return_tensors="tf")  # length = 4
tokenized_input_2 = tokenizer(input_2, return_tensors="tf")  # length = 5
print(f"`tokenized_input_1` shape = {tokenized_input_1.input_ids.shape}")
print(f"`tokenized_input_2` shape = {tokenized_input_2.input_ids.shape}")

print("Calling XLA generation with tokenized_input_1...")
print("(will be slow as it is the first call)")
start = time.time_ns()
xla_generate(**tokenized_input_1)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 9565.1 ms

print("Calling XLA generation with tokenized_input_2...")
print("(has a different length = will trigger tracing again)")
start = time.time_ns()
xla_generate(**tokenized_input_2)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 6815.0 ms

padding_kwargs = {"pad_to_multiple_of": 8, "padding": True}
tokenized_input_1_with_padding = tokenizer(
    input_1, return_tensors="tf", **padding_kwargs
)  # length = 8
tokenized_input_2_with_padding = tokenizer(
    input_2, return_tensors="tf", **padding_kwargs
)  # length = 8
print(
    "`tokenized_input_1_with_padding` shape = ",
    f"{tokenized_input_1_with_padding.input_ids.shape}"
)
print(
    "`tokenized_input_2_with_padding` shape = ",
    f"{tokenized_input_2_with_padding.input_ids.shape}"
)

print("Calling XLA generation with tokenized_input_1_with_padding...")
print("(slow, first time running with this length)")
start = time.time_ns()
xla_generate(**tokenized_input_1_with_padding)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 6815.4 ms

print("Calling XLA generation with tokenized_input_2_with_padding...")
print("(will be fast!)")
start = time.time_ns()
xla_generate(**tokenized_input_2_with_padding)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 19.3 ms


print("Calling XLA generation with the same input, but with new options...")
print("(slow again)")
start = time.time_ns()
xla_generate(**tokenized_input_1_with_padding, num_beams=2)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 9644.2 ms