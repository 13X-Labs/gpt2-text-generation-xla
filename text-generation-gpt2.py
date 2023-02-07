# Requires transformers >= 4.21.0;
# Sampling outputs may differ, depending on your hardware.
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
inputs = tokenizer(["Vietnam is"], return_tensors="tf")

generated = model.generate(**inputs, do_sample=True, seed=(42, 0), max_new_tokens=50, temperature=0.9)
print("Sampling output: ", tokenizer.decode(generated[0]))


# generated = model.generate(**inputs, do_sample=True, seed=(42, 0), max_new_tokens=50, temperature=0.7, num_beams=5, num_return_sequences=5)
# print(
#     "All generated hypotheses:",
#     "\n".join(tokenizer.decode(out) for out in generated)
# )

# However, it often results in sub-optimal outputs. You can increase the quality of the results through the num_beams argument.
# When it is larger than 1, it triggers Beam Search, which continuously explores high-probability sequences. 
# This exploration comes at the cost of additional resources and computational time.
# Finally, when running Sampling or Beam Search, you can use num_return_sequences to return several sequences. 
# For Sampling it is equivalent to running generate multiple times from the same input prompt, 
# while for Beam Search it returns the highest scoring generated beams in descending order.
# > Limiting to 30 new tokens: TensorFlow is a great learning platform for
# > Temperature 0.7: TensorFlow is a great way to do things like this........
# > Sampling output: TensorFlow is a great learning platform for learning about
# data structure and structure in data science..