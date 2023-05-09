from training.utils.data.data_utils import create_prompt_dataset
from transformers import AutoTokenizer
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("gpt2")
    create_prompt_dataset(
		0,
		["Dahoas/rm-static"],
		"2,4,4",
  		"./output",
		1,
		1,
		tok,
		512,
		"<|endoftext|>",
  		[]
	)