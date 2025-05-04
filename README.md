# llm.zig

For a complete and detailed explanation of the repo you can check the [DeepWiki](https://deepwiki.com/Saimirbaci/llm.zig)

LLM training in Zig, inspired by [Andrej Karpathy](https://github.com/karpathy/llm.c).

The goal is to have a simple, clean, and fast implementation of LLMs in Zig. 
The first working example is GPT-2, which is the grand-daddy of LLMs. The first time the modern stack was put together.

Why Zig? New to zig, and zig looks simple, clean, and fast language that is easy to learn and understand. 
And I personally like the philosophy of Zig and the way it is designed to be dev friendly and its strictness on type safety and many other features.

## Quick Start

### Zig setup version 0.12.0

Install zig for your machine following  [Zig](https://ziglang.org/learn/getting-started/)

### Dataset
Download and tokenize a sample dataset. 
Andrej uses the tinyshakespeare dataset is the fastest to download and tokenize:
In your python virtual environment run the following command:
```bash
pip install -r requirements.txt
python prepro_tinyshakespeare.py
```

### Generate pretrained data using the Nano Gpt2 model
Download the GPT-2 weights and save them as a checkpoint we can load in Zig:

The following command will download the GPT-2 124M a tokenizer model and save it in the data folder. 

```bash
python train_gpt2.py  
mv gpt2_124M.bin data/
mv gpt2_tokenizer.bin data/
mv gpt2_124M_debug_state.bin data/
```

### Compile the code

```bash
zig build-exe -O ReleaseFast train_gpt2.zig 
```

### Run the code

```bash
./train_gpt2
``` 

