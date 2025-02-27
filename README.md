# ExLlamaV2

ExLlamaV2 is an inference library for running local LLMs on modern consumer GPUs.

The official and recommended backend server for ExLlamaV2 is [TabbyAPI](https://github.com/theroyallab/tabbyAPI/),
which provides an OpenAI-compatible API for local or remote inference, with extended features like HF model
downloading, embedding model support and support for HF Jinja2 chat templates.

See the [wiki](https://github.com/theroyallab/tabbyAPI/wiki/1.-Getting-Started) for help getting started.


## New in v0.1.0+:

- ExLlamaV2 now supports paged attention via [Flash Attention](https://github.com/Dao-AILab/flash-attention) 2.5.7+
- New generator with dynamic batching, smart prompt caching, K/V cache deduplication and simplified API

![alt_text](doc/dynamic_gen.gif)

## Dynamic generator

The dynamic generator supports all inference, sampling and speculative decoding features of the previous two 
generators, consolidated into one API (with the exception of FP8 cache, though the Q4 cache mode is supported and
performs better anyway, see [here](doc/qcache_eval.md).)

The generator is explained in detail [here](doc/dynamic.md).

- Single generation:
  ```python
  output = generator.generate(prompt = "Hello, my name is", max_new_tokens = 200)
  ```
- Batched generation:
    ```python
    outputs = generator.generate(
        prompt = [
            "Hello, my name is",
            "Once upon a time,",
            "Large language models are",
        ], 
        max_new_tokens = 200
    )
    ```
- Streamed generation with `asyncio`:
    ```python
    job = ExLlamaV2DynamicJobAsync(
        generator,
        input_ids = tokenizer.encode("You can lead a horse to water"),
        banned_strings = ["make it drink"],
        gen_settings = ExLlamaV2Sampler.Settings.greedy(),
        max_new_tokens = 200
    )  
    async for result in job:
        text = result.get("text", "")
        print(text, end = "")       
    ``` 
See the full, updated examples [here](https://github.com/turboderp/exllamav2/tree/master/examples).


## Performance

Some quick tests to compare performance with ExLlama V1. There may be more performance optimizations in the future,
and speeds will vary across GPUs, with slow CPUs still being a potential bottleneck:

| Model      | Mode         | Size  | grpsz | act | 3090Ti  | 4090        |
|------------|--------------|-------|-------|-----|---------|-------------|
| Llama      | GPTQ         | 7B    | 128   | no  | 181 t/s | **205** t/s |
| Llama      | GPTQ         | 13B   | 128   | no  | 110 t/s | **114** t/s |
| Llama      | GPTQ         | 33B   | 128   | yes | 44 t/s  | **48** t/s  |
| OpenLlama  | GPTQ         | 3B    | 128   | yes | 259 t/s | **296** t/s |
| CodeLlama  | EXL2 4.0 bpw | 34B   | -     | -   | 44 t/s  | **50** t/s  |
| Llama2     | EXL2 3.0 bpw | 7B    | -     | -   | 217 t/s | **257** t/s |
| Llama2     | EXL2 4.0 bpw | 7B    | -     | -   | 185 t/s | **211** t/s |
| Llama2     | EXL2 5.0 bpw | 7B    | -     | -   | 164 t/s | **179** t/s |
| Llama2     | EXL2 2.5 bpw | 70B   | -     | -   | 33 t/s  | **38** t/s  |
| TinyLlama  | EXL2 3.0 bpw | 1.1B  | -     | -   | 656 t/s | **770** t/s |
| TinyLlama  | EXL2 4.0 bpw | 1.1B  | -     | -   | 602 t/s | **700** t/s |


## How to

To install from the repo you'll need the CUDA Toolkit and either gcc on Linux or (Build Tools for) Visual Studio
on Windows). Also make sure you have an appropriate version of [PyTorch](https://pytorch.org/get-started/locally/), then run:

```sh
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt
pip install .

python test_inference.py -m <path_to_model> -p "Once upon a time,"
# Append the '--gpu_split auto' flag for multi-GPU inference
```

A simple console chatbot is included. Run it with:

```sh
python examples/chat.py -m <path_to_model> -mode llama -gs auto
```


The `-mode` argument chooses the prompt format to use. `raw` will produce a simple chatlog-style chat that works with base 
models and various other finetunes. Run with `-modes` for a list of all available prompt formats. You can also provide
a custom system prompt with `-sp`. 


## Integration and APIs

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is a FastAPI-based server that provides an OpenAI-style web API
compatible with [SillyTavern](https://sillytavernai.com/) and other frontends.  

- [ExUI](https://github.com/turboderp/exui) is a simple, standalone single-user web UI that serves an ExLlamaV2 instance
directly with chat and notebook modes.

- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) supports ExLlamaV2 through the **exllamav2**
and **exllamav2_HF** loaders.

- [lollms-webui](https://github.com/ParisNeo/lollms-webui) supports ExLlamaV2 through the exllamav2 binding.

## Installation

### Method 1: Install from source

To install the current dev version, clone the repo and run the setup script:

```sh
git clone https://github.com/turboderp/exllamav2
cd exllamav2
cd /home/kesong.yk/anaconda3/envs/decdiff_env/bin/../lib/
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
pip install -r requirements.txt
pip install .
```

By default this will also compile and install the Torch C++ extension (`exllamav2_ext`) that the library relies on. 
You can skip this step by setting the `EXLLAMA_NOCOMPILE` environment variable:

```sh
EXLLAMA_NOCOMPILE= pip install .
```

This will install the "JIT version" of the package, i.e. it will install the Python components without building the
C++ extension in the process. Instead, the extension will be built the first time the library is used, then cached in 
`~/.cache/torch_extensions` for subsequent use.

### Method 2: Install from release (with prebuilt extension)

Releases are available [here](https://github.com/turboderp/exllamav2/releases), with prebuilt wheels that contain the extension binaries. Make sure to grab
the right version, matching your platform, Python version (`cp`) and CUDA version. Crucially, you must also match
the prebuilt wheel with your PyTorch version, since the Torch C++ extension ABI breaks with every new version of 
PyTorch.

Either download an appropriate wheel or install directly from the appropriate URL:

```sh
pip install https://github.com/turboderp/exllamav2/releases/download/v0.0.12/exllamav2-0.0.12+cu121-cp311-cp311-linux_x86_64.whl
```

The `py3-none-any.whl` version is the JIT version which will build the extension on first launch. The `.tar.gz` file
can also be installed this way, and it will build the extension while installing.

### Method 3: Install from PyPI

A PyPI package is available as well. This is the same as the JIT version (see above). It can be installed with:

```sh
pip install exllamav2
```


## EXL2 quantization

ExLlamaV2 supports the same 4-bit GPTQ models as V1, but also a new "EXL2" format. EXL2 is based on the same
optimization method as GPTQ and supports 2, 3, 4, 5, 6 and 8-bit quantization. The format allows for mixing quantization
levels within a model to achieve any average bitrate between 2 and 8 bits per weight.

Moreover, it's possible to apply multiple quantization levels to each linear layer, producing something akin to sparse 
quantization wherein more important weights (columns) are quantized with more bits. The same remapping trick that lets
ExLlama work efficiently with act-order models allows this mixing of formats to happen with little to no impact on
performance.

Parameter selection is done automatically by quantizing each matrix multiple times, measuring the quantization 
error (with respect to the chosen calibration data) for each of a number of possible settings, per layer. Finally, a
combination is chosen that minimizes the maximum quantization error over the entire model while meeting a target
average bitrate.

In my tests, this scheme allows Llama2 70B to run on a single 24 GB GPU with a 2048-token context, producing coherent 
and mostly stable output with 2.55 bits per weight. 13B models run at 2.65 bits within 8 GB of VRAM, although currently
none of them uses GQA which effectively limits the context size to 2048. In either case it's unlikely that the model
will fit alongside a desktop environment. For now.

[![chat_screenshot](doc/llama2_70b_chat_thumb.png)](doc/llama2_70b_chat.png)
[![chat_screenshot](doc/codellama_13b_instruct_thumb.png)](doc/codellama_13b_instruct.png)

### Conversion

A script is provided to quantize models. Converting large models can be somewhat slow, so be warned. The conversion
script and its options are explained in [detail here](doc/convert.md)

### Evaluation

A number of evaluaion scripts are provided. See [here](doc/eval.md) for details.

### Community

A test community is provided at https://discord.gg/NSFwVuCjRq 
Quanting service free of charge is provided at #bot test. The computation is generiously provided by the Bloke powered by Lambda labs. 

### HuggingFace repos

- I've uploaded a few EXL2-quantized models to Hugging Face to play around with, [here](https://huggingface.co/turboderp).

- [LoneStriker](https://huggingface.co/LoneStriker) provides a large number of EXL2 models on Hugging Face. 

- [bartowski](https://huggingface.co/bartowski) has some more EXL2 models on HF.
