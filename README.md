# literAI

literAI demo: [https://literai.hooloovoo.ai](https://literai.hooloovoo.ai) ([source](https://github.com/hooloovoo-ai/literAI-website)) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D61fr09JikQNuNErtYIMXg_D_Bmt5Z-1?usp=sharing)

[literAI](https://github.com/jquesnelle/literAI) is an experiment in open source AI composition written by [emozilla](https://twitter.com/theemozilla).
Originally inspired by [scribepod](https://github.com/yacineMTB/scribepod) by [yacine](https://twitter.com/yacineMTB), it creates a podcast where the two hosts, Alice and Bob, analyze a novel they both purportedly recently read, along with associated images generated from inferred descriptions of scenes in the novel.
Cricually, literAI uses exclusively open source AI models (no API calls) and is designed to run on (admittedly high-end) consumer-grade hardware.
It requires 24 GB of VRAM, although it is likely possible it could be tweaked to work with less.

### Models used

| Model | Purpose |
| -     | -       |
| [pszemraj/long-t5-tglobal-xl-16384-book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary) | Generate summaries of the novel text |
| [allenai/cosmo-xl](https://huggingface.co/allenai/cosmo-xl) | Conversation generation |
| [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) | Scene description summarization from novel passages |
| [dreamlike-art/dreamlike-diffusion-1.0](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0) | Image generation |

### Packages/tools used

| Package | Purpose |
| -       | -       |
| [transformers](https://github.com/huggingface/transformers) | Run LLMs |
| [diffusers](https://github.com/huggingface/diffusers) | Run diffusion models |
| [textsum](https://github.com/pszemraj/textsum) | Automate summary batching |
| [LangChain](https://github.com/hwchase17/langchain) | LLM context and prompt construction |
| [TorToiSe](https://github.com/neonbjb/tortoise-tts) | Audio generation |
| [pydub](https://github.com/jiaaro/pydub) | Audio stiching |

## Running

To run, clone the repository and install neccessary requirements.

```sh
git clone https://github.com/jquesnelle/literAI
cd literAI
python -m pip install -r ./requirements.txt
```

Then, pass the novel's title, author, and path to the raw UTF-8 encoded text file to the `literai` module.

```sh
python -m literai "Alice's Adventures in Wonderland" "Lewis Carroll" alice-in-wonderland.txt
```

Note: this may take a while.
A 24 GB CUDA-capable video card is _highly_ recommended.
The generated data will be in the `output/` folder.

### Running incrementally

Generating a literAI podcast is done in six steps, which the main `literai` command combines together.
The steps are:

1. Generate summaries
2. Generate dialogue script
3. Generate image descriptions
4. Generate images
5. Generate audio
6. (optional) Add to index file and upload to Google Cloud Storage

Each of these steps can be invoked separately.
For example, to re-create the dialogue script (it's random each time)

```sh
python -m literai.steps.step2 "Alice's Adventures in Wonderland" "Lewis Carroll"
```
