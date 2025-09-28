# Project Changelog

This document summarizes the changes made to the Fractal Neurons project.

## Summary of Changes

### Conversational AI Enhancements

The primary focus of the recent changes was to improve the model's conversational abilities. This involved a series of upgrades to the data, training, and generation pipelines.

- **New Conversational Datasets**:
    - Created a script to generate a synthetic conversational dataset (`generate_conversations.py`).
    - Added a sample conversational dataset at `data/conversational_corpus/conversations.jsonl`.
    - Created a script to format the conversational data into a chat format with special tokens (`tools/make_chat_sft.py`).
    - Generated a supervised fine-tuning (SFT) dataset for chat at `data/chat_sft.jsonl`.
    - Created a script to mix the conversational data with the English corpus (`mix_datasets.py`), producing a mixed dataset at `data/mixed_corpus/mixed_data.jsonl`.

- **New Training Configurations**:
    - Added `configs/conversational_v1.yaml` for training a conversational model from scratch.
    - Added `configs/mixed_corpus_v1.yaml` for training on the mixed conversational and English corpus data.
    - Added `configs/finetune_chat.yaml` for fine-tuning a model specifically for chat.

- **Improved Generation Capabilities**:
    - Implemented **repetition penalty** in the generation script (`fractal_neurons/generate.py`) to control text repetitiveness. This is available via the `--repetition_penalty` command-line argument.
    - Implemented **nucleus sampling (`top_p`)** in the generation script for more diverse and coherent text generation. This is available via the `--top_p` command-line argument.

### Tokenizer and Decoding Fixes

- **Corrected Tokenizer Decoding**: Addressed a critical bug where the tokenizer would produce artifacts (e.g., `Ġ`) in the generated text. This was fixed by:
    - Adding a `decode` method to all tokenizer classes in `fractal_neurons/tokenizer.py`.
    - Updating the generation and inference scripts to use the new `decode` method.
    - Creating a `fix_tokenizer_decoder.py` script to patch existing tokenizers with the correct decoder configuration.
- **Special Tokens**: Added special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) to the tokenizer to support the new chat format. This was done via the `add_special_tokens.py` script.

### Menu and Usability Improvements

- **New Menu Options**: The main menu (`menu.py`) has been updated with new, easy-to-use options for training conversational models:
    - "Train Conversational Model"
    - "Train on Mixed Corpus"
    - "Finetune for Chat"
- **Improved Defaults**: The default generation parameters in the menu have been updated to more sensible values for conversational AI (`temperature=0.8`, `top_k=50`).
- **Bug Fixes**: Fixed several `SyntaxError` and `ModuleNotFoundError` issues in the menu and other scripts.

### Other Changes

- **Distillation Filtering**: Improved the filtering logic in `fractal_neurons/distill.py` to be more robust for chat-based self-distillation.
- **Project Documentation**: Created this `project.md` file to log all the changes.

## How to Proceed

To train a conversational model, you can now use the new options in the main menu. The recommended workflow is:

1.  Run `python menu.py`.
2.  Select the "Train on Mixed Corpus" option to create the mixed dataset and start training.
3.  After training is complete, use the "Chat Session" option to interact with your new model, making sure to select the correct checkpoint from the `runs/mixed_corpus_v1` directory.
