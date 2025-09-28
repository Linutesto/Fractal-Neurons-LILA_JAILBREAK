from pathlib import Path

from fractal_neurons.finetune_english import build_config


def test_build_config_values():
    tokenizer = {
        "type": "hf",
        "name_or_path": "bert-base-uncased",
        "truncation_side": "right",
        "pad_to_multiple_of": 128,
    }
    cfg = build_config(
        Path("/tmp/corpus"),
        "base.pt",
        1000,
        1e-4,
        16,
        512,
        4,
        2,
        0.999,
        200,
        1.0,
        tokenizer_cfg=tokenizer,
    )
    assert cfg["data"]["text_root"] == "/tmp/corpus"
    assert cfg["init"]["from_checkpoint"] == "base.pt"
    assert cfg["train"]["steps"] == 1000
    assert cfg["model"]["num_experts"] == 4
    assert cfg["tokenizer"]["name_or_path"] == "bert-base-uncased"
