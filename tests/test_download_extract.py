from fractal_neurons.download_english import extract_text


def test_extract_text_priority():
    sample = {"text": "hello", "content": "ignored", "other": "foo"}
    assert extract_text(sample, "text") == "hello"
    sample2 = {"content": "hi"}
    assert extract_text(sample2, None) == "hi"
    sample3 = {"foo": "bar"}
    assert extract_text(sample3, None) == "bar"
