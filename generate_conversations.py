#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


TOPICS = [
    "machine learning", "black holes", "photosynthesis", "the Roman Empire", "quantum physics",
    "climate change", "the history of the internet", "the human brain", "the stock market",
    "the future of space exploration", "the importance of bees", "the life of Marie Curie",
    "the principles of design", "the basics of cooking", "the benefits of exercise"
]

CONCEPTS = [
    "neural networks", "general relativity", "the meaning of life", "the industrial revolution",
    "democracy", "capitalism", "socialism", "artificial intelligence", "love", "happiness",
    "justice", "freedom", "creativity", "consciousness", "beauty"
]

PROMPT_TEMPLATES = [
    "What is {topic}?",
    "Tell me about {topic}.",
    "Explain {concept} in simple terms.",
    "Can you give me a summary of {topic}?",
    "What are your thoughts on {concept}?",
    "What is the difference between {topic} and {concept}?",
    "How does {topic} work?",
    "Why is {concept} important?",
    "Give me an example of {topic}.",
    "What are the pros and cons of {concept}?",
]

RESPONSE_TEMPLATES = [
    "I'm still learning about {topic}, but I can tell you that it's a very interesting field.",
    "{topic} is a complex subject, but I can give you a brief overview.",
    "In simple terms, {concept} is about...",
    "I don't have personal thoughts, but I can provide you with information about {concept}.",
    "The difference between {topic} and {concept} is...",
    "{topic} works by...",
    "{concept} is important because...",
    "A good example of {topic} is...",
    "The pros of {concept} are... and the cons are...",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic conversational dataset")
    parser.add_argument("--output", default="data/conversational_corpus/conversations.jsonl")
    parser.add_argument("--count", type=int, default=500, help="Number of prompt/response pairs to create")
    args = parser.parse_args()

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conversations = []
    for _ in range(max(1, args.count)):
        prompt_template = random.choice(PROMPT_TEMPLATES)
        response_template = random.choice(RESPONSE_TEMPLATES)

        topic1 = random.choice(TOPICS)
        topic2 = random.choice(TOPICS)
        concept1 = random.choice(CONCEPTS)
        concept2 = random.choice(CONCEPTS)

        prompt = prompt_template.format(topic=topic1, concept=concept1)
        response = response_template.format(topic=topic2, concept=concept2)

        conversations.append({"prompt": prompt, "response": response})

    with out_path.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Generated {len(conversations)} conversational examples → {out_path}")


if __name__ == "__main__":
    main()
