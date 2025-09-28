#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
import requests
import random

def get_ollama_response(prompt: str, model: str, system_prompt: str) -> str:
    """Fetches a response from the Ollama API."""
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return ""

def get_file_size_mb(file_path: Path) -> float:
    """Returns the size of a file in megabytes."""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic conversational data using Ollama.")
    parser.add_argument("--output", default="data/ollama_conversations/conversations.jsonl",
                        help="Path to the output JSONL file.")
    parser.add_argument("--model", default="gpt-oss:20b",
                        help="Ollama model to use for generation (e.g., 'gpt-oss:20b').")
    parser.add_argument("--target_mb", type=float, default=10.0,
                        help="Target size of the dataset in megabytes.")
    parser.add_argument("--system_prompt", default="You are a helpful AI assistant. Engage in a natural conversation with the user.",
                        help="System prompt for the AI in the conversation.")
    parser.add_argument("--initial_user_prompt", default="Start a conversation about a random interesting topic.",
                        help="Initial prompt to kick off the conversation.")
    args = parser.parse_args()

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting Ollama conversation generation with model '{args.model}'...")
    print(f"Target dataset size: {args.target_mb:.2f} MB. Output file: {out_path}")

    current_size_mb = get_file_size_mb(out_path)
    print(f"Current dataset size: {current_size_mb:.2f} MB")

    # If file exists, read existing conversations to continue
    existing_conversations = []
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue # Skip malformed lines

    with out_path.open("a" if existing_conversations else "w", encoding="utf-8") as f:
        while current_size_mb < args.target_mb:
            messages = []
            user_prompt = args.initial_user_prompt

            # If resuming, try to pick up from the last conversation's context
            if existing_conversations:
                last_conv = existing_conversations.pop() # Get last conversation
                if last_conv and "messages" in last_conv and last_conv["messages"]:
                    # Reconstruct context from last conversation
                    for msg in last_conv["messages"]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    # Try to continue from the last assistant message
                    if messages[-1]["role"] == "assistant":
                        user_prompt = "Continue the conversation."
                    elif messages[-1]["role"] == "user":
                        user_prompt = messages[-1]["content"] # Last user prompt
                existing_conversations = [] # Clear to avoid reprocessing

            # Start a new conversation if no valid context from existing
            if not messages:
                messages.append({"role": "user", "content": user_prompt})

            try:
                # Generate first AI response
                ai_response = get_ollama_response(user_prompt, args.model, args.system_prompt)
                if not ai_response:
                    print("Failed to get initial AI response. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                messages.append({"role": "assistant", "content": ai_response})

                # Continue conversation for a few turns
                for _ in range(random.randint(2, 5)): # 2 to 5 additional turns
                    last_assistant_message = messages[-1]["content"]
                    follow_up_prompt = get_ollama_response(
                        f"Based on the last AI response: '{last_assistant_message}', generate a short, natural user follow-up question or statement.",
                        args.model,
                        args.system_prompt # Use system prompt for user follow-up generation too
                    )
                    if not follow_up_prompt:
                        break # End conversation if follow-up generation fails
                    messages.append({"role": "user", "content": follow_up_prompt})

                    ai_response = get_ollama_response(follow_up_prompt, args.model, args.system_prompt)
                    if not ai_response:
                        break # End conversation if AI response fails
                    messages.append({"role": "assistant", "content": ai_response})

                # Write the complete conversation
                if len(messages) > 1: # Ensure at least one user and one assistant message
                    conversation_record = {"messages": messages}
                    f.write(json.dumps(conversation_record, ensure_ascii=False) + "\n")
                    f.flush() # Ensure data is written to disk
                    print(f"Generated conversation (length: {len(messages)} messages).")

            except Exception as e:
                print(f"An error occurred during conversation generation: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                continue

            current_size_mb = get_file_size_mb(out_path)
            print(f"Current dataset size: {current_size_mb:.2f} MB / {args.target_mb:.2f} MB")
            time.sleep(1) # Small delay to avoid hammering Ollama

    print(f"Dataset generation complete. Final size: {current_size_mb:.2f} MB at {out_path}")

if __name__ == "__main__":
    main()
