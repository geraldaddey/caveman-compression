#!/usr/bin/env python3
"""
Caveman Compression Tool
Converts normal English to caveman compression and vice versa using OpenAI API
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError, AuthenticationError

# Try to get API key from environment variable or local .env file
API_KEY = os.getenv('OPENAI_API_KEY')

if not API_KEY:
    # Try to read from local .env file
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    API_KEY = line.split('=', 1)[1].strip().strip('"\'')
                    break

if not API_KEY:
    print("Error: OPENAI_API_KEY not found.", file=sys.stderr)
    print("Please set the OPENAI_API_KEY environment variable or create a .env file in the project root.", file=sys.stderr)
    print("You can get an API key from https://platform.openai.com/account/api-keys", file=sys.stderr)
    sys.exit(1)

# Load prompts from files
PROMPTS_DIR = Path(__file__).parent / 'prompts'
MAX_TEXT_LENGTH = 100000 # Maximum characters for input text

def load_prompt(filename):
    """Load prompt from prompts directory"""
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        print(f"Please ensure '{filename}' exists in the '{PROMPTS_DIR}' directory.", file=sys.stderr)
        sys.exit(1)
    try:
        return prompt_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        print(f"Error: Could not read prompt file '{prompt_path}' due to invalid UTF-8 encoding.", file=sys.stderr)
        sys.exit(1)

COMPRESSION_PROMPT = load_prompt('compression.txt')
DECOMPRESSION_PROMPT = load_prompt('decompression.txt')


def count_tokens(text):
    """Estimate tokens using character count / 4"""
    return len(text.strip()) // 4


def is_text_content(text):
    """Detect if content is natural language text vs code/structured data"""
    # Check for common code indicators
    code_indicators = [
        'def ', 'class ', 'function ', 'import ', 'const ', 'let ', 'var ',
        'public ', 'private ', 'protected ', '#include', 'package ',
        '=>', '->', '::', '!=', '==', '<=', '>=', '&&', '||',
        'print(', 'console.log(', 'return ', 'if (', 'for (', 'while (',
    ]

    # Count code-like patterns
    code_score = sum(1 for indicator in code_indicators if indicator in text)

    # Check for balanced braces/brackets (common in code)
    brace_count = text.count('{') + text.count('}') + text.count('[') + text.count(']') + text.count('(') + text.count(')')

    # Check for natural language indicators
    words = text.split()
    if len(words) < 5:
        return True  # Short text, treat as natural language

    # If high code indicators or many braces, treat as code
    if code_score >= 2 or brace_count > len(words) * 0.2:
        return False

    return True


def split_sentences(text):
    """Split text into sentences by period, preserving sentence boundaries"""
    # If no period in text, return as single sentence
    if '.' not in text:
        return [text]

    # Simple sentence splitting by periods followed by space or end of string
    import re
    # Split on '. ' or '.\n' or '. \n' but keep the period
    sentences = re.split(r'\.(\s+)', text)

    # Reconstruct sentences with their periods
    result = []
    i = 0
    while i < len(sentences):
        if sentences[i].strip():
            sentence = sentences[i]
            # Add back the period if this isn't the last fragment
            if i + 1 < len(sentences):
                sentence = sentence + '.'
            result.append(sentence.strip())
        i += 2 if i + 1 < len(sentences) else 1

    return result if result else [text]


def compress_text(text, model="gpt-4o"):
    """Compress normal English to caveman compression"""
    # Input Validation
    if not text or not text.strip():
        print("Error: Input text cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if len(text) > MAX_TEXT_LENGTH:
        print(f"Error: Input text too long ({len(text)} characters). Maximum allowed is {MAX_TEXT_LENGTH}.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY)

    # Detect if this is natural language text
    is_text = is_text_content(text)

    compressed = ""
    try:
        # If it's text, use sentence-by-sentence compression with gpt-4o-mini
        if is_text:
            sentences = split_sentences(text)

            # If only one sentence or very short, compress as whole
            if len(sentences) <= 1:
                model = "gpt-4o-mini"
                prompt = COMPRESSION_PROMPT.format(text=text)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert at caveman compression. Always compress the provided text, never ask for clarification."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                )
                compressed = response.choices[0].message.content.strip()
            else:
                # Compress sentence by sentence with gpt-4o-mini
                compressed_sentences = []
                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    prompt = COMPRESSION_PROMPT.format(text=sentence)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert at caveman compression. Always compress the provided text, never ask for clarification."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                    )
                    compressed_sent = response.choices[0].message.content.strip()
                    compressed_sentences.append(compressed_sent)

                compressed = ' '.join(compressed_sentences)
        else:
            # For code/structured data, use original model and compress as whole
            prompt = COMPRESSION_PROMPT.format(text=text)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at caveman compression. Always compress the provided text, never ask for clarification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            compressed = response.choices[0].message.content.strip()

    except AuthenticationError:
        print("Error: OpenAI authentication failed. Please check your API key.", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        if e.status_code == 429:
            print("Error: OpenAI rate limit exceeded. Please wait and try again later.", file=sys.stderr)
        elif e.status_code >= 500:
            print(f"Error: OpenAI server error ({e.status_code}). Service may be temporarily unavailable.", file=sys.stderr)
        else:
            print(f"Error: OpenAI API returned an unexpected status code {e.status_code}: {e.response}", file=sys.stderr)
        sys.exit(1)
    except APITimeoutError:
        print("Error: OpenAI API request timed out. Please check your internet connection or try again later.", file=sys.stderr)
        sys.exit(1)
    except APIConnectionError:
        print("Error: Could not connect to OpenAI API. Please check your internet connection.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during compression: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate statistics
    original_tokens = count_tokens(text)
    compressed_tokens = count_tokens(compressed)
    reduction = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0

    return compressed, original_tokens, compressed_tokens, reduction


def decompress_text(text, model="gpt-4o"):
    """Decompress caveman compression to normal English"""
    # Input Validation
    if not text or not text.strip():
        print("Error: Input text cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if len(text) > MAX_TEXT_LENGTH:
        print(f"Error: Input text too long ({len(text)} characters). Maximum allowed is {MAX_TEXT_LENGTH}.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY)

    decompressed = ""
    try:
        prompt = DECOMPRESSION_PROMPT.format(text=text)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at expanding compressed text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        decompressed = response.choices[0].message.content.strip()

    except AuthenticationError:
        print("Error: OpenAI authentication failed. Please check your API key.", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        if e.status_code == 429:
            print("Error: OpenAI rate limit exceeded. Please wait and try again later.", file=sys.stderr)
        elif e.status_code >= 500:
            print(f"Error: OpenAI server error ({e.status_code}). Service may be temporarily unavailable.", file=sys.stderr)
        else:
            print(f"Error: OpenAI API returned an unexpected status code {e.status_code}: {e.response}", file=sys.stderr)
        sys.exit(1)
    except APITimeoutError:
        print("Error: OpenAI API request timed out. Please check your internet connection or try again later.", file=sys.stderr)
        sys.exit(1)
    except APIConnectionError:
        print("Error: Could not connect to OpenAI API. Please check your internet connection.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during decompression: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate statistics
    caveman_tokens = count_tokens(text)
    normal_tokens = count_tokens(decompressed)
    expansion = ((normal_tokens - caveman_tokens) / caveman_tokens * 100) if caveman_tokens > 0 else 0

    return decompressed, caveman_tokens, normal_tokens, expansion


def main():
    parser = argparse.ArgumentParser(
        description='Caveman Compression Tool - Compress or decompress text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress normal English to caveman
  python caveman_compress.py compress "In order to optimize the database..."

  # Decompress caveman to normal English
  python caveman_compress.py decompress "Need fast queries. Add index..."

  # Read from file
  python caveman_compress.py compress -f input.txt

  # Save output to file
  python caveman_compress.py compress -f input.txt -o output.txt
        """
    )

    parser.add_argument(
        'mode',
        choices=['compress', 'decompress', 'c', 'd'],
        help='Mode: compress (c) or decompress (d)'
    )
    parser.add_argument(
        'text',
        nargs='?',
        help='Text to process (omit if using -f)'
    )
    parser.add_argument(
        '-f', '--file',
        help='Read input from file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Write output to file'
    )
    parser.add_argument(
        '-m', '--model',
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )

    args = parser.parse_args()

    # Normalize mode
    mode = 'compress' if args.mode in ['compress', 'c'] else 'decompress'

    # Get input text
    input_text = ""
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        try:
            input_text = input_path.read_text(encoding='utf-8').strip()
        except PermissionError:
            print(f"Error: Permission denied when reading file: {args.file}", file=sys.stderr)
            sys.exit(1)
        except UnicodeDecodeError:
            print(f"Error: Input file '{args.file}' contains invalid UTF-8 encoding.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: An unexpected error occurred while reading file '{args.file}': {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        input_text = args.text
    else:
        parser.error("Must provide either text argument or -f/--file option")

    # Process text
    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper()}")
    print(f"MODEL: {args.model}")
    print(f"{'='*60}\n")

    if mode == 'compress':
        print("ORIGINAL TEXT:")
        print(f"{input_text}\n")

        print("Compressing...\n")
        result, orig_tokens, comp_tokens, reduction = compress_text(input_text, args.model)

        print("CAVEMAN COMPRESSED:")
        print(f"{result}\n")

        print(f"{'='*60}")
        print("STATISTICS:")
        print(f"  Original:   {len(input_text):4d} chars ≈ {orig_tokens:3d} tokens")
        print(f"  Compressed: {len(result):4d} chars ≈ {comp_tokens:3d} tokens")
        print(f"  Reduction:  {reduction:.1f}%")
        print(f"{'='*60}\n")

    else:  # decompress
        print("CAVEMAN TEXT:")
        print(f"{input_text}\n")

        print("Decompressing...\n")
        result, cave_tokens, norm_tokens, expansion = decompress_text(input_text, args.model)

        print("NORMAL ENGLISH:")
        print(f"{result}\n")

        print(f"{'='*60}")
        print("STATISTICS:")
        print(f"  Caveman:  {len(input_text):4d} chars ≈ {cave_tokens:3d} tokens")
        print(f"  Normal:   {len(result):4d} chars ≈ {norm_tokens:3d} tokens")
        print(f"  Expansion: {expansion:.1f}%")
        print(f"{'='*60}\n")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.write_text(result, encoding='utf-8')
            print(f"Output saved to: {args.output}\n")
        except PermissionError:
            print(f"Error: Permission denied when writing to file: {args.output}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: An unexpected error occurred while writing to file '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
