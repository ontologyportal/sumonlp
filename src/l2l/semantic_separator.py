import re
import random
import string
import json
import os

class SemanticSeparator:
    def __init__(self):
        self.word_map = {}

    def generate_random_code(self):
        return ''.join(random.choices(string.ascii_uppercase, k=5))

    def scramble_text(self, text):
        # Match quoted strings and unquoted text separately
        parts = re.split(r'(".*?")', text)  # split but keep the quotes in results

        token_pattern = re.compile(r'\b\w+\b|\?')

        def replace_token(match):
            token = match.group(0)
            if token not in self.word_map:
                self.word_map[token] = self.generate_random_code()
            return self.word_map[token]

        scrambled_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Inside quotes: skip scrambling
                scrambled_parts.append(part)
            else:  # Outside quotes: scramble
                scrambled_parts.append(token_pattern.sub(replace_token, part))

        return ''.join(scrambled_parts)

    def scramble_file(self, input_path, output_path):
        with open(input_path, 'r', encoding='utf-8') as infile:
            text = infile.read()

        scrambled = self.scramble_text(text)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(scrambled)

        # Save map in same folder as output file
        map_file_path = os.path.splitext(output_path)[0] + '.map'
        with open(map_file_path, 'w', encoding='utf-8') as mapfile:
            json.dump(self.word_map, mapfile, indent=2)

        print(f"✅ Scrambled file written to: {output_path}")
        print(f"🗺️  Mapping saved to: {map_file_path}")

    def unscramble_file(self, scrambled_path, output_path, map_path=None):
        with open(scrambled_path, 'r', encoding='utf-8') as infile:
            scrambled_text = infile.read()

        if not map_path:
            map_path = os.path.splitext(scrambled_path)[0] + '.map'

        with open(map_path, 'r', encoding='utf-8') as mapfile:
            scramble_map = json.load(mapfile)

        reverse_map = {v: k for k, v in scramble_map.items()}
        missing_tokens = set()

        unscrambled_text = ""
        chunk = ""
        i = 0
        while i < len(scrambled_text):
            if len(chunk) < 5:
                if scrambled_text[i] == '"':
                    unscrambled_text += chunk + scrambled_text[i]
                    i = i+1
                    while i < len(scrambled_text) and scrambled_text[i] != '"':
                        unscrambled_text += scrambled_text[i]
                        i = i+1
                    if i < len(scrambled_text):
                        unscrambled_text += scrambled_text[i]
                    chunk = ""
                elif not (scrambled_text[i].isalpha() and scrambled_text[i].isupper()):
                    unscrambled_text += chunk + scrambled_text[i]
                    chunk = ""
                else:
                    chunk += scrambled_text[i]
                i = i+1
            else:
                if chunk in reverse_map:
                    unscrambled_text += reverse_map[chunk]
                else:
                    missing_tokens.add(chunk)
                    unscrambled_text += chunk
                chunk = ""

        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(unscrambled_text)

        print(f"✅ Unscrambled file written to: {output_path}")
        if missing_tokens:
            print("\n⚠️  Warning: The following tokens were not found in the map:")
            for t in sorted(missing_tokens):
                print(f"  - {t}")

    def unscramble_string(self, scrambled_string, map_path):
        with open(map_path, 'r', encoding='utf-8') as mapfile:
            scramble_map = json.load(mapfile)

        reverse_map = {v: k for k, v in scramble_map.items()}
        missing_tokens = set()

        token_pattern = re.compile(r'\b[A-Z]{5}\b')

        def replace_token(match):
            token = match.group(0)
            if token in reverse_map:
                return reverse_map[token]
            else:
                missing_tokens.add(token)
                return token

        result = token_pattern.sub(replace_token, scrambled_string)

        print("\n🔓 Unscrambled Output:")
        print(result)

        if missing_tokens:
            print("\n⚠️ Warning: The following tokens were not found in the map:")
            for t in sorted(missing_tokens):
                print(f"  - {t}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Separate semantic grounding in text using SemanticSeparator.')
    parser.add_argument('mode', choices=['scramble', 'unscramble', 'unscramble-string'],
                        help='Mode: scramble a file, unscramble a file, or unscramble a string\nExample: python semantic_separator.py --map $SUMO_NLP_HOME/L2L_model/t5_model/combined-log-encoded.map unscramble output_l2l.txt unscrambled.txt')
    parser.add_argument('input_file', nargs='?', help='Input file (required for scramble or unscramble)')
    parser.add_argument('output_file', nargs='?', help='Output file (required for scramble or unscramble)')
    parser.add_argument('--map', help='Path to .map file (optional, defaults to output_file.map)', default=None)

    args = parser.parse_args()
    tool = SemanticSeparator()

    if args.mode == 'scramble':
        if not args.input_file or not args.output_file:
            print("❌ scramble mode requires input_file and output_file.")
        else:
            tool.scramble_file(args.input_file, args.output_file)

    elif args.mode == 'unscramble':
        if not args.input_file or not args.output_file:
            print("❌ unscramble mode requires input_file and output_file.")
        else:
            tool.unscramble_file(args.input_file, args.output_file, args.map)

    elif args.mode == 'unscramble-string':
        scrambled_string = input("🔤 Enter scrambled string: ")
        if not args.map:
            print("❌ Please provide a --map path for unscrambling the string.")
        else:
            tool.unscramble_string(scrambled_string, args.map)


