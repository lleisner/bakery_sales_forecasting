import re

def clean_and_replace(text):
    # Replace umlauts
    text = text.replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe')
    text = text.replace('Ä', 'Ae').replace('Ü', 'Ue').replace('Ö', 'Oe')
    # Remove any non-printable characters and null characters
    text = re.sub(r'[^\x20-\x7E]', '', text).strip()
    return text

def load_item_mapping(file_path="data/raw_sources/item_mapping.txt"):
    item_mapping = {}
    with open(file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the header
            # Strip the line of leading/trailing whitespace and skip if it's empty
            line = line.strip()
            if not line:
                continue
            # Split by tab
            parts = line.split('\t')
            if len(parts) >= 2:
                # Handle non-printable characters
                key = clean_and_replace(parts[0])
                value = clean_and_replace(parts[1])
                item_mapping[key] = value

    return item_mapping


if __name__ == "__main__":
    print(load_item_mapping())