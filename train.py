import csv
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from collections import Counter
import re
import matplotlib.pyplot as plt
from unidecode import unidecode
import numpy as np
import pronouncing
import textwrap


theme_list = [
        "relationships", "love", "partying", "empowerment", "heartbreak",
        "self-discovery", "friendship", "dreams", "nostalgia", "ambition",
        "freedom", "youth", "identity", "rebellion", "loss",
        "happiness", "growing up", "breakups", "romance", "desire",
        "adventure", "regret", "acceptance", "struggles", "moving on",
        "conflict", "passion", "hope", "change", "betrayal",
        "longing", "confidence", "memories", "loneliness", "anxiety",
        "fame", "independence", "expression", "vulnerability", "courage",
        "temptation", "pride", "forgiveness", "home", "discovery",
        "resilience", "introspection", "redemption", "escapism", "travel",
        "growth", "transformation", "fear", "trust", "uncertainty",
        "jealousy", "healing", "support", "determination", "compromise",
        "transience", "wanderlust", "belonging", "inspiration", "sacrifice",
        "perspective", "reinvention", "balance", "authenticity", "endurance",
        "survival", "persistence", "self-love", "recovery", "confrontation",
        "exploration", "experimentation", "power", "self-worth", "insecurity",
        "unity", "empathy", "letting go", "release", "celebration",
        "equality", "individuality", "maturity", "wisdom", "spirituality",
        "commitment", "loyalty", "loss of innocence", "grief", "rejection",
        "sympathy", "compassion", "sincerity", "miscommunication", "closure",
        "vices", "decisions", "co-dependency", "separation", "obsession",
        "resentment", "boundaries", "catharsis", "responsibility", "self-reflection",
        "expectations", "gratitude", "self-acceptance", "oppression", "liberation",
        "cultural identity", "legacy", "mortality", "time", "nostalgia",
        "inner strength", "adaptation", "illusion", "reality", "mindfulness",
        "solitude", "connection", "social commentary", "injustice", "resistance",
        "materialism", "technology", "nature", "urban life", "simplicity",
        "complexity", "art", "creativity", "innovation", "restlessness",
        "contemplation", "ambivalence", "desperation", "inner turmoil", "renewal",
        "harmony", "self-expression", "intuition", "duality", "enlightenment",
        "perseverance", "optimism", "pessimism", "paradox", "humor",
        "satire", "whimsy", "escapes", "daydreams", "imagination",
        "surrealism", "identity crisis", "disorientation", "revelation", "deception",
        "temptation", "paranoia", "alienation", "isolation", "reconciliation",
        "disillusionment", "determination", "perseverance", "introspection", "unrequited love",
        "devotion", "obsession", "serendipity", "mystery", "supernatural",
        "fairy tales", "mythology", "legends", "fantasy", "reality vs. illusion",
        "alchemy", "magic", "science fiction", "alternate realities", "parallel universes",
        "metamorphosis", "origins", "destiny", "fate", "providence",
        "cycles", "seasons", "time travel", "premonitions", "prophecy",
        "divination", "clairvoyance", "psychic phenomena", "intuition", "visions",
        "dreams vs. reality", "nightmares", "sleep", "awakening", "subconscious"
    ]


def create_ban_list_from_csv(file_path):
    ban_list = []
    severity_threshold = 1.5  # Adjust this value to change the probability of adding a word to the ban list

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            severity = float(row['severity_rating'])
            if should_ban_word(severity, severity_threshold):
                ban_list.append(row['text'])
                ban_list.append(row['canonical_form_1'])
                ban_list.append(row['canonical_form_2'])
                ban_list.append(row['canonical_form_3'])

    return ban_list


def should_ban_word(severity, severity_threshold):
    if severity <= severity_threshold:
        return True

    probability = (severity - severity_threshold) / (5 - severity_threshold)
    return random.random() < probability


def contains_banned_words(text, ban_list):
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        if word in ban_list:
            return True
    return False



def load_dataset(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def split_data(data, split_ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data


def save_datasets_to_files(train_data, val_data, train_file, val_file):
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(item['lyrics'] + "\n")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(item['lyrics'] + "\n")

data = load_dataset('processed_songs.csv')
train_data, val_data = split_data(data)

# Save the train and validation datasets to text files
save_datasets_to_files(train_data, val_data, "train.txt", "val.txt")


# Load a pre-trained GPT-2 model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

# Create dataset for training and validation
train_dataset = TextDataset(tokenizer=tokenizer, file_path="train.txt", block_size=128)
val_dataset = TextDataset(tokenizer=tokenizer, file_path="val.txt", block_size=128)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configure training
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()

model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")


def create_theme_matrix(theme_list):
    theme_matrix = {}

    # Count the number of themes in the list
    theme_count = len(theme_list)

    # Generate the segment size
    total_combinations = 10000
    segment_size = total_combinations // theme_count
    remaining_combinations = total_combinations % theme_count

    # Assign the 4-digit number ranges to themes
    start = 0
    for i, theme in enumerate(theme_list):
        end = start + segment_size - 1
        if remaining_combinations > 0:
            end += 1
            remaining_combinations -= 1
        theme_matrix[(start, end)] = theme
        start = end + 1

    return theme_matrix



def find_rhymes(word):
    return pronouncing.rhymes(word)

def enforce_rhyme_scheme(lines, rhyme_scheme):
    if rhyme_scheme == "AABB":
        rhyme_indices = [0, 0, 1, 1]
    elif rhyme_scheme == "ABAB":
        rhyme_indices = [0, 1, 0, 1]
    elif rhyme_scheme == "ABBA":
        rhyme_indices = [0, 1, 1, 0]
    elif rhyme_scheme == "ABCC":
        rhyme_indices = [0, 1, 2, 2]
    elif rhyme_scheme == "ABCB":
        rhyme_indices = [0, 1, 2, 1]

    rhyming_lines = [None] * 4
    for i, line in enumerate(lines[:4]):
        if not line.strip():
            line = " "  # Assign an empty string if the line is empty
        words = line.split()
        if not words:  # Add this check
            continue
        word = words[-1]
        if rhyming_lines[rhyme_indices[i]] is None:
            rhyming_lines[rhyme_indices[i]] = line
        else:
            rhymes = find_rhymes(word)
            for rhyme in rhymes:
                if rhyme in rhyming_lines[rhyme_indices[i]].split():
                    rhyming_lines[i] = line
                    break
            # Add this line to ensure a non-None value is always assigned
            if rhyming_lines[i] is None:
                rhyming_lines[i] = line

    return rhyming_lines

def count_word_frequencies(text):
    words = re.findall(r'\b\w+\b', text.lower())
    counter = Counter(words)
    return counter


def generate_lyrics(model, tokenizer, prompt, max_length=75, max_word_frequency_ratio=0.1, ban_list=None):
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    while True:

        lyrics = generator(prompt, max_length=max_length, do_sample=True, top_p=0.9, top_k=30)
        generated_text = unidecode(lyrics[0]['generated_text'])  # Convert non-ASCII characters to ASCII
        word_frequencies = count_word_frequencies(generated_text)
        total_words = sum(word_frequencies.values())

        # Check if any word exceeds the allowed frequency ratio
        if not any(count / total_words > max_word_frequency_ratio for count in word_frequencies.values()):
            # Check if the generated text contains any banned words
            if ban_list is None or not contains_banned_words(generated_text, ban_list):
                break

    # Limit repetitions with probabilities
    lines = generated_text.split('\n')
    limited_repetition_lines = limit_repetitions(lines)
    max_repeats = 4
    repeats = 0
    prev_line = None
    repeat_probabilities = [np.random.randint(40, 60) / 100, np.random.randint(25, 35) / 100, np.random.randint(10, 20) / 100, np.random.randint(1, 10) / 100]

    for line in lines:
        if line != prev_line:
            repeats = 1
            limited_repetition_lines.append(line)
        else:
            if repeats < max_repeats:
                repeat_prob = repeat_probabilities[repeats - 1]
                if np.random.rand() < repeat_prob:
                    limited_repetition_lines.append(line)
                    repeats += 1
        prev_line = line

    # Enforce rhyme scheme
    rhyme_scheme = random.choice(["AABB", "ABAB", "ABBA", "ABCC", "ABCB"])
    rhyming_lyrics = []
    for i, line in enumerate(limited_repetition_lines):
        rhyming_lyrics.append(line)
        if i % 4 == 3:
            rhyming_lyrics[-4:] = enforce_rhyme_scheme(rhyming_lyrics[-4:], rhyme_scheme)

    filtered_rhyming_lyrics = [lyric for lyric in rhyming_lyrics if lyric is not None]
    generated_text = '\n'.join(filtered_rhyming_lyrics)
    return generated_text


def create_title(model, tokenizer, prompt, existing_titles):
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    title = ""
    while True:
        titles = generator(prompt, max_length=10, do_sample=True, top_p=0.9, top_k=30)
        title = unidecode(titles[0]['generated_text'].strip())
        if title not in existing_titles:
            break
    return title

def find_repeating_phrases(lyrics):
    words = lyrics.split()
    n = len(words)
    phrases = []
    for i in range(n):
        for j in range(i + 2, n + 1):
            phrase = ' '.join(words[i:j])
            if lyrics.count(phrase) > 1 and phrase not in phrases:
                phrases.append(phrase)
    return phrases

def limit_repetitions(lines):
    limited_repetition_lines = []
    max_repeats = 4
    repeats = 0
    prev_line = None
    repeat_probabilities = [np.random.randint(40, 60) / 100, np.random.randint(25, 35) / 100, np.random.randint(10, 20) / 100, np.random.randint(1, 10) / 100]

    for line in lines:
        if line != prev_line:
            repeats = 1
            limited_repetition_lines.append(line)
        else:
            if repeats < max_repeats:
                repeat_prob = repeat_probabilities[repeats - 1]
                if np.random.rand() < repeat_prob:
                    limited_repetition_lines.append(line)
                    repeats += 1
        prev_line = line

    return limited_repetition_lines


def insert_chorus(lyrics, chorus):
    lines = lyrics.split("\n")
    lines_with_chorus = []
    verse_counter = 0

    for line in lines:
        lines_with_chorus.append(line)
        verse_counter += 1

        if verse_counter % 4 == 0:
            lines_with_chorus.append(chorus)
            verse_counter = 0

    return "\n".join(lines_with_chorus)


def generate_songs(vault_code, model, tokenizer, prompt, theme_list, ban_list, num_songs=10, max_length=600):
    songs = []
    titles = set()

    # Generate theme matrix
    theme_matrix = create_theme_matrix(theme_list)

    # Generate album theme
    album_number = int(vault_code)
    album_theme = None
    for start, end in theme_matrix:
        if start <= album_number <= end:
            album_theme = theme_matrix[(start, end)]
            break

    for i in range(num_songs):
        # Generate song themes
        num_song_themes = random.randint(1, 3)
        song_themes = random.sample(theme_list, num_song_themes)

        # Update the prompt to include the album theme and song themes
        prompt_with_themes = f"Vault code: {vault_code}. Album theme: {album_theme}. Song themes: {', '.join(song_themes)}."


        # Generate lyrics
        generated_lyrics = generate_lyrics(model, tokenizer, prompt_with_themes, max_length, ban_list=ban_list)



        generated_lyrics = re.sub(r'((?:.*?\n){4})\1+', r'\1', generated_lyrics)  # Limit repeating lines to 4
        generated_lyrics = re.sub(r'\W+$', '', generated_lyrics)

        # Change 3: Create and insert choruses into the generated songs
        repeating_phrases = find_repeating_phrases(generated_lyrics)
        if repeating_phrases:
            chorus = random.choice(repeating_phrases)
            generated_lyrics = insert_chorus(generated_lyrics, chorus)

        # Change 2: Use GPT-2 to generate song titles
        title = create_title(model, tokenizer, prompt, titles)
        titles.add(title)
        songs.append((title, generated_lyrics))

    return songs


def generate_song_structure():
    structure = ["intro"]
    sections = ["verse", "pre-chorus", "chorus", "bridge", "outro"]
    probabilities = [
        [0.8, 0.1, 0.1, 0, 0],  # intro
        [0.2, 0.3, 0.4, 0.1, 0],  # verse
        [0.1, 0.1, 0.7, 0.1, 0],  # pre-chorus
        [0.1, 0, 0.1, 0.4, 0.4],  # chorus
        [0.2, 0, 0.6, 0.1, 0.1],  # bridge
        [0, 0, 0, 0, 1]  # outro
    ]
    verse_count = 1
    current_section = 0

    for _ in range(6):
        next_section = np.random.choice(5, p=probabilities[current_section])
        section_name = sections[next_section]

        if section_name == "verse":
            section_name += f" {verse_count}"
            verse_count += 1

        structure.append(section_name)
        current_section = next_section

    # Reset verse counter for the next song
    verse_count = 1

    # Format the structure with gaps between sections
    formatted_structure = "\n".join(structure)

    return formatted_structure

def is_valid_vault_code(code):
    return code.isdigit() and len(code) == 4

vault_code = input("Enter the vault code: ")

while not is_valid_vault_code(vault_code):
    print("Invalid input. Please enter a 4-digit number.")
    vault_code = input("Enter the vault code: ")

prompt = f"Vault code: {vault_code}"

ban_list = create_ban_list_from_csv('profanity_en.csv')

generated_songs = generate_songs(vault_code, model, tokenizer, prompt, theme_list, ban_list, num_songs=10, max_length=600)


def separator():
    return "─" * 30

for index, (title, lyrics) in enumerate(generated_songs, start=1):
    print(separator())
    print(f"Song {index}: {title}")
    print(separator())

    for line in lyrics.splitlines():
        wrapped_line = textwrap.wrap(line, width=30)
        for wrapped in wrapped_line:
            print(wrapped)
        print()

    print("\n")


def plot_metrics(train_loss, eval_loss, eval_perplexity):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, eval_loss, label="Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Evaluation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_perplexity, label="Evaluation Perplexity")
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.title("Evaluation Perplexity")

    plt.savefig("loss_and_perplexity.png")  # Save the plots as a PNG file
    plt.show()

train_loss = [entry['loss'] for entry in trainer.state.log_history[::2]]  # Extract training loss from log history
eval_loss = [entry['eval_loss'] for entry in trainer.state.log_history[1::2]]  # Extract evaluation loss from log history
eval_perplexity = [2**(loss / 0.693) for loss in eval_loss]  # Compute perplexity from evaluation loss

plot_metrics(train_loss, eval_loss, eval_perplexity)
