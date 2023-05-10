import time

from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import csv
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from transformers import pipeline
from collections import Counter
import re
import matplotlib.pyplot as plt
from unidecode import unidecode
import numpy as np
import pronouncing
import textwrap
import openai
from cryptography.fernet import Fernet
import urllib.request
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO


app = Flask(__name__)

# Replace this with the encrypted version of your OpenAI API key
encrypted_api_key = b'gAAAAABkWwbwk-uIRQkm8NXaOHpypTl8K6NK50_7A384ksI4uqV_worDJ5trP3VSN7k5Nqf9FBfzNV1QlUMVzLrrECwkJTWJfBzON25X3nqKpkeX0-6mSCUERKFTujrENoW4WdvZO9MLxOrHrPEGJqhvTHUSsQd0Ow=='

# Replace this with the encryption key used to encrypt the API key
key = 

# Create a Fernet object with the key
fernet = Fernet(key)

# Decrypt the API key
api_key = fernet.decrypt(encrypted_api_key).decode()

openai.api_key = str(api_key)

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


# Load a pre-trained GPT-2 model
model_path = "trained_model"

# Load the model
model = GPT2LMHeadModel.from_pretrained('trained_model')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("trained_model")

def generate_lyrics(model, tokenizer, prompt, max_length=75, max_word_frequency_ratio=0.1, ban_list=None):
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    while True:
        generated_text = ""  # Reset the variable at the beginning of each iteration
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

    all_themes = []

    for i in range(num_songs):
        # Generate song themes
        num_song_themes = random.randint(1, 3)
        song_themes = random.sample(theme_list, num_song_themes)
        all_themes.extend(song_themes)  # Adding song themes to the list

        # Update the prompt to include the album theme and song themes
        prompt_with_themes = f"Vault code: {vault_code}. Album theme: {album_theme}. Song themes: {', '.join(song_themes)}."

        # Generate lyrics
        generated_lyrics = generate_lyrics(model, tokenizer, prompt_with_themes, max_length, ban_list=ban_list)
        generated_lyrics = re.sub(r'((?:.*?\n){4})\1+', r'\1', generated_lyrics)  # Limit repeating lines to 4
        generated_lyrics = re.sub(r'\W+$', '', generated_lyrics)

        # Create and insert choruses into the generated songs
        repeating_phrases = find_repeating_phrases(generated_lyrics)
        if repeating_phrases:
            chorus = random.choice(repeating_phrases)
            generated_lyrics = insert_chorus(generated_lyrics, chorus)

        # Use GPT-2 to generate song titles
        title = create_title(model, tokenizer, prompt, titles)
        titles.add(title)
        songs.append((title, generated_lyrics))

    # Create a string containing all themes
    all_themes_prompt = ', '.join(all_themes)
    return songs, all_themes_prompt


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


def separator():
    return "─" * 30


def limit_word_occurrences(text, phrase, max_occurrences):
    occurrences = text.count(phrase)
    if occurrences > max_occurrences:
        # Remove extra occurrences
        text = text.replace(phrase, '', occurrences - max_occurrences)
    return text



prompt = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        vault_code = request.form.get('vault_code')  # Get the vault code from the form
        if not is_valid_vault_code(vault_code):
            return "Invalid input. Please enter a 4-digit number.", 400
        global prompt  # Make sure to update the global variable
        prompt = f"Vault code: {vault_code}"
        global vault_code_input
        vault_code_input = vault_code
        return redirect(url_for('generate_songs_route'))  # Redirect to the vault transition route
    return render_template('index.html')  # Render the form

@app.route('/vault')
def vault():
    return render_template('vault.html')

@app.route('/generate_songs')
def generate_songs_route():
    # Using the globally stored prompt now
    if prompt is None:
        return "No Vault Code provided. Please try again.", 400
    ban_list = create_ban_list_from_csv('profanity_en.csv')

    generated_songs, all_themes_prompt = generate_songs(vault_code_input, model, tokenizer, prompt, theme_list, ban_list, num_songs=10, max_length=600)

    # Create a list of dictionaries with song titles and lyrics
    songs_data = []
    for title, lyrics in generated_songs:
        lyrics = limit_word_occurrences(lyrics, "Song", 1)
        lyrics = limit_word_occurrences(lyrics, "Vault code", 1)
        lyrics = limit_word_occurrences(lyrics, "themes", 1)
        lyrics = limit_word_occurrences(lyrics, "theme", 1)
        lyrics = limit_word_occurrences(lyrics, "Album", 1)
        lyrics = lyrics.replace('\n', '<br>')  # replace newline characters with <br> tags
        songs_data.append({'title': title.replace('\n', ' '), 'lyrics': lyrics})

    # Write the songs data to a json file
    with open('static/songs.json', 'w') as json_file:
        json.dump(songs_data, json_file)

    user_prompt = "Charli XCX album cover, cute, beautiful eyes, vibrant, realistic, hyper realism," + ",".join(all_themes_prompt)

    # Generate the image
    response = openai.Image.create(
        prompt=user_prompt,
        n=1,
        size="512x512"
    )

    # Get the image URL
    image_url = response['data'][0]['url']

    # Use requests to get the image data
    response = requests.get(image_url)

    # Convert the response content to an image
    image = Image.open(BytesIO(response.content))

    # Save the image to your static folder using the vault code as its name
    image_path = os.path.join(app.root_path, 'static', f'{vault_code_input}.png')
    image.save(image_path)

    if not os.path.exists("static"):
        os.makedirs("static")

    return render_template('songs.html', songs=generated_songs, image_path=f'{vault_code_input}.png')


@app.route('/lyrics')
def lyrics_route():
    with open('static/songs.json', 'r') as f:
        songs = json.load(f)
    return render_template('lyrics.html', songs=songs)  # assumes you have a lyrics.html file in templates folder

if __name__ == '__main__':
    app.run(debug=True)