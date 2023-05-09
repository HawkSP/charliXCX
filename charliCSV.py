import csv
import random
import re


def clean_lyrics(raw_lyrics):
    # Remove song sections (e.g., [Chorus], [Verse 1])
    cleaned_lyrics = re.sub(r'\[.*?\]\n', '', raw_lyrics)

    # Remove any extra blank lines
    cleaned_lyrics = re.sub(r'\n{2,}', '\n', cleaned_lyrics).strip()

    return cleaned_lyrics


def get_input():
    song_number = input("Enter the song number (1-10, 'q' to quit): ")
    if song_number.lower() == 'q':
        return None
    lyrics = input("Enter the lyrics for this song: ")
    cleaned_lyrics = clean_lyrics(lyrics)
    return int(song_number), cleaned_lyrics


def main():
    data = []
    while True:
        song_info = get_input()
        if not song_info:
            break
        song_number, cleaned_lyrics = song_info
        code = "{:04}".format(random.randint(0, 9999))  # Generate random code as a zero-padded string
        data.append({"song_number": song_number, "code": code, "lyrics": cleaned_lyrics})

    with open("songs.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["song_number", "code", "lyrics"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    main()
