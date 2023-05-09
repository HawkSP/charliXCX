import csv
import re

# List of artist names to remove
artist_names = ["Rihanna", "feat. Rihanna", "Candy", "Cupid", "Charli XCX", "Iggy Azalea","intro", "verse", "pre-chorus", "chorus", "post-chorus", "hook", "bridge", "breakdown", "instrumental", "solo", "interlude", "outro", "coda", "adlib", "refrain", "Charli XCX", "Iggy Azalea", "Hannah Diamond", "Mike G", "Christine and the Queens", "Troye Sivan", "Kim Petras", "Tommy Cash", "Brooke Candy", "Pabllo Vittar", "Big Freedia", "CupcakKe", "Caroline Polachek", "Sky Ferreira", "Clairo", "Yaeji", "Dorian Electra", "Mykki Blanco", "Tommy Genesis", "Jay Park", "Slayyyter", "MØ", "Lizzo", "Lil Yachty", "Tove Lo", "Mura Masa", "Carly Rae Jepsen", "MNEK", "Rita Ora", "Big Freedia", "CupcakKe", "Brooke Candy", "Pabllo Vittar", "&", "Clairo", "Yaeji", "Rina Sawayama", "MØ", "Tove Lo", "Alma", "Jay Park", "Starrah", "RAYE", "Uffie", "ABRA", "Carly Rae Jepsen", "Brooke Candy", "Sophie", "Allie X", "A.G. Cook", "Azealia Banks", "Banks", "FKA Twigs", "Grimes", "Halsey", "Hayley Kiyoko", "Kehlani", "Kero Kero Bonito", "Kesha", "Lady Gaga", "Lana Del Rey", "Lorde", "Lykke Li", "Marina"]

# Define a function to remove artist names from a given text
def remove_artist_names(text, names):
    for name in names:
        # Escape any special characters in the name
        name = re.escape(name)
        # Create a regular expression pattern to match the name, case-insensitive
        pattern = re.compile(name, re.IGNORECASE)
        # Replace any occurrence of the name with an empty string
        text = pattern.sub("", text)
    return text

# Read the original CSV file
input_file = "songs.csv"
output_file = "processed_songs.csv"

with open(input_file, "r", newline="", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header row to the new CSV file
    header = next(reader)
    writer.writerow(header)

    # Process each row, remove artist names, and write to the new CSV file
    for row in reader:
        lyrics = row[1]  # Assuming lyrics are in the second column
        processed_lyrics = remove_artist_names(lyrics, artist_names)
        row[1] = processed_lyrics
        writer.writerow(row)
