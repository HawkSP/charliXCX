# Charli XCX AI Lyric Generation Project

## Overview
This repository houses the Charli XCX AI Lyric Generation Project. Our primary objective is to create an AI model capable of generating lyrics mimicking the unique style of Charli XCX, a prominent figure in the music industry.

## Key Features
- **Advanced NLP and Deep Learning:** We leverage state-of-the-art techniques in natural language processing and deep learning to achieve our goal.
- **Open to Collaboration:** This project is open to contributions and invites collaborators interested in music, AI, or both.

Project Structure
The project is structured as follows:

main.py - This is the main Python file that orchestrates the functionality of the project, including loading the data, training the GPT-2 model, and generating songs.

profanity_en.csv - This is a CSV file that contains a list of words to be banned from the generated lyrics. It contains the words in various forms, as well as a 'severity_rating' indicating the severity of each word.

processed_songs.csv - This is a CSV file containing the processed song lyrics dataset that is used to train the GPT-2 model.

output/ - This is a directory where the trained GPT-2 model is saved along with its configuration and tokenizer.

trained_model/ - This is a directory where the fine-tuned GPT-2 model is saved for future use.

# Project Structure

The project is structured as follows:

- `train.py` - This is the main Python file that orchestrates the functionality of the project, including loading the data, training the GPT-2 model, and generating songs.

- `profanity_en.csv` - This is a CSV file that contains a list of words to be banned from the generated lyrics. It contains the words in various forms, as well as a 'severity_rating' indicating the severity of each word.

- `processed_songs.csv` - This is a CSV file containing the processed song lyrics dataset that is used to train the GPT-2 model.

- `output/` - This is a directory where the trained GPT-2 model is saved along with its configuration and tokenizer.

- `trained_model/` - This is a directory where the fine-tuned GPT-2 model is saved for future use.

# Getting Started

To get started with the project, you need to clone the project and install the necessary dependencies. Here's how you can do this:

## Prerequisites

- Python 3.7 or above.
- Git installed on your system.

## Steps

1. Clone the project using git.

```bash
git clone charliXCX
cd charliXCX
```

2. Install the necessary python packages. It is recommended to create a virtual environment before proceeding with this step.

```bash
pip install -r requirements.txt
```

3. Run the main Python file.

```bash
python train.py
```

During execution, the program will ask you to input a 4-digit vault code. Enter any 4-digit number and the program will begin generating songs based on this vault code.

## Notes

- The program might take a while to execute depending on your system configuration because it involves training a GPT-2 model.

- Ensure that you have a stable internet connection for downloading the pre-trained GPT-2 model and its tokenizer.

- Depending on the vault code and the generated themes, the output of the program can vary widely.


## License
This project is licensed under the MIT License. For more information, please refer to the [LICENSE](LICENSE) file.

## Contact
If you have any questions, issues, or if you would like to contribute to the project, please feel free to reach out.

elliott@iamdedeye.com
