# Speech Deepfake Detection


## Prerequisites

Before running this script, ensure you have Python 3.9 installed on your system. This version is recommended for the best compatibility with the dependencies used in this script.

### Installing Dependencies

To install all required dependencies, use the provided `requirements.txt` file:

```
pip install -r requirements.txt
```


### Installation

1. Clone the repository or download the script to your local machine.
2. Clone the Pengi repository:
   ```
   cd models/
   git clone https://github.com/microsoft/Pengi
   ```
3. Download Pengi weights `base_no_text_enc` and move to Pengi/configs folder
4. Ensure that any additional libraries for models (`wrapper`, `msclap`) are properly installed and configured.

## Usage

To use the script, you will need to provide the path to the folder containing the `.WAV` files and the type of embedding to process. The script supports the following embedding types: `pengi`, `clap`, `wavlm`, `wav2vec`.

### Command Line Arguments

- \`folder\`: The path to the folder containing your `.WAV` files.
- \`embedding_type\`: The type of embedding to process. Valid options are `pengi`, `clap`, `wavlm`, `wav2vec`.

### Running the Script

Run the script from the command line by navigating to the script's directory and executing the following command:

```
python extract_embeddings.py [path_to_folder] [embedding_type]
```

For example, to process audio files in the folder \`timit/\` with the \`wavlm\` embedding type, you would run:

```
python extract_embeddings.py timit/ wavlm
```


### Output

The script will process each `.WAV` file in the specified folder and generate a corresponding `.<embedding_type>.npy` file containing the audio embeddings. For example, if the chosen embedding type is `wavlm`, the output file for `sample.WAV` will be named `sample.wavlm.npy`. These `.npy` files, named to reflect the embedding type used, will be saved in the same directory as the source `.WAV` files.

