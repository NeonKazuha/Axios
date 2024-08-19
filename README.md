# Axios

Axios is a PyTorch-based transformer model trained from scratch on a dataset containing subtitles from some Marvel Cinematic Universe (MCU) movies. The primary purpose of this project is to generate text in the style of MCU dialogues.

## Features

- **Transformer Model**: Trained from scratch using PyTorch.
- **Text Generation**: Generates text based on input sequences, mimicking MCU-style dialogues.
- **Streamlit Frontend**: A user-friendly interface to interact with the model and generate text.
- **MIT License**: This project is open-source and available under the MIT license.

## Installation

To run this project locally, you'll need to have Python installed. Follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/axios.git
    cd axios
    ```

2. **Install Dependencies**:
    Use the provided `requirements.txt` file to install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Launch the Streamlit app using the command provided above.
2. Enter some starting text in the text area.
3. Adjust the number of tokens you want to generate using the slider.
4. Click the "Generate Text" button to see the generated dialogue.

## Built With

- **PyTorch**: For training the transformer model.
- **Streamlit**: For creating the interactive frontend.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions to this project are not accepted at this time.

---

*Note: This project was developed as a demonstration and may not cover all aspects of a production-ready application.*
