# Axios

Axios is a PyTorch-based transformer model trained from scratch on a dataset containing subtitles from some Marvel Cinematic Universe (MCU) movies. The primary purpose of this project is to generate text in the style of MCU dialogues.

## Live Demo

Experience the Avengers Story Generator live at: [https://axiosav.streamlit.app/](https://axiosav.streamlit.app/)

## Features

- **Transformer Model**: Trained from scratch using PyTorch.
- **Text Generation**: Generates text based on input sequences, mimicking MCU-style dialogues.
- **Streamlit Frontend**: A user-friendly web interface to interact with the model and generate text.
- **MIT License**: This project is open-source and available under the MIT license.

## Getting Started

You can use the application directly through the [live demo](https://axiosav.streamlit.app/) or set it up locally by following the instructions below.

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Local Installation

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

3. **Download the Pre-trained Model**:
    - Ensure you have the `avengers.pth` model file in the project directory. If not, download it from [this link](#) and place it in the root directory of the project.

4. **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

    The application will open in your default web browser at `http://localhost:8501/`.

## Usage

### Online Usage

1. Visit the [Avengers Story Generator](https://axiosav.streamlit.app/).
2. Enter some starting text in the text area provided.
3. Adjust the number of tokens you want to generate using the slider.
4. Click the "Generate Text" button to see the generated dialogue.

### Local Usage

1. Follow the **Local Installation** steps above.
2. Open your browser and go to `http://localhost:8501/`.
3. Enter your desired starting text and specify the number of tokens to generate.
4. Click "Generate Text" to see the output.

## Project Structure

axios/
├── axiosthingy.py # Contains the model and preprocessing classes
├── app.py # Streamlit application
├── avengers.pth # Pre-trained model weights
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # MIT License


## Built With

- [PyTorch](https://pytorch.org/): For training and implementing the transformer model.
- [Streamlit](https://streamlit.io/): For building the interactive web interface.
- [Kaggle](https://www.kaggle.com/): Sourced the dataset of MCU subtitles.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions to this project are not accepted at this time.

## Acknowledgments

- Inspired by the amazing work of the Marvel Cinematic Universe.
- Thanks to Kaggle for providing access to valuable datasets.

---

*Note: This project was developed for educational purposes and may not cover all aspects of a production-ready application.*
