# Age and Gender Prediction from Handwriting

This project uses ResNet18 models to predict the age and gender of a person based on their handwriting.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Eyalevi21/ageAndGender.git
    cd ageAndGender
    ```

2.  **Install dependencies:**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs the CPU version of PyTorch by default. If you have a CUDA-capable GPU, please follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally/) to install the appropriate version.*

## Usage

1.  **Ensure Model Files exist:**
    Make sure the trained model weights are present in the project directory:
    - `gender_model_final.pth`
    - `age_model_best_66acc.pth`

2.  **Run the GUI:**
    ```bash
    python gui.py
    ```

3.  **Using the App:**
    - Click **"📂 Image"** to select a handwriting image (supports .jpg, .png, .tif, etc.).
    - The application will automatically analyze the image and display the predicted Gender and Age.
