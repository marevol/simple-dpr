# Simple DPR (Work in Progress)

This project implements a simplified version of **Dense Passage Retrieval (DPR)**, focusing on effective document retrieval and question-answering tasks using the **Amazon ESCI dataset**. The model utilizes dual encoders to separately encode queries and documents for efficient information retrieval.

## Features

- **Dual Encoder Architecture**: Train the DPR model with separate encoders for questions and contexts, enabling fast and scalable retrieval.
- **In-Batch Negatives**: Utilize in-batch negatives during training for effective contrastive learning.
- **Efficient Retrieval**: Encode and index large document corpora for efficient information retrieval tasks.

## Installation

### Requirements

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/marevol/simple-dpr.git
   cd simple-dpr
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

   This will create a virtual environment and install all the necessary dependencies listed in `pyproject.toml`.

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Data Preparation

This project relies on the **Amazon ESCI dataset** for training and evaluating the model. You need to download the dataset and place it in the correct directory.

1. **Download the dataset**:
   - Obtain the `shopping_queries_dataset_products.parquet` and `shopping_queries_dataset_examples.parquet` files from the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data).

2. **Place the downloaded files** in the `downloads` directory within your project folder:
   ```bash
   ./downloads/shopping_queries_dataset_products.parquet
   ./downloads/shopping_queries_dataset_examples.parquet
   ```

3. **Verify data paths**:
   - The `main.py` script is set to load the dataset from the `downloads` directory by default. If you place the files elsewhere, modify the paths in the script accordingly.

## Usage

### Running the Training Script

The `main.py` script demonstrates how to use the **Amazon ESCI dataset** to train the DPR model and evaluate its performance.

To run the training and evaluation:

```bash
poetry run python main.py
```

This script performs the following steps:

1. **Data Loading**: Loads product titles and queries from the Amazon ESCI dataset.
2. **Model Initialization**: Initializes the DPR model using a pre-trained language model (e.g., `bert-base-uncased`).
3. **Training**: Trains the DPR model using in-batch negatives for contrastive learning.
4. **Evaluation**: Evaluates the trained model on a test set and outputs performance metrics.

You can modify the script or dataset paths as needed.

### File Structure

- `main.py`: The main entry point for training and evaluating the DPR model with the Amazon ESCI dataset.
- `simple_dpr/model.py`: Defines the `DPRQuestionEncoder` and `DPRContextEncoder` model architectures.
- `simple_dpr/train.py`: Handles the training process, including loss calculation and optimization steps.
- `simple_dpr/evaluate.py`: Contains functions for evaluating the model's performance.
- `simple_dpr/`: Other utility modules used in the project.

### Output

Upon completion of the script:

1. **Model Saving**: A trained model will be saved in the `dpr_model` directory.
2. **Logging**: Training and evaluation logs, including loss and accuracy metrics, will be saved in `train_dpr.log`.
3. **Console Output**: Key performance metrics and progress information will be printed to the console.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
