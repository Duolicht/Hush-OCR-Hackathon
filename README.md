# OCR-to-JSON Conversion using Fine-Tuned PHI-3-Mini Model

Welcome to the OCR-to-JSON project repository. This project, developed by Siddhant and Pavana as part of a hackathon, focuses on fine-tuning the PHI-3-Mini model to convert OCR (Optical Character Recognition) results from receipts and invoices into structured JSON format.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook Details](#notebook-details)
- [Deployment](#deployment)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This repository contains code for:
- Fine-tuning the PHI-3-Mini model with OCR data.
- Converting OCR results from receipts and invoices into structured JSON.
- Deploying the fine-tuned model using a deployment script.

## Installation

Clone the repository and install the necessary packages:

```bash
git clone https://github.com/yourusername/ocr-to-json.git
cd ocr-to-json
pip install -r requirements.txt
```

## Usage

To interact with the model and convert OCR results to JSON, follow these steps:

1. **Run the Jupyter Notebook**: Start by running the `ocr-to-json.ipynb` notebook to understand the data preparation, model fine-tuning, and evaluation process.
2. **Deploy the Model**: Use the `app.py` script to deploy the fine-tuned model.

```bash
python app.py
```

## Notebook Details

### Installing and Importing Packages

The notebook installs and imports essential libraries such as `transformers`, `peft`, `accelerate`, `bitsandbytes`, and others required for data manipulation, visualization, and model interaction.

### Loading and Preparing Data

The dataset, containing OCR results from receipts and invoices, is loaded and split into training and evaluation subsets to ensure a representative distribution for effective model training and validation.

### Alpaca Prompt Template

The Alpaca prompt template structures input for the PHI model fine-tuning, setting the context for generating responses based on the instructional input.

### Configuring Parameters

Detailed configuration includes bits and bytes parameters, training arguments, and SFT (Supervised Fine-Tuning) parameters. Key configurations include using 4-bit precision, gradient checkpointing, and the AdamW optimizer.

### Training the Model

Fine-tuning is performed using QLoRA (Quantized Low-Rank Adaptation) and SFT training techniques. The trained model is then merged with LoRA weights and pushed to the Huggingface hub.

### Merging and Pushing to Huggingface

After training, the model is reloaded in FP16, merged with LoRA weights, and then pushed to the Huggingface hub for easy access and deployment.

## Deployment

An `app.py` script is provided for deploying the fine-tuned model. Run the script to start the application:

```bash
python app.py
```

## Demo

Watch the demo video to see the model in action.

https://github.com/Duolicht/Hush-OCR-Hackathon/assets/174272054/e3c20547-058c-4b27-a7f6-216b48973bf9

## Contributing

We welcome contributions to improve this project. Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the Huggingface team for providing the models and tools necessary for this project.
