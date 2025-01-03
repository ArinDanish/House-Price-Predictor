# House Price Predictor (Rust-based ML Project)

This project is a **House Price Predictor** built using Rust. It includes a machine learning training pipeline, model deployment to AWS S3, and a Dockerized environment for easy setup and execution. A `Makefile` is provided to streamline project tasks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Makefile Commands](#makefile-commands)
6. [Model Deployment](#model-deployment)
7. [Technologies Used](#technologies-used)

---

## Project Overview

The **House Price Predictor** aims to predict house prices based on features such as location, size, number of rooms, and more. It uses a machine learning model trained on a dataset and deployed in an efficient and scalable manner.

Key highlights:
- Built with Rust for performance and reliability.
- Training pipeline for pre-processing data, training the model, and evaluating results.
- Trained model is automatically pushed to an AWS S3 bucket for deployment.
- Containerized using Docker for easy deployment and consistency.

---

## Features

- **Training Pipeline:** 
  - Data cleaning, feature engineering, model training, and evaluation.
- **Model Storage:** 
  - Trained models are stored securely in an AWS S3 bucket.
- **Docker Integration:** 
  - Ensures portability and reproducibility.
- **Automation with Makefile:** 
  - Simplifies common tasks like training, testing, and deployment.

---

## Project Structure

house-price-predictor/ ├── src/ │ ├── lib.rs │  └── main.rs ├── Dockerfile ├── Makefile ├── config/ │ └── settings.toml ├── data/ │ ├── raw/ │ └── processed/ ├── models/ │ └── house_price_model.bin ├── README.md └── LICENSE
---

## Setup and Installation

### Prerequisites

- Rust (latest stable version)
- Docker
- AWS CLI configured with necessary credentials
- `make`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-predictor.git
   cd house-price-predictor
   
### Model Deployment
The trained model is stored in the models/ directory and automatically uploaded to an AWS S3 bucket during the pipeline. Ensure your AWS credentials are configured properly.

### Technologies Used
Programming Language: Rust
Machine Learning Framework: TBD (e.g., SmartCore, Linfa, etc.)
Cloud Storage: AWS S3
Containerization: Docker
Build Automation: Makefile

