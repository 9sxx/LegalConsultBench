# Legal Consultation Data Processing Pipeline

---

## Project Overview

This project is a comprehensive pipeline for processing legal consultation datasets, handling raw data with questions, categories, content, and responses. It covers data cleaning, automated labeling, sampling, dataset construction, model inference, and performance evaluation, ideal for legal AI benchmarking. Tasks include issue classification, dispute focus identification, legal provision citation, solution generation, and comprehensive response generation. The pipeline uses Pandas, Ollama (for LLM inference), and BERTScore (for evaluation), with support for GPU acceleration and parallel processing. Manual validation by legal Ph.D. students ensures label quality.

---

## Project Structure

The pipeline consists of seven Python scripts and a manual validation step, executed sequentially:

1. **data_cleaning_1.py**: Initial cleaning, retaining key columns, extracting core response content, and removing invalid rows.
2. **data_labeling.py**: Automated labeling using Ollama (qwen2:72b) to extract legal provisions, solutions, and dispute focuses.
3. **data_cleaning_2.py**: Secondary cleaning, filtering low-quality labels and duplicates.
4. **data_analysis.py**: Sampling, selecting 100 samples per issue category (with >100 occurrences).
5. **Manual Validation**: Legal Ph.D. students validate sampled data for accuracy.
6. **data_processing.py**: Constructs five JSON datasets for issue classification, dispute focus, legal provisions, solutions, and comprehensive responses.
7. **main.py**: Performs batch inference using Ollama models to generate predictions.
8. **calculate_scores.py**: Evaluates model performance (Accuracy for issue classification, BERTScore for other tasks).

---

## Prerequisites

- **Python Version**: Python 3.10+ recommended.
- **Dependencies**: See `requirements.txt`, including:
  - `pandas`, `numpy`, `ollama`, `bert-score`, `torch`, `psutil`, `openpyxl`
  - Install with: `pip install -r requirements.txt`
- **Ollama Service**:
  - Install Ollama (see [Ollama website](https://ollama.ai/)).
  - Run service: `ollama serve`.
  - Pull model: `ollama pull qwen2:72b`.
- **Hardware**: GPU support optional (for faster inference and evaluation).
- **Input Data**: `datas/consultations_data.csv` (with columns: 咨询问题, 问题类别, 内容, 回复).

---

## Installation and Usage

1. **Create Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Ollama**:
   - Start service: `ollama serve`.
   - Pull model: `ollama pull qwen2:72b`.

4. **Verify GPU Support** (Optional):
   - Check GPU availability:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```
   - If no GPU, install CPU-only torch:
     ```bash
     pip uninstall torch
     pip install torch==2.3.0+cpu
     ```

5. **Verify Dependencies**:
   - Check installed packages:
     ```bash
     pip list
     ```
   - Ensure `pandas`, `numpy`, `ollama`, `bert-score`, `torch`, `psutil`, `requests`, `openpyxl` are installed.

6. **Run Scripts** (in order):
   ```bash
   python data_cleaning_1.py
   python data_labeling.py
   python data_cleaning_2.py
   python data_analysis.py
   # Manually validate sampled_data.xlsx to produce sampled_data_process.xlsx
   python data_processing.py
   python main.py
   python calculate_scores.py
   ```

---

## Directory Structure

```plaintext
project_root/
├── datas/
│   └── consultations_data.csv  # Initial dataset
├── sampled_data.xlsx           # Sampled dataset
├── sampled_data_process.xlsx   # Validated dataset
├── 问题类别.json               # Issue classification dataset
├── 争议焦点.json               # Dispute focus dataset
├── 法条引用.json               # Legal provision dataset
├── 解决方案.json               # Solution dataset
├── 整体回复.json               # Comprehensive response dataset
├── results/
│   ├── logs/                  # Logs
│   ├── <model_name>/
│   │   ├── <task_name>/
│   │   │   └── predictions.json  # Model predictions
├── scores.csv                  # Model scores
├── data_cleaning_1.py
├── data_labeling.py
├── data_cleaning_2.py
├── data_analysis.py
├── data_processing.py
├── main.py
├── calculate_scores.py
├── requirements.txt            # 依赖文件 / Dependencies
└── README.md                  # 本文件 / This file
```

---

## Notes

- **File Paths**: Ensure input files (e.g., `datas/consultations_data.csv`) are in the correct paths.
- **Ollama Service**: Verify Ollama is running and models are available before executing `data_labeling.py` and `main.py`.
- **Log Checking**: Review logs in `results/logs/` and `score_calculation_*.log` for errors.
- **Interrupt Handling**: `data_labeling.py` supports interruption with progress saved in `progress.txt`.
- **Manual Validation**: Requires legal experts to validate `sampled_data.xlsx`, producing `sampled_data_process.xlsx`.

---

## License

This project is for academic and research purposes. Ensure compliance with data privacy laws when handling legal consultation data.