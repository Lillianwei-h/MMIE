<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/65941852f0152a21fc860f79/hLXIpSd7PzouZx0FULvgP.png'  width=300px>
  
# <p align="center"><b>MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models</b></p>
<p align="center">
<a href="https://mmie-bench.github.io">[📖 Project]</a>
<a href="https://arxiv.org/abs/xxxxx">[📄 Paper]</a>
<a href="https://github.com/Lillianwei-h/MMIE">[💻 Code]</a>
<a href="https://huggingface.co/datasets/MMIE/MMIE">[📝 Dataset]</a>
<a href="https://huggingface.co/MMIE/MMIE-Eval">[🤖 Eval Model]</a>
<a href="https://huggingface.co/spaces/MMIE/Leaderboard">[🏆 Leaderboard]</a>
<a href="#%F0%9F%8C%9F-overview">[🌟 Overview]</a>
<a href="#%F0%9F%94%A7-benchmark-details">[🔧 Benchmark Details]</a>
<a href="#%F0%9F%9A%A9-citation">[🚩 Citation]</a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

---
</div>

## 🌟 Overview

<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/65941852f0152a21fc860f79/Ks9yJtJh7fcyNJSUcKg-0.jpeg'>
</div>

We present **MMIE**, a **M**assive **M**ultimodal **I**nterleaved understanding **E**valuation benchmark, designed for Large Vision-Language Models (LVLMs). MMIE offers a robust framework for evaluating the interleaved comprehension and generation capabilities of LVLMs across diverse fields, supported by reliable automated metrics.

MMIE is curated from four multimodal datasets, encompassing:
- **3 categories**: Situational analysis, project-based learning, and multi-step reasoning.
- **12 fields**: Mathematics, physics, coding, statistics, literature, philosophy, education, finance, health, sports, art, and Electrical Engineering and Computer Science (EECS).
- **102 subfields**: Offering in-depth coverage across multiple domains.

The dataset contains **20,103 multimodal questions** that support both interleaved inputs and outputs. It includes a mix of multiple-choice and open-ended questions, evaluating a wide range of competencies and reasoning skills. Each query is paired with a **ground truth reference**, enabling effective evaluation.

In addition, we propose an **automated evaluation metric** powered by a scoring model, which is available for use at [MMIE-Eval](https://huggingface.co/MMIE/MMIE-Eval). This robust, automated evaluation metric, powered by Intern-VL2, to assess interleaved comprehension and generation capabilities across diverse fields.

This automated evaluation metric provides a reliable, streamlined approach to scoring LVLMs based on their performance in multimodal reasoning tasks. It is tailored to handle interleaved inputs and outputs, ensuring unbiased and consistent evaluation results.


🎯 **Key Features:**
Dataset:
- **Comprehensive Dataset**: Over 20K meticulously curated examples in interleaved multimodal format, merged into one all-in-one JSON file for easy access.
- **Diverse Evaluation**: Spanning 12 major fields and 102 subfields, MMIE ensures a broad evaluation of LVLM competencies.
- **Automated Scoring**: Test your model’s results against our evaluation metric available at [MMIE-Eval](https://huggingface.co/MMIE/MMIE-Eval).
---
Metric:
- **Automated Scoring System**: Fine-tuned **InternVL-2-4B** is employed as the foundation of the scoring system, offering high performance and support for multi-image input.
- **Bias Mitigation**: The model is fine-tuned to minimize biases and provide fair, objective scoring across all models tested.
- **Multimodal Focus**: Tailored to handle **interleaved multimodal inputs and outputs**, ensuring models are judged on their ability to integrate and reason with both text and images.
- **Human-like Evaluation**: Our metric shows high correlation with human annotations, surpassing alternative automated metrics like GPT-4o, especially in nuanced multimodal tasks.
- **Scalable and Consistent**: The evaluation metric is built to handle large-scale datasets, offering consistent and reproducible scoring results, making it perfect for model benchmarking and comparison.

---

## 🔧 Benchmark Details

<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/65941852f0152a21fc860f79/vrsgjTBcBYfZTdMQiJ1uC.png' width=500px>
</div>

### Dataset
MMIE is curated to evaluate models' comprehensive abilities in interleaved multimodal comprehension and generation. The dataset features diverse examples, categorized and distributed across different fields as illustrated above. This ensures balanced coverage across various domains of interleaved input/output tasks, supporting accurate and detailed model evaluations.

---

### Metric
#### Pipeline
<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/65941852f0152a21fc860f79/62MV7dB2_p2ptb2JXb6GH.png' width=50%>
</div>

To ensure a comprehensive and unbiased evaluation of various **LVLMs**, we propose an **automated evaluation metric** powered by **InternVL-2-4B**. This model was selected for its **strong performance in multimodal reasoning tasks** and its ability to support **multi-image inputs**. Furthermore, we fine-tuned the model to mitigate potential biases and provide accurate, consistent scoring.

The evaluation pipeline leverages the **internally fine-tuned LVLM** to assess models based on key dimensions such as **text quality**, **image quality**, **text-image coherence**, and **stylistic consistency**. This ensures models are rigorously tested on their multimodal reasoning capabilities.

#### Results

<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/65941852f0152a21fc860f79/YmDZxBR7OtWra5F016igi.png' width=90%>
</div>

*Note: In the image, higher values indicate better performance for Pearson and Cosine Similarity, while lower values are better for MSE and MAE.*

The MMIE evaluation metric demonstrates superior performance in scoring, achieving the highest correlation with **human annotations** in all aspects of multimodal comprehension and generation. It consistently outperforms GPT-4o and other standard evaluation metrics, proving its reliability for large-scale model benchmarking.

---
---


## Setup
We have host MMIE dataset on [HuggingFace]("https://huggingface.co/datasets/MMIE/MMIE"), where you should request access on this page first and shall be automatically approved.
Please download all the files in this repository and unzip `images.tar.gz` to get all images. We also provide `overview.json`, which is an example of the format of our dataset.


## Model Evaluation
### Setup
#### Dataset Preparation
Your to-eval data format should be:
```
[
    {
        "id": "",
        "question": [
            {
                "text": "...",
                "image": LOCAL_PATH_TO_THE_IMAGE or null
            },
            ...
        ],
        "answer": [
            {
                "text": "...",
                "image": LOCAL_PATH_TO_THE_IMAGE or null
            },
            ...
        ],
        "model": "gt",
        "gt_answer": [
            {
                "text": "...",
                "image": LOCAL_PATH_TO_THE_IMAGE or null
            },
            ...
        ]
    },
    ...
]
```

Make sure the file structure be:
```
INPUT_DIR
    |INPUT_FILE(data.json)
    |images
        |0.png
        |1.png
        |...
```

#### Installation
- Clone code from this repo
```bash
git clone https://github.com/Lillianwei-h/MMIE
cd MMIE
```
- Build environment
```bash
conda create -n MMIE python=3.11
conda activate MMIE
pip install -r requirements.txt
```

#### Model Preparation
You can download our [MMIE-Eval](https://huggingface.co/Lillianwei/MMIE-Eval) model on huggingface.
You can also refer to the document of [InternVL 2.0](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html) to find more details since MMIE-Eval is a fine-tuned version of InternVL 2.0 4B.

#### Run
```bash
python main.py --model_path PATH_TO_MMIE-Eval --input_dir INPUT_DIR --input_file INPUT_FILE
```

The output file should be at `./eval_outputs/eval_result.json` by default. You can also use arguments `--output_dir` and `--output_file` to specify your intended output position.

## Citation

If you find our benchmark useful in your research, please kindly consider citing us:

```bibtex
@article{xxx
}
