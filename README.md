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
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

---

</div>

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
