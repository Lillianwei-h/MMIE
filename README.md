# MMIE-Eval
## Preparation
### Data
Your data format should be:
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

### Environment
```bash
conda create -n MMIE python=3.11
conda activate MMIE
pip install -r requirements.txt
```

### Model
You can download our [MMIE-Eval](https://huggingface.co/Lillianwei/MMIE-Eval) model on huggingface.
You can also refer to the document of [InternVL 2.0](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html) to find more details since MMIE-Eval is a fine-tuned version of InternVL 2.0 4B.

## Run
```bash
python main.py --model_path PATH_TO_MMIE-Eval --input_dir INPUT_DIR --input_file INPUT_FILE
```

The output file should be at `./eval_outputs/eval_result.json` by default. You can also use arguments `--output_dir` and `--output_file` to specify your intended output position.