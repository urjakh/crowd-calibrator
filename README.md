# crowd-calibrator

This is the code for ["Crowd-Calibrator: Can Annotator Disagreement Inform Calibration in Subjective Tasks?"](https://openreview.net/pdf?id=VWWzO3ewMS), accepted to COLM 2024.

models, data, and remaining code will be uploaded and shared soon!

## setup. 
To run the training and evaluation for this paper, please set up the environment: 
```bash 
# Create environment.
conda create -n crowd-calibrator python=3.9
conda activate crowd-calibrator

# Install packages.
python setup.py develop
pip install -r requirements.txt
```

## training.
First, create a config file (see `configs/train/example_config.json` for an example). 

Then, run the following:
```bash
crowd_calibrator/train.py -c configs/CONFIG_FILE_NAME.json
```

## evaluation. 
To evaluate a model on the test set, create a config file (see `configs/test/example_config.json`) and run the following: 
```bash
crowd_calibrator/test.py -c configs/CONFIG_FILE_NAME.json
```
## responses from gpt-4.
To get predictions from GPT-4 run the following: 
```bash
scripts/get_llm_responses.py -d PATH_TO_HUGGINGFACE_DATASET -s DATA_SPLIT -o PATH_TO_SAVE_OUTPUTS
```
