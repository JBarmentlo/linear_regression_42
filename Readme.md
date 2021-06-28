# ft_linear_regression

## Prerequisites:
If you do not have python3.8, run:   
```bash
apt-get install python3.8
``` 
To create a virtual environment, run:   
```bash
python3.8 -m venv venv
```   
Then:
```bash
source venv/bin/activate
```   
Finally:
```bash
pip install --upgrade pip
pip install -r requirements.txt
sudo apt-get install python3-tk
```

## I. Visualise Data
Run ```python plot_data.py``` to visualise the dataset.

## II. Training
Run ```python learn.py``` to train the model on the dataset that is given by the subject and stored in ```../datasets/data.csv```.

## III. Predicting
Run ```python3 predictor.py``` to make one prediction.

## IIII. Evaluate
Run ```python3 evaluate.py``` to evaluate the regression on the given dataset.
