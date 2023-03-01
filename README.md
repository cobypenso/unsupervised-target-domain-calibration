Code release for ["Network Calibration under Distribution Shift based on Estimating the  Target Domain Accuracy"]


# Histogram-Transfer Temperature scaling (HTTS) Calibration:
### Step 1: Train a domain adaptation model using various DA methods
Run 'python3.6 1_train_DA_models.py'

### Step 2: Generate features by the fixed DA models for the source train, source validation and target dataset
Run 'python3.6 2_generate_features.py'

### Step 3: Estimate network accuracy on target data
Run 'python3.6 3_estimate_target_acc_using_metasets.py'

### Step 4: Calibration
Simply run 'python3.6 4_calibrate.py'

The code contains our method (all variants) and baselines.


## Dataset

### Office-31
Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/). 

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).
