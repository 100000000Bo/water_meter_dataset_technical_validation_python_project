# Water_meter_dataset_technical_validation_python_project
## Instructions for Using the Validation Code

This code is provided to validate the effectiveness of the proposed water meter dataset through **segmentation**, **recognition**, and **classification-assisted segmentation** experiments.

* * *

### Notes

*   File paths in each script should be adjusted according to the local directory structure.
*   All datasets used in the validation phase have been preprocessed to a unified size and fully labeled. **The data can be directly used for training and validation without any additional preprocessing.**

* * *

### 1\. Environment Setup

The validation code is implemented in **Python**.  
The recommended environment can be created using the provided environment configuration file (env_validation.yml), which defines and validates all required dependencies:

`conda env create -f env_validation.yml`
`conda activate <environment_name>`

Main dependencies include:

*   Python ≥ 3.8
*   PyTorch
*   torchvision
*   NumPy
*   Pandas
*   OpenCV
*   matplotlib
    

* * *

### 2\. Water Meter Segmentation Validation

The folder **`water meter segmentation/`** contains implementations of multiple semantic segmentation models used to evaluate the segmentation quality of the dataset.

#### 2.1 Directory Structure

The following scripts are provided, where each pair consists of a training script and a testing script:

*   `unet_train.py`, `unet_test.py`
*   `deeplabv3+_train.py`, `deeplabv3+_test.py`
*   `pspnet_train.py`, `pspnet_test.py`
*   `segformer_train.py`, `segformer_test.py`
    
All scripts share a similar structure and differ mainly in the network architecture.

*   `dataset/`: segmentation training and testing data
*   `model/`: saved trained segmentation models
*   `dataframe/`: experimental results from multiple training runs
*   `fig.py`: script for plotting evaluation curves
    

#### 2.2 Usage

Before running the code, place the prepared dataset into the `dataset/` directory and ensure that the dataset paths are correctly set in the corresponding scripts.
To train a segmentation model:
`python unet_train.py`
After training, the trained model weights will be automatically saved in the `model/` directory for subsequent loading and inference.
To test the trained model:
`python unet_test.py`
Repeat the above steps for other segmentation models as needed by modifying the corresponding training and testing scripts.
Running `fig.py` generates the segmentation performance figure: `seg.png`

* * *

### 3\. Water Meter Recognition Validation

The folder **`water meter recognition/`** contains scripts for validating the effectiveness of the digit recognition dataset.

#### 3.1 Directory Structure

*   `dataset/`: digit image datasets
*   `model/`: saved recognition models
*   `dataframe/`: recognition results across multiple runs
*   `fig.py`: script for result visualization

The following recognition models are provided:
*   `cnn_train.py`, `cnn_test.py`
*   `vgg16_train.py`, `vgg16_test.py`
*   `densenet_train.py`, `densenet_test.py`
*   `resnet_train.py`, `resnet_test.py`


#### 3.2 Usage

The usage procedure for the recognition model is the same as described in **Section 2.3**.  
Please ensure that the dataset is placed in the `dataset/` directory and that all paths are correctly configured.
To train a recognition model:
`python resnet_train.py`
To evaluate the trained model:
`python resnet_test.py`
The trained model will be saved in the `model/` directory for subsequent loading and inference.
Running `fig.py` generates the recognition performance figure: `rec.png`


* * *

### 4\. Water Meter Classification and Joint Validation

The folder **`water meter classification/`** is used to evaluate classification performance and to analyze whether classification information can enhance segmentation accuracy.

#### 4.1 Directory Structure

*   `Resnet_train+test.py`: training and testing script for water meter image classification

*   `seg_model/`: pretrained segmentation models used for joint testing
*   `dataframe/`: experimental results for plotting 
*   `fig.py`: visualization script

The following scripts introduce classification results into segmentation models:
*   `class+seg_deeplabv3+.py`
*   `class+seg_unet.py`
*   `class+seg_pspnet.py`
*   `class+seg_segformer.py`

#### 4.2 Classification Model Usage
Place the classification labels (provided as a CSV file) into the `dataset/` directory together with the corresponding images, and ensure that the dataset paths are correctly configured in the script.
Run the script to train and evaluate the classification model:
`python Resnet_train+test.py`
The trained classification model will be saved in the `class_model/` directory for subsequent loading and joint validation.

#### 4.3 Usage of classification-assisted segmentation
Before running the joint validation, place the pretrained segmentation models obtained in previous experiments into the `seg_model/` directory, and ensure that the dataset is correctly placed in the `dataset/` directory with proper path settings.
Run the corresponding classification-assisted segmentation script, for example:
`python class+seg_unet.py`
The script will automatically perform joint validation and output the final evaluation metrics, which are saved in the `dataframe/` directory for further analysis and visualization.
