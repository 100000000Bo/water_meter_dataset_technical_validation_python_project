# Water_meter_dataset_technical_validation_python_project
## Instructions for Data Preprocessing

This code is provided to validate the effectiveness of the proposed water meter dataset through **segmentation**, **recognition**, and **classification-assisted segmentation** experiments.

* * *

### Notes

*   **The dataset used in this study can be downloaded from Dryad: <u>https://doi.org/10.5061/dryad.7d7wm3860</u>.**
*   File paths in each script should be adjusted according to the local directory structure.
*   We provide data preprocessing code to illustrate the data processing and dataset construction procedures.
However, all datasets used in this work have already been fully preprocessed and can be directly used **without any additional preprocessing**.
*   Some one-time data cleaning operations are omitted, as they do not affect the final dataset or the reproducibility of the experiments.

* * *

### Environment Setup

The preprocessing code is implemented in **Python**.  
The recommended environment can be created using the provided `env_preprocess.yml` file:

`conda env create -f env_preprocess.yml`
`conda activate <environment_name>`

Key dependencies include:

*   Python ≥ 3.8
*   PyTorch
*   LabelMe
*   OpenCV (cv2)
*   PIL
*   NumPy 
*   Pandas
*   imgviz

* * *

### Preprocessing Pipeline

1.  Image resizing
2.  LabelMe annotation
3.  Label format conversion
4.  Binary label generation
5.  Dataset construction
6.  Digit region extraction

* * *
### 1\. Image Resizing

The script `imgSize.py` is used to resize images to a fixed resolution and unify the image format to `.png`.

### 2\. Annotation with LabelMe

Images are annotated using **LabelMe**.

**Steps:**

1.  Activate the LabelMe environment and launch LabelMe: `conda activate labelme` `labelme`
2.  Define the label set in `labels.txt`: `__ignore__ _background_ number`
3.  Annotate images and save the generated `.json` files into the `label_json/` directory.
    
* * *

### 3\. Convert LabelMe Annotations to VOC Format

The script `labelme2voc.py` converts LabelMe annotations into VOC-style segmentation labels.

**Command:** `python labelme2voc.py label_json result --labels labels.txt --noobject`

**Input:** `label_json`: directory containing LabelMe `.json` files
    
**Output (generated in `result/`):**
*   `JPEGImages/`: original images
*   `SegmentationClass/`: segmentation label images

* * *

### 4\. Binary Label Generation

The scripts `rgb2binary.py` and `verify_preprocess.py` are used to convert RGB segmentation labels into binary masks.

**Process:**
*   Convert RGB label images to grayscale
*   Apply binary thresholding
*   Foreground pixels are set to 255, background to 0
*   The binary label images are saved in: `result/binary_label/`

* * *

### 5\. Dataset Renaming and Construction

The script `rec_set_construct.py` renames images and updates the corresponding Excel file in the recognition set.

**Functionality:**

*   Rename images with consistent indexing (e.g., `train0.jpg`, `train1.jpg`)
*   Update the first column of the Excel file accordingly

* * *

### 6\. Digit Region Extraction

The script `digit_seg.py` extracts and normalizes digit regions using binary segmentation masks.

**Inputs:**

*   Original images
*   Binary segmentation labels
*   A CSV file specifying which images should be processed

**Output:**

*   Normalized digit images with fixed resolution `200 × 60`
*   Saved in the `output_images/` directory

* * *
