# BA: Computer Vision for Cost-Efficient Part Identification in Additive Manufacturing

![alt text](images/pipeline_img.png)

## ID Pipeline

### LM
* upload.py : send upload message to *imagegeneration* of the training pipeline
    * set *ml* to *false* to train for the registration pipeline
    * to generate more than one variation of dataset remove if statement at the end
* log.py : download log messages and saves to csv
* motorcontroller.py : turn motor when recieving a message 
* scan.py: scans when recieves message of turn and then upload pcd to *containerlmdp*

### Data Prep
*edit config before using*
* Config
    * runs if config.pkl is unavailable
    * finds center turntable
    * finds ellipse distortion
    * finds y-axis distortion
    * ![alt text](images/config.png)
* Normal
    * runs if config.pkl is available
    * loads pcd from storage *containerlmdp*
    * loads center and turntable plane
    * removes cylinder around center
    * runs DBSCAN
    * Turns pcd and applies 
    * filter out bad registration results with low correspondence pairs and high RMSE
    * uploads to *containerdpir*
    * ![alt text](images/AnimationICP.gif)

### ID Registration
*Only runs in linux with teaser++ installation*
Runs parallelized over multiple containers
Follow steps:
1. Edit config to access service hub and storage account
2. Docker build and push image to azure container registry
3. Send out initialization message and ID message with the help of the [run_id.ipynb](IdentificationReg/run_id.ipynb)
4. Start multiple container instances of the same image
5. Accumulate results and save to csv in [run_id.ipynb](IdentificationReg/run_id.ipynb)

### ID NN
* Uses modified DGCNN to identify point cloud, uses model from ../TrainML. Choose the variation number to 
* The model is **not** translation variant all point clouds need to be centered using their average center of points using the open3d library. 
* Identification and evaluation with [test_ID.ipynb](IdentificationML/test_ID.ipynb). *remember to correct label order in labels.txt when generating new Dataset*

## Training Pipeline
### Train Reg
Generate Point Cloud from STL, File dictionary and Annoy tree for ID

### Train ML
Generate Point Clouds from STL, create dataset and the Train a model using this dataset. Google Colab is used to train the Model



