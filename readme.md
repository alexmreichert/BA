# BA: Computer Vision for Cost-Efficient Part Identification in Additive Manufacturing

![alt text](images/pipeline_img.png)

## ID Pipeline

### LM

### Data Prep


### ID Registration
*Only runs in linux with teaser++ installation*
Runs parallelized over multiple containers
Follow steps:
1. Edit config to access service hub and storage account
2. Docker build and push image to azure container registry
3. Send out initialization message and ID message with the help of the [run_id.ipynb]run_id.ipynb)
4. 

### ID NN
Uses modified DGCNN to identify point cloud, uses model from ../TrainML

## Training Pipeline
Generate point cloud from STL


