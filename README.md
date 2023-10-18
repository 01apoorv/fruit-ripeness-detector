# Vision model to detect ripeness of fruit
Created using a YOLOv8 pretrained model and the Banana Ripening Process dataset, available from [here](https://universe.roboflow.com/fruit-ripening/banana-ripening-process/dataset/2). All credit for the dataset goes to them.

## Running the model
The `main.py` file containes a script to run the model using the front-facing camera of a laptop. The script was written on a MacBook and therefore uses Metal Performance Shaders (MPS) for graphics processing. This can be changed to use the PyTorch backend of your choice, for example, CUDA. The VideoCapture value may also need to be changed.

## Requirements
* PyTorch
* OpenCV
* NumPy
* Ultralytics (YOLOv8)
