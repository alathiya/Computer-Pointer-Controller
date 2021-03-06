# Computer Pointer Controller

This project uses gaze detection model to control mouse pointer of computer based on gaze of user's eye and changes the pointer position accordingly. Projects runs multiple
model inference in workflow establishing preprocessing and inferencing pipeline. Worflow is breifly described below:
1. User image is preprocessed and given input to face detection model. This model detects face coordinates and return array of min and max coordinates for face. 
Output from this model is used to crop the image of face which is returned along with coordinates. 
2. Cropped face image from above step is preprocessed and input is given to head pose estimation model. This model detects Tait-Bryan angles (yaw, pitch or roll) from face image. 
3. Cropped face image from step 1 is preprocessed and input is given to facial landmark detection model. Model detects facial landmark coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
Here (x0, y0, x1, y1) are coordinates of left and right eye retrieved for further processing. 
4. Output from step 2 and step 3 are proprocessed and these inputs are given to gaze estimation model. This model estimates cartesian coordinates of gaze direction vector.
5. Gaze vector from step 5 is send to mouse controller to position the movement of pointer.    

## Project Set Up and Installation

### Setup and Dependencies

- Virtual environment is create from Anaconda Naviator. Name - "Computer Controller Project".
- Select python 3.7 during environ create step. This will build environ with python 3.7 installation.
- Launch the terminal window from navigator once the environ has launched. 
- Install the dependency package for OpenCV using command *pip install opencv-python*. 
- Install the dependency package for pyautogui using command *pip install PyAutoGUI*. 
- Install Pillow from Anaconda Navigator. 
- Install intel distribution of openvino for Windows 10 [here](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html)

### Initialize OpenVino environment

 As a first step we have to initialize openVINO Environment for windows:
```
C:\Program Files (x86)\IntelSWTools\openvino\bin>setupvars.bat
```

### Directory Structure

```
├───intel
│   ├───face-detection-adas-0001
│   │   ├───FP16
│   │   ├───FP32
│   │   └───INT8
│   ├───gaze-estimation-adas-0002
│   │   ├───FP16
│   │   ├───FP32
│   │   └───INT8
│   ├───head-pose-estimation-adas-0001
│   │   ├───FP16
│   │   └───FP32
│   └───landmarks-regression-retail-0009
│       ├───FP16
│       └───FP32
└───starter
    └───starter
        ├───bin
        └───src
            ├───.vs
            │   └───src
            │       └───v16
            └───__pycache__
 ```           

- src folder contains all the source python class for each model:-
  * face_detection.py 
     - This class contains code to load and intialize the face detection model, methods to preprocess input, run prediction and postprocess output. This model predicts the face coordinates from input image. 
     
  * facial_landmarks_detection.py
     - This class contains code to load and intialize the facial landmarks detection model, methods to preprocess input, run prediction and postprocess output. This model predicts the facial landmark coordinates from cropped face image.
     
  * head_pose_estimation.py
     - This class contains code to load and intialize the head pose estimation model, methods to preprocess input, run prediction and postprocess output. This model detect the head postion by predicting yaw - roll - pitch angles from input cropped face image.
     
  * gaze_estimation.py
     - This class contains code to load and intialize the gaze estimation model, methods to preprocess input, run prediction and postprocess output. This model predicts the gaze vector from input left eye image, right eye image and head pose angles.
     
  * input_feeder.py
     - Contains InputFeeder class which initialize VideoCapture as per the user argument and return the frames one by one.
     
  * mouse_controller.py
     - Contains MouseController class which take x, y coordinates value, speed, precisions and according these values it  moves the mouse pointer by using pyautogui library.

  * main.py
     - Contains workflow implementation for project. Argument parser is used to read all arguments from the command line. 
 
- bin folder contains demo video which user can use for testing the app and director structure image.

- intel folder contains subfolder for each model where corresponding inference files are downloaded for FP32, FP16 and INT8. 

### Downloading Model from OpenVino Zoo 

  *  [Face detection model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html)

    Run below command in terminal to download model inference files.

``` 
python <openvino directory>\deployment_tools\tools\model_downloader\downloader.py --name face-detection-adas-0001 --precisions FP32,FP16,INT8
```
  
  *  [Facial Landmark detection model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

    Run below command in terminal to download model inference files.

``` 
python <openvino directory>\deployment_tools\tools\model_downloader\downloader.py --name landmarks-regression-retail-0009 --precisions FP32,FP16,INT8
```

  *  [Head Pose estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)

    Run below command in terminal to download model inference files.

``` 
python <openvino directory>\deployment_tools\tools\model_downloader\downloader.py --name head-pose-estimation-adas-0001 --precisions FP32,FP16,INT8
```

  *  [Gaze estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

    Run below command in terminal to download model inference files.

``` 
python <openvino directory>\deployment_tools\tools\model_downloader\downloader.py --name gaze-estimation-adas-0002 --precisions FP32,FP16,INT8
```

## Demo

To run model on CPU use below command in terminal window:
```
python ./starter/starter/src/main.py -face_m ./intel/face-detection-adas-0001/FP32/face-detection-adas-0001 -head_pose_m ./intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -facial_m ./intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -gaze_m ./intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -i ./starter/starter/bin/demo.mp4 -l C:\Program Files (x86)\IntelSWTools\<openvino directory>\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll -d CPU -pt 0.6 -flags face_detect face_landmark_detect head_pose gaze_est
```

To run model on GPU use below command in terminal window:
```
python ./starter/starter/src/main.py -face_m ./intel/face-detection-adas-0001/FP32/face-detection-adas-0001 -head_pose_m ./intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -facial_m ./intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -gaze_m ./intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -i ./starter/starter/bin/demo.mp4 -d GPU -pt 0.6 -flags face_detect face_landmark_detect head_pose gaze_est
```

Link to demo can be found [here](https://youtu.be/lKKtN5I2mqM) 

## Documentation

### Arguments Documentation 

* main.py has below arguments: 

  * -h: Get information about all the command line arguments
  * -face_m: (required) Specify the path of Face Detection model's name as  shown below for specific precision FP32, FP16, INT8
  * -facial_m: (required) Specify the path of Facial landmarks Detection model's name for specific precision FP32, FP16
  * -head_pose_m: (required) Specify the path of hose pose Detection model's name for specific precision FP32, FP16
  * -gaze_m: (required) Specify the path of gaze estimation model's name for specific precision FP32, FP16, INT8
  * -i: (required) Specify the path of input video file or enter cam for taking input video from webcam.
  * -l: (optional) Specify the absolute path of cpu extension if some layers of models are not supported on the device.  
  * -d: (optional) Specify the target device to infer the video file on the model. Supported devices are: CPU, GPU, FPGA (For running on FPGA used HETERO:FPGA,CPU), MYRIAD. 
  * -pt: (optional) Specify the probability threshold for face detection model to detect the face accurately from video frame.
  * -flag: (required) Specify the flags from face_detect, face_landmark_detect, head_pose, gaze_est to visualize the output of corresponding models of each frame seperated by space.


## Benchmarks

  * I ran the model inference on CPU and GPU device on local machine given same input video and same virtual environment. Listed below are hardware versions:
    * CPU device - Intel(R) Core(TM) i7 - 8850H CPU @ 2.60GHz
    * GPU device - Intel(R) UHD Graphics 630
  * Due to non availability of FPGA and VPU in local machine, I did not run inference for these device types.


* FP32

| Type of Hardware | Total inference time              | Total load time | fps |
  |------------------|----------------------------------------------|----------------------------|------
  | CPU              |  79.88s                                          |  0.74s                       |  0.74  |
  | GPU              |  83.60s                                          |  40.00s                        | 0.71   |

* FP16
  
  
  | Type of Hardware | Total inference time              | Total load time | fps |
  |------------------|----------------------------------------------|----------------------------|------
  | CPU              |  79.40s                                          |  0.70s                       |  0.74  |
  | GPU              |  82.43s                                         |  37.58s                      |  0.72  |



* INT8
  
  
  | Type of Hardware | Total inference time              | Total load time | fps |
  |------------------|----------------------------------------------|----------------------------|------
  | CPU              |  79.64s                                          |  0.65s                       |  0.74  |
  | GPU              |  83.60s                                          |  39.35s                          |  0.71  |


## Results

Some of the observations from benchmark results: 
  * Model load time significantly increases when device is switch from CPU to GPU as expected. Model takes considerably longer to load in GPU as we analyzed from prior project study. 
  * Model load time is reduced when preicison is changed from FP32 to FP16 to INT8. This is because as weight are quantized and reduced in size then load time is also reduced. 
  * Model inference time is slighly improved when changing precision from FP32 to FP16 and from FP32 to INT8. This is expected as accuracy decrease performance improves. There is slight increase in inference time when changing precision from FP16 to INT8. 
  * Model inference time is increased when changing device from CPU to GPU. Here we can get better performance only when batch inferencing is done on GPU to leverage full compute capacity of GPU. 
  * Frames per second is pretty consistent irrespective of precision but it slightly reduces as device is changed from CPU to GPU. Here again fps can be increased on GPU device if batch inferencing is done.     

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
  * When performing inference using web cam feed as input, mouse controller crashed when pointer moved to corner of the screen. To overcome this problem I had to set pyautogui.FailSafe to false in MouseController class. This feature is enabled by default so that you can easily stop execution of your pyautogui program by manually moving the mouse to the upper left corner of the screen.  
