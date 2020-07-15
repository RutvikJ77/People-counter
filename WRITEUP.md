# People Counter application

A people counter app is optimized to work with very little latency on the edge. The model is optimized via Intel [Open Vino toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html). This repository aims to showcase the potential of leveraging AI at the edge.

## Explaining Custom Layers

The workflow of the Intel's Optimized layer is as follows: 
- The model optimizer searches for the list of known layers in 
each layer for the given model. 
- The inference engine then loads the  
layers from the input model Intermediate Representation files into the specified device plugin, 
which will then search a list of known layer implementations for the given device. 
 -- If your model architecure contains a layer/layers that is not in the 
list of known layers for the device, the Inference engiene considers it to be unsupported. 
Supported layers list [Supported Devices documentation](https://docs.openvinotoolkit.org/2019_R1.1/_docs_IE_DG_supported_plugins_Supported_Devices.html).

### Custom Layers implementation

When implementing a custom layer for your pre-trained model in the
Open Vino toolkit, you will need to add extensions to both the Model 
Optimizer and the Inference Engine.

Model Optimizer - cross platform command line tool used for optimizing the layers using statistical analysis for the end user specified device.

Inference Engine - The core heart of the deep learning performance.

### Reasons for handling custom layers

For smooth transition among teams it is expected that you must be aware regarding the custom layers so that if one team is facing an issue or wants to hop in on a new project you can easily provide sustainable base for them.

`lambda` layers are another reason for using custom layers to solve various quick experimental code pieces.

## Comparing Model Performance

Person detection model from OpenVino Model zoo was the best in terms of performance and accuracy when compared to the rest. Being heavily focused on the accuracy and inference time I choose to go for openvino Model.

### Model size

| |SSD MobileNet V2|SSD Inception V2|SSD Coco MobileNet V1|
|-|-|-|-|
|Before Conversion|67 MB|98 MB|28 MB|
|After Conversion|65 MB|96 MB|26 MB|

### Inference Time

| |SSD MobileNet V2|SSD Inception V2|SSD Coco MobileNet V1|
|-|-|-|-|
|Before Conversion|50 ms|150 ms|55 ms|
|After Conversion|60 ms|155 ms|60 ms|

## Assess Model Use Cases

I beleive a similar application could be widely useful, here are a few of which I feel have a great impact

1. At retail stores

At retail stores there are variety of people who come to shop and analysing the optimal situation by

* Compare movement across different areas
* Calculating people movement
* Optimize product placing strategies
* Optimize staff placement

2. Space management

Being in a crowded city we know how much people crave for personal space therefore a probable solution is space management

* Know the most crowded places.
* Knowing when stores generate the most traffic might tell new sale opurtunities.

3. At Airports

Airports are one of the busiest places and the most with thriving data on a daily basis.

* Identify which shops generate more customer attention.
* Knowing an approx. position for optimal opening of shop
* Managing queues.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

* In dimly light situations the model tends to perform with weak accuracy. Solution to use this would be a better piece of hardware in these conditions.

* The natural descent of model's accuracy during conversion can lead to an irreparable loss, therefore to avoid this we can perform federated learning or threading simultaneously to achieve the accuracy required.

* Distorted images through input cameras can lead to an unknowing decline of the accuracy to avoid this data augmentation is must and should be performed.

## Model Research

While investigating the potential people counter models, I tried the following three models:

- Model 1: SSD Mobilenet v2
  - [Model Source](https://github.com/chuanqi305/MobileNet-SSD)
  - Converting the model to an Intermediate Representation :
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```

  - I tried to improve the model by performing transfer learning techniques, but in vain.
  
- Model 2: SSD Inception V2]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - Converting the model to an Intermediate Representation :
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```
  
  - The model was insufficient for the app since it had a pretty high latency in making predictions ~155 ms which is nowhere near to the model I currently use ~40 ms. It made accurate predictions though due to a very huge tradeoff in inference time, I neglected it.
  - I tried to improve the model by reducing precision of weights, however this had a negative impact on accuracy.

- Model 3: SSD Coco MobileNet V1
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - Converting the model to an Intermediate Representation :

```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```

  - Pretty inefficient model, generally ignored people with their backs on.

## The Model and learning

Facing these issues I particularly went for OpenVino Model Zoo.
 
- [person-detection-retail-0002](https://docs.openvinotoolkit.org/latest/person-detection-retail-0002.html)
- [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

These models are based on the MobileNet model. The MobileNet model performed well, considering latency and size apart from the few inference errors.

In general, [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) had a higher overall accuracy.

Make sure you check the case of false positives where the object is specified as correct which in case is not. Probable solutions for avoiding them is to set a threshold for the frames, another one would be to change the model in use or perform transfer learning for improving the model accuracy.

### Steps for Downloading model

Download all the pre-requisite libraries and source the openvino installation using the following commands:

```sh
pip install requests pyyaml -t /usr/local/lib/python3.5/dist-packages && clear && 
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

Navigate to the directory containing the Model Downloader:

```sh
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
```

Within there, you'll notice a `downloader.py` file, and can use the `-h` argument with it to see available arguments, `--name` for model name, and `--precisions`, used when only certain precisions are desired, are the important arguments. Use the following command to download the model

```sh
sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace
```

## Performing Inference

Open a new terminal

Execute the following commands:

```sh
  cd webservice/server
  npm install
```
After installation, run:

```sh
  cd node-server
  node ./server.js
```

If succesful you should receive a message-

```sh
Mosca Server Started.
```

Open another terminal

These commands will compile the UI for you

```sh
cd webservice/ui
npm install
```

After installation, run:

```sh
npm run dev
```

If succesful you should receive a message-

```sh
webpack: Compiled successfully
```

Open another terminal and run:

This will set up the `ffmpeg` for you

```sh
sudo ffserver -f ./ffmpeg/server.conf
```

Finally execute the following command in another terminal

This peice of code specifies the testing video provided in `resources/` folder and run it on port `3004`

```sh
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```
