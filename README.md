# NeuroScope
This is the code repo of our paper "NeuroScope: Reverse Engineering Deep Neural Network on Edge Devices using Dynamic Analysis". 



### Environment  
1. Create a Python virtual environment
2. `pip install -r ./requirements.txt `

### Usage

#### Online part (Dynamic analysis part)
NXP i.MX RT1050 Evaluation Kit is required for conducting dynamic analysis. 
1. Use MCUXpresso, NXP's IDE, to download (flash) the [binary that implements resnet](https://github.com/purseclab/NeuroScope/blob/0d98310860a248df4cd73fad144a7042249b56b5/online/bin/evkbimxrt1050_tensorflow_lite_micro_cifar10_deployment.axf) onto the board
2. Launch debug session with MCUXpresso IDE LinkServer (CMSIS-DAP) probe
3. In the debug console, use the following command to use our gdb script: `source $YOUR_PATH/online/io_dumper/io_dumper.py`, and then invoke `io-dump`

Our script will dump the I/O data to a folder: `$YOUR_PATH/online/dump`. 
These data will be processed in the offline part for DNN recovery. 


#### Offline part (Dataset synthesis, Model training, Model recovery)
##### Dataset synthesis 
Go into the folder `offline`, and invoke `python dataset_synthesizer.py`. 
You can configure the number of datapoints (i.e., size of the synthesized dataset) and the path where the synthesized dataset will be generated with global variables `SIZE` and `PATH`, respectively. 

##### Model training 
Go into the folder `offline`, and invoke `python train.py`. 
The trained models will be placed in a folder `./offline/saved_models`. 

##### Model recovery 
Go into the folder `offline`, and invoke `python recover.py`. 
It would infer the network information from the dumped I/O data with the trained models, and generate a DNN model description file in ONNX format (as `recovered_model.onnx`). 
