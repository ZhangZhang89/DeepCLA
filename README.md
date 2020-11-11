# DeepCLA
DeepCLA uses a deep learning apprach to predict clathrin, allowing users to run programs using specified protein sequences.

# Installation
* Install [python 3.6](https://www.python.org/downloads/) in Linux and Windows.
* Since the program is written in python 3.6, python 3.6 with pip tool must be installed first. DeepCLA uses the following dependencies: numpy, pandas, matplotlib and scikit-learn. You can install these packages first, by the following commands:

```
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
```
* If you want to run on a GPU, you will also need to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn), please refer to their websites for corresponding versions.

# Running DeepCLA
open cmd in windows or terminal in Linux, then cd to the DeepCLA-master/Codes folder which contains predict.py
</br>**For clathrin prediction using our model, run:**
</br>`python predict.py  -input [custom predicting data in txt format]  -threshold [threshold value]  -output [predicting results in csv format]`  

</br>**Example:**
</br>`python predict.py -input ../Codes/Example.txt -threshold 0.5 -output ../Codes/Results.csv`
</br>-input and -threshold are required parameters, while -output is optional parameter. Prediction results will show in the cmd or terminal. If you don't want to save results, you need not input -output.
</br>**Example:**
</br>`python predict.py -input ../Codes/example.txt -threshold 0.5`

</br>**For details of -input,-threshold and -output, run:**
</br>`python predict.py -h`


* If you want to use the model to predict your test data, you must prepared the test data as a txt format. Users can refer to the example.txt under the Codes folder. Also of note, each protein name should be added by '>', otherwise the program will occur error.
* To save the prediction results, the -output should be saved as an csv file.
