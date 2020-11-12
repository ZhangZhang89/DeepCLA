# DeepCLA
DeepCLA uses a deep learning apprach to predict clathrin, which is implemented by deep learning library Keras and Tensorflow backend. It allows users to run programs using specified protein sequences for clathrin prediction.

# Requirement
```
numpy>=1.14.5
backend==tensorflow
```
# Installation
* Install [python 3.6](https://www.python.org/downloads/) in Linux and Windows.
* Since the program is written in python 3.6, python 3.6 with pip tool must be installed first. DeepCLA uses the following dependencies: numpy, pandas, matplotlib, and scikit-learn. You can install these packages first, by the following commands:

```
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tensorflow
```
* If you want to run on a GPU, you will also need to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn), please refer to their websites for corresponding versions. You will need to uninstall the previous version of Tensorflow and install the corresponding Tensorflow-GPU version, the installation form as follows:
```
pip install tensorflow-gpu == 'your version'
```
* If you meet an error after operating improt tensorflow, the specific contents are as follows:

FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'._np_quint8 = np.dtype([("quint8", np.uint8, 1)]). That's probably because the version of numpy is too high and not compatible with Tensorflow, you need to uninstall numpy and reinstall the lower version.

# Running DeepCLA
open cmd in windows or terminal in Linux, then cd to the DeepCLA-master/Codes folder which contains predict.py
</br>**For clathrin prediction using our model, run:**
</br>`python predict.py  -input [custom predicting data in txt format]  -threshold [threshold value]  -output [predicting results in csv format]`  

</br>**Example:**
</br>`python predict.py -input ../Data/Example.txt -threshold 0.5 -output ../Data/Results.csv`
</br>-input and -threshold are required parameters, while -output is optional parameter. Prediction results will show in the cmd or terminal. If you don't want to save results, you need not input -output.
</br>**Example:**
</br>`python predict.py -input ../Data/Example.txt -threshold 0.5`

</br>**For details of -input,-threshold and -output, run:**
</br>`python predict.py -h`

# Announcements
* If the best_model fails to be unloaded after downloading, it can be downloaded separately and then unloaded into a folder containing predict.py.
* If you want to use the model to predict your test data, you must prepare the test data as a txt format. Users can refer to the Example.txt under the Codes folder. Also of note, each protein name should be added by '>', otherwise the program will occur error.
* To save the prediction results, the -output should be saved as a csv file.
