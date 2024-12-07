# Fewshot-VDT
Interactive Fusion and Correlation Network for Three-Modal Images Few-Shot Semantic Segmentation. As for the details, please refer to the paper:
- The paper has been accepted by IEEE SPL, you can refer to this [link](https://ieeexplore.ieee.org/document/10669915). 

## Usage
We have provided the prediction results of our network in this repository.

### Code
The code for 1 shot model is in the folder `./1-shot` and the code for 5 shot model is in the folder `./5shot`. You can find the structure of our network in `./1-shot/model/mymodel.py` and `./5-shot/model/mymodel.py`.

### usage
1. Download the dataset to `./VDT-2048-5i`.
2. Prepare the environment.
3. Run the code:
   - training: 1-shot `python./1-shot/train.py`; 5-shot `python./5-shot/train.py`.
   - testing: 1-shot `python./1-shot/test.py`; 5-shot `python./5-shot/test.py`.

### Dataset
The dataset can be downloaded from the link below:
-  [VDT-2048-5i](https://github.com/VDT-2048/SEMA)