The following repository contains the replication package to train an LSTM model for next word generation conditioned on literary genres.  
### Files and Folder Structure
    - `data`: contains training, test and validation data in form of pickle files. It also contains vocab dictionary and the genres mapping. They can be downloaded from the following Google Drive folder: https://drive.google.com/drive/folders/1SaBaq1KmAvj9K-4a9vvsdxzp8iOniRYL?usp=sharing
    - `logs`: Contains Tensorboard logs from my experiment.
    - `model2`: This package contains the model implementation, data classes and utility functions for I/O operations and training.
    - `Data_Preparation.ipynb`: Jupyter notebook used to generate the data provided in the Google Drive folder.
    - `train_attn.py` Main script that contains the logging, hyperparemeters values, and training/testing loops.


### Training
(Option) First step is to generate the data files. This an optional step as training files are already provided.  
To train the model, execute the the following:  
```
python train_attn.py
```