# AVDNet

This is the code base for training and optimizing the ADVNet architecture

The entry point to the codebase is optimization.py, running an optuna study to find the optimal hyperparameters for your own audio problem.  
The architecture is stored at Architectures/AVDNet.py  

To create and run a model:  


# Dataset structure

The data should be stored in a file structure as described below. (The path to the dataset need to be updated at constants.py)
To fit another database structure you can provide your own torch.utils.data.Dataset provided in data_methods.py  

Dataset_folder file structure  
Real/  
├── source_1/  
│   ├── real_1.wav  
│   ├── real_2.wav  
│   ├── real_3.wav  
├── source_2/  
│   ├── real_1.wav  
│   ├── real_2.wav  
│   ├── real_3.wav  
├── source_x/  
│   ├── real_1.wav  
│   ├── real_2.wav  
│   ├── real_3.wav  

Fake/  
├── source_1/  
│   ├── fake_1.wav  
│   ├── fake_2.wav  
│   ├── fake_3.wav  
├── source_2/  
│   ├── fake_1.wav  
│   ├── fake_2.wav  
│   ├── fake_3.wav  
├── source_x/  
│   ├── fake_1.wav  
│   ├── fake_2.wav  
│   ├── fake_3.wav  


# Constants - constants.py

Make sure to modify the constants.py following fields as needed:  
-DATASET_FOLDERS (path to the datset folder)  
-LOAD_TRAINING = True (loading or not from an existing optuna study)  
-EPOCHS = 100 (maximum numbers of epochs per trial)  
-TRIALS = 10 (number of trials)  
-PATIENCE = 4 (number of epochs before exiting the current trial if loss is not improved)  
-PARTIAL_TRAINING = 1 (the ratio of the data to train on, between 0-1)  
-DEBUGMODE = False (for debug print)  
-BATCH_SIZE = 16 (batch size for training)  
-DROP_OUT = 0.3 (drop out rate)  
