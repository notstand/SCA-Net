## SCA-Net

Current models for emotion recognition with hyper complex multi-5
modal signals face limitations due to fusion methods and insufficient attention mechanisms,6
preventing further enhancement in classification performance.To address these challenges, we7
propose a new model framework named Signal Channel Attention Network(SCA-Net), which8
comprises three main components: an encoder, an attention fusion module, and a decoder. 

### How to use :scream:

#### Install requirements

`pip install -r requirements.txt`

#### Data preprocessing

1) Download the data from the [official website](https://mahnob-db.eu/hci-tagging/).
2) Preprocess the data: `python data/preprocessing.py`
   - This will create a folder for each subject with CSV files containing the preprocessed data and save everything inside `args.save_path`.
4) Create torch files with augmented and split data: `python data/create_dataset.py`

#### Training

- please import the attention module you are about to use in the "modals/modal.py" and Modify of part of the code
- To train with specific hyperparameters run: `python main.py`
