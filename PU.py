import os
import yaml
from PU.tools import set_random_seed
import torch
import warnings
from PU.tools import my_print, print1, print_data_info
from PU.matdeeplearn.train import train
from PU.matdeeplearn.process import datasets
import pandas as pd


def get_positive_unlabled(params):
    file_name = os.path.join(params['data_path'], params['targets_file_name'])
    data = pd.read_csv(file_name).iloc[:, 1]
    positive = data[data == 1].index.values.tolist()
    unlabeled = data[data == 0].index.values.tolist()
    positive = pd.Series(positive)
    unlabeled = pd.Series(unlabeled)
    return positive, unlabeled

def job_setup():
    warnings.filterwarnings('ignore')
    print1()
    my_print('Starting')
    # parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
   
    ##Open provided config file
    assert os.path.exists('./PU/config.yml'), "Config file was not found!"
    with open('./PU/config.yml', "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader) 
    
    my_print("Settings")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False, indent=4))
    # args_dict.update(vars(args))
    config['Processing']['run_mode'] = config['Global']["run_mode"]
    ##Update config values from command line
    # 注册更新configure
    # config = update_config(args, config)

    ##Print and write settings for job

    run_mode = config['Global']['run_mode']
    ################################################################################
    #  Begin processing
    ################################################################################

    # 设置随机种子方便重复实验
    set_random_seed(config["Global"]['seed'])

    ################################################################################
    #  Training begins
    ################################################################################

    # 读取数据
    my_print('dataset')
    dataset = datasets.CrystalDataset(config['Processing'])

    print_data_info(dataset)
    # 获得设备
    my_print('device')
    print("GPU is available:"+str(torch.cuda.is_available())+", Quantity: "+str(torch.cuda.device_count())+'\n')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    return run_mode, device, dataset, config

def main():
    # 任务初始化
    run_mode, device, dataset, config = job_setup()
    positive, unlabeled = get_positive_unlabled(config['Processing'])
    ##Regular training
    if run_mode == "Training":
        my_print("training")
        train.trainer(device, dataset, config["Training"], positive, unlabeled)
    ##Predicting from a trained model
    if run_mode == "Predicting":
        # 读取数据
        my_print("Predicting")
        train.predictor(device, dataset, config['Predicting'])
        

if __name__ == "__main__":
    main()
