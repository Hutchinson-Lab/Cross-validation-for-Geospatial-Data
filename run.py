import numpy as np
import pandas as pd
import argparse
from cv_base import run_regr, run_clf, run_sim
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alaska")
    args = parser.parse_args()
    
    if args.dataset == 'alaska':
        features = ['water','wetland','shrub','dshrub','dec','mixed','spruce','baresnow']
        y = 'GCSP'
        k = 10 #number of folds
        trfile ="./data/alaska/alaska_tr.csv"
        tefile = "./data/alaska/alaska_te.csv"
        bcvfile = "./bcv/alaska_b0.82.csv"
        outfile = './alaska.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1800_sd':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1800/HEWA1800_train_scenario134.csv"
        tefile = "./data/hewa1800/HEWA1800_test_scenario1.csv"
        bcvfile = "./bcv/hewa1800_s134_b0.3.csv"
        outfile = './hewa1800_sd.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1800_si':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1800/HEWA1800_train_scenario2.csv"
        tefile = "./data/hewa1800/HEWA1800_test_scenario2.csv"
        bcvfile = "./bcv/hewa1800_s2_b0.32.csv"
        outfile = './hewa1800_si.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1800_sdcs':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1800/HEWA1800_train_scenario134.csv"
        tefile = "./data/hewa1800/HEWA1800_test_scenario3.csv"
        bcvfile = "./bcv/hewa1800_s134_b0.3.csv"
        outfile = './hewa1800_sdcs.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1800_sics':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1800/HEWA1800_train_scenario134.csv"
        tefile = "./data/hewa1800/HEWA1800_test_scenario4.csv"
        bcvfile = "./bcv/hewa1800_s134_b0.3.csv"
        outfile = './hewa1800_sics.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1000_sd':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1000/HEWA1000_train_scenario134.csv"
        tefile = "./data/hewa1000/HEWA1000_test_scenario1.csv"
        bcvfile = "./bcv/hewa1000_s134_b0.28.csv"
        outfile = './hewa1000_sd.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1000_si':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1000/HEWA1000_train_scenario2.csv"
        tefile = "./data/hewa1000/HEWA1000_test_scenario2.csv"
        bcvfile = "./bcv/hewa1000_s2_b0.33.csv"
        outfile = './hewa1000_si.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1000_sdcs':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1000/HEWA1000_train_scenario134.csv"
        tefile = "./data/hewa1000/HEWA1000_test_scenario3.csv"
        bcvfile = "./bcv/hewa1000_s134_b0.28.csv"
        outfile = './hewa1000_sdcs.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'hewa1000_sics':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/hewa1000/HEWA1000_train_scenario134.csv"
        tefile = "./data/hewa1000/HEWA1000_test_scenario4.csv"
        bcvfile = "./bcv/hewa1000_s134_b0.28.csv"
        outfile = './hewa1000_sics.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'weta1800_sd':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/weta1800/WETA1800_train_scenario134.csv"
        tefile = "./data/weta1800/WETA1800_test_scenario1.csv"
        bcvfile = "./bcv/weta1800_s134_b0.27.csv"
        outfile = './weta1800_sd.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'weta1800_si':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/weta1800/WETA1800_train_scenario2.csv"
        tefile = "./data/weta1800/WETA1800_test_scenario2.csv"
        bcvfile = "./bcv/weta1800_s2_b0.28.csv"
        outfile = './weta1800_si.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'weta1800_sdcs':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/weta1800/WETA1800_train_scenario134.csv"
        tefile = "./data/weta1800/WETA1800_test_scenario3.csv"
        bcvfile = "./bcv/weta1800_s134_b0.27.csv"
        outfile = './weta1800_sdcs.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'weta1800_sics':
        features = ['summer_nbr_TCA_mean_600','summer_nbr_TCB_mean_600','summer_nbr_TCG_mean_600','summer_nbr_TCW_mean_600']
        y = 'Occur'
        k = 9
        trfile ="./data/weta1800/WETA1800_train_scenario134.csv"
        tefile = "./data/weta1800/WETA1800_test_scenario4.csv"
        bcvfile = "./bcv/weta1800_s134_b0.27.csv"
        outfile = './weta1800_sics.csv'
        run_clf(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'house_bay':
        features = ['housing_median_age','median_income','total_rooms','total_bedrooms','population','households']
        y = 'median_house_value'
        k = 10
        trfile ="./data/housing/house_bay_tr.csv"
        tefile = "./data/housing/house_bay_te.csv"
        bcvfile = "./bcv/house_bay_b0.25.csv"
        outfile = './house_bay.csv'
        run_regr(features, y, k, trfile, tefile, bcvfile, outfile)
        
    elif args.dataset == 'house_latitude':
        features = ['housing_median_age','median_income','total_rooms','total_bedrooms','population','households']
        y = 'median_house_value'
        k = 10
        trfile ="./data/housing/house_latitude_tr.csv"
        tefile = "./data/housing/house_latitude_te.csv"
        bcvfile = "./bcv/house_latitude_b0.23.csv"
        outfile = './house_latitude.csv'
        run_regr(features, y, k, trfile, tefile, bcvfile, outfile)
    
    # set block size = 20 grid for blcv, bfcv and ibcv as follows
    elif args.dataset == 'sim_sd':
        bcvfile = "bcv/sim_b20/sac%d/bcv_c1_%d.csv"
        outfile = "sac%d_sd.csv"
        run_sim('sd',bcvfile,outfile)
        
    elif args.dataset == 'sim_si':
        bcvfile = "bcv/sim_b20/sac%d/bcv_c1_%d.csv"
        outfile = "sac%d_si.csv"
        run_sim('si',bcvfile,outfile)
        
    elif args.dataset == 'sim_sdcs':
        bcvfile = "bcv/sim_b20/sac%d/bcv_c3_%d.csv"
        outfile = "sac%d_sdcs.csv"
        run_sim('sdcs',bcvfile,outfile)
        
    elif args.dataset == 'sim_sics':
        bcvfile = "bcv/sim_b20/sac%d/bcv_c1_%d.csv"
        outfile = "sac%d_sics.csv"
        run_sim('sics',bcvfile,outfile)
        
    elif args.dataset == 'sim_sirs':
        bcvfile = "bcv/sim_b20/sac%d/bcv_c1_%d.csv"
        outfile = "sac%d_sirs.csv"
        run_sim('sirs',bcvfile,outfile)
        
    elif args.dataset == 'sim_sipcs':
        bcvfile = "bcv/sim_b20/sac%d/bcv_c1_%d.csv"
        outfile = "sac%d_sipcs.csv"
        run_sim('sipcs',bcvfile,outfile)
        
    else:
        print("Dataset is not provided!")
