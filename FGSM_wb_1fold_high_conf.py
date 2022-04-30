from arml.exp import exp_fgsm_impact_wb_1fold_high_conf

file_path = 'data/RML2016.10a_dict.pkl'
# number of cross validation runs 
n_runs = 5
# verbose ? 
verbose = 1
# # type of experiment 
# scenario = 'A'
# attack epsilons 
epsilons = [0.00025, 0.0005, 0.001, 0.0025,  0.005]
# epsilons = [0.00025, 0.0005, 0.001]

# For white box attack the adversarial mode are assume to be the
# same as the defenders model
# defenders model 
train_params = {'type': 'vtcnn2', 
                'dropout': 0.5, 
                'val_split': 0.9, 
                'batch_size': 1024, 
                'nb_epoch': 50, 
                'verbose': verbose, 
                'NHWC': [220000, 2, 128, 1],
                'tpu': False, 
                'file_path': 'FGSM_CNN2_1fold_high_conf.wts.h5'}
# name for the logger     
logger_name_1 = 'vtcnn2_FGSM_wb_1fold_high_conf_o'
logger_name_2 = 'vtcnn2_FGSM_wb_1fold_high_conf_w'
# output path
output_path_1 = 'outputs/vtcnn2_FGSM_wb_1fold_high_conf_o_op.pkl'  
output_path_2 = 'outputs/vtcnn2_FGSM_wb_1fold_high_conf_w_op.pkl'  

exp_fgsm_impact_wb_1fold_high_conf(file_path=file_path,
                n_runs=n_runs, 
                name="FGSM_CNN2_5fold.wts.h5",
                verbose=verbose, 
                # scenario=scenario,
                epsilons=epsilons, 
                train_params=train_params, 
                # train_adversary_params=train_adversary_params, 
                logger_name_1 = logger_name_1,
                output_path_1 = output_path_1,
                logger_name_2 = logger_name_2,
                output_path_2 = output_path_2,               
                )      