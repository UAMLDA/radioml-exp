from arml.exp import exp_fgsm_impact_wb_1fold_shift10

file_path = 'data/RML2016.10a_dict.pkl'
# number of cross validation runs 
n_runs = 5
# verbose ? 
verbose = 1
# # type of experiment 
# scenario = 'A'
# attack epsilons 
epsilons = [0.00025, 0.0005, 0.001, 0.0025,  0.005]
# epsilons = [0.00025, 0.0005]

# For white box attack the adversarial mode are assume to be the
# same as the defenders model
# defenders model 
train_params = {'type': 'vtcnn2', 
                'dropout': 0.5, 
                'val_split': 0.9, 
                'batch_size': 1024, 
                'nb_epoch': 35, 
                'verbose': verbose, 
                'NHWC': [220000, 2, 128, 1],
                'tpu': False, 
                'file_path': 'FGSM_CNN2_1fold_shift10.wts.h5'}
# name for the logger     
logger_name = 'vtcnn2_FGSM_wb_1fold_shift10'
# output path
output_path = 'outputs/vtcnn2_FGSM_wb_1fold_shift10_op.pkl'  
exp_fgsm_impact_wb_1fold_shift10(file_path=file_path,
                n_runs=n_runs, 
                verbose=verbose, 
                # scenario=scenario,
                epsilons=epsilons, 
                train_params=train_params, 
                # train_adversary_params=train_adversary_params, 
                logger_name = logger_name,
                output_path = output_path)      