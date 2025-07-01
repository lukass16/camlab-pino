import itertools
import subprocess
import json
import os
import sys

def run_grid_search():
    """
    Grid search over learning rates, boundary weights, and lambda values
    for both CNO and FNO architectures on the Helmholtz equation.
    """
    
    # Define parameter grids
    learning_rates = [1e-4, 5e-4, 1e-5]
    boundary_weights = [1, 10, 100]
    lambdas = [100, 1000, 10000]
    
    # Base training properties for CNO
    base_cno_training_properties = {
        "weight_decay": 1e-10,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 100,
        "batch_size": 16,
        "exp": 1,
        "training_samples": 1024,
        "pad_factor": 0,
        "patience": 1.0
    }
    
    # Base model architecture for CNO
    base_cno_architecture = {
        "N_layers": 4,
        "channel_multiplier": 16,
        "N_res": 5,
        "N_res_neck": 5,
        "in_size": 128,
        "retrain": 4,
        "kernel_size": 3,
        "FourierF": 0,
        "activation": 'cno_lrelu',
        "cutoff_den": 2.0001,
        "lrelu_upsampling": 2,
        "half_width_mult": 0.8,
        "filter_size": 6,
        "radial_filter": 0,
    }
    
    # Base training properties for FNO
    base_fno_training_properties = {
        "weight_decay": 1e-10,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 30,
        "batch_size": 16,
        "exp": 3,
        "training_samples": 1024,
        "pad_factor": 0,
        "patience": 1.0,
        "gradient_clip_value": 5
    }
    
    # Base architecture for FNO
    base_fno_architecture = {
        "width": 128,
        "modes": 16,
        "FourierF": 0,
        "n_layers": 4,
        "padding": 0,
        "include_grid": 1,
        "retrain": 4,
    }
    
    # Info for pretrained network (None for training from scratch)
    info_pretrained_cno = {
        "Path to pretrained model": "TrainedModels/helmholtz/CNO_1024helmholtz",
        "Pretrained Samples": 1024,
    }
    
    info_pretrained_fno = {
        "Path to pretrained model": "TrainedModels/helmholtz/FNO_1024helmholtz",
        "Pretrained Samples": 1024,
    }
    
    which_example = "helmholtz"
    
    # Create base directory
    base_dir = "TrainedModels/helmholtz_gridsearch"
    os.makedirs(base_dir, exist_ok=True)
    
    # Total number of experiments
    total_experiments = len(learning_rates) * len(boundary_weights) * len(lambdas) * 2  # 2 for CNO and FNO
    current_experiment = 0
    
    print(f"Starting grid search with {total_experiments} total experiments...")
    print(f"Learning rates: {learning_rates}")
    print(f"Boundary weights: {boundary_weights}")
    print(f"Lambda values: {lambdas}")
    print("-" * 80)
    
    # Run grid search
    for lr, boundary_weight, lambda_val in itertools.product(learning_rates, boundary_weights, lambdas):
        
        # CNO experiment
        current_experiment += 1
        print(f"[{current_experiment}/{total_experiments}] Running CNO: lr={lr}, boundary_weight={boundary_weight}, lambda={lambda_val}")
        
        cno_training_props = base_cno_training_properties.copy()
        cno_training_props["learning_rate"] = lr
        cno_training_props["boundary_weight"] = boundary_weight
        cno_training_props["lambda"] = lambda_val
        
        cno_folder = f"{base_dir}/CNO_lr_{lr}_bw_{boundary_weight}_lambda_{lambda_val}"
        
        cmd_cno = f"python TrainPINO_CNO.py {cno_folder} '{json.dumps(cno_training_props)}' '{json.dumps(base_cno_architecture)}' '{json.dumps(info_pretrained_cno)}' {which_example}"
        
        try:
            result = subprocess.run(cmd_cno, shell=True, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            if result.returncode != 0:
                print(f"CNO experiment failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
            else:
                print(f"CNO experiment completed successfully")
        except subprocess.TimeoutExpired:
            print(f"CNO experiment timed out after 1 hour")
        except Exception as e:
            print(f"CNO experiment failed with exception: {e}")
        
        # FNO experiment
        current_experiment += 1
        print(f"[{current_experiment}/{total_experiments}] Running FNO: lr={lr}, boundary_weight={boundary_weight}, lambda={lambda_val}")
        
        fno_training_props = base_fno_training_properties.copy()
        fno_training_props["learning_rate"] = lr
        fno_training_props["boundary_weight"] = boundary_weight
        fno_training_props["lambda"] = lambda_val
        
        fno_folder = f"{base_dir}/FNO_lr_{lr}_bw_{boundary_weight}_lambda_{lambda_val}"
        
        cmd_fno = f"python TrainPINO_FNO.py {fno_folder} '{json.dumps(fno_training_props)}' '{json.dumps(base_fno_architecture)}' '{json.dumps(info_pretrained_fno)}' {which_example}"
        
        try:
            result = subprocess.run(cmd_fno, shell=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout for FNO (fewer epochs)
            if result.returncode != 0:
                print(f"FNO experiment failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
            else:
                print(f"FNO experiment completed successfully")
        except subprocess.TimeoutExpired:
            print(f"FNO experiment timed out after 30 minutes")
        except Exception as e:
            print(f"FNO experiment failed with exception: {e}")
        
        print("-" * 40)
    
    print("Grid search completed!")
    print(f"Results saved in: {base_dir}")

def analyze_results():
    """
    Analyze the results from the grid search and create a summary.
    """
    import pandas as pd
    import glob
    
    base_dir = "TrainedModels/helmholtz_gridsearch"
    
    results = []
    
    # Find all experiment folders
    experiment_folders = glob.glob(f"{base_dir}/*/")
    
    for folder in experiment_folders:
        folder_name = os.path.basename(folder.rstrip('/'))
        
        # Parse folder name to extract parameters
        if folder_name.startswith('CNO_'):
            model_type = 'CNO'
            params_str = folder_name[4:]  # Remove 'CNO_'
        elif folder_name.startswith('FNO_'):
            model_type = 'FNO'
            params_str = folder_name[4:]  # Remove 'FNO_'
        else:
            continue
        
        # Parse parameters from folder name
        try:
            parts = params_str.split('_')
            lr = float(parts[1])
            bw = int(parts[3])
            lambda_val = int(parts[5])
        except:
            print(f"Could not parse folder name: {folder_name}")
            continue
        
        # Read the best testing error if available
        errors_file = os.path.join(folder, 'errors.txt')
        best_error = None
        if os.path.exists(errors_file):
            try:
                with open(errors_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'Best Testing Error:' in line:
                            best_error = float(line.split(':')[1].strip())
                            break
            except:
                pass
        
        results.append({
            'model_type': model_type,
            'learning_rate': lr,
            'boundary_weight': bw,
            'lambda': lambda_val,
            'best_testing_error': best_error,
            'folder': folder_name
        })
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['model_type', 'best_testing_error'])
        summary_file = f"{base_dir}/grid_search_summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"Results summary saved to: {summary_file}")
        
        # Print top 5 results for each model type
        print("\nTop 5 CNO results:")
        cno_results = df[df['model_type'] == 'CNO'].dropna(subset=['best_testing_error']).head(5)
        print(cno_results[['learning_rate', 'boundary_weight', 'lambda', 'best_testing_error']])
        
        print("\nTop 5 FNO results:")
        fno_results = df[df['model_type'] == 'FNO'].dropna(subset=['best_testing_error']).head(5)
        print(fno_results[['learning_rate', 'boundary_weight', 'lambda', 'best_testing_error']])
    else:
        print("No results found to analyze.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_results()
    else:
        run_grid_search() 