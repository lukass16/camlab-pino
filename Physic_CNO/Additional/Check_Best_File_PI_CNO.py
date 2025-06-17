import os

def Find_text(file_to_check,Print_all_errors):
    if os.path.exists(file_to_check):
        with open(file_to_check, 'r') as file:
                for line in file:
                    if line.startswith("Best Testing Error: "):
                        best_error_file = float(line.split(": ")[1])
                        if Print_all_errors:
                            print( best_error_file )
                        break
        return best_error_file
    else:
        print('No best Model')
        return 1e20
    
def check_best(target_folder,samples,Print_all_errors=False,training_properties_print=True):
    target_file='errors.txt'
    best_error=1e20
    for name in os.listdir(target_folder):
        if name.startswith(str(samples)+'S'):
          file_to_check=os.path.join(target_folder,name,target_file)
          
          best_error_file=Find_text(file_to_check,Print_all_errors)
                    
          if best_error_file<best_error:
             best_error= best_error_file
             best_model=os.path.join(target_folder,name)
          print(file_to_check)

    if training_properties_print:
        with open(os.path.join(best_model,'training_properties.txt'), 'r') as file:
            contents = file.read()
            
            lines = contents.split('\n')
            for line in lines:
                print(line)
    print(f'Best error: {best_error}') 
    print(f'Best file:  {best_model}')          
    return best_error, best_model

Path_to_scratch='/cluster/scratch/harno'
target_folder='MODEL_SELECTION_CNO_and_PhysicCNO_poisson'

Path=os.path.join(Path_to_scratch,target_folder)

samples=128
copy=False
best_error, best_model=check_best(Path,samples,Print_all_errors=True)

folder_name=os.path.join(Path_to_scratch,'Best_Model_CNO_and_Physics-Informed_CNO')

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

if copy:
   Old_folder=os.path.join(folder_name,f'Best_CNO_and_PI_CNO_{samples}')
   if os.path.isdir(Old_folder):
        print(Old_folder)
        os.system(f'rm -r {Old_folder}')
   
   os.system(f'cp -r {best_model}  {Old_folder}')
   print(f'New Best Model for {samples} samples')
