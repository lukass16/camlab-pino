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

def check_best(Path,presamples,samples,Check_all=False,training_properties_print=False,model_print=False,which_example='poisson'):
    target_file='errors.txt'
    Print_all_errors=True
    file_best=f'/cluster/scratch/harno/Best_Model_Physic_CNO/Best_CNO_{which_example}_'+str(presamples)+'_'+str(samples)
    current_best_error=Find_text(os.path.join(file_best,target_file),Print_all_errors=False)
    best_error=1e20
    for name in os.listdir(Path):
        if name.startswith(str(samples)+'S') or Check_all:
          file_to_check=os.path.join(Path,name,target_file)
          
          best_error_file=Find_text(file_to_check,Print_all_errors)
                    
          if best_error_file<best_error:
             best_error= best_error_file
             best_model=os.path.join(Path,name)
          print(file_to_check)
    if best_error>current_best_error:
       print('Old still the best')
       best_error=current_best_error
       best_model=file_best

    
    if training_properties_print:
        with open(os.path.join(best_model,'training_properties.txt'), 'r') as file:
            contents = file.read()
            
            lines = contents.split('\n')
            for line in lines:
                print(line)
    
    if model_print:
        with open(os.path.join(best_model,'net_architecture.txt'), 'r') as file:
            contents = file.read()
            
            lines = contents.split('\n')
            for line in lines:
                print(line)

    print(f'Best error: {best_error}')
    
    return f'scp -r harno@euler.ethz.ch:{best_model} Best_CNO_{which_example}_{presamples}_{samples}'


presamples=0
which_example ='poisson'
#Define path
Path_CNO=f'/cluster/scratch/harno/MODEL_SELECTION_CNO_{which_example}_'+str(presamples)

#Define path for physic informed

#samples=1024
Check_all=False
make_file=False
write_new=False
samples=1024
Path_Physic_CNO=f'/cluster/scratch/harno/MODEL_SELECTION_PhysicCNO_{which_example}_'+str(presamples)
line=check_best(Path_Physic_CNO,presamples,samples,training_properties_print=True,Check_all=False,model_print=False)
print(line)

if write_new:
   new_best=f'/cluster/scratch/harno/Best_Model_Physic_CNO/Best_CNO_{which_example}_{str(presamples)}_{str(samples)}'
   line=line.split()[2]
   best_model=line.split(':')[1]
   print(best_model)
   if best_model==new_best:
      print('Old model best model')
   else:
      print(f'/cluster/scratch/harno/Best_Model_Physic_CNO/Best_CNO_{which_example}_{str(presamples)}_{str(samples)}')
      os.system(f'rm -r /cluster/scratch/harno/Best_Model_Physic_CNO/Best_CNO_{which_example}_{str(presamples)}_{str(samples)}')
      os.system(f'cp -r {best_model}  {new_best}')
      print('Made new best')

if make_file:
  text_file='scp.txt'
  with open(text_file, "w") as file:
       for i in range(0,6):
          presamples=2**i
          #presamples=0
          for j in range(4,11):
              samples=2**j
              Path_Physic_CNO=f'/cluster/scratch/harno/MODEL_SELECTION_PhysicCNO_{which_example}_'+str(presamples)
              line=check_best(Path_Physic_CNO,presamples,samples,training_properties_print=False,which_example=which_example)
              file.write(line+'\n')