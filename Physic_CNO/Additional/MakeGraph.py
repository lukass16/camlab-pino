import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from helper_functions.helper_for_plots import Find_text, make_graph

which_example='poisson'
print(f'Checking {which_example} equation')
Path_to_Best_Physic_CNO='/cluster/scratch/harno/Best_Model_Physic_CNO'
Path_to_Best_CNO='/cluster/scratch/harno/Best_Models_CNO'

PreSamplelist=[1,2,4,8,16,32,64,128,256,512,1024]
PhysicSamplelist=[16,32,64,128,256,512,1024]

Folder_name=f'Best_CNO_{which_example}'

target_file='errors.txt'
trainings_file='training_properties.txt'

#Get Best Testing errors for the corresponding samples
Graph={}
for Pretrainedsampels in PreSamplelist:
   Path=os.path.join(Path_to_Best_Physic_CNO,Folder_name+'_'+str(Pretrainedsampels))
   Path_Pretrained=os.path.join(Path_to_Best_CNO,Folder_name+'_'+str(Pretrainedsampels))
   Graph[str(Pretrainedsampels),'0']=Find_text(os.path.join(Path_Pretrained,target_file),"Best Testing Error: ")
   
   for item in PhysicSamplelist:
          Path_from_scratch=os.path.join(Path_to_Best_Physic_CNO,Folder_name+'_'+str(0)+'_'+str(item))
          Graph['0',str(item)]=Find_text(os.path.join(Path_from_scratch,target_file),"Best Testing Error: ")
          error_file=os.path.join(Path+'_'+str(item),target_file)
          train_file=os.path.join(Path+'_'+str(item),trainings_file)
         
          best_error_file=Find_text(error_file,"Best Testing Error: ")
          samples=int(Find_text(train_file,"training_samples,"))

          assert (samples==item)
      
          Graph[str(Pretrainedsampels),str(int(samples))]=best_error_file
       

#Create Matrix       
table_data = np.zeros((12,8))
for i, Pretrainedsampels in enumerate(PreSamplelist):
    table_data[i+1,0]=Graph[str(Pretrainedsampels),str(0)]
    for j,item in enumerate(PhysicSamplelist):
       table_data[i+1,j+1]=Graph[str(Pretrainedsampels),str(item)]
       table_data[0,j+1]=Graph[str(0),str(item)]
table_data[0,0]=None


#Print Testing errors
Samplelist=[0,1,2,4,8,16,32,64,128,256,512,1024]
Samplelist2=[0,16,32,64,128,256,512,1024]
df = pd.DataFrame(table_data, columns=Samplelist2, index=Samplelist)
print(df)

#2D plot of errors
k, l=0, 0 
Samplelist=Samplelist[k:]
Samplelist2=Samplelist2[l:]
plt.figure(figsize=(8, 8))
plt.imshow(table_data[k:,l:], cmap="hsv", norm=LogNorm())
cbar=plt.colorbar(shrink=0.8)
cbar.set_label('Testing error in log scale')
plt.xlabel('Training Samples for CNO')
plt.ylabel('Training Samples for Physics-Informed CNO')
plt.xticks(np.arange(len(Samplelist2)), Samplelist2, rotation=45)
plt.yticks(np.arange(len(Samplelist)), Samplelist)
plt.show()
plt.savefig('Plots/TestingError_DifferentSamples.png')

#make_graph(table_data,'Plots/Graphs.png')

#1 D Plot errors
l,k=1,4
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,8))
labels = Samplelist2
ax[0].axhline(y=1, color='red', linestyle='--', label='1%')
ax[1].axhline(y=1, color='red', linestyle='--', label='1%')
for i in range(l,table_data[:,0].shape[0]-k):
    ax[0].plot(np.arange(len(Samplelist2)),table_data[i,:],label=Samplelist[i]) 
ax[0].set_xticks(np.arange(len(labels)))
ax[0].set_xticklabels(labels)
ax[0].set_xlabel('Number of Training Samples for Physics-Informed CNO')
ax[0].set_ylabel('Relative Testing Error')
ax[0].legend(title='Number of Training Samples\n for pre-training')

ax[1].set_xticks(np.arange(len(labels)))
ax[1].set_xticklabels(labels)
ax[1].plot(np.arange(len(Samplelist2)),table_data[0,:],label='No Pretraining') 
ax[1].set_xlabel('Number of Training Samples for Physics-Informed CNO')
ax[1].set_ylabel('Relative Testing Error')
ax[1].legend(title='Number of Training Samples\n for pre-training')

plt.tight_layout()

plt.savefig('Plots/plot.png')










