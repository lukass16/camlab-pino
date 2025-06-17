import os
import re
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def Find_text(file_path,text):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
              for line in file:
                  if line.startswith(text):
                      ref = float(re.split(': |,|\n',line)[1])
                      break 
        return ref
    else:
        raise FileExistsError(f'file {file_path} doesnt exist')
    
def plot_1(error_list,Samplelist,namefigsave,xname,yname='Testing Error',title=None):
    plt.figure()
    plt.plot(np.arange(len(Samplelist)),error_list)
    plt.xticks(np.arange(len(Samplelist)),labels=Samplelist)
    plt.xlabel(xname)
    plt.ylabel(yname)
    if title is not None:
        plt.title(title)
    plt.savefig(namefigsave)



def make_graph(data,name,M=4,Not_show=False):
    '''data : matrix of errors
       name : Name of how the plot should be saved
       M    : Amount of rows for the subplot
       Not_show: Do you want to add graph were there is no pretraining'''
    label_font_size = 80

    Pretrained_data=data[1:,0]
    Only_Physic_data=data[0,1:]
    data=data[1:,1:]

    n=data[0,:].shape[0]+1
    l=data[:,0].shape[0]

    #Create Graph
    adjacency_matrix=np.zeros((n,n))
    adjacency_matrix[1:,0]=1
    adjacency_matrix[0,1:]=adjacency_matrix[1:,0]
    G = nx.Graph(adjacency_matrix)
    custom_positions = {0: ((n-1)//2,2)}
    edge_labels = {}
    for j in range(1,n):
            edge_labels[(0,j)]=str(2**(j+3))
            custom_positions[j]= (j,1)
    
    #Create Plot
    fig, axs = plt.subplots(M, int(round(l/M,0)), figsize=(120, 120))
    for i, ax in enumerate(axs.flat):
        if i>=M*l//M and Not_show:
            #Blank image for nice plot
            ax.imshow(np.ones((n, n)), cmap='gray', vmin=0, vmax=1)  
            ax.axis('off')
        else:
           if i>=M*l//M:
               labels = {0: 'None'}
               Not_show=True
           else:
               labels = {0: f'{2**i} : {round(Pretrained_data[i],3)}'}
           
           for j in range(1,n):
               if i>=M*l//M:
                   labels[j]=str(round(Only_Physic_data[j-1],3))
               else:    
                   labels[j]=str(round(data[i,j-1],3))
       
           
           nx.set_edge_attributes(G, edge_labels, "label")
           nx.set_node_attributes(G, labels, "label")
           nx.draw(G, pos=custom_positions, labels=nx.get_node_attributes(G, "label"), with_labels=True, node_size=0, arrows=False,ax=ax,font_size=label_font_size)
           edge_labels = nx.get_edge_attributes(G, "label")
           nx.draw_networkx_edge_labels(G, pos=custom_positions, edge_labels=edge_labels,ax=ax,font_size=label_font_size)
    
    plt.suptitle('Error for different training samples', fontsize=128)   
    plt.savefig(name)
