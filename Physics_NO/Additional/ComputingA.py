import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def Laplace(u,D=2):
    s=u.size(-1)
    u_hat=torch.fft.fft2(u,dim=[-2,-1])
    assert (u.device==u_hat.device) 
    k_max=s//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(s, 1).repeat(1, s).reshape(1,s,s)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1, s).repeat(s, 1).reshape(1,s,s)

    Laplace_u_hat =-4*(torch.pi/D)**2*(k_x**2+k_y**2)*u_hat
    Laplace_u=torch.fft.irfft2(Laplace_u_hat[:, :, :k_max + 1], dim=[-2, -1])
    return Laplace_u

def computeA(model, data_set, D):
    total_parameters = sum(p.numel() for p in model.parameters())
    A = torch.zeros((total_parameters, total_parameters))
    N = len(data_set)
    
    for f in data_set:
        counter1 = 0
        counter2 = 0
        for l, theta1 in enumerate(model.parameters()):
            for k, theta2 in enumerate(model.parameters()):
                if k <= l:
                    model.train()
                    model_f = model(f)

                    # Computing derivative with respect to theta (still a vector because of the structure of the network)
                    phi1 = compute_phi(model_f=model_f, theta=theta1)
                    phi2 = compute_phi(model_f=model_f, theta=theta2)

                    for i in range(torch.numel(theta1)):
                        for j in range(torch.numel(theta2)):
                            # Getting the i derivative of theta
                            phi_i = get_derivative(phi1, i)
                            phi_j = get_derivative(phi2, j)
                            
                            Dphi_i = Laplace(phi_i, D)
                            Dphi_j = Laplace(phi_j, D)


                            index_i = counter1 + i
                            index_j = counter2 + j                     

                            # Monte Carlo integrate over Dphi_i * Dphi_j with Domain D
                            Integral = D**2 * torch.mean((Dphi_i * Dphi_j).view(1, -1))
                            A[index_i, index_j] += Integral
                            if index_i!=index_j:
                               A[index_j, index_i] += Integral

                counter2 += torch.numel(theta2)
                
            counter1 += torch.numel(theta1)
            counter2 = 0
            #print(counter1,counter2)

    # Take mean over all functions in the dataset
    A = 1 / N * A
    return A
                  
def compute_phi(model_f,theta):
    theta_model_f=torch.zeros([model_f.shape[-2],model_f.shape[-1],*theta.shape])
    for i in range(model_f.shape[-2]):
       for j in range(model_f.shape[-1]):
           #Compute gradient for each output point of the sxs grid
           x_0=model_f[0,i,j]
           theta_model_f[i,j,:] = torch.autograd.grad(x_0,theta,retain_graph=True)[0] 
    theta_model_f = theta_model_f.view(model_f.shape[-2], model_f.shape[-1], -1)
    return theta_model_f

def get_derivative(theta_model_f,i):
    theta_i=theta_model_f[:,:,i]
    return theta_i



class Test_Conv(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Test_Conv, self).__init__()
        self.fc1 = nn.Conv2d(input_size, hidden_size,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_size, output_size,kernel_size=3,padding=1)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

net = Test_Conv(1,2,1)
net.train()
D=2
i=0
total_parameters = sum(p.numel() for p in net.parameters())
num_samples = 10  # You can adjust this number based on your needs
training_data = torch.rand(num_samples, 16, 16)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

#I think this is not possible for CNO to big!!
A=computeA(net,train_dataloader,D)
eigenvalues, _ = np.linalg.eigh(A)

plt.hist(eigenvalues, bins='auto', color='grey', alpha=0.7, rwidth=0.85)
plt.title('Eigenvalue Histogram')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
plt.savefig('test')







