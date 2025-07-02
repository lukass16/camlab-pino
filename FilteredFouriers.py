import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LowPassFilter(nn.Module):
    """
    A simple low-pass filter using FFT and ideal filtering.
    Can return either filtered frequency domain or filtered real space.
    """
    def __init__(self, s, D=1.0, k_cut=None):
        super(LowPassFilter, self).__init__()
        self.s = s  # define size of the grid
        self.D = D  # define domain size
        self.h = self.D / self.s  # define step size
        
        # Set default k_cut to be 2/3 of the maximum frequency if not provided
        if k_cut is None:
            # Maximum frequency is pi/h, so default to 2/3 of that
            self.k_cut = (2.0/3.0) * np.pi / self.h
        else:
            self.k_cut = k_cut
            
        # Pre-compute frequency grids for efficiency
        self._setup_frequency_grids()
        
    def _setup_frequency_grids(self):
        """Pre-compute the frequency grids for FFT operations"""
        # Create frequency grids
        kx = torch.fft.fftfreq(self.s, d=self.h) * 2 * np.pi
        ky = torch.fft.fftfreq(self.s, d=self.h) * 2 * np.pi
        
        # Create 2D meshgrid
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')
        
        # Compute k-magnitude squared
        k_squared = kx_grid**2 + ky_grid**2
        k_magnitude = torch.sqrt(k_squared)
        
        # Create low-pass filter
        self.low_pass_filter = (k_magnitude <= self.k_cut).float()
        
        # Store k_squared for Laplacian computation
        self.k_squared = k_squared
        
    def forward(self, u, return_frequency_domain=False):
        """
        Apply low-pass filter to input.
        
        Args:
            u: Input tensor of shape [batch, height, width] or [batch, channels, height, width]
            return_frequency_domain: If True, returns filtered FFT. If False, returns filtered real space.
            
        Returns:
            filtered_result: Either filtered frequency domain or filtered real space
        """
        # Handle input dimensions
        if u.dim() == 3:
            u_input = u.unsqueeze(1)
            squeeze_output = True
        elif u.dim() == 4:
            u_input = u
            squeeze_output = False
        else:
            raise ValueError(f"Expected 3D or 4D input tensor, got {u.dim()}D")
        
        batch_size, channels, height, width = u_input.shape
        
        # Ensure square input for this implementation
        if height != self.s or width != self.s:
            raise ValueError(f"Expected input size {self.s}x{self.s}, got {height}x{width}")
        
        # Move frequency grids to same device as input
        if self.low_pass_filter.device != u_input.device:
            self.low_pass_filter = self.low_pass_filter.to(u_input.device)
            self.k_squared = self.k_squared.to(u_input.device)
        
        # Process each channel separately
        results = []
        
        for c in range(channels):
            u_channel = u_input[:, c, :, :]  # [batch, height, width]
            
            # Apply the filtering to each sample in the batch
            batch_results = []
            for b in range(batch_size):
                # Take 2D FFT
                u_fft = torch.fft.fft2(u_channel[b])
                
                # Apply low-pass filter
                u_fft_filtered = u_fft * self.low_pass_filter
                
                if return_frequency_domain:
                    # Return filtered frequency domain
                    batch_results.append(u_fft_filtered)
                else:
                    # Inverse FFT to get filtered real space
                    u_filtered_real = torch.fft.ifft2(u_fft_filtered).real
                    batch_results.append(u_filtered_real)
            
            # Stack batch results
            channel_result = torch.stack(batch_results, dim=0)
            results.append(channel_result)
        
        # Stack channel results
        result = torch.stack(results, dim=1)
        
        # Remove channel dimension if input didn't have it
        if squeeze_output:
            result = result.squeeze(1)
            
        return result

class Laplace(nn.Module):
    """
    Modular spectral Laplacian using FFT and ideal low-pass filtering.
    Step 1: Apply low-pass filter
    Step 2: Compute spectral derivative
    """
    def __init__(self, s, D=1.0, k_cut=None, cut_size=0):
        super(Laplace, self).__init__()
        self.s = s
        self.D = D
        self.h = self.D / self.s
        self.cut_size = cut_size
        
        # Create the low-pass filter
        self.filter = LowPassFilter(s, D, k_cut)
        
    def forward(self, u):
        """
        Compute the spectral Laplacian using modular approach:
        1. Low-pass filter
        2. Compute spectral derivative
        
        Args:
            u: Input tensor of shape [batch, height, width] or [batch, channels, height, width]
            
        Returns:
            laplace_result: Laplacian of the input after applying spectral filtering
            cut_size: Number of boundary points that should be cut
        """
        # Handle input dimensions
        if u.dim() == 3:
            u_input = u.unsqueeze(1)
            squeeze_output = True
        elif u.dim() == 4:
            u_input = u
            squeeze_output = False
        else:
            raise ValueError(f"Expected 3D or 4D input tensor, got {u.dim()}D")
        
        batch_size, channels, height, width = u_input.shape
        
        # Ensure square input for this implementation
        if height != self.s or width != self.s:
            raise ValueError(f"Expected input size {self.s}x{self.s}, got {height}x{width}")
        
        # Move frequency grids to same device as input
        if self.filter.k_squared.device != u_input.device:
            self.filter.k_squared = self.filter.k_squared.to(u_input.device)
        
        # Process each channel separately
        laplace_results = []
        
        for c in range(channels):
            u_channel = u_input[:, c, :, :]  # [batch, height, width]
            
            # Apply the spectral method to each sample in the batch
            batch_results = []
            for b in range(batch_size):
                # Step 1: Take 2D FFT and apply low-pass filter
                u_fft = torch.fft.fft2(u_channel[b])
                u_fft_filtered = u_fft * self.filter.low_pass_filter
                
                # Step 2: Compute spectral Laplacian: ∇²ũ = F^(-1){-(k_x^2 + k_y^2) H(k) û}
                laplace_fft = -self.filter.k_squared * u_fft_filtered
                
                # Inverse FFT to get the Laplacian in real space
                laplace_real = torch.fft.ifft2(laplace_fft).real
                
                batch_results.append(laplace_real)
            
            # Stack batch results
            channel_result = torch.stack(batch_results, dim=0)
            laplace_results.append(channel_result)
        
        # Stack channel results
        laplace_result = torch.stack(laplace_results, dim=1)
        
        # Apply boundary cutting if specified
        if self.cut_size > 0:
            laplace_result = laplace_result[:, :, self.cut_size:-self.cut_size, self.cut_size:-self.cut_size]
        
        # Remove channel dimension if input didn't have it
        if squeeze_output:
            laplace_result = laplace_result.squeeze(1)
            
        return laplace_result, self.cut_size

class HybridLaplacian(nn.Module):
    """
    Hybrid Laplacian approach:
    1. Apply spectral low-pass filter
    2. Convert back to real space  
    3. Apply finite difference Laplacian
    """
    def __init__(self, s, D=1.0, k_cut=None, fd_type="9-point"):
        super(HybridLaplacian, self).__init__()
        self.s = s
        self.D = D
        self.h = self.D / self.s
        self.fd_type = fd_type
        
        # Create the low-pass filter
        self.filter = LowPassFilter(s, D, k_cut)
        
        # Set up finite difference kernels (copied from FiniteDifferences.py)
        if self.fd_type == "9-point": # 9-point Patra-Karttunen (default)
            self.laplace_kernel = torch.tensor([[1,4,1],
                                               [4,-20,4],
                                               [1,4,1]], dtype=torch.float32).view(1,1,3,3)/(6*self.h**2)
            self.cut_size = 1
        elif self.fd_type == "9-point-OP": # 9-point Oono-Puri
            self.laplace_kernel = torch.tensor([[ 1,  2,  1],
                                               [ 2, -12,  2],
                                               [ 1,  2,  1]], dtype=torch.float32).view(1,1,3,3) / (4*self.h**2)
            self.cut_size = 1
        elif self.fd_type == "13-point":
            self.laplace_kernel = torch.tensor([[ 0.,   0.,  -1.,   0.,   0.],
                                               [ 0.,   3.,   4.,   3.,   0.],
                                               [-1.,   4., -24.,   4.,  -1.],
                                               [ 0.,   3.,   4.,   3.,   0.],
                                               [ 0.,   0.,  -1.,   0.,   0.]], dtype=torch.float32).view(1, 1, 5, 5) / (6 * self.h**2)
            self.cut_size = 2
        else:
            raise ValueError("Invalid laplacian type, choose from 9-point, 9-point-OP, 13-point")
        
    def forward(self, u):
        """
        Compute hybrid Laplacian:
        1. Apply spectral low-pass filter
        2. Convert back to real space
        3. Apply finite difference Laplacian
        
        Args:
            u: Input tensor of shape [batch, height, width] or [batch, channels, height, width]
            
        Returns:
            laplace_result: Laplacian of the filtered input using finite differences
            cut_size: Number of boundary points that should be cut
        """
        # Handle input dimensions
        if u.dim() == 3:
            u_input = u.unsqueeze(1)
            squeeze_output = True
        elif u.dim() == 4:
            u_input = u
            squeeze_output = False
        else:
            raise ValueError(f"Expected 3D or 4D input tensor, got {u.dim()}D")
        
        # Step 1 & 2: Apply low-pass filter and return to real space
        u_filtered = self.filter(u_input, return_frequency_domain=False)
        
        # Step 3: Apply finite difference Laplacian
        # Move kernel to same device as input
        if self.laplace_kernel.device != u_filtered.device:
            self.laplace_kernel = self.laplace_kernel.to(u_filtered.device)
        
        # Apply convolution
        laplace_result = F.conv2d(u_filtered, self.laplace_kernel, padding=0)
        
        # Remove channel dimension if input didn't have it
        if squeeze_output:
            laplace_result = laplace_result.squeeze(1)
            
        return laplace_result, self.cut_size 