import torch
import torch.nn as nn
import numpy as np
from .spline import *
from .utils import sparse_mask
from .LayerScaling import LayerScaling


class DropKANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device
    
    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward 
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False, drop_rate=0.0, drop_mode='postact', drop_scale=True, neuron_fun=None, input_preprocessing=None):
        ''''
        initialize a DropKANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            drop_rate: float
                The rate of drop for the DropKAN mask. Default: 0.0.
            drop_mode: str
                Accept the following values 'postspline' the drop mask is applied to the layer's postsplines, 'postact' the drop mask is applied to the layer's postacts, 'dropout' applies a standard dropout layer to the inputs. Default: 'postact'.
             drop_scale: bool
                If true, the retained postsplines/postacts are scaled by a factor of 1/(1-drop_rate). Default: True
            device : str
                device
        Returns:
        --------
            self
            
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        (3, 5)
        '''
        super(DropKANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        ### grid size: (batch, in_dim, out_dim, G + 1) => (batch, in_dim, G + 2*k + 1)
        
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / num
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        #if isinstance(scale_base, float):
        if sparse_init:
            mask = sparse_mask(in_dim, out_dim)
        else:
            mask = 1.
        
        self.scale_base = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_base * mask).requires_grad_(sb_trainable)  # make scale trainable
        #else:
        #self.scale_base = torch.nn.Parameter(scale_base.to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False),

        #self.register_buffer('mask', None)
        self.grid_eps = grid_eps
        self.drop_rate = drop_rate
        self.drop_mode = drop_mode
        self.drop_scale = drop_scale
        self.neuron_fun = neuron_fun
        self.neuron_train = torch.nn.Parameter(torch.ones(1)).requires_grad_(True)
        self.count = 0
        self.total = 0
        self.input_preprocessing = input_preprocessing

        ### remove weight_sharing & lock parts
        #self.weight_sharing = torch.arange(out_dim*in_dim).reshape(out_dim, in_dim)
        #self.lock_counter = 0
        #self.lock_id = torch.zeros(out_dim*in_dim).reshape(out_dim, in_dim)
        

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([100, 5]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]))
        '''
        batch = x.shape[0]

        if self.input_preprocessing == 'ls':
            #print('before', x)
            x = LayerScaling()(x)
            #print('after', x)
        elif self.input_preprocessing == 'ln':
            x = nn.LayerNorm(self.in_dim, elementwise_affine=False)(x)

        #print(self.drop_mode, self.drop_rate, self.drop_scale)
        if self.training:
            if self.drop_mode == 'dropout' and self.drop_rate > 0 and self.drop_scale:
                    #print('dropout with scale')
                    mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - self.drop_rate)
                    x = x * mask / (1 - self.drop_rate)
            elif self.drop_mode == 'dropout' and self.drop_rate > 0 and not self.drop_scale:
                    mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - self.drop_rate)
                    x = x * mask
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)  # y shape: (batch, in_dim, out_dim)
        
        postspline = y.clone().permute(0,2,1) # postspline shape: (batch, out_dim, in_dim)

        if self.training:
            if self.drop_mode == 'postspline' and self.drop_rate > 0 and self.drop_scale:
                    mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                    y = y * mask / (1 - self.drop_rate)
            elif self.drop_mode == 'postspline' and self.drop_rate > 0 and not self.drop_scale:
                    mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                    y = y * mask

        if self.training and self.drop_mode == 'dropkan' and self.drop_rate > 0:
            # Decide which input features to drop (same for the whole batch)
            feature_mask = torch.empty((1, y.shape[1]), device=y.device).bernoulli_(1 - self.drop_rate)

            feature_mask = feature_mask.view(1, 1, -1)  # shape: (1, 1, in_dim)


            # Reshape to apply mask on feature dimension
            feature_mask = feature_mask.permute(0, 2, 1)  # shape becomes [1, 11, 1]
            feature_mask = feature_mask.repeat(1, 1, y.shape[2])  # shape becomes [1, 11, 2]
            #print('here')
            #print(y)
            #print(feature_mask)


            if self.drop_scale:
                y = y * feature_mask / (1 - self.drop_rate)
            else:
                y = y * feature_mask

            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        #y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)

        if self.training:
            if self.drop_mode == 'postact' and self.drop_rate > 0 and self.drop_scale:
                    mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                    y = y * mask / (1 - self.drop_rate)
            elif self.drop_mode == 'postact' and self.drop_rate > 0 and not self.drop_scale:
                    mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                    y = y * mask

        if self.drop_mode == 'postact1' and self.drop_rate > 0:
                    y = torch.nn.functional.dropout(y, p=self.drop_rate, training=self.training)


        if self.neuron_fun == 'sum':
                y = torch.sum(y, dim=1)
                #print(y.shape)
        elif self.neuron_fun == 'min':
                y = torch.min(y, dim=1).values  # torch.min returns a tuple (values, indices)
        elif self.neuron_fun == 'max':
                y = torch.max(y, dim=1).values  # torch.max returns a tuple (values, indices)
        elif self.neuron_fun == 'multiply':
                y = torch.prod(y, dim=1)  # Element-wise product along dim=2
        elif self.neuron_fun == 'mean':
                y = torch.mean(y, dim=1)
        elif self.neuron_fun == 'std':
                y = torch.std(y, dim=1)
        elif self.neuron_fun == 'var':
                y = torch.var(y, dim=1)
        elif self.neuron_fun == 'median':
                y = torch.median(y, dim=1).values  # torch.median returns a tuple (values, indices)
        elif self.neuron_fun == 'norm':
                y = torch.norm(y, dim=1)
        elif self.neuron_fun == 'attention':
            #print("Before attention:", y.shape)  # [32, 42, 84]

            # Transpose to apply attention across in_dim
            y = y.transpose(1, 2)  # [32, 84, 42]
            attn_scores = torch.softmax(y, dim=2)  # [32, 84, 42]
            print(attn_scores)

            # Weighted sum over dim=2 (i.e., out = sum_i (x_i * attention_i))
            y = (y * attn_scores).sum(dim=2)  # [32, 84]

            #print("After attention:", y.shape)
        #elif self.neuron_fun == 'any':
        #        y = torch.any(y, dim=1).float()  # Convert boolean result to float
        #elif self.neuron_fun == 'all':
        #        y = torch.all(y, dim=1).float()  # Convert boolean result to float
        #elif self.neuron_fun == 'cumsum':
        #        y = torch.cumsum(y, dim=1)  # Cumulative sum along dim=2
        #elif self.neuron_fun == 'cumprod':
        #        y = torch.cumprod(y, dim=1)  # Cumulative product along dim=2
        #elif self.neuron_fun == 'amax':
        #        y = torch.amax(y, dim=1)  # Maximum value along dim=2
        #elif self.neuron_fun == 'amin':
        #        y = torch.amin(y, dim=1)  # Minimum value along dim=2
        #elif self.neuron_fun == 'argmax':
        #        y = torch.argmax(y, dim=1).float()  # Indices of max values along dim=2
        #elif self.neuron_fun == 'argmin':
        #        y = torch.argmin(y, dim=1).float()  # Indices of min values along dim=2
        #elif self.neuron_fun == 'sumln':
        #        y = torch.sum(y, dim=1)
        #        y = nn.LayerNorm(self.out_dim, elementwise_affine=False)(y)
        #elif self.neuron_fun == 'sumlnkan':
        #        y = torch.sum(y, dim=1)
        #        y = LayerNormKAN()(y)
        else:
            raise ValueError(
                "Unsupported function. Choose from 'sum', 'min', 'max', 'multiply', 'mean', 'std', 'var', 'median', 'norm', 'any', 'all', 'prod', 'cumsum', 'cumprod', 'amax', 'amin', 'argmax', 'argmin'.")

        # Count the number of elements between -1 and 1 (inclusive)
        #print(y)
        #print(((y >= -1) & (y <= 1)).sum().item())
        #print(y[0].shape[0])
        self.count += ((y >= -1) & (y <= 1)).sum().item()
        self.total += y.numel()

        #y = torch.sum(y, dim=1)  # shape (batch, out_dim)
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''
        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[ids, :].permute(1,0)
        margin = 0.01
        h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
        grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        '''
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        #x_eval = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_eval = x
        pgrid = parent.grid # (in_dim, G+2*k+1)
        pk = parent.k
        y_eval = coef2curve(x_eval, pgrid, parent.coef, pk)
        
        h = (pgrid[:,[-pk]] - pgrid[:,[pk]])/self.num
        grid = pgrid[:,[pk]] + torch.arange(self.num+1,) * h
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:,out_id]
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    
    def swap(self, i1, i2, mode='in'):
        
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

