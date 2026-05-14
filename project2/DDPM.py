import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels, 
            self.modelconfig.num_feat, 
            self.modelconfig.num_classes, 
            self.modelconfig.input_dim
        )

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.modelconfig.beta_1, self.modelconfig.beta_T, self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.  
        
        t_idxs = t_s.long()
        
        beta_all = torch.linspace(beta_1, beta_T, T).to(t_s.device)
        
        beta_t = beta_all[t_idxs-1]
        sqrt_beta_t = torch.sqrt(beta_t)
        
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)

        alpha_t_bar = torch.cumprod(1 - beta_all, 0)[t_idxs-1].view(-1, 1)
        
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)

        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function.  
                
        if (torch.rand(1) < self.modelconfig.mask_p).all():
            conditions = torch.zeros(conditions.shape[0], self.modelconfig.num_classes).to(images.device) \
                            + self.modelconfig.condition_mask_value
        else:  
            conditions = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).to(images.device)

        t = torch.randint(1, self.modelconfig.T+1, (conditions.shape[0],), device=images.device) / self.modelconfig.T
        
        eps = torch.randn_like(images)
        
        schedule = self.scheduler(t)
                
        x = schedule["sqrt_alpha_bar"].view(-1, 1, 1, 1) * images \
            + schedule["sqrt_oneminus_alpha_bar"].view(-1, 1, 1, 1) * eps
                    
        out = self.network(x, t, conditions)
        
        noise_loss = self.loss_fn(out, eps)

        # ==================================================== #
        return noise_loss

    def sample(self, conditions, omega):
        T = self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  
        
        masked_conditions = torch.zeros(conditions.shape[0], self.modelconfig.num_classes) \
                                + self.modelconfig.condition_mask_value
        X_t = torch.randn(conditions.shape[0], 28, 28)
        
        schedule = self.scheduler(torch.arange(1, self.modelconfig.T+1)) # 1, ..., T
        
        for t in range(self.modelconfig.T, 0, -1):
            z = torch.randn(conditions.shape[0], 28, 28) if t > 1 else 0
            
            t = torch.tensor(t, device=X_t.device)
                        
            eps = (1 + self.modelconfig.omega) * self.network(X_t, conditions, t) \
                    - self.modelconfig.omega * self.network(X_t, masked_conditions, t)
                    
            X_t = schedule["oneover_sqrt_alpha"][t-1] * (X_t - (schedule["beta_t"][t-1] / schedule["sqrt_oneminus_alpha_bar"][t-1]))\
                    + schedule["sqrt_beta_t"][t-1]*z
            
        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1)
        return generated_images 