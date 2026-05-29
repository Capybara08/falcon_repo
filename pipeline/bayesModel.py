import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
from pipeline.features import analyte_meta_tensor

utils_path = os.path.abspath('/Users/JasL/Desktop/research/sci-fair/pfa_code.py/my_utils.py')
sys.path.append(utils_path)
preprocess_path = os.path.abspath('/Users/JasL/Desktop/research/sci-fair/pfa_code.py/pipeline/preprocess.py')
sys.path.append(preprocess_path)

# pos_weight = torch.tensor([10.0]) # Give 10x importance to a detection.
OPTIM_FN = nn.BCEWithLogitsLoss()
        
class PFASDataset(Dataset):
    def __init__(self, df, GLOBAL_FEATS, LOCAL_FEATS, GLOBAL_CHAIN, LOCAL_CHAIN=None):
        """
        PyTorch Dataset for PFAS detection data. Concatenates a missingness mask to each
        feature vector so the model can distinguish true zeros from missing values.
    
        Parameters
        df           : preprocessed DataFrame
        GLOBAL_FEATS : list of global feature column names (post scale/encode)
        LOCAL_FEATS  : list of local feature column names
        GLOBAL_CHAIN : list of global PFAS target column names
        LOCAL_CHAIN  : list of local PFAS target column names (optional)
        """
        self.df = df
        self.df_X_glob = df[GLOBAL_FEATS]
        self.df_Y_glob = df[GLOBAL_CHAIN]
        self.df_X_loc = df[LOCAL_FEATS]
        
        self.y_loc_chain = LOCAL_CHAIN

        self.X_glob_tensors = torch.tensor(df[GLOBAL_FEATS].apply(pd.to_numeric, errors='coerce').values, dtype=torch.float32) # median impute for msising env data --> likely no missing env data
        self.X_loc_tensors = torch.tensor(df[LOCAL_FEATS].apply(pd.to_numeric, errors='coerce').values, dtype=torch.float32)

        y_glob_numeric = df[GLOBAL_CHAIN].apply(pd.to_numeric, errors='coerce').values
        self.Y_glob_tensors = torch.tensor(y_glob_numeric, dtype=torch.float32)

        if LOCAL_CHAIN:
            y_loc_numeric = df[LOCAL_CHAIN].apply(pd.to_numeric, errors='coerce').values
            self.Y_loc_tensors = torch.tensor(y_loc_numeric, dtype=torch.float32)
        else:
            self.Y_loc_tensors = torch.zeros((len(df), 0)) 

    def __len__(self): 
        return len(self.X_glob_tensors) 

    def __getitem__(self, idx):
        # Original data loading.
        x_glob = self.X_glob_tensors[idx].clone()
        x_loc = self.X_loc_tensors[idx].clone()
        
        # Create the Mask: 1.0 where data exists, 0.0 where it was NaN.
        # (Do this BEFORE nan_to_num or median imputation)
        mask_glob = (~torch.isnan(x_glob)).float()
        mask_loc = (~torch.isnan(x_loc)).float()

        # Impute NaNs with 0 (since the mask now handles the signal).
        x_glob = torch.nan_to_num(x_glob, nan=0.0)
        x_loc = torch.nan_to_num(x_loc, nan=0.0)

        # Concatenate the mask to the features.
        # This doubles the input size: [Value1, Value2, Mask1, Mask2].
        x_glob_combined = torch.cat([x_glob, mask_glob], dim=-1)
        x_loc_combined = torch.cat([x_loc, mask_loc], dim=-1)

        # Return the combined tensors.
        return x_glob_combined, self.Y_glob_tensors[idx], x_loc_combined, self.Y_loc_tensors[idx]

    #### THIS WAS BEFORE DOING NAN MASKING INTO THE X INPUTS (PROVIDING A MASK THAT ADDS DIMENSIONS) 
    #### SHOWS THAT DATA WAS MISSING WHERE --> HELP MODEL LEARN MNAR VS SMTH ELSE
    # def __getitem__(self, idx): 
    #     # Returns global X, global Y, and local X/Y
    #     return self.X_global[idx], self.y_global[idx], self.x_local[idx], self.y_local[idx]

# Helper class above PFASGlobal.
class BayesianLinearLayer(nn.Module):
    """
    Single Bayesian linear layer with weight modulation from a chemical prior.
    Samples weights from a learned posterior during training; uses the mean at inference.
    Modulation is applied via a sigmoid gate so it can only dampen or pass through the learned weight distribution.
 
    Parameters
    in_features : input dimensionality
    """
    def __init__(self, in_features):
        super().__init__()
        # Bayesian ML parameters are stored as Module attributes.
        self.in_features = in_features
        self.w_mu = nn.Parameter(torch.Tensor(1, in_features).normal_(0, 0.1))
        self.w_logvar = nn.Parameter(torch.Tensor(1, in_features).normal_(-3, 0.1))
        self.b_mu = nn.Parameter(torch.Tensor(1).normal_(0, 0.1))
        self.b_logvar = nn.Parameter(torch.Tensor(1).normal_(-3, 0.1))
        self.mod_proj = nn.Linear(32, in_features)
    
    def forward(self, x, prior_mod=None, sample=True):
        w_logvar = self.w_logvar.clamp(-10, 5)
        b_logvar = self.b_logvar.clamp(-10, 5)

        # modulation with chem meta
        # can only dampen or pass thru (squashed between 0 to 1)
        if prior_mod is not None:
            """
            target_w_u is a learned parameter 
            prior_mod is coming from the hypernet (which contains reLU activations)
            Prior can output values that are positive or have differently scaled magnitudes 
            in comparison to the scale of the learned parameter.
            
            By adding prior_mod + target_w_mu we bias the weights in a way the
            model didnt "learn" to handle
            """
            modulation = torch.sigmoid(self.mod_proj(prior_mod)) 
            target_w_mu = self.w_mu * modulation 
        else:
            print("WARNING PRIOR MOD IS NONE")
            with open('warnings.txt', 'a') as f:
                f.write(f"Prior has none value") 

        # Extract the Python boolean value from the tensor or variable.
        if isinstance(sample, torch.Tensor):
            # If it's a tensor with more than 1 element, it's definitely not the 'sample' flag.
            do_sample = bool(sample.item()) if sample.numel() == 1 else True
        else:
            do_sample = sample

        if do_sample:
            w_sigma = torch.exp(0.5 * w_logvar)
            b_sigma = torch.exp(0.5 * b_logvar)
            weights = target_w_mu + w_sigma * torch.randn_like(w_sigma)
            bias = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            weights, bias = target_w_mu, self.b_mu
        
        return F.linear(x, weights, bias)

    def kl_divergence(self):
        w_logvar = self.w_logvar.clamp(-10, 5)
        b_logvar = self.b_logvar.clamp(-10, 5)
        kl_w = -0.5 * torch.sum(1 + w_logvar - self.w_mu.pow(2) - w_logvar.exp())
        kl_b = -0.5 * torch.sum(1 + b_logvar - self.b_mu.pow(2) - b_logvar.exp())
        return kl_w + kl_b

class BayesianMLP(nn.Module):
    """
    Two-layer MLP with a deterministic first layer and a Bayesian second layer.
    Accepts an optional chemical prior vector to modulate the Bayesian weights.
 
    Parameters
    in_features : input dimensionality
    hidden_dim  : hidden layer size
    dropout     : dropout rate applied after the first layer
    """
    def __init__(self, in_features, hidden_dim=32, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn = nn.LayerNorm(hidden_dim)      # Stabilizes input to Bayesian layer.
        self.fc2 = BayesianLinearLayer(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, prior_mod=None, sample=False):
        """
        Modulated prior through the molecular descriptor flags (hypernet).
        """
        x = F.relu(self.fc1(x))
        if x.shape[0] > 1: # BatchNorm needs batch_size > 1.
            x = self.bn(x)
        x = self.dropout(x)
        return self.fc2(x, prior_mod=prior_mod, sample=sample)
    
    def kl_divergence(self):
        return self.fc2.kl_divergence()

#### WORK IN PROGRESS
class PFASGlobal(nn.Module):
    def __init__(self, input_env_dim, pfas_num_classes, molecular_descriptors, hidden_dim=64, dropout=0.2): # ridded of meta data for now
        """
        x_glob: [batch, d_env]
        meta_tensor: [n_analytes, d_meta]

        Create the classifer chain, input dims, output dims
        """
        super().__init__()
        self.analyte_weights = nn.Parameter(torch.ones(pfas_num_classes), requires_grad=False)
        self.input_env_dim = input_env_dim
        self.pfas_num_classes = pfas_num_classes
        self.hidden_dim = hidden_dim
        self.register_buffer('meta_tensor', molecular_descriptors)
        num_meta = molecular_descriptors.shape[1] # num meta features
        self.hyper_scale = nn.Parameter(torch.tensor([0.1])) # Start small
        # Scale the meta data
        self.hypernet_scaled = nn.Sequential(
            nn.Linear(num_meta, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # # Hypernetwork to modulate bayesian priors
        # self.hypernet = nn.Sequential(
        #     nn.Linear(5, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, hidden_dim)
        # )

        self.chain = nn.ModuleList([
            BayesianMLP(self.input_env_dim + i, hidden_dim=hidden_dim, dropout=dropout) for i in range(self.pfas_num_classes) # Changed it to MLP
        ])

    # PFAS chemical metadata is injected here
    def forward(self, x, sample=True, y_true=None, tf_ratio=1.0): # Pass in the actual
        """
        Pass in the env features
        """
        current_input = x # start at just x and append
        logits_list = []
        # print(f"DEBUG: Meta tensor shape in forward: {self.pfca_meta_tensor.shape}")
        # PFCA chain

        for i, layer in enumerate(self.chain): # classifier chain
            layer_meta = self.meta_tensor[i:i+1] # .expand(current_input.size(0), -1)
            raw_prior = self.hypernet_scaled(layer_meta)
            hyper_prior = torch.tanh(raw_prior) * self.hyper_scale            # step_input = torch.cat([current_input], dim=1)
            
            # The logic is now encapsulated in the layer's forward method            
            logits = layer(current_input, prior_mod=hyper_prior, sample=sample)
            logits_list.append(logits) # grab logits and append to list
            pred = torch.sigmoid(logits)

            if y_true is not None and self.training: # Teacher forcing: use actual label when available, pred fallback
                actual = y_true[:, i:i+1]
                tf_mask = torch.rand(actual.shape[0], 1, device=actual.device) < tf_ratio
                mask = (~torch.isnan(actual)) & tf_mask
                context = torch.where(mask, actual, pred)
            else: # teacher force backup
                context = pred
            current_input = torch.cat([current_input, context], dim=1)
        
        return torch.cat(logits_list, dim=1)
    
    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in self.chain)

class PFASLocal(nn.Module):
    def __init__(self, global_model, local_chain, input_dim_loc, molecular_descriptors):
        super().__init__()
        """
        molecular desciprtors is a list of dicts with each dict represnting a pfas analyte
        """
        self.global_head = global_model
        self.global_chain_dim = global_model.pfas_num_classes
        self.loc_chain_dim = len(local_chain) # the personalization pfas targets
        
        # Local specific adjustment params
        self.local_adjusts = nn.Parameter(torch.zeros(len(local_chain), global_model.hidden_dim))

        # Register descriptors as buffers to move with the model to GPU --> this tells model its not a feature to be optimizer.step
        self.register_buffer('meta_tensor', molecular_descriptors)
        num_meta = molecular_descriptors.shape[1] # shape: [len(local_chain), num_descriptors]
        tail_input_dim = input_dim_loc + self.global_chain_dim

        self.local_tail = nn.ModuleList([
            BayesianLinearLayer(tail_input_dim + i)  # ← each link gets exactly the right input size
            for i in range(len(local_chain))
        ])
        self.local_pfas_names = local_chain

    def forward(self, x_glob, x_loc=None, y_glob_tar=None, y_loc_tar=None):
        #self.global_head.eval() -> we put global head into eval mode/freeze it in the training loop instead of here. bc its a "race" condition
        with torch.no_grad():
            global_logits = self.global_head(x_glob) # grab global logits by calling forward method
            global_preds = torch.sigmoid(global_logits)
        if y_glob_tar is not None:
            mask = ~torch.isnan(y_glob_tar)
            refined_global_context = torch.where(mask, y_glob_tar, global_preds)
        else:
            refined_global_context = global_preds

        if x_loc is None:
            batch_size = x_glob.size(0)
            expected_dim = self.local_tail[0].in_features - self.global_chain_dim
            x_loc = torch.zeros((batch_size, expected_dim), device=x_glob.device)
        
        current_input = torch.cat([x_loc, refined_global_context], dim=1)
        
        local_outputs = []

        for i, layer in enumerate(self.local_tail):
            prior_mod = self.global_head.hypernet_scaled(self.meta_tensor[i:i+1])            
            
            logits = layer(current_input, prior_mod=prior_mod)           # ← BayesianLinearLayer does NOT expand internally
            local_outputs.append(logits)
            pred_y = torch.sigmoid(logits)
            
            # Teacher forcing
            if y_loc_tar is not None and self.training:
                actual = y_loc_tar[:, i:i+1]
                mask = ~torch.isnan(actual)
                context = torch.where(mask, actual, pred_y)
            else:
                context = pred_y
            current_input = torch.cat([current_input, context], dim=1)  # ← only one expansion, correct

        return global_logits, torch.cat(local_outputs, dim=1) if local_outputs else None

######################################### WORKING ON THE REPTILE ADAPT CODE
def reptile_adapt_fin(local_model, local_dloader, local_chain_names, inner_steps=5, inner_lr=0.01, meta_lr=0.1):
    # 1. Save original state
    """
    Model being passed in is: local model
    """
    # Set eval mode. Freeze global head
    num_local = len(local_chain_names)
    local_alpha = torch.ones(num_local, device='cpu')
    local_model.global_head.eval()
    for param in local_model.global_head.parameters():
        param.requires_grad_(False)

    full_local_list = local_dloader.dataset.y_loc_chain 
    target_indices = [i for i, name in enumerate(full_local_list) if name in local_chain_names]
    
    if not target_indices:
        print("Warning: No target analytes found in local dataset for personalization.")
        return local_model.state_dict()

    # Freeze Global: Ensure only the tail learns
    N = len(local_dloader.dataset)
    # Save original state of the tail only - want to meta learn the personalized part
    old_weights = {k: v.clone() for k, v in local_model.state_dict().items()}
    opt = torch.optim.SGD(local_model.local_tail.parameters(), lr=inner_lr)
    local_model.train()
    
    for i, (x_glob, y_glob, x_loc, y_loc) in enumerate(local_dloader):
        if i >= inner_steps: 
            break
        
        # Unpacking the "global model" inputs and local personalization inputs
        opt.zero_grad()
        y_loc_tar = y_loc[:, target_indices]
        _, logits = local_model(x_glob, x_loc, y_glob_tar=y_glob, y_loc_tar=y_loc_tar)
        
        loss = adaptive_masked_loss(logits, y_loc_tar, adaptive_alpha=local_alpha) # compute masked loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.local_tail.parameters(), max_norm=1.0)
        opt.step()
    # 3. Meta-Update (The "Reptile" Step)
    # After the inner loop, build updated state and load it
    new_weights = local_model.state_dict()
    updated_state = {}
    with torch.no_grad():
        for k in old_weights:
            updated_state[k] = old_weights[k] + meta_lr * (new_weights[k] - old_weights[k])
    
    # Unfreeze the global head
        # Unfreeze global head
    for param in local_model.global_head.parameters():
        param.requires_grad_(True)
    local_model.global_head.train()

    local_model.load_state_dict(updated_state)

    return local_model   # ✅ actually writes to the model

def train_global(global_model, global_dataloader, optimizer, epochs, kl_beta=0.001, N_global=None, adaptive_alpha=None): # N_global is ... local client's datasize is N but small clients have 100* stronger KL regularization
    N = N_global if N_global else len(global_dataloader.dataset) # Entire dataset len --> has to be weighted wrt entirety
    history = {'total': [], 'data': [], 'kl': []}
    
    # this is additive
    adaptive_factors = torch.ones(global_model.pfas_num_classes, device='cpu')
    
    global_model.train()
    for epoch in range(epochs):
        epoch_data = 0
        epoch_kl = 0

        all_preds  = []
        all_true = []
        for x_glob, y_glob, x_loc, y_loc in global_dataloader:
            optimizer.zero_grad()
            logits = global_model(x_glob, y_true=y_glob, tf_ratio=max(0, 1.0 - epoch/epochs))
            
            # Likelihood loss

            # additive
            data_loss = adaptive_masked_loss(
                logits, y_glob, 
                adaptive_alpha=adaptive_factors.to(logits.device) if adaptive_alpha is not None else None)
            """
            data_loss = masked_loss(logits, y_glob)
            """
            # KL divergence scaled by total dataset size (Standard Bayesian approach)
            # This prevents KL from overwhelming the data loss
            batch_n = x_glob.shape[0] # num samples in this mini batch
            kl_div = kl_beta * global_model.kl_divergence() * (batch_n / N_global)     
            
            kl_beta = 0.001   
            total_loss = data_loss + (kl_div / N)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_true.append(y_glob.detach().cpu())
            
            epoch_data += data_loss.item()
            epoch_kl += kl_div.item()
        with torch.no_grad():
            preds = torch.cat(all_preds, dim=0)
            trues = torch.cat(all_true, dim=0)
            
            for i in range(global_model.pfas_num_classes):
                mask = ~torch.isnan(trues[:, i])
                if mask.any():
                    # Calculate current recall
                    y_t = trues[mask, i]
                    y_p = (preds[mask, i] > 0.5).float()
                    
                    positives = y_t.sum()
                    if positives > 0:
                        recall = (y_t * y_p).sum() / positives
                        # If recall is low, increase the factor for the next epoch
                        # This formula slowly increases weight for difficult chemicals
                        if recall < 0.5:
                            adaptive_factors[i] += 0.1 
                        elif recall > 0.8:
                            adaptive_factors[i] = max(1.0, adaptive_factors[i] - 0.05)

        history['data'].append(epoch_data / len(global_dataloader))
        history['kl'].append(epoch_kl / len(global_dataloader))
        history['total'].append(history['data'][-1] + history['kl'][-1])
        
    print(f"Global Loss (combined KL and BCE loss): {history['total'][-1]:.4f} (BCE Loss: {history['data'][-1]:.4f}, KL: {history['kl'][-1]:.4f})")
    return global_model.state_dict(), history

def train_local(local_model, local_dloader, local_chain_names, epochs):
    # Set eval mode. Freeze global head
    num_local = len(local_chain_names)
    local_adaptive_factors = torch.ones(num_local, device='cpu')    
    
    local_model.global_head.eval()
    for param in local_model.global_head.parameters():
        param.requires_grad_(False)

    full_local_list = local_dloader.dataset.y_loc_chain 
    target_indices = [i for i, name in enumerate(full_local_list) if name in local_chain_names]
    
    if not target_indices:
        print("Warning: No target analytes found in local dataset for personalization.")
        return local_model.state_dict()

    local_model.train()
    # Freeze Global: Ensure only the tail learns
    optimizer = optim.Adam(local_model.local_tail.parameters(), lr=0.0005)
    N = len(local_dloader.dataset)

    for epoch in range(epochs):
        epoch_loss = 0
        all_preds, all_true = [], []
        for x_glob, y_glob, x_loc, y_loc in local_dloader:
            optimizer.zero_grad()
        
            # Forward pass: 
            # We pass x_loc and y_glob (for context) as defined in your PFASLocal.forward
            y_loc_tar = y_loc[:, target_indices]
            _, local_logits = local_model(x_glob, x_loc, y_glob_tar=y_glob, y_loc_tar=y_loc_tar)
            
            # Masked Loss: Use the fixed version that handles NaNs and raw concentrations
            data_loss = adaptive_masked_loss(local_logits, y_loc_tar, adaptive_alpha=local_adaptive_factors)
            
            # KL Divergence: Summing over the Bayesian layers in the tail
            total_kl = sum(layer.kl_divergence() for layer in local_model.local_tail)
            # Scale KL by N to keep data likelihood dominant
            kl_beta = 0.001
            total_loss = data_loss + (kl_beta * total_kl / N)
            total_loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(local_model.local_tail.parameters(), max_norm=1.0)
            optimizer.step() # gradient compute
                        
            # Check for corruption
            for name, param in local_model.named_parameters():
                if torch.isnan(param).any():
                    print(f"Weights in {name} corrupted!")
            epoch_loss += total_loss.item()
            all_preds.append(torch.sigmoid(local_logits).detach().cpu())
            all_true.append(y_loc_tar.detach().cpu())
        
        with torch.no_grad():
            preds = torch.cat(all_preds)
            trues = torch.cat(all_true)
            for i in range(num_local):
                mask = ~torch.isnan(trues[:, i])
                if mask.any() and trues[mask, i].sum() > 0:
                    # Calculate local recall for this specific chemical
                    recall = ((preds[mask, i] > 0.5) * trues[mask, i]).sum() / trues[mask, i].sum()
                    if recall < 0.4: # If failing locally, boost weight
                        local_adaptive_factors[i] += 0.1
        # Optional: Print every 20 epochs to monitor personalization convergence
        if (epoch + 1) % 2 == 0:
            print(f"Personalization Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(local_dloader):.4f}")
    
    # Unfreeze global head
    for param in local_model.global_head.parameters():
        param.requires_grad_(True)
    local_model.global_head.train()

    return local_model # return actual local model just like reptile

def finetune_global(agents, config, GLOBAL_CHAIN_FIN, adaptive_alpha):
    """
    Finetuning for the global heads. The global backbone is partially unfrozen.
    Uses the global chain of PFAS.
    Gets the personalized local env input features.
    """
    # List storing data for compare and loss comp
    epochs = config.get('finetune_epochs', 10)
    adaptive_factors = torch.ones(model.pfas_num_classes, device='cpu')
    # Grab the local features needed
    # Each node has the global training phase + local training phase
    # Grab the local data w local_dloader for the node
    for node_id, agent in agents.items():
        model = agent['model'] # Grab the model
        model.train() # Training mode...
        # Freeze all but classifier layer (fine tuning)
        for name, param in model.named_parameters():
            param.requires_grad = 'classifier' in name or 'output' in name
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.002
        )
        for epoch in range(epochs):
            for x_gb, y_gb, x_lc, y_lc in agent['train_loader']:
                # iterate through and grab local
                optimizer.zero_grad()
                inputs = x_lc
                outputs = y_gb # The global PFAS data.
                logits = model(x_lc, y_true=y_gb) # Forward pass; global model.
                # Do i need it?? --> tf_ratio=max(0, 1.0 - epoch/epochs)
                adaptive_alpha=adaptive_factors.to(logits.device) if adaptive_alpha is not None else None
                loss = adaptive_masked_loss(logits, y_gb, adaptive_alpha=adaptive_alpha)
                loss.backward()
                optimizer.step()
        
        agent['finetuned_global_state'] = model.state_dict() # Do save to global model so CFL can have its effects
        # Unfreeze for next federated run if needed
        for param in model.parameters():
            param.requires_grad_(True)

def get_bayes_pred(model, x_glob, x_loc=None, n_samples=50, debug_name=None, target_type='global'):
    # Returns mean prediction and "epistemic" uncertainties (std dev)
    # First check for any NaNs
    # What % is nan
    #print("Nan percent of x_glob: ", (torch.isnan(x_glob).all() / len(x_glob))*100)
    
    # The DataLoader/Dataset already provided the mask. Just remove the NaNs.
    x_glob_combined = torch.nan_to_num(x_glob, nan=0.0)

    if x_loc is not None:
        x_loc_combined = torch.nan_to_num(x_loc, nan=0.0)
    else:
        x_loc_combined = None

    """
        global_mask = (~torch.isnan(x_glob)).float()
        x_glob = torch.nan_to_num(x_glob, nan=0.0)
        x_glob_combined = torch.cat([x_glob, global_mask], dim=-1) # the data and the mask

        if x_loc is not None:
            m_loc = (~torch.isnan(x_loc)).float()
            x_loc = torch.nan_to_num(x_loc, nan=0.0)
            x_loc_combined = torch.cat([x_loc, m_loc], dim=-1)
        else:
            x_loc_combined = None
    """

    ##### PREVIOUS CODE WHEN USING IMPUTATION

    # if torch.isnan(x_glob).any() or (x_loc is not None and torch.isnan(x_loc).any()):
    #     # Return NaNs so the calling function knows the prediction failed
    #     return (
    #         np.full((x_glob.shape[0], model.num_outputs), np.nan), 
    #         np.full((x_glob.shape[0], model.num_outputs), np.nan)
    #     )
    
    # if x_loc is not None:
    #     #print("Nan percent of x_loc: ", (torch.isnan(x_loc).all() / len(x_loc))*100)

    # if torch.isnan(x_glob).any():
    #     print(f"WARNING: x_glob still has NaN after impute: {torch.isnan(x_glob).sum()} values")
    #     x_glob = torch.nan_to_num(x_glob, nan=0.0)
    # if x_loc is not None and torch.isnan(x_loc).any():
    #     print(f"WARNING: x_loc still has NaN after impute: {torch.isnan(x_loc).sum()} values")
    #     x_loc = torch.nan_to_num(x_loc, nan=0.0)
    
    model.eval()

    if isinstance(model, PFASLocal) and x_loc is None:
            # Create a zero-tensor matching x_glob batch size and the model's expected local dim
            device = x_glob.device
            # We divide by 2 because your Dataset doubles dim for masking [Value, Mask]
            local_dim = model.local_tail[0].in_features - model.global_chain_dim
            x_loc = torch.zeros((x_glob.size(0), local_dim), device=device)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples): # Number of samples
            if isinstance(model, PFASLocal):
                output = model(x_glob_combined, x_loc_combined)  # PFASLocal.forward(x_glob, x_loc)
            else:
                output = model(x_glob_combined)  # PFASGlobal.forward only takes x, nothing else
            
            # Check if output is tuple meaning Local model OR single tensor (global model)
            if isinstance(output, tuple):
                # output[0] is Global, output[1] is Local
                logits = output[0] if target_type == 'global' else output[1]
            else:
                logits = output

            if logits is None:
                print(f"WARNING: logits is None in get_bayes_pred ({debug_name})")
                return np.full((len(x_glob), 0), np.nan), np.full((len(x_glob), 0), np.nan)

            if torch.isnan(logits).any():
                print(f"WARNING: NaN logits in get_bayes_pred ({debug_name})")
                print(f"  logits shape: {tuple(logits.shape)}")
                print(f"  x_glob nan count: {torch.isnan(x_glob).sum().item()}")
                if x_loc is not None:
                    print(f"  x_loc nan count: {torch.isnan(x_loc).sum().item()}")
                return (
                    np.full(tuple(logits.shape), np.nan, dtype=float),
                    np.full(tuple(logits.shape), np.nan, dtype=float)
                )
            samples.append(torch.sigmoid(logits))
            
    samples = torch.stack(samples) # [num_samples, batch, num_pfas]
    means = samples.mean(dim=0)
    stds = samples.std(dim=0) # Measurement of uncertainty
    return means.cpu().numpy(), stds.cpu().numpy()

def adaptive_masked_loss(logits, targets, adaptive_alpha=None, pos_weight_cap=25.0):
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    num_analytes = targets.shape[1]
    pw = torch.ones(num_analytes, device=logits.device)
    
    for col in range(num_analytes):
        col_mask = mask[:, col]
        col_targets = targets[col_mask, col]
        if col_targets.numel() == 0:
            continue
        
        pos = col_targets.sum()
        neg = (1 - col_targets).sum()
        
        # Batch-based weight
        batch_pw = (neg / pos.clamp(min=1)).clamp(max=pos_weight_cap)
        
        # INTEGRATION: Multiply by the adaptive scaling factor if provided
        if adaptive_alpha is not None:
            pw[col] = batch_pw * adaptive_alpha[col]
        else:
            pw[col] = batch_pw

    # Standard BCE logic follows...
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.nan_to_num(0),
        pos_weight=pw.unsqueeze(0),
        reduction='none'
    )
    return losses[mask].mean()

def masked_loss(logits, targets, pos_weight_cap=10.0):
    # mask handles NaNs (missing tests)
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)        # If NO data is present in this batch, return a zero loss 

    # Per analyte pos_weight form the batch
    num_analytes = targets.shape[1]
    pw = torch.ones(num_analytes, device=logits.device) # cpu where the tensors are stored --> we can begin integraitng cpu and gpu
    for col in range(num_analytes):
        col_mask = mask[:, col] # one specific analyte per loop
        col_targets = targets[col_mask, col] # valid labels for this analyte only
        if col_targets.numel() == 0:
            continue # no data for this analyte
        pos = col_targets.sum()
        neg = (1 - col_targets).sum()
        pw[col] = (neg/pos.clamp(min=1)).clamp(max=pos_weight_cap)

    # Nan - do not include into the loss
    safe_targets = targets.clone()
    safe_targets[~mask] = 0.0 # nan mask
    safe_logits = logits.clone()
    safe_logits[~mask] = 0.0

    # positive weight (pw) [num_analytes] broadcasts across [batch, num_analytes]
    losses = F.binary_cross_entropy_with_logits(
        safe_logits, safe_targets,
        pos_weight=pw.unsqueeze(0),
        reduction='none'
    )
    return losses[mask].mean()

def create_personalized_tail(cluster_df, LOCAL_CHAIN, thres_ratio=0.05, min_detections=2):
    fin_local_chain = []
    available_cols = [c for c in LOCAL_CHAIN if c in cluster_df.columns]
    
    for analyte in available_cols:
        series = cluster_df[analyte]
        total_records = len(cluster_df)
        if total_records == 0:
            continue

        # How many rows were actually tested for this analyte
        tested = series.notna().sum()
        availability = tested / total_records

        # Of those tested, how many were detections
        detections = (series == 1.0).sum()

        # Require: analyte was tested in enough rows AND has minimum positive hits
        # This prevents training a personalized head on all-zero local labels
        if availability >= thres_ratio and detections >= min_detections:
            fin_local_chain.append(analyte)

    return fin_local_chain

if __name__=='__main__':
    # Unpickle the df_clustered - most updated
    filepth = 'data/df.pkl'
    with open(filepth, 'rb') as f:
        df = pickle.load(f)

        # --- COLUMNS DISCOVERY BLOCK ---
    print("\n--- HUNTING FOR KEYS ---")
    # List of keywords that Class B chemicals usually contain
    keywords = ['HFPO', 'ADONA', '11Cl', '9Cl', 'F53B', 'DONA', 'CIPF']

    # Find any column that matches these keywords (case-insensitive)
    found_cols = []
    for word in keywords:
        matches = [c for c in df.columns if word.lower() in c.lower().replace('_', '').replace('-', '')]
        if matches:
            print(f"Match for '{word}': {matches}")
            found_cols.extend(matches)

    # Also look for anything starting with 'PF' or ending in 'FTS'
    pfas_like = [c for c in df.columns if c.startswith('PF') or 'FTS' in c]
    print(f"General PFAS-like columns found: {pfas_like}")
    print("--- END HUNTING ---\n")
