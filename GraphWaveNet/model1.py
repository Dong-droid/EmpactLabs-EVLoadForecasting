import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGATAttention(nn.Module):
    """
    GAT-style spatial attention for replacing GWP
    """
    def __init__(self, in_channels, gso=None, num_heads=2, dropout=0.3, use_layer_norm=True):
        super(SpatialGATAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.gso = gso  # Graph structure (adjacency matrix)
        self.use_layer_norm = use_layer_norm
        
        # Ensure head_dim is valid
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        # Multi-head attention parameters
        self.W_q = nn.Linear(in_channels, in_channels, bias=False)
        self.W_k = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, in_channels, bias=False)
        
        # GAT-style attention parameters
        self.a_src = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(in_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(self, x):
        """
        x: (B, C, T, N) 
        """
        B, C, T, N = x.shape
        device = x.device
        
        # Reshape for processing: (B*T, N, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B*T, N, C)
        
        # Generate Q, K, V
        Q = self.W_q(x_reshaped)  # (B*T, N, C)
        K = self.W_k(x_reshaped)  # (B*T, N, C)
        V = self.W_v(x_reshaped)  # (B*T, N, C)
        
        # Reshape for multi-head attention: (B*T, N, num_heads, head_dim)
        Q = Q.view(B*T, N, self.num_heads, self.head_dim)
        K = K.view(B*T, N, self.num_heads, self.head_dim)
        V = V.view(B*T, N, self.num_heads, self.head_dim)
        
        # Process each head separately to avoid memory issues
        output_heads = []
        for h in range(self.num_heads):
            q_h = Q[:, :, h, :]  # (B*T, N, head_dim)
            k_h = K[:, :, h, :]  # (B*T, N, head_dim)
            v_h = V[:, :, h, :]  # (B*T, N, head_dim)
            
            # GAT-style scoring - ensure tensors are on the same device
            a_src_h = self.a_src[h].to(device)  # (head_dim,)
            a_dst_h = self.a_dst[h].to(device)  # (head_dim,)
            
            scores_src = torch.sum(q_h * a_src_h, dim=-1, keepdim=True)  # (B*T, N, 1)
            scores_dst = torch.sum(k_h * a_dst_h, dim=-1, keepdim=True)  # (B*T, N, 1)
            
            # Broadcast and add: (B*T, N, 1) + (B*T, 1, N) -> (B*T, N, N)
            scores = self.leaky_relu(scores_src + scores_dst.transpose(-2, -1))
            
            # Apply graph structure mask (only attend to connected nodes) if available
            if self.gso is not None:
                # Ensure GSO is on the same device
                gso_device = self.gso.to(device) if isinstance(self.gso, torch.Tensor) else torch.tensor(self.gso, device=device)
                mask = (gso_device == 0).float() * -1e9
                scores = scores + mask.unsqueeze(0)  # Broadcast to (B*T, N, N)
            
            # Softmax attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention: (B*T, N, N) @ (B*T, N, head_dim) -> (B*T, N, head_dim)
            out_h = torch.bmm(attn_weights, v_h)
            output_heads.append(out_h)
        
        # Concatenate heads: (B*T, N, C)
        output = torch.cat(output_heads, dim=-1)
        
        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        # Reshape back to original format: (B, C, T, N)
        output = output.reshape(B, T, N, C).permute(0, 3, 1, 2)
        
        return output


class TemporalGATAttention(nn.Module):
    """
    GAT-style temporal attention for replacing TWP
    """
    def __init__(self, in_channels, num_heads=2, dropout=0.3, use_layer_norm=True, causal=True):
        super(TemporalGATAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.causal = causal
        self.use_layer_norm = use_layer_norm
        
        # Ensure head_dim is valid
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        # Multi-head attention parameters
        self.W_q = nn.Linear(in_channels, in_channels, bias=False)
        self.W_k = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, in_channels, bias=False)
        
        # GAT-style attention parameters
        self.a_src = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.FloatTensor(num_heads, self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(in_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(self, x):
        """
        x: (B, C, T, N)
        For temporal attention, we process along time dimension
        """
        B, C, T, N = x.shape
        device = x.device
        
        # Reshape for processing: (B*N, T, C)
        x_reshaped = x.permute(0, 3, 2, 1).reshape(B*N, T, C)
        
        # Generate Q, K, V
        Q = self.W_q(x_reshaped)  # (B*N, T, C)
        K = self.W_k(x_reshaped)  # (B*N, T, C)
        V = self.W_v(x_reshaped)  # (B*N, T, C)
        
        # Reshape for multi-head attention: (B*N, T, num_heads, head_dim)
        Q = Q.view(B*N, T, self.num_heads, self.head_dim)
        K = K.view(B*N, T, self.num_heads, self.head_dim)
        V = V.view(B*N, T, self.num_heads, self.head_dim)
        
        # Process each head separately to avoid memory issues
        output_heads = []
        for h in range(self.num_heads):
            q_h = Q[:, :, h, :]  # (B*N, T, head_dim)
            k_h = K[:, :, h, :]  # (B*N, T, head_dim)
            v_h = V[:, :, h, :]  # (B*N, T, head_dim)
            
            # GAT-style scoring - ensure tensors are on the same device
            a_src_h = self.a_src[h].to(device)  # (head_dim,)
            a_dst_h = self.a_dst[h].to(device)  # (head_dim,)
            
            scores_src = torch.sum(q_h * a_src_h, dim=-1, keepdim=True)  # (B*N, T, 1)
            scores_dst = torch.sum(k_h * a_dst_h, dim=-1, keepdim=True)  # (B*N, T, 1)
            
            # Broadcast and add: (B*N, T, 1) + (B*N, 1, T) -> (B*N, T, T)
            scores = self.leaky_relu(scores_src + scores_dst.transpose(-2, -1))
            
            # Apply causal mask for temporal attention
            if self.causal:
                causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
            
            # Softmax attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention: (B*N, T, T) @ (B*N, T, head_dim) -> (B*N, T, head_dim)
            out_h = torch.bmm(attn_weights, v_h)
            output_heads.append(out_h)
        
        # Concatenate heads: (B*N, T, C)
        output = torch.cat(output_heads, dim=-1)
        
        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        # Reshape back to original format: (B, C, T, N)
        output = output.reshape(B, N, T, C).permute(0, 3, 2, 1)
        
        return output

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnetgat(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2,num_heads=2, gat_dropout=0.3, layernorm=True):
        super(gwnetgat, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.sgat = nn.ModuleList()
        self.tgat = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, # from 1d to 2d
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, # 1d->2d
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                
                # Get GSO for spatial attention - handle case where supports might be None or empty
                gso = None
                if supports is not None and len(supports) > 0:
                    gso = supports[0]
                
                self.sgat.append(SpatialGATAttention(dilation_channels, num_heads=num_heads, dropout=gat_dropout, gso=gso, use_layer_norm=layernorm))
                self.tgat.append(TemporalGATAttention(dilation_channels, num_heads=num_heads, dropout=gat_dropout,use_layer_norm=layernorm, causal=True))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            
            # GAT attention - ensure proper tensor format
            x_swp = x.permute(0,1,3,2)  # (B, C, N, T) -> (B, C, T, N) for spatial
            x_twp = x.permute(0,1,3,2)  # (B, C, N, T) -> (B, C, T, N) for temporal
            
            # Apply GAT attention
            swp_out = self.sgat[i](x_swp).permute(0,1,3,2)  # (B, C, T, N) -> (B, C, N, T)
            twp_out = self.tgat[i](x_twp).permute(0,1,3,2)  # (B, C, T, N) -> (B, C, N, T)
            
            x = x + swp_out + twp_out
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
