
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SpatialGATAttention(nn.Module):
    """
    GAT-style spatial attention for replacing GWP
    """
    def __init__(self, in_channels, gso, num_heads=2, dropout=0.1, use_layer_norm=True):
        super(SpatialGATAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.gso = gso  # Graph structure (adjacency matrix)
        self.use_layer_norm = use_layer_norm
        
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
        x: (B, C, T, N) or (B, C, N, T) depending on usage
        For spatial attention, we expect (B, C, T, N)
        """
        B, C, T, N = x.shape
        
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
        
        # GAT-style attention computation
        # Compute attention scores for each head
        attention_scores = []
        for h in range(self.num_heads):
            q_h = Q[:, :, h, :]  # (B*T, N, head_dim)
            k_h = K[:, :, h, :]  # (B*T, N, head_dim)
            
            # GAT-style scoring
            scores_src = torch.sum(q_h * self.a_src[h], dim=-1, keepdim=True)  # (B*T, N, 1)
            scores_dst = torch.sum(k_h * self.a_dst[h], dim=-1, keepdim=True)  # (B*T, N, 1)
            
            # Broadcast and add: (B*T, N, 1) + (B*T, 1, N) -> (B*T, N, N)
            scores = self.leaky_relu(scores_src + scores_dst.transpose(-2, -1))
            
            # Apply graph structure mask (only attend to connected nodes)
            if self.gso is not None:
                # Create mask from adjacency matrix
                mask = (self.gso == 0).float() * -1e9
                scores = scores + mask.unsqueeze(0)  # Broadcast to (B*T, N, N)
            
            # Softmax attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attention_scores.append(attn_weights)
        
        # Apply attention and aggregate heads
        output_heads = []
        for h in range(self.num_heads):
            v_h = V[:, :, h, :]  # (B*T, N, head_dim)
            attn_h = attention_scores[h]  # (B*T, N, N)
            
            # Apply attention: (B*T, N, N) @ (B*T, N, head_dim) -> (B*T, N, head_dim)
            out_h = torch.bmm(attn_h, v_h)
            output_heads.append(out_h)
        
        # Concatenate heads: (B*T, N, C)
        output = torch.cat(output_heads, dim=-1)
        
        # Add residual connection
        # output = output + x_reshaped
        
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
    def __init__(self, in_channels, num_heads=2, dropout=0.1, use_layer_norm=True, causal=True):
        super(TemporalGATAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.causal = causal
        self.use_layer_norm = use_layer_norm
        
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
        
        # GAT-style attention computation
        attention_scores = []
        for h in range(self.num_heads):
            q_h = Q[:, :, h, :]  # (B*N, T, head_dim)
            k_h = K[:, :, h, :]  # (B*N, T, head_dim)
            
            # GAT-style scoring
            scores_src = torch.sum(q_h * self.a_src[h], dim=-1, keepdim=True)  # (B*N, T, 1)
            scores_dst = torch.sum(k_h * self.a_dst[h], dim=-1, keepdim=True)  # (B*N, T, 1)
            
            # Broadcast and add: (B*N, T, 1) + (B*N, 1, T) -> (B*N, T, T)
            scores = self.leaky_relu(scores_src + scores_dst.transpose(-2, -1))
            
            # Apply causal mask for temporal attention
            if self.causal:
                causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
                causal_mask = causal_mask.to(x.device)
                scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
            
            # Softmax attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attention_scores.append(attn_weights)
        
        # Apply attention and aggregate heads
        output_heads = []
        for h in range(self.num_heads):
            v_h = V[:, :, h, :]  # (B*N, T, head_dim)
            attn_h = attention_scores[h]  # (B*N, T, T)
            
            # Apply attention: (B*N, T, T) @ (B*N, T, head_dim) -> (B*N, T, head_dim)
            out_h = torch.bmm(attn_h, v_h)
            output_heads.append(out_h)
        
        # Concatenate heads: (B*N, T, C)
        output = torch.cat(output_heads, dim=-1)
        
        # Add residual connection
        # output = output + x_reshaped
        
        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        # Reshape back to original format: (B, C, T, N)
        output = output.reshape(B, N, T, C).permute(0, 3, 2, 1)
        
        return output


class STConvBlockParallelGAT(nn.Module):
    """
    ST-Conv block using GAT-style attention instead of WeightedPooling
    """
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels,
                 act_func, graph_conv_type, gso, bias, droprate,
                num_heads=2, gat_dropout=0.1):
        super(STConvBlockParallelGAT, self).__init__()
        
        # Original temporal and graph convolution layers
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0],
                                           n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0],
                                         channels[1], Ks, gso, bias)
        self.relu = nn.ReLU()
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2],
                                           n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.dropout = nn.Dropout(p=droprate)
        
        # GAT-style attention modules
        use_gat = True
        print("Number of GAT heads:", num_heads, "GAT dropout:", gat_dropout)
        self.spatial_gat = SpatialGATAttention(
            in_channels=channels[2],
            gso=gso,
            num_heads=num_heads,
            dropout = gat_dropout,
            use_layer_norm=True,
        )
        self.temporal_gat = TemporalGATAttention(
            in_channels=channels[2],
            num_heads=num_heads,
            dropout=gat_dropout,
            use_layer_norm=True,
            causal=True
        )
        
        self.use_gat = use_gat
    
    def forward(self, x):
        # Standard ST-Conv processing
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        
        if self.use_gat:
            # Apply GAT-style attention in parallel
            spatial_out = self.spatial_gat(x)
            temporal_out = self.temporal_gat(x)
            
            # Combine original output with GAT outputs
            x = x + spatial_out + temporal_out
        
        return x


# Helper classes (ensuring compatibility with existing code)
class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func, use_wp=False):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, 
                                          kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, 
                                          kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func
        self.use_wp = use_wp

    def _create_temporal_mask(self, kernel_size):
        """Create temporal causal mask"""
        K = kernel_size * kernel_size
        causal_positions = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                pos_idx = i * kernel_size + j
                if i <= kernel_size // 2:
                    if i < kernel_size // 2 or (i == kernel_size // 2 and j <= kernel_size // 2):
                        causal_positions.append(pos_idx)
        
        mask = torch.zeros(K, 1, 1, dtype=torch.bool)
        for pos in causal_positions:
            mask[pos, 0, 0] = True
        return mask

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
       
        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]
                        
            if self.act_func == 'glu':
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
            else:
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))
        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        
        return x


class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias, use_wp=False):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        self.use_wp = use_wp
        
        if self.graph_conv_type == 'cheb_graph_conv' or self.graph_conv_type == 'wp' or self.graph_conv_type == 'gat':
            self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        elif self.graph_conv_type == 'graph_conv':
            self.graph_conv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x):
        x_gc_in = self.align(x)
        
        if self.graph_conv_type == 'cheb_graph_conv' or self.graph_conv_type == 'wp' or self.graph_conv_type == 'gat':
            x_gc = self.cheb_graph_conv(x_gc_in)
        elif self.graph_conv_type == 'graph_conv':
            x_gc = self.graph_conv(x_gc_in)
        
        x_gc = x_gc.permute(0, 3, 1, 2) 
        x_gc_out = torch.add(x_gc, x_gc_in)
        return x_gc_out


# Required helper classes (simplified versions)
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        return x


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        if self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])
        
        x = torch.stack(x_list, dim=2)
        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        return cheb_graph_conv


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)
        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul
        return graph_conv

class OutputBlock(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        # x : B, C, T, N
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x