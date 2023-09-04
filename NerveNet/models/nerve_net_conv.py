from typing import Optional, Tuple, Dict, List, Union
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import torch
from stable_baselines3.common.utils import get_device
from stable_baselines3.common import logger
from torch import Tensor
from torch_geometric.utils import add_self_loops, degree
from torch import nn
from torch.nn import Parameter 
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv, GATConv, SAGEConv, GCNConv, ChebConv, GINConv
from torch_geometric.nn.models import GraphSAGE, GIN
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes 
from torch_geometric.typing import Adj, OptTensor, PairTensor
from NerveNet.models.nerve_net_opt import dropout_manager, DynamicDropout, SimulatedAnnealingDropout
from NerveNet.models.utils import glorot, zeros, uniform
from torch_geometric.nn import Aggregation, MessagePassing

        
class NerveNetConv(MessagePassing):
    r"""

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 update_masks: Dict[str, Tuple[List[int], int]],
                 device: Union[torch.device, str] = "auto",
                 cached: bool = False,
                 bias: bool = True,
                 dropout_optimizer: SimulatedAnnealingDropout = None,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(NerveNetConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_masks = update_masks
        self.use_bias = bias
        self.cached = cached
        self.dropout_optmizier = dropout_optimizer
        self.device = get_device(device)

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.update_models_parameter = {}
        for group_name, _ in update_masks.items():
            self.update_models_parameter[group_name] = {}
            self.update_models_parameter[group_name]["weights"] = Parameter(
                torch.Tensor(in_channels, out_channels)).to(self.device)

            if self.use_bias:
                self.update_models_parameter[group_name]["bias"] = Parameter(
                    torch.Tensor(out_channels)).to(self.device)
            else:
                self.update_models_parameter[group_name]["bias"] = None

        self.reset_parameters()

    def reset_parameters(self):
        for _, params in self.update_models_parameter.items():
            glorot(params["weights"])
            zeros(params["bias"])
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                update_masks: dict = None) -> Tensor:
        """
        """
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # print(f"x {x} \n update_masks {update_masks}")
        if update_masks is not None:
            self.update_masks = update_masks
        if self.dropout_optmizier is not None:
            self.dropout_rate = self.dropout_optmizier.dropout_rate
        else:
            # A default value or some other mechanism in case the manager isn't provided
            self.dropout_rate = 0
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        out = self.propagate(edge_index, x=x, update_masks=update_masks,
                             size=None)
        return out

    def message(self, x_j: Tensor, update_masks: dict) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        embedding = torch.zeros(
            (*inputs.shape[:-1], self.out_channels)).to(self.device)

        for group_name, (update_mask, _) in self.update_masks.items():
            masked_inputs = inputs[:, update_mask]
            embedding[:, update_mask] = torch.matmul(
                masked_inputs,
                self.update_models_parameter[group_name]["weights"])
            if self.use_bias:
                embedding[:, update_mask] += self.update_models_parameter[group_name]["bias"]

        return embedding

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class NerveNetConv_v1(MessagePassing): ## try out this architecture to see results 
    """
        A message passing schema for nervenet with "Ego- and Neighbor-embedding Separation" 
        as described in "Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs"
    """

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """

        """
        # remove self loops, so we only aggregate neighbours and can add the ego features seperately
        edge_index, _ = remove_self_loops(edge_index)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        aggregated_neighbors = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                              size=None)

        # combine ego and aggregated neighbours
        out = torch.cat([x, aggregated_neighbors], dim=-1)

        return out

class NerveNetSage(SAGEConv):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 update_masks: Dict[str, Tuple[List[int], int]],
                 device: Union[torch.device, str] = "cpu",
                 aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
                 normalize: bool = False,
                 bias: bool =  True,
                 project: bool = False,
                 root_weight: bool = True,
                 **kwargs):
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        print(f"in{in_channels}")
        super(SAGEConv, self).__init__(aggr, **kwargs)
         
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_masks = update_masks
        self.device = device  
        self.project = project  
        self.bias = bias 
        self.root_weight = root_weight 
        self.normalize = normalize


        self.update_models_parameter = {}
        for group_name, _ in update_masks.items():
            self.update_models_parameter[group_name] = {}
            self.lin_l = nn.Linear(in_channels[0], out_channels, bias=bias)
            self.update_models_parameter[group_name]["lin_l"] = Parameter(
                torch.Tensor(self.lin_l.weight)).to(self.device)

            if self.project:
                self.lin = nn.Linear(in_channels[0], in_channels[0], bias=True)
                self.update_models_parameter[group_name]["lin"] = Parameter(
                    torch.Tensor(self.lin.weight)).to(self.device)
            else:
                self.update_models_parameter[group_name]["lin"] = None
            
            if self.root_weight:
                self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)
                self.update_models_parameter[group_name]["lin_r"] = Parameter(
                torch.Tensor(self.lin_r.weight)).to(self.device)
            else:
                self.update_models_parameter[group_name]["lin_r"] = None

        self.reset_parameters() 
    
    def reset_parameters(self):
        super().reset_parameters()
        for _, params in self.update_models_parameter.items():
            if self.project:
                self.lin.reset_parameters()
            self.lin_l.reset_parameters()
            if self.root_weight:
                self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                update_masks: dict = None) -> Tensor:
        
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])
       
        if update_masks is not None:
            self.update_masks = update_masks 
        
        out = self.propagate(edge_index, x=x, update_masks=update_masks,
                             size=None)
        out = self.lin_l(out)
        # print(f"out {out.shape}")
        ## dont know if the rest of this part is needed
        x_r = x[1]
    
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        
        return out 

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)


class NerveNetCheb(ChebConv):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 update_masks: Dict[str, Tuple[List[int], int]],
                 K: int, 
                 device: Union[torch.device, str] = "cpu",
                 normalization: str | None = 'sym', 
                 bias: bool = True, 
                 **kwargs
    ):
        kwargs.setdefault('aggr', 'add')

        super(ChebConv).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self.device = device

        self.update_models_parameter = {}
        for group_name, _ in update_masks.items():
            self.update_models_parameter[group_name] = {}
            self.lins = nn.Linear(in_channels, out_channels, bias=False)
            self.update_models_parameter[group_name]["lins"] = Parameter(
                torch.Tensor(self.lins.weight)).to(self.device)

            if self.bias:
                self.bias = Parameter(torch.Tensor(out_channels))
                self.update_models_parameter[group_name]["bias"] = self.bias.to(self.device)
            else: 
                self.update_models_parameter[group_name]["bias"] = None
            
class ConvDrop(MessagePassing):
    r"""

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
     ## allowing this sort of dropout to distringuish between soimilar neighbourhoods would be benefical for analysing 
     # slight modification in agent structure 
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 update_masks: Dict[str, Tuple[List[int], int]],
                 device: Union[torch.device, str] = "auto",
                 cached: bool = False,
                 bias: bool = True,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(ConvDrop, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_masks = update_masks
        self.use_bias = bias
        self.cached = cached
        self.device = get_device(device)

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.update_models_parameter = {}
        for group_name, _ in update_masks.items():
            self.update_models_parameter[group_name] = {}
            self.update_models_parameter[group_name]["weights"] = Parameter(
                torch.Tensor(in_channels, out_channels)).to(self.device)

            if self.use_bias:
                self.update_models_parameter[group_name]["bias"] = Parameter(
                    torch.Tensor(out_channels)).to(self.device)
            else:
                self.update_models_parameter[group_name]["bias"] = None

        # print(f"update_models_parameter {self.update_models_parameter[group_name]['weights'].shape}")
        self.reset_parameters()

    def reset_parameters(self):
        for _, params in self.update_models_parameter.items():
            glorot(params["weights"])
            zeros(params["bias"])
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                update_masks: dict = None) -> Tensor:
        """

        """

        gamma =  x.shape[1]
        p = 2 * 1 /(1+gamma)
        num_runs = gamma
        x = x.view(-1, x.size(-1)) 
        # print(f"x1{x.size()}") ## [1,9,12] [BatchSize, number of nodes, features per node]
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1), x.size(2)], device=x.device) * p).bool()
        x[drop] = torch.zeros_like(x[drop], device=x.device)
        del drop
        # run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)


        if update_masks is not None:
            self.update_masks = update_masks
        out = self.propagate(edge_index, x=x, update_masks=update_masks,
                             size=None)
        x = x.view(num_runs, -1, x.size(-1))
        out_mean = torch.mean(out, dim=0, keepdim=True)
        # print(f"out{out_mean.shape}") 

        return out_mean

    def message(self, x_j: Tensor, update_masks: dict) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        inputs = torch.mean(inputs, dim=0, keepdim=True)

        # print(f"inputs {inputs.shape}")
        embedding = torch.zeros(
            (*inputs.shape[:-1], self.out_channels)).to(self.device)
        for group_name, (update_mask, _) in self.update_masks.items():
            masked_inputs = inputs[:, update_mask]
            embedding[:, update_mask] = torch.matmul(
                masked_inputs,
                self.update_models_parameter[group_name]["weights"])
            if self.use_bias:
                embedding[:, update_mask] += self.update_models_parameter[group_name]["bias"]
        return embedding

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
   
class GCN(GCNConv):
    cached_edge_index: Optional[OptPairTensor]
    cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, update_masks: Dict[str, Tuple[List[int], int]],
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.update_models_parameter = {}
        for group_name, _ in update_masks.items():
            self.update_models_parameter[group_name] = {}
            self.update_models_parameter[group_name]["weights"] = Parameter(
                torch.Tensor(in_channels, out_channels)).to(self.device)

            if self.bias:
                self.update_models_parameter[group_name]["bias"] = Parameter(
                    torch.Tensor(out_channels)).to(self.device)
            else:
                self.update_models_parameter[group_name]["bias"] = None

        self.lin = nn.Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

class NerveNetConvGRU(GatedGraphConv):

    def __init__(self,
                 out_channels: int,
                 gnn_channels: int,
                 num_layers: int,
                 update_masks: Dict[str, Tuple[List[int], int]],
                 aggr: str = 'add',
                 bias: bool = True,
                 **kwargs):

        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.gnn_channels = gnn_channels
        self.num_layers = num_layers
        self.update_masks = update_masks

        self.weights = Parameter(
            Tensor(num_layers, out_channels, gnn_channels))

        self.rnns = dict()
        for group_name, (update_mask, _) in self.update_masks.items():
            self.rnns[group_name] = torch.nn.GRUCell(
                gnn_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weights)
        for group_name, (update_mask, _) in self.update_masks.items():
            self.rnns[group_name].reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        batch_size = x.shape[0]
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weights[i])
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               size=None)

            # use x_next to get arround the "no gradient for inplace operation" error we get when trying to directly update x
            x_next = torch.zeros_like(x)
            for group_name, (update_mask, _) in self.update_masks.items():

                x_group = x[:, update_mask]
                m_group = m[:, update_mask]

                x_next[:, update_mask] += self.rnns[group_name](m_group.view(-1, self.gnn_channels),
                                                                x_group.view(-1, self.out_channels)).view(batch_size, -1, self.out_channels)
            x = x_next

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)

class NerveNetConvGAT(GATConv):
    r"""
    Wrapper for the graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = nn.Linear(
                in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            ## look at his line in case x happens to be a tuple 
            self.lin_l = nn.Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = nn.Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()
    

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        batch_size = x.shape[0]
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            #assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            #assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(batch_size, -1, self.heads * self.out_channels)
        else:
            out = out.view(batch_size, -1).mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


import torch
from torch.nn import Parameter, functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj

class GCNConvGroups(GCNConv):
    def __init__(self, in_channels, out_channels, update_masks, device: Union[torch.device, str] = "auto",
 improved=False, cached=False, bias=True, normalize=True, dropout=0.5):
        super(GCNConvGroups, self).__init__(in_channels, out_channels, improved=improved, cached=cached, bias=bias, normalize=normalize)
        
        self.update_masks = update_masks
        self.dropout = dropout
        self.device = device 
        
        self.update_models_parameter = {}
        for group_name, _ in update_masks.items():
            self.update_models_parameter[group_name] = {}
            self.update_models_parameter[group_name]["weights"] = Parameter(torch.Tensor(in_channels, out_channels)).to(self.device)
            self.update_models_parameter[group_name]["bias"] = Parameter(torch.Tensor(out_channels)).to(self.device)
        
        self.reset_parameters_masked()

    def reset_parameters_masked(self):
        for _, params in self.update_models_parameter.items():
            torch.nn.init.xavier_uniform_(params["weights"])
            torch.nn.init.zeros_(params["bias"])

    def forward(self, x, edge_index, edge_weight=None):
        # Dropout on edges
        edge_index, _ = dropout_adj(edge_index, p=self.dropout, training=self.training)
        
        # Apply masks before propagation
        masked_x = torch.zeros_like(x)
        for group_name, (update_mask, _) in self.update_masks.items():
            masked_inputs = x[:, update_mask]
            masked_x[:, update_mask] = torch.matmul(masked_inputs, self.update_models_parameter[group_name]["weights"])
            masked_x[:, update_mask] += self.update_models_parameter[group_name]["bias"]
        
        # Original GCN propagation on masked features
        return super(GCNConvGroups, self).forward(masked_x, edge_index, edge_weight)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
