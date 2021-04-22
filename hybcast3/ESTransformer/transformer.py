import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder
from PE import generate_original_PE, generate_regular_PE

class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.
    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.
    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.
    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self, mc):

                #  d_input: int,
                #  d_model: int,
                #  d_output: int,
                #  q: int,
                #  v: int,
                #  h: int,
                #  N: int,
                #  attention_size: int = None,
                #  dropout: float = 0.3,
                #  chunk_mode: str = 'chunk',
                #  pe: str = None,
                #  pe_period: int = 24):a
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self.mc = mc
        self._d_model = mc.d_model

        self.layers_encoding = nn.ModuleList([Encoder(mc.d_model,
                                                      mc.q,
                                                      mc.v,
                                                      mc.h,
                                                      attention_size = mc.attention_size,
                                                      dropout = mc.dropout,
                                                      chunk_mode = mc.chunk_mode) for _ in range(mc.N)])
        self.layers_decoding = nn.ModuleList([Decoder(mc.d_model,
                                                      mc.q,
                                                      mc.v,
                                                      mc.h,
                                                      attention_size = mc.attention_size,
                                                      dropout = mc.dropout,
                                                      chunk_mode = mc.chunk_mode) for _ in range(mc.N)])

        self._embedding = nn.Linear(mc.d_input, mc.d_model)
        self._linear = nn.Linear(mc.d_model, mc.d_output)
      
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if mc.pe in pe_functions.keys():
            self._generate_PE = pe_functions[mc.pe]
            self._pe_period = mc.pe_period
        elif mc.pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{mc.pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer
        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.
        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).
        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        K = x.shape[1]
      
        # Embeddin module
        encoding = self._embedding(x)
      
        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = encoding

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)
        
        # Output module
        output = self._linear(decoding)
        # print('Linear shape:', output.shape)
        output = torch.sigmoid(output)
        return output