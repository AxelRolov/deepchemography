import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for SMILES reconstruction.
    
    Optimal architecture:
    - Bidirectional LSTM encoder with 2 layers, 128 units per direction (256 total)
    - Bottleneck layer: 256 units
    - LSTM decoder with 2 layers, 256 units per layer
    - Batch Normalization (crucial for high reconstruction accuracy)
    
    Expected performance: 99.71% reconstruction accuracy
    """
    
    def __init__(self, vocab, config):
        super().__init__()

        # Store config first (needed for initialization)
        self.config = config
        
        self.vocabulary = vocab
        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if config.q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        elif config.q_cell == 'lstm':
            self.encoder_rnn = nn.LSTM(
                d_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of ('gru', 'lstm')"
            )

        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        
        # For bidirectional LSTM with 2 layers, we concatenate h and c from both layers
        # Total: n_layers * 2_directions * (hidden_dim for h + hidden_dim for c)
        # But we only concatenate hidden states, not cell states in standard practice
        # Actually, per paper: "final cell states and hidden states of both layers are concatenated"
        encoder_output_dim = q_d_last * config.q_n_layers * 2  # *2 for h and c
        
        # Batch normalization for encoder output
        self.encoder_bn = nn.BatchNorm1d(encoder_output_dim) if config.use_batch_norm else None
        
        # Encoder to latent space (bottleneck)
        self.encoder_fc = nn.Linear(encoder_output_dim, config.d_z)

        # Decoder
        if config.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + config.d_z,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0
            )
        elif config.d_cell == 'lstm':
            self.decoder_rnn = nn.LSTM(
                d_emb + config.d_z,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of ('gru', 'lstm')"
            )

        # Batch normalization for decoder latent
        self.decoder_lat_bn = nn.BatchNorm1d(config.d_z) if config.use_batch_norm else None
        
        # Four parallel dense layers to initialize decoder LSTM states
        # As per paper: "decoded by four parallel dense layers to form the initial 
        # cell and hidden states for each LSTM layer of the decoder"
        self.decoder_h0_layer1 = nn.Linear(config.d_z, config.d_d_h)
        self.decoder_c0_layer1 = nn.Linear(config.d_z, config.d_d_h)
        if config.d_n_layers >= 2:
            self.decoder_h0_layer2 = nn.Linear(config.d_z, config.d_d_h)
            self.decoder_c0_layer2 = nn.Linear(config.d_z, config.d_d_h)
        
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)

        # Grouping the model's parameters
        encoder_modules = [self.encoder_rnn, self.encoder_fc]
        if self.encoder_bn is not None:
            encoder_modules.append(self.encoder_bn)
        self.encoder = nn.ModuleList(encoder_modules)
        
        decoder_modules = [
            self.decoder_rnn, 
            self.decoder_h0_layer1, 
            self.decoder_c0_layer1,
            self.decoder_fc
        ]
        if self.config.d_n_layers >= 2:
            decoder_modules.extend([self.decoder_h0_layer2, self.decoder_c0_layer2])
        if self.decoder_lat_bn is not None:
            decoder_modules.append(self.decoder_lat_bn)
        self.decoder = nn.ModuleList(decoder_modules)
        
        self.autoencoder = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )
        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def forward(self, x):
        """Forward pass through autoencoder

        :param x: list of tensors of longs, input sentence x
        :return: float, reconstruction loss
        """
        # Encoder: x -> z
        z = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        return recon_loss

    def forward_encoder(self, x):
        """Encoder step, encoding x to latent representation z
        
        As per paper: "final cell states and hidden states of both layers 
        are concatenated and passed to a dense layer"

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, latent vector z
        """
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        output = self.encoder_rnn(x, None)
        
        # Handle LSTM (returns tuple of (output, (h, c))) or GRU (returns (output, h))
        if isinstance(output[1], tuple):  # LSTM
            h_all_layers = output[1][0]  # Hidden states: (num_layers * num_directions, batch, hidden_size)
            c_all_layers = output[1][1]  # Cell states: (num_layers * num_directions, batch, hidden_size)
            
            # For bidirectional LSTM with 2 layers, h_all_layers has shape: (4, batch, hidden_size)
            # [layer0_forward, layer0_backward, layer1_forward, layer1_backward]
            # We concatenate all of them
            h_concat = torch.cat([h_all_layers[i] for i in range(h_all_layers.size(0))], dim=-1)
            c_concat = torch.cat([c_all_layers[i] for i in range(c_all_layers.size(0))], dim=-1)
            
            # Concatenate hidden and cell states from all layers
            encoder_output = torch.cat([h_concat, c_concat], dim=-1)
        else:  # GRU (only has hidden states)
            h_all_layers = output[1]
            encoder_output = torch.cat([h_all_layers[i] for i in range(h_all_layers.size(0))], dim=-1)

        # Apply batch normalization if enabled
        if self.encoder_bn is not None:
            encoder_output = self.encoder_bn(encoder_output)

        # Encode to latent space (bottleneck)
        z = self.encoder_fc(encoder_output)

        return z

    def forward_decoder(self, x, z):
        """Decoder step, reconstructing x from latent z
        
        As per paper: "decoded by four parallel dense layers to form the initial 
        cell and hidden states for each LSTM layer of the decoder"

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, reconstruction loss
        """
        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        # Apply batch normalization to latent vector if enabled
        z_norm = self.decoder_lat_bn(z) if self.decoder_lat_bn is not None else z
        
        # Use four parallel dense layers to initialize LSTM states
        # h_0 and c_0 for each layer
        h_0_list = [self.decoder_h0_layer1(z_norm)]
        c_0_list = [self.decoder_c0_layer1(z_norm)]
        
        if self.config.d_n_layers >= 2:
            h_0_list.append(self.decoder_h0_layer2(z_norm))
            c_0_list.append(self.decoder_c0_layer2(z_norm))
        
        # Stack to form initial states: (num_layers, batch, hidden_size)
        h_0 = torch.stack(h_0_list, dim=0)
        c_0 = torch.stack(c_0_list, dim=0)

        # Handle LSTM vs GRU initial state
        if isinstance(self.decoder_rnn, nn.LSTM):
            output, _ = self.decoder_rnn(x_input, (h_0, c_0))
        else:
            output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )

        return recon_loss

    def encode(self, x):
        """Encode input to latent representation (for inference)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, latent vector z
        """
        with torch.no_grad():
            return self.forward_encoder(x)

    def sample(self, n_batch, max_len=100, z=None, temp=1.0, decode='sample'):
        """Generate samples from latent representation

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None (if None, uses zero vector)
        :param temp: temperature of softmax (only used when decode='sample')
        :param decode: decoding mode - 'greedy' (default) for deterministic, 'sample' for stochastic
        :return: list of strings, generated sequences
        """
        with torch.no_grad():
            if z is None:
                z = torch.zeros(n_batch, self.encoder_fc.out_features,
                               device=self.device)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            # Apply batch normalization to latent vector if enabled
            z_norm = self.decoder_lat_bn(z) if self.decoder_lat_bn is not None else z
            
            # Use four parallel dense layers to initialize LSTM states
            h_0_list = [self.decoder_h0_layer1(z_norm)]
            c_0_list = [self.decoder_c0_layer1(z_norm)]
            
            if self.config.d_n_layers >= 2:
                h_0_list.append(self.decoder_h0_layer2(z_norm))
                c_0_list.append(self.decoder_c0_layer2(z_norm))
            
            h = torch.stack(h_0_list, dim=0)
            c = torch.stack(c_0_list, dim=0)
            
            # Handle LSTM vs GRU initial state
            if isinstance(self.decoder_rnn, nn.LSTM):
                state = (h, c)
            else:
                state = h
            
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool,
                                   device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, state = self.decoder_rnn(x_input, state)
                y = self.decoder_fc(o.squeeze(1))
                
                # Decode based on mode
                if decode == 'greedy':
                    w = y.argmax(dim=-1)  # Deterministic
                else:
                    y = F.softmax(y / temp, dim=-1)  # Stochastic
                    w = torch.multinomial(y, 1)[:, 0]
                
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]







