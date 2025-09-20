import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Implementazione del positional encoding per Transformer.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerPredictor(nn.Module):
    """
    Modello Transformer per predire il centroide di clustering 
    dato un punto latente dall'encoding del VAE.
    """
    
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        """
        Args:
            input_dim: Dimensione del vettore latente di input (dimensione VAE)
            output_dim: Dimensione del centroide di output 
            d_model: Dimensione del modello Transformer
            nhead: Numero di attention heads
            num_layers: Numero di layer del Transformer
            dropout: Probabilità di dropout
        """
        super(TransformerPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Embedding per trasformare l'input alla dimensione del modello
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head di output per predire il centroide
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Normalization layer
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass del modello.
        
        Args:
            x: Tensor di input [batch_size, input_dim] (vettori latenti VAE)
            
        Returns:
            Tensor di output [batch_size, output_dim] (centroidi predetti)
        """
        batch_size = x.size(0)
        
        # Embedding dell'input
        x = self.input_embedding(x)  # [batch_size, d_model]
        x = self.layer_norm(x)
        
        # Aggiungi dimensione di sequenza (per compatibilità con Transformer)
        # Trattiamo ogni punto latente come una sequenza di lunghezza 1
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Positional encoding
        x = x.transpose(0, 1)  # [1, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, 1, d_model]
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        
        # Rimuovi la dimensione di sequenza
        x = x.squeeze(1)  # [batch_size, d_model]
        
        # Output prediction
        output = self.output_head(x)  # [batch_size, output_dim]
        
        return output


class MultiHeadTransformerPredictor(nn.Module):
    """
    Versione avanzata del Transformer con multi-head attention più sofisticata.
    """
    
    def __init__(self, input_dim, output_dim, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(MultiHeadTransformerPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(3)
        ])
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)  # [batch_size, d_model]
        
        # Multi-scale features
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(x))
        
        # Concatena e riduce le features
        x = torch.stack(features, dim=1)  # [batch_size, 3, d_model]
        
        # Transformer processing
        x = self.transformer(x)  # [batch_size, 3, d_model]
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Output prediction
        output = self.output_layers(x)  # [batch_size, output_dim]
        
        return output


def create_transformer_model(input_dim, output_dim, model_type='simple', **kwargs):
    """
    Factory function per creare modelli Transformer.
    
    Args:
        input_dim: Dimensione dell'input (dimensione latente VAE)
        output_dim: Dimensione dell'output (dimensione centroide)
        model_type: Tipo di modello ('simple' o 'multi_head')
        **kwargs: Parametri aggiuntivi per il modello
        
    Returns:
        Modello Transformer inizializzato
    """
    if model_type == 'simple':
        return TransformerPredictor(input_dim, output_dim, **kwargs)
    elif model_type == 'multi_head':
        return MultiHeadTransformerPredictor(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")
