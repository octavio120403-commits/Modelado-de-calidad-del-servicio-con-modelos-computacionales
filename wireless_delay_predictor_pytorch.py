"""
=============================================================================
 Probabilistic Delay Forecasting in 5G Using Recurrent and
 Attention-Based Architectures
=============================================================================
 Implementación Python basada en:
   Mostafavi et al., "Probabilistic Delay Forecasting in 5G Using
   Recurrent and Attention-Based Architectures", arXiv:2503.15297v1

 Este script replica:
   1. Generación de datos sintéticos de delay 5G (patrones realistas)
   2. Tokenización de contexto de paquetes
   3. Modelo MLP (single-step, baseline)
   4. Modelo LSTM-SS (single-step con historia)
   5. Modelo LSTM (multi-step)
   6. Modelo Transformer encoder-decoder (multi-step)
   7. Mixture Density Network (MDN) con Gaussian Mixture Model (GMM)
   8. Entrenamiento con Negative Log-Likelihood (NLL)
   9. Evaluación: NLL, MAE, Coverage empírica
  10. Visualizaciones completas de resultados
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
import time
from copy import deepcopy

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1: GENERACIÓN DE DATOS SINTÉTICOS 5G
# ─────────────────────────────────────────────────────────────────────────────

def generate_5g_delay_data(
    n_packets: int = 15000,
    config: str = "reduced_gain",
    interarrival_ms: float = 50.0,
    seed: int = 42
) -> dict:
    """
    Genera datos sintéticos de delay de red 5G que simulan los dos
    escenarios del artículo:

    - 'reduced_gain':   MCS fluctúa entre 12-18, BLER ~10%, más retransmisiones
    - 'stable_high_gain': MCS fijo ~20, retransmisiones raras, delay más predecible

    El delay tiene dos componentes clave del artículo:
      a) Patrón diente de sierra (sawtooth) por desincronización del reloj
         de aplicación con el slot 5G TDD
      b) Saltos aleatorios por retransmisiones HARQ/RLC (~7.5ms cada uno)
    """
    rng = np.random.default_rng(seed)
    delays = []
    mcs_values = []
    harq_retx = []
    rlc_retx = []
    slots = []

    # Parámetros base según configuración
    if config == "reduced_gain":
        base_delay_ms = 18.0
        sawtooth_amplitude = 3.0
        harq_jump_ms = 7.5
        harq_prob = 0.10        # ~10% BLER
        rlc_prob = 0.005
        mcs_mean, mcs_std = 15, 2
        mcs_range = (12, 18)
    else:  # stable_high_gain
        base_delay_ms = 20.0
        sawtooth_amplitude = 2.5
        harq_jump_ms = 7.5
        harq_prob = 0.01
        rlc_prob = 0.0005
        mcs_mean, mcs_std = 20, 0.5
        mcs_range = (19, 21)

    # Frecuencia del diente de sierra: depende del interarrival
    sawtooth_period = max(10, int(interarrival_ms * 0.8))
    phase_offset = rng.integers(0, sawtooth_period)

    # Correlación temporal: estado de canal persiste
    channel_state = 0.0  # variable oculta de calidad de canal

    for i in range(n_packets):
        # MCS index (modulación y codificación)
        mcs = int(np.clip(rng.normal(mcs_mean, mcs_std),
                          mcs_range[0], mcs_range[1]))

        # Slot de llegada del paquete (0-9 en frame TDD de 10 slots)
        slot = int((i + phase_offset) % 10)

        # Delay base + diente de sierra (desalineación TDD)
        sawtooth = sawtooth_amplitude * ((i % sawtooth_period) / sawtooth_period)
        slot_penalty = 1.5 if slot in [4, 5, 9] else 0.0  # slots DL penalizan UL

        # Evolución del estado de canal (correlación serial)
        channel_state = 0.92 * channel_state + 0.08 * rng.standard_normal()
        channel_noise = 0.4 * channel_state

        # Retransmisiones HARQ
        n_harq = 0
        p_harq_adjusted = harq_prob * (1 + 0.5 * max(0, -channel_state))
        while rng.random() < min(p_harq_adjusted, 0.4) and n_harq < 3:
            n_harq += 1

        # Retransmisiones RLC (más raras, mayor penalización)
        n_rlc = 0
        if rng.random() < rlc_prob:
            n_rlc = 1

        # Delay total
        d = (base_delay_ms
             + sawtooth
             + slot_penalty
             + n_harq * harq_jump_ms
             + n_rlc * 25.0
             + channel_noise
             + rng.normal(0, 0.3))
        d = max(d, 5.0)  # mínimo físico

        delays.append(d)
        mcs_values.append(mcs)
        harq_retx.append(n_harq)
        rlc_retx.append(n_rlc)
        slots.append(slot)

    return {
        "delay_ms":        np.array(delays, dtype=np.float32),
        "mcs":             np.array(mcs_values, dtype=np.float32),
        "harq_retx":       np.array(harq_retx, dtype=np.float32),
        "rlc_retx":        np.array(rlc_retx, dtype=np.float32),
        "slot":            np.array(slots, dtype=np.float32),
        "packet_size_b":   np.full(n_packets, 200.0, dtype=np.float32),
        "interarrival_ms": np.full(n_packets, interarrival_ms, dtype=np.float32),
        "config":          config,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2: DATASET PYTORCH
# ─────────────────────────────────────────────────────────────────────────────

class PacketDelayDataset(Dataset):
    """
    Construye muestras de entrenamiento con ventana histórica H y
    horizonte futuro L, tal como describe el artículo (ecuación 6):
        D = { (X_m, y_m, ..., y_{m+L-1}) }_{m=1}^{N}
    """
    def __init__(self, data: dict, H: int = 20, L: int = 20,
                 normalize: bool = True, stats: dict = None):
        self.H = H
        self.L = L

        delays   = data["delay_ms"]
        mcs      = data["mcs"]
        harq     = data["harq_retx"]
        rlc      = data["rlc_retx"]
        slot     = data["slot"]
        pkt_size = data["packet_size_b"]
        iat      = data["interarrival_ms"]

        # Normalización Z-score para features continuas
        if stats is None:
            self.stats = {
                "delay_mean": delays.mean(), "delay_std": delays.std() + 1e-6,
                "mcs_mean":   mcs.mean(),    "mcs_std":   mcs.std() + 1e-6,
                "pkt_mean":   pkt_size.mean(),"pkt_std":  pkt_size.std() + 1e-6,
                "iat_mean":   iat.mean(),    "iat_std":   iat.std() + 1e-6,
            }
        else:
            self.stats = stats

        s = self.stats
        delay_n   = (delays   - s["delay_mean"]) / s["delay_std"]
        mcs_n     = (mcs      - s["mcs_mean"])   / s["mcs_std"]
        pkt_n     = (pkt_size - s["pkt_mean"])   / s["pkt_std"]
        iat_n     = (iat      - s["iat_mean"])   / s["iat_std"]

        # Vector de contexto de paquete: 6 features como en el artículo
        # [delay_norm, mcs_norm, harq_retx, rlc_retx, slot/9, pkt_norm, iat_norm]
        self.context = np.stack([
            delay_n,
            mcs_n,
            harq.astype(np.float32) / 3.0,
            rlc.astype(np.float32),
            slot / 9.0,
            pkt_n,
            iat_n,
        ], axis=1).astype(np.float32)   # shape: (N, 7)

        self.delay_raw = delays   # para cálculo de MAE en escala original
        self.delay_norm = delay_n

        N = len(delays)
        self.indices = list(range(H, N - L + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        m = self.indices[idx]
        # Contexto histórico: H pasos hasta m (inclusive)
        ctx = torch.tensor(self.context[m - self.H: m], dtype=torch.float32)
        # Targets futuros normalizados: L pasos desde m
        fut = torch.tensor(self.delay_norm[m: m + self.L], dtype=torch.float32)
        return ctx, fut


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3: TOKENIZADOR DE CONTEXTO (Feature Embedding)
# ─────────────────────────────────────────────────────────────────────────────

class DelayContextTokenizer(nn.Module):
    """
    Mapea el vector de contexto de paquete x_n ∈ R^M a un token
    u_n ∈ R^S mediante una función de embedding aprendible φ.

    Sigue la descripción de la Sección III-A del artículo:
    concatena embeddings individuales y los procesa con un MLP de 3 capas.
    """
    def __init__(self, input_dim: int = 7, token_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, x):
        # x: (batch, H, input_dim) → (batch, H, token_dim)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4: MIXTURE DENSITY NETWORK (MDN) con GMM
# ─────────────────────────────────────────────────────────────────────────────

class MDNHead(nn.Module):
    """
    Capa de salida que produce los parámetros de una Gaussian Mixture:
        - K pesos de mezcla (π_k)
        - K medias (μ_k)
        - K desviaciones estándar (σ_k)

    Para cada paso temporal futuro l ∈ {0, ..., L-1}, la distribución
    predicha es:  P(y) = Σ_k π_k · N(y; μ_k, σ_k²)
    """
    def __init__(self, in_dim: int, n_components: int = 8):
        super().__init__()
        self.K = n_components
        self.head = nn.Linear(in_dim, 3 * n_components)

    def forward(self, h):
        """
        h: (..., in_dim)
        Returns: (π, μ, σ) cada una de shape (..., K)
        """
        out = self.head(h)
        pi_raw, mu, log_sigma = out.split(self.K, dim=-1)
        pi = torch.softmax(pi_raw, dim=-1)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        return pi, mu, sigma


def gmm_nll(y, pi, mu, sigma):
    """
    Negative Log-Likelihood de una Gaussian Mixture Model.
    Ecuación (7) del artículo.

    y:     (batch, L) o (batch,)
    pi:    (batch, L, K) o (batch, K)
    mu:    (batch, L, K) o (batch, K)
    sigma: (batch, L, K) o (batch, K)
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)  # (batch, 1)
    if pi.dim() == 2:
        pi    = pi.unsqueeze(1)
        mu    = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)

    y = y.unsqueeze(-1)   # (batch, L, 1)
    # log N(y; μ_k, σ_k)
    log_prob = (
        -0.5 * ((y - mu) / sigma) ** 2
        - torch.log(sigma)
        - 0.5 * np.log(2 * np.pi)
    )
    # log Σ_k π_k · N(...)
    log_mix = torch.logsumexp(torch.log(pi + 1e-8) + log_prob, dim=-1)
    return -log_mix.mean()


def gmm_mean(pi, mu):
    """Media de la mezcla: E[Y] = Σ_k π_k · μ_k"""
    return (pi * mu).sum(dim=-1)


def gmm_percentile(pi, mu, sigma, p: float = 0.99, n_samples: int = 2000):
    """
    Percentil p de la mezcla mediante muestreo Monte Carlo.
    Devuelve tensor de shape (batch, L).
    """
    batch = pi.shape[0]
    L_    = pi.shape[1] if pi.dim() == 3 else 1
    if pi.dim() == 2:
        pi    = pi.unsqueeze(1)
        mu    = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)

    results = []
    for b in range(batch):
        row = []
        for l in range(L_):
            k_idx = torch.multinomial(pi[b, l], n_samples, replacement=True)
            samp  = mu[b, l, k_idx] + sigma[b, l, k_idx] * torch.randn(n_samples)
            row.append(torch.quantile(samp, p).item())
        results.append(row)
    return torch.tensor(results)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5: ARQUITECTURAS DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a. MLP (Single-step baseline) ──────────────────────────────────────────
class MLPPredictor(nn.Module):
    """
    Baseline de red feed-forward totalmente conectada.
    Sólo usa el token más reciente (single-step).
    Parámetros ≈ 37k como en el artículo.
    """
    def __init__(self, input_dim: int = 7, token_dim: int = 16,
                 n_components: int = 8, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = DelayContextTokenizer(input_dim, token_dim, dropout)
        self.net = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 4, token_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 4, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.mdn = MDNHead(token_dim, n_components)

    def forward(self, x):
        # x: (batch, H, input_dim) — sólo usa el último token
        tokens = self.tokenizer(x)          # (batch, H, S)
        last   = tokens[:, -1, :]           # (batch, S)
        h      = self.net(last)             # (batch, S)
        pi, mu, sigma = self.mdn(h)
        return pi, mu, sigma                # (batch, K) — single-step


# ── 5b. LSTM Single-Step ─────────────────────────────────────────────────────
class LSTMSingleStep(nn.Module):
    """
    LSTM que procesa toda la historia pero produce un único conjunto
    de parámetros GMM (single-step prediction).
    Parámetros ≈ 33k.
    """
    def __init__(self, input_dim: int = 7, token_dim: int = 16,
                 n_layers: int = 2, n_components: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.tokenizer = DelayContextTokenizer(input_dim, token_dim, dropout)
        self.lstm = nn.LSTM(token_dim, token_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.proj = nn.Linear(token_dim, token_dim)
        self.mdn  = MDNHead(token_dim, n_components)

    def forward(self, x):
        tokens = self.tokenizer(x)              # (batch, H, S)
        out, _ = self.lstm(tokens)              # (batch, H, S)
        h_last  = out[:, -1, :]                 # (batch, S)
        h_last  = torch.relu(self.proj(h_last))
        pi, mu, sigma = self.mdn(h_last)
        return pi, mu, sigma                    # (batch, K)


# ── 5c. LSTM Multi-Step ──────────────────────────────────────────────────────
class LSTMMultiStep(nn.Module):
    """
    LSTM con decodificación multi-paso usando padding tokens (Figura 4).
    Produce distribuciones para L pasos futuros simultáneamente.
    Parámetros ≈ 33k.
    """
    def __init__(self, input_dim: int = 7, token_dim: int = 16,
                 n_layers: int = 2, n_components: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.token_dim = token_dim
        self.n_layers  = n_layers
        self.tokenizer = DelayContextTokenizer(input_dim, token_dim, dropout)
        self.lstm      = nn.LSTM(token_dim, token_dim, n_layers,
                                 batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0)
        self.pad_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.proj      = nn.Linear(token_dim, token_dim)
        self.mdn       = MDNHead(token_dim, n_components)

    def forward(self, x, L: int = None):
        if L is None:
            L = x.shape[1]  # por defecto, L = H
        batch = x.shape[0]

        # Fase de codificación: procesar historia
        tokens  = self.tokenizer(x)                     # (B, H, S)
        _, (h, c) = self.lstm(tokens)                   # estado final

        # Fase de decodificación: L pasos con padding tokens
        pad = self.pad_token.expand(batch, L, -1)       # (B, L, S)
        out, _ = self.lstm(pad, (h, c))                 # (B, L, S)
        out  = torch.relu(self.proj(out))

        pi, mu, sigma = self.mdn(out)                   # (B, L, K)
        return pi, mu, sigma


# ── 5d. Transformer Encoder-Decoder Multi-Step ───────────────────────────────
class TransformerPredictor(nn.Module):
    """
    Transformer encoder-decoder para predicción multi-horizonte.
    Sigue exactamente la arquitectura de las Figuras 4-5 y ecuaciones (4)-(5).

    - Encoder: procesa la secuencia histórica U ∈ R^{H×S}
    - Decoder: genera Θ ∈ R^{L×V} usando query learnable Q ∈ R^{L×S}
    - MDN head convierte Θ a parámetros GMM
    Parámetros ≈ 78k.
    """
    def __init__(self, input_dim: int = 7, token_dim: int = 16,
                 n_heads: int = 4, n_enc_layers: int = 6,
                 n_dec_layers: int = 6, ff_dim: int = 512,
                 n_components: int = 8, dropout: float = 0.2,
                 max_seq_len: int = 200):
        super().__init__()
        self.token_dim = token_dim

        # Tokenizador compartido
        self.tokenizer = DelayContextTokenizer(input_dim, token_dim, dropout)

        # Positional encoding sinusoidal
        self.register_buffer("pos_enc",
                             self._make_pos_enc(max_seq_len, token_dim))

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # Decoder: query embeddings aprendibles (Figura 5)
        self.query_embed = nn.Embedding(200, token_dim)  # hasta 200 pasos futuros

        dec_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # Proyección final → parámetros MDN
        self.mdn = MDNHead(token_dim, n_components)

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        """Positional encoding sinusoidal estándar."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x, L: int = None):
        if L is None:
            L = x.shape[1]
        batch, H, _ = x.shape

        # Tokenizar + positional encoding
        tokens = self.tokenizer(x) + self.pos_enc[:, :H, :]    # (B, H, S)

        # Encoder (ecuación 4)
        memory = self.encoder(tokens)                           # (B, H, S)

        # Decoder queries Q^(0) (Figura 5)
        q_idx = torch.arange(L, device=x.device)
        Q = self.query_embed(q_idx).unsqueeze(0).expand(batch, -1, -1)
        Q = Q + self.pos_enc[:, :L, :]                         # (B, L, S)

        # Máscara causal para el decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device)

        # Decoder (ecuación 5)
        out = self.decoder(Q, memory, tgt_mask=tgt_mask)       # (B, L, S)

        # MDN head
        pi, mu, sigma = self.mdn(out)                          # (B, L, K)
        return pi, mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6: ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, n_epochs: int = 30,
                lr: float = 1e-3, device: str = "cpu",
                model_name: str = "Model", L: int = 20) -> dict:
    """
    Entrena un modelo minimizando la NLL (ecuación 7 del artículo).
    Retorna historial de pérdidas y tiempo de entrenamiento.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_val_nll = float("inf")
    best_state   = None
    history = {"train_nll": [], "val_nll": [], "train_time_s": []}

    print(f"\n{'='*55}")
    print(f"  Entrenando: {model_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros entrenables: {n_params:,}")
    print(f"{'='*55}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total_nll = 0.0

        for ctx, fut in train_loader:
            ctx, fut = ctx.to(device), fut.to(device)
            optimizer.zero_grad()

            pi, mu, sigma = model(ctx, L) if hasattr(model, 'query_embed') \
                            or isinstance(model, LSTMMultiStep) \
                            else (lambda r: r)(model(ctx))

            # Adaptar según single-step / multi-step
            if pi.dim() == 2:
                # single-step: replicar para todas las posiciones L
                nll = gmm_nll(fut.mean(dim=1), pi, mu, sigma)
            else:
                nll = gmm_nll(fut, pi, mu, sigma)

            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_nll += nll.item()

        scheduler.step()
        train_nll = total_nll / len(train_loader)
        dt = time.time() - t0

        # Validación
        model.eval()
        val_nll = 0.0
        with torch.no_grad():
            for ctx, fut in val_loader:
                ctx, fut = ctx.to(device), fut.to(device)
                pi, mu, sigma = model(ctx, L) if hasattr(model, 'query_embed') \
                                or isinstance(model, LSTMMultiStep) \
                                else (lambda r: r)(model(ctx))
                if pi.dim() == 2:
                    v = gmm_nll(fut.mean(dim=1), pi, mu, sigma)
                else:
                    v = gmm_nll(fut, pi, mu, sigma)
                val_nll += v.item()
        val_nll /= len(val_loader)

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["train_time_s"].append(dt)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state   = deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"Train NLL: {train_nll:.4f} | "
                  f"Val NLL: {val_nll:.4f} | "
                  f"Time: {dt:.2f}s")

    model.load_state_dict(best_state)
    print(f"  Mejor Val NLL: {best_val_nll:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7: EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, test_loader, device: str = "cpu",
                   L: int = 20, delay_std: float = 1.0,
                   delay_mean: float = 0.0) -> dict:
    """
    Evalúa el modelo en el conjunto de test.
    Métricas: NLL estandarizada, MAE (ms), Coverage empírica.
    """
    model.eval()
    all_nll = []
    all_mae = []
    all_preds_mean = []
    all_targets    = []
    all_pi, all_mu, all_sigma = [], [], []

    for ctx, fut in test_loader:
        ctx, fut = ctx.to(device), fut.to(device)

        pi, mu, sigma = model(ctx, L) if hasattr(model, 'query_embed') \
                        or isinstance(model, LSTMMultiStep) \
                        else (lambda r: r)(model(ctx))

        if pi.dim() == 2:
            nll = gmm_nll(fut.mean(dim=1), pi, mu, sigma).item()
            pred_mean = gmm_mean(pi, mu)           # (batch, K) → mean
            pred_mean_exp = pred_mean.unsqueeze(1).expand(-1, fut.shape[1])
            pi_exp    = pi.unsqueeze(1).expand(-1, fut.shape[1], -1)
            mu_exp    = mu.unsqueeze(1).expand(-1, fut.shape[1], -1)
            sigma_exp = sigma.unsqueeze(1).expand(-1, fut.shape[1], -1)
        else:
            nll = gmm_nll(fut, pi, mu, sigma).item()
            pred_mean = gmm_mean(pi, mu)           # (batch, L)
            pred_mean_exp = pred_mean
            pi_exp, mu_exp, sigma_exp = pi, mu, sigma

        all_nll.append(nll)

        # MAE en escala original (ms)
        mae = (torch.abs(pred_mean_exp - fut).mean().item()) * delay_std
        all_mae.append(mae)

        all_preds_mean.append(pred_mean_exp.cpu())
        all_targets.append(fut.cpu())
        all_pi.append(pi_exp.cpu())
        all_mu.append(mu_exp.cpu())
        all_sigma.append(sigma_exp.cpu())

    # Coverages empíricas a distintos niveles nominales
    targets_all = torch.cat(all_targets, 0)     # (N_test, L)
    pi_all      = torch.cat(all_pi, 0)
    mu_all      = torch.cat(all_mu, 0)
    sigma_all   = torch.cat(all_sigma, 0)

    coverages = {} #importante
    for level in [0.50, 0.70, 0.90, 0.99]:
        alpha = 1 - level
        lo_p  = alpha / 2
        hi_p  = 1 - alpha / 2
        cover_count = 0
        total       = 0
        for b in range(min(200, targets_all.shape[0])):  # muestra para velocidad
            for l in range(targets_all.shape[1]):
                k_s = torch.multinomial(pi_all[b, l], 1000, replacement=True)
                s   = mu_all[b, l, k_s] + sigma_all[b, l, k_s] * torch.randn(1000)
                lo  = torch.quantile(s, lo_p).item()
                hi  = torch.quantile(s, hi_p).item()
                y   = targets_all[b, l].item()
                if lo <= y <= hi:
                    cover_count += 1
                total += 1
        coverages[level] = cover_count / max(total, 1)

    return {
        "nll":      np.mean(all_nll),
        "mae_ms":   np.mean(all_mae),
        "coverages": coverages,
        "preds":    torch.cat(all_preds_mean, 0).numpy(),
        "targets":  targets_all.numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 8: PREDICCIÓN DE MUESTRA (para visualización)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_sample(model, ctx_tensor, L: int, device: str = "cpu",
                   n_mc: int = 3000):
    """
    Genera la distribución predictiva completa para un único ejemplo.
    Retorna cuantiles [50%, 70%, 90%, 99%] para cada paso futuro.
    """
    model.eval()
    ctx = ctx_tensor.unsqueeze(0).to(device)

    pi, mu, sigma = model(ctx, L) if hasattr(model, 'query_embed') \
                    or isinstance(model, LSTMMultiStep) \
                    else (lambda r: r)(model(ctx))

    if pi.dim() == 2:
        pi    = pi.unsqueeze(1).expand(-1, L, -1)
        mu    = mu.unsqueeze(1).expand(-1, L, -1)
        sigma = sigma.unsqueeze(1).expand(-1, L, -1)

    pi    = pi[0]     # (L, K)
    mu    = mu[0]
    sigma = sigma[0]

    means   = gmm_mean(pi, mu).numpy()        # (L,)
    quants  = {}
    for q in [0.005, 0.025, 0.15, 0.85, 0.975, 0.995]:
        vals = []
        for l in range(L):
            k_s = torch.multinomial(pi[l], n_mc, replacement=True)
            s   = mu[l, k_s] + sigma[l, k_s] * torch.randn(n_mc)
            vals.append(torch.quantile(s, q).item())
        quants[q] = np.array(vals)
    return means, quants


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 9: VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_sample(history_vals, future_vals, pred_mean,
                           quantiles, model_name: str,
                           delay_mean: float = 0.0,
                           delay_std:  float = 1.0,
                           ax=None):
    """
    Replica la Figura 8 del artículo: predicción con bandas de cobertura.
    """
    H = len(history_vals)
    L = len(future_vals)
    t_hist = np.arange(-H, 0)
    t_fut  = np.arange(0, L)

    # Desnormalizar
    def dn(x): return x * delay_std + delay_mean

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t_hist, dn(history_vals), color="#2196F3", lw=1.5, label="Delay History")
    ax.plot(t_fut,  dn(future_vals),  color="#FF5722", lw=1.5,
            linestyle="--", label="Ground Truth")
    ax.plot(t_fut,  dn(pred_mean),    color="#4CAF50", lw=2.0,
            linestyle=":", label="Prediction Mean")

    # Bandas de cobertura (de oscuro a claro)
    bands = [
        (0.005, 0.995, "#9C27B0", 0.15, "99% CI"),
        (0.025, 0.975, "#7B1FA2", 0.22, "95% CI"),
        (0.150, 0.850, "#AB47BC", 0.30, "70% CI"),
        (0.150, 0.850, "#CE93D8", 0.40, "50% CI"),
    ]
    for lo_q, hi_q, color, alpha, label in bands:
        ax.fill_between(t_fut,
                        dn(quantiles[lo_q]),
                        dn(quantiles[hi_q]),
                        color=color, alpha=alpha, label=label)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Packet Delay [ms]")
    ax.set_title(f"{model_name} — Probabilistic Delay Forecast")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_nll_vs_horizon(results_by_model: dict, title: str = ""):
    """Replica la Figura 9/11a: NLL vs horizonte de predicción."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"MLP": "o", "LSTM-SS": "^", "LSTM": "s", "Transformer": "D"}
    colors  = {"MLP": "#2196F3", "LSTM-SS": "#FF9800",
               "LSTM": "#4CAF50", "Transformer": "#F44336"}

    for name, data in results_by_model.items():
        horizons = sorted(data.keys())
        nlls     = [data[h]["nll"] for h in horizons]
        ax.plot(horizons, nlls,
                marker=markers.get(name, "x"),
                color=colors.get(name, "gray"),
                label=name, linewidth=2, markersize=7)

    ax.set_xlabel("Prediction Horizon (L)", fontsize=12)
    ax.set_ylabel("Standardized NLL", fontsize=12)
    ax.set_title(f"Model Accuracy vs Prediction Horizon\n{title}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mae_vs_horizon(results_by_model: dict, title: str = ""):
    """Replica la Figura 11b: MAE vs horizonte."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"MLP": "o", "LSTM-SS": "^", "LSTM": "s", "Transformer": "D"}
    colors  = {"MLP": "#2196F3", "LSTM-SS": "#FF9800",
               "LSTM": "#4CAF50", "Transformer": "#F44336"}

    for name, data in results_by_model.items():
        horizons = sorted(data.keys())
        maes     = [data[h]["mae_ms"] for h in horizons]
        ax.plot(horizons, maes,
                marker=markers.get(name, "x"),
                color=colors.get(name, "gray"),
                label=name, linewidth=2, markersize=7)

    ax.set_xlabel("Prediction Horizon (L)", fontsize=12)
    ax.set_ylabel("Delay MAE [ms]", fontsize=12)
    ax.set_title(f"Model MAE vs Prediction Horizon\n{title}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_coverage(coverage_dict: dict, title: str = ""):
    """Replica la Figura 14: cobertura empírica vs nominal."""
    fig, ax = plt.subplots(figsize=(7, 7))
    nominal = [0.50, 0.70, 0.90, 0.99]

    colors = {"MLP": "#2196F3", "LSTM-SS": "#FF9800",
              "LSTM": "#4CAF50", "Transformer": "#F44336"}
    markers = {"MLP": "o", "LSTM-SS": "^", "LSTM": "s", "Transformer": "D"}

    for name, covs in coverage_dict.items():
        empirical = [covs.get(n, n) for n in nominal]
        ax.plot(nominal, empirical,
                marker=markers.get(name, "x"),
                color=colors.get(name, "gray"),
                label=name, linewidth=2, markersize=8)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
    ax.set_xlabel("Target Coverage (Nominal)", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title(f"Calibration Plot\n{title}", fontsize=13)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_training_curves(histories: dict):
    """Curvas de pérdida de entrenamiento y validación."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"MLP": "#2196F3", "LSTM-SS": "#FF9800",
              "LSTM": "#4CAF50", "Transformer": "#F44336"}

    for name, hist in histories.items():
        c = colors.get(name, "gray")
        axes[0].plot(hist["train_nll"], color=c, label=f"{name} Train", lw=2)
        axes[0].plot(hist["val_nll"],   color=c, label=f"{name} Val",
                     lw=2, linestyle="--")

    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("NLL")
    axes[0].set_title("Training & Validation NLL")
    axes[0].legend(fontsize=8);  axes[0].grid(True, alpha=0.3)

    # Tiempo de entrenamiento por época
    for name, hist in histories.items():
        c = colors.get(name, "gray")
        axes[1].plot(hist["train_time_s"], color=c, label=name, lw=2)
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("Time per epoch [s]")
    axes[1].set_title("Training Time per Epoch")
    axes[1].legend(fontsize=9);  axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_data_overview(data: dict):
    """Visualización de los datos generados (delay + features)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"5G Synthetic Delay Data — Config: {data['config']}",
                 fontsize=14, fontweight="bold")
    n_show = min(500, len(data["delay_ms"]))

    ax = axes[0, 0]
    ax.plot(data["delay_ms"][:n_show], lw=1, color="#2196F3")
    ax.set_xlabel("Packet index");  ax.set_ylabel("Delay [ms]")
    ax.set_title("Delay Time Series (first 500 packets)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(data["delay_ms"], bins=60, color="#4CAF50", edgecolor="white",
            alpha=0.85, density=True)
    ax.set_xlabel("Delay [ms]");  ax.set_ylabel("Density")
    ax.set_title("Delay Distribution")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(data["mcs"][:n_show], data["delay_ms"][:n_show],
               alpha=0.3, s=8, color="#FF5722")
    ax.set_xlabel("MCS Index");  ax.set_ylabel("Delay [ms]")
    ax.set_title("Delay vs MCS Index")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    harq_vals = data["harq_retx"]
    for k in range(4):
        mask = harq_vals == k
        ax.hist(data["delay_ms"][mask], bins=40, alpha=0.6,
                label=f"HARQ={k}", density=True)
    ax.set_xlabel("Delay [ms]");  ax.set_ylabel("Density")
    ax.set_title("Delay Distribution by HARQ Retransmissions")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_gmm_example():
    """Replica la Figura 6: ejemplo de GMM con 2 componentes."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.linspace(5, 45, 500)

    # Componente 1
    mu1, sigma1, w1 = 18.0, 1.5, 0.65
    g1 = w1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))

    # Componente 2 (retransmisión HARQ)
    mu2, sigma2, w2 = 25.5, 2.0, 0.35
    g2 = w2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))

    mixture = g1 + g2

    ax.fill_between(x, g1,      alpha=0.5, color="#4CAF50",
                    label=f"Component 1 (μ={mu1}, σ={sigma1}, w={w1})")
    ax.fill_between(x, g2,      alpha=0.5, color="#2196F3",
                    label=f"Component 2 (μ={mu2}, σ={sigma2}, w={w2})")
    ax.plot(x, mixture,         color="#F44336", lw=2.5, label="GMM Mixture")

    ax.axvline(mu1, color="#4CAF50", linestyle="--", lw=1.5, alpha=0.8)
    ax.axvline(mu2, color="#2196F3", linestyle="--", lw=1.5, alpha=0.8)

    ax.set_xlabel("Packet Delay [ms]", fontsize=12)
    ax.set_ylabel("Density",           fontsize=12)
    ax.set_title("Gaussian Mixture Model — 2 Components\n"
                 "(Base delay + HARQ retransmission mode)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison_bar(results: dict, metric: str = "nll"):
    """Gráfico de barras comparativo de modelos."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names  = list(results.keys())
    values = [results[n][metric] for n in names]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    bars = ax.bar(names, values, color=colors[:len(names)],
                  edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(values),
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    label = "Standardized NLL" if metric == "nll" else "MAE [ms]"
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f"Model Comparison — {label}", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 10: PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT = "./resultados2/"

    import os
    os.makedirs(OUT, exist_ok=True)

    print("\n" + "=" * 65)
    print("  PROBABILISTIC DELAY FORECASTING IN 5G")
    print("=" * 65)

    # ── Configuración ───────────────────────────────────────────────────
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
    H        = 20       # ventana histórica
    L        = 20       # horizonte de predicción
    BATCH    = 64
    EPOCHS   = 25
    N_DATA   = 12000    # paquetes totales por configuración
    LR       = 8e-4
    N_COMP   = 8        # componentes GMM
    TOKEN_DIM = 16      # dimensión del token S

    print(f"\n  Dispositivo: {DEVICE}")
    print(f"  H={H}, L={L}, Epochs={EPOCHS}, BatchSize={BATCH}")
    print(f"  Token dim={TOKEN_DIM}, GMM components={N_COMP}")

    # ── 1. Generar datos ─────────────────────────────────────────────────
    print("\n[1/6] Generando datos sintéticos de delay 5G...")
    data_rg = generate_5g_delay_data(N_DATA, config="reduced_gain",
                                     interarrival_ms=50.0)
    data_hg = generate_5g_delay_data(N_DATA, config="stable_high_gain",
                                     interarrival_ms=50.0)

    fig_data = plot_data_overview(data_rg)
    plt.savefig(OUT + "fig1_data_overview.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_data)
    print("  ✓ fig1_data_overview.png guardado")

    fig_gmm = plot_gmm_example()
    plt.savefig(OUT + "fig2_gmm_example.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_gmm)
    print("  ✓ fig2_gmm_example.png guardado")

    # ── 2. Preparar datasets ─────────────────────────────────────────────
    print("\n[2/6] Preparando datasets...")
    ds_rg   = PacketDelayDataset(data_rg, H=H, L=L)
    stats   = ds_rg.stats

    n       = len(ds_rg)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds_rg, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=0)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ── 3. Instanciar modelos ────────────────────────────────────────────
    print("\n[3/6] Instanciando modelos...")
    models = {
        "MLP": MLPPredictor(
            input_dim=7, token_dim=TOKEN_DIM, n_components=N_COMP),
        "LSTM-SS": LSTMSingleStep(
            input_dim=7, token_dim=TOKEN_DIM, n_components=N_COMP),
        "LSTM": LSTMMultiStep(
            input_dim=7, token_dim=TOKEN_DIM, n_components=N_COMP),
        "Transformer": TransformerPredictor(
            input_dim=7, token_dim=TOKEN_DIM, n_heads=4,
            n_enc_layers=4, n_dec_layers=4, ff_dim=128,
            n_components=N_COMP),
    }
    for name, m in models.items():
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:12s}: {n_p:,} parámetros")

    # ── 4. Entrenamiento ─────────────────────────────────────────────────
    print("\n[4/6] Entrenando modelos...")
    histories = {}
    for name, model in models.items():
        hist = train_model(model, train_loader, val_loader,
                           n_epochs=EPOCHS, lr=LR, device=DEVICE,
                           model_name=name, L=L)
        histories[name] = hist

    fig_curves = plot_training_curves(histories)
    plt.savefig(OUT + "fig3_training_curves.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_curves)
    print("  ✓ fig3_training_curves.png guardado")

    # ── 5. Evaluación base (L=20) ─────────────────────────────────────
    print("\n[5/6] Evaluando modelos en test set...")
    d_mean = float(stats["delay_mean"])
    d_std  = float(stats["delay_std"])

    results_base = {}
    coverage_dict = {}
    for name, model in models.items():
        res = evaluate_model(model, test_loader, device=DEVICE,
                             L=L, delay_std=d_std, delay_mean=d_mean)
        results_base[name] = res
        coverage_dict[name] = res["coverages"]
        print(f"  {name:12s} | NLL: {res['nll']:.4f} | "
              f"MAE: {res['mae_ms']:.2f} ms | "
              f"Cov@99%: {res['coverages'].get(0.99, 0):.3f}")

    # Gráfico de barras comparativo
    fig_bar_nll = plot_model_comparison_bar(
        {n: {"nll": results_base[n]["nll"]} for n in results_base},
        metric="nll"
    )
    plt.savefig(OUT + "fig4_model_comparison_nll.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_bar_nll)

    fig_bar_mae = plot_model_comparison_bar(
        {n: {"mae": results_base[n]["mae_ms"]} for n in results_base},
        metric="mae"
    )
    plt.savefig(OUT + "fig5_model_comparison_mae.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_bar_mae)

    fig_cov = plot_coverage(coverage_dict,
                            title="Reduced Gain Config, L=20, N_train=10k")
    plt.savefig(OUT + "fig6_coverage_plot.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_cov)
    print("  ✓ fig4-6 guardados")

    # ── 5b. NLL vs horizonte (L = 10, 20, 50, 100 simulado) ──────────────
    # Usamos los modelos ya entrenados con L=20 y evaluamos sobre sub-horizontes
    print("\n  Evaluando NLL vs horizonte de predicción...")
    horizons    = [5, 10, 15, 20]
    nll_by_h    = {name: {} for name in models}
    mae_by_h    = {name: {} for name in models}

    for l_eval in horizons:
        ds_tmp = PacketDelayDataset(data_rg, H=H, L=l_eval, stats=stats)
        n_tmp  = len(ds_tmp)
        n_tst_tmp = int(0.15 * n_tmp)
        _, _, tst_tmp = torch.utils.data.random_split(
            ds_tmp, [n_tmp - 2 * n_tst_tmp, n_tst_tmp, n_tst_tmp],
            generator=torch.Generator().manual_seed(42))
        loader_tmp = DataLoader(tst_tmp, batch_size=BATCH, shuffle=False)

        for name, model in models.items():
            r = evaluate_model(model, loader_tmp, device=DEVICE,
                               L=l_eval, delay_std=d_std, delay_mean=d_mean)
            nll_by_h[name][l_eval] = {"nll": r["nll"]}
            mae_by_h[name][l_eval] = {"mae_ms": r["mae_ms"]}

    fig_nll_h = plot_nll_vs_horizon(nll_by_h,
                                    "Reduced Gain Config")
    plt.savefig(OUT + "fig7_nll_vs_horizon.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_nll_h)

    fig_mae_h = plot_mae_vs_horizon(mae_by_h,
                                    "Reduced Gain Config")
    plt.savefig(OUT + "fig8_mae_vs_horizon.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_mae_h)
    print("  ✓ fig7-8 guardados")

    # ── 6. Visualización de predicciones (Figura 8 del artículo) ──────────
    print("\n[6/6] Generando visualizaciones de predicciones...")

    # Tomar una muestra del test set
    sample_idx = 0
    sample_ctx_norm, sample_fut_norm = test_ds[sample_idx]

    fig_pred, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig_pred.suptitle(
        "Probabilistic Delay Forecasting — Sample Prediction\n"
        "(Replica of Figure 8 from Mostafavi et al.)",
        fontsize=13, fontweight="bold"
    )

    for ax, (name, model) in zip(axes, [
        ("Transformer (Multi-Step)", models["Transformer"]),
        ("MLP (Single-Step)", models["MLP"]),
    ]):
        pred_mean, quants = predict_sample(
            model, sample_ctx_norm, L=L, device=DEVICE)

        # Reconstruir historia: último valor de delay de cada token
        hist_vals = sample_ctx_norm[:, 0].numpy()  # delay normalizado

        plot_prediction_sample(
            hist_vals, sample_fut_norm.numpy(),
            pred_mean, quants,
            model_name=name,
            delay_mean=d_mean, delay_std=d_std,
            ax=ax
        )

    plt.tight_layout()
    plt.savefig(OUT + "fig9_prediction_sample.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig_pred)
    print("  ✓ fig9_prediction_sample.png guardado")

    # ── Resumen final en tabla ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTADOS FINALES (Test Set, Reduced Gain, L=20)")
    print("=" * 65)
    print(f"  {'Modelo':<14} | {'NLL':>8} | {'MAE (ms)':>10} | "
          f"{'Cov@50%':>8} | {'Cov@99%':>8}")
    print("  " + "-" * 60)
    for name in ["MLP", "LSTM-SS", "LSTM", "Transformer"]:
        r = results_base[name]
        print(f"  {name:<14} | {r['nll']:>8.4f} | "
              f"{r['mae_ms']:>10.3f} | "
              f"{r['coverages'].get(0.50, 0):>8.3f} | "
              f"{r['coverages'].get(0.99, 0):>8.3f}")
    print("=" * 65)

    print("\n  Todos los gráficos guardados en ./resultados2/")
    print("  Archivos generados:")
    files = [
        "fig1_data_overview.png        — Datos 5G generados",
        "fig2_gmm_example.png          — Ejemplo GMM (Figura 6 del artículo)",
        "fig3_training_curves.png      — Curvas de entrenamiento",
        "fig4_model_comparison_nll.png — Comparación NLL (barras)",
        "fig5_model_comparison_mae.png — Comparación MAE (barras)",
        "fig6_coverage_plot.png        — Calibración empírica (Figura 14)",
        "fig7_nll_vs_horizon.png       — NLL vs horizonte (Figura 9/11a)",
        "fig8_mae_vs_horizon.png       — MAE vs horizonte (Figura 11b)",
        "fig9_prediction_sample.png    — Muestra predicción (Figura 8)",
    ]
    for f in files:
        print(f"    • {f}")


if __name__ == "__main__":
    main()
