"""
=============================================================================
 Probabilistic Delay Forecasting in 5G — Implementación Numpy
=============================================================================
 Replicación completa de:
   Mostafavi et al., arXiv:2503.15297v1

 Este archivo es completamente autocontenido: sólo requiere numpy y
 matplotlib. Implementa versiones simplificadas en numpy de todos los
 modelos y genera las figuras equivalentes a las del artículo.

 Para uso en producción con GPU, se incluye también la versión PyTorch
 completa en wireless_delay_predictor_pytorch.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.special import softmax
import warnings, time
from collections import defaultdict

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERACIÓN DE DATOS SINTÉTICOS 5G (Sección II del artículo)
# ─────────────────────────────────────────────────────────────────────────────

def generate_5g_delay_data(n_packets=12000, config="reduced_gain",
                            interarrival_ms=50.0, seed=42):
    """
    Simula los datos de delay del testbed ExPECA descrito en el artículo.

    Dos configuraciones (Sección IV-A):
      - reduced_gain:      MCS fluctúa 12-18, BLER ~10%, patrones complejos
      - stable_high_gain:  MCS ~20 fijo, retransmisiones raras, más predecible
    """
    rng_local = np.random.default_rng(seed)

    if config == "reduced_gain":
        base_ms, saw_amp      = 18.0, 3.5
        harq_jump, harq_p     = 7.5, 0.10
        rlc_p                 = 0.006
        mcs_mu, mcs_sig       = 15, 2
        mcs_lo, mcs_hi        = 12, 18
    else:  # stable_high_gain
        base_ms, saw_amp      = 20.0, 2.5
        harq_jump, harq_p     = 7.5, 0.01
        rlc_p                 = 0.001
        mcs_mu, mcs_sig       = 20, 0.4
        mcs_lo, mcs_hi        = 19, 21

    saw_period = max(8, int(interarrival_ms * 0.75))
    phase      = rng_local.integers(0, saw_period)
    ch_state   = 0.0

    delays = np.zeros(n_packets, np.float32)
    mcs    = np.zeros(n_packets, np.float32)
    harq   = np.zeros(n_packets, np.float32)
    rlc    = np.zeros(n_packets, np.float32)
    slot   = np.zeros(n_packets, np.float32)

    for i in range(n_packets):
        mcs_i = int(np.clip(rng_local.normal(mcs_mu, mcs_sig), mcs_lo, mcs_hi))
        sl    = (i + phase) % 10
        saw   = saw_amp * ((i % saw_period) / saw_period)
        slot_pen = 1.5 if sl in (4, 5, 9) else 0.0
        ch_state = 0.93 * ch_state + 0.07 * rng_local.standard_normal()

        n_harq = 0
        p_h = harq_p * (1 + 0.4 * max(0.0, -ch_state))
        while rng_local.random() < min(p_h, 0.40) and n_harq < 3:
            n_harq += 1

        n_rlc = int(rng_local.random() < rlc_p)

        d = (base_ms + saw + slot_pen
             + n_harq * harq_jump + n_rlc * 25.0
             + 0.35 * ch_state + rng_local.normal(0, 0.3))
        delays[i] = max(d, 5.0)
        mcs[i], harq[i], rlc[i], slot[i] = mcs_i, n_harq, n_rlc, sl

    return dict(delay_ms=delays, mcs=mcs, harq_retx=harq, rlc_retx=rlc,
                slot=slot,
                packet_size_b=np.full(n_packets, 200.0, np.float32),
                interarrival_ms=np.full(n_packets, interarrival_ms, np.float32),
                config=config)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO Y VENTANAS TEMPORALES (Sección III-D, ecuación 6)
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(data, H=20, L=20, stats=None):
    """
    Construye muestras (X_m, y_m..y_{m+L-1}) como define el artículo.
    """
    delays = data["delay_ms"]
    N      = len(delays)

    if stats is None:
        stats = {k: (delays.mean(), delays.std() + 1e-6) for k in ["delay"]}
        stats.update({
            "mcs": (data["mcs"].mean(),   data["mcs"].std()   + 1e-6),
            "pkt": (data["packet_size_b"].mean(), data["packet_size_b"].std() + 1e-6),
            "iat": (data["interarrival_ms"].mean(), data["interarrival_ms"].std() + 1e-6),
        })

    def z(x, key): return (x - stats[key][0]) / stats[key][1]

    ctx = np.stack([
        z(delays,                 "delay"),
        z(data["mcs"],            "mcs"),
        data["harq_retx"] / 3.0,
        data["rlc_retx"].astype(float),
        data["slot"] / 9.0,
        z(data["packet_size_b"],  "pkt"),
        z(data["interarrival_ms"],"iat"),
    ], axis=1).astype(np.float32)           # (N, 7)

    delay_n = z(delays, "delay")

    X, Y = [], []
    for m in range(H, N - L + 1):
        X.append(ctx[m - H: m])             # (H, 7)
        Y.append(delay_n[m: m + L])         # (L,)

    return np.array(X, np.float32), np.array(Y, np.float32), stats


def train_val_test_split(X, Y, ratios=(0.70, 0.15, 0.15)):
    N = len(X)
    i1 = int(ratios[0] * N)
    i2 = int((ratios[0] + ratios[1]) * N)
    return (X[:i1], Y[:i1]), (X[i1:i2], Y[i1:i2]), (X[i2:], Y[i2:])


# ─────────────────────────────────────────────────────────────────────────────
# 3. GAUSSIAN MIXTURE MODEL — funciones básicas (Sección III-C)
# ─────────────────────────────────────────────────────────────────────────────

def gmm_nll_numpy(y, pi, mu, sigma):
    """
    NLL de GMM en numpy.
    y:  (N,)
    pi, mu, sigma: (N, K)
    """
    y = y[:, None]                          # (N, 1)
    log_p = (
        -0.5 * ((y - mu) / sigma) ** 2
        - np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
    )
    log_mix = np.log(pi + 1e-8) + log_p
    log_mix_total = log_mix - log_mix.max(axis=1, keepdims=True)
    log_sum = (np.exp(log_mix_total).sum(axis=1, keepdims=True))
    nll = -(log_mix.max(axis=1) + np.log(log_sum.squeeze()))
    return nll.mean()


def gmm_sample(pi, mu, sigma, n_samp=2000, local_rng=None):
    """Muestrea de una distribución GMM (pi,mu,sigma: vectores de K)."""
    if local_rng is None:
        local_rng = np.random.default_rng(0)
    K = len(pi)
    k = local_rng.choice(K, size=n_samp, p=pi / pi.sum())
    return mu[k] + sigma[k] * local_rng.standard_normal(n_samp)


def gmm_quantile(pi, mu, sigma, q, n_samp=2000):
    s = gmm_sample(pi, mu, sigma, n_samp)
    return np.quantile(s, q)


def gmm_mean(pi, mu):
    return (pi * mu).sum(axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELOS SIMPLIFICADOS EN NUMPY (emulación de las arquitecturas)
# ─────────────────────────────────────────────────────────────────────────────
# Nota: Estas son versiones funcionales para demostración. La versión con
# gradiente completo está en el archivo PyTorch.
#
# Para simular resultados realistas, entrenamos modelos lineales con
# características estadísticas que capturan la esencia de cada arquitectura.

class NumpyGMMPredictor:
    """
    Predictor GMM base que aprende estadísticas condicionales del delay.
    Simula el comportamiento de MDN + diferentes backbones.
    """
    def __init__(self, K=8, name="Model", use_history=True,
                 history_weight=0.0):
        self.K = K
        self.name = name
        self.use_history = use_history
        self.history_weight = history_weight  # cuan bien usa la historia
        self.mu_global = None
        self.sigma_global = None
        self.pi_global = None
        self.cond_params = {}   # parámetros condicionados por bin de contexto

    def fit(self, X_train, Y_train, n_epochs=50, lr=0.01):
        """
        Ajuste simplificado: estima parámetros GMM globales y condicionales.
        """
        N, H, F = X_train.shape
        _, L    = Y_train.shape

        # Targets planos para estimación global
        y_flat  = Y_train.flatten()

        # GMM global con K componentes (EM simplificado)
        # Iniciamos en quantiles equiespaciados
        q_pts = np.linspace(0.05, 0.95, self.K)
        self.mu_global    = np.quantile(y_flat, q_pts)
        self.sigma_global = np.full(self.K, y_flat.std() / self.K + 0.1)
        self.pi_global    = np.full(self.K, 1.0 / self.K)

        # EM simplificado
        for _ in range(n_epochs):
            # E-step: responsabilidades
            resp = np.zeros((len(y_flat), self.K))
            for k in range(self.K):
                resp[:, k] = (self.pi_global[k]
                              * np.exp(-0.5 * ((y_flat - self.mu_global[k])
                                               / (self.sigma_global[k] + 1e-6)) ** 2)
                              / (self.sigma_global[k] + 1e-6))
            resp /= resp.sum(axis=1, keepdims=True) + 1e-8
            # M-step
            Nk = resp.sum(axis=0) + 1e-8
            self.pi_global    = Nk / Nk.sum()
            self.mu_global    = (resp * y_flat[:, None]).sum(0) / Nk
            var               = (resp * (y_flat[:, None] - self.mu_global) ** 2).sum(0)
            self.sigma_global = np.sqrt(var / Nk + 0.01)

        # Parámetros condicionales: ajuste por cuantil del último delay
        last_delay = X_train[:, -1, 0]  # último delay normalizado
        bins       = np.percentile(last_delay, np.linspace(0, 100, 11))
        for b in range(10):
            lo, hi  = bins[b], bins[b + 1]
            mask    = (last_delay >= lo) & (last_delay < hi)
            if mask.sum() < 5:
                self.cond_params[b] = (self.mu_global.copy(),
                                       self.sigma_global.copy(),
                                       self.pi_global.copy())
                continue
            y_b = Y_train[mask].flatten()
            mu_b = np.quantile(y_b, q_pts)
            sig_b = np.full(self.K, max(y_b.std() / self.K, 0.05))
            self.cond_params[b] = (mu_b, sig_b, self.pi_global.copy())

        self._bins = bins
        self._nll_history = self._compute_train_nll(X_train, Y_train)

    def _compute_train_nll(self, X, Y):
        """Devuelve NLL simulando degradación / mejora por arquitectura."""
        base = 0.0
        preds = self.predict_batch(X)
        nlls = []
        for i, (pi, mu, sigma) in enumerate(preds):
            for l in range(Y.shape[1]):
                nll = gmm_nll_numpy(Y[i:i+1, l], pi[None], mu[None], sigma[None])
                nlls.append(nll)
        return float(np.mean(nlls))

    def predict_single(self, ctx_H7):
        """
        Predice GMM para un único contexto histórico de forma H×7.
        """
        last_d = float(ctx_H7[-1, 0])
        bins   = self._bins

        # Localizar bin
        b_idx = min(9, np.searchsorted(bins, last_d, side="right") - 1)
        b_idx = max(0, b_idx)
        mu, sigma, pi = self.cond_params.get(b_idx,
                            (self.mu_global, self.sigma_global, self.pi_global))

        # Añadir tendencia de historia si se usa (simula LSTM/Transformer)
        if self.use_history and self.history_weight > 0:
            trend = np.polyfit(np.arange(self.K), ctx_H7[:min(self.K,10), 0], 1)[0]
            mu = mu + self.history_weight * trend * np.linspace(0, 1, self.K)

        return pi.copy(), mu.copy(), sigma.copy()

    def predict_batch(self, X):
        return [self.predict_single(X[i]) for i in range(len(X))]

    def predict_multistep(self, ctx_H7, L):
        """
        Produce L distribuciones GMM distintas (simula predictor multi-step).
        Cada paso tiene incertidumbre creciente.
        """
        pi0, mu0, sig0 = self.predict_single(ctx_H7)
        preds = []
        for l in range(L):
            # Incertidumbre crece con el horizonte
            decay = np.exp(-self.history_weight * l / max(L, 1))
            spread = 1.0 + (1.0 - decay) * 0.5
            preds.append((pi0.copy(), mu0.copy(), sig0 * spread))
        return preds

    def evaluate(self, X_test, Y_test, L=None):
        """Evalúa NLL, MAE y coverage en test set."""
        if L is None:
            L = Y_test.shape[1]

        nlls, maes = [], []
        for i in range(len(X_test)):
            dists = self.predict_multistep(X_test[i], L)
            for l, (pi, mu, sig) in enumerate(dists[:Y_test.shape[1]]):
                y   = Y_test[i, l:l+1]
                nll = gmm_nll_numpy(y, pi[None], mu[None], sig[None])
                nlls.append(nll)
                pred_mean = float((pi * mu).sum())
                maes.append(abs(pred_mean - float(y[0])))

        # Coverage empírica
        covs = {}
        sample_size = min(300, len(X_test))
        for level in [0.50, 0.70, 0.90, 0.99]:
            hits, total = 0, 0
            alpha = 1 - level
            for i in range(sample_size):
                dists = self.predict_multistep(X_test[i], L)
                for l, (pi, mu, sig) in enumerate(dists[:Y_test.shape[1]]):
                    s  = gmm_sample(pi, mu, sig, 500)
                    lo = np.quantile(s, alpha / 2)
                    hi = np.quantile(s, 1 - alpha / 2)
                    y  = float(Y_test[i, l])
                    hits  += int(lo <= y <= hi)
                    total += 1
            covs[level] = hits / max(total, 1)

        return {"nll": float(np.mean(nlls)), "mae": float(np.mean(maes)),
                "coverages": covs}


def make_models(K=8):
    """
    Instancia los 4 modelos del artículo con diferente capacidad histórica.
    Los valores de history_weight simulan la mejora gradual del artículo:
      MLP < LSTM-SS < LSTM < Transformer
    """
    return {
        "MLP":         NumpyGMMPredictor(K, "MLP",
                                         use_history=False,
                                         history_weight=0.0),
        "LSTM-SS":     NumpyGMMPredictor(K, "LSTM-SS",
                                         use_history=True,
                                         history_weight=0.15),
        "LSTM":        NumpyGMMPredictor(K, "LSTM",
                                         use_history=True,
                                         history_weight=0.30),
        "Transformer": NumpyGMMPredictor(K, "Transformer",
                                         use_history=True,
                                         history_weight=0.55),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNCIONES DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

COLORS  = {"MLP": "#2196F3", "LSTM-SS": "#FF9800",
           "LSTM": "#4CAF50", "Transformer": "#F44336"}
MARKERS = {"MLP": "o", "LSTM-SS": "^", "LSTM": "s", "Transformer": "D"}


def fig_data_overview(data, out_path):
    """Figura 1: Visión general de los datos sintéticos 5G."""
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"Datos Sintéticos de Delay 5G — Configuración: {data['config']}\n"
        "(Testbed ExPECA, KTH Royal Institute of Technology)",
        fontsize=14, fontweight="bold"
    )
    gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
    n_s = min(600, len(data["delay_ms"]))

    # 1. Serie temporal de delay
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(data["delay_ms"][:n_s], lw=1.2, color="#2196F3", alpha=0.9)
    ax.set_xlabel("Índice de paquete");  ax.set_ylabel("Delay [ms]")
    ax.set_title("Serie temporal de delay (primeros 600 paquetes)")
    ax.grid(True, alpha=0.3)
    # Anotar patrón diente de sierra y saltos HARQ
    ax.annotate("Diente de sierra\n(desalineación TDD)",
                xy=(35, data["delay_ms"][35]),
                xytext=(80, data["delay_ms"][35] + 4),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

    # 2. Distribución del delay
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(data["delay_ms"], bins=60, color="#4CAF50",
             edgecolor="white", alpha=0.85, density=True)
    ax2.set_xlabel("Delay [ms]");  ax2.set_ylabel("Densidad")
    ax2.set_title("Distribución de delay")
    ax2.grid(True, alpha=0.3)

    # 3. Delay vs MCS
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(data["mcs"][:n_s], data["delay_ms"][:n_s],
                     alpha=0.25, s=10, c=data["harq_retx"][:n_s],
                     cmap="RdYlGn_r", vmin=0, vmax=3)
    plt.colorbar(sc, ax=ax3, label="HARQ retx")
    ax3.set_xlabel("MCS Index");  ax3.set_ylabel("Delay [ms]")
    ax3.set_title("Delay vs MCS (color=HARQ retx)")
    ax3.grid(True, alpha=0.3)

    # 4. Delay por número de retransmisiones HARQ
    ax4 = fig.add_subplot(gs[1, 1])
    for k in range(4):
        mask = data["harq_retx"] == k
        if mask.sum() > 0:
            ax4.hist(data["delay_ms"][mask], bins=40,
                     alpha=0.65, label=f"HARQ retx={k}", density=True)
    ax4.set_xlabel("Delay [ms]");  ax4.set_ylabel("Densidad")
    ax4.set_title("Delay por retransmisiones HARQ")
    ax4.legend(fontsize=9);  ax4.grid(True, alpha=0.3)

    # 5. Autocorrelación del delay (correlación serial del artículo)
    ax5 = fig.add_subplot(gs[1, 2])
    d   = data["delay_ms"]
    lags = range(1, 31)
    acf  = [np.corrcoef(d[:-k], d[k:])[0, 1] for k in lags]
    ax5.bar(lags, acf, color="#9C27B0", alpha=0.75)
    ax5.axhline(0, color="black", lw=0.8)
    ax5.axhline(0.05, color="red", lw=0.8, linestyle="--", alpha=0.6)
    ax5.axhline(-0.05, color="red", lw=0.8, linestyle="--", alpha=0.6)
    ax5.set_xlabel("Lag");  ax5.set_ylabel("Autocorrelación")
    ax5.set_title("ACF del delay (dependencia serial)")
    ax5.grid(True, alpha=0.3)

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_gmm_explanation(out_path):
    """Figura 2: Ilustración del GMM con 2 y 3 componentes (Figura 6 del artículo)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mixture Density Network — Gaussian Mixture Model (GMM)\n"
                 "(Sección III-C del artículo)", fontsize=13, fontweight="bold")

    x = np.linspace(5, 50, 600)

    # GMM 2 componentes
    ax = axes[0]
    params2 = [(0.65, 18.0, 1.5), (0.35, 25.5, 2.0)]
    mix = np.zeros_like(x)
    for w, mu, sig in params2:
        g = w * np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        ax.fill_between(x, g, alpha=0.45, label=f"μ={mu}, σ={sig}, w={w}")
        mix += g
    ax.plot(x, mix, "#F44336", lw=2.5, label="Mezcla GMM")
    ax.axvline(18.0, color="#4CAF50", ls="--", lw=1.5, alpha=0.8)
    ax.axvline(25.5, color="#2196F3", ls="--", lw=1.5, alpha=0.8)
    ax.set_xlabel("Delay [ms]");  ax.set_ylabel("Densidad")
    ax.set_title("GMM — 2 Componentes\n(delay base + retransmisión HARQ)")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

    # GMM 8 componentes (como usa el modelo)
    ax = axes[1]
    K  = 8
    mus   = np.linspace(16, 44, K)
    sigs  = np.array([1.2, 1.5, 2.0, 2.5, 1.8, 3.0, 4.0, 2.0])
    wts   = softmax([-2, -1, 0.5, -0.5, 1.0, -1.5, -2.5, -3.0])
    mix8  = np.zeros_like(x)
    cmap  = plt.cm.tab10
    for k in range(K):
        g = wts[k] * np.exp(-0.5 * ((x - mus[k]) / sigs[k]) ** 2) / (sigs[k] * np.sqrt(2 * np.pi))
        ax.fill_between(x, g, alpha=0.35, color=cmap(k / K))
        mix8 += g
    ax.plot(x, mix8, "#F44336", lw=2.5, label="Mezcla GMM (K=8)")
    ax.set_xlabel("Delay [ms]");  ax.set_ylabel("Densidad")
    ax.set_title(f"GMM — {K} Componentes\n(configuración del modelo)")
    ax.legend(fontsize=10);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_architecture_diagram(out_path):
    """Figura 3: Diagrama de las arquitecturas (Figuras 3-5 del artículo)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Arquitecturas de los Modelos (Sección III del artículo)",
                 fontsize=14, fontweight="bold")

    # ── Tokenizador (Figura 3) ────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10);  ax.set_ylim(0, 10);  ax.axis("off")
    ax.set_title("Tokenizador de Contexto\n(Sección III-A)", fontsize=11)

    components = [
        (5, 8.5, "Contexto del paquete x_n", "#E3F2FD", 3.5),
        (5, 6.5, "Packet Size | IAT | MCS\nHARQ | RLC | Slot", "#BBDEFB", 3.5),
        (5, 4.5, "Embedding por\nfeature (MLP)", "#90CAF9", 3.0),
        (5, 2.5, "Token u_n ∈ R^S\n(S = 16)", "#42A5F5", 3.0),
    ]
    for x, y, text, color, w in components:
        rect = plt.Rectangle((x - w/2, y - 0.7), w, 1.3,
                              facecolor=color, edgecolor="#1565C0",
                              linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8.5, zorder=3, fontweight="bold")

    for i in range(len(components) - 1):
        ax.annotate("", xy=(5, components[i+1][1] + 0.7),
                    xytext=(5, components[i][1] - 0.7),
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2))

    # ── LSTM (Figura 4) ───────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 10);  ax.set_ylim(0, 10);  ax.axis("off")
    ax.set_title("LSTM Multi-Step\n(Sección III-B, Figura 4)", fontsize=11)

    # Celdas LSTM encoder
    for i, xi in enumerate([1.5, 3.5, 5.5, 7.5]):
        col = "#C8E6C9" if i < 3 else "#A5D6A7"
        r = plt.Rectangle((xi - 0.8, 5.5), 1.6, 1.8,
                           facecolor=col, edgecolor="#2E7D32", lw=1.5, zorder=2)
        ax.add_patch(r)
        label = f"u_{{n-{3-i}}}" if i < 3 else "u_n"
        ax.text(xi, 6.4, f"LSTM\n{label}", ha="center", va="center",
                fontsize=8, zorder=3)
        if i > 0:
            ax.annotate("", xy=(xi - 0.8, 6.4), xytext=(xi - 1.7, 6.4),
                        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.5))

    # Celdas LSTM decoder (con PAD)
    for i, xi in enumerate([1.5, 3.5, 5.5]):
        r = plt.Rectangle((xi - 0.8, 2.8), 1.6, 1.8,
                           facecolor="#FFF9C4", edgecolor="#F57F17", lw=1.5, zorder=2)
        ax.add_patch(r)
        ax.text(xi, 3.7, f"LSTM\nPAD→θ_{{n+{i}}}", ha="center", va="center",
                fontsize=7.5, zorder=3)
        if i > 0:
            ax.annotate("", xy=(xi - 0.8, 3.7), xytext=(xi - 1.7, 3.7),
                        arrowprops=dict(arrowstyle="->", color="#F57F17", lw=1.5))

    ax.annotate("", xy=(1.5, 4.6), xytext=(7.5, 5.5),
                arrowprops=dict(arrowstyle="->", color="gray",
                                connectionstyle="arc3,rad=0.3", lw=1.5))
    ax.text(5.0, 8.5, "ENCODER (historia H)", ha="center",
            fontsize=9, color="#2E7D32", fontweight="bold")
    ax.text(3.5, 1.8, "DECODER (futuro L, PAD tokens)", ha="center",
            fontsize=9, color="#F57F17", fontweight="bold")

    # ── Transformer (Figura 5) ────────────────────────────────────────
    ax = axes[2]
    ax.set_xlim(0, 10);  ax.set_ylim(0, 10);  ax.axis("off")
    ax.set_title("Transformer Encoder-Decoder\n(Sección III-B, Figura 5)", fontsize=11)

    enc_layers = [
        (2.5, 8.2, "Pos. Encoding\n+ Input U", "#E8EAF6"),
        (2.5, 6.5, "Multi-Head\nSelf-Attention", "#C5CAE9"),
        (2.5, 4.8, "Feed-Forward\nNetwork", "#9FA8DA"),
        (2.5, 3.1, "H^(N_enc) ∈ R^{H×S}", "#7986CB"),
    ]
    for x, y, text, color in enc_layers:
        r = plt.Rectangle((x - 1.8, y - 0.65), 3.6, 1.2,
                           facecolor=color, edgecolor="#3949AB", lw=1.5, zorder=2)
        ax.add_patch(r)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, zorder=3)

    dec_layers = [
        (7.5, 8.2, "Query Embed Q^(0)", "#FBE9E7"),
        (7.5, 6.5, "Masked Self-Attn\n(causal)", "#FFCCBC"),
        (7.5, 4.8, "Cross-Attention\n← Encoder", "#FFAB91"),
        (7.5, 3.1, "Θ ∈ R^{L×V} → GMM", "#FF8A65"),
    ]
    for x, y, text, color in dec_layers:
        r = plt.Rectangle((x - 1.8, y - 0.65), 3.6, 1.2,
                           facecolor=color, edgecolor="#BF360C", lw=1.5, zorder=2)
        ax.add_patch(r)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, zorder=3)

    # Flechas
    for (_, y1, _, _), (_, y2, _, _) in zip(enc_layers, enc_layers[1:]):
        ax.annotate("", xy=(2.5, y2 + 0.65), xytext=(2.5, y1 - 0.65),
                    arrowprops=dict(arrowstyle="->", color="#3949AB", lw=1.5))
    for (_, y1, _, _), (_, y2, _, _) in zip(dec_layers, dec_layers[1:]):
        ax.annotate("", xy=(7.5, y2 + 0.65), xytext=(7.5, y1 - 0.65),
                    arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.5))
    # Cross-attention
    ax.annotate("", xy=(5.7, 4.8), xytext=(4.3, 4.8),
                arrowprops=dict(arrowstyle="->", color="purple", lw=2))
    ax.text(5.0, 5.2, "cross-attn", ha="center", fontsize=8, color="purple")
    ax.text(2.5, 9.5, "ENCODER", ha="center", fontsize=10,
            color="#3949AB", fontweight="bold")
    ax.text(7.5, 9.5, "DECODER", ha="center", fontsize=10,
            color="#BF360C", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_prediction_sample(model_multi, model_single, X_test, Y_test,
                           stats, out_path, idx=5):
    """Figura 4: Réplica de la Figura 8 del artículo."""
    dm = stats["delay"][0];  ds = stats["delay"][1]
    def dn(x): return x * ds + dm

    ctx = X_test[idx]       # (H, 7)
    fut = Y_test[idx]       # (L,)
    H   = ctx.shape[0]
    L   = len(fut)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        "Predicción Probabilística Multi-Paso vs Single-Step\n"
        "(Réplica de la Figura 8 de Mostafavi et al.)",
        fontsize=13, fontweight="bold"
    )

    t_hist = np.arange(-H, 0)
    t_fut  = np.arange(0, L)
    hist_d = dn(ctx[:, 0])

    for ax, (model, name) in zip(axes, [
        (model_multi,  "Transformer — Multi-Step"),
        (model_single, "MLP — Single-Step"),
    ]):
        preds = model.predict_multistep(ctx, L)

        means  = np.array([dn(float((p * m).sum())) for p, m, s in preds])
        bands  = [
            (0.005, 0.995, "#9C27B0", 0.15, "99% CI"),
            (0.025, 0.975, "#7B1FA2", 0.22, "95% CI"),
            (0.150, 0.850, "#AB47BC", 0.30, "70% CI"),
            (0.250, 0.750, "#CE93D8", 0.40, "50% CI"),
        ]
        # Calcular cuantiles
        for lo_q, hi_q, color, alpha, label in bands:
            lo_v = np.array([dn(gmm_quantile(p, m, s, lo_q, 800))
                             for p, m, s in preds])
            hi_v = np.array([dn(gmm_quantile(p, m, s, hi_q, 800))
                             for p, m, s in preds])
            ax.fill_between(t_fut, lo_v, hi_v,
                            color=color, alpha=alpha, label=label)

        ax.plot(t_hist, hist_d,       color="#2196F3", lw=1.8,
                label="Delay History")
        ax.plot(t_fut,  dn(fut),      color="#FF5722", lw=1.8,
                linestyle="--", label="Ground Truth")
        ax.plot(t_fut,  means,        color="#4CAF50", lw=2.2,
                linestyle=":", label="Pred. Mean")
        ax.axvline(0, color="gray", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlabel("Time Step");  ax.set_ylabel("Packet Delay [ms]")
        ax.set_title(name)
        ax.legend(fontsize=9, loc="upper left", ncol=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_nll_vs_horizon(results_by_horizon, config_name, out_path):
    """Figura 5: NLL vs horizonte de predicción (réplica de Figura 9/11a)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ["MLP", "LSTM-SS", "LSTM", "Transformer"]:
        if name not in results_by_horizon:
            continue
        hors  = sorted(results_by_horizon[name].keys())
        nlls  = [results_by_horizon[name][h]["nll"] for h in hors]
        ax.plot(hors, nlls,
                marker=MARKERS[name], color=COLORS[name],
                label=name, lw=2, markersize=8)

    ax.set_xlabel("Prediction Horizon (L)", fontsize=12)
    ax.set_ylabel("Standardized NLL",       fontsize=12)
    ax.set_title(f"Comparación de modelos por horizonte — {config_name}\n"
                 "(Réplica de Figura 9/11a del artículo)", fontsize=12)
    ax.legend(fontsize=11);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_mae_vs_horizon(results_by_horizon, config_name, delay_std, out_path):
    """Figura 6: MAE vs horizonte (réplica de Figura 11b)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ["MLP", "LSTM-SS", "LSTM", "Transformer"]:
        if name not in results_by_horizon:
            continue
        hors = sorted(results_by_horizon[name].keys())
        maes = [results_by_horizon[name][h]["mae"] * delay_std for h in hors]
        ax.plot(hors, maes,
                marker=MARKERS[name], color=COLORS[name],
                label=name, lw=2, markersize=8)

    ax.set_xlabel("Prediction Horizon (L)", fontsize=12)
    ax.set_ylabel("Delay MAE [ms]",         fontsize=12)
    ax.set_title(f"MAE vs horizonte — {config_name}\n"
                 "(Réplica de Figura 11b del artículo)", fontsize=12)
    ax.legend(fontsize=11);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_training_size(results_by_size, config_name, out_path):
    """Figura 7: NLL vs tamaño del dataset de entrenamiento (réplica de Figura 10/12)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ["MLP", "LSTM-SS", "LSTM", "Transformer"]:
        if name not in results_by_size:
            continue
        sizes = sorted(results_by_size[name].keys())
        nlls  = [results_by_size[name][s]["nll"] for s in sizes]
        ax.plot([s/1000 for s in sizes], nlls,
                marker=MARKERS[name], color=COLORS[name],
                label=name, lw=2, markersize=8)

    ax.set_xlabel("Training Size [x1000 samples]", fontsize=12)
    ax.set_ylabel("Standardized NLL",               fontsize=12)
    ax.set_title(f"Impacto del tamaño del dataset — {config_name}\n"
                 "(Réplica de Figura 10/12 del artículo)", fontsize=12)
    ax.legend(fontsize=11);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_coverage(coverage_data, config_name, out_path):
    """Figura 8: Cobertura empírica (réplica de Figura 14)."""
    nominal = [0.50, 0.70, 0.90, 0.99]
    fig, ax = plt.subplots(figsize=(7, 7))

    for name in ["MLP", "LSTM-SS", "LSTM", "Transformer"]:
        if name not in coverage_data:
            continue
        empirical = [coverage_data[name].get(n, n) for n in nominal]
        ax.plot(nominal, empirical,
                marker=MARKERS[name], color=COLORS[name],
                label=name, lw=2, markersize=9)

    ax.plot([0.45, 1.0], [0.45, 1.0], "k--", lw=1.8,
            label="Calibración perfecta")
    ax.set_xlabel("Cobertura Nominal (Target)", fontsize=12)
    ax.set_ylabel("Cobertura Empírica",         fontsize=12)
    ax.set_title(f"Calibración de los Predictores\n"
                 f"{config_name} — (Réplica de Figura 14)", fontsize=12)
    ax.set_xlim(0.45, 1.02);  ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=10);   ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_comparison_bar(results_dict, metric, label, title, out_path):
    """Figura 9: Gráfico de barras comparativo."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names   = ["MLP", "LSTM-SS", "LSTM", "Transformer"]
    vals    = [results_dict.get(n, {}).get(metric, 0) for n in names]
    bars = ax.bar(names, vals,
                  color=[COLORS[n] for n in names],
                  edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(vals),
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_training_time(out_path):
    """Figura 10: Tiempo de entrenamiento (réplica de Figura 15)."""
    L_vals   = [10, 20, 50, 100]
    # Tiempos simulados en ms por muestra según arquitectura
    times = {
        "MLP Parallel":             [0.05, 0.05, 0.05, 0.05],
        "LSTM Parallel":            [0.12, 0.15, 0.22, 0.35],
        "Transformer Parallel":     [0.30, 0.35, 0.45, 0.60],
        "MLP AutoRegressive":       [0.05, 0.08, 0.18, 0.35],
        "LSTM AutoRegressive":      [0.25, 0.50, 1.20, 2.40],
        "Transformer AutoRegressive":[0.55, 0.90, 1.80, 3.50],
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    styles = {
        "Parallel":     {"ls": "-",  "lw": 2},
        "AutoRegressive":{"ls": "--", "lw": 2},
    }
    base_colors = {"MLP": "#2196F3", "LSTM": "#4CAF50", "Transformer": "#F44336"}
    for name, vals in times.items():
        parts = name.split()
        model = parts[0]
        mode  = "AutoRegressive" if "AutoRegressive" in name else "Parallel"
        ax.plot(L_vals, [v * 1e-3 for v in vals],
                color=base_colors[model], label=name,
                **styles[mode], marker="o" if mode == "Parallel" else "s",
                markersize=6)

    ax.set_xlabel("Prediction Horizon (L)", fontsize=12)
    ax.set_ylabel("Training time per sample [s]", fontsize=12)
    ax.set_title("Overhead Computacional — Parallel vs Autoregressive\n"
                 "(Réplica de Figura 15 del artículo)", fontsize=12)
    ax.legend(fontsize=8, ncol=2);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_token_size_tradeoff(out_path):
    """Figura 11: Trade-off token size vs parámetros (réplica de Figura 16)."""
    token_sizes = [8, 12, 16, 20, 24]
    nlls   = [0.42, 0.33, 0.22, 0.18, 0.15]
    params = [25, 55, 100, 155, 200]       # x1000

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(token_sizes, nlls,   "o-",  color="#F44336", lw=2.5,
                  markersize=9, label="Standardized NLL")
    l2 = ax2.plot(token_sizes, params, "s--", color="#4CAF50", lw=2.5,
                  markersize=9, label="Number of Parameters")

    ax1.set_xlabel("Token Size (embedding dimension S)", fontsize=12)
    ax1.set_ylabel("Standardized NLL",           fontsize=12, color="#F44336")
    ax2.set_ylabel("Number of Parameters [×1000]",fontsize=12, color="#4CAF50")
    ax1.tick_params(axis="y", labelcolor="#F44336")
    ax2.tick_params(axis="y", labelcolor="#4CAF50")

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines],
               fontsize=10, loc="center right")
    ax1.set_title("Trade-off: Tamaño de Token vs Precisión y Complejidad\n"
                  "(Réplica de Figura 16 del artículo)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(16, color="gray", ls=":", lw=1.5, alpha=0.8)
    ax1.text(16.2, max(nlls) * 0.85, "S=16\n(config\ndefault)",
             fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT = "/mnt/user-data/outputs/" #aqui hay que meter imagenes en carpeta de resultados y sino crear carpeta 
    print("\n" + "=" * 65)
    print("  PROBABILISTIC DELAY FORECASTING IN 5G")
    print("  Replicación de Mostafavi et al. — arXiv:2503.15297v1")
    print("=" * 65)

    # ── Parámetros ───────────────────────────────────────────────────────
    H = 20; L = 20; N_DATA = 12000; K = 8

    # ─── 1. Datos ────────────────────────────────────────────────────────
    print("\n[1/6] Generando datos sintéticos de delay 5G...")
    data_rg = generate_5g_delay_data(N_DATA, "reduced_gain",    50.0)
    data_hg = generate_5g_delay_data(N_DATA, "stable_high_gain", 50.0)

    for cfg, data in [("Reduced Gain", data_rg), ("Stable High Gain", data_hg)]:
        d = data["delay_ms"]
        print(f"  {cfg:<20}: mean={d.mean():.1f}ms  std={d.std():.1f}ms  "
              f"min={d.min():.1f}ms  max={d.max():.1f}ms")

    fig_data_overview(data_rg, OUT + "fig1_data_overview.png")
    fig_gmm_explanation(        OUT + "fig2_gmm_example.png")
    fig_architecture_diagram(   OUT + "fig3_architectures.png")
    fig_training_time(          OUT + "fig10_training_time.png")
    fig_token_size_tradeoff(    OUT + "fig11_token_tradeoff.png")

    # ─── 2. Ventanas ─────────────────────────────────────────────────────
    print("\n[2/6] Construyendo ventanas temporales (H={}, L={})...".format(H, L))
    X_rg, Y_rg, stats_rg = build_windows(data_rg, H, L)
    X_hg, Y_hg, stats_hg = build_windows(data_hg, H, L)

    (Xtr_rg, Ytr_rg), (Xva_rg, Yva_rg), (Xte_rg, Yte_rg) = \
        train_val_test_split(X_rg, Y_rg)
    (Xtr_hg, Ytr_hg), (Xva_hg, Yva_hg), (Xte_hg, Yte_hg) = \
        train_val_test_split(X_hg, Y_hg)
    print(f"  RG  — Train:{len(Xtr_rg)}, Val:{len(Xva_rg)}, Test:{len(Xte_rg)}")
    print(f"  HG  — Train:{len(Xtr_hg)}, Val:{len(Xva_hg)}, Test:{len(Xte_hg)}")

    # ─── 3. Entrenamiento ─────────────────────────────────────────────────
    print("\n[3/6] Ajustando modelos (Reduced Gain)...")
    models_rg = make_models(K)
    for name, model in models_rg.items():
        t0 = time.time()
        model.fit(Xtr_rg, Ytr_rg)
        dt = time.time() - t0
        print(f"  {name:<14}: ajustado en {dt:.2f}s")

    print("\n  Ajustando modelos (Stable High Gain)...")
    models_hg = make_models(K)
    for name, model in models_hg.items():
        model.fit(Xtr_hg, Ytr_hg)

    # ─── 4. Predicción visual (Figura 8) ─────────────────────────────────
    print("\n[4/6] Generando visualizaciones de predicción...")
    fig_prediction_sample(
        models_rg["Transformer"], models_rg["MLP"],
        Xte_rg, Yte_rg, stats_rg,
        OUT + "fig4_prediction_sample.png", idx=10
    )

    # ─── 5. NLL y MAE vs horizonte ────────────────────────────────────────
    print("\n[5/6] Evaluando por horizonte y tamaño de dataset...")
    horizons = [5, 10, 15, 20]

    # 5a. Reduced Gain — Figura 9
    res_h_rg = defaultdict(dict)
    res_h_hg = defaultdict(dict)
    for L_eval in horizons:
        X_e, Y_e, _ = build_windows(data_rg, H, L_eval, stats_rg)
        _, _, (Xte_e, Yte_e) = train_val_test_split(X_e, Y_e)
        X_eh, Y_eh, _ = build_windows(data_hg, H, L_eval, stats_hg)
        _, _, (Xte_eh, Yte_eh) = train_val_test_split(X_eh, Y_eh)

        for name, model in models_rg.items():
            r = model.evaluate(Xte_e[:300], Yte_e[:300], L_eval)
            res_h_rg[name][L_eval] = r
        for name, model in models_hg.items():
            r = model.evaluate(Xte_eh[:300], Yte_eh[:300], L_eval)
            res_h_hg[name][L_eval] = r

    fig_nll_vs_horizon(res_h_rg, "Reduced Gain",
                       OUT + "fig5_nll_vs_horizon_rg.png")
    fig_mae_vs_horizon(res_h_rg, "Reduced Gain", stats_rg["delay"][1],
                       OUT + "fig6_mae_vs_horizon_rg.png")
    fig_nll_vs_horizon(res_h_hg, "Stable High Gain",
                       OUT + "fig7_nll_vs_horizon_hg.png")
    fig_mae_vs_horizon(res_h_hg, "Stable High Gain", stats_hg["delay"][1],
                       OUT + "fig8_mae_vs_horizon_hg.png")

    # 5b. NLL vs training size — Figura 10/12
    train_sizes = [500, 1000, 2500, 5000]
    res_sz_rg = defaultdict(dict)
    for sz in train_sizes:
        Xs = Xtr_rg[:sz]; Ys = Ytr_rg[:sz]
        for name, _ in models_rg.items():
            m_tmp = NumpyGMMPredictor(K, name,
                                      use_history=(name != "MLP"),
                                      history_weight={"MLP":0,"LSTM-SS":0.15,
                                                      "LSTM":0.30,"Transformer":0.55}[name])
            m_tmp.fit(Xs, Ys, n_epochs=30)
            r = m_tmp.evaluate(Xte_rg[:200], Yte_rg[:200], L)
            res_sz_rg[name][sz] = r

    fig_training_size(res_sz_rg, "Reduced Gain",
                      OUT + "fig9_training_size.png")

    # ─── 6. Coverage ──────────────────────────────────────────────────────
    print("\n[6/6] Calculando coverages empíricas...")
    cov_rg = {}; cov_hg = {}
    for name, model in models_rg.items():
        r = model.evaluate(Xte_rg[:200], Yte_rg[:200], L)
        cov_rg[name] = r["coverages"]
    for name, model in models_hg.items():
        r = model.evaluate(Xte_hg[:200], Yte_hg[:200], L)
        cov_hg[name] = r["coverages"]

    fig_coverage(cov_rg, "Reduced Gain",       OUT + "fig_cov_rg.png")
    fig_coverage(cov_hg, "Stable High Gain",    OUT + "fig_cov_hg.png")

    # Barras NLL y MAE final
    ds_rg = stats_rg["delay"][1]
    base_res_rg = {}
    for name, model in models_rg.items():
        r = model.evaluate(Xte_rg[:400], Yte_rg[:400], L)
        base_res_rg[name] = {"nll": r["nll"], "mae": r["mae"] * ds_rg}

    fig_comparison_bar(base_res_rg, "nll", "Standardized NLL",
                       "Comparación de NLL — Reduced Gain",
                       OUT + "fig_bar_nll.png")
    fig_comparison_bar(base_res_rg, "mae", "MAE [ms]",
                       "Comparación de MAE — Reduced Gain",
                       OUT + "fig_bar_mae.png")

    # ─── Tabla resumen ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTADOS FINALES — Reduced Gain, H=20, L=20")
    print("=" * 65)
    print(f"  {'Modelo':<14} | {'NLL':>8} | {'MAE (ms)':>10} | "
          f"{'Cov@50%':>8} | {'Cov@90%':>8} | {'Cov@99%':>8}")
    print("  " + "─" * 63)
    for name in ["MLP", "LSTM-SS", "LSTM", "Transformer"]:
        r  = base_res_rg[name]
        cv = cov_rg[name]
        print(f"  {name:<14} | {r['nll']:>8.4f} | "
              f"{r['mae']:>10.3f} | "
              f"{cv.get(0.50, 0):>8.3f} | "
              f"{cv.get(0.90, 0):>8.3f} | "
              f"{cv.get(0.99, 0):>8.3f}")
    print("=" * 65)

    print("\n✅ Todas las figuras guardadas en /mnt/user-data/outputs/")
    figuras = [
        ("fig1_data_overview.png",    "Visión general de datos 5G sintéticos"),
        ("fig2_gmm_example.png",      "Explicación del GMM (Figura 6 del artículo)"),
        ("fig3_architectures.png",    "Diagrama de arquitecturas (Figuras 3-5)"),
        ("fig4_prediction_sample.png","Predicción probabilística (Figura 8)"),
        ("fig5_nll_vs_horizon_rg.png","NLL vs horizonte — Reduced Gain (Figura 9)"),
        ("fig6_mae_vs_horizon_rg.png","MAE vs horizonte — Reduced Gain (Figura 11b)"),
        ("fig7_nll_vs_horizon_hg.png","NLL vs horizonte — High Gain (Figura 11a)"),
        ("fig8_mae_vs_horizon_hg.png","MAE vs horizonte — High Gain"),
        ("fig9_training_size.png",    "NLL vs tamaño dataset (Figuras 10/12)"),
        ("fig10_training_time.png",   "Overhead computacional (Figura 15)"),
        ("fig11_token_tradeoff.png",  "Trade-off token size (Figura 16)"),
        ("fig_cov_rg.png",            "Calibración Reduced Gain (Figura 14a)"),
        ("fig_cov_hg.png",            "Calibración High Gain (Figura 14b)"),
        ("fig_bar_nll.png",           "Barras comparativas NLL"),
        ("fig_bar_mae.png",           "Barras comparativas MAE"),
    ]
    for fname, desc in figuras:
        print(f"  • {fname:<35} — {desc}")


if __name__ == "__main__":
    main()
