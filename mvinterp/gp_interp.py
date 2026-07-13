"""GPyTorch (RBF-kernel Gaussian Process) variant of the cosmology interpolator.

Same interpolation *mean* as the RBF method, but adds two things a plain RBF cannot:
  1. principled hyperparameters -- the RBF lengthscale and noise are LEARNED by
     marginal-likelihood optimisation (GPyTorch), instead of a heuristic/LOO grid;
  2. an uncertainty -- the GP posterior variance, which depends only on theta, so
     it is one error bar per query cosmology (shared across all particles).

Architecture (why this scales to N=2048^3):
  * GPyTorch fits the GP on the M(~56) cosmologies -- a tiny M x M kernel. To learn
    hyperparameters shared across the per-particle field we train a *batch* of GPs
    on a particle subsample and take the median lengthscale/noise.
  * The massive per-particle mean is then predicted with a tiled kernel-ridge solve
    mean(theta*) = m + k*(K+sigma^2 I)^{-1}(Y - m); (K+sigma^2 I)^{-1} is M x M,
    built once and reused across particle tiles -- GPyTorch's own predict would try
    to materialise the full solve and never fits in memory.

Backend is torch (GPU via .to(device)); this is the one module that uses torch
rather than the CuPy backend, because GPyTorch is a torch library.
"""
import numpy as np
import torch
import gpytorch


def _pick_device(pref=None):
    if pref is not None:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _BatchGP(gpytorch.models.ExactGP):
    """Independent GPs over C output channels sharing the same input points."""
    def __init__(self, x, y, likelihood, C, d):
        super().__init__(x, y, likelihood)
        bs = torch.Size([C])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=bs, ard_num_dims=d), batch_shape=bs)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class GPCosmologyInterpolator:
    def __init__(self, iters=100, lr=0.1, device=None, dtype=torch.float64):
        self.iters = iters
        self.lr = lr
        self.device = _pick_device(device)
        self.dtype = dtype
        self._ready = False

    # ---- hyperparameter learning (GPyTorch) ----
    def fit(self, theta_train, Y_sample, verbose=False):
        """Learn RBF lengthscale + noise on a particle subsample.

        theta_train: (M,3);  Y_sample: (M, n_sample, 3) small subsample of the field.
        """
        th = np.asarray(theta_train, dtype=np.float64)
        self._mu = th.mean(0, keepdims=True)
        self._sd = th.std(0, keepdims=True) + 1e-12
        ts = (th - self._mu) / self._sd
        M, d = ts.shape
        Ys = np.asarray(Y_sample, dtype=np.float64).reshape(M, -1)      # (M, C)
        C = Ys.shape[1]

        x = torch.tensor(ts, dtype=self.dtype, device=self.device)
        xb = x.unsqueeze(0).expand(C, M, d).contiguous()               # (C,M,d)
        yb = torch.tensor(Ys.T, dtype=self.dtype, device=self.device).contiguous()  # (C,M)

        lik = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([C])).to(self.device, self.dtype)
        model = _BatchGP(xb, yb, lik, C, d).to(self.device, self.dtype)
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        for i in range(self.iters):
            opt.zero_grad()
            loss = -mll(model(xb), yb).sum()
            loss.backward(); opt.step()
            if verbose and i % 20 == 0:
                print(f"  gp iter {i:3d}  loss {loss.item():.3f}")

        ls = model.covar_module.base_kernel.lengthscale.detach().reshape(C, d)
        os = model.covar_module.outputscale.detach().reshape(C)
        nz = lik.noise.detach().reshape(C)
        self.lengthscale = ls.median(0).values                         # (d,)
        self.outputscale = os.median()
        self.noise = nz.median()
        self._ready = True
        return self

    # ---- kernel ridge predictor (tiled) ----
    def _K(self, A, B):
        Al, Bl = A / self.lengthscale, B / self.lengthscale
        d2 = (Al * Al).sum(1, keepdim=True) + (Bl * Bl).sum(1) - 2.0 * Al @ Bl.T
        return self.outputscale * torch.exp(-0.5 * d2.clamp_min(0.0))

    def prepare(self, theta_train):
        """Precompute (K + noise I)^{-1} on the M cosmologies (M x M, once)."""
        assert self._ready, "call fit() first"
        ts = (np.asarray(theta_train, float) - self._mu) / self._sd
        self._X = torch.tensor(ts, dtype=self.dtype, device=self.device)
        self._M = self._X.shape[0]
        K = self._K(self._X, self._X) + self.noise * torch.eye(
            self._M, dtype=self.dtype, device=self.device)
        self._Kinv = torch.linalg.inv(K)
        return self

    def _weights(self, theta_query):
        tqs = (np.atleast_2d(np.asarray(theta_query, float)) - self._mu) / self._sd
        Xq = torch.tensor(tqs, dtype=self.dtype, device=self.device)
        kstar = self._K(Xq, self._X)                                   # (Q,M)
        return Xq, kstar, kstar @ self._Kinv                           # W = (Q,M)

    def predict(self, Y_train, theta_query, tile=2_000_000):
        """Posterior-mean per-particle field. Y_train: (M,N,3) -> (Q,N,3) host."""
        _, _, W = self._weights(theta_query)
        Q, N = W.shape[0], Y_train.shape[1]
        out = np.empty((Q, N, 3), dtype=np.float32)
        for s in range(0, N, tile):
            e = min(s + tile, N)
            Yt = torch.tensor(Y_train[:, s:e, :].reshape(self._M, -1),
                              dtype=self.dtype, device=self.device)
            m = Yt.mean(0, keepdim=True)                               # constant mean
            pred = m + W @ (Yt - m)                                    # (Q, t*3)
            out[:, s:e, :] = pred.reshape(Q, e - s, 3).cpu().numpy()
        return out

    def predict_uncertainty(self, theta_query):
        """Posterior std per query cosmology (theta-only; shared across particles)."""
        _, kstar, W = self._weights(theta_query)
        kss = self.outputscale                                        # k(theta*,theta*)
        var = kss - (kstar * W).sum(1)                                # (Q,)
        var = var + self.noise                                        # predictive
        return torch.sqrt(var.clamp_min(0)).cpu().numpy()
