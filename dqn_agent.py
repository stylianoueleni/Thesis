import numpy as np
import random
from collections import deque

def _he_init(fan_in: int, fan_out: int, rng) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float64)

class _NumpyMLP:
    """fc1 → relu → fc2 → relu → fcOut"""

    def __init__(self, obs_dim: int, num_actions: int,hidden: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)

        self.W1 = _he_init(obs_dim, hidden, rng)
        self.b1 = np.zeros((1, hidden), dtype=np.float64)
        self.W2 = _he_init(hidden, hidden, rng)
        self.b2 = np.zeros((1, hidden), dtype=np.float64)
        self.W3 = _he_init(hidden, num_actions, rng)
        self.b3 = np.zeros((1, num_actions), dtype=np.float64)

        self._params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

        self._cache: dict = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X   : (B, obs_dim) - batch of observations
        out : (B, num_actions)

        """
        z1  = X  @ self.W1 + self.b1
        a1  = np.maximum(0.0, z1)           # ReLU  (relu1)
        z2  = a1 @ self.W2 + self.b2
        a2  = np.maximum(0.0, z2)           # ReLU  (relu2)
        out = a2 @ self.W3 + self.b3        # fcOut (no activation)
        self._cache = dict(X=X, z1=z1, a1=a1, z2=z2, a2=a2)
        return out
    
    def update(
        self,
        Q_pred:    np.ndarray,
        Q_target:  np.ndarray,
        lr:        float = 1e-3,
        l2:        float = 1e-4,
        delta:     float = 1.0,
        clip_norm: float = 10.0,
    ) -> float:
        B    = float(Q_pred.shape[0])
        diff = Q_pred - Q_target
        absd = np.abs(diff)

        dL = np.where(absd <= delta, diff, delta * np.sign(diff)) / B

        c  = self._cache
        
        dW3 = c["a2"].T @ dL
        db3 = dL.sum(axis=0, keepdims=True)
        
        dA2 = dL  @ self.W3.T
        dZ2 = dA2 * (c["z2"] > 0).astype(np.float64)
        dW2 = c["a1"].T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)
        
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (c["z1"] > 0).astype(np.float64)
        dW1 = c["X"].T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)

        grads = [dW1, db1, dW2, db2, dW3, db3]

        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)
            grads = [g * scale for g in grads]

        self._t += 1
        t = self._t
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self._params, grads)):
            wd = l2 if i % 2 == 0 else 0.0
            g_wd = g + wd * p
            self._m[i] = b1 * self._m[i] + (1 - b1) * g_wd
            self._v[i] = b2 * self._v[i] + (1 - b2) * g_wd**2
            m_hat = self._m[i] / (1 - b1**t)
            v_hat = self._v[i] / (1 - b2**t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        return float(np.where(absd <= delta,
                               0.5 * diff**2,
                               delta * (absd - 0.5 * delta)).mean())
    
    def copy_weights_from(self, other: "_NumpyMLP"):
        for ps, po in zip(self._params, other._params):
            ps[:] = po

    def get_weights(self):
        return [p.copy() for p in self._params]
    
    def set_weights(self, weights):
        for p, w in zip(self._params, weights):
            p[:] = w

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((
            np.asarray(obs, dtype=np.float64),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float64),
            float(done),
        ))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.stack(obs) .astype(np.float64),
            np.array(act, dtype=np.int32),
            np.array(rew, dtype=np.float64),
            np.stack(nobs).astype(np.float64),
            np.array(done, dtype=np.float64),
        )

    def __len__(self):
        return len(self.buf)
    
class DQNAgent:
    _mat_eps_start   = 1.0
    _mat_eps_decay   = 0.005
    _mat_eps_min     = 0.01
    _mat_LR          = 0.001
    _mat_L2          = 0.0001
    _mat_gamma       = 0.99
    _mat_batch       = 64
    _mat_buffer      = 100_000
    _mat_target_freq = 4

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        lr: float = _mat_LR,
        l2: float = _mat_L2,
        gamma: float = _mat_gamma,
        batch_size: int = _mat_batch,
        buffer_capacity: int = _mat_buffer,
        target_update_freq: int = _mat_target_freq,
        eps_start: float = _mat_eps_start,
        eps_min: float = _mat_eps_min,
        eps_decay: float = _mat_eps_decay,
        hidden: int = 128,
        seed: int = 0,
        **_,
    ):
        self.num_actions        = num_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.lr                 = lr
        self.l2                 = l2
        self.eps                = float(eps_start)
        self.eps_min            = float(eps_min)
        self.eps_decay          = float(eps_decay)
        self._grad_steps        = 0

        self.online = _NumpyMLP(obs_dim, num_actions, hidden=hidden, seed=seed)
        self.target = _NumpyMLP(obs_dim, num_actions, hidden=hidden, seed=seed + 1)
        self.target.copy_weights_from(self.online)

        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.eps:
            return random.randint(0, self.num_actions - 1)
        q = self.online.predict(obs[np.newaxis, :])[0]
        return int(np.argmax(q))
    
    def select_actions(self, obs_array: np.ndarray,
                       greedy: bool = False) -> np.ndarray:
        """obs_array: (N, obs_dim)  →  actions: (N,)"""
        return np.array(
            [self.select_action(obs_array[i], greedy)
             for i in range(len(obs_array))],
            dtype=int,
        )
    
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def store_swarm(
            self, 
            obs_arr: np.ndarray, 
            action_arr: np.ndarray, 
            reward_arr: np.ndarray, 
            next_obs_arr: np.ndarray,
            done_arr: np.ndarray,
    ):
        for i in range(len(obs_arr)):
            self.store(obs_arr[i], action_arr[i], reward_arr[i], next_obs_arr[i], done_arr[i])

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)
        B = self.batch_size

        best_a = np.argmax(self.online.predict(nobs), axis=1)
        q_next   = self.target.predict(nobs)[np.arange(B), best_a]
        td_target = (rew + self.gamma * q_next * (1.0 - done)).astype(np.float64)

        q_pred = self.online.predict(obs)
        q_target = q_pred.copy()
        q_target[np.arange(B), act] = td_target

        loss = self.online.update(q_pred, q_target, lr=self.lr, l2=self.l2)

        self._grad_steps += 1
        
        if self._grad_steps % self.target_update_freq == 0:
            self.target.copy_weights_from(self.online)

        return loss
    
    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps - self.eps_decay)

    def save(self, path: str):
        np.save(path, {
            "online_w": self.online.get_weights(),
            "target_w": self.target.get_weights(),
            "eps": self.eps,
            "grad_steps": self._grad_steps,
        }, allow_pickle=True)
        print(f" [DQNAgent] Saved -> {path}")
    
    def load(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        self.online.set_weights(data["online_w"])
        self.target.set_weights(data["target_w"])
        self.eps = float(data.get("eps", self.eps_min))
        self._grad_steps = int(data.get("grad_steps", 0))
        print(f" [DQNAgent] Loaded -> {path}")