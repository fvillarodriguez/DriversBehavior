import os
import subprocess
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xml.etree.ElementTree as ET
import json
import time
from dataclasses import dataclass

from src.utils import FlowColumns

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except Exception:
    torch = None
    class _UnavailableNN:
        Module = object

        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar MA-AIRL.")

    nn = _UnavailableNN()
    optim = None
    Normal = None

ROOT_DIR = Path(__file__).resolve().parents[1]

# Define parameter ranges for calibration (Actions)
PARAM_BOUNDS = {
    "maxSpeed": (10.0, 50.0), # m/s (36 - 180 km/h)
    "accel": (0.5, 4.0),
    "decel": (2.0, 6.0),
    "minGap": (1.0, 5.0),
    "sigma": (0.0, 1.0),
    "impatience": (0.0, 1.0)
}

PARAM_KEYS = list(PARAM_BOUNDS.keys())

class ClusterAgent:
    """
    Agent representing a driver cluster.
    Uses Cross-Entropy Method (CEM) or simple Gaussian distribution to sample parameters.
    """
    def __init__(self, cluster_id: str, init_params: Optional[Dict[str, float]] = None, target_stats: Optional[Dict[str, float]] = None):
        self.cluster_id = cluster_id
        self.target_stats = target_stats or {}
        
        # Initialize means based on target stats if available
        self.param_means = np.array([np.mean(PARAM_BOUNDS[k]) for k in PARAM_KEYS])
        
        if self.target_stats:
            # Map observed stats to SUMO params (Heuristic mappings)
            if "avg_speed_kmh" in self.target_stats:
                # maxSpeed in SUMO is m/s. We set it slightly higher than avg speed to allow flow.
                target_speed_ms = self.target_stats["avg_speed_kmh"] / 3.6
                if "maxSpeed" in PARAM_KEYS:
                    idx = PARAM_KEYS.index("maxSpeed")
                    # Set maxSpeed mean ~ 1.1x avg speed, clamped to bounds
                    val = np.clip(target_speed_ms * 1.1, PARAM_BOUNDS["maxSpeed"][0], PARAM_BOUNDS["maxSpeed"][1])
                    self.param_means[idx] = val
            
            if "avg_headway_s" in self.target_stats:
                # tau (reaction time) or minGap. Let's influence minGap.
                # Smaller headway -> Smaller gap
                target_headway = self.target_stats["avg_headway_s"]
                if "minGap" in PARAM_KEYS:
                    idx = PARAM_KEYS.index("minGap")
                    # Heuristic: minGap ~ headway * 2 (rough approx for gaps at speed)
                    val = np.clip(target_headway * 0.5, PARAM_BOUNDS["minGap"][0], PARAM_BOUNDS["minGap"][1])
                    self.param_means[idx] = val

            if "lane_change_rate" in self.target_stats:
                 # Higher LC rate -> Higher sigma or impatience?
                 lc_rate = self.target_stats["lane_change_rate"]
                 if "sigma" in PARAM_KEYS:
                     idx = PARAM_KEYS.index("sigma")
                     val = np.clip(lc_rate, 0.1, 0.9) # Normalize if needed
                     self.param_means[idx] = val
        
        # Override with manual init_params if provided
        if init_params:
            for k, v in init_params.items():
                if k in PARAM_KEYS:
                    self.param_means[PARAM_KEYS.index(k)] = v

        self.param_stds = np.array([
            (PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0]) / 6.0 # Tighter initial std if we have targets
            for k in PARAM_KEYS
        ])
        self.best_params = self.param_means.copy()
        self.best_reward = -float('inf')

    def sample_action(self) -> Dict[str, float]:
        """Sample parameters from current distribution."""
        sample = np.random.normal(self.param_means, self.param_stds)
        # Clip to bounds
        for i, k in enumerate(PARAM_KEYS):
            low, high = PARAM_BOUNDS[k]
            sample[i] = np.clip(sample[i], low, high)
        return dict(zip(PARAM_KEYS, sample))

    def update(self, elite_samples: List[np.ndarray]):
        """Update distribution based on elite samples (CEM)."""
        if not elite_samples:
            return
        elites = np.array(elite_samples)
        new_mean = np.mean(elites, axis=0)
        new_std = np.std(elites, axis=0)
        # Smooth update
        alpha = 0.7
        self.param_means = alpha * new_mean + (1 - alpha) * self.param_means
        self.param_stds = alpha * new_std + (1 - alpha) * self.param_stds
        # Ensure minimum exploration
        min_std = 0.05
        self.param_stds = np.maximum(self.param_stds, min_std)

class MARLManager:
    def __init__(self, simulation_dir: Path, sumo_cmd: List[str] = None):
        self.simulation_dir = simulation_dir
        self.sumo_cmd = sumo_cmd or ["sumo"]
        self.agents: Dict[str, ClusterAgent] = {}
        self.history: List[Dict] = []
        
    def add_agent(self, cluster_id: str, target_stats: Optional[Dict[str, float]] = None):
        self.agents[cluster_id] = ClusterAgent(cluster_id, target_stats=target_stats)

    def generate_vtypes(self, actions: Dict[str, Dict[str, float]]) -> Path:
        """
        Generate a partial XML with vTypes for the simulation.
        actions: {cluster_id: {param: value}}
        """
        root = ET.Element("additional")
        for cluster_id, params in actions.items():
            # SUMO vType definition
            attr = {"id": f"vType_{cluster_id}"}
            attr.update({k: f"{v:.4f}" for k, v in params.items()})
            # Add defaults for safety
            attr["vClass"] = "passenger"
            attr["color"] = "0,1,0" # Dynamic color could be nice
            ET.SubElement(root, "vType", attrib=attr)
        
        path = self.simulation_dir / "marl_vtypes.add.xml"
        tree = ET.ElementTree(root)
        tree.write(path)
        return path

    def run_episode(self, episode_id: int) -> Dict[str, float]:
        """
        Run one simulation episode.
        1. Sample actions for all agents.
        2. Write vTypes.
        3. Run SUMO.
        4. Compute Reward.
        """
        # 1. Sample
        actions = {cid: ag.sample_action() for cid, ag in self.agents.items()}
        
        # 2. Write
        self.generate_vtypes(actions)
        
        # 3. Run SUMO
        # We assume the main sumo cfg includes marl_vtypes.add.xml
        # and has output tripinfo.xml
        output_tripinfo = self.simulation_dir / f"tripinfo_{episode_id}.xml"
        cmd = self.sumo_cmd + [
            "--additional-files", str(self.simulation_dir / "marl_vtypes.add.xml"),
            "--tripinfo-output", str(output_tripinfo),
            "--no-step-log", "true"
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"SUMO failed: {e}")
            return {}

        # 4. Compute Reward (Placeholder)
        # Parse tripinfo
        try:
            df_trips = self._parse_tripinfo(output_tripinfo)
            reward_dict = self._compute_reward(df_trips)
        except Exception as e:
            print(f"Reward comp failed: {e}")
            reward_dict = {cid: -100.0 for cid in self.agents}

        # Log
        self.history.append({
            "episode": episode_id,
            "actions": actions,
            "rewards": reward_dict
        })
        
        return reward_dict

    def _parse_tripinfo(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        # Simple parsing logic
        # In a real scenario, use xml parsing
        try:
            tree = ET.parse(path)
            rows = []
            for child in tree.getroot():
                if child.tag == "tripinfo":
                    rows.append(child.attrib)
            return pd.DataFrame(rows)
        except:
            return pd.DataFrame()

    def _compute_reward(self, df_trips: pd.DataFrame) -> Dict[str, float]:
        """
        Compute reward based on difference between target stats and sim stats.
        Currently a dummy reward for demonstration.
        """
        rewards = {}
        for cid in self.agents:
            # Filter trips by vType ~ cluster
            # Assuming vType name contains cluster_id
            mask = df_trips["vType"].str.contains(cid) if "vType" in df_trips.columns else pd.Series([False]*len(df_trips))
            
            if mask.any():
                # Reward: Minimize difference from target speed (e.g. 50km/h)
                # This requires Real Data target. Hardcoding for now.
                sim_speed = pd.to_numeric(df_trips.loc[mask, "arrivalSpeed"], errors='coerce').mean()
                target_speed = 13.89 # 50 km/h
                reward = -abs(sim_speed - target_speed)
            else:
                reward = -50.0 # No vehicles finished
            rewards[cid] = reward
        return rewards


# =========================
# MA-AIRL (Multi-Agent Adversarial IRL) core
# =========================

DEFAULT_STATE_COLS = (
    "state_speed_kmh",
    "state_lane",
    "state_headway_s",
    "state_relative_speed_kmh",
    "flow_context_vehicle_count",
    "flow_context_mean_speed_kmh",
    "flow_context_density_veh_per_km",
    "vms_active",
)
DEFAULT_ACTION_COLS = (
    "action_delta_speed_kmh",
    "action_accel_m_s2",
    "action_lane_change",
)
DEFAULT_ACTION_BOUNDS = {
    "action_delta_speed_kmh": (-80.0, 80.0),
    "action_accel_m_s2": (-8.0, 8.0),
    "action_lane_change": (0.0, 1.0),
}


def _resolve_device(prefer: str = "auto") -> str:
    if torch is None:
        return "cpu"
    if prefer == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _safe_numpy(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    values = df[cols].copy()
    for col in cols:
        if values[col].dtype == bool:
            values[col] = values[col].astype(float)
    return values.to_numpy(dtype=np.float32, copy=True)


def _normalize_action_array(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    denom = (highs - lows)
    denom = np.where(denom == 0, 1.0, denom)
    normalized = 2.0 * (values - lows) / denom - 1.0
    return np.clip(normalized, -1.0, 1.0)


def _denormalize_action_array(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return lows + (values + 1.0) * 0.5 * (highs - lows)


def _infer_feature_cols(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> List[str]:
    if feature_cols:
        return [c for c in feature_cols if c in df.columns]
    cols = [c for c in DEFAULT_STATE_COLS if c in df.columns]
    if cols:
        return cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"cluster_label", "agent_id", *DEFAULT_ACTION_COLS}
    return [c for c in numeric_cols if c not in excluded and not c.startswith("next_")]


def _infer_action_cols(df: pd.DataFrame, action_cols: Optional[List[str]] = None) -> List[str]:
    if action_cols:
        return [c for c in action_cols if c in df.columns]
    return [c for c in DEFAULT_ACTION_COLS if c in df.columns]


def _next_feature_col(feature_col: str, df: pd.DataFrame) -> str:
    if feature_col.startswith("state_"):
        candidate = "next_" + feature_col[len("state_") :]
        if candidate in df.columns:
            return candidate
    if feature_col in df.columns:
        return feature_col
    raise ValueError(f"No se encontró columna de siguiente estado para '{feature_col}'.")


def _resolve_agent_col(df: pd.DataFrame, agent_col: Optional[str]) -> str:
    if agent_col and agent_col in df.columns:
        return agent_col
    for candidate in ("agent_id", "vehicle_class", "cluster_label", "vehicle_type_id"):
        if candidate in df.columns:
            return candidate
    return "__shared_agent"


def _action_bounds(df: pd.DataFrame, action_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    lows = []
    highs = []
    for col in action_cols:
        if col in DEFAULT_ACTION_BOUNDS:
            low, high = DEFAULT_ACTION_BOUNDS[col]
        else:
            values = pd.to_numeric(df[col], errors="coerce")
            low = float(values.min())
            high = float(values.max())
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = -1.0, 1.0
        lows.append(low)
        highs.append(high)
    return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)


class StandardScaler:
    def __init__(self, eps: float = 1e-6):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted.")
        return (values - self.mean) / self.std

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)


@dataclass
class MAIRLExpertDataset:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray


def build_expert_datasets(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    action_cols: Optional[List[str]] = None,
    agent_ids: Optional[List[str]] = None,
    agent_col: Optional[str] = None,
) -> Tuple[Dict[str, MAIRLExpertDataset], StandardScaler, List[str], List[str], np.ndarray, np.ndarray]:
    df = df.copy()
    resolved_agent_col = _resolve_agent_col(df, agent_col)
    if resolved_agent_col == "__shared_agent":
        df[resolved_agent_col] = "shared_policy"
    df[resolved_agent_col] = df[resolved_agent_col].astype(str)

    feature_cols = _infer_feature_cols(df, feature_cols)
    if not feature_cols:
        raise ValueError("No se encontraron columnas numéricas para usar como estado.")
    next_feature_cols = [_next_feature_col(col, df) for col in feature_cols]

    action_cols = _infer_action_cols(df, action_cols)
    if not action_cols:
        raise ValueError(
            "El dataset de expertos debe contener acciones segment-level "
            "(action_delta_speed_kmh, action_accel_m_s2, action_lane_change)."
        )

    clean_df = df.dropna(subset=feature_cols + next_feature_cols + action_cols)
    if clean_df.empty:
        raise ValueError("No hay transiciones válidas para entrenar (NaNs en estado, acción o siguiente estado).")

    lows, highs = _action_bounds(clean_df, action_cols)

    scaler = StandardScaler()
    states_raw = _safe_numpy(clean_df, feature_cols)
    next_states_raw = _safe_numpy(clean_df, next_feature_cols)
    all_states = np.vstack([states_raw, next_states_raw])
    scaler.fit(all_states)

    datasets: Dict[str, MAIRLExpertDataset] = {}
    group = clean_df.groupby(resolved_agent_col)

    if agent_ids:
        agent_ids = [str(a) for a in agent_ids]

    for group_id, group_df in group:
        if agent_ids and group_id not in agent_ids:
            continue
        states = scaler.transform(_safe_numpy(group_df, feature_cols))
        next_states = scaler.transform(_safe_numpy(group_df, next_feature_cols))
        actions = _safe_numpy(group_df, action_cols)
        actions_norm = _normalize_action_array(actions, lows, highs)
        datasets[group_id] = MAIRLExpertDataset(
            states=states,
            actions=actions_norm,
            next_states=next_states,
        )

    if not datasets:
        raise ValueError("No se encontraron clústeres con datos válidos para entrenamiento.")

    return datasets, scaler, feature_cols, action_cols, lows, highs


class MAIRLDataloader:
    def __init__(self, dataset: MAIRLExpertDataset, device: str):
        if torch is None:
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar MA-AIRL.")
        self.states = torch.tensor(dataset.states, dtype=torch.float32, device=device)
        self.actions = torch.tensor(dataset.actions, dtype=torch.float32, device=device)
        self.next_states = torch.tensor(dataset.next_states, dtype=torch.float32, device=device)
        self.num_samples = self.states.shape[0]

    def sample(self, batch_size: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        idx = torch.randint(0, self.num_samples, (batch_size,), device=self.states.device)
        return self.states[idx], self.actions[idx], self.next_states[idx]

    def sample_states(self, batch_size: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        idx = torch.randint(0, self.num_samples, (batch_size,), device=self.states.device)
        return self.states[idx], self.next_states[idx]

    def mean_state(self) -> "torch.Tensor":
        return self.states.mean(dim=0, keepdim=True)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.backbone = _MLP(state_dim, hidden_sizes, action_dim * 2)
        self.action_dim = action_dim

    def forward(self, state: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        out = self.backbone(state)
        mean, log_std = out[..., : self.action_dim], out[..., self.action_dim :]
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def sample(self, state: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, state: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
        mean, log_std = self(state)
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(action).sum(dim=-1)


class RewardNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.net = _MLP(state_dim + action_dim, hidden_sizes, 1)

    def forward(self, state: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class PotentialNet(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.net = _MLP(state_dim, hidden_sizes, 1)

    def forward(self, state: "torch.Tensor") -> "torch.Tensor":
        return self.net(state).squeeze(-1)


class MAIRLAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...],
        gamma: float,
        policy_lr: float,
        reward_lr: float,
        device: str,
    ):
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_sizes).to(device)
        self.reward_net = RewardNet(state_dim, action_dim, hidden_sizes).to(device)
        self.potential_net = PotentialNet(state_dim, hidden_sizes).to(device)
        self.gamma = gamma
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=policy_lr)
        reward_params = list(self.reward_net.parameters()) + list(self.potential_net.parameters())
        self.reward_opt = optim.Adam(reward_params, lr=reward_lr)

    def f(self, state: "torch.Tensor", action: "torch.Tensor", next_state: "torch.Tensor") -> "torch.Tensor":
        return self.reward_net(state, action) + self.gamma * self.potential_net(next_state) - self.potential_net(state)


class MAIRLManager:
    def __init__(
        self,
        expert_df: pd.DataFrame,
        agent_config: List[Dict[str, Any]],
        feature_cols: Optional[List[str]] = None,
        action_cols: Optional[List[str]] = None,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        gamma: float = 0.99,
        policy_lr: float = 3e-4,
        reward_lr: float = 3e-4,
        device: str = "auto",
    ):
        if torch is None:
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar MA-AIRL.")
        self.device = _resolve_device(device)
        self.agent_config = agent_config
        self.agent_ids = [str(agent["id"]) for agent in agent_config]
        self.agent_names = {str(agent["id"]): agent.get("name", str(agent["id"])) for agent in agent_config}

        datasets, scaler, features, actions, lows, highs = build_expert_datasets(
            expert_df,
            feature_cols=feature_cols,
            action_cols=action_cols,
            agent_ids=self.agent_ids,
        )
        self.feature_cols = features
        self.action_cols = actions
        self.action_lows = lows
        self.action_highs = highs
        self.scaler = scaler
        self.datasets = datasets
        self.loaders = {cid: MAIRLDataloader(ds, self.device) for cid, ds in datasets.items()}

        state_dim = len(self.feature_cols)
        action_dim = len(self.action_cols)
        self.agents: Dict[str, MAIRLAgent] = {}
        for cid in self.datasets:
            self.agents[cid] = MAIRLAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                gamma=gamma,
                policy_lr=policy_lr,
                reward_lr=reward_lr,
                device=self.device,
            )

    def train_step(self, batch_size: int) -> Dict[str, float]:
        metrics = {}
        disc_losses = []
        policy_losses = []
        reward_means = []

        for cid, agent in self.agents.items():
            loader = self.loaders[cid]
            state_e, action_e, next_state_e = loader.sample(batch_size)
            state_p, next_state_p = loader.sample_states(batch_size)
            action_p, logp_p = agent.policy.sample(state_p)

            logp_e = agent.policy.log_prob(state_e, action_e)

            f_e = agent.f(state_e, action_e, next_state_e)
            f_p = agent.f(state_p, action_p.detach(), next_state_p)

            logp_e_det = logp_e.detach()
            logp_p_det = logp_p.detach()

            logsum_e = torch.logsumexp(torch.stack([f_e, logp_e_det], dim=-1), dim=-1)
            logsum_p = torch.logsumexp(torch.stack([f_p, logp_p_det], dim=-1), dim=-1)

            log_d_e = f_e - logsum_e
            log_1_minus_d_p = logp_p_det - logsum_p

            disc_loss = -(log_d_e.mean() + log_1_minus_d_p.mean())
            agent.reward_opt.zero_grad()
            disc_loss.backward()
            agent.reward_opt.step()

            policy_loss = -(f_p.detach() - logp_p).mean()
            agent.policy_opt.zero_grad()
            policy_loss.backward()
            agent.policy_opt.step()

            disc_losses.append(disc_loss.item())
            policy_losses.append(policy_loss.item())
            reward_means.append(f_e.detach().mean().item())

        metrics["disc_loss"] = float(np.mean(disc_losses)) if disc_losses else 0.0
        metrics["policy_loss"] = float(np.mean(policy_losses)) if policy_losses else 0.0
        metrics["reward_mean"] = float(np.mean(reward_means)) if reward_means else 0.0
        metrics["agents"] = float(len(self.agents))
        metrics["transitions"] = float(sum(loader.num_samples for loader in self.loaders.values()))
        return metrics

    def get_policy_actions(self) -> Dict[str, Dict[str, float]]:
        params = {}
        for cid, agent in self.agents.items():
            loader = self.loaders[cid]
            state = loader.mean_state()
            with torch.no_grad():
                mean, _ = agent.policy(state)
            action = mean.squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
            denorm = _denormalize_action_array(action, self.action_lows, self.action_highs)
            params[cid] = {k: float(v) for k, v in zip(self.action_cols, denorm)}
        return params

    def get_policy_params(self) -> Dict[str, Dict[str, float]]:
        """Backward-compatible alias for UI code that displayed policy outputs."""
        return self.get_policy_actions()

    def export_policy_profiles(self, output_path: Path) -> Path:
        profiles = {
            "feature_cols": self.feature_cols,
            "action_cols": self.action_cols,
            "policy_actions": self.get_policy_actions(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        return output_path
