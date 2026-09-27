import math
from itertools import combinations

import numpy as np
import torch
from fingr.constants import SHARPA_RIGHT_JOINT_NAMES, TACTILE_NAMES
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F



finger_names = ("index", "ring", "pinky")
finger_joints = tuple(tuple(i for i, name in enumerate(SHARPA_RIGHT_JOINT_NAMES) if f"_{finger}_" in name) for finger in finger_names)
teleop_joints = np.array([i for joints in finger_joints for i in joints])
action_indices = 7 + teleop_joints
tactile_indices = np.array([TACTILE_NAMES.index(finger) for finger in finger_names])
layer_combinations = np.array([group for group in combinations(range(8), 4) if 0 in group])
layer_complements = np.array([np.setdiff1d(np.arange(8), group) for group in layer_combinations])
action_dim = len(action_indices)
image_size = 96
cube_point_count = 32
task_axis_directions = {"U": np.array([1, 1, 0]), "L": np.array([-1, 1, 0])}
task_rotation_signs = {"U": -1, "L": 1}
task_indices = {"U": 0, "L": 1}




def cube_turn_start(cubies, task):
    layers, complements = cubies[layer_combinations], cubies[layer_complements]
    axes = layers.mean(1) - complements.mean(1)
    candidates = np.linalg.norm(axes, axis=1).argsort()[-3:]
    axis = candidates[(abs(axes[candidates] @ task_axis_directions[task]) / np.linalg.norm(axes[candidates], axis=1)).argmax()]
    return axis, cube_angle_to_solved(cubies, (axis, 0, cubies), task, 0)[1], cubies.copy()



def cube_angle_to_solved(cubies, turn_start, task, previous_rotation):
    axis, initial_rotation, initial_cubies = turn_start
    layers, complements = cubies[layer_combinations], cubies[layer_complements]
    _, layer_values, layer_axes = np.linalg.svd(layers - layers.mean(1, keepdims=True), full_matrices=False)
    _, complement_values, complement_axes = np.linalg.svd(complements - complements.mean(1, keepdims=True), full_matrices=False)
    scale = (layer_values[:, 0] + complement_values[:, 0]) / 2
    scores = layer_values[:, 2] + complement_values[:, 2] + abs(layer_values[:, 0] - layer_values[:, 1]) + abs(complement_values[:, 0] - complement_values[:, 1]) + scale * (1 - abs((layer_axes[:, 2] * complement_axes[:, 2]).sum(1)))
    rotation = 0
    if axis == scores.argmin():
        vector = layers[axis].mean(0) - complements[axis].mean(0)
        vector /= np.linalg.norm(vector)
        basis = np.eye(3)[abs(vector).argmin()]
        basis -= (basis @ vector) * vector
        basis /= np.linalg.norm(basis)
        perpendicular = np.cross(vector, basis)
        phases = []
        for points in (layers[axis], complements[axis]):
            points = points - points.mean(0)
            phases.append(np.angle(((points @ basis + 1j * (points @ perpendicular)) ** 4).sum()) / 4)
        rotation = task_rotation_signs[task] * np.angle(np.exp(4j * (phases[0] - phases[1]))) / 4
    rotations = []
    for group in (layer_combinations[axis], layer_complements[axis]):
        first = initial_cubies[group] - initial_cubies[group].mean(0)
        current = cubies[group] - cubies[group].mean(0)
        rotations.append(Rotation.align_vectors(current, first)[0])
    vector = initial_cubies[layer_combinations[axis]].mean(0) - initial_cubies[layer_complements[axis]].mean(0)
    vector /= np.linalg.norm(vector)
    reference = task_rotation_signs[task] * (rotations[1].inv() * rotations[0]).as_rotvec() @ vector
    reference += 2 * np.pi * np.round((previous_rotation - reference) / (2 * np.pi))
    rotation += (np.pi / 2) * np.round((initial_rotation + reference - rotation) / (np.pi / 2))
    return np.pi / 2 - rotation, rotation - initial_rotation



def make_features(model, qpos, qvel, torque, tactile_force, cubies):
    poses = model.link_transforms(qpos[:7], qpos[7:])
    palm = poses["right_hand_C_MC"]
    observed_fingers = (*finger_names, "thumb", "middle")
    tips = np.stack([poses[f"right_{finger}_fingertip"][:3, 3] for finger in observed_fingers])
    features = np.zeros((5, 24), np.float32)
    for i, finger in enumerate(observed_fingers):
        joints = np.array([7 + j for j, name in enumerate(SHARPA_RIGHT_JOINT_NAMES) if f"_{finger}_" in name])
        features[i, :len(joints)] = qpos[joints]
        features[i, 5:5 + len(joints)] = qvel[joints]
        features[i, 10:10 + len(joints)] = torque[joints]
        features[i, 15:21] = tactile_force[TACTILE_NAMES.index(finger)]
    features[:, 21:] = (tips - palm[:3, 3]) @ palm[:3, :3]
    center = cubies.mean(0)
    layers, complements = cubies[layer_combinations], cubies[layer_complements]
    _, layer_values, layer_axes = np.linalg.svd(layers - layers.mean(1, keepdims=True), full_matrices=False)
    _, complement_values, complement_axes = np.linalg.svd(complements - complements.mean(1, keepdims=True), full_matrices=False)
    scale = (layer_values[:, 0] + complement_values[:, 0]) / 2
    scores = layer_values[:, 2] + complement_values[:, 2] + abs(layer_values[:, 0] - layer_values[:, 1]) + abs(complement_values[:, 0] - complement_values[:, 1]) + scale * (1 - abs((layer_axes[:, 2] * complement_axes[:, 2]).sum(1)))
    i = scores.argmin()
    face_centers = []
    for group, other in ((layer_combinations[i], layer_complements[i]), (layer_complements[i], layer_combinations[i])):
        points = cubies[group]
        neighbors = np.linalg.norm(points[:, None] - points[None], axis=2).argsort(1)[:, 1:3]
        face_centers.extend(((1.5 * points[:, None] - 0.5 * points[neighbors]).reshape(-1, 3), points + (points.mean(0) - cubies[other].mean(0)) / 2))
    cube_points = np.concatenate((*face_centers, center + 2 * (cubies - center)))
    return features, ((cube_points - palm[:3, 3]) @ palm[:3, :3]).astype(np.float32)



def normalize_observation(features, cube_points, statistics, local_geometry=False):
    features = (features - statistics[0]) / statistics[1]
    cube_points = (cube_points - statistics[2]) / statistics[3]
    if local_geometry:
        features = torch.cat((features[..., :21].clip(-5, 5), features[..., 21:]), -1)
    else:
        features = features.clip(-5, 5)
        cube_points = cube_points.clip(-5, 5)
    return features, cube_points


class FlowBlock(nn.Module):
    def __init__(self, hidden_dim, heads, feedforward_dim, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, heads, dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, dropout, batch_first=True)
        self.feedforward = nn.Sequential(nn.Linear(hidden_dim, feedforward_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(feedforward_dim, hidden_dim))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim, elementwise_affine=False) for _ in range(3))
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, actions, memory, condition):
        shift_self, scale_self, shift_cross, scale_cross, shift_ff, scale_ff = self.modulation(condition).chunk(6, -1)
        query = self.norms[0](actions) * (1 + scale_self[:, None]) + shift_self[:, None]
        actions = actions + self.self_attention(query, query, query, need_weights=False)[0]
        query = self.norms[1](actions) * (1 + scale_cross[:, None]) + shift_cross[:, None]
        actions = actions + self.cross_attention(query, memory, memory, need_weights=False)[0]
        query = self.norms[2](actions) * (1 + scale_ff[:, None]) + shift_ff[:, None]
        return actions + self.feedforward(query)


class ObservationEncoder(nn.Module):
    def __init__(self, hidden_dim=384, feedforward_dim=1536, encoder_layers=4, dropout=0.0, geometry_dim=64, geometry_pool="mean", prediction_horizons=(1, 5, 10), prediction_mode="dynamics"):
        super().__init__()
        channels = (32, 64, 128, hidden_dim)
        self.tactile_backbone = nn.ModuleList(nn.Conv2d(a, b, 5 if i == 0 else 3, 2, 2 if i == 0 else 1) for i, (a, b) in enumerate(zip((1, *channels[:-1]), channels)))
        self.tactile_film = nn.ParameterList(nn.Parameter(torch.zeros(3, 2, channel)) for channel in channels)
        self.feature_projections = nn.ModuleList(nn.Linear(24, hidden_dim) for _ in range(5))
        self.cube_pointnet = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.angle_to_solved_projection = nn.Linear(1, hidden_dim)
        self.task_embedding = nn.Embedding(len(task_indices), hidden_dim)
        self.finger_embedding = nn.Parameter(torch.randn(5, hidden_dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        encoder = nn.TransformerEncoderLayer(hidden_dim, 8, feedforward_dim, dropout, batch_first=True, norm_first=True)
        self.memory_encoder = nn.TransformerEncoder(encoder, encoder_layers, norm=nn.LayerNorm(hidden_dim), enable_nested_tensor=False)
        self.hidden_dim = hidden_dim
        self.local_geometry = geometry_dim > 0
        self.geometry_pool = geometry_pool
        if self.local_geometry:
            self.local_pointnet = nn.Sequential(nn.Linear(3, geometry_dim), nn.GELU(), nn.Linear(geometry_dim, geometry_dim))
            self.local_projection = nn.Linear(geometry_dim, hidden_dim)
            self.register_buffer("tip_mean", torch.zeros(5, 3))
            self.register_buffer("tip_std", torch.ones(5, 3))
            self.register_buffer("point_mean", torch.zeros(3))
            self.register_buffer("point_std", torch.ones(3))
        self.prediction_horizons = tuple(prediction_horizons)
        self.prediction_mode = prediction_mode
        self.memory_tokens = 9 + (len(self.prediction_horizons) if prediction_mode == "dynamics" else 0)
        if self.prediction_horizons and prediction_mode == "dynamics":
            self.future_queries = nn.Parameter(torch.randn(len(self.prediction_horizons), hidden_dim) * 0.02)

    def set_geometry_statistics(self, statistics):
        if self.local_geometry:
            self.tip_mean.copy_(statistics[0][:, 21:])
            self.tip_std.copy_(statistics[1][:, 21:])
            self.point_mean.copy_(statistics[2])
            self.point_std.copy_(statistics[3])

    def encode_observation(self, features, heatmaps, cube_points, angle_to_solved, task):
        batch = len(features)
        if self.local_geometry:
            tips = features[..., 21:] * self.tip_std + self.tip_mean
            points = cube_points * self.point_std + self.point_mean
            local = self.local_pointnet((points[:, None] - tips[:, :, None]) / 0.05)
            local = local.amax(2) if self.geometry_pool == "max" else local.mean(2)
            local = self.local_projection(local)
            features = features.clip(-5, 5)
            cube_points = cube_points.clip(-5, 5)
        tactile = heatmaps.flatten(0, 1)
        for convolution, film in zip(self.tactile_backbone, self.tactile_film):
            tactile = convolution(tactile).unflatten(0, (batch, 3))
            scale, shift = film.unbind(1)
            tactile = F.gelu(tactile * (1 + scale[None, :, :, None, None]) + shift[None, :, :, None, None]).flatten(0, 1)
        tactile = F.adaptive_avg_pool2d(tactile, 1).flatten(1).unflatten(0, (batch, 3))
        numeric = torch.stack([projection(features[:, i]) for i, projection in enumerate(self.feature_projections)], 1)
        if self.local_geometry:
            numeric = numeric + local
        position = self.finger_embedding[None]
        fingers = torch.stack((numeric[:, :3] + position[:, :3] + self.modality_embedding[None, None, 0], tactile + position[:, :3] + self.modality_embedding[None, None, 1]), 2).flatten(1, 2)
        cube = self.cube_pointnet(cube_points).amax(1) + self.modality_embedding[None, 2]
        tokens = torch.cat((fingers, cube[:, None], numeric[:, 3:] + position[:, 3:] + self.modality_embedding[None, None, 0]), 1) + (self.angle_to_solved_projection(angle_to_solved) + self.task_embedding(task))[:, None]
        if self.prediction_horizons and self.prediction_mode == "dynamics":
            tokens = torch.cat((tokens, self.future_queries[None].expand(batch, -1, -1)), 1)
        return self.memory_encoder(tokens)


class FlowPolicy(ObservationEncoder):
    def __init__(self, hidden_dim=384, feedforward_dim=1536, encoder_layers=4, decoder_layers=4, inference_steps=4, dropout=0.0, action_horizon=20, geometry_dim=64, geometry_pool="mean", prediction_horizons=(1, 5, 10), prediction_mode="dynamics"):
        super().__init__(hidden_dim, feedforward_dim, encoder_layers, dropout, geometry_dim, geometry_pool, prediction_horizons, prediction_mode)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.action_position = nn.Parameter(torch.randn(action_horizon, hidden_dim) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.flow_blocks = nn.ModuleList(FlowBlock(hidden_dim, 8, feedforward_dim, dropout) for _ in range(decoder_layers))
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.x0_head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("time_period", torch.logspace(math.log10(0.004), math.log10(4), hidden_dim // 2))
        self.inference_steps = inference_steps
        self.action_horizon = action_horizon
        if self.prediction_horizons:
            output_dim = action_dim + 10 if prediction_mode == "dynamics" else len(self.prediction_horizons) * 10
            self.prediction_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def predict_velocity(self, noisy_actions, time, memory):
        angle = time[:, None] * (2 * math.pi / self.time_period[None])
        condition = self.time_mlp(torch.cat((angle.sin(), angle.cos()), 1))
        actions = self.action_projection(noisy_actions) + self.action_position[None]
        for block in self.flow_blocks:
            actions = block(actions, memory, condition)
        denoised = self.x0_head(self.action_norm(actions))
        return (noisy_actions - denoised) / time[:, None, None]

    def forward(self, features, heatmaps, cube_points, angle_to_solved, task, noisy_actions=None, time=None):
        memory = self.encode_observation(features, heatmaps, cube_points, angle_to_solved, task)
        if noisy_actions is None:
            if self.prediction_mode == "dynamics":
                return self.prediction_head(memory[:, -len(self.prediction_horizons):])
            return self.prediction_head(memory.mean(1)).reshape(len(features), len(self.prediction_horizons), 10)
        return self.predict_velocity(noisy_actions, time, memory)

    def sample(self, features, heatmaps, cube_points, angle_to_solved, task, noise):
        memory = self.encode_observation(features, heatmaps, cube_points, angle_to_solved, task)
        return self.sample_from_memory(memory, noise)

    def sample_from_memory(self, memory, noise):
        actions = noise
        step_size = -1 / self.inference_steps
        for step in range(self.inference_steps):
            time = torch.full((len(actions),), 1 + step * step_size, device=actions.device)
            actions = actions + step_size * self.predict_velocity(actions, time, memory)
        return actions
