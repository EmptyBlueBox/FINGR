import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from fingr.kinematics import ArmHandModel, DEFAULT_ARM_URDF
from fingr.policy import action_dim, action_indices, cube_angle_to_solved, cube_point_count, cube_turn_start, finger_names, image_size, layer_combinations, layer_complements, make_features, task_indices


DATA_REVISION = '59cb5aca47da3bc4fd3d0f82efa7e84eafc56111'


def sessions(path):
    return {session['episode_index']: session for filename in sorted((Path(path) / 'meta/segments').glob('episode_*.json')) for session in [json.loads(filename.read_text())]}


def open_dataset(path, episodes=None):
    path = Path(path)
    return LeRobotDataset(repo_id='EmptyBlue/FINGR', root=path, episodes=episodes, return_uint8=True, video_backend='pyav')


def data_identity(path, episodes):
    path = Path(path)
    digest = hashlib.sha256(b'fingr-v1')
    digest.update(DEFAULT_ARM_URDF.read_bytes())
    digest.update(np.asarray(sorted(episodes), np.int64).tobytes())
    for filename in sorted([*path.glob('data/**/*.parquet'), *path.glob('videos/**/*.mp4'), *path.glob('meta/segments/*.json')]):
        digest.update(str(filename.relative_to(path)).encode())
        with filename.open('rb') as source:
            digest.update(hashlib.file_digest(source, 'sha256').digest())
    return digest.hexdigest()


def load_data(path, episodes, workers):
    path = Path(path)
    cache_key = data_identity(path, episodes)
    cache_path = Path('output/trajectory-cache') / f'{cache_key}_{image_size}.npz'
    if cache_path.exists():
        print(f"loading cache: {cache_path}", flush=True)
        with np.load(cache_path) as cache:
            data = dict(cache)
        return data
    dataset = open_dataset(path, episodes)
    model = ArmHandModel()
    count = len(dataset)
    data = {
        "features": np.empty((count, 5, 24), np.float32),
        "cube_points": np.empty((count, cube_point_count, 3), np.float32),
        "angle_to_solved": np.empty((count, 1), np.float32),
        "heatmaps": np.empty((count, 3, image_size, image_size), np.uint8),
        "qpos": np.empty((count, action_dim), np.float32),
        "actions": np.empty((count, action_dim), np.float32),
        "episode": np.empty(count, np.int64),
        "segment": np.empty(count, np.int64),
        "segment_end": np.empty(count, np.int64),
        "task": np.empty(count, np.int64),
        "sample_time_ns": np.empty(count, np.int64),
        "valid": np.empty(count, bool),
    }
    loader = DataLoader(dataset, 32, num_workers=workers, generator=torch.Generator())
    turn_starts = {}
    previous_rotations = {}
    recorded_angles = {}
    for episode in episodes:
        session = json.loads((path / "meta/segments" / f"episode_{episode:06d}.json").read_text())
        if "angle_to_solved_deg" in session:
            recorded_angles[episode] = np.asarray(session['angle_to_solved_deg'], np.float64)
    i = 0
    with tqdm(total=count, desc="processing data", unit="frames") as bar:
        for batch in loader:
            for j in range(len(batch["episode_index"])):
                qpos = batch["observation.state"][j].numpy()
                qvel = batch["observation.qvel"][j].numpy()
                torque = batch["observation.torque"][j].numpy()
                cubies = batch["observation.cube_cubie_positions"][j].numpy()
                episode = int(batch["episode_index"][j])
                segment = int(batch["segment_index"][j])
                task = batch["task"][j]
                key = episode, segment
                finite = np.isfinite(cubies).all()
                if finite:
                    data["features"][i], data["cube_points"][i] = make_features(model, qpos, qvel, torque, batch["observation.tactile_force"][j].numpy(), cubies)
                else:
                    data['features'][i] = np.nan
                    data['cube_points'][i] = np.nan
                if episode in recorded_angles:
                    data["angle_to_solved"][i] = recorded_angles[episode][int(batch["frame_index"][j])] / 90
                elif finite:
                    if key not in turn_starts:
                        turn_starts[key] = cube_turn_start(data["cube_points"][i, -8:], task)
                        previous_rotations[key] = 0
                    angle_to_solved, previous_rotations[key] = cube_angle_to_solved(cubies, turn_starts[key], task, previous_rotations[key])
                    data["angle_to_solved"][i] = angle_to_solved / (np.pi / 2)
                else:
                    data['angle_to_solved'][i] = np.nan
                data["heatmaps"][i] = np.stack([cv2.resize(batch[f"observation.images.tactile_heatmap_{finger}"][j, 0].numpy(), (image_size, image_size), interpolation=cv2.INTER_AREA) for finger in finger_names])
                data["qpos"][i] = qpos[action_indices]
                data["actions"][i] = batch["action"][j].numpy()[action_indices]
                data["episode"][i] = episode
                data["segment"][i] = segment
                data["task"][i] = task_indices[task]
                data['sample_time_ns'][i] = int(batch['observation.sample_time_ns'][j])
                data['valid'][i] = finite and np.isfinite(data['angle_to_solved'][i]).all()
                i += 1
            bar.update(len(batch["episode_index"]))
    boundaries = np.r_[0, np.flatnonzero((np.diff(data["episode"]) != 0) | (np.diff(data["segment"]) != 0)) + 1, count]
    for start, end in zip(boundaries, boundaries[1:]):
        data["segment_end"][start:end] = end
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **data)
    print(f"saved cache: {cache_path}", flush=True)
    return data


def off_axis_angles(cubies, task):
    cubies = cubies.astype(np.float64)
    target = cube_turn_start(cubies[0], task)[0]
    layers, complements = cubies[:, layer_combinations], cubies[:, layer_complements]
    vectors = layers.mean(2) - complements.mean(2)
    layers = layers - layers.mean(2, keepdims=True)
    complements = complements - complements.mean(2, keepdims=True)
    _, values, axes = np.linalg.svd(layers, full_matrices=False)
    _, other_values, other_axes = np.linalg.svd(complements, full_matrices=False)
    scale = (values[:, :, 0] + other_values[:, :, 0]) / 2
    scores = values[:, :, 2] + other_values[:, :, 2] + abs(values[:, :, 0] - values[:, :, 1]) + abs(other_values[:, :, 0] - other_values[:, :, 1]) + scale * (1 - abs((axes[:, :, 2] * other_axes[:, :, 2]).sum(2)))
    active = scores.argmin(1)
    index = np.arange(len(cubies))
    vector = vectors[index, active]
    vector /= np.linalg.norm(vector, axis=1)[:, None]
    basis = np.eye(3)[abs(vector).argmin(1)]
    basis -= (basis * vector).sum(1)[:, None] * vector
    basis /= np.linalg.norm(basis, axis=1)[:, None]
    perpendicular = np.cross(vector, basis)
    phases = [np.angle((((points * basis[:, None]).sum(2) + 1j * (points * perpendicular[:, None]).sum(2)) ** 4).sum(1)) / 4 for points in (layers[index, active], complements[index, active])]
    angles = abs(np.rad2deg(np.angle(np.exp(4j * (phases[0] - phases[1]))) / 4))
    return np.where(active == target, 0, angles)


def ensure_data(path):
    from huggingface_hub import snapshot_download

    path = Path(path)
    if not path.exists():
        staging = path.with_name(path.name + '.download')
        snapshot_download('EmptyBlue/FINGR', repo_type='dataset', revision=DATA_REVISION, local_dir=staging, allow_patterns=['data/**', 'meta/**', 'videos/**'])
        staging.rename(path)
    return path


def load_training_data(path, workers):
    metadata = sessions(path)
    episodes = sorted(metadata)
    data = load_data(path, episodes, workers)
    fps = json.loads((Path(path) / 'meta/info.json').read_text())['fps']
    keep = np.zeros(len(data['episode']), bool)
    turns = {'U': 0, 'L': 0}
    for episode, session in metadata.items():
        for segment, step in enumerate(session['steps']):
            indices = np.flatnonzero((data['episode'] == episode) & (data['segment'] == segment))
            if (step['completed'] and len(indices) >= 3
                    and data['valid'][indices[-3:]].all()
                    and (abs(data['angle_to_solved'][indices[-3:], 0]) <= 10 / 90).all()
                    and len(indices) <= 8 * fps
                    and (off_axis_angles(data['cube_points'][indices[data['valid'][indices]], -8:], step['move']) <= 10).all()):
                keep[indices] = True
                turns[step['move']] += 1
    keep &= data['valid']
    data = {key: value[keep] for key, value in data.items()}
    data['segment_end'] = np.cumsum(keep)[data['segment_end'] - 1]
    print(f"episodes={len(episodes)} turns={turns} action_frames={len(data['features'])}", flush=True)
    return data, episodes


def prediction_targets(data, horizons, fps):
    count = len(data['features'])
    future = np.arange(count)[:, None] + np.asarray(horizons)[None]
    indices = np.minimum(future, count - 1)
    broken = ~data['valid'].copy()
    elapsed = np.diff(data['sample_time_ns'])
    broken[1:] |= (np.diff(data['episode']) != 0) | (np.diff(data['segment']) != 0) | (elapsed <= 0) | (elapsed > 1.5e9 / fps)
    boundaries = np.cumsum(broken)
    mask = (future < count) & data['valid'][:, None] & (boundaries[indices] == boundaries[:, None])
    contact = data['features'][:, :3, 15:18].reshape(count, 9)
    targets = np.concatenate((contact[indices] - contact[:, None], data['angle_to_solved'][:, None] - data['angle_to_solved'][indices], data['qpos'][indices] - data['qpos'][:, None]), -1)
    return np.where(mask[..., None], targets, 0).astype(np.float32), mask


def prepare_prediction_data(path, workers, horizons):
    raw = load_data(path, sorted(sessions(path)), workers)
    fps = json.loads((Path(path) / 'meta/info.json').read_text())['fps']
    targets, mask = prediction_targets(raw, horizons, fps)
    keep = mask.any(1)
    data = {key: raw[key][keep] for key in ('features', 'heatmaps', 'cube_points', 'angle_to_solved', 'task')}
    data.update(prediction_targets=targets[keep], prediction_mask=mask[keep])
    scale = np.stack([data['prediction_targets'][data['prediction_mask'][:, i], i].std(0) for i in range(len(horizons))])
    scale = np.maximum(scale, 1e-6)
    data['prediction_targets'] /= scale
    print(f"prediction_frames={keep.sum()} pairs={mask.sum(0).tolist()}", flush=True)
    return data, scale
