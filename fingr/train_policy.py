import argparse
import copy
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fingr.data import ensure_data, load_training_data, prepare_prediction_data
from fingr.policy import FlowPolicy, action_dim, normalize_observation


STATISTICS = ("feature_mean", "feature_std", "cube_point_mean", "cube_point_std", "action_offset", "action_scale")


def compute_statistics(data, episodes, action_horizon):
    rows = np.flatnonzero(np.isin(data["episode"], episodes))
    features = data["features"][rows].astype(np.float64)
    feature_std = features.std(0) + 1e-6
    feature_std[:, :5] = np.maximum(feature_std[:, :5], 0.01)
    cube_points = data["cube_points"][rows].astype(np.float64)
    deltas = []
    for row in rows:
        end = data["segment_end"][row]
        deltas.append(data["actions"][row:min(row + action_horizon, end)] - data["qpos"][row])
    action_q01, action_q99 = np.quantile(np.concatenate(deltas), (0.01, 0.99), axis=0)
    return tuple(torch.tensor(value, dtype=torch.float32) for value in (features.mean(0), feature_std, cube_points.mean((0, 1)), cube_points.std((0, 1)) + 1e-6, (action_q01 + action_q99) / 2, np.maximum((action_q99 - action_q01) / 2, 0.01)))


def action_targets(data, horizon):
    rows = np.arange(len(data["actions"]))[:, None] + np.arange(horizon)
    ends = data["segment_end"][:, None]
    return data["actions"][np.minimum(rows, ends - 1)] - data["qpos"][:, None], rows < ends


def move_batch(batch, statistics, augment=False, local_geometry=False):
    features, cube_points = normalize_observation(batch["features"].cuda(non_blocking=True), batch["cube_points"].cuda(non_blocking=True), statistics, local_geometry)
    angle_to_solved = batch["angle_to_solved"].cuda(non_blocking=True)
    if augment:
        features += torch.randn_like(features) * 0.01
        cube_points += torch.randn_like(cube_points) * 0.01
    return features, batch["heatmaps"].cuda(non_blocking=True).float().div_(255), cube_points, angle_to_solved, batch["task"].cuda(non_blocking=True), batch["actions"].cuda(non_blocking=True), batch["mask"].cuda(non_blocking=True)[..., None]


def policy_loss(model, batch, statistics, time_distribution):
    features, heatmaps, cube_points, angle_to_solved, task, actions, mask = move_batch(batch, statistics, True, model.local_geometry)
    action_offset, action_scale = statistics[-2:]
    normalized_actions = (actions - action_offset) / action_scale
    noise = torch.randn_like(normalized_actions)
    flow_time = time_distribution.sample((len(actions),))
    expanded_time = flow_time[:, None, None]
    noisy_actions = expanded_time * noise + (1 - expanded_time) * normalized_actions
    target_velocity = noise - normalized_actions
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(features, heatmaps, cube_points, angle_to_solved, task, noisy_actions, flow_time)
        flow_loss = ((prediction - target_velocity).square() * mask).sum() / (mask.sum() * action_dim)
        denoised = noisy_actions - expanded_time * prediction
        mse = ((denoised * action_scale + action_offset - actions).square() * mask).sum() / (mask.sum() * action_dim)
        loss = mse / action_scale.square().mean()
    return loss, flow_loss, mse


def prediction_loss(model, batch, statistics):
    features, points = normalize_observation(batch['features'], batch['cube_points'], statistics, model.local_geometry)
    mask = batch['prediction_mask']
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(features, batch['heatmaps'][:, :, None].float() / 255, points, batch['angle_to_solved'], batch['task'])
        squared = (prediction.float() - batch['prediction_targets']).square()
        contact = (squared[..., :9].mean(-1) * mask).sum() / mask.sum()
        progress = (squared[..., 9] * mask).sum() / mask.sum()
        loss = (contact + progress) / 2
        metrics = {'train/contact_loss': contact, 'train/progress_loss': progress}
        if model.prediction_mode == 'dynamics':
            motion = (squared[..., 10:].mean(-1) * mask).sum() / mask.sum()
            loss = (contact + progress + motion) / 3
            metrics['train/motion_loss'] = motion
    return loss, metrics


@torch.no_grad()
def update_ema(model, ema_model):
    torch._foreach_lerp_(list(ema_model.parameters()), list(model.parameters()), 0.01)
    for target, source in zip(ema_model.buffers(), model.buffers()):
        target.copy_(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/training')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='checkpoint')
    args = parser.parse_args()
    path = ensure_data(args.data)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    config = dict(hidden_dim=384, feedforward_dim=1536, encoder_layers=4, decoder_layers=4,
                  inference_steps=4, dropout=0.0, action_horizon=20, geometry_dim=64,
                  geometry_pool='mean', prediction_horizons=[1, 5, 10], prediction_mode='dynamics')
    data, episodes = load_training_data(path, args.workers)
    data['targets'], data['mask'] = action_targets(data, config['action_horizon'])
    statistics = tuple(value.cuda() for value in compute_statistics(data, episodes, config['action_horizon']))
    prediction_data, prediction_scale = prepare_prediction_data(path, args.workers, config['prediction_horizons'])
    prediction_data = {key: torch.from_numpy(value).cuda() for key, value in prediction_data.items()}
    prediction_generator = torch.Generator().manual_seed(args.seed + 1)
    train_frames = len(data['features'])
    data = {key: torch.from_numpy(data[key]).cuda() for key in ('features', 'cube_points', 'angle_to_solved', 'task', 'heatmaps', 'targets', 'mask')}
    loader = DataLoader(range(train_frames), args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    total_steps = len(loader) * args.epochs
    model = FlowPolicy(**config).cuda()
    model.set_geometry_statistics(statistics)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    training_model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4, fused=True)
    warmup_steps = max(1, min(1000, total_steps // 20))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup_steps if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))))
    time_distribution = torch.distributions.Beta(torch.tensor(1.5, device='cuda'), torch.tensor(1.0, device='cuda'))
    output = Path(args.output) / datetime.now().strftime('%y%m%d_%H%M%S')
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps(vars(args) | {'model': config, 'total_steps': total_steps}, indent=2) + '\n')
    step = 0
    with (output / 'metrics.jsonl').open('w', buffering=1) as metrics_file:
        for epoch in range(args.epochs):
            model.train()
            start = time.perf_counter()
            totals = np.zeros(4)
            prediction_batches = 0
            for rows in loader:
                rows = rows.cuda(non_blocking=True)
                batch = {key: data[key][rows] for key in ('features', 'cube_points', 'angle_to_solved', 'task')}
                batch.update(heatmaps=data['heatmaps'][rows, :, None], actions=data['targets'][rows], mask=data['mask'][rows])
                step += 1
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = policy_loss(training_model, batch, statistics, time_distribution)
                loss.backward()
                totals[0] += loss.item()
                prediction_weight = 0.003 * max(0, 1 - step / (total_steps * 0.5))
                if prediction_weight:
                    rows = torch.randint(len(prediction_data['features']), (args.batch_size,), generator=prediction_generator).cuda()
                    prediction_batch = {key: value[rows] for key, value in prediction_data.items()}
                    future_loss, prediction_metrics = prediction_loss(training_model, prediction_batch, statistics)
                    (prediction_weight * future_loss).backward()
                    totals[1:] += [value.item() for value in prediction_metrics.values()]
                    prediction_batches += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                update_ema(model, ema_model)
            torch.cuda.synchronize()
            metrics = dict(epoch=epoch + 1, step=step, action_loss=totals[0] / len(loader),
                           lr=scheduler.get_last_lr()[0], samples_per_second=train_frames / (time.perf_counter() - start))
            if prediction_batches:
                metrics.update(zip(('contact_loss', 'progress_loss', 'motion_loss'), totals[1:] / prediction_batches))
            metrics_file.write(json.dumps(metrics) + '\n')
            print(json.dumps(metrics), flush=True)
            if (epoch + 1) % 100 == 0 or epoch + 1 == args.epochs:
                checkpoint = dict(ema_model=ema_model.state_dict(), model_config=config, train_episodes=episodes,
                                  epoch=epoch + 1, step=step, training_config=vars(args), prediction_scale=prediction_scale)
                checkpoint.update({key: value.cpu() for key, value in zip(STATISTICS, statistics)})
                target = output / f'epoch_{epoch + 1:04d}.pt'
                temporary = target.with_suffix('.tmp')
                torch.save(checkpoint, temporary)
                temporary.replace(target)
    print(f'checkpoint={output}', flush=True)


if __name__ == '__main__':
    main()
