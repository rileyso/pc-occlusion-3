"""Cache a timestamp-ordered DAIR batch using one model load per requested run.

Keeps the full official split intact for stage-one indexing. Each frame JSON is
compatible with coalign_visualisation and coalign_ego_view; manifest.json records
playback timestamps and inference provenance. Rerunning resumes the same cache.


   python -u -m dairv2x.cache_sequence \
     --coalign-repo /home/riley/CoAlign \
     --single /path/to/DAIR_pointpillar_single_checkpoint \
     --root data/coalign_full \
     --split val \
     --batch-id 106 \
     --output data/coalign_sequences/batch_106_single
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from dairv2x.tools.coalign import infer_frames, result_record


def select_sequence(pairs, vehicle_info, infrastructure_info, split_ids, batch_id, max_gap=0.2):
    """Select one acquisition batch, rejecting gaps instead of silently stitching scenes."""
    paired = {Path(p['vehicle_pointcloud_path']).stem: (i, p) for i, p in enumerate(pairs)}
    infrastructure = {Path(p['pointcloud_path']).stem: p for p in infrastructure_info}
    allowed = set(split_ids)
    frames = []
    for row in vehicle_info:
        frame_id = Path(row['pointcloud_path']).stem
        if str(row['batch_id']) != str(batch_id) or frame_id not in paired or frame_id not in allowed:
            continue
        index, pair = paired[frame_id]
        infra_id = Path(pair['infrastructure_pointcloud_path']).stem
        infra = infrastructure[infra_id]
        frames.append({
            'vehicle_frame_id': frame_id, 'infrastructure_frame_id': infra_id,
            'pair_index': index, 'timestamp_us': int(row['pointcloud_timestamp']),
            'infrastructure_timestamp_us': int(infra['pointcloud_timestamp']),
            'intersection': row['intersection_loc'], 'infrastructure_batch_id': str(infra['batch_id']),
            'prediction_file': f'{frame_id}.json',
        })
    frames.sort(key=lambda f: f['timestamp_us'])
    if not frames:
        raise ValueError(f'No paired split frames in batch {batch_id}')
    if len({f['vehicle_frame_id'] for f in frames}) != len(frames):
        raise ValueError('Duplicate vehicle frames in sequence metadata')
    if len({(f['intersection'], f['infrastructure_batch_id']) for f in frames}) != 1:
        raise ValueError('Selection crosses intersections or infrastructure batches')
    for a, b in zip(frames, frames[1:]):
        if not 0 < b['timestamp_us'] - a['timestamp_us'] <= max_gap * 1e6:
            raise ValueError(f"Timestamp gap at {a['vehicle_frame_id']} -> {b['vehicle_frame_id']}; choose another batch")
        if b['infrastructure_timestamp_us'] <= a['infrastructure_timestamp_us']:
            raise ValueError('Infrastructure timestamps are not strictly increasing')
    return frames


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--coalign-repo', type=Path, required=True)
    parser.add_argument('--coalign', type=Path, help='collaborative CoAlign checkpoint')
    parser.add_argument('--single', type=Path, help='vehicle-only checkpoint')
    parser.add_argument('--stage1', type=Path, help='required with --coalign')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.iou <= 1:
        parser.error('--iou must be between 0 and 1')
    if args.coalign is None and args.single is None:
        parser.error('provide --coalign, --single, or both')
    if args.coalign is not None and args.stage1 is None:
        parser.error('--stage1 is required with --coalign')
    root, repo, output = args.root.resolve(), args.coalign_repo.resolve(), args.output.resolve()
    coalign_model = args.coalign.resolve() if args.coalign else None
    single_model = args.single.resolve() if args.single else None
    stage1 = args.stage1.resolve() if args.stage1 else None
    if not (repo / 'opencood').is_dir():
        parser.error('Not a CoAlign checkout')
    read_json = lambda path: json.loads(path.read_text())
    pairs = sorted(read_json(root / 'cooperative/data_info.json'), key=lambda p: p['vehicle_pointcloud_path'])
    split_ids = read_json(root / f'{args.split}.json')
    frames = select_sequence(pairs, read_json(root / 'vehicle-side/data_info.json'),
                             read_json(root / 'infrastructure-side/data_info.json'), split_ids, args.batch_id)
    if stage1 is not None:
        stage_indices = read_json(stage1)
        indices = {frame_id: i for i, frame_id in enumerate(split_ids)}
        for frame in frames:
            if str(indices[frame['vehicle_frame_id']]) not in stage_indices:
                raise ValueError(f"Missing stage-one entry for {frame['vehicle_frame_id']}")
    models = [p for p in (coalign_model, single_model) if p is not None]
    inputs = []
    for model in models:
        weights = sorted(model.glob('*.pth'))
        if not weights:
            raise FileNotFoundError(f'No checkpoint weights in {model}')
        inputs.extend([model / 'config.yaml', *weights])
    if stage1 is not None:
        inputs.append(stage1)
    inputs.extend([root / f'{args.split}.json', root / 'cooperative/data_info.json',
                   root / 'vehicle-side/data_info.json', root / 'infrastructure-side/data_info.json',
                   Path(__file__), Path(__file__).with_name('coalign.py')])
    manifest = {
        'batch_id': args.batch_id, 'split': args.split, 'root': str(root),
        'model': str(coalign_model or single_model), 'coalign_repo': str(repo),
        'stage1': str(stage1) if stage1 else None,
        'coalign_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'iou_threshold': args.iou, 'numpy_seed_per_frame': 303,
        'noise_policy': 'checkpoint config unchanged; fixed per-frame seed, not a temporally simulated noise process',
        'sha256': {str(p): sha256(p) for p in inputs}, 'frames': frames,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError('Cache provenance differs; use a new output directory')
    elif any(output.glob('*.json')):
        raise ValueError('Output contains JSON without a matching manifest; use a new directory')
    else:
        save_json(manifest_path, manifest)
    pending = []
    prediction_key = 'vehicle_only' if coalign_model is None else 'coalign'
    for frame in frames:
        path = output / frame['prediction_file']
        if path.exists():
            cached = read_json(path)
            if (cached['vehicle_frame_id'] != frame['vehicle_frame_id']
                    or cached['pair_index'] != frame['pair_index'] or cached['split'] != args.split
                    or cached['iou_threshold'] != args.iou
                    or prediction_key not in cached['predictions']):
                raise ValueError(f'Cache identity mismatch: {path}')
        else:
            pending.append(frame)
    print(f"Batch {args.batch_id}: {len(frames)} frames, {len(pending)} pending, "
          f"{(frames[-1]['timestamp_us'] - frames[0]['timestamp_us']) / 1e6:.3f} seconds", flush=True)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    if pending:
        frame_ids = [f['vehicle_frame_id'] for f in pending]
        collaborative = (infer_frames(coalign_model, root, args.split, frame_ids,
                                      'coalign', stage1, args.iou) if coalign_model else None)
        vehicle_only = (infer_frames(single_model, root, args.split, frame_ids,
                                     'single', None, args.iou) if single_model else None)
        for count, frame in enumerate(pending, len(frames) - len(pending) + 1):
            primary = next(collaborative) if collaborative else next(vehicle_only)
            single = next(vehicle_only) if collaborative and vehicle_only else None
            record = result_record(frame['pair_index'], pairs[frame['pair_index']], args.split,
                                   args.iou, primary, single, prediction_key=prediction_key)
            save_json(output / frame['prediction_file'], record)
            print(f"[{count}/{len(frames)}] {frame['vehicle_frame_id']}: "
                  f"{len(record['predictions'][prediction_key])} predictions", flush=True)
    print(f'Cache complete: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
