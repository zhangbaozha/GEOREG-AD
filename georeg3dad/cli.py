"""One CLI on Windows and Linux; paths are inputs, never machine constants."""
import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .runtime import configure_threads


def main(argv=None):
    parser = argparse.ArgumentParser(description='GeoReg3DAD: inspect, run or smoke-test either dataset')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect','run','smoke'):
        sub = commands.add_parser(command)
        sub.add_argument('--config', type=Path, required=True)
        sub.add_argument('--preset')
        sub.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
        sub.add_argument('--data-root', type=Path, default=os.environ.get('GEOREG_DATA_ROOT'))
        sub.add_argument('--scope', choices=('all','pcd','new_pcd'))
        sub.add_argument('--categories', nargs='+', default=[])
        sub.add_argument('--input-protocol', choices=('legacy-pcd', 'real3dad-official'),
                         help='Real3D defaults to official TXT xyz plus centering; legacy-pcd replays the old input protocol')
        if command != 'inspect':
            sub.add_argument('--run-root', type=Path, required=True)
            sub.add_argument('--workers', type=int, default=4)
            sub.add_argument('--threads', type=int, default=2)
            sub.add_argument('--resume', action='store_true')
    child = commands.add_parser('_category', help=argparse.SUPPRESS)
    child.add_argument('--run-root', type=Path, required=True)
    child.add_argument('--category', required=True)
    child.add_argument('--threads', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == '_category':
            configure_threads(args.threads)
            from .runner import evaluate_category
            evaluate_category(args.run_root, args.category, args.threads)
            return
        settings, _ = load_config(args.config, args.preset, args.set)
        if args.data_root is None:
            parser.error('Provide --data-root or GEOREG_DATA_ROOT')
        dataset = settings['dataset']
        scope = args.scope or ('pcd' if dataset == 'shapenet' else 'all')
        if dataset == 'real3dad' and scope != 'all':
            parser.error('Real3D uses --scope all')
        from .datasets import inspect_dataset
        categories = args.categories
        if args.command == 'smoke' and not categories:
            from .datasets import protocol
            categories = [r['category'] for r in protocol(dataset)['categories'] if scope=='all' or r['source_split']==scope][:2]
        input_protocol = args.input_protocol or ('real3dad-official' if dataset == 'real3dad' else 'legacy-pcd')
        manifest = inspect_dataset(dataset, args.data_root, categories, scope,
                                   smoke=args.command=='smoke', input_protocol=input_protocol)
        if args.command == 'inspect':
            print(json.dumps({'dataset':dataset,'scope':scope,'categories':len(manifest['categories']),
                'test_scans':manifest['test_scans'],'point_valid_scans':manifest['point_valid_scans'],
                'input_protocol': input_protocol, 'center_inputs': manifest['center_inputs'],
                'configuration':settings}, indent=2))
            return
        configure_threads(args.threads)
        from .runner import run
        run(manifest, settings, args.run_root, args.workers, args.threads, args.resume)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
