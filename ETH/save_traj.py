
"""test models nocol
and train cmd"""
import multiprocessing
import os
import glob
from pathlib import Path
import torch
import argparse
import subprocess


def get_cmds(args):
    cmds = []
    for dset_i, dset in enumerate(['eth', 'hotel', 'univ', 'zara1', 'zara2']):
        cmd = f'python test.py --cfg {dset}  --info reproduce'#  --gpu {dset_i % torch.cuda.device_count()}'
        cmds.append(cmd)
    return cmds

def spawn(cmds, args):
    """launch cmds in separate threads, max_cmds_at_a_time at a time, until no more cmds to launch"""
    print(f"launching at most {args.max_cmds_at_a_time} cmds at a time:")

    sps = []
    num_gpus = len(args.gpus_available)
    total_cmds_launched = args.start_from  # total cmds launched so far
    gpu_i = 0

    while total_cmds_launched < len(cmds):
        cmd = cmds[total_cmds_launched]
        # assign gpu and launch on separate thread
        # gpu_i = args.gpus_available[total_cmds_launched % num_gpus]
        ngpn = args.num_gpus_per_node
        gpu_str = []
        for _ in range(ngpn):
            gpu_str.append(str(args.gpus_available[gpu_i % num_gpus]))
            gpu_i += 1
        gpu_str = ','.join(gpu_str)
        print(gpu_str, cmd)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_str}
        if not args.trial:
            if args.redirect_output:
                output_filename = 'logs_output'
                # output_filename = cmd.replace(' ', '_').replace('/', '_').replace('=', '_').replace('-', '_').replace('.', '_')
                cmd = f"sudo {cmd} >> {output_filename}.txt 2>&1"
            sp = subprocess.Popen(cmd, env=env, shell=True)
            sps.append(sp)
            if len(sps) >= args.max_cmds_at_a_time:
                # this should work if all subprocesses take the same amount of time;
                # otherwise we might be waiting longer than necessary
                sps[0].wait()
                sps = sps[1:]
        total_cmds_launched += 1

    print("total cmds launched:", total_cmds_launched)
    [sp.wait() for sp in sps]
    print(f"finished all {total_cmds_launched} processes")


def main(args):
    if args.wandb:
        cmds = get_cmds_wandb(args)
    else:
        cmds = get_cmds(args)

    spawn(cmds, args)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_cmds', '-mc', type=int, default=10000)
    argparser.add_argument('--max_cmds_at_a_time', '-c', type=int, default=max(1, multiprocessing.cpu_count()-3))
    argparser.add_argument('--start_from', '-sf', type=int, default=0)
    argparser.add_argument('--train', '-tr', action='store_true')
    argparser.add_argument('--num_gpus_per_node', '-ng', type=int, default=1)
    argparser.add_argument('--cfgs', '-cf', nargs='+', default=None)
    argparser.add_argument('--glob_str', '-g', nargs='+', default=None)
    argparser.add_argument('--wandb', '-w', action='store_true')
    try:
        cuda_visible = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    except KeyError:
        cuda_visible = list(range(torch.cuda.device_count()))
    argparser.add_argument('--gpus_available', '-ga', nargs='+', type=int, default=cuda_visible)
    argparser.add_argument('--no_trial', '-nt', dest='trial', action='store_false', help='if not trial, then actually run the commands')
    argparser.add_argument('--redirect_output', '-ro', action='store_true')
    argparser.add_argument('--recompute_test', '-rt', action='store_true')
    argparser.add_argument('--extra_flags', '-xf', nargs='+', default=None)

    main(argparser.parse_args())
