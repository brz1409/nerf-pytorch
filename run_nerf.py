import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import nerfacc

from run_nerf_helpers import *

from load_dataset import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays_chunk(
    rays_o,
    rays_d,
    near,
    far,
    viewdirs,
    network_fn,
    network_query_fn,
    estimator,
    render_step_size,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    render_bkgd=None,
    alpha_thre=0.0,
    cone_angle=0.0,
    near_plane=0.0,
    far_plane=1e10,
    pytest=False,
):
    device = rays_o.device
    n_rays = rays_o.shape[0]

    if estimator is None:
        raise ValueError("render_rays_chunk requires a nerfacc estimator instance.")

    if render_bkgd is None:
        bg_color = torch.ones(3, device=device) if white_bkgd else torch.zeros(3, device=device)
    else:
        bg_color = render_bkgd.to(device)

    render_step_size = float(render_step_size)

    dir_norm = torch.norm(rays_d, dim=-1, keepdim=True).clamp_min(1e-6)
    rays_d_unit = rays_d / dir_norm
    t_min = near.squeeze(-1) * dir_norm.squeeze(-1)
    t_max = far.squeeze(-1) * dir_norm.squeeze(-1)

    def run_rgb_sigma(current_network, t_starts, t_ends, ray_indices, add_noise):
        if t_starts.numel() == 0:
            empty_rgb = torch.empty((0, 3), device=device)
            empty_sigma = torch.empty((0,), device=device)
            return empty_rgb, empty_sigma
        mid = 0.5 * (t_starts + t_ends)
        pts = rays_o[ray_indices] + rays_d_unit[ray_indices] * mid[:, None]
        dirs = viewdirs[ray_indices] if viewdirs is not None else None
        raw = network_query_fn(pts, dirs, current_network)
        sigma_raw = raw[..., 3]
        if add_noise and raw_noise_std > 0.0 and not pytest:
            sigma_raw = sigma_raw + torch.randn_like(sigma_raw) * raw_noise_std
        sigma = F.relu(sigma_raw)
        rgb = torch.sigmoid(raw[..., :3])
        return rgb, sigma

    def sigma_fn(t_starts, t_ends, ray_indices):
        _, sigmas = run_rgb_sigma(network_fn, t_starts, t_ends, ray_indices, add_noise=False)
        return sigmas

    ray_indices, t_starts, t_ends = estimator.sampling(
        rays_o=rays_o,
        rays_d=rays_d_unit,
        sigma_fn=sigma_fn,
        near_plane=near_plane,
        far_plane=far_plane,
        t_min=t_min,
        t_max=t_max,
        render_step_size=render_step_size,
        stratified=perturb > 0.0,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
    )

    if t_starts.numel() == 0:
        rgb_bg = bg_color.expand(n_rays, -1)
        zeros = torch.zeros(n_rays, device=device)
        extras = {
            'rgb0': rgb_bg,
            'disp0': zeros,
            'acc0': zeros,
        }
        return rgb_bg, zeros, zeros, extras

    rgb_coarse, opacity_coarse, depth_coarse, extras_coarse = nerfacc.rendering(
        t_starts,
        t_ends,
        ray_indices,
        n_rays=n_rays,
        rgb_sigma_fn=lambda ts, te, ridx: run_rgb_sigma(
            network_fn, ts, te, ridx, add_noise=True
        ),
        render_bkgd=bg_color,
    )

    acc_coarse = opacity_coarse.squeeze(-1)
    depth_coarse_param = depth_coarse.squeeze(-1) / dir_norm.squeeze(-1)
    eps = 1e-10
    disp_coarse = 1.0 / torch.max(
        eps * torch.ones_like(depth_coarse_param),
        depth_coarse_param / acc_coarse.clamp_min(eps),
    )

    rgb_final = rgb_coarse
    disp_final = disp_coarse
    acc_final = acc_coarse

    extras = {
        'rgb0': rgb_coarse,
        'disp0': disp_coarse,
        'acc0': acc_coarse,
    }

    if N_importance > 0 and network_fine is not None:
        weights_coarse = extras_coarse['weights']
        intervals_coarse, cdf_edges, packed_info = build_nerfacc_intervals(
            ray_indices, t_starts, t_ends, weights_coarse, n_rays
        )

        if intervals_coarse is not None and cdf_edges is not None:
            intervals_fine, _ = nerfacc.importance_sampling(
                intervals_coarse,
                cdf_edges,
                N_importance,
                stratified=perturb > 0.0,
            )

            if intervals_fine.packed_info is None:
                fine_vals = intervals_fine.vals
                n_samples_fine = fine_vals.shape[-1] - 1
                fine_t_starts = fine_vals[..., :-1].reshape(-1)
                fine_t_ends = fine_vals[..., 1:].reshape(-1)
                fine_ray_indices = (
                    torch.arange(n_rays, device=device)
                    .unsqueeze(-1)
                    .expand(-1, n_samples_fine)
                    .reshape(-1)
                )
            else:
                fine_t_starts = intervals_fine.vals[intervals_fine.is_left]
                fine_t_ends = intervals_fine.vals[intervals_fine.is_right]
                fine_ray_indices = intervals_fine.ray_indices[intervals_fine.is_left]

            t_starts_all = torch.cat([t_starts, fine_t_starts], dim=0)
            t_ends_all = torch.cat([t_ends, fine_t_ends], dim=0)
            ray_indices_all = torch.cat([ray_indices, fine_ray_indices], dim=0)
            packed_all = nerfacc.pack_info(ray_indices_all, n_rays=n_rays)
            packed_all_cpu = packed_all.detach().cpu()
            mids_all = 0.5 * (t_starts_all + t_ends_all)
            gather_indices = []
            for ray_id in range(n_rays):
                start_idx = int(packed_all_cpu[ray_id, 0].item())
                count = int(packed_all_cpu[ray_id, 1].item())
                if count == 0:
                    continue
                idx_range = torch.arange(start_idx, start_idx + count, device=device)
                sorted_idx = idx_range[torch.argsort(mids_all[start_idx:start_idx + count])]
                gather_indices.append(sorted_idx)
            if gather_indices:
                gather_indices = torch.cat(gather_indices, dim=0)
                t_starts_all = t_starts_all[gather_indices]
                t_ends_all = t_ends_all[gather_indices]
                ray_indices_all = ray_indices_all[gather_indices]

                rgb_fine, opacity_fine, depth_fine, _ = nerfacc.rendering(
                    t_starts_all,
                    t_ends_all,
                    ray_indices_all,
                    n_rays=n_rays,
                    rgb_sigma_fn=lambda ts, te, ridx: run_rgb_sigma(
                        network_fine, ts, te, ridx, add_noise=True
                    ),
                    render_bkgd=bg_color,
                )

                acc_final = opacity_fine.squeeze(-1)
                depth_final_param = depth_fine.squeeze(-1) / dir_norm.squeeze(-1)
                disp_final = 1.0 / torch.max(
                    eps * torch.ones_like(depth_final_param),
                    depth_final_param / acc_final.clamp_min(eps),
                )
                rgb_final = rgb_fine

    return rgb_final, disp_final, acc_final, extras


def _merge_chunk_extras(extras_list):
    if not extras_list:
        return {}
    merged = {}
    for chunk in extras_list:
        for key, value in chunk.items():
            if value is None:
                continue
            merged.setdefault(key, []).append(value)
    for key, values in merged.items():
        if isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = values
    return merged


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays using nerfacc-based sampling."""
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    else:
        viewdirs = None

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near_full = near * torch.ones_like(rays_d[..., :1])
    far_full = far * torch.ones_like(rays_d[..., :1])

    total_rays = rays_o.shape[0]
    rgb_chunks, disp_chunks, acc_chunks, extras_chunks = [], [], [], []

    estimator = kwargs.get('estimator')
    render_step_size = kwargs.get('render_step_size')
    render_bkgd = kwargs.get('render_bkgd')
    alpha_thre = kwargs.get('occ_alpha_thre', 0.0)
    cone_angle = kwargs.get('occ_cone_angle', 0.0)
    near_plane = kwargs.get('occ_near_plane', 0.0)
    far_plane = kwargs.get('occ_far_plane', 1e10)
    perturb = kwargs.get('perturb', 0.0)
    N_importance = kwargs.get('N_importance', 0)
    network_fine = kwargs.get('network_fine')
    white_bkgd = kwargs.get('white_bkgd', False)
    raw_noise_std = kwargs.get('raw_noise_std', 0.0)
    pytest = kwargs.get('pytest', False)

    for i in range(0, total_rays, chunk):
        slc = slice(i, min(i + chunk, total_rays))
        chunk_viewdirs = viewdirs[slc] if viewdirs is not None else None
        rgb_chunk, disp_chunk, acc_chunk, extras_chunk = render_rays_chunk(
            rays_o[slc],
            rays_d[slc],
            near_full[slc],
            far_full[slc],
            chunk_viewdirs,
            kwargs['network_fn'],
            kwargs['network_query_fn'],
            estimator,
            render_step_size,
            perturb=perturb,
            N_importance=N_importance,
            network_fine=network_fine,
            white_bkgd=white_bkgd,
            raw_noise_std=raw_noise_std,
            render_bkgd=render_bkgd,
            alpha_thre=alpha_thre,
            cone_angle=cone_angle,
            near_plane=near_plane,
            far_plane=far_plane,
            pytest=pytest,
        )
        rgb_chunks.append(rgb_chunk)
        disp_chunks.append(disp_chunk)
        acc_chunks.append(acc_chunk)
        extras_chunks.append(extras_chunk)

    rgb_map = torch.cat(rgb_chunks, dim=0)
    disp_map = torch.cat(disp_chunks, dim=0)
    acc_map = torch.cat(acc_chunks, dim=0)
    extras = _merge_chunk_extras(extras_chunks)

    rgb_map = rgb_map.reshape(list(sh[:-1]) + [3])
    disp_map = disp_map.reshape(list(sh[:-1]))
    acc_map = acc_map.reshape(list(sh[:-1]))

    for key, value in list(extras.items()):
        if isinstance(value, torch.Tensor):
            extras[key] = value.reshape(list(sh[:-1]) + list(value.shape[1:]))

    return rgb_map, disp_map, acc_map, extras


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, scene_aabb):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    roi_aabb = scene_aabb.to(device)
    estimator = nerfacc.OccGridEstimator(
        roi_aabb=roi_aabb,
        resolution=args.occ_grid_resolution,
        levels=args.occ_grid_levels,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None and ckpt.get('network_fine_state_dict') is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if 'estimator_state_dict' in ckpt:
            estimator.load_state_dict(ckpt['estimator_state_dict'])

    ##########################

    bg_color = torch.ones(3, device=device) if args.white_bkgd else torch.zeros(3, device=device)

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'estimator' : estimator,
        'render_step_size' : args.render_step_size,
        'render_bkgd' : bg_color,
        'occ_alpha_thre' : args.occ_alpha_threshold,
        'occ_cone_angle' : args.occ_cone_angle,
        'occ_near_plane' : args.nerfacc_near_plane,
        'occ_far_plane' : args.nerfacc_far_plane,
    }

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_step_size", type=float, default=None,
                        help='ray marching step size for nerfacc (defaults to (far-near)/N_samples)')
    parser.add_argument("--occ_grid_resolution", type=int, default=128,
                        help='resolution of each nerfacc occupancy grid level')
    parser.add_argument("--occ_grid_levels", type=int, default=1,
                        help='number of nerfacc occupancy grid levels')
    parser.add_argument("--occ_alpha_threshold", type=float, default=0.01,
                        help='alpha threshold used to skip empty space during sampling')
    parser.add_argument("--occ_ema_decay", type=float, default=0.95,
                        help='exponential moving average decay for occupancy updates')
    parser.add_argument("--occ_warmup_steps", type=int, default=256,
                        help='number of warmup steps before probabilistic occupancy updates')
    parser.add_argument("--occ_update_every", type=int, default=16,
                        help='update occupancy grid every N training steps')
    parser.add_argument("--occ_cone_angle", type=float, default=0.0,
                        help='cone angle for nerfacc ray marching (0 keeps constant step size)')
    parser.add_argument("--occ_grid_aabb", type=float, nargs=6, default=None,
                        help='optional ROI AABB (xmin ymin zmin xmax ymax zmax) for the occupancy grid')
    parser.add_argument("--nerfacc_near_plane", type=float, default=None,
                        help='global near plane for nerfacc sampling')
    parser.add_argument("--nerfacc_far_plane", type=float, default=None,
                        help='global far plane for nerfacc sampling')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data via unified dataset loader
    K = None
    images, poses, hwf, dataset_near, dataset_far, _ = load_dataset(
        args.datadir,
        downsample=args.factor if args.factor and args.factor > 1 else None,
    )
    poses = poses.astype(np.float32)
    images = images.astype(np.float32) / 255.0
    bottom = np.broadcast_to(np.array([0., 0., 0., 1.], dtype=np.float32), (poses.shape[0], 1, 4))
    render_poses = np.concatenate([poses, bottom], axis=1)
    print('Loaded dataset', images.shape, render_poses.shape, hwf, args.datadir)

    num_images = images.shape[0]
    if num_images == 0:
        raise ValueError(f'No images found in dataset {args.datadir}')

    if args.llffhold > 0 and num_images > 1:
        print('Auto holdout,', args.llffhold)
        i_test = np.arange(num_images)[::args.llffhold]
        if i_test.size == 0:
            i_test = np.array([num_images - 1])
    else:
        i_test = np.array([num_images - 1])

    i_val = i_test
    i_train = np.array([i for i in np.arange(num_images) if (i not in i_test and i not in i_val)])
    if i_train.size == 0:
        i_train = np.arange(num_images)

    if args.no_ndc:
        near = dataset_near
        far = dataset_far
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    if args.render_step_size is None:
        depth_span = float(far - near) if isinstance(far, (int, float)) else float(far)
        if abs(depth_span) < 1e-6:
            depth_span = 1.0
        args.render_step_size = max(depth_span / max(args.N_samples, 1), 1e-3)
    if args.nerfacc_near_plane is None:
        args.nerfacc_near_plane = float(near)
    if args.nerfacc_far_plane is None:
        args.nerfacc_far_plane = float(far)

    if args.occ_grid_aabb is not None:
        occ_aabb = np.array(args.occ_grid_aabb, dtype=np.float32)
    else:
        cam_centers = poses[:, :3, 3]
        bb_min = cam_centers.min(axis=0) - 0.5
        bb_max = cam_centers.max(axis=0) + 0.5
        occ_aabb = np.concatenate([bb_min, bb_max]).astype(np.float32)
    scene_aabb = torch.tensor(occ_aabb, dtype=torch.float32, device=device)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, scene_aabb)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    estimator = render_kwargs_train.get('estimator')
    if estimator is not None:
        estimator.train()

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            if estimator is not None:
                estimator.eval()
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            if estimator is not None:
                estimator.train()
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        if estimator is not None:
            with torch.no_grad():
                def occ_eval_fn(x):
                    dirs = torch.zeros_like(x) if render_kwargs_train['use_viewdirs'] else None
                    raw = render_kwargs_train['network_query_fn'](
                        x,
                        dirs,
                        render_kwargs_train['network_fn'],
                    )
                    sigma = F.relu(raw[..., 3:4])
                    return sigma * render_kwargs_train['render_step_size']

                estimator.update_every_n_steps(
                    step=global_step,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=args.occ_alpha_threshold,
                    ema_decay=args.occ_ema_decay,
                    warmup_steps=args.occ_warmup_steps,
                    n=args.occ_update_every,
                )

        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'estimator_state_dict': render_kwargs_train['estimator'].state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                if estimator is not None:
                    estimator.eval()
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                if estimator is not None:
                    estimator.train()
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                if estimator is not None:
                    estimator.eval()
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                if estimator is not None:
                    estimator.train()
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
