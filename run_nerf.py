import os, sys
import numpy as np
import imageio
import json
import random
import time
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
        if viewdirs.shape != inputs.shape:
            input_dirs = viewdirs[..., None, :].expand(inputs.shape)
        else:
            input_dirs = viewdirs
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k == 'num_samples':
                value = int(ret[k].sum().item()) if isinstance(ret[k], torch.Tensor) else int(ret[k])
                all_ret[k] = all_ret.get(k, 0) + value
            else:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

    for k in list(all_ret.keys()):
        if k == 'num_samples':
            device = rays_flat.device
            all_ret[k] = torch.tensor(all_ret[k], device=device, dtype=torch.int32)
        else:
            all_ret[k] = torch.cat(all_ret[k], 0)
    return all_ret


def compute_occ_aabb(poses: np.ndarray,
                     meta: Dict[str, Any],
                     near: float,
                     far: float,
                     use_ndc: bool,
                     padding_scale: float = 1.5) -> torch.Tensor:
    """Derive an occupancy-grid AABB from dataset poses and metadata."""
    if use_ndc:
        far_ndc = max(far, 1.0)
        aabb = torch.tensor(
            [-1.5, -1.5, 0.0,
             1.5, 1.5, far_ndc],
            dtype=torch.float32,
            device=device,
        )
        return aabb

    if poses.size == 0:
        center = np.zeros(3, dtype=np.float32)
        extent = np.ones(3, dtype=np.float32) * 0.5
    else:
        centers = poses[:, :3, 3]
        bbox_min = centers.min(axis=0)
        bbox_max = centers.max(axis=0)
        align_bbox = meta.get('alignment', {}).get('bbox_after') if meta else None
        if isinstance(align_bbox, dict):
            if 'min' in align_bbox:
                bbox_min = np.minimum(bbox_min, np.asarray(align_bbox['min'], dtype=np.float32))
            if 'max' in align_bbox:
                bbox_max = np.maximum(bbox_max, np.asarray(align_bbox['max'], dtype=np.float32))
        center = 0.5 * (bbox_min + bbox_max)
        extent = 0.5 * (bbox_max - bbox_min)
    extent = np.maximum(extent, np.ones(3, dtype=np.float32) * 0.5)
    radius = float(np.linalg.norm(extent))
    padding = max(far, radius * padding_scale, 1.0)
    half = extent + padding
    aabb_min = center - half
    aabb_max = center + half
    occ = np.concatenate([aabb_min, aabb_max]).astype(np.float32)
    return torch.tensor(occ, device=device)


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        if k == 'num_samples':
            continue
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


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


def create_nerf(args, occ_aabb: torch.Tensor):
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
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    occ_estimator = nerfacc.OccGridEstimator(
        roi_aabb=occ_aabb,
        resolution=128,
        levels=1,
    ).to(device)
    with torch.no_grad():
        occ_estimator.occs.fill_(1.0)
        occ_estimator.binaries.fill_(True)

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
        'occ_estimator': occ_estimator,
        'render_step_size': None,
        'alpha_thre': 1e-2,
        'early_stop_eps': 1e-4,
        'occ_update_every': 16,
        'occ_thre': 1e-2,
        'occ_ema_decay': 0.95,
        'occ_warmup_steps': 256,
        'update_occ_grid': True,
    }

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    occ_estimator.train()

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['update_occ_grid'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                **kwargs):
    """Volumetric rendering using nerfacc accelerated sampling."""

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    occ_estimator = kwargs.get('occ_estimator')
    if occ_estimator is None:
        raise ValueError("Expected nerfacc occupancy estimator in render kwargs.")

    device = rays_o.device
    rays_d_norm = torch.nn.functional.normalize(rays_d, dim=-1)

    t_min = near.squeeze(-1).contiguous()
    t_max = far.squeeze(-1).contiguous()
    near_plane = float(torch.min(t_min).detach().item())
    far_plane = float(torch.max(t_max).detach().item())

    use_viewdirs = kwargs.get('use_viewdirs', False)
    white_bkgd = kwargs.get('white_bkgd', False)
    raw_noise_std = kwargs.get('raw_noise_std', 0.0)

    perturb_enabled = perturb > 0.

    def _add_noise(raw_sigma: Tensor) -> Tensor:
        if raw_noise_std <= 0.0:
            return raw_sigma
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*raw_sigma.shape) * raw_noise_std
            noise = torch.tensor(noise, dtype=raw_sigma.dtype, device=raw_sigma.device)
        else:
            noise = torch.randn_like(raw_sigma) * raw_noise_std
        return raw_sigma + noise

    render_step_size = kwargs.get('render_step_size')
    if render_step_size is None:
        avg_span = torch.mean(torch.clamp(t_max - t_min, min=1e-3)).detach().item()
        render_step_size = max(avg_span / max(float(N_samples), 1.0), 1e-3)

    alpha_thre = kwargs.get('alpha_thre', 1e-2)
    early_stop_eps = kwargs.get('early_stop_eps', 1e-4)
    cone_angle = kwargs.get('cone_angle', 0.0)

    def _sigma_fn_est(t_s: Tensor, t_e: Tensor, r_idx: Tensor) -> Tensor:
        if t_s.shape[0] == 0:
            return torch.empty((0,), device=device, dtype=rays_o.dtype)
        mid = (t_s + t_e) * 0.5
        positions = rays_o[r_idx] + rays_d_norm[r_idx] * mid[:, None]
        dirs = viewdirs[r_idx] if viewdirs is not None else None
        raw = network_query_fn(positions, dirs, network_fn)
        sigmas = F.relu(raw[..., 3])
        return sigmas

    with torch.no_grad():
        ray_indices, t_starts, t_ends = occ_estimator.sampling(
            rays_o,
            rays_d_norm,
            sigma_fn=_sigma_fn_est,
            near_plane=near_plane,
            far_plane=far_plane,
            t_min=t_min,
            t_max=t_max,
            render_step_size=render_step_size,
            early_stop_eps=early_stop_eps,
            alpha_thre=alpha_thre,
            stratified=perturb_enabled,
            cone_angle=cone_angle,
        )

    num_samples = t_starts.shape[0]
    render_bkgd = torch.ones(3, device=device) if white_bkgd else None

    def _render_with_network(current_network: nn.Module):
        def rgb_sigma_fn(t_s: Tensor, t_e: Tensor, r_idx: Tensor):
            if t_s.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=device)
                sigmas = torch.empty((0,), device=device)
            else:
                mid = (t_s + t_e) * 0.5
                positions = rays_o[r_idx] + rays_d_norm[r_idx] * mid[:, None]
                dirs = viewdirs[r_idx] if viewdirs is not None else None
                raw = network_query_fn(positions, dirs, current_network)
                rgbs = torch.sigmoid(raw[..., :3])
                sigmas = F.relu(_add_noise(raw[..., 3]))
            return rgbs, sigmas

        colors, opacities, depths, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=N_rays,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        return colors, opacities, depths, extras

    colors_coarse, opacity_coarse, depth_coarse, extras_coarse = _render_with_network(network_fn)

    colors_final = colors_coarse
    opacity_final = opacity_coarse
    depth_final = depth_coarse
    extras_final = extras_coarse

    if N_importance > 0 and network_fine is not None:
        colors_fine, opacity_fine, depth_fine, extras_fine = _render_with_network(network_fine)
        colors_final = colors_fine
        opacity_final = opacity_fine
        depth_final = depth_fine
        extras_final = extras_fine

    depth_map = depth_final.squeeze(-1)
    acc_map = opacity_final.squeeze(-1)
    disp_map = 1. / torch.clamp(depth_map, min=1e-10)

    ret = {
        'rgb_map': colors_final,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'num_samples': torch.tensor(num_samples, device=device, dtype=torch.int32),
    }

    if N_importance > 0 and network_fine is not None:
        depth_coarse_map = depth_coarse.squeeze(-1)
        acc_coarse_map = opacity_coarse.squeeze(-1)
        disp_coarse = 1. / torch.clamp(depth_coarse_map, min=1e-10)
        ret['rgb0'] = colors_coarse
        ret['disp0'] = disp_coarse
        ret['acc0'] = acc_coarse_map

    weights_final = extras_final.get('weights')
    if weights_final is not None and weights_final.numel() > 0:
        z_mids = (t_starts + t_ends) * 0.5
        w_sum = torch.zeros(N_rays, device=device).scatter_add_(0, ray_indices, weights_final)
        weighted_mean = torch.zeros(N_rays, device=device).scatter_add_(0, ray_indices, weights_final * z_mids)
        mean = weighted_mean / torch.clamp(w_sum, min=1e-10)
        var = torch.zeros(N_rays, device=device).scatter_add_(
            0, ray_indices, weights_final * (z_mids - mean[ray_indices]) ** 2
        )
        std = torch.sqrt(var / torch.clamp(w_sum, min=1e-10))
    else:
        std = torch.zeros(N_rays, device=device)
    ret['z_std'] = std

    update_occ_grid = kwargs.get('update_occ_grid', False)
    if update_occ_grid:
        occ_update_every = kwargs.get('occ_update_every', 16)
        occ_thre = kwargs.get('occ_thre', 1e-2)
        occ_ema_decay = kwargs.get('occ_ema_decay', 0.95)
        occ_warmup_steps = kwargs.get('occ_warmup_steps', 256)
        global_step = kwargs.get('global_step', 0)

        def occ_eval_fn(x: Tensor) -> Tensor:
            dirs = torch.zeros_like(x) if use_viewdirs else None
            raw = network_query_fn(x, dirs, network_fn)
            sigma = F.relu(raw[..., 3:4])
            return sigma * render_step_size

        occ_estimator.update_every_n_steps(
            step=global_step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=occ_thre,
            ema_decay=occ_ema_decay,
            warmup_steps=occ_warmup_steps,
            n=occ_update_every,
        )

    for key in ('rgb_map', 'disp_map', 'acc_map'):
        if (torch.isnan(ret[key]).any() or torch.isinf(ret[key]).any()) and DEBUG:
            print(f"! [Numerical Error] {key} contains nan or inf.")

    return ret


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
    images, poses, hwf, dataset_near, dataset_far, dataset_meta = load_dataset(
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

    use_ndc = not args.no_ndc
    occ_aabb = compute_occ_aabb(poses, dataset_meta, near, far, use_ndc)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, occ_aabb)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

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

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
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
        render_kwargs_train['global_step'] = global_step
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        num_samples_tensor = extras.get('num_samples')
        if num_samples_tensor is not None and num_samples_tensor.sum() == 0:
            continue

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
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
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
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
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
