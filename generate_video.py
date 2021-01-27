# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re

import imageio
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

# ----------------------------------------------------------------------------


def generate_video(
    network_pkl,
    seeds,
    interpolation_size,
    truncation_psi,
    outdir,
    class_idx,
    dlatents_npz,
):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:
        print(f'Generating images from dlatents file "{dlatents_npz}"')
        dlatents = np.load(dlatents_npz)["dlatents"]
        assert dlatents.shape[1:] == (18, 512)  # [N, 18, 512]
        imgs = Gs.components.synthesis.run(
            dlatents,
            output_transform=dict(
                func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
            ),
        )
        for i, img in enumerate(imgs):
            fname = f"{outdir}/dlatent{i:02d}.png"
            print(f"Saved {fname}")
            PIL.Image.fromarray(img, "RGB").save(fname)
        return

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        "output_transform": dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        "randomize_noise": False,
    }
    if truncation_psi is not None:
        Gs_kwargs["truncation_psi"] = truncation_psi

    noise_vars = [
        var
        for name, var in Gs.components.synthesis.vars.items()
        if name.startswith("noise")
    ]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    zs = []

    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        zs.append(z)

    # interpolate
    zs_int = [zs[0]]

    for i in range(len(zs) - 1):
        z1 = zs[i]
        z2 = zs[i + 1]

        zd = (z2 - z1) / (interpolation_size + 1)
        for j in range(interpolation_size + 1):
            z1 += zd
            zs_int.append(z1.copy())

    imgs = []

    for z in zs_int:
        noise_rnd = np.random.RandomState(1)  # fix noise
        tflib.set_vars(
            {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}
        )  # [height, width]
        images = Gs.run(z, label, **Gs_kwargs)  # [minibatch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], "RGB"))

    with imageio.get_writer(f"{outdir}/mov.mp4", mode="I", fps=60) as writer:
        for image in imgs:
            writer.append_data(np.array(image))


# ----------------------------------------------------------------------------


def _parse_num_range(s):
    """Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints."""

    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------

_examples = """examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
"""

# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using pretrained network pickle.",
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--network", help="Network pickle filename", dest="network_pkl", required=True
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--seeds", type=_parse_num_range, help="List of random seeds")
    g.add_argument(
        "--dlatents", dest="dlatents_npz", help="Generate images for saved dlatents"
    )
    parser.add_argument(
        "--interpolation_size",
        type=int,
        help="Number of interpolation steps",
    )
    parser.add_argument(
        "--trunc",
        dest="truncation_psi",
        type=float,
        help="Truncation psi (default: %(default)s)",
        default=0.5,
    )
    parser.add_argument(
        "--class",
        dest="class_idx",
        type=int,
        help="Class label (default: unconditional)",
    )
    parser.add_argument(
        "--outdir", help="Where to save the output images", required=True, metavar="DIR"
    )

    args = parser.parse_args()
    generate_video(**vars(args))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
