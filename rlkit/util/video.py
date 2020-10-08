import os
import os.path as osp
import time

import numpy as np
import scipy.misc
import skvideo.io

from rlkit.core import logger


class VideoSaveFunction:
    def __init__(self, variant):
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 50)

    def __call__(self, algo, epoch):
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            video_dir = osp.join(self.logdir,
                                 'videos_eval/{epoch}/'.format(epoch=epoch))
            eval_paths = algo.eval_data_collector.get_epoch_paths()
            dump_video_basic(video_dir, eval_paths)


def dump_video_basic(video_dir, paths):

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    for i, path in enumerate(paths):
        video = path['next_observations']
        frame_list = []
        for frame in video:
            # TODO(avi) Figure out why this hack is needed
            if isinstance(frame, np.ndarray):
                frame_list.append(frame[0]['image'])
            else:
                frame_list.append(frame['image'])
            # frame_list.append(frame)
        frame_list = np.asarray(frame_list)
        video_len = frame_list.shape[0]
        n_channels = 3
        imsize = int(np.sqrt(frame_list.shape[1] / n_channels))
        assert imsize*imsize*n_channels == frame_list.shape[1]

        video = frame_list.reshape(video_len, n_channels, imsize, imsize)
        video = np.transpose(video, (0, 2, 3, 1))
        video = (video*255.0).astype(np.uint8)
        filename = osp.join(video_dir, '{}.mp4'.format(i))
        FPS = float(np.ceil(video_len/3.0))
        writer = cv2.VideoWriter(filename, fourcc, FPS, (imsize, imsize))
        for j in range(video.shape[0]):
            writer.write(cv2.cvtColor(video[j], cv2.COLOR_RGB2BGR))
        writer = None


def dump_video(
        env,
        policy,
        filename,
        rollout_function,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        num_channels=3,
):
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    frames = []
    H = 3 * imsize
    W = imsize
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )
        is_vae_env = isinstance(env, VAEWrappedEnv)
        l = []
        for d in path['full_observations']:
            if is_vae_env:
                recon = np.clip(env._reconstruct_img(d['image_observation']), 0,
                                1)
            else:
                recon = d['image_observation']
            l.append(
                get_image(
                    d['image_desired_goal'],
                    d['image_observation'],
                    recon,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    imsize=imsize,
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    frames = np.array(frames, dtype=np.uint8)
    path_length = frames.size // (
            N * (H + 2 * pad_length) * (W + 2 * pad_length) * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, H + 2 * pad_length, W + 2 * pad_length,
                 num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)


def get_image(goal, obs, recon_obs, imsize=84, pad_length=1, pad_color=255):
    if len(goal.shape) == 1:
        goal = goal.reshape(-1, imsize, imsize).transpose()
        obs = obs.reshape(-1, imsize, imsize).transpose()
        recon_obs = recon_obs.reshape(-1, imsize, imsize).transpose()
    img = np.concatenate((goal, obs, recon_obs))
    img = np.uint8(255 * img)
    if pad_length > 0:
        img = add_border(img, pad_length, pad_color)
    return img


def add_border(img, pad_length, pad_color, imsize=84):
    H = 3 * imsize
    W = imsize
    img = img.reshape((3 * imsize, imsize, -1))
    img2 = np.ones((H + 2 * pad_length, W + 2 * pad_length, img.shape[2]),
                   dtype=np.uint8) * pad_color
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2
