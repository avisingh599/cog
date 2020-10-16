import numpy as np
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

# TODO Clean up this file


def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions


def add_data_to_buffer(data, replay_buffer):

    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_images(data[j]['observations']),
            next_observations=process_images(
                data[j]['next_observations']),
        )
        replay_buffer.add_path(path)


def load_data_from_npy(variant, expl_env, observation_key,
                       extra_buffer_size=100):
    with open(variant['buffer'], 'rb') as f:
        data = np.load(f, allow_pickle=True)

    num_transitions = get_buffer_size(data)
    buffer_size = num_transitions + extra_buffer_size

    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
    )
    add_data_to_buffer(data, replay_buffer)
    print('Data loaded from npy file', replay_buffer._top)
    return replay_buffer


def load_data_from_npy_chaining(variant, expl_env, observation_key,
                                extra_buffer_size=100):
    with open(variant['prior_buffer'], 'rb') as f:
        data_prior = np.load(f, allow_pickle=True)
    with open(variant['task_buffer'], 'rb') as f:
        data_task = np.load(f, allow_pickle=True)

    buffer_size = get_buffer_size(data_prior)
    buffer_size += get_buffer_size(data_task)
    buffer_size += extra_buffer_size

    # TODO Clean this up
    if 'biased_sampling' in variant:
        if variant['biased_sampling']:
            bias_point = buffer_size - extra_buffer_size
            print('Setting bias point', bias_point)
            replay_buffer = ObsDictReplayBuffer(
                buffer_size,
                expl_env,
                observation_key=observation_key,
                biased_sampling=True,
                bias_point=bias_point,
                before_bias_point_probability=0.5,
            )
        else:
            replay_buffer = ObsDictReplayBuffer(
                buffer_size,
                expl_env,
                observation_key=observation_key,
            )
    else:
        replay_buffer = ObsDictReplayBuffer(
            buffer_size,
            expl_env,
            observation_key=observation_key,
        )

    add_data_to_buffer(data_prior, replay_buffer)
    top = replay_buffer._top
    print('Prior data loaded from npy file', top)
    replay_buffer._rewards[:top] = 0.0*replay_buffer._rewards[:top]
    print('Zero-ed the rewards for prior data', top)

    add_data_to_buffer(data_task, replay_buffer)
    print('Task data loaded from npy file', replay_buffer._top)
    return replay_buffer


def process_images(observations):
    output = []
    for i in range(len(observations)):
        image = observations[i]['image']
        if len(image.shape) == 3:
            image = np.transpose(image, [2, 0, 1])
            image = (image.flatten())/255.0
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(image=image))
    return output