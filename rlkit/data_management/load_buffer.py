import numpy as np
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer


# TODO (avi) Clean up
def load_data_from_npy(variant, expl_env, observation_key,
                       extra_buffer_size=100):
    with open(variant['buffer'], 'rb') as f:
        data = np.load(f, allow_pickle=True)

    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    buffer_size = num_transitions + extra_buffer_size

    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
    )

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
    print('Data loaded from npy file', replay_buffer._top)
    return replay_buffer


# TODO (avi) Clean up
def process_images(observations):
    output = []
    for i in range(len(observations)):
        image = observations[i]['image']
        if len(image.shape) == 3:
            image = np.transpose(image, [2, 0, 1])
            image = (image.flatten())/255.0
        # elif len(image.shape) == 1:
        #     assert 48*48*3 == image.shape[0]
        #     image = np.reshape(image, (48, 48, 3))
        #     image = np.transpose(image, [2, 0, 1])
        #     image = image.flatten()
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(image=image))
    return output