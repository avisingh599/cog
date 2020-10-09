# CQL

Our code is built off of [CQL](https://github.com/aviralkumar2907/CQL). 

# TODO
- [ ] Fix checkpoint saving
- [ ] Fix/update input args (including with-lagrange) for `examples/cql_image.py`
- [ ] Add script for running chaining experiments. 
- [ ] Remove hard-coded image key in line 114 in `rlkit/samplers/rollout_functions.py`
- [ ] `ObsDictReplayBuffer` and `ObsDictRelabelingBuffer` share a lot of code, but are currently implemented independently.
- [ ] Hard-coded `.cuda()` in line 234 in `rlkit/torch/sac/cql.py`
- [ ] Clean up `rlkit/data_management/load_buffer.py`
- [ ] Perhaps the `if` statement (line 437) for image format conversion is a bit too hacky in `ObsDictReplayBuffer`. 
- [ ] Add setup instructions. 
- [x] Add video logging
- [x] Fix logging dir
