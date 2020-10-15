# CQL

Our code is built off of [CQL](https://github.com/aviralkumar2907/CQL). 

## TODO
### High priority
- [ ] Add support for biased sampling
- [ ] Remove hard-coded image key in line 114 in `rlkit/samplers/rollout_functions.py`
- [ ] Add setup instructions. 

### Soon
- [ ] Fix OpenCV/ffmpeg video saving warning
- [ ] Clean up `rlkit/data_management/load_buffer.py`
- [ ] `ObsDictReplayBuffer` and `ObsDictRelabelingBuffer` share a lot of code, but are currently implemented independently.
- [ ] Hard-coded `.cuda()` in line 234 in `rlkit/torch/sac/cql.py`
- [ ] Perhaps the `if` statement (line 437) for image format conversion is a bit too hacky in `ObsDictReplayBuffer`. 

### Done
- [x] Fix/update input args (including with-lagrange) for `examples/cql_image.py`
- [x] Add script for running chaining experiments. 
- [x] Fix checkpoint saving
- [x] Add video logging
- [x] Fix logging dir
- [x] Add script for online finetuning