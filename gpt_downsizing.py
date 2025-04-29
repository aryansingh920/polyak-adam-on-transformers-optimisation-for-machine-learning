# gpt_downsizing.py - Keep only the config definitions
from new_gpt import ModelConfig

# initial downsize
def create_custom_config():
    return ModelConfig(
        batch_size=16,
        block_size=32,
        n_embd=32,
        n_head=2,
        n_layer=2,
        dropout=0.0,
        learning_rate=1e-3,
        max_iters=5000,
        eval_interval=100,
        eval_iters=50,
        file_name="input_childSpeech_trainingSet.txt"
    )

# deeper but thinner
def create_custom_config_1():
    return ModelConfig(
        batch_size=16,
        block_size=32,
        n_embd=24,
        n_head=3,
        n_layer=3,
        dropout=0.1,
        learning_rate=1e-3,
        max_iters=5000,
        eval_interval=100,
        eval_iters=50,
        file_name="input_childSpeech_trainingSet.txt"
    )

# wider but shallower
def create_custom_config_2():
    return ModelConfig(
        batch_size=16,
        block_size=32,
        n_embd=48,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        learning_rate=1e-3,
        max_iters=5000,
        eval_interval=100,
        eval_iters=50,
        file_name="input_childSpeech_trainingSet.txt"
    )