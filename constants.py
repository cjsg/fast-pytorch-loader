IMG_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 8
N_EPOCHS = 10
WARMUP_STEPS = 10

tqdm_args= {
    'smoothing': 0,  # avg-speed
    'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '\
        '{rate_noinv_fmt}{postfix}]'
}
