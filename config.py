from dynaconf import Dynaconf, Validator


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml'],
    validators=[
        Validator("agent.replay_buffer_size", gte=1e4, lte=1e6, must_exist=True),
        Validator("agent.batch_size", gte=8, must_exist=True),
        Validator("agent.gamma", gte=0.9, lt=1.0, must_exist=True),
        Validator("agent.tau", gte=1e-3, lt=1.0, must_exist=True),
        Validator("agent.lr", gte=1e-4, lt=1.0, must_exist=True),
        Validator("agent.update_every", gte=4, must_exist=True),
    ]
)

settings.validators.validate()

assert isinstance(settings.seed, int)
assert isinstance(settings.episodes, int)
assert isinstance(settings.max_t, int)
assert isinstance(settings.eps_start, float)
assert isinstance(settings.eps_end, float)
assert isinstance(settings.eps_decay, float)

assert isinstance(settings.agent.replay_buffer_size, int)
assert isinstance(settings.agent.batch_size, int)
assert isinstance(settings.agent.gamma, float)
assert isinstance(settings.agent.tau, float)
assert isinstance(settings.agent.lr, float)
assert isinstance(settings.agent.update_every, int)

