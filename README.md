# jaxgptc4
training gpt on c4 with [jax](https://jax.readthedocs.io/en/latest/index.html) and [equinox](https://docs.kidger.site/equinox/) (and [jaxamp](https://github.com/acutkosky/jaxamp)).

View sample logs [here](https://api.wandb.ai/links/optimizedlearning/uj90xkyl)


When running on the SCC, you may wish to run `source scc_setup.sh` first. Then you may want to run `wandb init` to setup wandb logging.

Note that sometimes the cuda versions get a little messed up. You can try: (1) do not have any scc cuda modules loaded (`module unload cuda`) (2) reinstalling jax after pytorch (follow instructions [here](https://jax.readthedocs.io/en/latest/installation.html))

Then `python trainer.py` to train.

The various options are specified in `conf/config_gpt2.yaml`. You can override them in the command line like so: `python trainer.py model.num_blocks=8 train.wandb_project=projectname`

