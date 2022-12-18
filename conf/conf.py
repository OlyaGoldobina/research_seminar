import logging
from dynaconf import Dynaconf


# specifying logging level
logging.basicConfig(level=logging.INFO)


# setting toml file
settings = Dynaconf(settings_file="conf/settings.toml",
                    envvar_prefix="DYNACONF",
                    env_switcher="ENV_FOR_DYNACONF")


logging.info("Dynaconf settings created")
