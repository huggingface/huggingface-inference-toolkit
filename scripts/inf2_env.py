#!/usr/bin/env python

"""
This script is here to specify all missing environment variables that would be required to run some encoder models on
inferentia2.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from huggingface_hub import constants
from transformers import AutoConfig

from optimum.neuron.utils import get_hub_cached_entries
from optimum.neuron.utils.version_utils import get_neuronxcc_version

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

env_config_peering = [
    ("HF_BATCH_SIZE", "static_batch_size"),
    ("HF_OPTIMUM_SEQUENCE_LENGTH", "static_sequence_length"),
]

# By the end of this script all env vars should be specified properly
env_vars = list(map(lambda x: x[0], env_config_peering))

# Currently not used for encoder models
# available_cores = get_available_cores()

neuronxcc_version = get_neuronxcc_version()


def parse_cmdline_and_set_env(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    if not argv:
        argv = sys.argv
    # All these are params passed to tgi and intercepted here
    parser.add_argument(
        "--batch-size",
        type=int,
        default=os.getenv("HF_BATCH_SIZE", os.getenv("BATCH_SIZE", 0)),
    )
    parser.add_argument(
        "--sequence-length", type=int,
        default=os.getenv("HF_OPTIMUM_SEQUENCE_LENGTH",
                          os.getenv("SEQUENCE_LENGTH", 0))
    )

    parser.add_argument("--model-id", type=str, default=os.getenv("HF_MODEL_ID", os.getenv("HF_MODEL_DIR")))
    parser.add_argument("--revision", type=str, default=os.getenv("REVISION"))

    args = parser.parse_known_args(argv)[0]

    if not args.model_id:
        raise Exception(
            "No model id provided ! Either specify it using --model-id cmdline or MODEL_ID env var"
        )

    # Override env with cmdline params
    os.environ["MODEL_ID"] = args.model_id

    # Set all tgi router and tgi server values to consistent values as early as possible
    # from the order of the parser defaults, the tgi router value can override the tgi server ones
    if args.batch_size > 0:
        os.environ["HF_BATCH_SIZE"] = str(args.batch_size)

    if args.sequence_length > 0:
        os.environ["HF_OPTIMUM_SEQUENCE_LENGTH"] = str(args.sequence_length)

    if args.revision:
        os.environ["REVISION"] = str(args.revision)

    return args


def neuron_config_to_env(neuron_config):
    with open(os.environ["ENV_FILEPATH"], "w") as f:
        for env_var, config_key in env_config_peering:
            f.write("export {}={}\n".format(env_var, neuron_config[config_key]))


def sort_neuron_configs(dictionary):
    return -dictionary["static_batch_size"]


def lookup_compatible_cached_model(
        model_id: str, revision: Optional[str]
) -> Optional[Dict[str, Any]]:
    # Reuse the same mechanic as the one in use to configure the tgi server part
    # The only difference here is that we stay as flexible as possible on the compatibility part
    entries = get_hub_cached_entries(model_id, "inference")

    logger.debug(
        "Found %d cached entries for model %s, revision %s",
        len(entries),
        model_id,
        revision,
    )

    all_compatible = []
    for entry in entries:
        if check_env_and_neuron_config_compatibility(
                entry, check_compiler_version=True
        ):
            all_compatible.append(entry)

    if not all_compatible:
        logger.debug(
            "No compatible cached entry found for model %s, env %s, neuronxcc version %s",
            model_id,
            get_env_dict(),
            neuronxcc_version,
        )
        return None

    logger.info("%d compatible neuron cached models found", len(all_compatible))

    all_compatible = sorted(all_compatible, key=sort_neuron_configs)

    entry = all_compatible[0]

    logger.info("Selected entry %s", entry)

    return entry


def check_env_and_neuron_config_compatibility(
        neuron_config: Dict[str, Any], check_compiler_version: bool
) -> bool:
    logger.debug(
        "Checking the provided neuron config %s is compatible with the local setup and provided environment",
        neuron_config,
    )

    # Local setup compat checks
    # if neuron_config["num_cores"] > available_cores:
    #     logger.debug(
    #         "Not enough neuron cores available to run the provided neuron config"
    #     )
    #     return False

    if (
            check_compiler_version
            and neuron_config["compiler_version"] != neuronxcc_version
    ):
        logger.debug(
            "Compiler version conflict, the local one (%s) differs from the one used to compile the model (%s)",
            neuronxcc_version,
            neuron_config["compiler_version"],
        )
        return False

    for env_var, config_key in env_config_peering:
        try:
            neuron_config_value = str(neuron_config[config_key])
        except KeyError:
            logger.debug("No key %s found in neuron config %s", config_key, neuron_config)
            return False
        env_value = os.getenv(env_var, str(neuron_config_value))
        if env_value != neuron_config_value:
            logger.debug(
                "The provided env var '%s' and the neuron config '%s' param differ (%s != %s)",
                env_var,
                config_key,
                env_value,
                neuron_config_value,
            )
            return False

    return True


def get_env_dict() -> Dict[str, str]:
    d = {}
    for k in env_vars:
        d[k] = os.getenv(k)
    return d


def main():
    """
    This script determines proper default TGI env variables for the neuron precompiled models to
    work properly
    :return:
    """
    args = parse_cmdline_and_set_env()

    for env_var in env_vars:
        if not os.getenv(env_var):
            break
    else:
        logger.info(
            "All env vars %s already set, skipping, user know what they are doing",
            env_vars,
        )
        sys.exit(0)

    cache_dir = constants.HF_HUB_CACHE

    logger.info("Cache dir %s, model %s", cache_dir, args.model_id)

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision)
    neuron_config = getattr(config, "neuron", None)
    if neuron_config is not None:
        compatible = check_env_and_neuron_config_compatibility(
            neuron_config, check_compiler_version=False
        )
        if not compatible:
            env_dict = get_env_dict()
            msg = (
                "Invalid neuron config and env. Config {}, env {}, neuronxcc version {}"
            ).format(neuron_config, env_dict, neuronxcc_version)
            logger.error(msg)
            raise Exception(msg)
    else:
        neuron_config = lookup_compatible_cached_model(args.model_id, args.revision)

    if not neuron_config:
        neuron_config = {'static_batch_size': 1, 'static_sequence_length': 128}
        msg = (
            "No compatible neuron config found. Provided env {}, neuronxcc version {}. Falling back to default"
        ).format(get_env_dict(), neuronxcc_version, neuron_config)
        logger.info(msg)

    logger.info("Final neuron config %s", neuron_config)

    neuron_config_to_env(neuron_config)


if __name__ == "__main__":
    main()
