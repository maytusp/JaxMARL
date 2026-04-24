from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import List, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict
from jaxmarl.environments.overcooked_v2.common import (
    ACTION_TO_DIRECTION,
    MAX_INGREDIENTS,
    Actions,
    StaticObject,
    DynamicObject,
    Direction,
    Position,
    Agent,
)
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts, Layout
from jaxmarl.environments.overcooked_v2.settings import (
    DELIVERY_REWARD,
    INDICATOR_ACTIVATION_COST,
    INDICATOR_ACTIVATION_TIME,
    POT_COOK_TIME,
    SHAPED_REWARDS,
)
from jaxmarl.environments.overcooked_v2.utils import (
    OvercookedPathPlanner,
    compute_view_box,
    get_closest_true_pos_no_directions,
    mark_adjacent_cells,
    tree_select,
    compute_enclosed_spaces,
)


class ObservationType(str, Enum):
    DEFAULT = "default"
    FEATURIZED = "featurized"


@chex.dataclass
class State:
    agents: Agent

    # width x height x 3
    # First channel: static items
    # Second channel: dynamic items (plates and ingredients)
    # Third channel: extra info
    grid: chex.Array

    time: chex.Array
    terminal: bool

    recipe: int

    new_correct_delivery: bool = False

    ingredient_permutations: Optional[chex.Array] = None


from dataclasses import replace as _replace

from jaxmarl.environments.overcooked_v2.overcooked import (
    ObservationType as _BaseObservationType,
    OvercookedV2 as _BaseOvercookedV2,
    State as _BaseState,
)


ObservationType = _BaseObservationType
State = _BaseState


class OvercookedSingleAgent(_BaseOvercookedV2):
    """Single-agent Overcooked V2 by removing all but one agent position.

    Reset, step, observations, rewards, action semantics, and interaction
    mechanics come from the original Overcooked V2 implementation. This wrapper
    only converts the selected layout to contain one agent before initializing
    the base environment.
    """

    def __init__(
        self,
        layout: Union[str, Layout] = "cramped_room",
        max_steps: int = 400,
        observation_type: Union[
            ObservationType, List[ObservationType]
        ] = ObservationType.DEFAULT,
        agent_view_size: Optional[int] = None,
        random_reset: bool = False,
        random_agent_positions: bool = False,
        start_cooking_interaction: bool = False,
        negative_rewards: bool = False,
        sample_recipe_on_delivery: bool = False,
        indicate_successful_delivery: bool = False,
        op_ingredient_permutations: List[int] = None,
        initial_state_buffer: Optional[State] = None,
        force_path_planning: bool = False,
        active_agent_idx: int = 0,
        fixed_agent_idx: Optional[int] = None,
    ):
        if isinstance(layout, str):
            if layout not in overcooked_v2_layouts:
                raise ValueError(
                    f"Invalid layout: {layout}, allowed layouts: {overcooked_v2_layouts.keys()}"
                )
            layout = overcooked_v2_layouts[layout]
        elif not isinstance(layout, Layout):
            raise ValueError("Invalid layout, must be a Layout object or a string key")

        if len(layout.agent_positions) < 1:
            raise ValueError("Single-agent environment requires at least 1 agent position in layout")

        # Backward compatibility with the previous wrapper API: callers could
        # specify which teammate to freeze. In the one-agent version that means
        # selecting the other spawn as the retained active spawn.
        if fixed_agent_idx is not None and len(layout.agent_positions) > 1:
            active_agent_idx = 0 if fixed_agent_idx != 0 else 1

        if active_agent_idx < 0 or active_agent_idx >= len(layout.agent_positions):
            raise ValueError(
                f"active_agent_idx must be in [0, {len(layout.agent_positions) - 1}]"
            )

        single_agent_layout = _replace(
            layout,
            agent_positions=[layout.agent_positions[active_agent_idx]],
        )

        super().__init__(
            layout=single_agent_layout,
            max_steps=max_steps,
            observation_type=observation_type,
            agent_view_size=agent_view_size,
            random_reset=random_reset,
            random_agent_positions=random_agent_positions,
            start_cooking_interaction=start_cooking_interaction,
            negative_rewards=negative_rewards,
            sample_recipe_on_delivery=sample_recipe_on_delivery,
            indicate_successful_delivery=indicate_successful_delivery,
            op_ingredient_permutations=op_ingredient_permutations,
            initial_state_buffer=initial_state_buffer,
            force_path_planning=force_path_planning,
        )

        self.active_agent_idx = active_agent_idx

    def get_legacy_env(self):
        """Return a vanilla Overcooked V2 env using this one-agent layout."""
        return _BaseOvercookedV2(
            layout=self.layout,
            max_steps=self.max_steps,
            observation_type=self.observation_type,
            agent_view_size=self.agent_view_size,
            random_reset=self.random_reset,
            random_agent_positions=self.random_agent_positions,
            start_cooking_interaction=self.start_cooking_interaction,
            negative_rewards=self.negative_rewards,
            sample_recipe_on_delivery=self.sample_recipe_on_delivery,
            indicate_successful_delivery=self.indicate_successful_delivery,
            op_ingredient_permutations=self.op_ingredient_permutations,
            initial_state_buffer=self.initial_state_buffer,
            force_path_planning=False,
        )
