from dataclasses import replace as _replace
from functools import partial
from typing import List, Optional, Union

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_single.overcooked_single_agent import (
    ObservationType,
    OvercookedSingleAgent,
    State,
)
from jaxmarl.environments.overcooked_v2.common import DynamicObject
from jaxmarl.environments.overcooked_v2.layouts import Layout, overcooked_v2_layouts



class OvercookedSingleRare(OvercookedSingleAgent):
    """Single-agent Overcooked rare-goal environment.

    The default `overcooked_single_rare` setup has 10 ingredient types. In
    training mode, 95% of episode recipes contain 3 unique common ingredients
    and 5% contain exactly 1 rare ingredient plus 2 common ingredients. Use
    `recipe_sampling_mode="rare_only"` or `"balanced"` to evaluate rare-goal
    performance separately.
    """

    VALID_RECIPE_SAMPLING_MODES = ("train", "common_only", "rare_only", "balanced")

    def __init__(
        self,
        layout: Union[str, Layout] = "rare_single_room",
        max_steps: int = 400,
        observation_type: Union[
            ObservationType, List[ObservationType]
        ] = ObservationType.DEFAULT,
        agent_view_size: Optional[int] = None,
        random_reset: bool = False,
        random_agent_positions: bool = False,
        start_cooking_interaction: bool = False,
        front_obs: bool = False,
        negative_rewards: bool = False,
        sample_recipe_on_delivery: bool = True,
        indicate_successful_delivery: bool = False,
        op_ingredient_permutations: List[int] = None,
        initial_state_buffer: Optional[State] = None,
        force_path_planning: bool = False,
        active_agent_idx: int = 0,
        fixed_agent_idx: Optional[int] = None,
        rare_recipe_prob: float = 0.05,
        num_common_ingredients: int = 3,
        num_rare_ingredients: int = 1,
        recipe_size: int = 3,
        rare_ingredients_per_rare_recipe: int = 1,
        recipe_sampling_mode: str = "train",
        shuffle_recipe: bool = True,
    ):

        if isinstance(layout, str):
            if layout not in overcooked_v2_layouts:
                raise ValueError(
                    f"Invalid layout: {layout}, allowed layouts: {overcooked_v2_layouts.keys()}"
                )
            layout = overcooked_v2_layouts[layout]
        elif not isinstance(layout, Layout):
            raise ValueError("Invalid layout, must be a Layout object or a string key")

        self.rare_recipe_prob = float(rare_recipe_prob)
        self.num_common_ingredients = int(num_common_ingredients)
        self.num_rare_ingredients = int(num_rare_ingredients)
        self.recipe_size = int(recipe_size)
        self.rare_ingredients_per_rare_recipe = int(rare_ingredients_per_rare_recipe)
        self.recipe_sampling_mode = recipe_sampling_mode
        self.shuffle_recipe = bool(shuffle_recipe)

        self._validate_rare_recipe_config(layout)

        # Recipes are sampled by this subclass, not by the layout's full recipe
        # list. Keep the layout's ingredient piles and geometry unchanged.
        layout = _replace(layout, possible_recipes=[[0, 1, 2]])

        super().__init__(
            layout=layout,
            max_steps=max_steps,
            observation_type=observation_type,
            agent_view_size=agent_view_size,
            random_reset=random_reset,
            random_agent_positions=random_agent_positions,
            start_cooking_interaction=start_cooking_interaction,
            front_obs=front_obs,
            negative_rewards=negative_rewards,
            sample_recipe_on_delivery=sample_recipe_on_delivery,
            indicate_successful_delivery=indicate_successful_delivery,
            op_ingredient_permutations=op_ingredient_permutations,
            initial_state_buffer=initial_state_buffer,
            force_path_planning=force_path_planning,
            active_agent_idx=active_agent_idx,
            fixed_agent_idx=fixed_agent_idx,
        )

    def _validate_rare_recipe_config(self, layout: Layout) -> None:
        if self.recipe_sampling_mode not in self.VALID_RECIPE_SAMPLING_MODES:
            raise ValueError(
                "recipe_sampling_mode must be one of "
                f"{self.VALID_RECIPE_SAMPLING_MODES}"
            )
        if not 0.0 <= self.rare_recipe_prob <= 1.0:
            raise ValueError("rare_recipe_prob must be in [0, 1]")
        if self.recipe_size < 1:
            raise ValueError("recipe_size must be positive")
        if self.num_common_ingredients < self.recipe_size:
            raise ValueError("num_common_ingredients must be at least recipe_size")
        if self.rare_ingredients_per_rare_recipe < 1:
            raise ValueError("rare_ingredients_per_rare_recipe must be positive")
        if self.num_rare_ingredients < self.rare_ingredients_per_rare_recipe:
            raise ValueError(
                "num_rare_ingredients must be at least "
                "rare_ingredients_per_rare_recipe"
            )
        common_per_rare = self.recipe_size - self.rare_ingredients_per_rare_recipe
        if common_per_rare < 0:
            raise ValueError(
                "rare_ingredients_per_rare_recipe cannot exceed recipe_size"
            )
        if self.num_common_ingredients < common_per_rare:
            raise ValueError(
                "num_common_ingredients is too small for the rare recipe size"
            )
        total_ingredients = self.num_common_ingredients + self.num_rare_ingredients
        if layout.num_ingredients < total_ingredients:
            raise ValueError(
                "layout must expose at least "
                f"{total_ingredients} ingredient piles, got {layout.num_ingredients}"
            )

    def _sample_recipe(self, key: chex.PRNGKey) -> int:
        key_mode, key_recipe = jax.random.split(key)

        if self.recipe_sampling_mode == "common_only":
            is_rare = jnp.array(False)
        elif self.recipe_sampling_mode == "rare_only":
            is_rare = jnp.array(True)
        elif self.recipe_sampling_mode == "balanced":
            is_rare = jax.random.bernoulli(key_mode, 0.5)
        else:
            is_rare = jax.random.bernoulli(key_mode, self.rare_recipe_prob)

        recipe = jax.lax.cond(
            is_rare,
            self._sample_rare_recipe_ids,
            self._sample_common_recipe_ids,
            key_recipe,
        )
        return DynamicObject.get_recipe_encoding(recipe)

    def _sample_common_recipe_ids(self, key: chex.PRNGKey) -> chex.Array:
        key_sample, key_shuffle = jax.random.split(key)
        common_ids = jnp.arange(self.num_common_ingredients)
        recipe = jax.random.permutation(key_sample, common_ids)[: self.recipe_size]
        return self._maybe_shuffle_recipe(key_shuffle, recipe)

    def _sample_rare_recipe_ids(self, key: chex.PRNGKey) -> chex.Array:
        key_common, key_rare, key_shuffle = jax.random.split(key, 3)
        common_per_rare = self.recipe_size - self.rare_ingredients_per_rare_recipe
        common_ids = jnp.arange(self.num_common_ingredients)
        rare_ids = jnp.arange(self.num_rare_ingredients) + self.num_common_ingredients

        common_recipe = jax.random.permutation(key_common, common_ids)[:common_per_rare]
        rare_recipe = jax.random.permutation(key_rare, rare_ids)[
            : self.rare_ingredients_per_rare_recipe
        ]
        recipe = jnp.concatenate([rare_recipe, common_recipe])
        return self._maybe_shuffle_recipe(key_shuffle, recipe)

    def _maybe_shuffle_recipe(self, key: chex.PRNGKey, recipe: chex.Array) -> chex.Array:
        if self.shuffle_recipe:
            return jax.random.permutation(key, recipe)
        return recipe

    def recipe_ingredient_ids(self, recipe: chex.Array) -> chex.Array:
        return DynamicObject.get_ingredient_idx_list_jit(recipe)

    def is_rare_recipe(self, recipe: chex.Array) -> chex.Array:
        ingredient_ids = self.recipe_ingredient_ids(recipe)
        rare_start = self.num_common_ingredients
        rare_end = self.num_common_ingredients + self.num_rare_ingredients
        return jnp.any((ingredient_ids >= rare_start) & (ingredient_ids < rare_end))

    @partial(jax.jit, static_argnums=(0,))
    def reset_from_state(
        self,
        state: State,
        key: chex.PRNGKey,
    ):
        key_recipe, key_perm = jax.random.split(key)

        ingredient_permutations = None
        if self.op_ingredient_permutations:
            ingredient_permutations = self._sample_op_ingredient_permutations(key_perm)

        state = state.replace(
            time=0,
            terminal=False,
            recipe=self._sample_recipe(key_recipe),
            new_correct_delivery=False,
            ingredient_permutations=ingredient_permutations,
        )

        obs = self.get_obs(state)

        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions,
    ):
        obs, state, rewards, dones, info = super().step_env(key, state, actions)
        info["is_rare_recipe"] = self.is_rare_recipe(state.recipe)
        info["recipe_ingredient_ids"] = self.recipe_ingredient_ids(state.recipe)
        return obs, state, rewards, dones, info
