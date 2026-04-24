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
from jaxmarl.environments.overcooked_single.layouts import overcooked_v2_layouts, Layout
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


class OvercookedSingleAgent(MultiAgentEnv):
    """Single-agent Overcooked environment for pretraining.
    
    This environment wraps the multi-agent Overcooked environment but exposes
    only a single agent. The second agent's position is fixed and their actions
    are set to no-op. This allows pretraining a single agent that can later
    generalize to multi-agent settings.
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
        fixed_agent_idx: int = 1,
    ):
        """
        Initializes the Single-Agent Overcooked environment.

        Args:
            layout (Layout): The layout configuration for the environment.
            max_steps (int): The maximum number of steps in the environment.
            observation_type (Union[ObservationType, List[ObservationType]]): The type of observation.
            agent_view_size (Optional[int]): The number of blocks the agent can view in each direction.
            random_reset (bool): Whether to reset with random agent positions.
            random_agent_positions (bool): Whether to randomize agent positions.
            start_cooking_interaction (bool): If false pot starts cooking automatically.
            negative_rewards (bool): Whether to use negative rewards.
            sample_recipe_on_delivery (bool): Whether to sample a new recipe on delivery.
            indicate_successful_delivery (bool): Whether to indicate successful delivery.
            op_ingredient_permutations (list): List of ingredient indices to permute.
            initial_state_buffer (State): Initial state buffer.
            force_path_planning (bool): Whether to force path planning.
            fixed_agent_idx (int): Index of the agent to keep fixed (1 = second agent).
        """

        if isinstance(layout, str):
            if layout not in overcooked_v2_layouts:
                raise ValueError(
                    f"Invalid layout: {layout}, allowed layouts: {overcooked_v2_layouts.keys()}"
                )
            layout = overcooked_v2_layouts[layout]
        elif not isinstance(layout, Layout):
            raise ValueError("Invalid layout, must be a Layout object or a string key")

        num_agents = len(layout.agent_positions)

        if num_agents < 1:
            raise ValueError("Single-agent environment requires at least 1 agent position in layout")

        # Expose one controlled agent. If a layout has extra agents, keep the
        # existing behavior of freezing one teammate in place.
        self._num_agents_internal = num_agents
        self._fixed_agent_idx = fixed_agent_idx if num_agents > 1 else None
        self._active_agent_idx = 0 if self._fixed_agent_idx != 0 else 1

        super().__init__(num_agents=1)  # External interface: 1 agent

        self.height = layout.height
        self.width = layout.width

        self.layout = layout

        self.initial_state_buffer = initial_state_buffer

        self.agents = ["agent_0"]  # External agent list
        self.action_set = jnp.array(list(Actions))

        self.observation_type = observation_type

        self.agent_view_size = agent_view_size
        self.indicate_successful_delivery = indicate_successful_delivery
        self.obs_shape = self._get_obs_shape()

        self.max_steps = max_steps

        self.possible_recipes = jnp.array(layout.possible_recipes, dtype=jnp.int32)

        self.random_reset = random_reset
        self.random_agent_positions = random_agent_positions

        self.start_cooking_interaction = jnp.array(
            start_cooking_interaction, dtype=jnp.bool_
        )
        self.negative_rewards = negative_rewards
        self.sample_recipe_on_delivery = jnp.array(
            sample_recipe_on_delivery, dtype=jnp.bool_
        )

        self.enclosed_spaces = compute_enclosed_spaces(
            layout.static_objects == StaticObject.EMPTY,
        )

        self.op_ingredient_permutations = op_ingredient_permutations

        if (
            force_path_planning
            or observation_type == ObservationType.FEATURIZED
        ):
            move_area = layout.static_objects == StaticObject.EMPTY
            self.path_planer = OvercookedPathPlanner(move_area)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        # Convert single agent action to internal multi-agent actions
        active_action = actions["agent_0"]
        
        # Create internal actions: active agent gets the real action; any fixed
        # teammate gets no-op (stay).
        internal_actions = jnp.zeros((self._num_agents_internal,), dtype=jnp.int32)
        internal_actions = internal_actions.at[self._active_agent_idx].set(active_action)
        if self._fixed_agent_idx is not None:
            internal_actions = internal_actions.at[self._fixed_agent_idx].set(Actions.stay)

        state, reward, shaped_rewards = self.step_agents(key, state, internal_actions)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        # Return single agent rewards
        rewards = {"agent_0": reward}
        shaped_rewards = {"agent_0": shaped_rewards[0] if shaped_rewards else 0.0}

        dones = {"agent_0": done, "__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {"shaped_reward": shaped_rewards},
        )

    @partial(jax.jit, static_argnums=(0,))
    def _sample_op_ingredient_permutations(self, key: chex.PRNGKey) -> chex.Array:
        perm_indices = jnp.array(self.op_ingredient_permutations)

        def _ingredient_permutation(key):
            full_perm = jnp.arange(self.layout.num_ingredients)
            perm = jax.random.permutation(key, perm_indices)
            full_perm = full_perm.at[perm_indices].set(full_perm[perm])
            return full_perm

        key, subkey = jax.random.split(key)
        ing_keys = jax.random.split(subkey, self._num_agents_internal)
        ingredient_permutations = jax.vmap(_ingredient_permutation)(ing_keys)

        return ingredient_permutations

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        if self.initial_state_buffer is not None:
            num_states = jax.tree_util.tree_flatten(self.initial_state_buffer)[0][
                0
            ].shape[0]
            print("num_states in buffer: ", num_states)
            sampled_state_idx = jax.random.randint(key, (), 0, num_states)
            sampled_state = jax.tree_util.tree_map(
                lambda x: x[sampled_state_idx], self.initial_state_buffer
            )
            return self.reset_from_state(sampled_state, key)

        layout = self.layout

        static_objects = layout.static_objects
        grid = jnp.stack(
            [
                static_objects,
                jnp.zeros_like(static_objects),  # ingredient channel
                jnp.zeros_like(static_objects),  # extra info channel
            ],
            axis=-1,
            dtype=jnp.int32,
        )

        num_agents = self._num_agents_internal
        x_positions, y_positions = map(jnp.array, zip(*layout.agent_positions))
        agents = Agent(
            pos=Position(x=x_positions, y=y_positions),
            dir=jnp.full((num_agents,), Direction.UP),
            inventory=jnp.zeros((num_agents,), dtype=jnp.int32),
        )

        key, subkey = jax.random.split(key)
        recipe = self._sample_recipe(subkey)

        ingredient_permutations = None
        if self.op_ingredient_permutations:
            ingredient_permutations = self._sample_op_ingredient_permutations(key)

        state = State(
            agents=agents,
            grid=grid,
            time=0,
            terminal=False,
            recipe=recipe,
            new_correct_delivery=False,
            ingredient_permutations=ingredient_permutations,
        )

        key, key_randomize = jax.random.split(key)
        if self.random_reset:
            state = self._randomize_state(state, key_randomize)
        elif self.random_agent_positions:
            state = self._randomize_agent_positions(state, key_randomize)

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    @partial(jax.jit, static_argnums=(0,))
    def reset_from_state(
        self,
        state: State,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Reset the environment from a given state."""

        print("reset_from_state")

        ingredient_permutations = None
        if self.op_ingredient_permutations:
            ingredient_permutations = self._sample_op_ingredient_permutations(key)

        state = state.replace(
            time=0,
            terminal=False,
            new_correct_delivery=False,
            ingredient_permutations=ingredient_permutations,
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def _sample_recipe(self, key: chex.PRNGKey) -> int:
        fixed_recipe_idx = jax.random.randint(
            key, (), 0, self.possible_recipes.shape[0]
        )

        fixed_recipe = self.possible_recipes[fixed_recipe_idx]

        return DynamicObject.get_recipe_encoding(fixed_recipe)

    def _randomize_agent_positions(self, state: State, key: chex.PRNGKey) -> State:
        """Randomize agent positions."""
        num_agents = self._num_agents_internal
        agents = state.agents

        def _select_agent_position(taken_mask, x):
            pos, key = x

            allowed_positions = (
                self.enclosed_spaces == self.enclosed_spaces[pos.y, pos.x]
            ) & ~taken_mask
            allowed_positions = allowed_positions.flatten()

            p = allowed_positions / jnp.sum(allowed_positions)
            agent_pos_idx = jax.random.choice(key, allowed_positions.size, (), p=p)
            agent_position = Position(
                x=agent_pos_idx % self.width, y=agent_pos_idx // self.width
            )

            new_taken_mask = taken_mask.at[agent_position.y, agent_position.x].set(True)
            return new_taken_mask, agent_position

        taken_mask = jnp.zeros_like(self.enclosed_spaces, dtype=jnp.bool_)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_agents)
        _, agent_positions = jax.lax.scan(
            _select_agent_position, taken_mask, (agents.pos, keys)
        )

        key, subkey = jax.random.split(key)
        directions = jax.random.randint(subkey, (num_agents,), 0, len(Direction))

        return state.replace(agents=agents.replace(pos=agent_positions, dir=directions))

    def _randomize_state(self, state: State, key: chex.PRNGKey) -> State:
        """Randomize the state of the environment."""

        key, subkey = jax.random.split(key)
        state = self._randomize_agent_positions(state, subkey)

        num_agents = self._num_agents_internal
        agents = state.agents
        grid = state.grid

        # Agent inventory
        def _sample_inventory(key):
            key_dish, key_ing, key_inv = jax.random.split(key, 3)

            def _sample_dish(key):
                recipe_idx = jax.random.randint(key, (), 0, len(self.possible_recipes))
                recipe = self.possible_recipes[recipe_idx]
                return (
                    DynamicObject.get_recipe_encoding(recipe)
                    | DynamicObject.COOKED
                    | DynamicObject.PLATE
                )

            ingridient_idx = jax.random.randint(
                key_ing, (), 0, self.layout.num_ingredients
            )

            possible_inventory = jnp.array(
                [
                    DynamicObject.EMPTY,
                    DynamicObject.PLATE,
                    DynamicObject.ingredient(ingridient_idx),
                    _sample_dish(key_dish),
                ],
                dtype=jnp.int32,
            )

            inventory = jax.random.choice(
                key_inv, possible_inventory, (), p=jnp.array([0.5, 0.1, 0.25, 0.15])
            )
            return inventory

        key, subkey = jax.random.split(key)
        agent_inventories = jax.vmap(_sample_inventory)(
            jax.random.split(subkey, num_agents)
        )

        def _sample_grid_states_wrapper(cell, key):

            def _sample_pot_states(key):
                key, key_ing, key_num, key_timer = jax.random.split(key, 4)
                raw_ingridients = jax.random.randint(
                    key_ing, (3,), 0, self.layout.num_ingredients
                )
                raw_ingridients = jax.vmap(DynamicObject.ingredient)(raw_ingridients)

                partial_recipe = jax.random.randint(key_num, (), 1, 4)
                mask = jnp.arange(3) < partial_recipe

                pot_ingridients_masked = jnp.sum(raw_ingridients * mask)
                if self.start_cooking_interaction:
                    pot_ingridients_full = pot_ingridients_masked
                else:
                    pot_ingridients_full = jnp.sum(raw_ingridients)

                pot_timer = jax.random.randint(key_timer, (), 0, POT_COOK_TIME) + 1

                possible_states = jnp.array(
                    [
                        cell,
                        [cell[0], pot_ingridients_masked, 0],
                        [cell[0], pot_ingridients_full, pot_timer],
                        [cell[0], pot_ingridients_full | DynamicObject.COOKED, 0],
                    ]
                )
                return jax.random.choice(
                    key, possible_states, p=jnp.array([0.4, 0.35, 0.15, 0.1])
                )

            def _sample_counter_state(key):
                key, key_ing, key_dish = jax.random.split(key, 3)

                ingridient_idx = jax.random.randint(
                    key_ing, (), 0, self.layout.num_ingredients
                )
                dish_idx = jax.random.randint(
                    key_dish, (), 0, len(self.possible_recipes)
                )
                dish = (
                    DynamicObject.get_recipe_encoding(self.possible_recipes[dish_idx])
                    | DynamicObject.COOKED
                    | DynamicObject.PLATE
                )

                possible_states = jnp.array(
                    [
                        DynamicObject.EMPTY,
                        DynamicObject.PLATE,
                        DynamicObject.ingredient(ingridient_idx),
                        dish,
                    ]
                )

                ing_layer = jax.random.choice(
                    key, possible_states, p=jnp.array([0.5, 0.1, 0.3, 0.1])
                )
                return cell.at[1].set(ing_layer)

            is_pot = cell[0] == StaticObject.POT
            is_wall = cell[0] == StaticObject.WALL
            branch_idx = 1 * is_pot + 2 * is_wall

            return jax.lax.switch(
                branch_idx,
                [
                    lambda _: cell,
                    lambda key: _sample_pot_states(key),
                    lambda key: _sample_counter_state(key),
                ],
                key,
            )

        key, subkey = jax.random.split(key)
        key_grid = jax.random.split(subkey, (self.height, self.width))
        new_grid = jax.vmap(jax.vmap(_sample_grid_states_wrapper))(grid, key_grid)

        return state.replace(
            agents=agents.replace(inventory=agent_inventories),
            grid=new_grid,
        )

    def _get_obs_shape(self) -> Tuple[int]:
        if self.agent_view_size:
            view_size = self.agent_view_size * 2 + 1
            view_width = min(self.width, view_size)
            view_height = min(self.height, view_size)
        else:
            view_width = self.width
            view_height = self.height

        def _get_obs_shape_single(obs_type):
            match obs_type:
                case ObservationType.DEFAULT:
                    num_ingredients = self.layout.num_ingredients
                    # Single agent: no other agent layers
                    num_layers = 18 + 4 * (num_ingredients + 2)

                    if self.indicate_successful_delivery:
                        num_layers += 1

                    return (view_height, view_width, num_layers)
                case ObservationType.FEATURIZED:
                    num_pot_features = 10
                    base_features = 28
                    num_pots = 2
                    # Single agent: no other player features
                    total_features = num_pots * num_pot_features + base_features
                    return (total_features,)
                case _:
                    raise ValueError(
                        f"Invalid observation type: {self.observation_type}"
                    )

        return _get_obs_shape_single(self.observation_type)

    def get_obs(self, state: State) -> chex.Array:
        """Get observation for the single active agent."""
        obs_dict = self.get_obs_internal(state)
        return {"agent_0": obs_dict["agent_0"]}

    def get_obs_internal(self, state: State) -> Dict[str, chex.Array]:
        """Internal method to get observations for all agents."""
        if self.observation_type == ObservationType.DEFAULT:
            return self.get_obs_default(state)
        elif self.observation_type == ObservationType.FEATURIZED:
            return self.get_obs_featurized(state)
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

    def get_obs_default(self, state: State) -> Dict[str, chex.Array]:
        """Get default observation for single agent (no other agent info)."""

        width = self.width
        height = self.height
        num_ingredients = self.layout.num_ingredients

        static_objects = state.grid[:, :, 0]
        ingredients = state.grid[:, :, 1]
        extra_info = state.grid[:, :, 2]

        static_encoding = jnp.array(
            [
                StaticObject.WALL,
                StaticObject.GOAL,
                StaticObject.POT,
                StaticObject.RECIPE_INDICATOR,
                StaticObject.BUTTON_RECIPE_INDICATOR,
                StaticObject.PLATE_PILE,
            ]
        )
        static_layers = static_objects[..., None] == static_encoding

        def _ingridient_layers(ingredients, ingredient_mapping=None):
            shift = jnp.array([0, 1] + [2 * (i + 1) for i in range(num_ingredients)])
            mask = jnp.array([0x1, 0x1] + [0x3] * num_ingredients)

            layers = ingredients[..., None] >> shift
            layers = layers & mask

            if ingredient_mapping is not None:
                layers = layers.at[..., 2:].set(
                    layers[..., 2:][..., ingredient_mapping]
                )

            return layers

        recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        button_recipe_indicator_mask = (
            static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
        ) & (extra_info > 0)

        recipe_ingridients = jnp.where(
            recipe_indicator_mask | button_recipe_indicator_mask, state.recipe, 0
        )

        extra_info = state.grid[:, :, 2]
        pot_timer_layer = jnp.where(static_objects == StaticObject.POT, extra_info, 0)
        new_correct_delivery_layer = jnp.where(
            static_objects == StaticObject.GOAL, state.new_correct_delivery, 0
        )

        extra_layers = [pot_timer_layer]

        if self.indicate_successful_delivery:
            extra_layers.append(new_correct_delivery_layer)

        extra_layers = jnp.stack(extra_layers, axis=-1)

        def _agent_layers(agent, ingredient_mapping=None):
            pos = agent.pos
            direction = agent.dir
            inv = agent.inventory

            pos_layers = (
                jnp.zeros((height, width, 1), dtype=jnp.uint8)
                .at[pos.y, pos.x, 0]
                .set(1)
            )
            dir_layers = (
                jnp.zeros((height, width, 4), dtype=jnp.uint8)
                .at[pos.y, pos.x, direction]
                .set(1)
            )
            inv_grid = jnp.zeros_like(ingredients).at[pos.y, pos.x].set(inv)
            inv_layers = _ingridient_layers(
                inv_grid, ingredient_mapping=ingredient_mapping
            )

            return jnp.concatenate(
                [
                    pos_layers,
                    dir_layers,
                    inv_layers,
                ],
                axis=-1,
            )

        # Get observation for the active agent only
        active_agent = jax.tree_util.tree_map(
            lambda x: x[self._active_agent_idx], state.agents
        )
        
        ingredient_mapping = None
        if self.op_ingredient_permutations:
            ingredient_mapping = state.ingredient_permutations[self._active_agent_idx]

        agent_layer = _agent_layers(active_agent, ingredient_mapping)

        # No other agent layers for single-agent
        other_agent_layers = jnp.zeros_like(agent_layer)

        ingredients_layers = _ingridient_layers(
            ingredients, ingredient_mapping=ingredient_mapping
        )

        recipe_layers = _ingridient_layers(
            recipe_ingridients, ingredient_mapping=ingredient_mapping
        )

        ingredient_pile_encoding = jnp.array(
            [StaticObject.INGREDIENT_PILE_BASE + i for i in range(num_ingredients)]
        )
        if self.op_ingredient_permutations:
            ingredient_pile_encoding = ingredient_pile_encoding[ingredient_mapping]

        ingredient_pile_layers = (
            static_objects[..., None] == ingredient_pile_encoding
        )

        obs = jnp.concatenate(
            [
                agent_layer,
                other_agent_layers,
                static_layers,
                ingredient_pile_layers,
                ingredients_layers,
                recipe_layers,
                extra_layers,
            ],
            axis=-1,
        )

        # Apply view masking if needed
        if self.agent_view_size is not None:
            view_size = self.agent_view_size
            pos = active_agent.pos

            padded_obs = jnp.pad(
                obs,
                ((view_size, view_size), (view_size, view_size), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            obs = jax.lax.dynamic_slice(
                padded_obs,
                (pos.y, pos.x, 0),
                self.obs_shape,
            )

        return {"agent_0": obs}

    def get_obs_featurized(self, state: State) -> Dict[str, chex.Array]:
        """Get featurized observation for single agent."""
        if self.layout.num_ingredients > 1:
            raise NotImplementedError(
                "Featurized observation not implemented for more than 1 ingredient"
            )

        num_pots = 2

        onion = DynamicObject.ingredient(0)
        recipe = 3 * onion
        soup = recipe | DynamicObject.COOKED | DynamicObject.PLATE

        active_agent = jax.tree_util.tree_map(
            lambda x: x[self._active_agent_idx], state.agents
        )

        pos = active_agent.pos
        direction = active_agent.dir
        inv = active_agent.inventory

        reachable_area = self.enclosed_spaces == self.enclosed_spaces[pos.y, pos.x]
        reachable_area = mark_adjacent_cells(reachable_area)

        # pi_orientation: [NORTH, SOUTH, EAST, WEST]
        dir_features = jax.nn.one_hot(direction, 4)

        # pi_obj: ["onion", "soup", "dish", "tomato"]
        items = jnp.array(
            [
                DynamicObject.EMPTY,
                DynamicObject.ingredient(0),
                soup,
                DynamicObject.PLATE,
            ]
        )
        inv_features = jax.vmap(lambda item, inv: inv == item)(items, inv)
        inv_features = jnp.concatenate([inv_features, jnp.array([inv == 0])])

        # Find closest objects
        grid = state.grid

        def _find_closest(obj_encoding, default_pos):
            obj_mask = (grid[:, :, 1] == obj_encoding) | (grid[:, :, 1] == obj_encoding | DynamicObject.COOKED)
            obj_mask = obj_mask & (grid[:, :, 0] == StaticObject.EMPTY)
            
            if jnp.any(obj_mask):
                obj_positions = jnp.where(obj_mask)
                distances = jnp.abs(obj_positions[0] - pos.y) + jnp.abs(obj_positions[1] - pos.x)
                closest_idx = jnp.argmin(distances)
                closest_pos = Position(x=obj_positions[1][closest_idx], y=obj_positions[0][closest_idx])
                return jnp.array([closest_pos.x - pos.x, closest_pos.y - pos.y], dtype=jnp.float32)
            return jnp.array([0.0, 0.0], dtype=jnp.float32)

        # Closest onion, soup, dish, serving station, empty counter
        closest_onion = _find_closest(DynamicObject.ingredient(0), pos)
        closest_soup = _find_closest(soup, pos)
        closest_dish = _find_closest(DynamicObject.PLATE, pos)
        
        # Serving station (goal)
        serving_pos = jnp.where(grid[:, :, 0] == StaticObject.GOAL)
        if len(serving_pos[0]) > 0:
            distances = jnp.abs(serving_pos[0] - pos.y) + jnp.abs(serving_pos[1] - pos.x)
            closest_idx = jnp.argmin(distances)
            closest_serving = jnp.array([serving_pos[1][closest_idx] - pos.x, serving_pos[0][closest_idx] - pos.y], dtype=jnp.float32)
        else:
            closest_serving = jnp.array([0.0, 0.0], dtype=jnp.float32)

        # Empty counter
        empty_counter_mask = (grid[:, :, 0] == StaticObject.EMPTY) & (grid[:, :, 1] == DynamicObject.EMPTY)
        if jnp.any(empty_counter_mask):
            empty_positions = jnp.where(empty_counter_mask)
            distances = jnp.abs(empty_positions[0] - pos.y) + jnp.abs(empty_positions[1] - pos.x)
            closest_idx = jnp.argmin(distances)
            closest_empty = jnp.array([empty_positions[1][closest_idx] - pos.x, empty_positions[0][closest_idx] - pos.y], dtype=jnp.float32)
        else:
            closest_empty = jnp.array([0.0, 0.0], dtype=jnp.float32)

        # Pot features
        pot_positions = jnp.where(grid[:, :, 0] == StaticObject.POT)
        
        def _pot_features(pot_idx):
            if pot_idx < len(pot_positions[0]):
                px, py = pot_positions[1][pot_idx], pot_positions[0][pot_idx]
                pot_grid = grid[py, px, 1]
                pot_extra = grid[py, px, 2]
                
                is_empty = pot_grid == DynamicObject.EMPTY
                is_full = (pot_grid != DynamicObject.EMPTY) & (pot_extra == 0) & ~((pot_grid & DynamicObject.COOKED).astype(bool))
                is_cooking = pot_extra > 0
                is_ready = (pot_grid & DynamicObject.COOKED).astype(bool) & (pot_extra == 0)
                
                num_onions = jnp.sum(pot_grid == DynamicObject.ingredient(0))
                cook_time = jnp.where(pot_extra > 0, pot_extra, -1)
                
                return jnp.array([
                    1.0 if not (is_empty | is_full | is_cooking | is_ready) else 0.0,  # exists
                    float(is_empty),
                    float(is_full),
                    float(is_cooking),
                    float(is_ready),
                    float(num_onions),
                    float(cook_time),
                    float(px - pos.x),
                    float(py - pos.y),
                    0.0,  # padding
                    0.0,  # padding
                ], dtype=jnp.float32)
            return jnp.zeros(10, dtype=jnp.float32)

        pot_feats = jax.vmap(_pot_features)(jnp.arange(num_pots))

        # Wall features
        north_wall = pos.y == 0
        south_wall = pos.y == self.height - 1
        east_wall = pos.x == self.width - 1
        west_wall = pos.x == 0
        wall_features = jnp.array([north_wall, south_wall, east_wall, west_wall], dtype=jnp.float32)

        # Combine all features
        features = jnp.concatenate([
            dir_features,
            inv_features,
            closest_onion,
            closest_soup,
            closest_dish,
            closest_serving,
            closest_empty,
            pot_feats.flatten(),
            wall_features,
            jnp.array([float(pos.x), float(pos.y)], dtype=jnp.float32),
        ])

        return {"agent_0": features}

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, float, float]:
        """Internal step function for all agents."""
        grid = state.grid

        def _move_wrapper(agent, action):
            direction = ACTION_TO_DIRECTION[action]

            def _move(agent, dir):
                pos = agent.pos
                new_pos = pos.move_in_bounds(dir, self.width, self.height)

                new_pos = tree_select(
                    grid[new_pos.y, new_pos.x, 0] == StaticObject.EMPTY, new_pos, pos
                )

                return agent.replace(pos=new_pos, dir=direction)

            return jax.lax.cond(
                direction != -1,
                _move,
                lambda a, _: a,
                agent,
                direction,
            )

        new_agents = jax.vmap(_move_wrapper)(state.agents, actions)

        # Resolve collisions
        def _masked_positions(mask):
            return tree_select(mask, state.agents.pos, new_agents.pos)

        def _get_collisions(mask):
            positions = _masked_positions(mask)

            collision_grid = jnp.zeros((self.height, self.width))
            collision_grid, _ = jax.lax.scan(
                lambda grid, pos: (grid.at[pos.y, pos.x].add(1), None),
                collision_grid,
                positions,
            )

            collision_mask = collision_grid > 1

            collisions = jax.vmap(lambda p: collision_mask[p.y, p.x])(positions)
            return collisions

        initial_mask = jnp.zeros((self._num_agents_internal,), dtype=bool)
        mask = jax.lax.while_loop(
            lambda mask: jnp.any(_get_collisions(mask)),
            lambda mask: mask | _get_collisions(mask),
            initial_mask,
        )
        new_agents = new_agents.replace(pos=_masked_positions(mask))

        # Prevent swapping
        def _compute_swapped_agents(original_positions, new_positions):
            original_positions = original_positions.to_array()
            new_positions = new_positions.to_array()

            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)

            swapped_agents = jnp.any(swap_pairs, axis=0)
            return swapped_agents

        swap_mask = _compute_swapped_agents(state.agents.pos, new_agents.pos)
        new_agents = new_agents.replace(pos=_masked_positions(swap_mask))

        # Interact action
        def _interact_wrapper(carry, x):
            agent, action = x
            is_interact = action == Actions.interact

            def _interact(carry, agent):
                grid, correct_delivery, reward = carry

                (
                    new_grid,
                    new_agent,
                    new_correct_delivery,
                    interact_reward,
                    shaped_reward,
                ) = self.process_interact(
                    grid, agent, new_agents.inventory, state.recipe
                )

                carry = (
                    new_grid,
                    correct_delivery | new_correct_delivery,
                    reward + interact_reward,
                )
                return carry, (new_agent, shaped_reward)

            return jax.lax.cond(
                is_interact, _interact, lambda c, a: (c, (a, 0.0)), carry, agent
            )

        carry = (grid, False, 0.0)
        xs = (new_agents, actions)
        (new_grid, new_correct_delivery, reward), (new_agents, shaped_rewards) = (
            jax.lax.scan(_interact_wrapper, carry, xs)
        )

        # Update extra info
        def _timestep_wrapper(cell):
            def _cook(cell):
                is_cooking = cell[2] > 0
                new_extra = jax.lax.select(is_cooking, cell[2] - 1, cell[2])
                finished_cooking = is_cooking * (new_extra == 0)
                new_ingredients = cell[1] | (finished_cooking * DynamicObject.COOKED)

                return jnp.array([cell[0], new_ingredients, new_extra])

            def _indicator(cell):
                new_extra = jnp.clip(cell[2] - 1, min=0)
                return cell.at[2].set(new_extra)

            branches = (
                jnp.array(
                    [
                        StaticObject.POT,
                        StaticObject.BUTTON_RECIPE_INDICATOR,
                    ]
                )
                == cell[0]
            )

            branch_idx = jax.lax.select(
                jnp.any(branches),
                jnp.argmax(branches) + 1,
                0,
            )

            return jax.lax.switch(
                branch_idx,
                [
                    lambda x: x,
                    _cook,
                    _indicator,
                ],
                cell,
            )

        new_grid = jax.vmap(jax.vmap(_timestep_wrapper))(new_grid)

        sample_new_recipe = new_correct_delivery & self.sample_recipe_on_delivery

        key, subkey = jax.random.split(key)
        new_recipe = jax.lax.cond(
            sample_new_recipe,
            lambda _, key: self._sample_recipe(key),
            lambda r, _: r,
            state.recipe,
            subkey,
        )

        return (
            state.replace(
                agents=new_agents,
                grid=new_grid,
                recipe=new_recipe,
                new_correct_delivery=new_correct_delivery,
            ),
            reward,
            shaped_rewards,
        )

    def process_interact(
        self,
        grid: chex.Array,
        agent: Agent,
        all_inventories: jnp.ndarray,
        recipe: int,
    ):
        """Assume agent took interact action. Result depends on the faced cell."""

        inventory = agent.inventory
        fwd_pos = agent.get_fwd_pos()

        shaped_reward = jnp.array(0, dtype=float)

        interact_cell = grid[fwd_pos.y, fwd_pos.x]

        interact_item = interact_cell[0]
        interact_ingredients = interact_cell[1]
        interact_extra = interact_cell[2]
        plated_recipe = recipe | DynamicObject.PLATE | DynamicObject.COOKED

        object_is_plate_pile = interact_item == StaticObject.PLATE_PILE
        object_is_ingredient_pile = StaticObject.is_ingredient_pile(interact_item)

        object_is_pile = object_is_plate_pile | object_is_ingredient_pile
        object_is_pot = interact_item == StaticObject.POT
        object_is_goal = interact_item == StaticObject.GOAL
        object_is_wall = interact_item == StaticObject.WALL
        object_is_button_recipe_indicator = (
            interact_item == StaticObject.BUTTON_RECIPE_INDICATOR
        )

        object_has_no_ingredients = interact_ingredients == 0

        inventory_is_empty = inventory == 0
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        inventory_is_plate = inventory == DynamicObject.PLATE
        inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

        merged_ingredients = interact_ingredients + inventory

        pot_is_cooking = object_is_pot * (interact_extra > 0)
        pot_is_cooked = object_is_pot * (
            interact_ingredients & DynamicObject.COOKED != 0
        )
        pot_is_idle = object_is_pot * ~pot_is_cooking * ~pot_is_cooked

        successful_dish_pickup = pot_is_cooked * inventory_is_plate
        is_dish_pickup_useful = merged_ingredients == plated_recipe
        shaped_reward += (
            successful_dish_pickup
            * is_dish_pickup_useful
            * SHAPED_REWARDS["DISH_PICKUP"]
        )

        successful_pickup = (
            object_is_pile * inventory_is_empty
            + successful_dish_pickup
            + object_is_wall * ~object_has_no_ingredients * inventory_is_empty
        )

        successful_indicator_activation = (
            object_is_button_recipe_indicator
            * inventory_is_empty
            * object_has_no_ingredients
        )

        pot_full = DynamicObject.ingredient_count(interact_ingredients) == 3

        successful_pot_placement = pot_is_idle * inventory_is_ingredient * ~pot_full
        ingredient_selector = inventory | (inventory << 1)
        is_pot_placement_useful = (interact_ingredients & ingredient_selector) < (
            recipe & ingredient_selector
        )
        shaped_reward += (
            successful_pot_placement
            * is_pot_placement_useful
            * jax.lax.select(
                is_pot_placement_useful,
                1,
                -1 if self.negative_rewards else 0,
            )
            * SHAPED_REWARDS["PLACEMENT_IN_POT"]
        )

        successful_drop = (
            object_is_wall * object_has_no_ingredients * ~inventory_is_empty
            + successful_pot_placement
        )
        successful_delivery = object_is_goal * inventory_is_dish
        no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

        pile_ingredient = (
            object_is_plate_pile * DynamicObject.PLATE
            + object_is_ingredient_pile * StaticObject.get_ingredient(interact_item)
        )

        new_ingredients = (
            successful_drop * merged_ingredients + no_effect * interact_ingredients
        )
        pot_full_after_drop = DynamicObject.ingredient_count(new_ingredients) == 3

        successful_pot_start_cooking = (
            pot_is_idle
            * ~object_has_no_ingredients
            * inventory_is_empty
            * self.start_cooking_interaction
        )
        is_pot_start_cooking_useful = interact_ingredients == recipe
        shaped_reward += (
            successful_pot_start_cooking
            * is_pot_start_cooking_useful
            * SHAPED_REWARDS["POT_START_COOKING"]
        )
        auto_cook = pot_is_idle & pot_full_after_drop & ~self.start_cooking_interaction

        use_pot_extra = successful_pot_start_cooking | auto_cook
        new_extra = (
            use_pot_extra * POT_COOK_TIME
            + successful_indicator_activation * INDICATOR_ACTIVATION_TIME
            + ~use_pot_extra * ~successful_indicator_activation * interact_extra
        )

        new_cell = jnp.array([interact_item, new_ingredients, new_extra])

        new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

        new_inventory = (
            successful_pickup * (pile_ingredient + merged_ingredients)
            + no_effect * inventory
        )

        new_agent = agent.replace(inventory=new_inventory)

        is_correct_recipe = inventory == plated_recipe

        reward = jnp.array(0, dtype=float)
        reward += (
            successful_delivery
            * jax.lax.select(
                is_correct_recipe,
                1,
                -1 if self.negative_rewards else 0,
            )
            * DELIVERY_REWARD
        )

        reward -= successful_indicator_activation * INDICATOR_ACTIVATION_COST

        inventory_is_plate = new_inventory == DynamicObject.PLATE
        successful_plate_pickup = successful_pickup * inventory_is_plate
        num_plates_in_inventory = jnp.sum(all_inventories == DynamicObject.PLATE)
        num_nonempty_pots = jnp.sum(
            (grid[:, :, 0] == StaticObject.POT) & (grid[:, :, 1] != 0)
        )
        is_plate_pickup_useful = num_plates_in_inventory < num_nonempty_pots
        no_plates_on_counters = jnp.sum(grid[:, :, 1] == DynamicObject.PLATE) == 0
        shaped_reward += (
            no_plates_on_counters
            * is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP"]
        )

        correct_delivery = successful_delivery & is_correct_recipe
        return new_grid, new_agent, correct_delivery, reward, shaped_reward

    def is_terminal(self, state: State) -> bool:
        """Check if episode is done."""
        return jnp.logical_or(
            state.time >= self.max_steps,
            state.terminal,
        )

    @property
    def agent_ids(self) -> List[str]:
        return self.agents

    def get_legacy_env(self):
        """Return the underlying multi-agent environment for multi-agent training.
        
        This allows using the same environment for multi-agent training after
        pretraining a single agent.
        """
        from jaxmarl.environments.overcooked_single.overcooked import OvercookedV2
        
        return OvercookedV2(
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
