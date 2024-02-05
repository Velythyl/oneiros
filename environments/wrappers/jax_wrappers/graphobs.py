import functools

import jax
import jax.numpy as jp
from brax.envs.base import Wrapper, Env, State
from flax import struct


class GraphObsWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, graph_spec, replace_obs=True):
        super().__init__(env)
        self.graph_spec = graph_spec
        self.nodes, self.encode_state, self.edges, self.get_action, self.encode_reward = make_graph_modifyer(graph_spec, env)
        self.get_action = jax.jit(self.get_action)

        self.replace_obs = replace_obs
        self.env = env

        if replace_obs:
            def _possibly_replace_state(state, graph_obs):
                state = state.replace(obs=graph_obs)
                return state
        else:
            def _possibly_replace_state(state, graph_obs):
                return state
        possibly_replace_state = jax.jit(_possibly_replace_state)

        @jax.jit
        def _vmap_edges_util(nodes):
            return self.edges
        vmap_edges_util = jax.jit(jax.vmap(_vmap_edges_util))

        @jax.jit
        def encode_state(state):
            encoded_on_nodes = jax.vmap(self.encode_state)(pipeline_state=state.pipeline_state)
            nodes_obs = jax.vmap(self.encode_reward)(encoded_on_nodes, state.reward)

            edges = vmap_edges_util(nodes_obs)
            graph_obs = GraphObs(nodes_obs, edges)

            infos = state.info
            infos["nodes"] = nodes_obs
            infos["edges"] = edges
            infos["graphobs"] = graph_obs
            state = state.replace(info=infos)

            state = possibly_replace_state(state, graph_obs)

            return state

        if replace_obs:
            def _get_action(action):
                return jax.vmap(self.get_action)(action)
        else:
            def _get_action(action):
                return action
        get_action = jax.jit(_get_action)

        def step(state: State, action: jp.ndarray) -> State:
            action = get_action(action)
            return encode_state(self.env.step(state, action))
        self._step = jax.jit(step)

        def reset(rng):
            return encode_state(self.env.reset(rng))
        self._reset = jax.jit(reset)


    def reset(self, rng):
        return self._reset(rng)

    def step(self, state, action):
        return self._step(state, action)

    @property
    def observation_size(self):
        return self.nodes.shape if self.replace_obs else self.env.observation_size

@struct.dataclass
class GraphObs:
    nodes: jp.ndarray
    edges: jp.ndarray

from brax.io.torch import jax_to_torch
@jax_to_torch.register(GraphObs)
def _graphobs_to_tensor(
        value: GraphObs, device = None
):
    nodes = jax_to_torch(value.nodes, device)
    edges = jax_to_torch(value.edges, device)
    return GraphObs(nodes, edges)

@jax.jit
def concat_nodes_and_edges(nodes, edges):
    return jp.concatenate((nodes, edges), axis=1)

def get_body2braxidx(system):
    q_idx = 0
    qd_idx = 0
    q = {}
    qd = {}

    for i, joint_name in enumerate(system.link_names):
        from brax.base import Q_WIDTHS, QD_WIDTHS
        n_q_idx = Q_WIDTHS[system.link_types[i]] + q_idx
        q[joint_name] = jp.arange(q_idx, n_q_idx, 1)

        n_qd_idx = QD_WIDTHS[system.link_types[i]] + qd_idx
        qd[joint_name] = jp.arange(qd_idx, n_qd_idx, 1)

        q_idx = n_q_idx
        qd_idx = n_qd_idx

    return q, qd

def make_graph_modifyer(graph_spec, env):
    nodes = (graph_spec["nodes"])
    node_indices = graph_spec["node_indices"]
    edges = (graph_spec["edges"])
    attrs = graph_spec["attr_indices"]

    system = env.unwrapped.sys
    brax_q_idx, brax_qd_idx = get_body2braxidx(system)

    def extract_nodename(node):
        return "_".join(node.split("_")[1:])  # remove "body_"

    motor2body = graph_spec["motor2body"]
    action_in_attr = {}
    for actuator_link_id in system.actuator_link_id:
        corresponding_body_name = system.link_names[actuator_link_id]
        for i, (motor, body_name) in enumerate(motor2body.items()):
            FOUND = False
            if extract_nodename(body_name) == corresponding_body_name:
                action_in_attr[motor] = i
                FOUND = True
                break
        assert FOUND
    motor_nodes_in_graph = jp.array([node_indices[x] for x in motor2body.keys()])
    action_in_attr = jp.array(list(action_in_attr.values()))

    motor_node_indices = {}
    for node_name, indices in node_indices.items():
        if node_name.startswith("motor_"):
            motor_node_indices[node_name] = indices

    Q_nodename2braxidx = {}
    QD_nodename2braxidx = {}

    for key, node_index in node_indices.items():
        if key.startswith("body_"):  # todo subject to change, currently mapping brax's links to mjcf's bodies
            part = extract_nodename(key)

            if part in brax_q_idx:
                Q_nodename2braxidx[key] = brax_q_idx[part]
                QD_nodename2braxidx[key] = brax_qd_idx[part]
                del brax_q_idx[part]
                del brax_qd_idx[part]
    assert len(brax_q_idx) == len(brax_qd_idx) == 0

    # TODO assert that the attrs are all at 0 initially (should be true)
    @jax.jit
    def nodes_set_attr(nodes, node_id, attr_id, val):
        return nodes.at[node_id, attr_id].set(val)


    link_nodes_in_graph = []
    q_in_attr = []
    for body_node, idxs_in_brax in Q_nodename2braxidx.items():
        link_nodes_in_graph.append(node_indices[body_node])
        q_in_attr.append(idxs_in_brax)


    # this is only used by env
    @jax.jit
    def encode_state(nodes, pipeline_state):

        for bodypart, range in Q_nodename2braxidx.items():
            node_id = node_indices[bodypart]
            q_attr_id = attrs["state_pos"]
            qd_attr_id = attrs["state_vel"]

            q_val = pipeline_state.q[Q_nodename2braxidx[bodypart]]
            qd_val = pipeline_state.qd[QD_nodename2braxidx[bodypart]]

            nodes = nodes_set_attr(nodes, node_id, q_attr_id, q_val)
            nodes = nodes_set_attr(nodes, node_id, qd_attr_id, qd_val)

        return nodes

    @jax.jit
    def encode_reward(nodes, reward):
        for motor_indices in motor_node_indices.values():
            nodes = nodes_set_attr(nodes, motor_indices, attrs["reward"], reward)
        return nodes

    @jax.jit
    def get_action_brax(node_from_agent):
        return node_from_agent[motor_nodes_in_graph, action_in_attr]

    def get_action_torch(nodes_from_agent):
        actions = nodes_from_agent[:, motor_nodes_in_graph, action_in_attr]
        return actions

    return nodes, functools.partial(encode_state, nodes=nodes), edges, get_action_brax, encode_reward