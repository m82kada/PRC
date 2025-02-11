# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module with routing layers."""
import functools
from dataclasses import field
from typing import Any, Mapping, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import vmoe.moe

from vmoe.nn.soft_router import SoftRouter

Array = jnp.ndarray
BaseDispatcher = vmoe.moe.BaseDispatcher
DType = type(jnp.float32)
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, Array]


class NoisyTopExpertsPerItemRouter(nn.Module):
  """Noisy TopExpertsPerItem router used in https://arxiv.org/abs/2106.05974.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (item, expert). Noise is added to these logits. The logits are normalized
  using a softmax over the expert dimension. This score will be used to
  determine which items are dispatched to which experts and how the outputs of
  the experts are combined.

  Because the routing algorithm is non-differentiable, the only way to train the
  parameters of the dense (a.k.a. gating layer) is through the weights used
  to combine the output of the experts, and through two auxiliary losses that
  depend on the output of the gating.
  """
  num_experts: int
  num_selected_experts: int = 1
  noise_std: float = 1.0
  gshard_loss_weight: float = 0.0
  importance_loss_weight: float = 1.0
  load_loss_weight: float = 1.0
  similarity_loss_weight: float = 1.0
  dispatcher: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None
  additional_losses: list = field(default_factory=list)
  #similarity_loss_input: str = "original" # original, noisy
  #similarity_loss_type: str = "l2" # l2, (l2_nosqrt), jsd, l2_topk, jsd_topk, m_entropy
  #onehot_loss_type: str = "entropy" # minas_sum, entropy
  teacher_student: bool = False
  layer_num: int = 8
  similarity_loss_layer: list = field(default_factory=list)
  only_cls_contrastive_layer: list = field(default_factory=list)
  #barlow_lambd: float = 0.3
  cls_alpha: float = 0.
  cls_concat: bool = False
  stop_grad: bool = False
  auxiliary_loss_decay: str = "none" # none / linear / sigma

  @nn.compact
  def __call__(self, inputs: Array, patch_correspondence,
               return_router=False, learn_per=0., return_router_input=False) -> Tuple[BaseDispatcher, Metrics]:
    num_experts = self.num_experts
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    if not num_experts >= self.num_selected_experts >= 1:
      raise ValueError(f"num_experts >= num_selected_experts >= 1, but got "
                       f"num_experts = {num_experts} and "
                       f"num_selected_experts = {self.num_selected_experts}.")
    dtype = self.dtype or inputs.dtype

    importance_loss_weight, load_loss_weight = self.importance_loss_weight, self.load_loss_weight
    if self.auxiliary_loss_decay == "linear":
      importance_loss_weight = self.importance_loss_weight * (1.0 - learn_per)
      load_loss_weight = self.load_loss_weight * (1.0 - learn_per)
    if self.auxiliary_loss_decay == "sigma":
      importance_loss_weight = self.importance_loss_weight * jnp.where(learn_per > 0.5, 0., 1.)
      load_loss_weight = self.load_loss_weight * jnp.where(learn_per > 0.5, 0., 1.)

    # CLS Blend for Image Patch
    if self.cls_alpha > 0.:
      _shape = inputs.shape
      inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
      inputs = jax.lax.stop_gradient(self.cls_alpha * inputs[:,0,:].reshape(-1, 1, inputs.shape[-1])) + (1 - self.cls_alpha) * inputs
      inputs = inputs.reshape(_shape)

    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)

    # Compute the auxiliary losses defined in Appendix A.2, from
    # https://arxiv.org/abs/2106.05974. Notice that the "Load Loss" can only be
    # computed if the router is stochastic (i.e. deterministic = False).
    # Notice that the auxiliary losses are computed on each group independently
    # (i.e. through the vmaps surrounding the calls).
    gates_softmax = jax.nn.softmax(gates_logits)
    #gates_softmax = jnp.ones(gates_softmax.shape) / 8
    importance_loss = jax.vmap(self._importance_auxiliary_loss)(gates_softmax)

    #sum_loss = importance_loss * (importance_loss_weight if self.layer_num in self.similarity_loss_layer else 5e-3)
    sum_loss = importance_loss * importance_loss_weight
    metrics = {"importance_loss": importance_loss}

    if self.layer_num in self.only_cls_contrastive_layer:
      patch_correspondence = jnp.ones([inputs.shape[0], 1, 2], dtype=jnp.int32) # [B, P(cls), 2]
    
    if len(self.additional_losses) > 0 and patch_correspondence is not None and self.layer_num in self.similarity_loss_layer:
      _gates_softmax = jnp.concatenate([gates_softmax.reshape(patch_correspondence.shape[0], 2, -1, gates_softmax.shape[-1]), jnp.zeros([patch_correspondence.shape[0], 2, 1, gates_softmax.shape[-1]])], axis=2)
      pc_cnt = jnp.sum(jnp.where(patch_correspondence > -1, 0.5, 0.))
      # router: (B, 2, P+1, D), patch_correspondence: (B, X)
      l_not_skip_when_cnt_zero = ['two_hot', 'entropy']
      loss_functions = { # (router, patch_correspondence) -> loss
        'l2': lambda r, pc: jnp.sum(
          jax.vmap(
            lambda i, r:
              jnp.sum(
                jnp.where(i[:,0] > -1, jnp.sqrt(jnp.sum((r[0, i[:,0]] - r[1, i[:,1]]) ** 2, axis=-1) + 1e-9), jnp.zeros(i.shape[0]))
              ),
          )(pc, r) / pc_cnt
        ),
        'jsd': lambda r, pc: jnp.sum(
          jax.vmap(
            lambda i, r:
              jnp.sum(
                jnp.where(i[:,0] > -1, jnp.sum(jsd(r[0, i[:,0]], r[1, i[:,1]]), axis=-1), jnp.zeros(i.shape[0]))
              )
          )(pc, r) / pc_cnt
        ),
        'jsd_topk': lambda r, pc: jnp.sum(
          jax.vmap(
            lambda i, r:
              jnp.sum(
                jnp.where(i[:,0] > -1, jnp.sum(jsd_topk(r[0, i[:,0]], r[1, i[:,1]]), axis=-1), jnp.zeros(i.shape[0]))
              )
          )(pc, r) / pc_cnt
        ),
        'two_hot': lambda r, pc: jnp.average(
          jnp.sqrt(
            jnp.sum(jax.lax.sort(r)[:,:,:-1,2:] ** 2, axis=-1) + 1e-6
          ),
        ),
        'two_hot_mul': lambda r, pc: two_hot_mul(r),
        'entropy': lambda r, pc: -jnp.sum(r[:,:,:-1,:] * jnp.log(r[:,:,:-1,:] + 1e-7)) / r.shape[0],
        'old_matrix_top1': lambda r, pc: barlowtwins(gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: jnp.matmul(r[0, i[:,0]].transpose(), r[1, i[:, 1]]))(pc, r), axis=0) / pc_cnt, 10),
        #'matrix_top1': lambda r, pc: barlowtwins(gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: jnp.matmul(r[0, i[:,0]].transpose(), r[1, i[:, 1]]))(pc, r), axis=0) / pc_cnt, self.barlow_lambd),
        #'matrix_top2': lambda r, pc: barlowtwins2(gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: matrix_top2(i, r))(pc, r), axis=0) / cnt, self.barlow_lambd),
        'matrix_top1_diagonal': lambda r, pc: jnp.average((gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: r[0, i[:,0]] * r[1, i[:, 1]])(pc, r), axis=0) / pc_cnt - 1) ** 2),
        'matrix_top1_offdiagonal': lambda r, pc: barlowtwins_offdiagonal(gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: jnp.matmul(r[0, i[:,0]].transpose(), r[1, i[:, 1]]))(pc, r), axis=0) / pc_cnt),
        #'matrix_top2_diagonal': lambda r, pc: barlowtwins_diagonal(gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: matrix_top2(i, r))(pc, r), axis=0) / cnt),
        'matrix_top2_diagonal': lambda r, pc: barlowtwins2_diagonal(gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: matrix_top2(i, r, r.shape[0]))(pc, r), axis=0) / cnt),
        'matrix_top2_offdiagonal': lambda r, pc: barlowtwins_offdiagonal(gates_softmax.shape[-1] * jnp.sum(jax.vmap(lambda i, r: matrix_top2(i, r, r.shape[0]))(pc, r), axis=0) / cnt),
        'matrix_top1_diagonal_sqrt': lambda r, pc: jnp.average((gates_softmax.shape[-1] * jnp.sqrt(jnp.sum(jax.vmap(lambda i, r: r[0, i[:,0]] * r[1, i[:, 1]])(pc, r), axis=0) + 1e-9) / pc_cnt - 1) ** 2),
        'matrix_top1_offdiagonal_sqrt': lambda r, pc: barlowtwins_offdiagonal(gates_softmax.shape[-1] * jnp.sqrt(jnp.sum(jax.vmap(lambda i, r: jnp.matmul(r[0, i[:,0]].transpose(), r[1, i[:, 1]]))(pc, r), axis=0) + 1e-9) / pc_cnt),
      }
      for loss_name, loss_weight in self.additional_losses:
        # patch_correspondenceが全て-1とならないよう対策するのでここのコードは不要のはず
        if loss_name in l_not_skip_when_cnt_zero:
          loss = loss_functions[loss_name](_gates_softmax, patch_correspondence)
        else:
          # pc_cnt が 0 のときの zero-div 対策
          loss = jax.lax.cond(jnp.greater(pc_cnt, 0.), lambda: loss_functions[loss_name](_gates_softmax, patch_correspondence), lambda: jnp.array(0.))
        if loss_weight > 0.:
          sum_loss += loss_weight * loss
        metrics[loss_name] = loss

    if self.deterministic or self.noise_std == 0.0:
      dispatcher = self._create_dispatcher(gates_softmax)
      gshard_loss = jax.vmap(self._gshard_auxiliary_loss)(gates_softmax)
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      logits_noise = noise_std * jax.random.normal(
          key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)
      dispatcher = self._create_dispatcher(gates_softmax_noisy)
      load_loss = jax.vmap(
          functools.partial(
              self._load_auxiliary_loss,
              num_selected_experts=self.num_selected_experts,
              noise_std=noise_std))(gates_logits, gates_logits_noisy)
      sum_loss += load_loss_weight * load_loss
      #sum_loss += (load_loss_weight if self.layer_num in self.similarity_loss_layer else 5e-3) * load_loss
      metrics["load_loss"] = load_loss
      gshard_loss = jax.vmap(self._gshard_auxiliary_loss)(gates_softmax_noisy)

    sum_loss += self.gshard_loss_weight * gshard_loss
    metrics["gshard_loss"] = gshard_loss

    metrics["auxiliary_loss"] = sum_loss
    if return_router_input:
      metrics["router_input"] = inputs
    if return_router:
      metrics["router"] = gates_softmax
    return dispatcher, metrics

  @nn.nowrap
  def _create_dispatcher(self, gates_dispatch):
    # Creates a dispatcher implementing the TopExpertsPerItem routing algorithm,
    # that uses at most `num_selected_experts` per item. Notice that each
    # group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_experts_per_item_dispatcher_vmapped = jax.vmap(
        functools.partial(
            vmoe.moe.get_top_experts_per_item_dispatcher,
            num_selected_experts=self.num_selected_experts,
            **dispatcher_kwargs))
    if self.stop_grad:
      dispatcher = get_top_experts_per_item_dispatcher_vmapped(jax.lax.stop_gradient(gates_dispatch))
    else:
      dispatcher = get_top_experts_per_item_dispatcher_vmapped(gates_dispatch)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher

  @classmethod
  def _gshard_auxiliary_loss(cls, gates: Array) -> Array:
    # See `l_{aux}` in Algorithm 1 in https://arxiv.org/pdf/2006.16668.pdf.
    _, num_experts = gates.shape
    # Line (3) in Algorithm 1.
    mean_gates_per_expert = gates.mean(axis=0)
    # Lines (11, 13) in Algorithm 1.
    mean_top1_per_expert = jax.nn.one_hot(
        jnp.argmax(gates, axis=1), num_experts, dtype=jnp.int32).mean(axis=0)
    # Note: Only gradients through mean_gates_per_expert affect the gating,
    # since hard counts from top_k+one_hot are not differentiable.
    auxiliary_loss = jnp.mean(mean_top1_per_expert * mean_gates_per_expert)
    # Note: Not mentioned in the paper, but it's done in their source code.
    # https://github.com/tensorflow/lingvo/blob/84b85514d7ad3652bc9720cb45acfab08604519b/lingvo/core/gshard_layers.py#L2223
    auxiliary_loss *= num_experts**2
    return auxiliary_loss

  @classmethod
  def _importance_auxiliary_loss(cls, gates: Array) -> Array:
    axis = tuple(range(gates.ndim - 1))  # All except last.
    importance_per_expert = jnp.sum(gates, axis=axis)
    std_importance_per_expert = jnp.std(importance_per_expert)
    mean_importance_per_expert = jnp.mean(importance_per_expert)
    # Compute coefficient of variation (i.e. std/mean) squared.
    return (std_importance_per_expert / mean_importance_per_expert)**2

  @classmethod
  def _load_auxiliary_loss(cls, logits: Array, logits_noisy: Array,
                           noise_std: Array,
                           num_selected_experts: int) -> Array:
    # For each example, compute the weight required for an expert to be selected
    # among the top-k.
    # NOTE: DO NOT TRY TO SIMPLIFY THIS. This convoluted way of obtaining the
    # threshold_per_item avoids adding all-gather ops during backpropagation.
    num_experts = logits_noisy.shape[-1]
    threshold_per_item_index = jax.lax.top_k(
        logits_noisy, num_selected_experts)[-1][..., -1]
    threshold_per_item = jnp.sum(
        jax.nn.one_hot(threshold_per_item_index, num_experts) * logits_noisy,
        axis=-1)
    # For each example and expert, find how far they were from the threshold and
    # normalize this value by the noise_std to use the standard Gaussian CDF.
    noise_required_to_win = threshold_per_item[..., None] - logits
    noise_required_to_win /= noise_std
    # p is the probability of being above the threshold for each (item, expert)
    # if the random noise (with its std) was re-sampled again.
    p = 1. - jax.scipy.stats.norm.cdf(noise_required_to_win)
    # We compute the average such probability for each expert over examples.
    p_mean = jnp.mean(p, axis=0)
    # Compute p_mean's coefficient of variation squared.
    return (jnp.std(p_mean) / jnp.mean(p_mean))**2


class NoisyTopItemsPerExpertRouter(nn.Module):
  """Noisy TopItemsPerExpert router.

  Instead of picking the Top-K experts with highest score for each item, and
  then ignore choices that exceed the capacity (C) of any given expert, here we
  pick the Top-C items with highest score for each expert.

  This makes the load across experts automatically balanced, however the number
  of experts assigned to each item is not bounded and can vary. Some items may
  not be routed to any expert. In practice, though, this works very well.

  This was coined "Experts Choice Routing" in https://arxiv.org/abs/2202.09368.
  """
  num_experts: int
  noise_std: float = 1.0
  dispatcher: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None

  num_selected_experts: int = 1
  teacher_student: bool = False
  layer_num: int = 8
  similarity_loss_layer: list = field(default_factory=list)
  only_cls_contrastive_layer: list = field(default_factory=list)
  cls_alpha: float = 0.
  cls_concat: bool = False
  stop_grad: bool = False
  auxiliary_loss_decay: str = "none" # none / linear / sigma

  @nn.compact
  def __call__(self, inputs: Array, patch_correspondence=None,
               return_router=False, learn_per=0.) -> Tuple[BaseDispatcher, Metrics]:
    gates_softmax = self._compute_gates_softmax(inputs, self.num_experts)
    dispatcher, metrics = self._create_dispatcher_and_metrics(gates_softmax)
    metrics["auxiliary_loss"] = 0.
    return dispatcher, metrics

  @nn.nowrap
  def _compute_gates_softmax(self, inputs: Array, num_experts: int) -> Array:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    if self.deterministic or self.noise_std == 0.0:
      gates_softmax = jax.nn.softmax(gates_logits)
      return gates_softmax
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      logits_noise = noise_std * jax.random.normal(
          key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)
      return gates_softmax_noisy

  @nn.nowrap
  def _create_dispatcher_and_metrics(self, gates_dispatch):
    # Creates a dispatcher implementing the TopItemsPerExpert routing algorithm.
    # Notice that each group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_items_per_expert_dispatcher_vmapped = jax.vmap(
        functools.partial(
            vmoe.moe.get_top_items_per_expert_dispatcher, **dispatcher_kwargs))
    dispatcher, metrics = get_top_items_per_expert_dispatcher_vmapped(
        gates_dispatch)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher, metrics


def _weighted_sum(args):
  """Returns a weighted sum of [(weight, element), ...] for weights > 0."""
  # Note: some losses might be ill-defined in some scenarios (e.g. they may
  # have inf/NaN gradients), in those cases we don't apply them on the total
  # auxiliary loss, by setting their weights to zero.
  return sum(x * w for w, x in args if w > 0)

def kld(p, q):
  return jnp.sum(p * jnp.log(p / (q + 1e-7) + 1e-7), axis=-1)
  # softmaxなのでp>0を仮定
  #return jnp.sum(jnp.where(p != 0, p * jnp.log(p / q), 0))

def jsd(p, q):
  return (kld(p, (p+q)/2) + kld(q, (p+q)/2)) / 2

def kld_topk(p, q, k):
  return jnp.sum(jax.lax.sort_key_val(-p, p * jnp.log(p / (q + 1e-7) + 1e-7))[1][:,:k], axis=-1)

def jsd_topk(p, q, k=2):
  return (kld_topk(p, (p+q)/2, k) + kld_topk(q, (p+q)/2, k)) / 2

def barlowtwins(x, lambd):
  # TODO: sum? ave?
  on_diag = jnp.average((jnp.diag(x) - 1.) ** 2)
  off_diag = jnp.average(off_diagonal(x) ** 2)
  return on_diag + off_diag * lambd

def off_diagonal(x):
  n, m = x.shape
  assert n == m
  return x.flatten()[:-1].reshape(n-1, n+1)[:,1:].flatten()

def barlowtwins2(x, lambd):
  on_diag = jnp.average(jnp.diag(x - 0.25) ** 2) # ???
  off_diag = jnp.average(off_diagonal(x) ** 2)
  return on_diag + off_diag * lambd

def entropy_fn(i):
    return -jnp.sum(i * jnp.log(i), axis=-1)

def matrix_top2(i, r, experts_num=8):
  # r: (2, P, E), i: (X, 2)
  a = r[0, i[:,0]] # (X, E)
  b = r[1, i[:,1]] # (X, E)
  a = jnp.sqrt(jnp.matmul(a.transpose(), a)) # (E, E)
  b = jnp.sqrt(jnp.matmul(b.transpose(), b)) # (E, E)
  index = jnp.array([[i < j for j in range(experts_num)] for i in range(experts_num)])

  a = a[index]
  b = b[index]

  return jnp.matmul(a.transpose(), b)

def barlowtwins_offdiagonal(x):
  return jnp.average(off_diagonal(x) ** 2)

def barlowtwins2_diagonal(x):
  return jnp.average(jnp.diag(x - 0.25) ** 2)

def two_hot_mul(x):
  # x: (2, P, E)
  x = x.reshape([-1, x.shape[-1], 1])
  y = jnp.matmul(x, x.transpose((0, 2, 1))) * jnp.triu(jnp.ones([x.shape[-1], x.shape[-1]]), k=1) # (?, E, E)    i < j
  z = jnp.matmul(y.reshape([-1, x.shape[-1], x.shape[-1], 1]), x.transpose((0, 2, 1))) * jnp.triu(jnp.ones([x.shape[-1], x.shape[-1]]), k=1) # (?, E, E, E)   i < j < k
  return jnp.average(z)
