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

"""Module with several util functions."""
import ast
import collections.abc
import importlib
import re
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Type, Union
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

PRNGKey = jax.random.KeyArray


def make_rngs(rng_keys: Tuple[str, ...], seed: int) -> Dict[str, PRNGKey]:
  if not rng_keys:
    return dict()
  rngs = jax.random.split(jax.random.PRNGKey(seed), len(rng_keys))
  return dict(zip(rng_keys, rngs))


def multiply_no_nan(x, y):
  """Multiplies x and y and returns 0 if x is 0, even if y is not finite."""
  # Note: This is equivalent to tf.math.multiply_no_nan, with safe gradients.
  x_ok = x != 0.
  safe_x = jnp.where(x_ok, x, 1.)
  safe_y = jnp.where(x_ok, y, 1.)
  return jnp.where(x_ok, jax.lax.mul(safe_x, safe_y), jnp.zeros_like(x))


def parse_call(string: str, default_module: Union[str, Any]):
  """Parses a string representing a call.

  Examples:
    - parse_call('foo', module): Returns (module.foo, (), {}).
    - parse_call('foo(25)', module): Returns (module.foo, (25,), {}).
    - parse_call('foo.bar.baz', module): Returns ('foo.bar.baz', (), {}).

  Args:
    string: This can be either a name (it assumes no arguments) or a call string
      including the positional and keyword arguments. The call cannot include
      nested calls (e.g. "foo.bar().baz()" is not allowed). The optional args
      must be Python literals.
    default_module: Default module to use to import the function.

  Returns:
    Returns the callable (e.g. a class or function), a tuple of positional args,
    and a dictionary of keyword arguments.
  """
  expr = ast.parse(string, mode='eval').body
  if isinstance(expr, ast.Call):
    # Parses the positional and keyword arguments in strings like:
    # "foo.bar.baz(a, b=c)".
    args = tuple([ast.literal_eval(arg) for arg in expr.args])
    kwargs = {
        kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in expr.keywords
    }
    # Prepare to process the rest of the expression (e.g. "foo.bar.baz").
    expr = expr.func
  else:
    args, kwargs = (), {}
  if isinstance(expr, ast.Name):
    # After the (optional) call arguments, the expression is a name: e.g. "foo".
    module = default_module
    name = expr.id
  elif isinstance(expr, ast.Attribute):
    name = expr.attr
    expr = expr.value
    module = string[:expr.end_col_offset]
    # We check that the expression is something like:
    # "name.attribute_1.....attribute_n".
    # For instance, something like "foo.bar().baz" is not be accepted.
    while not isinstance(expr, ast.Name):
      if not isinstance(expr, ast.Attribute):
        raise ValueError(f'{string=!r} is not a supported callable string.')
      expr = expr.value
  else:
    raise ValueError(f'{string=!r} is not a supported callable string.')
  if isinstance(module, str):
    module = importlib.import_module(module)
  return getattr(module, name), args, kwargs


def partialclass(cls: Type[Any], *base_args, **base_kwargs):
  """Builds a subclass with partial application of the given args and keywords."""

  class _NewClass(cls):

    def __init__(self, *args, **kwargs):
      bound_args = base_args + args
      bound_kwargs = base_kwargs.copy()
      bound_kwargs.update(kwargs)
      super().__init__(*bound_args, **bound_kwargs)

  return _NewClass


class SafeZipIteratorError(RuntimeError):
  pass


class SafeZipIterator:
  """Lazy zip over multiple iterators, ensuring that all have the same length."""

  def __init__(self, *iterators):
    self.iterators = tuple(
        i if isinstance(i, collections.abc.Iterator) else iter(i)
        for i in iterators)

  def __iter__(self):
    return self

  def __next__(self) -> Tuple[Any, ...]:
    stop = None
    elements = []
    for i, iterator in enumerate(self.iterators):
      try:
        elements.append(next(iterator))
        if stop is not None:
          break
      except StopIteration:
        stop = i
    if stop is not None and elements:
      raise SafeZipIteratorError(
          f'The {stop}-th iterator raised StopIteration before the rest')
    if not elements:
      raise StopIteration
    return tuple(elements)


def safe_map(f: Callable[..., Any], *iterables) -> Iterator[Any]:
  for args in SafeZipIterator(*iterables):
    yield f(*args)


def safe_zip(*iterables) -> Iterator[Tuple[Any, ...]]:
  return SafeZipIterator(*iterables)


def tree_rngs_split(rngs, num_splits=2):
  """Splits a PyTree of PRNGKeys into num_splits PyTrees."""
  rngs = jax.tree_map(lambda rng: jax.random.split(rng, num_splits), rngs)
  slice_rngs = lambda rngs, i: jax.tree_map(lambda rng: rng[i], rngs)
  return tuple(slice_rngs(rngs, i) for i in range(num_splits))


def make_match_fn_from_regex_list(
    regexes: Optional[Sequence[str]]) -> Optional[Callable[[str], bool]]:
  """Creates a function returning True iff a str matches any of the regexes."""

  if not regexes:
    return None
  if isinstance(regexes, str):
    regexes = [regexes]
  joined_regex = re.compile(
      '(?:' + '|'.join([f'(?:{r})' for r in regexes]) + ')')

  def fn(string: str) -> bool:
    return joined_regex.search(string) is not None
  return fn

"""
def window_to_patch_correspondence(window, image_size=384, patch_size=32, use_cls=1, border_iou=0.6, return_num=289):
  # IoUもどきの計算をand/minとし，borderを0.5とすると，patch_correspondenceは高々patch_len^2+use_cls個に収まる
  # window: 2 x 4(crop window)

  def _window_conv(idx):
    patch_len = image_size // patch_size
    y1 = idx % patch_len
    x1 = (idx // patch_len) % patch_len
    y2 = (idx // (patch_len ** 2)) % patch_len
    x2 = (idx // (patch_len ** 3)) % patch_len

    # 画像1の元座標のx座標の(l, r) [必ずしもl<rとは限らない]
    tmp1 = jnp.array([
      (window[0][2] - window[0][0]) * y1 / patch_len + window[0][0],
      (window[0][2] - window[0][0]) * (y1 + 1) / patch_len + window[0][0]
    ])
    # 画像1の元座標のy座標の(l, r)
    tmp2 = jnp.array([
      (window[0][3] - window[0][1]) * x1 / patch_len + window[0][1],
      (window[0][3] - window[0][1]) * (x1 + 1) / patch_len + window[0][1]
    ])
    # 画像1の元座標の(x_r, y_r, x_l, y_l)
    res1 = jnp.array([jnp.max(tmp1), jnp.max(tmp2), jnp.min(tmp1), jnp.min(tmp2)])

    tmp3 = jnp.array([
      (window[1][2] - window[1][0]) * y2 / patch_len + window[1][0],
      (window[1][2] - window[1][0]) * (y2 + 1) / patch_len + window[1][0]
    ])
    tmp4 = jnp.array([
      (window[1][3] - window[1][1]) * x2 / patch_len + window[1][1],
      (window[1][3] - window[1][1]) * (x2 + 1) / patch_len + window[1][1]
    ])
    # 画像2の元座標の(x_r, y_r, x_l, y_l)
    res2 = jnp.array([jnp.max(tmp3), jnp.max(tmp4), jnp.min(tmp3), jnp.min(tmp4)])

    _height = jnp.maximum(0., jnp.minimum(res1[0], res2[0]) - jnp.maximum(res1[2], res2[2]))
    _width = jnp.maximum(0., jnp.minimum(res1[1], res2[1]) - jnp.maximum(res1[3], res2[3]))
    and_area = _height * _width
    res1_area = (res1[0] - res1[2]) * (res1[1] - res1[3])
    res2_area = (res2[0] - res2[2]) * (res2[1] - res2[3])
    iou = and_area / jnp.minimum(res1_area, res2_area)
    #iou = and_area / (res1_area + res2_area - and_area)
    #or_area = res1_area + res2_area - and_area
    #return and_area / or_area
    # IoU では無いので注意
    # ちなみにminの場合，値はパッチの位置によらずに一定となる
    # border_iouはIoUの場合は0.15程度だが，and/minの場合は0.4とする
    return jax.lax.cond(jnp.greater(iou, border_iou),
            lambda: jnp.array([y1 * patch_len + x1 + use_cls, y2 * patch_len + x2 + use_cls]),
            lambda: jnp.array([-1, -1]))
  patch_len = image_size // patch_size
  patch_correspondence = jax.lax.map(_window_conv, jnp.arange(patch_len ** 4 + use_cls))
  return patch_correspondence[jnp.argsort(-patch_correspondence[:, 0])][:return_num]

@jax.jit
def _generate_patch_correspondences(windows):
  return jax.lax.stop_gradient(jax.lax.map(window_to_patch_correspondence2, windows))

def generate_patch_correspondences(windows):
  _cpu = jax.devices("cpu")[0]
  with jax.default_device(_cpu):
    return _generate_patch_correspondences(jax.device_put(windows, _cpu))
"""

def get_patch_correspondences_shape(R_shape, image_size=384, patch_size=32, use_cls=1, cls_correspondence=False):
  return (R_shape[0], + (image_size // patch_size) ** 2 + (1 if cls_correspondence else 0), 2)

#@jax.jit
@partial(jax.jit, static_argnums=(1,2,3,4,5))
def generate_patch_correspondences(l_R, image_size=384, patch_size=32, use_cls=1, border_iou=0.6, cls_correspondence=False):
  def _window_to_patch_correspondence(R):
    l = image_size // patch_size
    image1_X = jnp.array([[[i // l * patch_size + patch_size // 2], [i % l * patch_size + patch_size // 2], [1]] for i in range(l * l)])
    tmp = jnp.matmul(R, image1_X)
    image2_X_patch = (tmp[:,:2,0] / jnp.stack([tmp[:,2,0], tmp[:,2,0]], axis=1)).reshape((-1, 2)).T / patch_size
    image2_grid = jnp.arange(l*l).reshape((l, l)) + use_cls
    image2_patchnum = jax.scipy.ndimage.map_coordinates(image2_grid, image2_X_patch - 0.5, order=0, mode='constant', cval=-1)
    image1_patchnum = jnp.where(image2_patchnum == -1,
                                -jnp.ones(image2_patchnum.shape, dtype=jnp.int32),
                                jnp.arange(l*l) + use_cls)
      
    # TODO: CHECK IoU + 4角
    res = jnp.stack([image1_patchnum, image2_patchnum]).T

    if use_cls == 1 and cls_correspondence:
      res = jnp.concatenate([res, jnp.array([[0, 0]])], 0)
    return res
  return jax.lax.stop_gradient(jax.lax.map(_window_to_patch_correspondence, l_R))
