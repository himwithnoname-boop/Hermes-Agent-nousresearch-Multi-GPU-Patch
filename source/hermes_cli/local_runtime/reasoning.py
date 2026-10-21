"""Reasoning controls proven by a selected managed GGUF's literal chat-template guards."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from types import MappingProxyType

from agent.reasoning_effort import EFFORT_LADDER


@dataclass(frozen=True)
class ReasoningCapabilities:
    efforts: tuple[str, ...]
    default_effort: str | None
    aliases: Mapping[str, str]
    supports_disable: bool


_UNKNOWN = ReasoningCapabilities((), None, MappingProxyType({}), False)
_CACHE_LIMIT = 64
_TEMPLATE_LIMIT = 256 * 1024
_cache: OrderedDict[tuple[str, int, int], ReasoningCapabilities | None] = OrderedDict()
_cache_lock = threading.RLock()


def _name(node, value: str) -> bool:
    from jinja2 import nodes

    return isinstance(node, nodes.Name) and node.name == value


def _test(node, name: str) -> bool:
    from jinja2 import nodes

    return (isinstance(node, nodes.Test) and _name(node.node, "enable_thinking")
            and node.name == name and not node.args and not node.kwargs)


def _thinking_condition(node, *, enabled: bool) -> bool:
    from jinja2 import nodes

    kind, tests = (nodes.Or, ("undefined", "true")) if enabled else (nodes.And, ("defined", "false"))
    return (isinstance(node, kind)
            and ((_test(node.left, tests[0]) and _test(node.right, tests[1]))
                 or (_test(node.right, tests[0]) and _test(node.left, tests[1]))))


def _comparison(node, variable: str, operator: str):
    from jinja2 import nodes

    if (isinstance(node, nodes.Compare) and _name(node.expr, variable)
            and len(node.ops) == 1 and node.ops[0].op == operator):
        return node.ops[0].expr
    return None


def _nonblank(body):
    from jinja2 import nodes

    def blank(value):
        if isinstance(value, nodes.TemplateData):
            return not value.data.strip()
        return isinstance(value, nodes.Const) and isinstance(value.value, str) and not value.value.strip()

    return [statement for statement in body
            if not (isinstance(statement, nodes.Output) and all(blank(value) for value in statement.nodes))]


def _raises(body) -> bool:
    from jinja2 import nodes

    body = _nonblank(body)
    outputs = [part for statement in body if isinstance(statement, nodes.Output)
               for part in statement.nodes
               if not (isinstance(part, nodes.TemplateData) and not part.data.strip())]
    if len(body) != 1 or not isinstance(body[0], nodes.Output) or len(outputs) != 1:
        return False
    call = outputs[0]
    return isinstance(call, nodes.Call) and _name(call.node, "raise_exception")


def _default_assignment(statement):
    from jinja2 import nodes

    if not isinstance(statement, nodes.Assign) or not isinstance(statement.target, nodes.Name):
        return None
    value = statement.node
    if (isinstance(value, nodes.Filter) and _name(value.node, "reasoning_effort")
            and value.name == "default" and len(value.args) == 1
            and isinstance(value.args[0], nodes.Const) and isinstance(value.args[0].value, str)
            and not value.kwargs):
        return statement.target.name, value.args[0].value
    return None


def _literal_efforts(guard, variable: str):
    from jinja2 import nodes

    values = _comparison(guard.test, variable, "notin") if isinstance(guard, nodes.If) else None
    if not isinstance(values, (nodes.Tuple, nodes.List)) or not _raises(guard.body):
        return None
    if guard.elif_ or guard.else_ or not values.items:
        return None
    if not all(isinstance(item, nodes.Const) and item.value in EFFORT_LADDER[1:] for item in values.items):
        return None
    levels = tuple(item.value for item in values.items)
    return levels if len(set(levels)) == len(levels) else None


def _alias(statement, variable: str):
    from jinja2 import nodes

    if not isinstance(statement, nodes.If) or statement.elif_ or statement.else_:
        return None
    body = _nonblank(statement.body)
    if len(body) != 1:
        return None
    source = _comparison(statement.test, variable, "eq")
    target = body[0]
    if (isinstance(source, nodes.Const) and source.value in EFFORT_LADDER[1:]
            and isinstance(target, nodes.Assign) and _name(target.target, variable)
            and isinstance(target.node, nodes.Const) and target.node.value in EFFORT_LADDER[1:]):
        return source.value, target.node.value, target
    return None


def _guard_capabilities(body):
    """Accept only a direct default -> optional literal aliases -> rejecting literal allow-list."""
    from jinja2 import nodes

    body = _nonblank(body)
    for index, statement in enumerate(body):
        default = _default_assignment(statement)
        if default is None:
            continue
        variable, default_effort = default
        aliases = {}
        for following in body[index + 1:]:
            pair = _alias(following, variable)
            if pair is not None:
                if pair[0] in aliases:
                    break
                aliases[pair[0]] = pair[1]
                continue
            levels = _literal_efforts(following, variable)
            if levels is None:
                break
            if any(value not in levels or key in levels for key, value in aliases.items()):
                break
            resolved_default = aliases.get(default_effort, default_effort)
            if resolved_default not in levels:
                break
            # A later reassignment could make the apparent allow-list irrelevant.
            assignments = [assign for part in body
                           for assign in ([part] if isinstance(part, nodes.Assign) else []) + list(part.find_all(nodes.Assign))
                           if _name(assign.target, variable)]
            expected = [statement] + [pair[2] for part in body[index + 1:]
                                      if (pair := _alias(part, variable)) is not None]
            if assignments != expected:
                break
            return tuple(level for level in EFFORT_LADDER if level in levels), resolved_default, aliases
    return None


def _conditional_capabilities(tree):
    """GLM's literal conditional allow-list falls back to a literal default."""
    from jinja2 import nodes

    found = []
    for statement in tree.body:
        if not isinstance(statement, nodes.Assign) or not isinstance(statement.target, nodes.Name):
            continue
        value = statement.node
        if not (isinstance(value, nodes.CondExpr) and _name(value.expr1, "reasoning_effort")
                and isinstance(value.expr2, nodes.Const) and value.expr2.value in EFFORT_LADDER[1:]):
            continue
        test = value.test
        if not (isinstance(test, nodes.And) and isinstance(test.left, nodes.Test)
                and _name(test.left.node, "reasoning_effort") and test.left.name == "defined"
                and not test.left.args and not test.left.kwargs):
            continue
        levels = _comparison(test.right, "reasoning_effort", "in")
        if not (isinstance(levels, (nodes.List, nodes.Tuple)) and levels.items
                and all(isinstance(item, nodes.Const) and item.value in EFFORT_LADDER[1:]
                        for item in levels.items)):
            continue
        assignments = [part for part in tree.find_all(nodes.Assign)
                       if _name(part.target, statement.target.name)]
        if assignments != [statement]:
            continue
        vocabulary = {item.value for item in levels.items} | {value.expr2.value}
        found.append((tuple(level for level in EFFORT_LADDER if level in vocabulary),
                      value.expr2.value, {}))
    return found[0] if len(found) == 1 else None


def _generation_ifs(body):
    """The prompt tail is active only under the known generation-prompt flag, not arbitrary guards."""
    from jinja2 import nodes

    for statement in body:
        if isinstance(statement, nodes.If):
            yield statement
            if _name(statement.test, "add_generation_prompt"):
                yield from _generation_ifs(statement.body)


def _can_disable(tree, guarded_block) -> bool:
    from jinja2 import nodes

    if guarded_block is None or not _thinking_condition(guarded_block.test, enabled=True):
        return False
    for branch in _generation_ifs(tree.body):
        if not _thinking_condition(branch.test, enabled=False):
            continue
        if len(branch.body) != 1 or not isinstance(branch.body[0], nodes.Output):
            continue
        values = branch.body[0].nodes
        if (all(isinstance(value, (nodes.Const, nodes.TemplateData)) for value in values)
                and re.fullmatch(r"\s*<think>\s*</think>\s*", "".join(
                    str(value.value if isinstance(value, nodes.Const) else value.data) for value in values))):
            return True
    return False


def capabilities_from_template(template: str) -> ReasoningCapabilities | None:
    """Parse syntax only; never render a model-supplied Jinja template or execute its calls."""
    if not isinstance(template, str) or not template or len(template) > _TEMPLATE_LIMIT:
        return None
    try:
        from jinja2 import Environment, TemplateSyntaxError, nodes
    except ImportError:
        return None
    try:
        tree = Environment(extensions=["jinja2.ext.loopcontrols"]).parse(template)
        candidates = []
        root = _guard_capabilities(tree.body)
        if root is not None:
            candidates.append((root, None))
        conditional = _conditional_capabilities(tree)
        if conditional is not None:
            candidates.append((conditional, None))
        for branch in tree.body:
            if isinstance(branch, nodes.If) and _thinking_condition(branch.test, enabled=True):
                found = _guard_capabilities(branch.body)
                if found is not None:
                    candidates.append((found, branch))
        if len(candidates) != 1:
            return None
        (efforts, default, aliases), guarded = candidates[0]
        return ReasoningCapabilities(efforts, default, MappingProxyType(aliases), _can_disable(tree, guarded))
    except (TemplateSyntaxError, RecursionError, TypeError, ValueError):
        return None


def model_reasoning_capabilities(model_id: str | None) -> ReasoningCapabilities | None:
    """Selected staged GGUF evidence; empty efforts mean unknown vocabulary, None means no model.

    Read only its header, without starting a server or inference. Unknown local templates
    must use the model default rather than inheriting an unsupported generic wire effort.
    """
    from hermes_cli.local_runtime.bootstrap import staged_models
    from hermes_cli.local_runtime.gguf import _read_part, model_id_from_stem

    path = None
    try:
        path = next((Path(path) for path in staged_models()
                     if model_id_from_stem(Path(path).stem) == str(model_id or "").strip()), None)
        if path is None:
            return None
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        with _cache_lock:
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]
            value = capabilities_from_template(_read_part(path).metadata.get("tokenizer.chat_template")) or _UNKNOWN
            _cache[key] = value
            while len(_cache) > _CACHE_LIMIT:
                _cache.popitem(last=False)
            return value
    except (OSError, ValueError, KeyError, RuntimeError):
        return _UNKNOWN if path is not None else None


def managed_reasoning_capabilities(model_id: str | None, base_url: str | None) -> ReasoningCapabilities | None:
    """Wire controls apply only to the ownership-checked managed endpoint and its selected model."""
    from hermes_cli.local_runtime.endpoint import _state_endpoint

    try:
        managed = _state_endpoint()
        if (not managed or str(base_url or "").strip().rstrip("/")
                != str(managed.get("base_url") or "").strip().rstrip("/")):
            return None
        return model_reasoning_capabilities(model_id)
    except (OSError, ValueError, RuntimeError):
        return None
