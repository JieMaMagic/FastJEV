"""Candidate-prefix trie execution for Qwen3.5 hybrid backbones.

Unlike a flattened attention tree, each fork here carries a complete, private
continuation cache for all retained layers, including convolution and GDN
states. Unary paths are evaluated as segments. Children of a fork are batched
against the same immutable parent cache. Internal segments always retain final
states; the share/drop factor can discard states only at terminal leaves.

This is an inference-only reference executor. Extra launches and intermediate
caches are included in measured request cost; sharing does not promise speedup.
Existing experiment modules are not modified.
"""
import copy
from collections import defaultdict
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass

import torch

from mso import branch as BR


@dataclass
class PrefixTrie:
    tokens: list
    parents: list
    depths: list
    children: list
    paths: list
    terminals: set

    @classmethod
    def build(cls, rows):
        if not rows or any(not row for row in rows):
            raise ValueError("Nonempty candidate rows are required")
        tokens, parents, depths, children, paths, terminals = [], [], [], [[]], [], set()
        edges = {}
        for row in rows:
            parent, path = -1, []
            for depth, token in enumerate(row):
                # branch_forward gives every row p0 + depth on all rope axes.
                # Parent identity also fixes the complete preceding history.
                key = (parent, int(token), depth)
                if key not in edges:
                    node = len(tokens)
                    edges[key] = node
                    tokens.append(int(token)); parents.append(parent); depths.append(depth)
                    children.append([])
                    children[parent + 1].append(node)
                parent = edges[key]
                path.append(parent)
            paths.append(path)
            terminals.add(parent)
        return cls(tokens, parents, depths, children, paths, terminals)

    def segments(self, parent):
        result = []
        for child in self.children[parent + 1]:
            path = [child]
            while child not in self.terminals and len(self.children[child + 1]) == 1:
                child = self.children[child + 1][0]
                path.append(child)
            result.append(path)
        return result


def continuation_row(cache, row, batch):
    """Materialize an independent row of a *keep* segment's valid cache.

The executor is the sole caller and never exports leaf/drop caches. Remove the
finished StateFactorialContext's instrumentation from the new ordinary cache.
Retained layer tensors are copied; skipped layers may still have batch one.
"""
    def select(value):
        if torch.is_tensor(value):
            if value.ndim and value.shape[0] == batch:
                return value[row:row + 1].clone()
            return value.clone()
        if isinstance(value, dict):
            return {k: select(v) for k, v in value.items()}
        if isinstance(value, list):
            return [select(v) for v in value]
        if isinstance(value, tuple):
            return tuple(select(v) for v in value)
        return value
    new = copy.copy(cache)
    for name in list(vars(new)):
        if name.startswith('_omnijev_'):
            delattr(new, name)
    if 'update_recurrent_state' in vars(new):
        raise RuntimeError('A live state-update patch cannot become a continuation')
    new.layers = []
    for layer in cache.layers:
        target = copy.copy(layer)
        for name, value in vars(layer).items():
            setattr(target, name, select(value))
        new.layers.append(target)
    return new


class HybridTrieContext(AbstractContextManager):
    """Install one request's trie executor around the unchanged serving path."""
    def __init__(self, controller, *, shared=False, layers=(), kernel='recurrent', inspect=False):
        if kernel not in ('recurrent', 'original'):
            raise ValueError(kernel)
        if shared and kernel != 'recurrent':
            raise ValueError('Shared states require the controlled recurrent kernel')
        self.controller, self.shared, self.layers = controller, shared, tuple(layers)
        self.kernel, self.inspect = kernel, inspect
        self.stats = {'trie': True, 'shared': shared, 'layers': list(layers), 'kernel': kernel,
                      'original_tokens': 0, 'unique_tokens': 0, 'padded_tokens': 0,
                      'segment_calls': 0, 'internal_calls': 0, 'leaf_calls': 0,
                      'prefix_calls': 0, 'state': [], 'component': [], 'requests': 0}
        self.old = None

    def __enter__(self):
        if self.old is not None or getattr(BR.branch_forward, '_hybrid_trie', False):
            raise RuntimeError('Trie contexts cannot be reused or nested')
        if torch.is_grad_enabled() or self.controller.backbone.training:
            raise RuntimeError('Trie requires inference_mode and eval')
        self.old = BR.branch_forward
        def forward(*args, **kwargs):
            return self.forward(*args, **kwargs)
        forward._hybrid_trie = True
        BR.branch_forward = forward
        return self

    def __exit__(self, *args):
        BR.branch_forward = self.old
        return False

    def forward(self, backbone, penc, rows, dev, open_ids, close_ids, lm_head=None,
                prefix_pos=None, pos_start=None, groups=None, chunk=None, pad_id=0, mrope_axes=3):
        if backbone is not self.controller.backbone:
            raise RuntimeError('Trie patched an unexpected backbone')
        tree = PrefixTrie.build(rows)
        chunk = chunk or 64
        if chunk < 1:
            raise ValueError('Invalid batch size')
        length = penc['input_ids'].shape[1]
        kw = {**penc, 'attention_mask': torch.ones_like(penc['input_ids']),
              'use_cache': True, 'output_hidden_states': True, 'logits_to_keep': 1}
        if prefix_pos is not None:
            kw['position_ids'] = prefix_pos
        # No pruning or state patch on the common image/question prefix.
        prefix = backbone(**kw)
        cache0 = prefix.past_key_values
        p0 = int(pos_start if pos_start is not None else
                 (int(prefix_pos.max()) + 1 if prefix_pos is not None else length))
        hidden = prefix.hidden_states[-1].new_empty((len(tree.tokens), prefix.hidden_states[-1].shape[-1]))
        del prefix
        self.stats['requests'] += 1
        self.stats['prefix_calls'] += 1
        self.stats['original_tokens'] += sum(map(len, rows))
        self.stats['unique_tokens'] += len(tree.tokens)
        visited = set()

        def run_segments(parent_cache, segments, internal):
            for start in range(0, len(segments), chunk):
                batch_segments = segments[start:start + chunk]
                count = len(batch_segments)
                width = max(map(len, batch_segments))
                offset = tree.depths[batch_segments[0][0]]
                assert all(tree.depths[s[0]] == offset for s in batch_segments)
                ids = torch.full((count, width), pad_id, dtype=torch.long, device=dev)
                att = torch.zeros((count, width), dtype=torch.long, device=dev)
                for i, segment in enumerate(batch_segments):
                    ids[i, :len(segment)] = torch.tensor([tree.tokens[n] for n in segment], device=dev)
                    att[i, :len(segment)] = 1
                full_att = torch.cat((torch.ones((count, length + offset), dtype=torch.long, device=dev), att), 1)
                pos = torch.arange(p0 + offset, p0 + offset + width, device=dev)[None, None].expand(mrope_axes, count, width).contiguous()
                with ExitStack() as stack:
                    state = None
                    if self.kernel == 'recurrent':
                        from mso.terminal_state_factorial import StateFactorialContext
                        mode = 'recurrent_' + ('share' if self.shared else 'copy') + ('_keep' if internal or not self.shared else '_drop')
                        state = stack.enter_context(StateFactorialContext(mode, inspect=self.inspect, max_tokens=4096))
                    component = stack.enter_context(self.controller.context(self.layers, 'block', inspect=self.inspect))
                    cache = BR.expand_cache(parent_cache, count)
                    out = backbone(input_ids=ids, attention_mask=full_att, position_ids=pos,
                                   cache_position=torch.arange(length + offset, length + offset + width, device=dev),
                                   past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
                if self.inspect:
                    if state is not None:
                        self.stats['state'].append(state.stats)
                    self.stats['component'].append(component.stats)
                self.stats['segment_calls'] += 1
                self.stats['internal_calls' if internal else 'leaf_calls'] += 1
                self.stats['padded_tokens'] += count * width - sum(map(len, batch_segments))
                next_work = []
                for i, segment in enumerate(batch_segments):
                    assert not visited.intersection(segment)
                    visited.update(segment)
                    hidden[torch.tensor(segment, device=dev)] = out.hidden_states[-1][i, :len(segment)]
                    if internal:
                        next_work.append((segment[-1], continuation_row(out.past_key_values, i, count)))
                del out, cache
                for node, next_cache in next_work:
                    visit(node, next_cache)

        def visit(parent, cache):
            internal_by_length, leaves = defaultdict(list), []
            for segment in tree.segments(parent):
                if tree.children[segment[-1] + 1]:
                    internal_by_length[len(segment)].append(segment)
                else:
                    leaves.append(segment)
            # Only terminal segments may be padded or omit final GDN states.
            for segments in internal_by_length.values():
                run_segments(cache, segments, True)
            if leaves:
                run_segments(cache, leaves, False)

        visit(-1, cache0)
        assert len(visited) == len(tree.tokens)
        marks = [BR._marks(row, open_ids, close_ids) for row in rows]
        ends = torch.tensor([path[m[1]] for path, m in zip(tree.paths, marks)], device=dev)
        question_nodes = [path[max(0, m[0] - 1)] for path, m in zip(tree.paths, marks)]
        qends = torch.tensor(question_nodes, device=dev)
        u, zq = hidden[ends], hidden[qends]
        feats = None
        if lm_head is not None:
            predictor, targets, owners, ordinals, first_rows, first_tokens = [], [], [], [], [], []
            for i, (row, path, mark) in enumerate(zip(rows, tree.paths, marks)):
                inner = mark[2]
                for ordinal, t in enumerate(inner):
                    predictor.append(path[t - 1]); targets.append(row[t]); owners.append(i)
                    ordinals.append(ordinal)
                if inner:
                    first_rows.append(i); first_tokens.append(row[inner[0]])
            q_nodes = [question_nodes[i] for i in first_rows]
            needed = list(dict.fromkeys(predictor + q_nodes))
            sums = torch.zeros(len(rows), device=dev, dtype=torch.float32)
            counts = torch.tensor([len(m[2]) for m in marks], device=dev, dtype=torch.float32)
            first = torch.zeros_like(sums)
            token_scores = torch.zeros((len(rows), max((len(m[2]) for m in marks), default=0)),
                                       device=dev, dtype=torch.float32)
            # Bound full-vocabulary projection memory without altering its normalization.
            node_to_predictors, node_to_first = defaultdict(list), defaultdict(list)
            for j, node in enumerate(predictor): node_to_predictors[node].append(j)
            for j, node in enumerate(q_nodes): node_to_first[node].append(j)
            for start in range(0, len(needed), 128):
                nodes = needed[start:start + 128]
                lp = torch.log_softmax(lm_head(hidden[torch.tensor(nodes, device=dev)]).float(), -1)
                ri, ti, own, position, fr, ft, fo = [], [], [], [], [], [], []
                for r, node in enumerate(nodes):
                    for j in node_to_predictors[node]:
                        ri.append(r); ti.append(targets[j]); own.append(owners[j])
                        position.append(ordinals[j])
                    for j in node_to_first[node]:
                        fr.append(r); ft.append(first_tokens[j]); fo.append(first_rows[j])
                if ri:
                    # Each destination is unique. Atomic index_add would reorder
                    # additions between calls and break ties between identical
                    # candidates, even when every backbone state is bitwise equal.
                    token_scores[own, position] = lp[ri, ti]
                if fr:
                    first[fo] = lp[fr, ft]
            sums = token_scores.sum(dim=1)
            valid = counts > 0
            masked = torch.where(valid, sums, torch.full_like(sums, -1e4))
            dist = torch.zeros_like(sums)
            for group in (groups or [list(range(len(rows)))]):
                dist[group] = torch.log_softmax(masked[group], 0)
            feats = torch.stack((sums / 10, torch.where(valid, sums / counts.clamp(min=1), 0),
                                 counts / 10, dist, first, valid.float()), 1)
        return {'u': u, 'zq': zq, 'feats': feats}
