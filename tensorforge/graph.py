"""Graph container + traversal utilities shared by every pass."""
from __future__ import annotations

from .ir import Node
from .tensor import Tensor


class Graph:
    def __init__(self, outputs, inputs):
        self.outputs = [o.node if isinstance(o, Tensor) else o for o in outputs]
        self.inputs = [i.node if isinstance(i, Tensor) else i for i in inputs]

    # --- traversal ---------------------------------------------------------
    def topo(self):
        """Return nodes reachable from outputs in dependency order."""
        seen, order = set(), []

        def visit(n: Node):
            if n.id in seen:
                return
            seen.add(n.id)
            for inp in n.inputs:
                visit(inp)
            order.append(n)

        for o in self.outputs:
            visit(o)
        return order

    def users(self):
        """Map node-id -> list of nodes that consume it (the use-def graph)."""
        users: dict[int, list[Node]] = {}
        for n in self.topo():
            for inp in n.inputs:
                users.setdefault(inp.id, []).append(n)
        return users

    def replace_all_uses(self, old: Node, new: Node):
        """Rewire every consumer of ``old`` to read ``new`` instead."""
        if old is new:
            return
        for n in self.topo():
            if old in n.inputs:
                n.inputs = tuple(new if i is old else i for i in n.inputs)
        self.outputs = [new if o is old else o for o in self.outputs]

    # --- introspection -----------------------------------------------------
    def num_ops(self):
        return sum(1 for n in self.topo()
                   if n.op not in ("const", "placeholder"))

    def pretty(self) -> str:
        lines = []
        for n in self.topo():
            if n.op in ("const", "placeholder"):
                lines.append(f"  {n.name}: {n.op} {n.shape}")
            else:
                ins = ", ".join(i.name for i in n.inputs)
                extra = ""
                if n.attrs:
                    kv = ", ".join(f"{k}={v}" for k, v in n.attrs.items()
                                   if not isinstance(v, (list, dict)))
                    extra = f"  [{kv}]"
                lines.append(f"  {n.name} = {n.op}({ins}){extra}  : {n.shape}")
        outs = ", ".join(o.name for o in self.outputs)
        return "graph(\n" + "\n".join(lines) + f"\n) -> {outs}"
