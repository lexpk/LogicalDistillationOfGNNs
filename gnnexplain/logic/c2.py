from abc import ABC, abstractmethod
from enum import Enum
from networkx import Graph


class Var(Enum):
    x = 0
    y = 1


class Formula(ABC):
    @abstractmethod
    def is_atomic(self) -> bool:
        pass

    @abstractmethod
    def free_variables(self) -> set[Var]:
        pass

    @abstractmethod
    def subformulas(self) -> set["Formula"]:
        pass

    @abstractmethod
    def is_gc2(self) -> bool:
        pass

    def evaluate(self, graph: Graph, x: int = None, y: int = None) -> bool:
        if not hasattr(graph, "evaluations"):
            graph.evaluations = {}
        if x is None and y is None:
            return Forall(
                Var.x,
                Forall(
                    Var.y,
                    self
                )
            )._evaluate(graph, x, y)
        elif x is None:
            return Forall(Var.x, self)._evaluate(graph, x, y)
        elif y is None:
            return Forall(Var.y, self)._evaluate(graph, x, y)
        else:
            return self._evaluate(graph, x, y)

    def __lt__(self, other) -> bool:
        return hash(self) < hash(other)

    @abstractmethod
    def _evaluate(self, graph: Graph, x, y) -> bool:
        pass

    @abstractmethod
    def simplify(self) -> "Formula":
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def complexity(
        self,
        guarded_weight: float = 1,
        unguarded_weight: float = 1
    ) -> float:
        pass


class UnaryConnective(Formula, ABC):
    def __init__(self, child: Formula) -> None:
        self.child = child
        self.hash = hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self.hash != other.hash:
                return False
            return self.child == other.child

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.child.free_variables()

    def is_gc2(self) -> bool:
        return self.child.is_gc2()

    def subformulas(self) -> set[Formula]:
        return self.child.subformulas() | {self}

    def __hash__(self) -> int:
        return self.hash

    def complexity(
        self,
        guarded_weight: float = 1,
        unguarded_weight: float = 1
    ) -> float:
        return self.child.complexity(guarded_weight, unguarded_weight)


class BinaryConnective(Formula, ABC):
    def __init__(self, left: Formula, right: Formula) -> None:
        self.left = left
        self.right = right
        self.hash = hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self.hash != other.hash:
                return False
            if self.left != other.left:
                return False
            return self.right == other.right

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.left.free_variables() | self.right.free_variables()

    def is_gc2(self) -> bool:
        return self.left.is_gc2() and self.right.is_gc2()

    def subformulas(self) -> set[Formula]:
        return self.left.subformulas() | self.right.subformulas() | {self}

    def __hash__(self) -> int:
        return self.hash

    def complexity(
        self,
        guarded_weight: float = 1,
        unguarded_weight: float = 1
    ) -> float:
        return self.left.complexity(
            guarded_weight, unguarded_weight
        ) + self.right.complexity(guarded_weight, unguarded_weight)


class AssociativeConnective(Formula, ABC):
    def __init__(self, *children: Formula) -> None:
        self.children = children
        self.hash = hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self.hash != other.hash:
                return False
            return self.children == other.children

    def is_atomic(self) -> bool:
        return bool(self.children)

    def free_variables(self) -> set[Var]:
        if self.children:
            return set.union(*[c.free_variables() for c in self.children])
        else:
            return set()

    def subformulas(self) -> set[Formula]:
        if self.children:
            return set.union(
                *[c.subformulas() for c in self.children]
            ) | {self}
        else:
            return {self}

    def is_gc2(self) -> bool:
        return all([c.is_gc2() for c in self.children])

    def __hash__(self) -> int:
        return self.hash

    def complexity(
        self,
        guarded_weight: float = 1,
        unguarded_weight: float = 1
    ) -> float:
        return max(1, sum(
            c.complexity(guarded_weight, unguarded_weight)
            for c in self.children
        ))


class QunatifiedFormula(Formula, ABC):
    def __init__(
        self,
        count: int,
        var: Var,
        formula: Formula,
        outgoing=True
    ) -> None:
        self.count = count
        self.var = var
        self.formula = formula
        self.outgoing = outgoing
        self.hash = hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            if self.hash != other.hash:
                return False
            if self.var != other.var:
                return False
            return self.formula == other.formula

    def is_atomic(self) -> bool:
        return False

    def free_variables(self) -> set[Var]:
        return self.formula.free_variables() - {self.var}

    def _formula_free_variables(self) -> set[Var]:
        if not hasattr(self, "_formula_free_variables_cache"):
            self._formula_free_variables_cache = self.formula.free_variables()
        return self._formula_free_variables_cache

    def is_gc2(self) -> bool:
        return self.formula.is_gc2()

    def subformulas(self) -> set[Formula]:
        return self.formula.subformulas() | {self}

    def simplify(self) -> Formula:
        simplified_formula = self.formula.simplify()
        if self.count is None:
            simplified = type(self)(self.var, simplified_formula)
        else:
            simplified = type(self)(self.count, self.var, simplified_formula)
        if str(simplified) == "⊤":
            return Top()
        if str(simplified) == "⊥":
            return Bottom()
        return simplified

    def __hash__(self) -> int:
        return self.hash

    def complexity(
        self,
        guarded_weight: float = 1,
        unguarded_weight: float = 1
    ) -> float:
        if isinstance(self, GuardedExistsGeq) or \
           isinstance(self, GuardedExistsLeq) or \
           isinstance(self, GuardedExistsEq) or \
           isinstance(self, GuardedExistsNeq):
            return guarded_weight * self.formula.complexity(
                guarded_weight, unguarded_weight
            )
        else:
            return unguarded_weight * self.formula.complexity(
                guarded_weight, unguarded_weight
            )


class And(AssociativeConnective):
    def __init__(self, *conjunts: Formula) -> None:
        super().__init__(*conjunts)

    def _evaluate(self, graph: Graph, x, y) -> bool:
        return all([c._evaluate(graph, x, y) for c in self.children])

    def simplify(self) -> Formula:
        simplified_conjunts = sorted(c.simplify() for c in self.children)
        if any([c == Bottom() for c in simplified_conjunts]):
            return Bottom()
        if len(simplified_conjunts) == 1:
            return simplified_conjunts[0]
        return And(*[c for c in simplified_conjunts if c != Top()])

    def __repr__(self) -> str:
        if self.children:
            return f"({' ∧ '.join([str(c) for c in self.children])})"
        else:
            return "⊤"


class Or(AssociativeConnective):
    def __init__(self, *disjuncts: Formula) -> None:
        super().__init__(*disjuncts)

    def _evaluate(self, graph: Graph, x, y):
        for c in self.children:
            if c._evaluate(graph, x, y):
                return True
        else:
            return False

    def simplify(self) -> Formula:
        simplified_disjuncts = sorted(c.simplify() for c in self.children)
        if any([c == Top() for c in simplified_disjuncts]):
            return Top()
        if len(simplified_disjuncts) == 1:
            return simplified_disjuncts[0]
        return Or(*[c for c in simplified_disjuncts if c != Bottom()])

    def __repr__(self) -> str:
        if self.children:
            return f"({' ∨ '.join([str(c) for c in self.children])})"
        else:
            return "⊥"


def Top():
    return And()


def Bottom():
    return Or()


class Not(UnaryConnective):
    def __init__(self, formula: Formula) -> None:
        super().__init__(formula)

    def _evaluate(self, graph: Graph, x, y):
        return not self.child._evaluate(graph, x, y)

    def simplify(self) -> Formula:
        if isinstance(self.child, Not):
            return self.child.child.simplify()
        if isinstance(self.child, And):
            return Or(*[Not(c) for c in self.child.children]).simplify()
        if isinstance(self.child, Or):
            return And(*[Not(c) for c in self.child.children]).simplify()
        if isinstance(self.child, Implies):
            return And(self.child.left.simplify(),
                       Not(self.child.right)).simplify()
        if isinstance(self.child, ExistsGeq):
            return ExistsLeq(self.child.count - 1, self.child.var,
                             self.child.formula).simplify()
        if isinstance(self.child, ExistsLeq):
            return ExistsGeq(self.child.count + 1, self.child.var,
                             self.child.formula).simplify()
        if isinstance(self.child, ExistsEq):
            return ExistsNeq(self.child.count, self.child.var,
                             self.child.formula).simplify()
        if isinstance(self.child, ExistsNeq):
            return ExistsEq(self.child.count, self.child.var,
                            self.child.formula).simplify()
        if isinstance(self.child, GuardedExistsGeq):
            return GuardedExistsLeq(self.child.count - 1, self.child.var,
                                    self.child.formula).simplify()
        if isinstance(self.child, GuardedExistsLeq):
            return GuardedExistsGeq(self.child.count + 1, self.child.var,
                                    self.child.formula).simplify()
        if isinstance(self.child, GuardedExistsEq):
            return GuardedExistsNeq(self.child.count, self.child.var,
                                    self.child.formula).simplify()
        if isinstance(self.child, GuardedExistsNeq):
            return GuardedExistsEq(self.child.count, self.child.var,
                                   self.child.formula).simplify()
        if isinstance(self.child, Forall):
            return Exists(self.child.var, Not(self.child.formula)).simplify()
        if isinstance(self.child, Exists):
            return Forall(self.child.var, Not(self.child.formula)).simplify()
        if isinstance(self.child, GuardedForall):
            return GuardedExists(self.child.var,
                                 Not(self.child.formula)).simplify()
        if isinstance(self.child, GuardedExists):
            return GuardedForall(self.child.var,
                                 Not(self.child.formula)).simplify()
        if self.child == Top():
            return Bottom()
        if self.child == Bottom():
            return Top()
        return self

    def __repr__(self) -> str:
        return f"¬{self.child}"


class Implies(BinaryConnective):
    def __init__(self, left: Formula, right: Formula) -> None:
        super().__init__(left, right)

    def _evaluate(self, graph: Graph, x, y):
        return not self.left._evaluate(graph, x, y) \
            or self.right._evaluate(graph, x, y)

    def simplify(self) -> Formula:
        simplified_left = self.left.simplify()
        simplified_right = self.right.simplify()
        if simplified_left == Bottom():
            return Top()
        if simplified_right == Top():
            return Top()
        if simplified_left == Top():
            return simplified_right
        if self.right == Bottom():
            return Not(simplified_left).simplify()

    def __repr__(self) -> str:
        return f"({self.left} → {self.right})"


class ExistsGeq(QunatifiedFormula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        super().__init__(count, var, formula)

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            count = 0
            for x in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, y] = True
                    return True
            else:
                graph.evaluations[self, y] = False
                return False
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            count = 0
            for y in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, x] = True
                    return True
            else:
                graph.evaluations[self, x] = False
                return False

    def __repr__(self) -> str:
        if hasattr(self, "_repr_"):
            return self._repr_
        elif self.formula == Top():
            self._repr_ = f"∃≥{self.count}{self.var.name}"
        elif self.formula == Bottom():
            self._repr_ = "⊥"
        else:
            self._repr_ = f"∃≥{self.count}{self.var.name}.{self.formula}"
        return self._repr_


class ExistsLeq(QunatifiedFormula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        super().__init__(count, var, formula)

    def _evaluate(self, graph: Graph, x, y):
        return not ExistsGeq(
            self.count + 1, self.var, self.formula
        )._evaluate(graph, x, y)

    def __repr__(self) -> str:
        if self.formula == Top():
            return f"∃≤{self.count}{self.var.name}"
        elif self.formula == Bottom():
            return "⊤"
        else:
            return f"∃≤{self.count}{self.var.name}.{self.formula}"


class ExistsEq(QunatifiedFormula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        super().__init__(count, var, formula)

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            if Var.x not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            count = 0
            for x in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, y] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, y] = False
                    return False
                graph.evaluations[self, y] = True
                return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            if Var.y not in self._formula_free_variables():
                return self.formula._evaluate(graph, x, y)
            count = 0
            for y in graph.nodes():
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, x] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, x] = False
                    return False
                graph.evaluations[self, x] = True
                return True

    def __repr__(self) -> str:
        if self.formula == Top():
            return f"∃={self.count}{self.var.name}"
        elif self.formula == Bottom():
            if self.count == 0:
                return "⊤"
            else:
                return f"∃≤{self.count - 1}{self.var.name}"
        else:
            return f"∃={self.count}{self.var.name}.{self.formula}"


class ExistsNeq(QunatifiedFormula):
    def __init__(self, count: int, var: Var, formula: Formula) -> None:
        super().__init__(count, var, formula)

    def _evaluate(self, graph: Graph, x, y):
        return not ExistsEq(
            self.count, self.var, self.formula
        )._evaluate(graph, x, y)

    def __repr__(self) -> str:
        if self.formula == Top():
            return f"∃≠{self.count}{self.var.name}"
        if self.formula == Bottom():
            if self.count == 0:
                return "⊥"
            return "⊤"
        return f"∃≠{self.count}{self.var.name}.{self.formula}"


class GuardedExistsGeq(QunatifiedFormula):
    def __init__(
        self,
        count: int,
        var: Var,
        formula: Formula,
        outgoing=True
    ) -> None:
        super().__init__(count, var, formula, outgoing)

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.neighbors(y):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, y] = True
                    return True
            else:
                graph.evaluations[self, y] = False
                return False
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.neighbors(x):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count >= self.count:
                    graph.evaluations[self, x] = True
                    return True
            else:
                graph.evaluations[self, x] = False
                return False

    def __repr__(self) -> str:
        v1, v2 = ('x', 'y') if (self.var == Var.y) == self.outgoing \
            else ('y', 'x')
        if self.formula == Top():
            return f"∃≥{self.count}{self.var.name}.E({v1}, {v2})"
        elif self.formula == Bottom():
            return "⊥"
        else:
            return f"∃≥{self.count}{self.var.name}." + \
                f"E({v1}, {v2}) ∧ {self.formula})"


class GuardedExistsLeq(QunatifiedFormula):
    def __init__(
        self,
        count: int,
        var: Var,
        formula: Formula,
        outgoing=True
    ) -> None:
        super().__init__(count, var, formula, outgoing)

    def _evaluate(self, graph: Graph, x, y):
        return not GuardedExistsGeq(
            self.count + 1, self.var, self.formula
        )._evaluate(graph, x, y)

    def __repr__(self) -> str:
        v1, v2 = ('x', 'y') if (self.var == Var.y) == self.outgoing \
            else ('y', 'x')
        if self.formula == Top():
            return f"∃≤{self.count}{self.var.name}.E({v1}, {v2})"
        elif self.formula == Bottom():
            return "⊤"
        else:
            return f"∃≤{self.count}{self.var.name}." + \
                    f"(E({v1}, {v2}) ∧ {self.formula})"


class GuardedExistsEq(QunatifiedFormula):
    def __init__(
        self,
        count: int,
        var: Var,
        formula: Formula,
        outgoing=True
    ) -> None:
        super().__init__(count, var, formula, outgoing)

    def _evaluate(self, graph: Graph, x, y):
        if self.var == Var.x:
            if (self, y) in graph.evaluations:
                return graph.evaluations[self, y]
            count = 0
            for x in graph.neighbors(y):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, y] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, y] = False
                    return False
                graph.evaluations[self, y] = True
                return True
        else:
            if (self, x) in graph.evaluations:
                return graph.evaluations[self, x]
            count = 0
            for y in graph.neighbors(x):
                if self.formula._evaluate(graph, x, y):
                    count += 1
                if count > self.count:
                    graph.evaluations[self, x] = False
                    return False
            else:
                if count < self.count:
                    graph.evaluations[self, x] = False
                    return False
                graph.evaluations[self, x] = True
                return True

    def __repr__(self) -> str:
        v1, v2 = ('x', 'y') if (self.var == Var.y) == self.outgoing \
            else ('y', 'x')
        if self.formula == Top():
            return f"∃={self.count}{self.var.name}.E({v1}, {v2})"
        elif self.formula == Bottom():
            if self.count == 0:
                return "⊤"
            else:
                return "⊥"
        else:
            return f"∃={self.count}{self.var.name}." + \
                f"(E({v1}, {v2}) ∧ {self.formula})"


class GuardedExistsNeq(QunatifiedFormula):
    def __init__(
        self,
        count: int,
        var: Var,
        formula: Formula,
        outgoing=True
    ) -> None:
        super().__init__(count, var, formula, outgoing)

    def _evaluate(self, graph: Graph, x, y):
        return not GuardedExistsEq(
            self.count, self.var, self.formula
        )._evaluate(graph, x, y)

    def __repr__(self) -> str:
        v1, v2 = ('x', 'y') if (self.var == Var.y) == self.outgoing \
            else ('y', 'x')
        if self.formula == Top():
            return f"∃≠{self.count}{self.var.name}.E({v1}, {v2})"
        if self.formula == Bottom():
            if self.count == 0:
                return "⊥"
            else:
                return "⊤"
        else:
            return f"∃≠{self.count}{self.var.name}." + \
                f"(E({v1}, {v2}) ∧ {self.formula})"


class Exists(QunatifiedFormula):
    def __init__(self, var: Var, formula: Formula) -> None:
        super().__init__(None, var, formula)

    def _evaluate(self, graph: Graph, x, y) -> bool:
        return ExistsGeq(1, self.var, self.formula)._evaluate(graph, x, y)

    def __repr__(self) -> str:
        if self.formula == Top():
            return "⊤"
        elif self.formula == Bottom():
            return "⊥"
        else:
            return f"∃{self.var.name}.{self.formula}"


class GuardedExists(QunatifiedFormula):
    def __init__(self, var: Var, formula: Formula, outgoing=True) -> None:
        super().__init__(None, var, formula, outgoing)

    def _evaluate(self, graph: Graph, x, y) -> bool:
        return GuardedExistsGeq(
            1, self.var, self.formula
        )._evaluate(graph, x, y)

    def __repr__(self) -> str:
        v1, v2 = ('x', 'y') if (self.var == Var.y) == self.outgoing \
            else ('y', 'x')
        if self.formula == Top():
            return "∃{self.var.name}.E({v1}, {v2})"
        if self.formula == Bottom():
            return "⊥"
        else:
            return f"∃{self.var.name}." + \
                f"(E({v1}, {v2}) ∧ {self.formula})"


class Forall(QunatifiedFormula):
    def __init__(self, var: Var, formula: Formula) -> None:
        super().__init__(None, var, formula)

    def _evaluate(self, graph: Graph, x, y) -> bool:
        return Not(
            ExistsGeq(1, self.var, Not(self.formula)))._evaluate(graph, x, y)

    def __repr__(self) -> str:
        if self.formula == Top():
            return "⊤"
        elif self.formula == Bottom():
            return "⊥"
        else:
            return f"∀{self.var.name}.{self.formula}"


class GuardedForall(QunatifiedFormula):
    def __init__(self, var: Var, formula: Formula, outgoing=True) -> None:
        super().__init__(None, var, formula, outgoing)

    def _evaluate(self, graph: Graph, x, y) -> bool:
        return not GuardedExistsGeq(
            1, Var.x, Not(self.formula)
        )._evaluate(graph, x, y)

    def __repr__(self) -> str:
        v1, v2 = ('x', 'y') if (self.var == Var.y) == self.outgoing \
            else ('y', 'x')
        if self.formula == Top():
            return "⊤"
        elif self.formula == Bottom():
            return f"∀{self.var.name}.¬E({v1}, {v2})"
        return f"∀{self.var.name}." + \
            f"(E({v1}, {v2}) → {self.formula})"


class E(Formula):
    def __init__(self, v: Var, w: Var) -> None:
        self.v = v
        self.w = w
        self.hash = hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, E):
            return self.v == other.v and self.w == other.w

    def is_atomic(self) -> bool:
        return True

    def free_variables(self) -> set[Var]:
        return {self.v, self.w}

    def subformulas(self) -> set[Formula]:
        return {self}

    def is_gc2(self) -> bool:
        return False

    def _evaluate(self, graph: Graph, x, y):
        if (x, y) in graph.edges():
            return True
        if (y, x) in graph.edges():
            return True
        return False

    def simplify(self) -> Formula:
        return self

    def __repr__(self) -> str:
        return f"E({self.v.name}, {self.w.name})"

    def __hash__(self) -> int:
        return self.hash

    def complexity(
        self,
        guarded_weight: float = 1,
        unguarded_weight: float = 1
    ) -> float:
        return 1


class Atom(Formula):
    def __init__(self, name: str, v: Var) -> None:
        self.name = name
        self.v = v
        self.hash = hash(str(self))

    def is_atomic(self) -> bool:
        return True

    def free_variables(self) -> set[Var]:
        return {self.v}

    def subformulas(self) -> set[Formula]:
        return {self}

    def is_gc2(self) -> bool:
        return True

    def _evaluate(self, graph: Graph, x, y):
        if self.v == Var.x:
            return bool(graph.nodes[x][self.name])
        else:
            return bool(graph.nodes[y][self.name])

    def simplify(self) -> Formula:
        return self

    def __repr__(self) -> str:
        return f"{self.name}({self.v.name})"

    def __hash__(self) -> int:
        return self.hash

    def complexity(
        self,
        guarded_weight: float = 1,
        unguarded_weight: float = 1
    ) -> float:
        return 1