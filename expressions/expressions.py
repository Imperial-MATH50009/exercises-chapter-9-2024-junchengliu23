from functools import singledispatch


class Expression:
    def __init__(self, operands=()):
        self.operands = operands

    def __repr__(self):
        return f"{type(self).__name__}{repr(self.operands)}"

    def __str__(self):
        if not self.operands:
            return super().__str__()  # fallback if terminal doesn't override

        # Binary infix operator case (for Add, Mul, etc.)
        left, right = self.operands
        return f"{self._parenthesize(left)} {self.op_symbol} \
              {self._parenthesize(right)}"

    def _parenthesize(self, operand):
        if not isinstance(operand, Expression):
            return str(operand)
        if getattr(operand, "precedence", 100) < self.precedence:
            return f"({operand})"
        return str(operand)

    def __eq__(self, other):
        return (  # noqa: E721
            type(self) == type(other)
            and getattr(self, 'value', None) == getattr(other, 'value', None)
            and getattr(self, 'operands', None) == getattr(other, 'operands', None)  # noqa: E501
        )

    def __hash__(self):
        return hash((type(self), getattr(self, 'value', None), tuple(getattr(self, 'operands', ()))))  # noqa: E501

    def __add__(self, other):
        return Add(self, self._coerce(other))

    def __radd__(self, other):
        return Add(self._coerce(other), self)

    def __sub__(self, other):
        return Sub(self, self._coerce(other))

    def __rsub__(self, other):
        return Sub(self._coerce(other), self)

    def __mul__(self, other):
        return Mul(self, self._coerce(other))

    def __rmul__(self, other):
        return Mul(self._coerce(other), self)

    def __truediv__(self, other):
        return Div(self, self._coerce(other))

    def __rtruediv__(self, other):
        return Div(self._coerce(other), self)

    def __pow__(self, other):
        return Pow(self, self._coerce(other))

    def __rpow__(self, other):
        return Pow(self._coerce(other), self)

    def _coerce(self, value):
        if isinstance(value, Expression):
            return value
        else:
            return Number(value)


# Terminal classes
class Terminal(Expression):
    def __init__(self):
        super().__init__(operands=())


class Number(Expression):
    def __init__(self, value):
        super().__init__(())
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Number({self.value})"


class Symbol(Expression):
    def __init__(self, value):
        super().__init__(())
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"Symbol({repr(self.value)})"


# Operator base class
class Operator(Expression):
    def __init__(self, *operands):
        self.operands = tuple(
            op if isinstance(op, Expression) else Number(op)
            for op in operands
        )


# Concrete operator subclasses
class Add(Expression):
    precedence = 1
    op_symbol = "+"

    def __init__(self, left, right):
        super().__init__((left, right))


class Sub(Expression):
    precedence = 1
    op_symbol = "-"

    def __init__(self, left, right):
        super().__init__((left, right))


class Mul(Expression):
    precedence = 2
    op_symbol = "*"

    def __init__(self, left, right):
        super().__init__((left, right))


class Div(Expression):
    precedence = 2
    op_symbol = "/"

    def __init__(self, left, right):
        super().__init__((left, right))


class Pow(Expression):
    precedence = 3
    op_symbol = "^"

    def __init__(self, left, right):
        super().__init__((left, right))


@singledispatch
def differentiate(expr, *args, var=None, **kwargs):
    raise NotImplementedError(f"Cannot differentiate a {type(expr).__name__}")


@differentiate.register(Symbol)
def _(expr, *args, var=None, **kwargs):
    """Differentiate Symbol with respect to var."""
    # If the symbol matches the variable, return 1, otherwise return 0
    return Number(1.0) if expr.value == var else Number(0.0)

@differentiate.register(Number)
def _(expr, *args, var=None, **kwargs):
    """Differentiate Number (constant) with respect to any variable."""
    return Number(0.0)

@differentiate.register(Add)
def _(expr, *args, var=None, **kwargs):
    """Differentiate Add expression with respect to var."""
    left, right = expr.operands
    left_diff = differentiate(left, var=var)
    right_diff = differentiate(right, var=var)
    return Add(left_diff, right_diff)

@differentiate.register(Sub)
def _(expr, *args, var=None, **kwargs):
    """Differentiate Sub expression with respect to var."""
    left, right = expr.operands
    left_diff = differentiate(left, var=var)
    right_diff = differentiate(right, var=var)
    return Sub(left_diff, right_diff)

@differentiate.register(Mul)
def _(expr, *args, var=None, **kwargs):
    """Differentiate Mul expression with respect to var using the product rule."""
    left, right = expr.operands
    left_diff = differentiate(left, var=var)
    right_diff = differentiate(right, var=var)
    return Add(Mul(left_diff, right), Mul(right_diff, left))

@differentiate.register(Div)
def _(expr, *args, var=None, **kwargs):
    """Differentiate Div expression with respect to var using the quotient rule."""
    nume, deno = expr.operands
    left_diff = differentiate(nume, var=var)
    right_diff = differentiate(deno, var=var)
    numerator = Sub(Mul(left_diff, deno), Mul(nume, right_diff))
    denominator = Pow(deno, Number(2))
    return Div(numerator, denominator)

@differentiate.register(Pow)
def _(expr, *args, var=None, **kwargs):
    """Differentiate Pow expression with respect to var."""
    base, exponent = expr.operands

    # If the exponent is constant, apply the power rule: d/dx[u^n] = n * u^(n-1) * u'
    if isinstance(exponent, Number):
        base_diff = differentiate(base, var=var)
        return Mul(Mul(Number(exponent.value), Pow(base, exponent.value - 1)), base_diff)
    else:
        raise NotImplementedError("Differentiating variable exponents is not supported in this exercise.")



def postvisitor(expr, visit, **kwargs):
    visited = set()    # tracks expressions that have been visited
    stack = [(expr, False)]  # (expression, visited_children)
    results = {}

    while stack:
        current, children_visited = stack.pop()

        if id(current) in visited:
            continue

        if children_visited:
            # Get evaluated results of operands
            operands_results = [results[id(op)] for op in getattr(current, 'operands', ())]
            result = visit(current, *operands_results, **kwargs)
            results[id(current)] = result
            visited.add(id(current))
        else:
            stack.append((current, True))
            for operand in reversed(getattr(current, 'operands', ())):
                if id(operand) not in visited:
                    stack.append((operand, False))

    return results[id(expr)]
