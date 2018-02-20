from armozeen.parser.pipeline import PipelineStage
from armozeen.utils import replace, check_expr, check_token
from armozeen.types import Expression, Expressions

from collections import defaultdict
from functools import partial


class GrammarAtom(object):
    def match(self, element):
        return False


class GrammarExpression(GrammarAtom):
    def __init__(self, expr_type=None):
        self.exprression_type = expr_type
        super(GrammarExpression, self).__init__()

    def match(self, element):
        return not self.exprression_type or check_expr(element, self.exprression_type)


class GrammarToken(GrammarAtom):
    def __init__(self, token_type):
        self.token_type = token_type
        super(GrammarToken, self).__init__()

    def match(self, element):
        return check_token(element, self.token_type)


class GrammarConsume(GrammarAtom):
    def __init__(self, group):
        self.group = group
        super(GrammarConsume, self).__init__()

    def match(self, element):
        return True


class GrammarConsumeTyped(GrammarConsume):
    def __init__(self, group, type_):
        self.type = type_
        super(GrammarConsumeTyped, self).__init__(group)

    def match(self, element):
        return check_expr(element, self.type) or check_token(element, self.type)


class Grammar(PipelineStage):
    def items(self):
        return []

    def create_item(self, tokens, items, consumed):
        tokens += items

    def _match(self, atom, node):
        return self.items[atom].match(node)

    def run(self, etstream):
        atom, consumed = 0, defaultdict(list)
        start, sl, toks, shadow = -1, len(etstream), [], []
        leftover, S = None, len(self.items)
        for i, n in enumerate(etstream):
            shadow.append(n)
            matched, opt = False, False
            # Consumer code needs to buffer data separately and we need to check it first
            # since if we consume the expression or token it shouldn't be processed any
            # further
            if isinstance(self.items[atom], GrammarConsume):
                ei = S > atom + 1
                if self._match(atom, n):
                    # If we have enough grammar atoms and the next atom matches the current
                    # value we need to stop consuming
                    if ei and self._match(atom + 1, n):
                        if i == sl - 1 or S - 1 == atom + 1:
                            self.create_item(toks, etstream[start:i], consumed)
                            atom, start = 0, -1
                        else:
                            atom += 2
                    elif (sl - 1) == i:
                        consumed[self.items[atom].group].append(n)
                        self.create_item(toks, etstream[start:i], consumed)
                        start = -1
                        break
                    else:
                        consumed[self.items[atom].group].append(n)
                    continue
                else:
                    if (sl - 1) == i:
                        self.create_item(toks, etstream[start:i], consumed)
                        start = -1
                        break
                    if not ei:
                        self.create_item(toks, etstream[start:i], consumed)
                        atom, start = 0, -1
                    else:
                        atom += 1

            if self._match(atom, n):
                if atom == 0:
                    start = i
                    atom += 1
                elif atom == S - 1:
                    inp = ([leftover] if leftover else []) + etstream[start:i + 1]

                    self.create_item(toks, inp, consumed)

                    consumed, created, start, atom = defaultdict(list), toks[-1], -1, 0

                    if self._match(0, created):
                        # first recurse into the created node
                        created.children = self.run(created.children)
                        if (sl - i) < (S - atom):
                            break
                        check = partial(check_expr, t=Expressions.Paren)
                        # Out current grammar matches the expression created above by the call
                        # to `create_items`, so we need to check whether some of the children
                        # of the created expression matches our *current* grammar as well,
                        # this is necessary to tackle cases like:
                        #
                        #      L    O R
                        #   (a * b) * c
                        #
                        # Where the left (L) or right (R) parts may require further care
                        # since they need to be processed by the current operator/grammar (O) as
                        # well, hence we check if some of the children contain scope worth
                        # inspecting (currently `paren`) and replace the original entry
                        # with the processed one
                        sub_expressions = list(
                            filter(check, created.children))
                        if sub_expressions:
                            for torep, repl in zip(sub_expressions, self.run(sub_expressions)):
                                replace(created.children, torep, repl)

                        toks.pop()

                        atom, start, leftover = 1, i + 1, created
                else:
                    atom += 1
            else:
                toprocess = [n]
                if start > -1:
                    toprocess += shadow[start:i]
                    toks += shadow[start:i]
                    atom, start = 0, -1

                consumed = defaultdict(list)
                for tp in toprocess:
                    if isinstance(tp, Expression):
                        tp.children[:] = self.run(tp.children)

                if isinstance(n, list):
                    n[:] = self.run(n)

                if leftover:
                    toks.append(leftover)

                leftover = None
                toks.append(n)

        if start > -1:
            if isinstance(self.items[atom], GrammarConsume) and atom == (S - 1):
                # The last element of the grammar is GrammarConsume atom and we have no nodes left
                # and since GrammarConsume atoms are optional we just create the node now
                self.create_item(toks, etstream[start:], consumed)
            else:
                if isinstance(n, Expression):
                    n.children[:] = self.run(n.children)

                toks.append(n)

        return toks


# Helpful aliases for grammar definitions
GExpr = GrammarExpression
GToken = GrammarToken
GConsume = GrammarConsume
GConsumeT = GrammarConsumeTyped
