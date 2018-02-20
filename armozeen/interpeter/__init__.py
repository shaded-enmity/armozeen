from armozeen import ast
from armozeen.parser.pipeline import PipelineStage
from armozeen.utils import check_expr, check_token, split_by
from armozeen.types import Expression, Expressions, Tokens, Token


class ExpressionNodeHandler(object):
    @property
    def exptype(self):
        return None

    def handle(self, node):
        pass


def slicebinary(items):
    assert len(items) == 2
    return items


def sliceternary(items):
    assert len(items) == 3
    return items


_registered_ast_nodes = []
def register_ast_node(C):
    
    _registered_ast_nodes.append(C())

    def inner(*args, **kwargs):
        return C

    return inner


@register_ast_node
class AssignENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'assign'

    def handle(self, node):
        lhs, _, rhs = sliceternary(node.children)
        left, right = None, None

        if check_expr(lhs, 'bracketfunc'):
            ''' LHS is a bracket func, this can be a write-bracket-function definition,
                which looks like:

                  R[integer r] = bits(32) value
                     .. block ..

                So we need to check for:

                  LHS: [Name, Load][Bracket[TypedVariable]]
                  RHS: [TypedVariable]

                Or this can be an actual invocation of write-bracket-functiion:

                  R[0] = '1010010...1'

                In which case we need to check for:

                  LHS: [Name, Load][Bracket[Name, Load, FunctionCall, Parens]]
                  RHS: [Expression]

                Any other case results in an error.
        
            items_are_typed_vars = all(check_expr(it, 'typed_variable') for it in lhs_item if not check_token(it, ','))
            if (not lhs_item or items_are_typed_vars) and check_expr(rhs, 'typed_variable'):
                lhs_name.children[0].char += '__Write'
                return ast.AstFunctionDefinition(
                        None, rhs, lhs_name, lhs_item, [], is_bracket=True
                )
            '''
            lhs_name, lhs_item = slicebinary(lhs.children)
            if not lhs_item or check_expr(lhs_item[0], [Expressions.Name, Expressions.Load, Expressions.Funcall, Expressions.Paren, Expressions.Number, 'op_num_*']):
                #lhs_name.children[0].char += '__Write'
                return ast.AstFunctionCall(
                        None, lhs_name, [lhs_item[0], rhs], is_bracket=True
                )
            else:
                raise Exception('AssignENH > Bracket func > {} |. {} <- {}'.format(lhs_name, lhs_item, rhs))
        elif check_expr(lhs, [Expressions.Name, 'typed_variable', 'tuple', 'load', Expressions.Substruct, 'bitselect']):
            left, right = lhs, rhs
        elif check_expr(lhs, 'destructure'):
            left, right = lhs, rhs
        else:
            raise Exception('AssignENH > LHS > {}'.format(lhs.type))

        return ast.AstBinaryOperator(None, ast.astnode.Assignment, left, right)


@register_ast_node
class FunctionCallENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'funcall'

    def handle(self, node):
        name, params = slicebinary(node.children)
        return ast.AstFunctionCall(None, name, params.children)


@register_ast_node
class FunctionDefENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'function_definition'

    def handle(self, node):
        return_type, name, params, impl = node.children
        return ast.AstFunctionDefinition(None, return_type[0], name[0], [x[0] for x in params[0]], impl)


@register_ast_node
class IfENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return Expressions.IfStatement

    @staticmethod
    def handle_branch(b):
        if len(b.children) != 2:
            raise Exception()
        if len(b.children[0]) != 1:
            raise Exception()
        return ast.AstBranch(None, b.children[0][0], b.children[1])

    def handle(self, node):
        L = len(node.children)

        assert L > 0

        if_branch = node.children[0]
        elsif_branches = node.children[1:-1] if L > 2 else []
        else_branch = node.children[-1] if L > 1 else []

        if not check_expr(if_branch, 'ifthen'):
            raise Exception()

        eib_branches = []
        for eib in elsif_branches:
            if not check_expr(eib, 'elsifthen'):
                raise Exception()
            eib_branches.append(
                self.handle_branch(eib)
            )

        if else_branch and not check_expr(else_branch, ['elsifthen', 'elseblock']):
            raise Exception("IfENH > Invalid else branch > {}".format(else_branch.type))
        
        return ast.AstIf(
                None, 
                self.handle_branch(if_branch),
                eib_branches, 
                ast.AstBranch(None, None, else_branch.children[0]) if else_branch else None
        )


@register_ast_node
class BitselectENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'bitselect'

    def handle(self, node):
        #print(node.children)
        return ast.AstBitselect(None, *node.children)


@register_ast_node
class DestructureENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'substructure'

    def handle(self, node):
        return ast.AstDestructure(None, *node.children)


@register_ast_node
class ConcatENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'op_concat'

    def handle(self, node):
        #print(node)
        l, _, r = sliceternary(node.children)
        return ast.AstBinaryOperator(None, ast.astnode.Concat, l, r)


@register_ast_node
class NumericOperationENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'op_num_*'

    def handle(self, node):
        l, _, r = sliceternary(node.children)
        typ = {
            'add': ast.astnode.Add,
            'sub': ast.astnode.Sub,
            'mult': ast.astnode.Mul,
            'div': ast.astnode.Div,
            'mod': ast.astnode.Mod,
            'bsr': ast.astnode.BitshiftRight,
            'bsl': ast.astnode.BitshiftLeft,
            'pow': ast.astnode.Power,
            'xor': ast.astnode.Xor
        }.get(node.type.rsplit('_', 1)[1])
        return ast.AstBinaryOperator(None, typ, l, r)


@register_ast_node
class LoadENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return Expressions.Load

    def handle(self, node):
        l, _, r = sliceternary(node.children)
        return ast.AstBinaryOperator(None, ast.astnode.Load, l, r)


@register_ast_node
class ForLoopENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'forloop'

    def handle(self, node):
        node.children[0] = node.children[0][0]
        return ast.AstForLoop(None, *node.children)


@register_ast_node
class StringENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return Expressions.String

    def handle(self, node):
        return ast.AstString(None, node.children[0])


@register_ast_node
class NumberENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return Expressions.Number

    def handle(self, node):
        return ast.AstNumber(None, node.children[0])


@register_ast_node
class SetENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'curly'

    def handle(self, node):
        return ast.AstSet(None, [x.pop() for x in split_by(node.children, ',', sanitize=True)])


@register_ast_node
class TupleENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'tuple'

    def handle(self, node):
        #print('ORLY? ', node)
        return ast.AstTuple(None, [x.pop() for x in split_by(node.children, ',', sanitize=True)])


@register_ast_node
class EnumENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'enum'

    def handle(self, node):
        return ast.AstEnumerationDefinition(
                None, 
                node.children[0], 
                [x.pop() for x in split_by(node.children[1], ',', sanitize=True)]
        )


@register_ast_node
class WBracketfundefENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'wbracketfundef'

    def handle(self, node):
        bf, value, impl = node.children

        ''' We need to mangle the name with the `__Write` suffix
            so that we're able to distinguish between read / write
            forms of array functions.

            If the name is a `load` operation we need to drill down
            to the last right-node, which must hold a `name` node
            that we can mangle.
            This has the following effect:

               load: [name: [A], ,, name: [B]]
               load: [name: [A], ,, name: [B__Write]]

        '''
        name = bf.children[0]
        while check_expr(name, Expressions.Load):
            name = name.children[2]
        assert check_expr(name, Expressions.Name)
        name.children[0].char += '__Write'

        return ast.AstFunctionDefinition(
                None, None, bf.children[0], bf.children[1] + [value], impl, is_bracket=True
        )


@register_ast_node
class ReturnENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return ['return', 'return_value']

    def handle(self, node):
        item = node.children[0] if node.children else None
        return ast.AstUnaryOperator(None, ast.astnode.Return, item)


@register_ast_node
class BracketfundefENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'bracketfundef'

    def handle(self, node):
        rt, bf, impl = node.children
        return ast.AstFunctionDefinition(
                None, rt, bf.children[0], bf.children[1], 
                impl, is_bracket=True
        )


@register_ast_node
class TypedefENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'typedef'

    def handle(self, node):
        return ast.AstTypeDefinition(
                None, node.children[0].children[0].char, 
                [x.children[0] for x in node.children[1]]
        )


@register_ast_node
class DestructureENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'destructure'

    def handle(self, node):
        return ast.AstDestructure(None, node.children[0], node.children[1])


@register_ast_node
class CaseOfENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'caseof'

    def handle(self, node):
        whens, otherwise = [], None
        for spec in node.children[1].children:
            if check_expr(spec, 'when*'):
                whens.append(spec)
            elif check_expr(spec, 'otherwise*'):
                if otherwise:
                    raise ValueError('More than 1 otherwise in block not permitted')
                otherwise = spec

        case = node.children[0][0]
        #if check_expr(case, 'funcall'):
        #   pass

        name = node.children[0][0].children[0]
        if isinstance(name, Token):
            name = name.char

        return ast.AstCaseOf(None,
                case,
                whens,
                otherwise
        )


@register_ast_node
class ComparisonENH(ExpressionNodeHandler):
    _Keys = ['op_equal', 'op_not_equal', 'op_greater_than', 'op_gteq', 'op_less_than', 
             'op_leq', 'op_in', 'op_logical_and', 'op_logical_or']
    _Values = [ast.astnode.Equal, ast.astnode.NotEqual, ast.astnode.GreaterThan, 
               ast.astnode.GraterThanOrEqual, ast.astnode.LessThan, 
               ast.astnode.LessThanOrEqual, ast.astnode.In, ast.astnode.And, ast.astnode.Or] 
    _Mapping = dict(zip(_Keys, _Values))

    @property
    def exptype(self):
        return self._Keys

    def handle(self, node):
        l, _, r = sliceternary(node.children)
        return ast.AstBinaryOperator(None, self._Mapping[node.type], l, r)


@register_ast_node
class ArrayDefENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'arraydef'

    def handle(self, node):
        type_, spec = slicebinary(node.children)
        return ast.AstArrayDef(None, type_, spec)


@register_ast_node
class BracketfuncENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'bracketfunc'

    def handle(self, node):
        name, params = slicebinary(node.children)
        return ast.AstFunctionCall(None, name, params, is_bracket=True)


@register_ast_node
class NameENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'name'

    def handle(self, node):
        return ast.AstName(None, node.children[0])


@register_ast_node
class TextStringENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'sstring'

    def handle(self, node):
        return ast.AstTextString(None, node.children[0])


@register_ast_node
class BlockENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'block'

    def handle(self, node):
        return ast.AstBlock(None, node.children)


@register_ast_node
class TypedVariableENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'typed_variable'

    def handle(self, node):
        return ast.AstTypedName(None, *node.children)


@register_ast_node
class TypeENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'typeexp'

    def handle(self, node):
        return ast.AstType(None, *node.children)


@register_ast_node
class ExpressionENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'paren'

    def handle(self, node):
        return ast.AstExpression(None, node.children)


@register_ast_node
class RealENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'real'

    def handle(self, node):
        value = float(str(node.children[0].children[0]) + '.' + str(node.children[2].children[0]))
        return ast.AstReal(None, value)


@register_ast_node
class SubstructureENH(ExpressionNodeHandler):
    @property
    def exptype(self):
        return 'substruct'

    def handle(self, node):
        return ast.AstSubstructure(None, node.children)


class AstBuilder(PipelineStage):
    def run(self, items):
        toks = []
        for item in items:
            if isinstance(item, list):
                toks.append(self.run(item))
            else:
                s = False
                for nt in _registered_ast_nodes:
                    if check_expr(item, nt.exptype):
                        toks.append(nt.handle(item))
                        toks[-1].children = self.run(toks[-1].children)
                        s = True
                if not s:
                    if isinstance(item, Expression):
                        item.children = self.run(item.children)
                    toks.append(item)
        return toks
