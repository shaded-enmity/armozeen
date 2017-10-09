from armozeen.types import Expressions, Tokens, Expression


class astnode(object):
    (Assignment, FunctionCall, FunctionDefinition, EnumerationDefinition, String, Set, Bitselect,
     Destructure, Branch, If, CaseOf, ForLoop, Load, Concat, Add, Sub, Mul, Div, Mod, Xor,
     BitshiftLeft, BitshiftRight, ArrayAccess, Compare, Number, And, Or, GreaterThan, LessThan,
     GraterThanOrEqual, LessThanOrEqual, Equal, NotEqual, In, Return, Tuple, TypeDefinition,
     ArrayDefinition, Power) = (
            'ast_assignment', 'ast_function_call', 'ast_function_definition',
            'ast_enumeration_definition', 'ast_string', 'ast_set', 'ast_bitselect',
            'ast_destructure', 'ast_branch', 'ast_if', 'ast_caseof', 'ast_forloop',
            'ast_load', 'ast_string_concat', 'ast_num_add', 'ast_num_sub', 'ast_num_mul',
            'ast_num_div', 'ast_num_mod', 'ast_num_xor', 'ast_num_bsl', 'ast_num_bsr', 'ast_array_access',
            'ast_compare', 'ast_number', 'ast_and', 'ast_or', 'ast_greater_than', 'ast_less_than',
            'ast_greater_than_or_equal', 'ast_less_than_or_equal', 'ast_equal', 'ast_not_equal', 'ast_in',
            'ast_return', 'ast_tuple', 'ast_type_definition', 'ast_array_definition', 'ast_power'
    )


astnodemap = {v:k for k, v in astnode.__dict__.iteritems() if not k.startswith('_')}
def childref(me, num):
    return me.children[num]

class AstNode(Expression): pass


class AstBinaryNode(AstNode):
    def __init__(self, parent, typ, left, right):
        super(AstBinaryNode, self).__init__(parent, 'ast_binary_node_' + typ)
        self.children = [left, right]

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]


class AstBinaryOperator(AstBinaryNode):
    def __init__(self, parent, typ, left, right):
        super(AstBinaryOperator, self).__init__(parent, typ, left, right)
        self.operator = typ


class AstFunctionDefinition(AstNode):
    def __init__(self, parent, return_type, name, parameters, impl, is_bracket=False):
        super(AstFunctionDefinition, self).__init__(parent, 'ast_function_definition')
        self.children = [return_type, name, parameters, is_bracket, impl]

    return_type = property(lambda self: self.children[0])
    name = property(lambda self: self.children[1])
    parameters = property(lambda self: self.children[2])
    is_bracket = property(lambda self: self.children[3])
    impl = property(lambda self: self.children[4])


class AstFunctionCall(AstNode):
    def __init__(self, parent, name, parameters, is_bracket=False):
        super(AstFunctionCall, self).__init__(parent, 'ast_function_call')
        self.children = [name, parameters, is_bracket]

    name = property(lambda self: self.children[0])
    parameters = property(lambda self: self.children[1])
    is_bracket = property(lambda self: self.children[2])


class AstTypeDefinition(AstNode):
    def __init__(self, parent, type_name, fields):
        super(AstTypeDefinition, self).__init__(parent, 'ast_type_definition')
        self.children = [type_name, fields]

    type_name = property(lambda self: self.children[0])
    fields = property(lambda self: self.children[1])


class AstEnumerationDefinition(AstNode):
    def __init__(self, parent, name, fields):
        super(AstEnumerationDefinition, self).__init__(parent, 'ast_enumeration_definition')
        self.children = [name, fields]

    name = property(lambda self: self.children[0])
    fields = property(lambda self: self.children[1])


class AstUnaryNode(AstNode):
    def __init__(self, parent, typ, item):
        super(AstUnaryNode, self).__init__(parent, 'ast_unary_node_' + typ)
        self.children = [item]

    @property
    def item(self):
        return self.children[0]


class AstUnaryOperator(AstUnaryNode):
    def __init__(self, parent, typ, item):
        super(AstUnaryOperator, self).__init__(parent, typ, item)
        self.operator = typ


class AstNumber(AstNode):
    def __init__(self, parent, value):
        super(AstNumber, self).__init__(parent, 'ast_number')
        self.children = [value]

    value = property(lambda self: self.children[0])


class AstString(AstNode):
    def __init__(self, parent, value):
        super(AstString, self).__init__(parent, 'ast_string')
        self.children = [value]

    value = property(lambda self: self.children[0])


class AstSet(AstNode):
    def __init__(self, parent, items):
        super(AstSet, self).__init__(parent, 'ast_set')
        self.children = items

    items = property(lambda self: self.children[0])

class AstTuple(AstNode):
    def __init__(self, parent, items):
        super(AstTuple, self).__init__(parent, 'ast_tuple')
        self.children = items

    items = property(lambda self: self.children[0])

class AstBitselect(AstNode):
    def __init__(self, parent, source, selection):
        super(AstBitselect, self).__init__(parent, 'ast_bitselect')
        self.children = [source, selection]

    source = property(lambda self: self.children[0])
    selection = property(lambda self: self.children[1])


class AstDestructure(AstNode):
    def __init__(self, parent, source, selection):
        super(AstDestructure, self).__init__(parent, 'ast_destructure')
        self.children = [source, selection]

    source = property(lambda self: self.children[0])
    selection = property(lambda self: self.children[1])


class AstBranch(AstNode):
    def __init__(self, parent, condition, block):
        super(AstBranch, self).__init__(parent, 'ast_branch')
        self.children = [condition, block]

    condition = property(lambda self: self.children[0])
    block = property(lambda self: self.children[1])


class AstIf(AstNode):
    def __init__(self, parent, if_branch, elsif_branches, else_branch):
        super(AstIf, self).__init__(parent, 'ast_if')
        self.children = [if_branch, elsif_branches, else_branch]

    if_branch = property(lambda self: self.children[0])
    elsif_branches = property(lambda self: self.children[1])
    else_branch = property(lambda self: self.children[2])


class AstCaseOf(AstNode):
    def __init__(self, parent, case, branches, otherwise):
        super(AstCaseOf, self).__init__(parent, 'ast_caseof')
        self.children = [case, branches, otherwise]

    case = property(lambda self: self.children[0])
    branches = property(lambda self: self.children[1])
    otherwise = property(lambda self: self.children[2])


class AstForLoop(AstNode):
    def __init__(self, parent, init, finish, block):
        super(AstForLoop, self).__init__(parent, 'ast_forloop')
        self.children = [init, finish, block]

    init = property(lambda self: self.children[0])
    finish = property(lambda self: self.children[1])
    block = property(lambda self: self.children[2])


class AstArrayDef(AstNode):
    def __init__(self, parent, type_, spec):
        super(AstArrayDef, self).__init__(parent, 'ast_array_definition')
        self.children = [type_, spec.children[0], spec.children[1]]

    item_type = property(lambda self: self.children[0])
    name = property(lambda self: self.children[1])
    size_spec = property(lambda self: self.children[2])

