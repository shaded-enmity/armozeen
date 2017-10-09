from armozeen.parser.pipeline import PipelineStage
from armozeen.ast import astnode
from armozeen.utils import check_ast, check_expr, swap, nest, check_token
from armozeen.types import Expression, Expressions, Token, ArmozeenException
from re import compile as regexp


# Regexp to recognize printf() style specifiers %s %i and %b
fmtre = regexp('%[sib]')


def enum(enums):
    return type('Enum', (), dict(zip(enums, enums)))


# Abbreviation of Runtime Exception Type
RET = enum(["DIFF_BIT_SIZE",
            "ARG_WRONG_TYPE", "ARG_WRONG_NUMBER",
            "BITSELECT_RANGE",
            "FMT_SPEC",
            "BAD_META",
            "NOT_FOUND",
            "INVALID_RETURN",
            "BAD_OPERATOR",
            "CONSTANT",
            "ALREADY_DECLARED",
            "RUNTIME_ERROR"])


class ArmozeenRuntimeException(ArmozeenException):
    def __init__(self, type_, message):
        self.type_ = type_
        super(ArmozeenRuntimeException, self).__init__(message)


def _raise(t, m):
    raise ArmozeenRuntimeException(t, m)


def match_types_strict(a, b, relax_const=False):
    ''' Perform a `strict` matching of types -

        * Check that names match
        * If the type is a bitstring, check that sizes match
    '''
    if a.const and not relax_const:
        return False
    if a.name != b.name:
        return False
    if a.name == ValueTypes.Bits and a.size != b.size:
        return False

    return True


def names_to_strings(names):
    ''' Unwrap [Name] -> [str]

        The actual string is stored in the first child of the given name.
        Note that the string was kept in the Token type withing the pipeline,
        so that we need to access it via `char`.
    '''
    for name in names:
        yield name.children[0].char


def value_cmp(a, b):
    ''' Compare `a` and `b` by first coercing types and then checking equality
        of values.

    '''
    aX, bX = Coercion.coerce(a, b)

    if aX.type_info.name == ValueTypes.Bits:
        A, B = aX.value, bX.value
        if aX.type_info.size != bX.type_info.size:
            _raise(RET.DIFF_BIT_SIZE, 
                  'Invalid bits comparison {} and {}'.format(_format_typeinfo(aX.type_info), 
                                                             _format_typeinfo(bX.type_info)))

        for i in range(aX.type_info.size):
            if A[i] == 'x':
                continue
            elif B[i] == 'x':
                continue
            else:
                if A[i] != B[i]:
                    return False
        return True

    return aX.value == bX.value


def _corece_single(elem, target_type):
    ''' Coerce `elem` to `target_type` or return back `elem`.

    '''
    if elem.type_info.name != target_type:
        funcaX = elem.type_info.name + '_' + target_type
        return Coercion.coercions[funcaX](elem)
    return elem


def value_lt(a, b):
    ''' Coerce `a` and `b` to integers and use the `<` operator to check the values

    '''
    aX, bX = Coercion._apply_native(a, b)
    aX = _corece_single(aX, ValueTypes.Integer)
    bX = _corece_single(aX, ValueTypes.Integer)

    if not (aX.type_info.name == ValueTypes.Integer and bX.type_info.name == ValueTypes.Integer):
        _raise(RET.ARG_WRONG_TYPE, 'Arguments to < must be integers')

    return aX.value < bX.value


def value_logic(a, b, op):
    ''' Coerce `a` and `b` to booleans and apply the `&&` or `||` operators

    '''
    aX, bX = Coercion._apply_native(a, b)
    aX = _corece_single(aX, ValueTypes.Bool)
    bX = _corece_single(aX, ValueTypes.Bool)

    opmap = {
        'or':  lambda x, y: x or y,
        'and': lambda x, y: x and y
    }

    return opmap[op](aX.value, bX.value)


def bitselect(node, state):
    ''' Perform a bit selection:

            ident<index>
            ident<upper_1:lower_1, upper_2:lower_2 ... upper_N:lower_N>

        Note that both of the specs can be mixed, so in reality we can have:

            some_value<63:60, 50:45, 41, 5:0>
            
        This returns a list of tuples in the form of (upper, lower) or (index, index).
    '''
    candidates = []
    if len(node.children[1]) == 1:
        upper = state._clean_eval(node.children[1][0])
        lower = upper
        candidates.append((lower, upper))
    else:
        stash, prev = None, None
        num = len(node.children[1])
        for i, cnd in enumerate(node.children[1]):
            if check_token(cnd, ':'):
                stash = prev
            else:
                if stash:
                    candidates.append((state._clean_eval(stash), state._clean_eval(cnd)))
                    stash = None
                    prev = None
                    continue
                elif check_ast(prev, astnode.Number):
                    v = state._clean_eval(prev)
                    candidates.append((v, v))
            prev = cnd if i != (num-1) else None
        if check_ast(prev, astnode.Number):
            v = state._clean_eval(prev)
            candidates.append((v, v))
    
    for a, b in candidates:
        if not (a.type_info == ValueTypes.Integer and b.type_info == ValueTypes.Integer):
            _raise(RET.BITSELECT_RANGE, 
                   'Invalid upper/lower bitselect: {}\n{}'.format(lower, upper))

    return candidates


def get_bool_conf(context, name):
    return context.config.get(name, False)


def builtin__backtrace(args, context):
    ''' Print backtrace

    '''
    context._backtrace()


def builtin__UInt(args, context):
    if not args or len(args) > 1:
        _raise(RET.ARG_WRONG_NUMBER, 'Invalid number of arguments')

    s = args[0]

    if s.type_info.name != ValueTypes.Bits:
        _raise(RET.ARG_WRONG_TYPE, 'Invalid type for conversion to integer')

    return context._clean_eval(s.to_uint())


def builtin__print(args, context):
    ''' Printf builtin function

    '''
    vtmap = {'bits': 's', 'integer': 'i', 'bool': 'b'}

    fmt, fmtargs = args[0].char, args[1:]

    formats = fmtre.findall(fmt)
    argtypes = [vtmap[a.type_info.name] for a in fmtargs]

    if not all(a[1] == f for a, f in zip(formats, argtypes)):
        _raise(RET.FMT_SPEC, 'Invalid format specifier')

    print(fmtre.sub(StringProvider([str(sa.value) for sa in fmtargs]), fmt).replace('\\n', '\n'))

    return NullResult


def _format_typeinfo(ti):
    if isinstance(ti.subtype, TypeInfo):
        sub = _format_typeinfo(ti.subtype)
    else:
        sub = ti.subtype
    return '{pref}{refs}{name}{size}{sub}'.format(
        name=ti.name,
        refs='&' if ti.ref else '',
        pref='constant ' if ti.const else '',
        size='({0})'.format(ti.size) if ti.name in ['bits', 'array'] else '',
        sub='<{0}>'.format(sub) if sub else ''
    )


def _dumpvar(args, context, sort=False):
    ''' Dumpvar builtin function, prints information about variable value / type

    '''
    try:
        from prettytable import PrettyTable
    except ImportError:
        PrettyTable = None

    def fmt_def_param(p):
        if check_expr(nest(p, 0), Expressions.Bits):
            tn  = 'bits(' + nest(p, 0, 0, 0).char + ')'
            return tn
        else:
            return nest(p, 0, 0).char

    direct = True
    if not args:
        direct = False
        args = list(context.locals.keys())

    if PrettyTable:
        def _dumptype(items):
             return "{\n" + "\n".join(items) + "\n}"

        pt = PrettyTable(["Variable", "Value", "Type"])
        for local in args:
            v = context.locals[local]

            if check_ast(v, astnode.FunctionDefinition):
                if not v.is_bracket:
                    nm = nest(v, 1, 0).char
                    fparams = []
                    for x in v.children[2]:
                        for p in x:
                            fparams.append(fmt_def_param(nest(p, 0)))
                    pt.add_row([nm, '{}({})'.format(nm, ', '.join(fparams)), 'constant function'])
                else:
                    nm = nest(v, 1, 0).char
                    is_write = nm.endswith('__Write')
                    if is_write:
                        fp = []
                        for p in nest(v, 2):
                            fp.append(fmt_def_param(p))
                                               
                        val, params =  fp[0], ', '.join(fp[1:])
                        pt.add_row([nm, '{nm}[{params}] = {val}'.format(nm=nm, val=val, params=params), 'array funtion'])
                    else:
                        fp = []
                        for p in nest(v, 2):
                            fp.append(fmt_def_param(p))
                        
                        val, params = fmt_def_param(v), ', '.join(fp)
                        pt.add_row([
                            nm, 
                            '{val} {nm}[{params}]'.format(nm=nm, val=val, params=params), 
                            'constant array funtion'
                        ])
            elif isinstance(v, (Value, Bitstring)):
                ti = v.type_info
                v = v.value if v.value is not None else '<unset>' if isinstance(v, Value) else v
                if ti.name == ValueTypes.Type and v != '<unset>':
                    if isinstance(v, StructType):
                        v = v.fields
                        v = _dumptype([
                            '  {} {} = {}'.format(_format_typeinfo(rr.type_info), nm, rr.value or '<usnet>')
                            for nm, rr in v
                        ])
                    else:
                        v = _dumptype(
                                ['  {} {}'.format(
                                    _format_typeinfo(context._eval(sv)), 
                                    nest(sv, 1, 0).char) 
                                 for sv in v]
                        )
                pt.add_row([local, v, _format_typeinfo(ti)])
            elif isinstance(v, Scope):
                pt.add_row([local, '', 'scope<{}>'.format(v.name)])
            elif isinstance(v, Array):
                ti = v.type_info
                if direct or get_bool_conf(context, 'dump_full_arrays'):
                    fmtitems = "\n".join(
                            [('{:' + str(len(str(len(v.items))) + 1) + '} {}').format(
                                str(i) + ':', item.value
                             ) for i, item in enumerate(v.items)]
                    ) or '<empty>'
                else:
                    fmtitems = str([item.value for item in v.items] or '') or '<empty>'
                    if len(fmtitems) > 80:
                        fmtitems = fmtitems[:80] + '...'
                pt.add_row([local, fmtitems, _format_typeinfo(ti)])
            else:
                continue

        pt.align = 'l'
        if sort:
            print(pt.get_string(sortby='Variable', sort_key=lambda x: x[0][0].lower()))
        else:
            print(pt)
    else:
        print("Locals dump:")
        for local in sorted(args):
            print('  {0} => {1}'.format(local, context.locals[local]))


def builtin__dumpvar(args, context):
    _dumpvar(args, context, sort=False)

def builtin__dumpvarsorted(args, context):
    _dumpvar(args, context, sort=True)


class Resolvable(object):
    ''' Protocol providing key/value semantics and key enumeration, this is used by things
        like Scope to implement namespace member lookup, or structs to implement member lookup.

        See classes implementing this protocol
    '''
    def __getitem__(self, item):
        return None
    
    def keys(self):
        return []


class ValueTypes(object):
    ''' Armozeen native types  '''
    (Struct, Integer, Real, Bits, Enum, Bool, Tuple, Type, Builtin, Array) = [
            'struct', 'integer', 'real', 'bits', 'enum', 'bool', 'tuple', 'type', 'builtin', 'array'
    ]


class EvalResult(object):
    ''' Protocol for encapsulating results from eval operations, which need to:
        1) Provide a way of cloning the object
        2) Provide meaningful type information
    
    '''
    def clone(self):
        return EvalResult()

    @property
    def type_info(self):
        return None


# Type tags
class ReturnResult(EvalResult): pass
class NullResult(EvalResult): pass


class Array(EvalResult):
    def __init__(self, items, type_):
        self.items = items
        self.type_ = type_
        self.value = items

    def clone(self):
        return Array([v.clone() for v in self.items], self.type_)

    @property
    def type_info(self):
        return TypeInfo(ValueTypes.Array, size=len(self.items), sub=self.type_)


class Bitstring(EvalResult):
    ''' Bitstring type
        --------------

        Represents a string of individual bits, the value
        in code can be specified such as:

            '110010' == '11 00 10'

        Bitstrings can also contain wildcard 'x' character:

            '1100xx' == {'110000', '110001', '110010', '110011'}
    '''
    def __init__(self, value):
        self.value = (''+value.char if not isinstance(value, str) else value).replace(' ', '').lower()

    def to_uint(self):
        return int(self.value, 2)

    def to_int(self):
        m = -1 if self.value[-1] == '1' else 1
        return m * int(self.value[:-1] or 0, 2)

    def clone(self):
        return Bitstring('' + self.value)

    def __repr__(self):
        return '<Bitstring value={}, uint={}>'.format(self.value, self.to_uint())

    @property
    def type_info(self):
        return TypeInfo(ValueTypes.Bits, size=len(self.value))


class TypeInfo(object):
    ''' Type information currently encodes:

        `name`  - Name of the type (Must be enumerated in ValueType as well)
        `size`  - Size of the type, interpretation of this value depends on `name`
        `ref`   - This is a reference to an actual value
        `const` - Const cannot be mutated
        `sub`   - Subtype where applicable (e.g. struct, enum)
    '''
    def __init__(self, name, size=1, ref=False, const=False, sub=None):
        self.name = name
        self.size = size
        self.ref = ref
        self.const = const
        self.subtype = sub

    def clone(self):
        return TypeInfo('' + self.name, self.size, self.ref, self.const)

    def __repr__(self):
        return '<TypeInfo name={}, size={}, ref={}, const={}>'.format(
            self.name, self.size, self.ref, self.const
        )

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        super(TypeInfo, self).__eq__(self, other)


class StructType(Resolvable):
    ''' When instantiated this represents the struct runtime information. 

        `name`   - Name of the struct
        `fields` - List of `Value` items
        `_fd`    - Key/Value mapping of member names/values 

        This class implements the `Resolvable` protocol - this means that
        it can be used for Load operation in variables, e.g.:

            Point p;
            p.x = 1;

        So on the first line we instantiate `StructType` from `Point` specification as `p`
        and set the `x` member to 1 - we need to resolve `x` from `p`. Locals are already stored
        in a dictionary, so if we conform to the same underlying protocol underlying them both
        we can drill down to values in a neat recursive fashion.
    '''
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields
        self._fd = dict(fields)

    def __getitem__(self, item):
        if check_expr(item, Expressions.Name):
            item = item.children[0].char
        return self._fd[item]

    def keys(self):
        return self._fd.keys()


class Value(EvalResult):
    ''' Value created via eval, this type is used for all runtime variables which
        store actual value

    '''
    def __init__(self, vt, v):
        self.value_type = vt
        self.value = v
    
    @property
    def type_info(self):
        return self.value_type

    def clone(self):
        v = None
        if self.value_type.name == ValueTypes.Tuple:
            v = Value(self.value_type.clone(), [it.clone() for it in self.value])
        else:
            v = self.value
        return Value(self.value_type.clone(), v)

    def __repr__(self):
        return '<Value type={}, value={}>'.format(self.value_type, self.value)

    def check_type(self, other):
        if isinstance(other, Value):
            return self.value_type == other.value_type
        elif isinstance(other, str):
            return self.value_type == other
        return False


class Scope(Resolvable):
    ''' Scope provides a namespacing feature to support modules:

            ModuleA.fun1()
            ModuleB.fun2()

        Since we want to allow for arbitrary nesting (see StructType comment above) this 
        class conforms to Resolvable.

        TODO: * Limit assignment to scopes? (i.e. scopes are not first class members)
              * Limit assignment of scopes (i.e. scope cannot be embedded in a struct directly)
    '''
    def __init__(self, name):
        self.name = name
        self.items = {}

    def __repr__(self):
        from pprint import pformat
        return '<Scope name={}, items=<{}>>'.format(self.name, pformat(self.items))

    def __contains__(self, item):
        if check_expr(item, Expressions.Name):
            item = item.children[0].char
        if isinstance(item, Token):
            item = item.char
        return item in self.items

    def keys(self):
        return self.items.keys()

    def __setitem__(self, item, value):
        if check_expr(item, Expressions.Name):
            item = item.children[0].char
        if isinstance(item, Token):
            item = item.char
        self.items[item] = value

    def __getitem__(self, item):
        if check_expr(item, Expressions.Name):
            item = item.children[0].char
        if isinstance(item, Token):
            item = item.char
        return self.items[item]


class Coercion(object):
    ''' Type coercion semantics for binary operators
        --------------------------------------------

        The `native` table specifies mapping between native Python types
        and their corresponding Armozeen type, native types are always
        coerced to Armozeen types first befire applying any further rules

        The `rules` table specifies how to handle possible (left, right) combinations
        and resulting coerced types. Type names are joined to lookup a coercion function
        e.g.:

           (_, 'integer'), (_, 'bits') => integer_bits -> bits

        General rule of the thumb is to coerce to the type LHS type.
    '''
    rules = [
        (('bits', 'integer'), ('bits', 'bits')),
        (('real', 'integer'), ('real', 'real')),
        (('integer', 'real'), ('integer', 'integer')),
        (('integer', 'bits'), ('integer', 'integer')),
        (('bool', 'integer'), ('bool', 'bool')),
        (('integer', 'bool'), ('integer', 'integer')),
        (('bits', 'bool'), ('bits', 'bits')),
        (('bool', 'bits'), ('bool', 'bool')),
    ]
    native = [
        ('!str', 'bits'),
        ('!int', 'integer'),
        ('!bool', 'bool')
    ]
    coercions = {
        'str': lambda x: Bitstring(x),
        'int': lambda x: Value(TypeInfo(ValueTypes.Integer), x),
        'bool': lambda x: Value(TypeInfo(ValueTypes.Bool), x),
        'integer_real': lambda x: Value(TypeInfo(ValueTypes.Real), float(x)),
        'real_integer': lambda x: Value(TypeInfo(ValueTypes.Integer), int(x)),
        'integer_bits': lambda x: Bitstring(bin(x.value)[2:]),
        'bits_integer': lambda x: Value(TypeInfo(ValueTypes.Integer), x.to_uint()),
        'bool_integer': lambda x: Value(TypeInfo(ValueTypes.Integer), int(x.value)),
        'integer_bool': lambda x: Value(TypeInfo(ValueTypes.Bool), bool(x.value)),
        'bits_bool': lambda x: Value(TypeInfo(ValueTypes.Bool), bool(x.value == '1')),
        'bool_bits': lambda x: Bitstring(chr(x.value))
    }

    @staticmethod
    def _apply_native(a, b):
        for lhs, rhs in Coercion.native:
            if a.__class__.__name__ == lhs[1:]:
                a = Coercion.coercions[lhs[1:]](a)
            if b.__class__.__name__ == lhs[1:]:
                b = Coercion.coercions[lhs[1:]](b)
        return a, b

    @staticmethod
    def _apply_coercions(a, b):
        aT = a.type_info.name
        bT = b.type_info.name

        if aT == bT:
            return a, b

        for ((lhsA, lhsB), (rhsA, rhsB)) in Coercion.rules:
            if lhsA == aT and lhsB == bT:
                fnameA = aT + '_' + lhsA
                fnameB = bT + '_' + rhsB
                if aT != lhsA:
                    a = Coercion.coercions[fnameA](a)
                if bT != rhsB:
                    b = Coercion.coercions[fnameB](b)

        return a, b

    @staticmethod
    def coerce(a, b):
        a, b = Coercion._apply_native(a, b)
        return Coercion._apply_coercions(a, b)


class MetavarSolution(object):
    ''' Computes solution to a meta-variable problem, this provides
        checking for bitstring sizes, e.g.:

            ((bits(N), bits(M), bits(N+M)) IdentityAndCombine(bits(N) first, bits(M) second)
                return ((first, second), first:second)

            IdentityAndCombine('11', '00') == (('11', '00'), '1100')
            
        This object validates that the return bitstring sizes in return value match the
        actually returned runtime value.
        If we changed the implementation such as:

            return ((first, second), first:second:'1')

        An error would be thrown, since the size of the rightmost string would be:
        '11100' = 5 and thus the N+M = 4 check would fail.

        This is essentially a post-condition checking.
    '''
    def __init__(self, outputs, context):
        self.outputs = outputs
        self.context = context

    def _incontext_eval(self, frame, n):
        self.context.state.frames.append(frame)
        toret = [self.context._eval(v) for v in n]
        self.context.state.frames.pop()
        return toret

    def validate(self, frame, input_sizes):
        fr = frame.returned
        if fr.type_info.name == ValueTypes.Tuple:
            def _pt(ggk):
                # Get metavars from return type
                values = []
                for it in ggk:
                    if it.type_info.name == ValueTypes.Tuple:
                        values += _pt(it.value)
                    else:
                        values.append(it)
                return values

            def _ot(oot):
                # Get metavars from passed in parameters
                values = []
                for otp in oot:
                    if isinstance(otp, list):
                        values += _ot(otp)
                    else:
                        values.append(otp)
                return values

            returned  = ([xx.type_info.size for xx in _pt(fr.value)])
            evaluated = self._incontext_eval(frame, _ot(self.outputs))

            if not all(value_cmp(a, b) for a, b in zip(returned, evaluated)):
                msg =  'Output size of meta parameter "{}" is wrong:\n'.format(self.outputs)
                msg += '  Received:\t{}\n'.format(returned)
                msg += '  Expected:\t{}'.format(evaluated)
                _raise(RET.BAD_META, msg)

        else:
            expected = fr.type_info.size
            evaluated = self._incontext_eval(frame, self.outputs)
            if evaluated and not value_cmp(expected, evaluated[0]):
                msg =  'Output size of meta parameter "{}" is wrong:\n'.format(self.outputs[0])
                msg += '  Received:\t{}\n'.format(expected)
                msg += '  Expected:\t{}'.format(evaluated[0].value)
                _raise(RET.BAD_META, msg)


class MetavarProblem(object):
    ''' Abstractions got leaky  '''
    def __init__(self, input_defined, output_defined):
        self.input_defined = input_defined
        self.output_defined = output_defined

    def solve(self, context):
        solution = MetavarSolution(self.output_defined, context)
        return solution


class StackFrame(object):
    ''' Stack frame object 
    
        `locals`           - Dictionary of local variables
        `returned`         - Value returned in the frame via `return`
        `backtrace_name`   - Name to show in backtrace
    '''
    def __init__(self, locals_, backtrace_name=None):
        self.locals = locals_
        self.returned = None
        self.backtrace_name = backtrace_name
        self.curnode = None

    def __repr__(self):
        return repr(self.locals)


class ProgramState(object):
    ''' 

    '''
    @property
    def frame(self):
        return self.frames[-1]

    def push_frame(self):
        curlocals = Scope('root')
        if self.frames:
            newitems = self.frame.locals.items.copy()
            curlocals.items = newitems
        self.frames.append(StackFrame(curlocals))

    def pop_frame(self):
        return self.frames.pop()

    def __init__(self):
        self.control_stack = []
        self.frames = []


class StringProvider(object):
    ''' Helper object used for replacing values via regular expressions.

        The `re.sub` family style functions can replace found occurences either with a
        string value or a callable. In the `printf` case we want to replace N format identifiers
        with N input values, so rather then matching each specifier seprately passing a single string,
        we match all of them and provide this callable which provides one string from list at a time.
    '''
    def __init__(self, strings):
        self.strings = iter(strings)

    def __call__(self, *args, **kwargs):
        return next(self.strings)


class Run(PipelineStage):
    Builtins = {
       'printf': Value(TypeInfo(ValueTypes.Builtin, const=True), builtin__print),
       'dumpvar': Value(TypeInfo(ValueTypes.Builtin, const=True), builtin__dumpvar),
       'dumpvarsorted': Value(TypeInfo(ValueTypes.Builtin, const=True), builtin__dumpvarsorted),
       'backtrace': Value(TypeInfo(ValueTypes.Builtin, const=True), builtin__backtrace),
       'UInt': Value(TypeInfo(ValueTypes.Builtin, const=True), builtin__UInt)
    }

    def _backtrace(self):
        print('Backtrace of {} frame(s)'.format(len(self.state.frames)))
        def get_pos(n):
            pos = (0, 0)
            def _drill_down_first(n):
                xx = n.children if isinstance(n, Expression) else n
                for c in n:
                    if isinstance(c, Token):
                        return c.position
                    elif isinstance(c, Expression):
                        f = _drill_down_first(c.children)
                        if f:
                            return f
                return None
            if n:
                p = _drill_down_first(n)
                if p:
                    pos = p
            return pos
        for i, f in enumerate(self.state.frames):
            pos = get_pos(f.curnode)
            if isinstance(pos[1], tuple):
                pos = (pos[0], pos[1][0])
            msg = '* {}: {} at line {} character {}'
            print(msg.format(i, f.backtrace_name, pos[0] + 1, pos[1]))

    def __init__(self, **config):
        self.config = config
        self.state = ProgramState()
        self.state.push_frame()
        self.state.frame.backtrace_name = 'program'

        for bname, bimpl in self.Builtins.items():
            self.locals[bname] = bimpl

    @property
    def control_stack(self):
        return self.state.control_stack

    @property
    def locals(self):
        return self.state.frame.locals

    def _clean_eval(self, node):
        ''' Evaluate node in a new stack frame and pop it afterwards so that
            variables from the evaluted scope do not bleed into current frame's locals
        '''
        self.state.push_frame()
        r = self._eval(node)
        self.state.pop_frame()
        return r

    def _resolve_load(self, node, create_intermediary=False):
        ''' Resolve `Load` operation from `Resovable` items

        '''
        assert check_ast(node, astnode.Load)

        scopes = []
        xx, yy = node, node
        while check_ast(xx, astnode.Load):
            # First right node holds the outermost name of the function
            # and we don't want to create intermediary scope for that
            if xx != yy:
                scopes.append(xx.right)
            yy = xx
            xx = xx.left
        scopes = names_to_strings([yy.left] + list(reversed(scopes)))
        last_scope = self.locals
        for scope in scopes:
            if scope in last_scope:
                last_scope = last_scope[scope]
            elif create_intermediary:
                last_scope[scope] = last_scope = Scope(scope)
            else:
                _raise(RET.NOT_FOUND, "Name not found: {}".format(str(scope)))
        if isinstance(last_scope, Value) and last_scope.type_info.name == ValueTypes.Type:
            # Descend into structure scope
            last_scope = last_scope.value
        return last_scope

    def _eval(self, node):
        ''' Evaluate `node` and mutate current `self.state` 

            Each branch here is self-contained unit handling a specific case, order of
            execution does not matter, and each case may opt-in to return some value.

            Node may refer to sub-classes of `ast.AstNode` or 
        '''

        self.state.frame.curnode = node

        if check_ast(node, astnode.EnumerationDefinition):
            ''' Define enumeration and put all items into current scope

            '''
            fields = node.fields
            typename = node.name.children[0].char
            value = Value(TypeInfo(ValueTypes.Enum, const=True, sub=typename), typename)
            self.locals[typename] = value

            for i, f in enumerate(fields):
                self.locals[nest(f, 0).char] = Value(
                        TypeInfo(ValueTypes.Enum, const=True, sub=typename), 
                        i
                )

            return NullResult

        elif check_ast(node, astnode.TypeDefinition):
            ''' Define structure in the current scope

            '''
            type_name = node.type_name
            fields = node.fields
            var = Value(TypeInfo(ValueTypes.Type, const=True, sub=type_name), fields)
            self.locals[type_name] = var
            return var

        elif check_expr(node, Expressions.Name) or isinstance(node, str):
            ''' Evaluate `name` expressions and raw strings as variable names

            '''
            try:
                return self.locals[node]
            except KeyError:
                name = node
                if check_expr(name, Expressions.Name):
                    name = name.children[0].char
                _raise(RET.NOT_FOUND, 'Item {} not found in current scope'.format(node))

        elif check_ast(node, astnode.Return):
            ''' Return statement evaluates the return node and puts the result
                into the current stack frame. Fail if we're not in a function.

            '''
            if len(self.state.frames) < 2:
                _raise(RET.INVALID_RETURN, 'Return from a non-function')
            if node.children:
                self.state.frame.returned = self._eval(node.children[0])
            return ReturnResult

        elif check_ast(node, [astnode.Equal, astnode.NotEqual, astnode.LessThan, 
                              astnode.LessThanOrEqual, astnode.GreaterThan, 
                              astnode.GraterThanOrEqual, astnode.And, astnode.Or]):
            ''' Comparison / bool operators

            '''
            l, r = self._eval(node.left), self._eval(node.right)
            lv, rv = l.value, r.value

            result = None
            if check_ast(node, astnode.Equal):
                result = value_cmp(l, r)
            elif check_ast(node, astnode.NotEqual):
                result = not value_cmp(l, r)
            elif check_ast(node, astnode.LessThan):
                result = value_lt(l, r)
            elif check_ast(node, astnode.LessThanOrEqual):
                result = value_lt(l, r) or value_cmp(l, r)
            elif check_ast(node, astnode.GreaterThan):
                result = value_lt(r, l)
            elif check_ast(node, astnode.GraterThanOrEqual):
                result = value_cmp(l, r) or value_lt(r, l)
            elif check_ast(node, astnode.And):
                result = value_logic(l, r, 'and')
            elif check_ast(node, astnode.Or):
                result = value_logic(l, r, 'or')
            else:
                _raise(RET.BAD_OPERATOR, 'Bad operator: {}'.format(str(node)))

            return Value(TypeInfo(ValueTypes.Bool), result)

        elif check_ast(node, astnode.Load):
            ''' Resolve load information and return the resolved node

            '''
            resolved = self._resolve_load(node, create_intermediary=False)
            if isinstance(resolved, Value) and resolved.type_info.name == ValueTypes.Type:
                resolved = resolved.value
            return resolved[node.right]

        elif check_ast(node, astnode.String):
            ''' Create bitstring

            '''
            return Bitstring(node.children[0])

        elif isinstance(node, int):
            ''' ints to Armozeen integers

            '''
            return Value(TypeInfo(ValueTypes.Integer), node)

        elif check_ast(node, astnode.Number):
            ''' Numbers to Armozeen Integers

            '''
            val = int(node.children[0].char)
            return Value(TypeInfo(ValueTypes.Integer), val)

        elif check_ast(node, astnode.FunctionCall):
            ''' Function calls
                --------------

                Order of operations:
                1) Validate that cardinality of input array is the same as that of
                   function parameters array
                2) Validate that input array and parameters array has the same types
                3) Find any meta-parameters and extract problem/solution variables
                4) Create a new stack frame and push input paramers directly as
                   local variables into the frame
                5) Evaluate the function block
                6) Find solution to meta variable problem from step (3)
            '''
            func = self._eval(node.name)

            # Special case - builtin function
            if isinstance(func, Value) and func.type_info.name == ValueTypes.Builtin:
                return func.value([self._eval(np) for np in node.parameters.children], self)

            # Special case - array access
            if isinstance(func, Array):
                # Write Case
                if len(node.children[1]) > 1:
                    index = self._clean_eval(node.children[1][0])
                    value = self._clean_eval(node.children[1][1])

                    if match_types_strict(func.items[index.value].type_info, value.type_info):
                        func.items[index.value].value = value.value
                    else:
                        _raise(RET.ARG_WRONG_TYPE, 
                               'Invalid types in array assignment "{}" and "{}"'.format(
                                   _format_typeinfo(func.items[index.value].type_info), 
                                   _format_typeinfo(value.type_info)))

                    return NullResult
                # Read case
                else:
                    index = self._clean_eval(node.children[1][0])
                    return func.items[index.value]

            mvp, in_values, value_inputs, rt_meta = None, {}, {}, []
            if not func.is_bracket:

                dp = node.parameters.children
                p = func.children[2][0].children

                if p != dp:
                    length_match = len(p) == len(dp)

                    if not length_match:
                        _raise(ARG_WRONG_NUMBER, 'invalid number of parameters for' + str(func.name))

                    dp = [self._eval(pp) for pp in dp]
                    value_inputs = []

                    def _find_metaparams_ex(nodes):
                        ''' Find meta parameters in return values, eg:

                                ((bits(N), bits(M)), bits(N+M)) IdentityAndCompose(bits(N) a, bits(M) b)
                                    return ((a, b), a:b)

                            This will yield the following nested list structure:

                                [['N', 'M'], ast_node: N + M]

                            Since we can reference meta variables in arbitrary nested structures
                            we can support arbitrary nested return tuples.
                        '''
                        if not isinstance(nodes, list):
                            nodes = [nodes]
                        r = []
                        for node in nodes:
                            if check_ast(node, astnode.Tuple):
                                # recurse into tuple 
                                r += [_find_metaparams_ex(node.children)]
                            if check_expr(node, Expressions.Bits):
                                if check_expr(node.children[0], Expressions.Name):
                                    r.append(node.children[0].children[0].char)
                                else:
                                    # What else could be valid here?
                                    if 'binary_node' in node.children[0].type:
                                       r.append(node.children[0])
                                    else:
                                       r.append(node.children[0])
                        return r

                    def check_param_type(p, t):
                        if check_expr(p, Expressions.Name):
                            return t.type_info.name == ValueTypes.Struct
                        if p.type == t.type_info.name:
                            return True
                        if check_expr(p, Expressions.TypeName):
                            return p.children[0].char == t.type_info.name
                        return False

                    for i in range(len(p)):
                        typ = p[i].children[0].children[0]
                        if not check_param_type(typ, dp[i]):
                            _raise(RET.ARG_WRONG_TYPE, 'Input argument does not match: {} != {}'.format(
                                str(typ), str(dp[i])    
                            ))

                        if check_expr(typ, Expressions.Bits):
                            if check_expr(typ.children[0], Expressions.Name):
                                value_inputs.append((i, typ.children[0].children[0].char))

                    in_values = dict([(iv[0], dp[iv[0]].type_info.size) for iv in value_inputs])
                    rt_meta = _find_metaparams_ex(func.children[0])

                    value_inputs = dict(map(swap, value_inputs))
                    mvp = MetavarProblem(value_inputs, rt_meta)

                # TODO: Implement bracket function invocation

                self.state.push_frame()
                self.state.frame.backtrace_name = func.name.children[0].char

                # Push parameters as local variables in the functions frame
                for inp in zip(dp, p):
                    inv = inp[0].clone()
                    self.locals[nest(inp[1], 0, 1, 0).char] = inv

                # Same for meta parameters
                for n, pos in value_inputs.items():
                    inv = self._eval(in_values[pos]).clone()
                    inv.type_info.const = True
                    self.locals[n] = inv
                
                self._eval(func.impl)
                frame = self.state.pop_frame()

                if mvp:
                    mvp.solve(self).validate(frame, dict(in_values))

                return frame.returned or NullResult

        elif check_ast(node, astnode.Tuple):
            ''' Create a tuple type
            
            '''
            ev = [self._eval(n) for n in node.children]
            return Value(TypeInfo(ValueTypes.Tuple, len(node.children), const=True), ev)

        elif check_ast(node, astnode.Concat):
            ''' Bitstring concatenation

                TODO: Validate both side are really Bitstrings
            '''
            l, r = self._eval(node.left), self._eval(node.right)
            lv, rv = l.value, r.value
            return Bitstring(lv + rv)

        elif check_ast(node, [astnode.Add, astnode.Sub, astnode.Mul, astnode.Mod, astnode.Xor, 
                              astnode.Div, astnode.Power, astnode.BitshiftLeft, astnode.BitshiftRight]):
            ''' Number operations

            '''
            l, r = self._eval(node.left), self._eval(node.right)
            l, r = Coercion.coerce(l, r)

            if l.type_info.name not in [ValueTypes.Integer, ValueTypes.Real]:
                _raise(RET.ARG_WRONG_TYPE, 
                       'Type {} is not integer / real'.format(_format_typeinfo(l.type_info)))

            lv, rv = l.value, r.value
            result = 0

            if check_ast(node, astnode.Add):
                result = lv + rv
            elif check_ast(node, astnode.Sub):
                result = lv - rv
            elif check_ast(node, astnode.Mul):
                result = lv * rv
            elif check_ast(node, astnode.Mod):
                result = lv % rv
            elif check_ast(node, astnode.Xor):
                result = lv ^ rv
            elif check_ast(node, astnode.Div):
                result = lv // rv
            elif check_ast(node, astnode.Power):
                result = lv ** rv
            elif check_ast(node, astnode.BitshiftLeft):
                result = lv << rv
            elif check_ast(node, astnode.BitshiftRight):
                result = lv >> rv
            else:
                _raise(RET.BAD_OPERATOR, 'Invalid operator'.format(str(node)))

            return Value(TypeInfo(l.type_info.name), result)

        elif check_ast(node, astnode.Bitselect):
            ''' Bitselect (read) operation

            '''
            v = self._clean_eval(node.children[0])
            selected = []
            x = list(reversed(v.value))
            bse = bitselect(node, self)
            for u, l in bse:
                l = l.value
                u = u.value
                if l != u:
                    selected += x[l : u+1]
                else:
                    selected.append(x[l])
            return Bitstring("".join(reversed(selected))) if v else NullResult

        elif check_ast(node, astnode.Assignment):
            ''' Assigment operation variants
                ----------------------------

                We need to handle number of distinct cases depending on what node
                is in LHS:
            
                * expr<Name>
                  This can be either assignment or definition with type inferred from
                  RHS. If this is an assignment we need to check that types match and that
                  we're not assigning to a `const` variable.

                * expr<typed_variable>
                  Declare a variable of specific type and assign a value to it. 

                * expr<Substruct>
                  Substruct into variables in current scope e.g.: <A, B> = '10';
                  
                * ast<Load>
                  Assign into result of a load operation (scope, struct)

                * ast<Tuple>
                  Assign multiple variables from a tuple

                * ast<Destructure>
                  Destructure a struct, similar to `Substruct` e.g.: someStruct.<A, B> = '10';

                * ast<Bitselect>
                  Assign to a range(s) of bits, sizes must match
            '''
            lhs_n, var = node.children[0], None
            if check_expr(lhs_n, Expressions.Name):
                lhs_s, ll = nest(node, 0, 0).char, None
                if lhs_s in self.locals:
                    ll = self.locals[lhs_s]
                    if ll.type_info.const:
                        _raise(RET.CONSTANT, 'Value {} is constant!'.format(lhs_s))
                var = self._eval(node.children[1])
                if var != NullResult:
                    if ll and not match_types_strict(ll.type_info, var.type_info):
                        msg = 'Invalid assigment of value of type {} to variable of type {}'
                        _raise(RET.ARG_WRONG_TYPE, msg.format(ll.type_info, var.type_info))
                    self.locals[lhs_s] = var
                else:
                    _raise(RET.RUNTIME_ERROR, 'Invalid name assignment {!s}'.format(node.children[1]))

            elif check_ast(lhs_n, astnode.Load):
                scope = self._resolve_load(lhs_n, create_intermediary=True)
                #r = self._eval(node.children[1])
                var = self._eval(node.children[1])
                if var != NullResult:
                    #var = Value(r.type_info, r)
                    k = nest(lhs_n, 1, 0).char
                    if isinstance(scope, StructType):
                        if match_types_strict(scope[k].type_info, var.type_info):
                            scope[k].value = var.value
                        else:
                            msg = 'Invalid assigment of value of type {} to variable of type {}'
                            _raise(RET.ARG_WRONG_TYPE, msg.format(scope[k].type_info, var.type_info))
                    else:
                        if k in scope:
                            if match_types_strict(scope[k].type_info, var.type_info):
                                scope[k] = var
                            else:
                                msg = 'Invalid assigment of value of type {} to variable of type {}'
                                _raise(RET.ARG_WRONG_TYPE, msg.format(scope[k].type_info, var.type_info))
                        else:
                            scope[k] = var
                else:
                    _raise(RET.RUNTIME_ERROR, 'Invalid load assignment {!s}'.format(node.children[1]))

            elif check_ast(lhs_n, astnode.Tuple):
                var = self._eval(node.children[1])
                def _rec(a, b):
                    a_tuple, b_tuple = check_ast(a, astnode.Tuple), b.type_info.name == ValueTypes.Tuple
                    if a_tuple != b_tuple:
                        _raise(RET.ARG_WRONG_TYPE, 'Invalid tuple unrwapping')
                    la = len(a.children)
                    lb = b.type_info.size if b_tuple else 1
                    if la != lb:
                        _raise(RET.ARG_WRONG_NUMBER, 
                               'Invalid number of sub-items {} and {}'.format(la, lb))
                    ret = []
                    if a_tuple:
                        for ir, it in enumerate(a):
                            ret += _rec(it, b.value[ir])
                    else:
                        ret.append((a, b))
                    return ret
                for varname, varval in _rec(node.children[0], var):
                    if varname in self.locals:
                        if match_types_strict(self.locals[varname].type_info, varval.type_info):
                            self.locals[varname] = varval
                        else:
                            msg = 'Invalid assigment of value of type {} to variable of type {}'
                            _raise(RET.ARG_WRONG_TYPE, msg.format(scope[k].type_info, var.type_info))
                    else:
                        self.locals[varname] = varval

            elif check_expr(lhs_n, 'typed_variable'):
                lhs_s = lhs_n.children[1].children[0].char
                var = self._eval(node.children[1])
                if lhs_s in self.locals:
                    _raise(RET.ALREADY_DECLARED, 'Variable {} already declared'.format(lhs_s))
                if var != NullResult:
                    td = self._eval(lhs_n)
                    if not match_types_strict(td, var.type_info, relax_const=True):
                        msg = 'Invalid assigment of value of type {} to variable of type {}'
                        _raise(RET.ARG_WRONG_TYPE, 
                               msg.format(_format_typeinfo(var.type_info), _format_typeinfo(td)))
                    self.locals[lhs_s].value = var.value
                else:
                    _raise(RET.RUNTIME_ERROR, 'Invalid typed assignment {!s}'.format(node.children[1]))

            elif check_ast(lhs_n, astnode.Destructure):
                v = self._clean_eval(lhs_n.children[0])
                if not v.type_info.name == ValueTypes.Type:
                    raise ValueError('Invalid destructure target ' + str(v))
                v = v.value
                var = self._eval(node.children[1])
                elts = [v[elt] for elt in lhs_n.children[1]]
                szl = sum(elt.type_info.size for elt in elts)
                szr = var.type_info.size
                vv = list(var.value)
                if szl != szr:
                    _raise(RET.DIFF_BIT_SIZE, 'Invalid destructuing source/target size')
                for elt in elts:
                    sz = elt.type_info.size
                    elt.value = "".join(vv[:sz])
                    vv = vv[sz:]

            elif check_expr(lhs_n, 'substruct'):
                var = self._eval(node.right)
                elts = [self.locals[elt.children[0].char] for elt in lhs_n.children]
                szl = sum(elt.type_info.size for elt in elts)
                szr = var.type_info.size
                if szl != szr:
                    _raise(RET.DIFF_BIT_SIZE, 'Invalid destructuing source/target size')
                vv = list(var.value)
                for elt in elts:
                    sz = elt.type_info.size
                    elt.value = "".join(vv[:sz])
                    vv = vv[sz:]

            elif check_ast(lhs_n, astnode.Bitselect):
                value = self._clean_eval(node.right)
                v = self._clean_eval(nest(node, 0, 0))
                lsize = len(v.value)
                rsize = len(value.value)
                vv = list(v.value)

                off = 0
                for u, l in bitselect(lhs_n, self):
                    l = l.value
                    u = u.value

                    diff = u - l

                    if diff != 0:
                        vv[lsize - (u+1):lsize - l] = value.value[off:off+diff+1]
                    else:
                        vv[lsize-u] = value.value[off]

                    off += l if l == u else diff

                if v.type_info.const:
                    _raise(RET.CONSTANT, 'Bitselect assignment to a constant value {}'.format(lhs_n))
                if v.type_info.size != len(vv):
                    msg = 'Bitselect assignment with RHS ({}) of different size than LHS ({})'
                    _raise(RET.DIFF_BIT_SIZE, msg.format(len(vv), v.type_info.size, ))

                v.value = "".join(vv)
                
            else:
                _raise(RET.RUNTIME_ERROR, 'Invalid assignment {!s}'.format(node.children[1]))

            return var or NullResult

        elif check_ast(node, astnode.Destructure):
            v = self._clean_eval(node.children[0])
            if not v.type_info.name == ValueTypes.Type:
                _raise(RET.ARG_WRONG_TYPE, 
                       'Invalid destructure target ' + _format_typeinfo(v.type_info))
            v = v.value
            elts = [v[elt].value for elt in node.children[1]]
            return Bitstring("".join(elts))
           
        elif check_ast(node, astnode.FunctionDefinition):
            ''' Define function by placing the definition node
                into the locals under the functions name
            '''
            cn = node.children[1]
            if check_ast(cn, astnode.Load):
                last_scope = self._resolve_load(cn, create_intermediary=True)
                last_scope[nest(cn, 1, 0).char] = node
            else:
                self.locals[nest(node, 1, 0).char] = node
            return NullResult

        elif check_ast(node, astnode.ForLoop):
            ''' For-To loop implementation

            '''

            name = node.children[0][0].left.children[0].char
            iterable = self._clean_eval(node.children[0])
            target = self._clean_eval(node.children[1])

            self.locals[name] = iterable

            while iterable.value < target.value:
                if self._eval(node.block) == ReturnResult:
                    break
                iterable.value += 1

        elif check_expr(node, 'typed_variable'):
            ''' Typed variable declaration

            '''
            t, v = None, None
            if node.children[1] in self.locals:
                _raise(RET.ALREADY_DECLARED, 
                       'Variable {} already declared'.format(node.children[1]))
            ts = node.children[0]
            if check_expr(ts, 'bits'):
                real_size = self._eval(ts.children[0])
                t = TypeInfo(ValueTypes.Bits, real_size.value)
            elif check_expr(ts, 'typename'):
                t = TypeInfo(ts.children[0].char)
            elif check_expr(ts, 'typeexp'):
                const = ts.const
                name = ts.name
                if check_expr(name, 'typename'):
                    t = TypeInfo(ts.name.children[0].char, const=const)
                elif check_expr(name, 'name'):
                    if name.children[0].char == 'bool':
                        t = TypeInfo(ValueTypes.Bool, const=const)
                else:
                    _raise(RET.RUNTIME_ERROR, 'Invalid typed variable name' + str(name))

            elif check_expr(ts, 'name'):
                nn = ts.children[0].char
                if nn == 'bool':
                    t = TypeInfo(ValueTypes.Bool)
                else:
                    subitem = self.locals[nn]
                    t = TypeInfo(
                            ValueTypes.Type, 
                            sub=subitem.type_info.subtype
                    )
                    names = [nest(si, 1, 0).char for si in subitem.value]
                    v = StructType(ts.children[0].char, [
                        (n, Value(self._clean_eval(x), None)) for x, n in zip(subitem.value, names)
                    ])

            self.locals[node.children[1]] = Value(t, v)
            return t

        elif check_expr(node, Expressions.Paren):
            if len(node.children) > 1:
                _raise(RET.RUNTIME_ERROR, 'Invalid parenthesis expression')
            return self._eval(node.children[0])


        elif check_expr(node, 'sstring'):
            ''' This is a double-quoted string

            '''
            return node.children[0]

        elif check_ast(node, astnode.CaseOf):
            ''' Evaluate a case ... of statement

            '''
            if isinstance(node.case, str):
                val, rv = Bitstring(node.case), None
            else:
                val, rv = self._clean_eval(node.case), None
            for branch in node.branches:
                brval = self._clean_eval(branch.children[0])
                if value_cmp(val, brval):
                    rv = self._eval(branch.children[1])
                    break
            if not rv:
                if node.otherwise:
                    rv = self._eval(node.otherwise.children[0])
            return rv

        elif check_expr(node, 'real'):
            floatval = node.children[0].children[0].char + '.' + node.children[2].children[0].char
            return Value(TypeInfo(ValueTypes.Real), float(floatval))

        elif check_expr(node, Expressions.Block):
            ''' Evaluate each children

            '''
            return self._eval(node.children)

        elif check_ast(node, astnode.If):
            ''' Evaluate if/elsif/else branches and execute block associated with
                the one that matched
            '''
            first_stage = [node.if_branch] + node.elsif_branches
            for branch in first_stage:
                r = self._eval(branch[0][0])
                if r.value:
                    return self._eval(branch[1])
            if node.else_branch:
                return self._eval(node.else_branch[1])

        elif check_ast(node, astnode.ArrayDefinition):

            name = node.name
            array_size = 0

            if name in self.locals:
                _raise(RET.ALREADY_DECLARED, 'Name {} already declared'.format(name))

            if check_ast(node.size_spec[-1], astnode.Number):
                array_size = int(node.size_spec[-1].children[0].char)
            else:
                _raise(RET.CONSTANT, 'Array size must be constant')

            # TODO: This code should go into a separate function and
            #       handle generic instantion of Armozeen types
            type_, items = None, []
            if check_expr(node.item_type, 'bits'):
                s = node.item_type.children[0]
                if check_ast(s, astnode.Number):
                    type_ = TypeInfo(ValueTypes.Bits, int(s.children[0].char))
                    items = [Bitstring('0'*type_.size) for _ in range(array_size)]
                else:
                    _raise(RET.CONSTANT, 'Array element size must be contant')
            else:
                _raise(RET.ARG_WRONG_TYPE, 'Invalid array type argument')

            var = Array(items, type_)
            self.locals[name] = var

            return var

        elif isinstance(node, list):
            ''' Evaluate each element of the list unless we've got a return value

            '''
            last, prev = None, None
            for nc in node:
                last = self._eval(nc)
                if self.state.frame.returned is not None:
                    return prev
                prev = last
            return last

        else:
            ''' Node unaccounted for

            '''
            _raise(RET.RUNTIME_ERROR, 'Unknown AST node')

        return NullResult


    def run(self, items):
        try:
            for item in items:
                x = self._eval(item)
                if x is ReturnResult:
                    break
        except ArmozeenRuntimeException as are:
            print('\nAn error was encountered: {}\n\n{}\n'.format(are.type_, are.message))
            self._backtrace()
        except Exception as exc:
            print('Fatal runtime error while processing:\n')
            print(self.state.frame.curnode)
            print('')
            import traceback
            traceback.print_exc()

        return self.state

