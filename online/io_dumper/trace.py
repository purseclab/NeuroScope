import pickle
import cxxfilt
import angr
from angr_utils import *


class CallSite:

    def __init__(self, addr, callee_addr, callee_name):
        self.addr = addr

        self.callee_addr = callee_addr
        self.callee_name = cxxfilt.demangle(callee_name)

    def __repr__(self) -> str:
        return f"CallSite({hex(self.addr)}, {self.callee_name}, {hex(self.callee_addr)})"


class FunctionNode:

    def __init__(self, name, addr, t_id, parent):
        self.name = cxxfilt.demangle(name)
        self.addr = addr
        self.t_id = t_id

        self.children = []
        self.parent = parent

        self.callsite = None

    def __repr__(self) -> str:
        return f"FunctionNode({self.name}, {hex(self.addr)}, {self.t_id})"

    def is_parent(self, node):
        if self is node:
            return True

        parent = node.parent
        while parent is not None:
            if parent is self:
                return True
            parent = parent.parent
        return False


def recover_trace(traces):
    '''
    Recover a function call tree from a trace
    '''
    root = FunctionNode("main", 0, 0, None)

    cur_node = root
    t_id = 0
    for id, (f_addr, f_name, direction) in enumerate(traces):
        # probably functions inlined
        if f_addr == cur_node.addr:
            continue

        t_id += 1
        if direction == 'up':
            child_node = FunctionNode(f_name, f_addr, t_id, cur_node)
            cur_node.children.append(child_node)
            cur_node = child_node
        elif direction == 'down':
            # we should go back to the parent or parent's parent (tail call)
            if f_addr == cur_node.parent.addr:
                cur_node = cur_node.parent
            elif f_addr == cur_node.parent.parent.addr:
                cur_node = cur_node.parent.parent
            else:
                print(id)
                print("cur_node: ", cur_node)
                print("parent: ", cur_node.parent)
                print("trace: ", (hex(f_addr), f_name, direction))
                assert False

            # break when we are back to the root
            if cur_node is None:
                print("STOP: hit the root")
                break
        else:
            assert False

    return root


def find_cand_op_node(node, cand_op, cand_op_node):
    '''
    Find the candidate operator nodes in the function call tree with `node` as the root
    '''
    for child in node.children:
        if child.addr in cand_op:
            cand_op_node.append(child)
        find_cand_op_node(child, cand_op, cand_op_node)


def identify_dispatcher(root, cand_op_node):
    '''
    dispatcher -> lowest common ancestor of all the candidate operator functions
    -> parent of all the candidate operator functions AND all of its children are not
    '''

    # assert root is the parent of all the candidate operator functions
    assert all([root.is_parent(node) for node in cand_op_node])

    def recursive(node, cand_op_node):
        legit_children = []
        for child in node.children:
            if all([child.is_parent(cand_op) for cand_op in cand_op_node]):
                legit_children.append(child)

        if len(legit_children) == 0:
            return node
        elif len(legit_children) == 1:
            return recursive(legit_children[0], cand_op_node)
        else:
            assert False

    return recursive(root, cand_op_node)


def identify_operator_entry(dispatcher, cand_op_node):
    '''
    Identify the entry function of the operator implementations,
    assuming that op_entry are the direct children of dispatcher
    '''
    op_entry = []
    for child in dispatcher.children:
        if any([child.is_parent(cand_op) for cand_op in cand_op_node]):
            op_entry.append(child)
    return op_entry


def prune_trace(dispatcher, op_entry, cand_op_node):
    '''
    prune the irrelevant part of the trace, only keep the paths:
        1. from dispatcher to op_entry
        2. from op_entry to candidate operator functions
    '''
    # 1
    new_children = []
    for child in dispatcher.children:
        if child in op_entry:
            new_children.append(child)
    dispatcher.children = new_children

    # 2
    to_prune_node = op_entry.copy()
    while to_prune_node:
        node = to_prune_node.pop()
        new_children = []
        for child in node.children:
            if any([child.is_parent(cand_op) for cand_op in cand_op_node]):
                to_prune_node.append(child)
                new_children.append(child)
        node.children = new_children


def parse_simarg(arg):
    '''
    arg: https://github.com/angr/angr/blob/fecb387de6d5581e4049a6ee42f2476ac9452005/angr/calling_conventions.py#L267
    return: (type, value)
    '''
    if isinstance(arg, angr.calling_conventions.SimRegArg):
        return ("reg", arg.reg_name)
    elif isinstance(arg, angr.calling_conventions.SimStackArg):
        return ("stack", arg.stack_offset)
    else:
        print("unsupported arg type: ", type(arg))
        assert False


def find_tracking_args(proj, op_entry):
    '''
    Find the function arguments of callee of each op_entry
    '''
    tracking_args = {}
    for op in op_entry:
        if len(op.children) == 1:
            op_func = proj.cfg.kb.functions[op.addr]
            callee_func = proj.cfg.kb.functions[op.children[0].addr]

            if callee_func.addr not in tracking_args:
                tracking_args[callee_func.addr] = [
                    parse_simarg(arg) for arg in callee_func.arguments
                ]
        elif len(op.children) == 0:
            # FIXME: function got inlined
            continue
        else:
            assert False

    with open("tracking_args.pkl", "wb") as f:
        pickle.dump(tracking_args, f)

    return tracking_args
