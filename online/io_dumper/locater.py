from collections import defaultdict
import pickle
import json

from angr_utils import *


def print_func_map_item(func_map, key):
    '''
    # FIXME: it is not the right way, we should use class
    '''
    f_addr = key
    f_range = func_map[key][0]
    f_name = func_map[key][1]
    callsites = func_map[key][2]

    print("f_addr: 0x%x, f_min: 0x%x, f_max: 0x%x, f_name: %s" %
          (f_addr, f_range[0], f_range[1], f_name))
    for callsite in callsites:
        print("callsite: 0x%x" % callsite)


def print_func_assembly(func):
    for bb in func.blocks:
        for inst in bb.capstone.insns:
            print(inst.mnemonic, inst.op_str)


def find_extra_call_sites(func):
    '''
    find triky call sites that angr fail to find

    pattern 1:
        pop {r4, lr}
        b.w callee
    '''
    callsites = []

    # pattern 1
    for bb in func.blocks:
        lr_flag = False
        for inst in bb.capstone.insns:
            if inst.mnemonic.startswith("pop") or inst.mnemonic.startswith(
                    "mov"):
                if 'lr' in inst.op_str:
                    lr_flag = True
                    continue

            if lr_flag and inst.mnemonic.startswith("b"):
                callsites.append(inst.address)

            if lr_flag:
                lr_flag = False

    return callsites


def is_simd_inst(mnemonic):
    '''
    Check if the instruction is SIMD instruction
    https://www.keil.com/pack/doc/CMSIS/Core/html/group__intrinsic__SIMD__gr.html
    '''
    if mnemonic.startswith("sm"):
        return True
    return False


def is_simd_func(func):
    '''
    Check if the SIMD instruction is used
    '''
    # SIMD
    for bb in func.blocks:
        for inst in bb.capstone.insns:
            if is_simd_inst(inst.mnemonic):
                return True
    return False


def identify_math_function(proj):
    '''
    Identify math function 
    '''
    math_funcs = set()

    for f_addr in proj.cfg.kb.functions:
        f = proj.cfg.kb.functions[f_addr]

        # simd
        if is_simd_func(f):
            math_funcs.add(f_addr)
            continue

        # many loops
        if f_addr in proj.loop_finder.loops_hierarchy and get_loop_depth(
                proj.loop_finder.loops_hierarchy[f_addr]) > 3:
            math_funcs.add(f_addr)
            continue

        # FIXME: math density

    proj.identified_math_funcs = list(math_funcs)


def has_math_func(proj, f_addr):
    '''
    Check if the function has math function as successor
        1. libc math function
        2. identified math function
    '''
    succ = get_succ(proj, f_addr)

    MATH_FUNC_NAME = [
        "tan", "tanf", "tanh", "tanhf", "exp", "expf", "log", "logf", "pow",
        "powf", "sqrt", "sqrtf"
    ]

    for s in succ:
        if s not in proj.cfg.kb.functions:
            continue
        # libc math function
        if proj.cfg.kb.functions[s].name in MATH_FUNC_NAME:
            return True
        # identified math function
        if s in proj.identified_math_funcs:
            return True

    return False


def should_be_skipped(func, callsites, cand_op):
    '''
    Check if the function should be skipped
    '''
    f_name = func.name
    f_addr = func.addr

    # libc functions
    if f_name == 'memcpy' or f_name == 'memset' or f_name == 'malloc' or f_name == 'free':
        return 1

    # FIXME: use addr
    # if f_addr in cand_op:
    # the idenfied operator: we dont need to trace into it
    if f_name == 'arm_convolve_s8' or f_name == "arm_fully_connected_s8":
        return 2

    # no call sites
    if len(callsites) == 0:
        return 3

    # others
    # other_list = [0x6000c951, 0x6000e28d]
    # if f_addr in other_list:
    #     return 4

    return 0


def preprocess(proj, cand_op, use_json=False):
    '''
    We need to store some key information in pkl, 
    since gdb's Python cannot import angr.

    # FIXME: use json
    func_map: f_addr -> ((start_addr, end_addr), name, [[call_sites, skipped]], should_be_skipped)
    `skipped` annotates if the callee in the corresponding call site should be skipped

    1. callsites 
    angr's callsites are not complete (e.g. some callsites are not identified),
    we complement the callsites with find_extra_call_sites()

    2. should_be_skipped
        0 -> shouldn't, 
        1 -> libc functions
        2 -> identified operators
        3 -> no call sites
        4 -> others
    '''
    func_map = {}
    for f_addr in proj.cfg.kb.functions:
        f = proj.cfg.kb.functions[f_addr]
        f_name = f.name

        start_addr, end_addr = get_func_boundary(f)

        # Remove the useless function stub recovered by angr
        if end_addr - start_addr <= 2:
            continue

        angr_callsites = [
            get_last_inst_addr_in_blk(f, addr)
            for addr in list(f.get_call_sites())
        ]
        extra_callsites = find_extra_call_sites(f)

        # FIXME: make sure the data structure is aligned with the how we use it in func_trace.py
        if use_json:
            callsites = [{
                'callsite_addr': cs,
                'callee_addr': None,
                'skip': False
            } for cs in (angr_callsites + extra_callsites)]
            skip_flag = should_be_skipped(f, callsites, cand_op)
            func_map[f_addr] = {
                "start_addr": start_addr,
                "end_addr": end_addr,
                "f_name": f_name,
                "callsites": callsites,
                "skip_flag": skip_flag
            }
        else:
            callsites = [[cs, False]
                         for cs in (angr_callsites + extra_callsites)]
            skip_flag = should_be_skipped(f, callsites, cand_op)
            func_map[f_addr] = ((start_addr, end_addr), f_name, callsites,
                                skip_flag)

    if use_json:
        with open("func_map.json", "w") as file:
            json.dump(func_map, file)
    else:
        # another pass to annotate the callsites
        for f_addr in func_map:
            for cs in func_map[f_addr][2]:
                callsite_addr = cs[0]
                # FIXME: not complete
                # callee_addr =

                cs[1] = func_map[cs[0]][3]

        with open("func_map.pkl", "wb") as file:
            pickle.dump(func_map, file)


def locate_op_cand(proj):
    LOOP_COUNT = 1

    identify_math_function(proj)

    cand_op_func = []
    for f_addr in proj.cfg.kb.functions:
        # nested loops
        loop_flag = False
        if f_addr in proj.loop_finder.loops_hierarchy:
            loop = proj.loop_finder.loops_hierarchy[f_addr]
            if get_loop_depth(loop) >= LOOP_COUNT:
                loop_flag = True

        # math func
        math_flag = has_math_func(proj, f_addr)

        if loop_flag and math_flag:
            if proj.cfg.kb.functions[f_addr].name != "main":
                cand_op_func.append(f_addr)

    return cand_op_func


def locate(proj):
    '''
    Locate the inference function and all the operator functions
    '''
    LOOP_COUNT = 2

    # look for candidate op_func
    cand_op_func = []
    for f_addr in proj.cfg.kb.functions:
        # nested loops
        loop_flag = False
        if f_addr in proj.loop_finder.loops_hierarchy:
            loop = proj.loop_finder.loops_hierarchy[f_addr]
            if get_loop_depth(loop) >= LOOP_COUNT:
                loop_flag = True

        # math func
        math_flag = has_math_func(proj, f_addr)

        if loop_flag or math_flag:
            cand_op_func.append(f_addr)

    # vote for "inference function"
    vote_dict = defaultdict(int)
    for f_addr in cand_op_func:
        preds = [
            get_func_addr_from_addr(proj.cfg, p)
            for p in get_pred(proj, f_addr)
        ]
        for p in preds:
            vote_dict[p] += 1

    # decide inference function
    max_count = max(vote_dict.values())
    infer_f_addr = [k for k, v in vote_dict.items() if v == max_count]

    # for debugging
    print("---op_func---")
    for f_addr in cand_op_func:
        print(proj.cfg.kb.functions[f_addr].name)
    print()
    print("---count_dict---")
    for k, v in vote_dict.items():
        print(proj.cfg.kb.functions[k].name, v)
    print()
    print("---infer_func---")
    for f_addr in infer_f_addr:
        print(proj.cfg.kb.functions[f_addr].name)

    # there should be only one infer
    if len(infer_f_addr) == 1:
        infer_f_addr = infer_f_addr[0]
    else:
        assert (False)

    # retrieve all the op_func
    op_func_addr = get_succ(proj, infer_f_addr)

    return infer_f_addr, op_func_addr
