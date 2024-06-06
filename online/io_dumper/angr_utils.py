import angr


def get_last_inst_addr_in_blk(func, blk_addr):
    """
    get blk's last inst addr
    """
    blk = func.get_block(blk_addr)
    assert isinstance(blk, angr.block.Block)
    return blk.instruction_addrs[-1]


def get_func_boundary(func):
    """
    Return the min and max addr of the given func.

    Note that addr + size is not the max, because size = sum of block.size,
    and it is possible that random bytes in the func.
    e.g. glow_mnist_8_ARM_M4 0x101af

    Also note the max_addr is not the addr of last inst, but the next one.
    """
    min_addr = func.addr

    last_blk_addr = max(func.block_addrs_set)
    last_blk = func._local_blocks[last_blk_addr]
    max_addr = last_blk.addr + last_blk.size

    return min_addr, max_addr


def check_in_func(func, addr):
    """
    check if addr is in func
    """

    min_addr, max_addr = get_func_boundary(func)

    if addr >= min_addr and addr < max_addr:
        return True
    return False


def get_func_addr_from_addr(cfg, addr):
    """
    return the func addr where the addr is in
    """
    for f_addr, func in cfg.kb.functions.items():
        if check_in_func(func, addr):
            return f_addr


def get_func_from_addr(cfg, addr):
    return cfg.kb.functions[get_func_addr_from_addr(cfg, addr)]


def get_succ(proj, f_addr):
    """
    Get the addr of callees of function
    """
    return list(proj.cfg.kb.functions.callgraph.successors(f_addr))


def get_pred(proj, f_addr):
    """
    Get the addr of callers of function
    """
    return list(proj.cfg.kb.functions.callgraph.predecessors(f_addr))


def get_loop_depth(loop):
    """
    Get the maximum depth of the nested loop
    """
    if not loop:
        return 0
    return max([get_loop_depth(l.subloops) for l in loop]) + 1


def check_in_block(block, addr):
    """
    Check if addr is in the block
    """
    if addr >= block.addr and addr < block.addr + block.size:
        return True
    return False


def get_block_from_addr(proj, addr):
    """
    Get the BlockNode that contains the addr
    """
    func = get_func_from_addr(proj.cfg, addr)

    for block_node in func.nodes:
        if check_in_block(block_node, addr):
            return block_node

    return None


def get_next_addr_from_addr(proj, addr):
    """
    Get the next addr of the given addr
    """
    # block_node = get_block_from_addr(proj, addr)
    # assert block_node is not None

    # HACK: get the next addr by adding 4
    return addr + 4
