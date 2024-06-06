import gdb
import os
import sys
import re
from collections import deque, defaultdict

TARGET = "imxrt1050-tflm"
DIR = "/home/ruoyu/workspace/dnd2/locater/"


def parse_ldr_instruction(instruction_str):
    # Define a regular expression pattern to match the "ldrsh.w" instruction
    pattern = re.compile(r"(\S+)\s+(\S+),\s*\[([^,]+), #(-?\d+)\]")

    # Use the pattern to match the instruction
    match = pattern.match(instruction_str)

    if match:
        # Extract the instruction, register, base register, and offset
        instruction = match.group(1)
        register = match.group(2)
        base_register = match.group(3)
        offset = int(match.group(4))
        return instruction, register, base_register, offset
    else:
        return None, None, None, None


class Base:
    def __init__(self):
        self.DEBUG = False

    def setup_gdb(self):
        # make sure the breakpoint is fired
        gdb.execute("set confirm off")
        gdb.execute("set breakpoint pending on")

        # disable the behavior that re-execute last command when typing blank line
        gdb.execute("dont-repeat")

        gdb.execute("set print pretty on")

        # set logging
        gdb.execute(
            "set logging file /home/ruoyu/workspace/dnd2/locater/func_trace.log"
        )
        gdb.execute("set logging on")

        # make sure we step into the function without symbol info
        gdb.execute("set step-mode on")

        # FIXME: this might cause issue
        # gdb.execute('set scheduler-locking on')

    def log(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")

    def debug(self, msg):
        if self.DEBUG:
            self.log(msg)

    def setup_log(self):
        self.log_file = open(DIR + "func_trace.log", "w")

    def bp(self, addrs, info=""):
        for addr in addrs:
            self.log("\nbp: " + info + " " + str(hex(addr)))
            gdb.execute("b *{}".format(hex(addr)))

    def _addr2num(self, addr):
        try:
            return int(addr)  # Python 3
        except:
            return long(addr)  # Python 2

    def dump(self):
        # TODO: dump the whole memory
        pass

    def read_mem(self, addr, size):
        return self._addr2num(gdb.selected_inferior().read_memory(addr, size))

    def read_stack(self, offset, size):
        return self.read_mem(self.read_reg("sp") + offset, size)

    def read_reg(self, reg):
        return self._addr2num(gdb.newest_frame().read_register(reg))

    def get_pc(self):
        # FIXME: arm-thumb
        return self.read_reg("pc") + 1

    def get_lr(self):
        # FIXME: should we plus 1?
        return self.read_reg("lr")

    def get_stack_depth(self):
        """
        FIXME: this might be too slow
        """
        # return gdb.newest_frame().level()
        depth = 1
        frame = gdb.newest_frame()
        while frame is not None:
            frame = frame.older()
            depth += 1
        return depth

    def is_outermost_frame(self):
        return gdb.newest_frame().older() is None

    def finish(self):
        """
        Continue running until just after function in the selected stack frame returns.
        """
        self.debug("finish\n")
        gdb.execute("finish")

    def until(self):
        self.debug("\nuntil")
        gdb.execute("until")

    def remove_bp(self, addr=None):
        if addr is None:
            self.log("\nremove all bp")
            gdb.execute("d")  # remove all breakpoints
        else:
            assert False

    def step_inst(self):
        gdb.execute("stepi")
        self.debug(
            "\nstep_inst. new pc: "
            + str(hex(self.get_pc()))
            + " new stack depth: "
            + str(self.get_stack_depth())
            + " new function name: "
            + self.get_f_name()
        )

    def step_n_inst(self, n):
        self.debug("\nstep_n_inst")
        gdb.execute("stepi " + str(n))

    def step_inst_over(self):
        """
        step a instruction (over a function call)
        """
        self.debug("\nstep_inst_over")
        gdb.execute("nexti")

    def step_continue(self):
        self.debug("\nstep_continue")
        gdb.execute("c")

    def log_exception(self, e):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        self.log(
            "\nException: "
            + str(exc_type)
            + " "
            + str(fname)
            + " "
            + str(exc_tb.tb_lineno)
            + " "
            + str(e)
        )

    def log_trace(self, current_pc, f_name, direction):
        if direction == "down":
            self.log(direction + " " + str(hex(current_pc)) + " " + f_name)
        elif direction == "up":
            self.log(direction + " " + str(hex(current_pc)) + " " + f_name)
        else:
            assert False

    def goto_entry(self):
        """
        Skip the booting process
        """
        if TARGET == "imxrt1050-tflm":
            gdb.execute("b *0x60005b22")
            gdb.execute("c")

            # gdb.execute("b main.cpp:32")
            # gdb.execute('c')

            # gdb.execute('b tflite::MicroInterpreter::Invoke()')
            # gdb.execute('c')
        elif TARGET == "imxrt1050-glow":
            gdb.execute("b *0x600034de")
            gdb.execute("c")

        else:
            raise Exception("Unknown target")

    def get_f_name(self):
        name = gdb.newest_frame().name()
        if name is None:
            return "None"
        return name


class SampleTracer(gdb.Command, Base):
    def __init__(self):
        super(SampleTracer, self).__init__("sample-trace", gdb.COMMAND_USER)

        self.setup_gdb()

        self.DEBUG = False
        self.setup_log()

        # remove all breakpoints
        self.remove_bp()

        self.goto_entry()

        self.step_stride = 100

    def invoke(self, argument, from_tty):
        self.trace()

    def trace(self):
        while True:
            try:
                gdb.execute("stepi " + str(self.step_stride), to_string=False)
                current_pc = self.get_pc()
                f_name = self.get_f_name()  # FIXME: this might be wrong
                self.log(
                    "\n------trace_step @ %s %s------" % (str(hex(current_pc)), f_name)
                )

            except Exception as e:
                self.log_exception(e)


class Function:
    def __init__(self, starting_addr, stack_depth, name):
        self._starting_addr = starting_addr
        self._stack_depth = stack_depth
        self._name = name
        self._addr_count = defaultdict(int)

    def trapped_in_loop(self):
        """
        Check if we are trapped in a loop for a while.
        """
        for addr, count in self._addr_count.items():
            if count > 10:
                return True
        return False

    def tostr(self):
        return self._name + " " + str(self._stack_depth)


class LoopControlledTracer(gdb.Command, Base):
    def __init__(self):
        super(LoopControlledTracer, self).__init__(
            "loop-controlled-trace", gdb.COMMAND_USER
        )

        self.setup_gdb()

        self.DEBUG = True
        self.setup_log()

        # remove all breakpoints
        self.remove_bp()

        # We set it in debug config
        # self.goto_entry()

    def invoke(self, argument, from_tty):
        self.trace()

    def should_skip(self, f_name):
        SKIP_LIBC_FUNC = [
            "memcpy",
            "memset",
            "malloc",
            "free",
            "calloc",
            "realloc",
            "strncpy",
            "strncat",
        ]

        SKIP_MATH_FUNC = [
            "tan",
            "tanf",
            "tanh",
            "tanhf",
            "exp",
            "expf",
            "log",
            "logf",
            "pow",
            "powf",
            "sqrt",
            "sqrtf",
        ]

        SKIP_CUSTOM_FUNC = ["MODEL_ProcessOutput"]

        if f_name in SKIP_LIBC_FUNC:
            return True
        if f_name in SKIP_MATH_FUNC:
            return True
        if f_name in SKIP_CUSTOM_FUNC:
            return True
        return False

    def sync_tracker(self, tracker):
        """
        Sync the tracker with the current stack frame.
        """
        cur_stack_depth = self.get_stack_depth()
        cur_pc = self.get_pc()
        cur_function_name = self.get_f_name()

    def trace(self):
        # It tracks the number of time each address of a function instance is executed. Function instance is an instance in a stack frame. {function instance: {function address: count}}
        tracker = deque()

        # inlined functions: there could be cases where one stepi results in stack frame depth increase that is greater than 1. In this case, we wont track the inlined functions, since likely tracking them is unnecessary.

        start_stack_depth = self.get_stack_depth()

        tracker.append(
            Function(self.get_pc(), self.get_stack_depth(), self.get_f_name())
        )

        while True:
            try:
                # step an instruction
                self.step_inst()
                cur_stack_depth = self.get_stack_depth()
                cur_pc = self.get_pc()
                cur_function_name = self.get_f_name()

                # At this address the stack frame depth is wrong
                if cur_pc == 0x6000952B:
                    self.log("skip weird address")
                    self.step_inst()
                    cur_stack_depth = self.get_stack_depth()
                    cur_pc = self.get_pc()
                    cur_function_name = self.get_f_name()

                if cur_stack_depth < start_stack_depth:
                    # Stop tracing
                    self.log("Finish tracing")
                    break

                # Inlined frame
                if gdb.newest_frame().type() == gdb.INLINE_FRAME:
                    self.log("Finish inlined frame")
                    self.finish()
                    continue

                # Calling other functions
                if cur_stack_depth > tracker[-1]._stack_depth:
                    # finish the inlined frame
                    if cur_stack_depth > tracker[-1]._stack_depth + 1:
                        if gdb.newest_frame().type() == gdb.INLINE_FRAME:
                            self.log("Finish inlined frame")
                            self.finish()
                        else:
                            assert False
                        continue

                    self.log_trace(cur_pc, cur_function_name, "down")

                    if self.should_skip(cur_function_name):
                        self.log("Skip " + cur_function_name + " at " + str(cur_pc))
                        self.finish()
                        cur_pc = self.get_pc()
                        cur_function_name = self.get_f_name()
                        self.log_trace(self.get_pc(), self.get_f_name(), "up")
                        continue

                    tracker.append(Function(cur_pc, cur_stack_depth, cur_function_name))

                # Returning
                elif cur_stack_depth < tracker[-1]._stack_depth:
                    if cur_stack_depth != tracker[-1]._stack_depth - 1:
                        # return to GDB for debugging
                        print("We need to debug")
                        return

                    self.log_trace(cur_pc, cur_function_name, "up")

                    # remove the function instance from the tracker
                    tracker.pop()

                # Same function
                else:
                    assert cur_stack_depth == tracker[-1]._stack_depth

                    if cur_function_name != tracker[-1]._name:
                        print(
                            "[Function name not matched] cur_function_name: "
                            + cur_function_name
                            + " tracker[-1]._name: "
                            + tracker[-1]._name
                        )

                    # Update tracker
                    tracker[-1]._addr_count[self.get_pc()] += 1

                    # check if we are trapped in a loop for a while. If so, we finish this function
                    if tracker[-1].trapped_in_loop():
                        self.log(
                            "Break loop in "
                            + cur_function_name
                            + " "
                            + str(hex(cur_pc))
                        )
                        self.finish()
                        self.log_trace(self.get_pc(), self.get_f_name(), "up")
                        tracker.pop()
                        continue

            except Exception as e:
                self.log_exception(e)
                break


class SimpleLoopControlledTracer(gdb.Command, Base):
    """
    This tracer steps instruction by instruction, tracks the number of time each address is visited, and if an address has been visited more than threshold, we call finish to skip the function.

    The distinguished characteristics of this tracer is that it holds a global count of each address, and it does not track the stack frame depth.
    """

    def __init__(self):
        super(SimpleLoopControlledTracer, self).__init__(
            "simple-loop-controlled-trace", gdb.COMMAND_USER
        )

        self.iteration_bound = 10

        self.setup_gdb()

        self.DEBUG = False
        self.setup_log()

        # remove all breakpoints
        self.remove_bp()

        # No need if we set it in debug config
        self.goto_entry()

    def invoke(self, argument, from_tty):
        self.trace()

    def should_skip(self, f_name):
        SKIP_LIBC_FUNC = [
            "memcpy",
            "memset",
            "malloc",
            "free",
            "calloc",
            "realloc",
            "strncpy",
            "strncat",
        ]

        SKIP_MATH_FUNC = [
            "tan",
            "tanf",
            "tanh",
            "tanhf",
            "exp",
            "expf",
            "log",
            "logf",
            "pow",
            "powf",
            "sqrt",
            "sqrtf",
        ]

        SKIP_CUSTOM_FUNC = ["MODEL_ProcessOutput"]

        if f_name in SKIP_LIBC_FUNC:
            return True
        if f_name in SKIP_MATH_FUNC:
            return True
        if f_name in SKIP_CUSTOM_FUNC:
            return True
        return False

    def trace(self):
        addr_counter = defaultdict(int)
        addr_name = {}

        while True:
            cur_pc = self.get_pc()
            cur_function_name = self.get_f_name()

            if cur_pc not in addr_name:
                addr_name[cur_pc] = cur_function_name

            addr_counter[cur_pc] += 1

            if self.should_skip(cur_function_name) or (
                addr_counter[cur_pc] >= self.iteration_bound
                and not self.is_outermost_frame()
            ):
                self.log("Finish at " + str(hex(cur_pc)) + " " + cur_function_name)
                self.finish()
            else:
                self.log(
                    "Step at "
                    + str(hex(cur_pc))
                    + " in function "
                    + cur_function_name
                    + " for "
                    + str(addr_counter[cur_pc])
                    + " times"
                )
                self.step_inst()


class LoopControlledIOFinder(gdb.Command, Base):
    def __init__(self):
        super(LoopControlledIOFinder, self).__init__(
            "loop-controlled-io-finder", gdb.COMMAND_USER
        )

        self.iteration_bound = 20

        self.setup_gdb()

        self.DEBUG = False
        self.setup_log()

        if TARGET == "imxrt1050-tflm":
            gdb.execute("b Conv")
            # gdb.execute("b micro_graph.cpp:174")
            gdb.execute("c")

    def invoke(self, argument, from_tty):
        self.find()

    def find(self):
        arch = gdb.selected_frame().architecture()

        addr_counter = defaultdict(int)
        addr_name = {}

        ldr_src_addr_list = []

        while True:
            cur_pc = self.get_pc()
            cur_function_name = self.get_f_name()

            if cur_pc not in addr_name:
                addr_name[cur_pc] = cur_function_name

            addr_counter[cur_pc] += 1

            disasm = arch.disassemble(cur_pc)[0]

            if addr_counter[cur_pc] >= self.iteration_bound:
                self.log(
                    "Reach iteration bound at "
                    + str(hex(cur_pc))
                    + " "
                    + cur_function_name
                )

                # log the ldr src addr
                self.log(
                    "ldr src addr list: "
                    + str([hex(addr) for addr in ldr_src_addr_list])
                )

                # for testing
                return

                self.step_continue()

            # step one inst
            self.step_inst()
            # self.log(
            #     "Step at "
            #     + str(hex(cur_pc))
            #     + " in function "
            #     + cur_function_name
            #     + " for "
            #     + str(addr_counter[cur_pc])
            #     + " times"
            # )

            if "ldr" in disasm["asm"]:
                op_code, dest_reg, src_reg, offset = parse_ldr_instruction(
                    disasm["asm"]
                )

                if src_reg != None and offset != None:
                    src_reg_addr = self.read_reg(src_reg)
                    src_addr = src_reg_addr + offset

                    ldr_src_addr_list.append(src_addr)

                    self.log(
                        "ldr inst at "
                        + str(hex(cur_pc))
                        + " loading from: "
                        + str(hex(src_addr))
                        + ", which has been visited "
                        + str(addr_counter[cur_pc])
                        + " times"
                    )

            if "str" in disasm["asm"]:
                pass


# tracer = LoopControlledTracer()
# simple_tracer = SimpleLoopControlledTracer()
simple_io_finder = LoopControlledIOFinder()
