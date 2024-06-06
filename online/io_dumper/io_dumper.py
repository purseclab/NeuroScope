import gdb

TARGET = "imxrt1050-tflm"
LOGGING_FILE_PATH = "/tmp/io_dumper.log"
DUMP_DIR_PATH = "/tmp/dump/"


def addr2num(addr):
    print(type(addr))
    try:
        return int(addr)  # Python 3
    except:
        return long(addr)  # Python 2


class IoDumper(gdb.Command):
    def __init__(self):
        super(IoDumper, self).__init__("io-dump", gdb.COMMAND_USER)

        self.logging_file_path = LOGGING_FILE_PATH
        self.logging_file = open(self.logging_file_path, "w")

        self.setup_gdb()

    def log(self, msg):
        print(msg)
        self.logging_file.write(msg + "\n")

    def setup_gdb(self):
        # make sure the breakpoint is fired
        gdb.execute("set confirm off")
        gdb.execute("set breakpoint pending on")

        # disable the behavior that re-execute last command when typing blank line
        gdb.execute("dont-repeat")

        gdb.execute("set print pretty on")

        # make sure we step into the function without symbol info
        gdb.execute("set step-mode on")

        # set logging
        gdb.execute("set logging file {}".format(self.logging_file_path))
        gdb.execute("set logging on")

    def remove_all_bp(self, addr=None):
        gdb.execute("d")  # remove all breakpoints

    def break_at(self, addr):
        gdb.execute("b *{}".format(addr))
        self.log("Breakpoint at {}".format(addr))

    def continue_execution(self):
        gdb.execute("c")
        self.log("Continue")

    def step_into(self):
        gdb.execute("s")
        self.log("Step into")

    def get_func_name(self):
        name = gdb.newest_frame().name()
        if name is None:
            return "None"
        return name

    def read_reg(self, reg):
        return addr2num(gdb.newest_frame().read_register(reg))

    def read_mem(self, addr, size):
        return addr2num(gdb.selected_inferior().read_memory(addr, size))

    def get_size(self, addr):
        # read the value at the address that is 4 bytes after the input address
        dim_addr = self.read_mem(addr + 4, 4)
        self.log("Dim address: {}".format(dim_addr))

        rank = self.read_mem(dim_addr, 4)
        self.log("Rank: {}".format(rank))

        dims = []
        for i in range(rank):
            dim = self.read_mem(dim_addr + 4 + i * 4, 4)
        self.log("Dims: {}".format(dims))

        def multiply_list_elements(lst):
            result = 1
            for element in lst:
                result *= element
            return result

        return multiply_list_elements(dims)

    def get_dump(self, addr, size):
        addr_val = gdb.Value(addr).cast(gdb.lookup_type("char").pointer())
        dump_mem = self.read_mem(addr_val, size)
        byte_array = bytearray(dump_mem)
        return byte_array

    def invoke(self, arg, from_tty):
        # breakpoint the operator functions and dump the input/output
        BP_LIST = []
        if TARGET == "imxrt1050-tflm":
            BP_LIST = [0x6000A6E2, 0x6000A72A]
        for bp in BP_LIST:
            self.break_at(bp)

        op_counter = 0
        counter = 0

        while True:
            self.continue_execution()
            self.log("Into function: {}".format(self.get_func_name()))

            buffer_addr = self.read_reg("r0")
            self.log("Buffer address: {}".format(buffer_addr))

            size = self.get_size(buffer_addr)
            dump = self.get_dump(buffer_addr, size)

            with open(DUMP_DIR_PATH + "dump_{}_{}.bin".format(counter), "wb") as f:
                f.write(dump)

            counter += 1
            if counter == 2:
                op_counter += 1
                counter = 0


io_dumper = IoDumper()
