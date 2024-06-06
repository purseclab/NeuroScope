import torch
import os
import struct
import onnx
from onnx import helper
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

from Datapoint import DataPoint
from Model import FeatureNet
from Dataset import Operator

DUMP_DIR = "../online/dump/"
MODEL_PATH = "./saved_models/mt_model.pth"
FEATURE_LEN = 24


def read_dump(dump_path):
    with open(dump_path, "rb") as file:
        data = file.read()

    float_size = 4
    floats = struct.unpack("<" + "f" * (len(data) // float_size), data)
    return torch.tensor(floats)


def main():
    # read file names from the dump directory
    dump_names = os.listdir(DUMP_DIR)

    # get the number of operators
    operator_count = 0
    for file_name in dump_names:
        parts = file_name.split("_")
        if int(parts[1]) > operator_count:
            operator_count = int(parts[1])

    # load the model
    device = torch.device("cpu")
    mt = FeatureNet(FEATURE_LEN, len(Operator))
    mt.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # create input node
    input_node = make_tensor_value_info("inputs", onnx.TensorProto.FLOAT, None)
    prev_node = input_node

    created_nodes = []

    for op_idx in range(operator_count + 1):
        # read the dump file
        input_dump_path = os.path.join(DUMP_DIR, "dump_{}_0.bin".format(op_idx))
        input_tensor = read_dump(input_dump_path)
        output_dump_path = os.path.join(DUMP_DIR, "dump_{}_1.bin".format(op_idx))
        output_tensor = read_dump(output_dump_path)

        # create the datapoint
        dp = DataPoint(input_tensor, output_tensor)
        dp.recompute()

        # infer the operator
        mt.eval()
        with torch.no_grad():
            output = mt(
                input_tensor, None, output_tensor, None, torch.tensor(dp.features())
            )
            _, label = torch.max(output, 0)
            # print("Operator: ", list(Operator)[label].__str__())
            op_type = list(Operator)[label.item()].__str__()

            if op_type == "Operator.CONV":
                node = helper.make_node(
                    op_type="Conv",
                    name="op_{}".format(op_idx),
                    inputs=[prev_node.name],
                    outputs=["op_{}".format(op_idx)],
                )
                created_nodes.append(node)
                prev_node = node
            elif op_type == "Operator.AVG_POOL":
                node = helper.make_node(
                    op_type="AveragePool",
                    name="op_{}".format(op_idx),
                    inputs=[prev_node.name],
                    outputs=["op_{}".format(op_idx)],
                )
                created_nodes.append(node)
                prev_node = node
            elif op_type == "Operator.FC":
                node = helper.make_node(
                    op_type="Gemm",
                    name="op_{}".format(op_idx),
                    inputs=[prev_node.name],
                    outputs=["op_{}".format(op_idx)],
                )
                created_nodes.append(node)
                prev_node = node
            elif op_type == "Operator.ADD":
                node = helper.make_node(
                    op_type="Add",
                    name="op_{}".format(op_idx),
                    inputs=[prev_node.name],
                    outputs=["op_{}".format(op_idx)],
                )
                created_nodes.append(node)
                prev_node = node
            elif op_type == "Operator.MAX_POOL":
                node = helper.make_node(
                    op_type="MaxPool",
                    name="op_{}".format(op_idx),
                    inputs=[prev_node.name],
                    outputs=["op_{}".format(op_idx)],
                )
                created_nodes.append(node)
                prev_node = node
            elif op_type == "Operator.SOFTMAX":
                node = helper.make_node(
                    op_type="Softmax",
                    name="op_{}".format(op_idx),
                    inputs=[prev_node.name],
                    outputs=["op_{}".format(op_idx)],
                )
                created_nodes.append(node)
                prev_node = node
            else:
                assert False, "Unknown operator type!"

    # create output node
    output_node = make_tensor_value_info(prev_node.name, onnx.TensorProto.FLOAT, None)

    # create graph
    graph = make_graph(created_nodes, "test", [input_node], [output_node], [])

    # create model
    model = helper.make_model(graph, producer_name="onnx-builder")

    # save model
    onnx.save_model(model, "recovered_model.onnx")


if __name__ == "__main__":
    main()
