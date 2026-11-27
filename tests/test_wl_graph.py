#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import onnx
import pytest
from ttsim.graph.wl_graph import WorkloadGraph, convert_torch_attrs_to_onnx
from ttsim.ops import SimOp, SimTensor

def test_graph2onnx_softmax_with_dim(session_temp_directory):
    """
    Test that a workload graph with Softmax op having 'dim' attribute
    is successfully exported to ONNX, with attribute conversion.
    """
    # Create a simple graph with Softmax op
    graph = WorkloadGraph("test_softmax")

    # Create input and output tensors
    input_tensor = SimTensor({
        'name': 'input',
        'shape': (1, 10),
        'dtype': 'float32',
        'op_in': ['softmax'],
        'op_out': []
    })
    output_tensor = SimTensor({
        'name': 'output',
        'shape': (1, 10),
        'dtype': 'float32',
        'op_in': [],
        'op_out': ['softmax']
    })

    # Add tensors to graph
    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)

    # Create Softmax op with 'dim' attribute
    softmax_op = SimOp({
        'name': 'softmax',
        'optype': 'Softmax',
        'inList': ['input'],
        'outList': ['output'],
        'attrs': {'dim': 1}
    })
    graph.add_op(softmax_op)

    # Construct the graph
    graph.construct_graph()

    # Export to ONNX
    onnx_filename = str(session_temp_directory / "test_softmax.onnx")
    graph.graph2onnx(onnx_filename)

    # Check that the file was created
    assert os.path.exists(onnx_filename), "ONNX file should be created"

    # Load the model to verify it's valid
    model = onnx.load(onnx_filename)
    onnx.checker.check_model(model)

    # Verify the Softmax node has 'axis' instead of 'dim'
    graph_def = model.graph
    softmax_node = None
    for node in graph_def.node:
        if node.op_type == "Softmax":
            softmax_node = node
            break
    assert softmax_node is not None, "Softmax node should be present"
    assert "axis" in [attr.name for attr in softmax_node.attribute], "Softmax should have 'axis' attribute"
    assert "dim" not in [attr.name for attr in softmax_node.attribute], "Softmax should not have 'dim' attribute"

    # Check the axis value
    axis_attr = next(attr for attr in softmax_node.attribute if attr.name == "axis")
    assert axis_attr.i == 1, "axis should be 1"


def test_convert_torch_attrs_to_onnx():
    """Test the convert_torch_attrs_to_onnx function for various cases."""
    # Test Softmax with dim -> should convert and return copy
    attrs = {'dim': 1, 'other': 'value'}
    result = convert_torch_attrs_to_onnx('Softmax', attrs)
    assert result == {'axis': 1, 'other': 'value'}
    assert result is not attrs  # Should be a copy
    assert attrs == {'dim': 1, 'other': 'value'}  # Original should remain unchanged

    # Test LogSoftmax with dim -> should convert
    attrs = {'dim': 2}
    result = convert_torch_attrs_to_onnx('LogSoftmax', attrs)
    assert result == {'axis': 2}
    assert result is not attrs  # Should be a copy
    assert attrs == {'dim': 2}  # Original should remain unchanged

    # Test Softmax without dim -> should return original
    attrs = {'other': 'value'}
    result = convert_torch_attrs_to_onnx('Softmax', attrs)
    assert result is attrs  # Should return the same object

    # Test other optype -> should return original
    attrs = {'dim': 1}
    result = convert_torch_attrs_to_onnx('SomeOp', attrs)
    assert result is attrs  # Should return the same object

    # Test empty attrs -> should return original
    attrs = {}
    result = convert_torch_attrs_to_onnx('Softmax', attrs)
    assert result is attrs  # Should return the same object

    # Test attrs with both dim and axis -> should raise ValueError
    attrs = {'dim': 1, 'axis': 0}
    with pytest.raises(ValueError, match="both 'dim' and 'axis' are present") as exc_info:
        result = convert_torch_attrs_to_onnx('Softmax', attrs)

    # Test Mean with dim -> should pop dim
    attrs = {'dim': 1, 'keepdims': True}
    result = convert_torch_attrs_to_onnx('Mean', attrs)
    assert result == {'keepdims': True}
    assert result is not attrs  # Should be a copy
    assert attrs == {'dim': 1, 'keepdims': True}  # Original should remain unchanged

    # Test MatMul with bias -> should pop bias
    attrs = {'bias': True, 'transA': False}
    result = convert_torch_attrs_to_onnx('MatMul', attrs)
    assert result == {'transA': False}
    assert result is not attrs  # Should be a copy
    assert attrs == {'bias': True, 'transA': False}  # Original should remain unchanged

def test_graph2onnx_logsoftmax_with_dim(session_temp_directory):
    """
    Test that a workload graph with LogSoftmax op having 'dim' attribute
    is successfully exported to ONNX, with attribute conversion.
    """
    # Create a simple graph with LogSoftmax op
    graph = WorkloadGraph("test_logsoftmax")

    # Create input and output tensors
    input_tensor = SimTensor({
        'name': 'input',
        'shape': (1, 10),
        'dtype': 'float32',
        'op_in': ['logsoftmax'],
        'op_out': []
    })
    output_tensor = SimTensor({
        'name': 'output',
        'shape': (1, 10),
        'dtype': 'float32',
        'op_in': [],
        'op_out': ['logsoftmax']
    })

    # Add tensors to graph
    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)

    # Create LogSoftmax op with 'dim' attribute
    logsoftmax_op = SimOp({
        'name': 'logsoftmax',
        'optype': 'LogSoftmax',
        'inList': ['input'],
        'outList': ['output'],
        'attrs': {'dim': 1}
    })
    graph.add_op(logsoftmax_op)

    # Construct the graph
    graph.construct_graph()

    # Export to ONNX
    onnx_filename = str(session_temp_directory / "test_logsoftmax.onnx")
    graph.graph2onnx(onnx_filename)

    # Check that the file was created
    assert os.path.exists(onnx_filename), "ONNX file should be created"

    # Load the model to verify it's valid
    model = onnx.load(onnx_filename)
    onnx.checker.check_model(model)

    # Verify the LogSoftmax node has 'axis' instead of 'dim'
    graph_def = model.graph
    logsoftmax_node = None
    for node in graph_def.node:
        if node.op_type == "LogSoftmax":
            logsoftmax_node = node
            break
    assert logsoftmax_node is not None, "LogSoftmax node should be present"
    assert "axis" in [attr.name for attr in logsoftmax_node.attribute], "LogSoftmax should have 'axis' attribute"
    assert "dim" not in [attr.name for attr in logsoftmax_node.attribute], "LogSoftmax should not have 'dim' attribute"

    # Check the axis value
    axis_attr = next(attr for attr in logsoftmax_node.attribute if attr.name == "axis")
    assert axis_attr.i == 1, "axis should be 1"


def test_graph2onnx_mean_with_dim(session_temp_directory):
    """
    Test that a workload graph with Mean op having 'dim' attribute
    is successfully exported to ONNX, with 'dim' stripped out.
    """
    graph = WorkloadGraph("test_mean")

    input_tensor = SimTensor({
        'name': 'input',
        'shape': (1, 10),
        'dtype': 'float32',
        'op_in': ['mean'],
        'op_out': []
    })
    output_tensor = SimTensor({
        'name': 'output',
        'shape': (1, 1),
        'dtype': 'float32',
        'op_in': [],
        'op_out': ['mean']
    })

    graph.add_tensor(input_tensor)
    graph.add_tensor(output_tensor)

    mean_op = SimOp({
        'name': 'mean',
        'optype': 'Mean',
        'inList': ['input'],
        'outList': ['output'],
        'attrs': {'dim': 1}
    })
    graph.add_op(mean_op)
    graph.construct_graph()

    onnx_filename = str(session_temp_directory / "test_mean.onnx")
    graph.graph2onnx(onnx_filename)

    assert os.path.exists(onnx_filename), "ONNX file should be created"

    model = onnx.load(onnx_filename)
    onnx.checker.check_model(model)

    graph_def = model.graph
    mean_node = None
    for node in graph_def.node:
        if node.op_type == "Mean":
            mean_node = node
            break
    assert mean_node is not None, "Mean node should be present"
    assert "dim" not in [attr.name for attr in mean_node.attribute], "Mean should not have 'dim' attribute"


def test_graph2onnx_matmul_with_bias(session_temp_directory):
    """
    Test that a workload graph with MatMul op having 'bias' attribute
    is successfully exported to ONNX, with 'bias' stripped out.
    """
    graph = WorkloadGraph("test_matmul")

    input_a = SimTensor({
        'name': 'input_a',
        'shape': (2, 3),
        'dtype': 'float32',
        'op_in': ['matmul'],
        'op_out': []
    })
    input_b = SimTensor({
        'name': 'input_b',
        'shape': (3, 4),
        'dtype': 'float32',
        'op_in': ['matmul'],
        'op_out': []
    })
    output_tensor = SimTensor({
        'name': 'output',
        'shape': (2, 4),
        'dtype': 'float32',
        'op_in': [],
        'op_out': ['matmul']
    })

    graph.add_tensor(input_a)
    graph.add_tensor(input_b)
    graph.add_tensor(output_tensor)

    matmul_op = SimOp({
        'name': 'matmul',
        'optype': 'MatMul',
        'inList': ['input_a', 'input_b'],
        'outList': ['output'],
        'attrs': {'bias': True}
    })
    graph.add_op(matmul_op)
    graph.construct_graph()

    onnx_filename = str(session_temp_directory / "test_matmul.onnx")
    graph.graph2onnx(onnx_filename)

    assert os.path.exists(onnx_filename), "ONNX file should be created"

    model = onnx.load(onnx_filename)
    onnx.checker.check_model(model)

    graph_def = model.graph
    matmul_node = None
    for node in graph_def.node:
        if node.op_type == "MatMul":
            matmul_node = node
            break
    assert matmul_node is not None, "MatMul node should be present"
    assert "bias" not in [attr.name for attr in matmul_node.attribute], "MatMul should not have 'bias' attribute"