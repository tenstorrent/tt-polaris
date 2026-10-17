#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

import networkx as nx
import numpy as np
#for graph2onnx
import onnx
from loguru import logger
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor, make_tensor_value_info

from ttsim.ops import SimOp, SimTensor

LOG   = logger
INFO  = LOG.info
DEBUG = LOG.debug

def convert_torch_attrs_to_onnx(optype: str, attrs: dict) -> dict:
    """
    Convert PyTorch/Torch-style attribute names to ONNX-compatible attribute names.

    This function handles attribute name mismatches between PyTorch/Torch conventions
    and ONNX specifications. It can be extended for future attribute conversions.

    Args:
        optype: The operator type (e.g., 'Softmax', 'LogSoftmax')
        attrs: Dictionary of operator attributes

    Returns:
        Dictionary with converted attribute names
    """
    # Convert 'dim' to 'axis' for Softmax/LogSoftmax (ONNX uses 'axis')
    if optype in ['Softmax', 'LogSoftmax'] and 'dim' in attrs:
        if 'axis' in attrs:
            raise ValueError(
                f"Cannot convert attributes for {optype}: both 'dim' and 'axis' are present"
            )
        converted_attrs = attrs.copy()
        converted_attrs['axis'] = converted_attrs.pop('dim')
        return converted_attrs

    # For Dropout (opset >= 12), 'prob' and 'train_mode' are NOT valid ONNX attributes;
    # they are already passed as input tensors (ratio, training_mode). Strip them here.
    if optype == 'Dropout':
        converted_attrs = attrs.copy()
        converted_attrs.pop('prob', None)
        converted_attrs.pop('train_mode', None)
        return converted_attrs

    # For Mean (ONNX Mean takes a list of tensors and has no 'dim' attribute)
    if optype == 'Mean':
        converted_attrs = attrs.copy()
        converted_attrs.pop('dim', None)
        return converted_attrs

    # For MatMul (ONNX MatMul does not take 'bias' attribute)
    if optype == 'MatMul':
        converted_attrs = attrs.copy()
        converted_attrs.pop('bias', None)
        return converted_attrs

    # Future extensions can be added here:
    # Example: if optype == 'SomeOp' and 'torch_attr' in attrs:
    #     converted_attrs = attrs.copy()
    #     converted_attrs['onnx_attr'] = converted_attrs.pop('torch_attr')
    #     return converted_attrs

    return attrs

class WorkloadGraph():
    # Type hint for instance attribute
    _graph: nx.MultiDiGraph
    def __init__(self, name):
        self._graph          = nx.MultiDiGraph() # NetworkX MultiDiGraph
        self._name           = name              # graph.name                  : Str
        self._hdr_info       = {}                # Workload Header Info        : Dict -- ONNX like
        self._ops            = {}                # All Ops in the Graph        : Dict[Str, SimOp]
        self._tensors        = {}                # All Tensors in the Graph    : Dict[Str, SimTensor]
        self._input_nodes    = []                # Input Nodes in the Graph    : List[Str] Op Names
        self._output_nodes   = []                # Output Nodes in the Graph   : List[Str] Op Names
        self._input_tensors  = []                # Input Tensors in the Graph  : List[Str] Op Names
        self._output_tensors = []                # Output Tensors in the Graph : List[Str] Op Names
        self._optype_hist    = defaultdict(int)  # Op Type Histogram           : Dict[Str, Int]
        # Per-graph sequence counter to assign unique sequence numbers to operators based on
        # addition order. This ensures deterministic ordering when multiple valid topological
        # orderings exist, which is critical for reproducible CSV stats output.
        self._seqcounter     = 0

    def get_node_count(self)     : return self._graph.number_of_nodes()
    def get_edge_count(self)     : return self._graph.number_of_edges()
    def get_node(self, name)     : return self._graph[name]['data']
    def has_node(self, name)     : return self._graph.has_node(name)
    def get_input_nodes(self)    : return self._input_nodes
    def get_output_nodes(self)   : return self._output_nodes
    def get_input_tensors(self)  : return self._input_tensors
    def get_output_tensors(self) : return self._output_tensors
    def get_ordered_nodes(self):
        """
        Return nodes in a deterministic topological order.

        Uses lexicographical topological sort with operator sequence numbers (seqno) as the
        tie-breaking key. This ensures that when multiple valid topological orderings exist,
        the same ordering is always chosen based on the addition order of operators to this graph.
        This determinism is essential for:
        - Consistent CSV stats output (opnum and row ordering)
        - Reproducible results across multiple runs

        Returns:
            List of operator names in deterministic topological order.
        """
        return list(nx.lexicographical_topological_sort(self._graph, key=lambda x: self._ops[x].seqno))

    def is_input_node(self, opname): return opname in self._input_nodes
    def is_output_node(self, opname): return opname in self._output_nodes

    def add_hdr_info(self, hdr_info):
        self._hdr_info = hdr_info

    def add_op(self, op: SimOp):
        for tensor_name in op.inList:
            assert tensor_name in self._tensors, \
                    f"Input SimTensor {tensor_name} for SimOp {op.name} not found in WorkloadGraph"
        for tensor_name in op.outList:
            assert tensor_name in self._tensors, \
                    f"Output SimTensor {tensor_name} for SimOp {op.name} not found in WorkloadGraph"
        assert op.name not in self._ops, f"SimOp({op.name}) is not unique!!!"
        # Assign sequence number based on addition order to this graph. This is used as a
        # tie-breaker in lexicographical topological sort to ensure deterministic operator
        # ordering, even when the graph has multiple valid orderings.
        op.seqno = self._seqcounter
        self._seqcounter += 1
        self._ops[op.name] = op

    def add_tensor(self, tensor: SimTensor):
        assert tensor.name not in self._tensors, f"SimTensor({tensor.name}) is not unique!!!"
        self._tensors[tensor.name] = tensor

    def construct_graph(self):
        #construct graph
        for op_count, (op_name, op_info) in enumerate(self._ops.items()):
            self._graph.add_node(op_name)
            self._optype_hist[op_info.optype] += 1
            for o in op_info.outList:
                for inode in self._tensors[o].op_in:
                    self._graph.add_edge(op_name, inode, name=o)
        #update graph summary data structures
        self._input_tensors  = list(set([tname for tname,tval in self._tensors.items() if tval.op_out == []]))
        self._output_tensors = list(set([tname for tname,tval in self._tensors.items() if tval.op_in == []]))
        self._input_nodes    = list(set([o for t in self._input_tensors for o in self._tensors[t].op_in]))
        self._output_nodes   = list(set([o for t in self._output_tensors for o in self._tensors[t].op_out]))

        return

    def rebuild_graph(self):
        """Rebuild the NetworkX graph + summary structures from the current _ops/_tensors.

        Use after in-place op-graph surgery that adds/removes SimOps (e.g. the backend
        arch-aware column-split transform), so get_ordered_nodes()/get_successors() and the
        input/output node/tensor sets reflect the mutated graph. Ops flagged
        removed_in_optimization and dangling consumer references are skipped. Idempotent.
        """
        self._graph = nx.MultiDiGraph()
        self._optype_hist = defaultdict(int)
        for op_name, op_info in self._ops.items():
            if getattr(op_info, 'removed_in_optimization', False):
                continue
            self._graph.add_node(op_name)
            self._optype_hist[op_info.optype] += 1
            for o in op_info.outList:
                if o not in self._tensors:
                    continue
                for inode in self._tensors[o].op_in:
                    if inode in self._ops and not getattr(self._ops[inode], 'removed_in_optimization', False):
                        self._graph.add_edge(op_name, inode, name=o)
        self._input_tensors  = list(set([tname for tname, tval in self._tensors.items() if tval.op_out == []]))
        self._output_tensors = list(set([tname for tname, tval in self._tensors.items() if tval.op_in == []]))
        # Filter dangling consumer/producer references (ops removed or flagged
        # removed_in_optimization) so the node sets honour this method's contract and never
        # surface op names absent from _ops after in-place graph surgery.
        def _op_live(name: str) -> bool:
            return name in self._ops and not getattr(self._ops[name], 'removed_in_optimization', False)
        self._input_nodes    = list(set([o for t in self._input_tensors for o in self._tensors[t].op_in if _op_live(o)]))
        self._output_nodes   = list(set([o for t in self._output_tensors for o in self._tensors[t].op_out if _op_live(o)]))
        return

    def __str__(self):
        indent = 4
        ostr = ""

        ostr += '\nSUMMARY:\n'
        ostr += f"{' '*indent}num_nodes         : {self.get_node_count()}\n"
        ostr += f"{' '*indent}num_edges         : {self.get_edge_count()}\n"
        ostr += f"{' '*indent}num_input_nodes   : {len(self.get_input_nodes())}\n"
        ostr += f"{' '*indent}num_output_nodes  : {len(self.get_output_nodes())}\n"
        ostr += f"{' '*indent}num_input_tensors : {len(self.get_input_tensors())}\n"
        ostr += f"{' '*indent}num_output_tensors: {len(self.get_output_tensors())}\n"
        ostr += f"{' '*indent}input_nodes       :\n"
        for op_in in self.get_input_nodes(): ostr += f"{' '*2*indent}{op_in}\n"
        ostr += f"{' '*indent}output_nodes      :\n"
        for op_out in self.get_output_nodes(): ostr += f"{' '*2*indent}{op_out}\n"
        ostr += f"{' '*indent}input_tensors     :\n"
        for t_in in self.get_input_tensors(): ostr += f"{' '*2*indent}{t_in}\n"
        ostr += f"{' '*indent}output_tensors    :\n"
        for t_out in self.get_output_tensors(): ostr += f"{' '*2*indent}{t_out}\n"


        ostr += '\nHDRINFO:\n'
        for k,v in self._hdr_info.items():
            ostr += f"{' '*indent}{k:25s}: {v}\n"

        ostr += '\nOPTYPE_HISTOGRAM:\n'
        for k,v in sorted(self._optype_hist.items(),
                          key=lambda item: item[1],
                          reverse=True):
            ostr += f"{' '*indent}{k:25s}: {v:5d}\n"

        ostr += '\nTENSORS:\n'
        for k,v in self._tensors.items():
            ostr += f"{' '*indent}{v}\n"

        ostr += '\nOPS:\n'
        for k,v in self._ops.items():
            ostr += f"{' '*indent}{v}\n"

        return ostr

    def reverse_graph(self): pass
    def find_forks_and_joins(self): pass
    def create_gradient_tensors(self): pass
    def backward(self): pass

    def get_op(self, opname):
        return self._ops[opname]

    def get_optype(self, opname):
        return self._ops[opname].optype

    def get_successors(self, opname):
        return list(self._graph.successors(opname))

    def is_removed(self, opname):
        return self._ops[opname].removed_in_optimization

    def remove_nodes(self, removal_spec):
        for opname in self.get_ordered_nodes():
            optype = self.get_optype(opname).upper()
            if removal_spec.is_included(optype):
                self._ops[opname].remove_in_optimization()
        return

    def fuse_nodes(self, fusion_spec):
        """
        op_type_fusion_list is a list of lists specifying optype-patterns to search for
         in the graph and then fuse. The patterns are specifed in order of priority
         e.g. [ [Matmul, Gelu, Add], [Matmul, Gelu] ] will match Matmul-Gelu-Add
           first and only then match the next pattern
        Steps:
            1) Find all matched nodes in the graph:        fusion_candidates
            2) Track nodes that have already been matched: already_matched_nodes_set

        """
        fusion_candidates = []
        already_matched_nodes_set = set()
        for pattern in fusion_spec.get_fused_layer_sequences():
            pattern_len = len(pattern)
            assert pattern_len > 1, f"Illegal fusion pattern specification {pattern}!!"
            for opname in self.get_ordered_nodes():
                if opname in already_matched_nodes_set:
                    continue

                if self.get_optype(opname).upper() == pattern[0]:
                    current_node          = opname
                    matched_nodes_list    = [current_node]
                    for i in range(1, pattern_len):
                        successors = self.get_successors(current_node)
                        if ( len(successors) == 1 and
                            self.get_optype(successors[0]).upper() == pattern[i] and
                            successors[0] not in already_matched_nodes_set):
                            current_node = successors[0]
                            matched_nodes_list.append(current_node)
                        elif len(successors) == 1 and self.is_removed(successors[0]):
                            """
                            Skip nodes that have been marked for removal during optimization
                            and continue looking at their successors instead
                            """
                            current_node = successors[0]
                            next_successors = self.get_successors(current_node)
                            if next_successors:
                                current_node = next_successors[0]
                                if (self.get_optype(current_node).upper() == pattern[i] and
                                    current_node not in already_matched_nodes_set):
                                    matched_nodes_list.append(current_node)
                                else:
                                    """Pattern doesn't match after skipping removed node"""
                                    break
                            else:
                                """No more successors after a removed node"""
                                break
                        else:
                            """pattern does not match break"""
                            break
                    if len(matched_nodes_list) == pattern_len:
                        fusion_candidates.append(matched_nodes_list)
                        already_matched_nodes_set.update(matched_nodes_list)
        return fusion_candidates

    def set_precision(self, data_type_spec):
        for opname in self.get_ordered_nodes():
            optype = self.get_optype(opname).upper()
            dt     = data_type_spec.layer_2_datatype(optype)
            self._ops[opname].precision = dt

        for opname in self.get_ordered_nodes():
            op = self.get_op(opname)
            assert op.precision is not None, f'Precision for {opname} is not set'

        return

    def set_resources(self, rsrc_spec):
        for opname in self.get_ordered_nodes():
            optype = self.get_optype(opname).upper()
            rsrc   = rsrc_spec.layer_2_pipe(optype)
            try:
                self._ops[opname].uses_compute_pipe = rsrc
            except KeyError as e:
                raise RuntimeError(f'node={opname} operator={optype} rsrc={rsrc} error!') from e
        return

    def graph2onnx(self, onnx_filename, /, producer_name="ttsim.functional.export",
                   do_model_check=True, filter_op_attrs=None):
        nptype_map = {
                np.float16: TensorProto.FLOAT16,
                np.float32: TensorProto.FLOAT,
                np.float64: TensorProto.DOUBLE,
                np.uint8:   TensorProto.UINT8,
                np.uint16:  TensorProto.UINT16,
                np.uint32:  TensorProto.UINT32,
                np.int32:   TensorProto.INT32,
                np.int64:   TensorProto.INT64,
                np.bool_:   TensorProto.BOOL,
                np.float16: TensorProto.FLOAT16,
                }
        #create tensors....
        onnx_tensors = {}
        onnx_constants = {}
        onnx_params = {}
        for tname, tval in self._tensors.items():
            assert tval.shape is not None, "Illegal tensor shape for make_tensor_value_info: {tval}"

            #TODO: ensure that dims are always int (or str)
            # somewhere in ttsim.front.functional we are creating float dims!!!
            # thens this stupid line to create shape0 can be removed
            shape0 = tuple([int(d) for d in tval.shape])
            dtype_type = np.dtype(tval.dtype).type
            assert dtype_type in nptype_map, f"dtype={tval.dtype} not yet supported. Pl. edit wl_graph accordingly!!"

            if shape0 == (): #rank-0 tensor
                if tval.data is None:
                    _data = np.random.randn(1).astype(tval.dtype)
                    tval.data = _data[0]
                val_list = [tval.data]
            else:
                if tval.data is None:
                    tval.data = np.random.randn(*shape0).astype(tval.dtype)
                val_list = tval.data.flatten().tolist()


            if tval.is_const:
                onnx_constants[tname] = make_tensor(name=tname,
                                                    data_type=nptype_map[dtype_type],
                                                    dims=shape0,
                                                    vals=val_list
                                                    )
            elif tval.is_param:
                #print("onnx_params", tval)
                #print("onnx_params", tval.data.shape, tval.data.size)
                onnx_params[tname] = make_tensor(name=tname,
                                                 data_type=nptype_map[dtype_type],
                                                 dims=shape0,
                                                 vals=val_list
                                                 )
            else:
                onnx_tensors[tname] = make_tensor_value_info(tname,
                                                             nptype_map[dtype_type],
                                                             shape0)

        onnx_nodes = {}
        for oname, op in self._ops.items():
            if filter_op_attrs is not None:
                onnx_attrs = filter_op_attrs(op.attrs)
            else:
                onnx_attrs = op.attrs

            # Remove attributes that are internal to ttsim but not valid in ONNX
            _ttsim_only_attrs = {
                'Slice':   {'out_shape'},
                'Split':   {'ipos'},
            }
            if op.optype in _ttsim_only_attrs:
                drop_keys = _ttsim_only_attrs[op.optype]
                onnx_attrs = {k: v for k, v in onnx_attrs.items() if k not in drop_keys}

            # Convert torch-style attributes to ONNX-compatible attributes
            onnx_attrs = convert_torch_attrs_to_onnx(op.optype, onnx_attrs)
            onnx_nodes[oname] = make_node(op.optype, op.inList, op.outList, name=oname, **onnx_attrs)

        input_list        = [onnx_tensors[x] for x in self.get_input_tensors() if x in onnx_tensors]
        output_list       = [onnx_tensors[x] for x in self.get_output_tensors() if x in onnx_tensors]
        initializer_list  = [onnx_constants[tname] for tname in onnx_constants] + [onnx_params[tname] for tname in onnx_params]
        node_list         = [onnx_nodes[node] for node in nx.topological_sort(self._graph)]

        onnx_graph = make_graph(
                nodes       = node_list,
                name        = self._name,
                inputs      = input_list,
                outputs     = output_list,
                initializer = initializer_list, #TODO: constants can go here...
                )
        model_def = make_model(onnx_graph, producer_name=producer_name)
        if do_model_check:
            check_model(model_def)
        onnx.save(model_def, onnx_filename)
        return