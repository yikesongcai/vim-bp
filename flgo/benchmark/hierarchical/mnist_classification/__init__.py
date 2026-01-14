from .model import default_model
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition
import flgo.benchmark.toolkits.visualization as v

default_model = default_model
default_partitioner = flgo.benchmark.toolkits.partition.BasicHierPartitioner
default_partition_para = {}
visualize = v.visualize_hier_by_class