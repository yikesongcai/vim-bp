from .model import default_model
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition

default_model = default_model
default_partitioner = flgo.benchmark.toolkits.partition.BasicHierPartitioner
default_partition_para = {}
visualize = flgo.benchmark.toolkits.visualization.visualize_hier_by_class