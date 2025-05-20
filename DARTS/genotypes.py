from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

CIFAR_10_arch = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
                                 ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0)],
                         normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2),
                                                            ('skip_connect', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0),
                                                            ('sep_conv_5x5', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

CIFAR_100_arch = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0),
                                  ('sep_conv_5x5', 3), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1)],
                          normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0),
                                                             ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3),
                                                             ('max_pool_3x3', 2), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))

ImageNet_arch = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2),
                                 ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)],
                         normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0),
                                                            ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3),
                                                            ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
