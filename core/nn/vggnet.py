import mxnet as mx

class VGGNet:

    @staticmethod
    def build(num_classes):

        data = mx.sym.Variable("data")

        # Block 1
        conv_1_1 = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv_1_1")
        act_1_1 = mx.sym.LeakyReLU(data=conv_1_1, act_type="prelu", name="act_1_1")
        bn_1_1 = mx.sym.BatchNorm(data=act_1_1, name="bn_1_1")
        conv_1_2 = mx.sym.Convolution(data=bn_1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv_1_2")
        act_1_2 = mx.sym.LeakyReLU(data=conv_1_2, act_type="prelu", name="act_1_2")
        bn_1_2 = mx.sym.BatchNorm(data=act_1_2, name="bn_1_2")
        pool_1_1 = mx.sym.Pooling(data=bn_1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool_1_1")
        dropout_1_1 = mx.sym.Dropout(data=pool_1_1, p=0.25)

        # Block 2
        conv_2_1 = mx.sym.Convolution(data=dropout_1_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv_2_1")
        act_2_1 = mx.sym.LeakyReLU(data=conv_2_1, act_type="prelu", name="act_2_1")
        bn_2_1 = mx.sym.BatchNorm(data=act_2_1, name="bn_2_1")
        conv_2_2 = mx.sym.Convolution(data=bn_2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv_2_2")
        act_2_2 = mx.sym.LeakyReLU(data=conv_2_2, act_type="prelu", name="act_2_2")
        bn_2_2 = mx.sym.BatchNorm(data=act_2_2, name="bn_2_2")
        pool_2_1 = mx.sym.Pooling(data=bn_2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool_2_1")
        dropout_2_1 = mx.sym.Dropout(data=pool_2_1, p=0.25)

        # Block 3
        conv_3_1 = mx.sym.Convolution(data=dropout_2_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv_3_1")
        act_3_1 = mx.sym.LeakyReLU(data=conv_3_1, act_type="prelu", name="act_3_1")
        bn_3_1 = mx.sym.BatchNorm(data=act_3_1, name="bn_3_1")
        conv_3_2 = mx.sym.Convolution(data=bn_3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv_3_2")
        act_3_2 = mx.sym.LeakyReLU(data=conv_3_2, act_type="prelu", name="act_3_2")
        bn_3_2 = mx.sym.BatchNorm(data=act_3_2, name="bn_3_2")
        conv_3_3 = mx.sym.Convolution(data=bn_3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv_3_3")
        act_3_3 = mx.sym.LeakyReLU(data=conv_3_3, act_type="prelu", name="act_3_3")
        bn_3_3 = mx.sym.BatchNorm(data=act_3_3, name="bn_3_3")
        pool_3_1 = mx.sym.Pooling(data=bn_3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool_3_1")
        dropout_3_1 = mx.sym.Dropout(data=pool_3_1, p=0.25)


        # Block 4
        conv_4_1 = mx.sym.Convolution(data=dropout_3_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv_4_1")
        act_4_1 = mx.sym.LeakyReLU(data=conv_4_1, act_type="prelu", name="act_4_1")
        bn_4_1 = mx.sym.BatchNorm(data=act_4_1, name="bn_4_1")
        conv_4_2 = mx.sym.Convolution(data=bn_4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv_4_2")
        act_4_2 = mx.sym.LeakyReLU(data=conv_4_2, act_type="prelu", name="act_4_2")
        bn_4_2 = mx.sym.BatchNorm(data=act_4_2, name="bn_4_2")
        conv_4_3 = mx.sym.Convolution(data=bn_4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv_4_3")
        act_4_3 = mx.sym.LeakyReLU(data=conv_4_3, act_type="prelu", name="act_4_3")
        bn_4_3 = mx.sym.BatchNorm(data=act_4_3, name="bn_4_3")
        pool_4_1 = mx.sym.Pooling(data=bn_4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool_4_1")
        dropout_4_1 = mx.sym.Dropout(data=pool_4_1, p=0.25)

        # Block 5
        conv_5_1 = mx.sym.Convolution(data=dropout_4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv_5_1")
        act_5_1 = mx.sym.LeakyReLU(data=conv_5_1, act_type="prelu", name="act_5_1")
        bn_5_1 = mx.sym.BatchNorm(data=act_5_1, name="bn_5_1")
        conv_5_2 = mx.sym.Convolution(data=bn_5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv_5_2")
        act_5_2 = mx.sym.LeakyReLU(data=conv_5_2, act_type="prelu", name="act_5_2")
        bn_5_2 = mx.sym.BatchNorm(data=act_5_2, name="bn_5_2")
        conv_5_3 = mx.sym.Convolution(data=bn_5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv_5_3")
        act_5_3 = mx.sym.LeakyReLU(data=conv_5_3, act_type="prelu", name="act_5_3")
        bn_5_3 = mx.sym.BatchNorm(data=act_5_3, name="bn_5_3")
        pool_5_1 = mx.sym.Pooling(data=bn_5_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool_5_1")
        dropout_5_1 = mx.sym.Dropout(data=pool_5_1, p=0.25)

        # Softmax
        flatten = mx.sym.Flatten(data=dropout_5_1, name="flatten")
        fc_6_1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096, name="fc_6_1")
        act_6_1 = mx.sym.LeakyReLU(data=fc_6_1, act_type="prelu", name="act_6_1")
        bn_6_1 = mx.sym.BatchNorm(data=act_6_1, name="bn_6_1")
        dropout_6_1 = mx.sym.Dropout(data=bn_6_1, p=0.5)
        fc_6_2 = mx.sym.FullyConnected(data=dropout_6_1, num_hidden=4096, name="fc_6_2")
        act_6_2 = mx.sym.LeakyReLU(data=fc_6_2, act_type="prelu", name="act_6_2")
        bn_6_2 = mx.sym.BatchNorm(data=act_6_2, name="bn_6_2")
        dropout_6_2 = mx.sym.Dropout(data=bn_6_2, p=0.5)
        fc_6_3 = mx.sym.FullyConnected(data=dropout_6_2, num_hidden=num_classes, name="fc_6_3")
        model = mx.sym.SoftmaxOutput(data=fc_6_3, name="softmax")

        return model
