import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers import GATConv
from spektral.layers import ops
from spektral.layers.ops import modes

class our_gat_conv(GATConv):
    def build(self, input_shape):
        super().build(input_shape)

        input_dim = input_shape[2][-1]

        self.edge_kernel = self.add_weight(
            name="edge_kernel",
            shape=[input_dim, input_dim],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )

        self.edge_kernel_self = self.add_weight(
            name="edge_kernel_self",
            shape=[input_dim, self.attn_heads],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )

        self.edge_kernel_add = self.add_weight(
            name="edge_kernel_add",
            shape=[input_dim, self.attn_heads, input_dim],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )

        self.output_kernel = self.add_weight(
            name="output_kernel",
            shape=[self.channels+input_dim, self.channels],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )

    def call(self, inputs, mask=None):
        x, a, e = inputs

        mode = ops.autodetect_mode(x, a)
        if mode == modes.SINGLE and K.is_sparse(a):
            output, attn_coef = self._call_single(x, a)
        else:
            if K.is_sparse(a):
                a = tf.sparse.to_dense(a)
            bond_output, output, attn_coef = self._call_dense(x, a, e)

        if self.concat_heads:
            shape = tf.concat((tf.shape(output)[:-2], [self.attn_heads * self.channels]), axis=0)
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)

        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]

        output = self.activation(output)
        bond_output = self.activation(bond_output)

        if self.return_attn_coef:
            return bond_output, output, attn_coef
        else:
            return bond_output, output

    def _call_dense(self, x, a, e):
        shape = tf.shape(a)[:-1]
        a = tf.linalg.set_diag(a, tf.zeros(shape, a.dtype))
        a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))

        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)

        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs)
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        ###
        e = tf.einsum("...NI , IO -> ...NO", e, self.edge_kernel)
        edgeAtt_self = tf.einsum("...NI , IH -> ...NH", e, self.edge_kernel_self)
        edgeAtt_self = tf.einsum("...AB -> ...BA", edgeAtt_self)
        attn_coef = attn_coef + edgeAtt_self
        ###
        attn_coef = tf.nn.elu(attn_coef)

        mask = -10e9 * (1.0 - a)
        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        ###
        edge_add = tf.einsum("...NI , IHO -> ...NHO", e, self.edge_kernel_add)
        edge_add = tf.einsum("...ABC -> ...BAC", edge_add)

        bond_sum = tf.einsum("...NHM , ...NHMI -> ...NHMI", attn_coef_drop, edge_add)
        bond_sum = tf.reduce_sum(bond_sum, axis=1)
        bond_sum = tf.einsum("...ABC -> ...BAC", bond_sum)
        ###
        atom_sum = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)
            
        output = tf.concat((atom_sum,bond_sum), axis=-1)
        output = tf.einsum("...NHI , IO -> ...NHO", output, self.output_kernel)

        edge_axis_sum = tf.reduce_sum(e, axis=1)
        bond_out = tf.einsum("...NMI , ...NI -> ...NMI", e, edge_axis_sum)
        edge_transpose = tf.einsum("...ABC -> ...BAC", e)
        bond_out = bond_out - edge_transpose

        return bond_out, output, attn_coef
