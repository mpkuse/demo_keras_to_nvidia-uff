
from keras import backend as K
from keras.engine.topology import Layer
import keras
import code
import numpy as np

# import cv2
import code


class NetVLADLayer( Layer ):

    def __init__( self, num_clusters, **kwargs ):
        self.num_clusters = num_clusters
        super(NetVLADLayer, self).__init__(**kwargs)

    def build( self, input_shape ):
        self.K = self.num_clusters
        self.D = input_shape[-1]


        self.kernel = self.add_weight( name='kernel',
                                    shape=(1,1,self.D,self.K),
                                    initializer='uniform',
                                    trainable=True )

        self.bias = self.add_weight( name='bias',
                                    shape=(1,1,self.K),
                                    initializer='uniform',
                                    trainable=True )

        self.C = self.add_weight( name='cluster_centers',
                                shape=[1,1,1,self.D,self.K],
                                initializer='uniform',
                                trainable=True)

    def call( self, x ):
        print 'input x.shape=', x.shape
        # soft-assignment.
        s = K.conv2d( x, self.kernel, padding='same' ) + self.bias
        print 's.shape=', s.shape
        a = K.softmax( s )
        print 'a.shape=',a.shape

        self.amap = K.argmax( a, -1 ) #<----- currently not needed for output. if need be uncomment this and will also have to change compute_output_shape
        print 'amap.shape', self.amap.shape

        # import code
        # code.interact( local=locals() )
        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        print 'a.shape (before)=', a.shape
        # a = K.expand_dims( a, -2 ) #original code
        # a = K.reshape( a, [ K.shape(a)[0], K.shape(a)[1], K.shape(a)[2], 1, K.shape(a)[3]] ) # I think only for unknown shapes should use K.shape(a)[0] etc
        a = K.reshape( a, [ K.shape(a)[0], a.shape[1].value, a.shape[2].value, 1, a.shape[3].value ] )
        print 'a.shape=',a.shape


        # Core
        print 'x.shape', x.shape
        # v = K.expand_dims(x, -1) + self.C #original code
        v_tmp = K.reshape( x, [ K.shape(x)[0],  x.shape[1].value, x.shape[2].value, x.shape[3].value, 1 ] )
        print 'v_tmp.shape', v_tmp.shape, '\tself.C.shape', self.C.shape
        v = v_tmp + self.C
        print 'v.shape', v.shape
        return v
        #-------------------------------------------#
        v = a * v
        # print 'v.shape', v.shape
        v = K.sum(v, axis=[1, 2])
        # print 'v.shape', v.shape
        v = K.permute_dimensions(v, pattern=[0, 2, 1])
        print 'v.shape', v.shape
        #v.shape = None x K x D

        # Normalize v (Intra Normalization)
        v = K.l2_normalize( v, axis=-1 )
        v = K.batch_flatten( v )
        v = K.l2_normalize( v, axis=-1 )

        # return [v, self.amap]
        print 'v.shape (final)', v.shape
        return v

    def compute_output_shape( self, input_shape ):

        # return [(input_shape[0], self.K*self.D ), (input_shape[0], input_shape[1], input_shape[2]) ]
        # return (input_shape[0], self.K*self.D )

        # return (input_shape[0], input_shape[1], input_shape[2], 1, self.K) #s
        return (input_shape[0], input_shape[1], input_shape[2], self.D, self.K) #s


    def get_config( self ):
        pass
        # base_config = super(NetVLADLayer, self).get_config()
        # return dict(list(base_config.items()))

        # As suggested by: https://github.com/keras-team/keras/issues/4871#issuecomment-269731817
        config = {'num_clusters': self.num_clusters}
        base_config = super(NetVLADLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
