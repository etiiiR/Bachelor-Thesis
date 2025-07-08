# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import os
import tensorflow.contrib.layers as tfcontriblayers

from modules.losses import mesh_loss_2, laplace_loss_2
from modules.layers import LocalGraphProjection, SampleHypothesis, DeformationReasoning


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'suffix'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        save_dir_suffix = kwargs.get('suffix', '')
        self.save_dir_suffix = save_dir_suffix

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output1_2 = None
        self.output2_2 = None

        self.loss = 0
        self.inc_loss = 0
        self.pose_loss = 0
        self.optimizer = None
        self.optimizer_inc = None
        self.opt_op_vp = None
        self.opt_op_vi = None
        self.opt_op = None
        self.summary = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None, ckpt_path=None, step=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')
        saver = tf.train.Saver(self.vars, max_to_keep=0)
        save_path = saver.save(sess, os.path.join(ckpt_path, '{}.ckpt'.format(self.name)), global_step=step)
        print('Model saved in file: {}, epoch {}'.format(save_path, step))

    def load(self, sess=None, ckpt_path=None, step=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')
        
        save_path = os.path.join(ckpt_path, '{}.ckpt-{}'.format(self.name, step))
        
        # Check if the checkpoint exists using TensorFlow's built-in method
        if not tf.train.checkpoint_exists(save_path):
            # Try alternative names in order of preference
            alternative_names = ['meshnet', 'meshnetp2mpp']
            found_path = None
            
            for alt_name in alternative_names:
                alt_save_path = os.path.join(ckpt_path, f'{alt_name}.ckpt-{step}')
                if tf.train.checkpoint_exists(alt_save_path):
                    found_path = alt_save_path
                    break
            
            if found_path:
                save_path = found_path
                print(f"Using alternative checkpoint: {save_path}")
            else:
                # List available checkpoints for debugging
                print(f"Checkpoint not found at: {save_path}")
                if os.path.exists(ckpt_path):
                    print(f"Available files in {ckpt_path}:")
                    for f in os.listdir(ckpt_path):
                        print(f"  {f}")
                raise FileNotFoundError(f"Checkpoint not found: {save_path}")
        
        # Create flexible variable mapping that handles scope mismatches
        print(f"Loading P2MPP checkpoint from: {save_path}")
        
        # Read checkpoint to understand its structure
        checkpoint_reader = tf.train.NewCheckpointReader(save_path)
        checkpoint_vars = checkpoint_reader.get_variable_to_shape_map()
        print(f"P2MPP Checkpoint contains {len(checkpoint_vars)} variables")
        
        try:
            # For P2MPP, filter out optimizer variables to only load model weights
            model_vars = []
            vars_list = list(self.vars.values())  # Get the actual variables, not keys
            for var in vars_list:
                # Skip Adam optimizer variables
                if 'Adam' not in var.name and 'beta' not in var.name and 'global_step' not in var.name:
                    model_vars.append(var)
            
            print(f"Filtered {len(model_vars)}/{len(vars_list)} variables (excluded optimizer state)")
            saver = tf.train.Saver(model_vars)
            saver.restore(sess, save_path)
            print('=> P2MPP Model restored from file: {}, epoch {}'.format(save_path, step))
            
        except Exception as e:
            print(f"Standard P2MPP loading failed: {e}")
            print("Attempting flexible mapping...")
            try:
                # Create flexible mapping between current variables and checkpoint variables
                var_list = {}
                unmapped_vars = []
                vars_list = list(self.vars.values())  # Get the actual variables, not keys
                
                for var in vars_list:
                    var_name = var.name
                    mapped = False
                    
                    # Skip Adam optimizer variables
                    if 'Adam' in var_name or 'beta' in var_name or 'global_step' in var_name:
                        continue
                    
                    # Try different scope mappings for P2MPP
                    possible_names = [
                        var_name,  # exact match
                        var_name.replace('meshnet/', 'meshnetp2mpp/'),  # meshnet -> p2mpp  
                        var_name.replace('meshnetp2mpp/', 'meshnet/'),  # p2mpp -> meshnet
                        var_name.replace('meshnet/', ''),  # remove meshnet scope
                        var_name.replace('meshnetp2mpp/', ''),  # remove p2mpp scope
                    ]
                    
                    for possible_name in possible_names:
                        if possible_name in checkpoint_vars:
                            var_list[var_name] = var  # Map variable name to variable object
                            mapped = True
                            break
                    
                    if not mapped:
                        unmapped_vars.append(var_name)
                
                if unmapped_vars:
                    print(f"Warning: Could not map {len(unmapped_vars)} P2MPP variables:")
                    for var_name in unmapped_vars[:10]:  # Show first 10
                        print(f"  - {var_name}")
                
                if not var_list:
                    raise ValueError("No matching variables found between P2MPP checkpoint and current model")
                
                print(f"Successfully mapped {len(var_list)}/{len(self.vars)} P2MPP variables")
                
                # Create saver with mapped variables and restore
                saver = tf.train.Saver(var_list)
                saver.restore(sess, save_path)
                print('=> P2MPP Model restored with flexible mapping from file: {}, epoch {}'.format(save_path, step))
                
            except Exception as fallback_error:
                print(f"Flexible P2MPP mapping also failed: {fallback_error}")
                raise e  # Re-raise the original error


class MeshNet(Model):
    def __init__(self, placeholders, args, **kwargs):
        super(MeshNet, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.placeholders = placeholders
        self.args = args
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        self.summary_loss = None
        self.merged_summary_op = None
        self.output1l = None
        self.output2l = None
        self.sample1 = None
        self.sample2 = None
        self.proj1 = None
        self.proj2 = None
        self.drb1 = None
        self.drb2 = None
        self.build()

    def loadcnn(self, sess=None, ckpt_path=None, step=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')
        
        # Try primary checkpoint path first
        save_path = os.path.join(ckpt_path, '{}.ckpt-{}'.format(self.name, step))
        
        # Check if checkpoint exists, if not try alternative names
        if not tf.train.checkpoint_exists(save_path):
            # Try meshnetmvp2m as fallback
            alt_save_path = os.path.join(ckpt_path, 'meshnetmvp2m.ckpt-{}'.format(step))
            if tf.train.checkpoint_exists(alt_save_path):
                save_path = alt_save_path
            else:
                # List available checkpoints for debugging
                print(f"CNN checkpoint not found at: {save_path}")
                print(f"Also tried: {alt_save_path}")
                if os.path.exists(ckpt_path):
                    print(f"Available files in {ckpt_path}:")
                    for f in os.listdir(ckpt_path):
                        print(f"  {f}")
                raise FileNotFoundError(f"CNN checkpoint not found: {save_path}")
        
        # Create flexible variable mapping that handles scope mismatches
        checkpoint_reader = tf.train.NewCheckpointReader(save_path)
        checkpoint_vars = checkpoint_reader.get_variable_to_shape_map()
        
        # Get current CNN variables
        current_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='meshnet/cnn/')
        
        # Create flexible mapping
        var_list = {}
        for var in current_vars:
            var_name = var.name
            # Try different scope mappings
            possible_names = [
                var_name,  # exact match
                var_name.replace('meshnet/', 'meshnetmvp2m/'),  # mvp2m scope
                var_name.replace('meshnet/', ''),  # no scope prefix
            ]
            
            found = False
            for possible_name in possible_names:
                if possible_name in checkpoint_vars:
                    var_list[possible_name] = var
                    print(f"Mapping: {possible_name} -> {var_name}")
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find checkpoint variable for {var_name}")
        
        if not var_list:
            raise ValueError("No matching variables found between checkpoint and current model")
        
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path)
        print('=> !!CNN restored from file: {}, epoch {}'.format(save_path, step))

    def _loss(self):
        # Pixel2mesh loss
        self.loss += mesh_loss_2(self.output1l, self.placeholders, 3)
        self.loss += laplace_loss_2(self.inputs, self.output1l, self.placeholders, 3)
        self.loss += mesh_loss_2(self.output2l, self.placeholders, 3)
        self.loss += laplace_loss_2(self.output1l, self.output2l, self.placeholders, 3)

        conv_layers = [self.drb1, self.drb2]
        for l in conv_layers:
            for var in l.vars.values():
                self.loss += 5e-6 * tf.nn.l2_loss(var)

    def _build(self):
        with tf.name_scope('pixel2mesh'):
            self.build_cnn18()  # update image feature
            # sample hypothesis points
            self.sample1 = SampleHypothesis(placeholders=self.placeholders, name='graph_sample_hypothesis_1_layer_0')
            # 1st projection block
            self.proj1 = LocalGraphProjection(placeholders=self.placeholders, name='graph_localproj_1_layer_1')
            # 1st DRB
            self.drb1 = DeformationReasoning(input_dim=self.args.stage2_feat_dim,
                                             output_dim=3,
                                             placeholders=self.placeholders,
                                             gcn_block=3,
                                             args=self.args,
                                             name='graph_drb_blk1_layer_2')
            # sample hypothesis points
            self.sample2 = SampleHypothesis(placeholders=self.placeholders, name='graph_sample_hypothesis_2_layer_3')
            # 2nd projection block
            self.proj2 = LocalGraphProjection(placeholders=self.placeholders, name='graph_localproj_2_layer_4')
            # 2nd DRB
            self.drb2 = DeformationReasoning(input_dim=self.args.stage2_feat_dim,
                                             output_dim=3,
                                             placeholders=self.placeholders,
                                             gcn_block=3,
                                             args=self.args,
                                             name='graph_drb_blk2_layer_5')

    def build_cnn18(self):
        x = self.placeholders['img_inp']
        # x = tf.expand_dims(x, 0)
# 224 224
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_1')
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_2')
        x0 = x
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_3')
# 112 112
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_4')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_5')
        x1 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_6')
# 56 56
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_7')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_8')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_9')
# 28 28
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_10')
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_11')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_12')
# 14 14
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_13')
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_14')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_15')
# 7 7
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_16')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_17')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_18')
        x5 = x
        # updata image feature
        self.placeholders.update({'img_feat': [tf.squeeze(x0), tf.squeeze(x1), tf.squeeze(x2)]})
        self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.3

    def build(self):
        ''' Wrapper for _build() '''
        with tf.variable_scope(self.name):
            self._build()

        blk1_sample = self.sample1(self.inputs)
        blk1_proj_feat = self.proj1(blk1_sample)
        blk1_out = self.drb1((blk1_proj_feat, self.inputs))

        blk2_sample = self.sample2(blk1_out)
        blk2_proj_feat = self.proj2(blk2_sample)
        blk2_out = self.drb2((blk2_proj_feat, blk1_out))

        self.output1l = blk1_out
        self.output2l = blk2_out

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/')
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)

        self.summary_loss = tf.summary.scalar('loss', self.loss)
        self.merged_summary_op = tf.summary.merge_all()
