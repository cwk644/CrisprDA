��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
�
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
:2*
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
:2*
dtype0
~
dense_230/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_namedense_230/kernel
w
$dense_230/kernel/Read/ReadVariableOpReadVariableOpdense_230/kernel* 
_output_shapes
:
��*
dtype0
u
dense_230/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_230/bias
n
"dense_230/bias/Read/ReadVariableOpReadVariableOpdense_230/bias*
_output_shapes	
:�*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	�*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/conv2d_24/kernel/m
�
+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*&
_output_shapes
:2*
dtype0
�
Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/conv2d_24/bias/m
{
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_230/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/dense_230/kernel/m
�
+Adam/dense_230/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_230/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_230/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/dense_230/bias/m
|
)Adam/dense_230/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_230/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/conv2d_24/kernel/v
�
+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*&
_output_shapes
:2*
dtype0
�
Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/conv2d_24/bias/v
{
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_230/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_nameAdam/dense_230/kernel/v
�
+Adam/dense_230/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_230/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_230/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/dense_230/bias/v
|
)Adam/dense_230/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_230/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�)
value�)B�) B�)
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api

	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratem[m\m]m^(m_)m`vavbvcvd(ve)vf
 
*
0
1
2
3
(4
)5
*
0
1
2
3
(4
)5
�

regularization_losses
	variables
3layer_metrics
4non_trainable_variables

5layers
6layer_regularization_losses
7metrics
trainable_variables
 
\Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
	variables
8layer_metrics
9non_trainable_variables

:layers
;layer_regularization_losses
<metrics
trainable_variables
 
 
 
 
�
regularization_losses
	variables
=layer_metrics
>non_trainable_variables

?layers
@layer_regularization_losses
Ametrics
trainable_variables
 
 
 
�
regularization_losses
	variables
Blayer_metrics
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
Fmetrics
trainable_variables
\Z
VARIABLE_VALUEdense_230/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_230/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
 regularization_losses
!	variables
Glayer_metrics
Hnon_trainable_variables

Ilayers
Jlayer_regularization_losses
Kmetrics
"trainable_variables
 
 
 
�
$regularization_losses
%	variables
Llayer_metrics
Mnon_trainable_variables

Nlayers
Olayer_regularization_losses
Pmetrics
&trainable_variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
*regularization_losses
+	variables
Qlayer_metrics
Rnon_trainable_variables

Slayers
Tlayer_regularization_losses
Umetrics
,trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
0
1
2
3
4
5
6
7
 

V0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Wtotal
	Xcount
Y	variables
Z	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

W0
X1

Y	variables
}
VARIABLE_VALUEAdam/conv2d_24/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_230/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_230/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_24/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_230/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_230/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_onehotPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_24/kernelconv2d_24/biasdense_230/kerneldense_230/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_178961
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$dense_230/kernel/Read/ReadVariableOp"dense_230/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp+Adam/dense_230/kernel/m/Read/ReadVariableOp)Adam/dense_230/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp+Adam/dense_230/kernel/v/Read/ReadVariableOp)Adam/dense_230/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_179263
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_24/kernelconv2d_24/biasdense_230/kerneldense_230/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/dense_230/kernel/mAdam/dense_230/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/dense_230/kernel/vAdam/dense_230/bias/vAdam/output/kernel/vAdam/output/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_179348��
�
d
+__inference_dropout_72_layer_call_fn_179129

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_72_layer_call_and_return_conditional_losses_1787842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�;
�

__inference__traced_save_179263
file_prefix/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_dense_230_kernel_read_readvariableop-
)savev2_dense_230_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop6
2savev2_adam_dense_230_kernel_m_read_readvariableop4
0savev2_adam_dense_230_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop6
2savev2_adam_dense_230_kernel_v_read_readvariableop4
0savev2_adam_dense_230_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_dense_230_kernel_read_readvariableop)savev2_dense_230_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop2savev2_adam_dense_230_kernel_m_read_readvariableop0savev2_adam_dense_230_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop2savev2_adam_dense_230_kernel_v_read_readvariableop0savev2_adam_dense_230_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :2:2:
��:�:	�:: : : : : : : :2:2:
��:�:	�::2:2:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:2: 

_output_shapes
:2:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:2: 

_output_shapes
:2:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::,(
&
_output_shapes
:2: 

_output_shapes
:2:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�
�
(__inference_model_7_layer_call_fn_178888
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_1788562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
e
F__inference_dropout_72_layer_call_and_return_conditional_losses_178784

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_230_layer_call_fn_179108

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_230_layer_call_and_return_conditional_losses_1787092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�*
�
C__inference_model_7_layer_call_and_return_conditional_losses_179028

inputsB
(conv2d_24_conv2d_readvariableop_resource:27
)conv2d_24_biasadd_readvariableop_resource:2<
(dense_230_matmul_readvariableop_resource:
��8
)dense_230_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity�� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp� dense_230/BiasAdd/ReadVariableOp�dense_230/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2Dinputs'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
conv2d_24/Conv2D�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������22
conv2d_24/Relu�
tf.reshape_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_14/Reshape/shape�
tf.reshape_14/ReshapeReshapeconv2d_24/Relu:activations:0$tf.reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_14/Reshape�
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim�
max_pooling1d_2/ExpandDims
ExpandDimstf.reshape_14/Reshape:output:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������22
max_pooling1d_2/ExpandDims�
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:���������
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_2/MaxPool�
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:���������
2*
squeeze_dims
2
max_pooling1d_2/Squeezes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_9/Const�
flatten_9/ReshapeReshape max_pooling1d_2/Squeeze:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_9/Reshape�
dense_230/MatMul/ReadVariableOpReadVariableOp(dense_230_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
dense_230/MatMul/ReadVariableOp�
dense_230/MatMulMatMulflatten_9/Reshape:output:0'dense_230/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_230/MatMul�
 dense_230/BiasAdd/ReadVariableOpReadVariableOp)dense_230_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_230/BiasAdd/ReadVariableOp�
dense_230/BiasAddBiasAdddense_230/MatMul:product:0(dense_230/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_230/BiasAddw
dense_230/ReluReludense_230/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_230/Relu�
dropout_72/IdentityIdentitydense_230/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_72/Identity�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_72/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAdd�
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^dense_230/BiasAdd/ReadVariableOp ^dense_230/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 dense_230/BiasAdd/ReadVariableOp dense_230/BiasAdd/ReadVariableOp2B
dense_230/MatMul/ReadVariableOpdense_230/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_230_layer_call_and_return_conditional_losses_178709

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_178657

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
e
F__inference_dropout_72_layer_call_and_return_conditional_losses_179146

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_model_7_layer_call_fn_178978

inputs!
unknown:2
	unknown_0:2
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_1787392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_output_layer_call_and_return_conditional_losses_179165

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_model_7_layer_call_fn_178995

inputs!
unknown:2
	unknown_0:2
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_1788562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_7_layer_call_and_return_conditional_losses_178936
input_onehot*
conv2d_24_178915:2
conv2d_24_178917:2$
dense_230_178924:
��
dense_230_178926:	� 
output_178930:	�
output_178932:
identity��!conv2d_24/StatefulPartitionedCall�!dense_230/StatefulPartitionedCall�"dropout_72/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_24_178915conv2d_24_178917*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_1786812#
!conv2d_24/StatefulPartitionedCall�
tf.reshape_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_14/Reshape/shape�
tf.reshape_14/ReshapeReshape*conv2d_24/StatefulPartitionedCall:output:0$tf.reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_14/Reshape�
max_pooling1d_2/PartitionedCallPartitionedCalltf.reshape_14/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_1786572!
max_pooling1d_2/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1786962
flatten_9/PartitionedCall�
!dense_230/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_230_178924dense_230_178926*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_230_layer_call_and_return_conditional_losses_1787092#
!dense_230/StatefulPartitionedCall�
"dropout_72/StatefulPartitionedCallStatefulPartitionedCall*dense_230/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_72_layer_call_and_return_conditional_losses_1787842$
"dropout_72/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_72/StatefulPartitionedCall:output:0output_178930output_178932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1787322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^dense_230/StatefulPartitionedCall#^dropout_72/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!dense_230/StatefulPartitionedCall!dense_230/StatefulPartitionedCall2H
"dropout_72/StatefulPartitionedCall"dropout_72/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
(__inference_model_7_layer_call_fn_178754
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_1787392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
C__inference_model_7_layer_call_and_return_conditional_losses_178912
input_onehot*
conv2d_24_178891:2
conv2d_24_178893:2$
dense_230_178900:
��
dense_230_178902:	� 
output_178906:	�
output_178908:
identity��!conv2d_24/StatefulPartitionedCall�!dense_230/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_24_178891conv2d_24_178893*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_1786812#
!conv2d_24/StatefulPartitionedCall�
tf.reshape_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_14/Reshape/shape�
tf.reshape_14/ReshapeReshape*conv2d_24/StatefulPartitionedCall:output:0$tf.reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_14/Reshape�
max_pooling1d_2/PartitionedCallPartitionedCalltf.reshape_14/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_1786572!
max_pooling1d_2/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1786962
flatten_9/PartitionedCall�
!dense_230/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_230_178900dense_230_178902*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_230_layer_call_and_return_conditional_losses_1787092#
!dense_230/StatefulPartitionedCall�
dropout_72/PartitionedCallPartitionedCall*dense_230/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_72_layer_call_and_return_conditional_losses_1787202
dropout_72/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_72/PartitionedCall:output:0output_178906output_178908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1787322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^dense_230/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!dense_230/StatefulPartitionedCall!dense_230/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
E__inference_conv2d_24_layer_call_and_return_conditional_losses_178681

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_dropout_72_layer_call_fn_179124

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_72_layer_call_and_return_conditional_losses_1787202
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_230_layer_call_and_return_conditional_losses_179119

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_72_layer_call_and_return_conditional_losses_179134

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_model_7_layer_call_and_return_conditional_losses_178739

inputs*
conv2d_24_178682:2
conv2d_24_178684:2$
dense_230_178710:
��
dense_230_178712:	� 
output_178733:	�
output_178735:
identity��!conv2d_24/StatefulPartitionedCall�!dense_230/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24_178682conv2d_24_178684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_1786812#
!conv2d_24/StatefulPartitionedCall�
tf.reshape_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_14/Reshape/shape�
tf.reshape_14/ReshapeReshape*conv2d_24/StatefulPartitionedCall:output:0$tf.reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_14/Reshape�
max_pooling1d_2/PartitionedCallPartitionedCalltf.reshape_14/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_1786572!
max_pooling1d_2/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1786962
flatten_9/PartitionedCall�
!dense_230/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_230_178710dense_230_178712*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_230_layer_call_and_return_conditional_losses_1787092#
!dense_230/StatefulPartitionedCall�
dropout_72/PartitionedCallPartitionedCall*dense_230/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_72_layer_call_and_return_conditional_losses_1787202
dropout_72/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_72/PartitionedCall:output:0output_178733output_178735*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1787322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^dense_230/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!dense_230/StatefulPartitionedCall!dense_230/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling1d_2_layer_call_fn_178663

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_1786572
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_24_layer_call_fn_179077

inputs!
unknown:2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_1786812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_output_layer_call_fn_179155

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1787322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_model_7_layer_call_and_return_conditional_losses_178856

inputs*
conv2d_24_178835:2
conv2d_24_178837:2$
dense_230_178844:
��
dense_230_178846:	� 
output_178850:	�
output_178852:
identity��!conv2d_24/StatefulPartitionedCall�!dense_230/StatefulPartitionedCall�"dropout_72/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24_178835conv2d_24_178837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_1786812#
!conv2d_24/StatefulPartitionedCall�
tf.reshape_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_14/Reshape/shape�
tf.reshape_14/ReshapeReshape*conv2d_24/StatefulPartitionedCall:output:0$tf.reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_14/Reshape�
max_pooling1d_2/PartitionedCallPartitionedCalltf.reshape_14/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_1786572!
max_pooling1d_2/PartitionedCall�
flatten_9/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1786962
flatten_9/PartitionedCall�
!dense_230/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_230_178844dense_230_178846*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_230_layer_call_and_return_conditional_losses_1787092#
!dense_230/StatefulPartitionedCall�
"dropout_72/StatefulPartitionedCallStatefulPartitionedCall*dense_230/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_72_layer_call_and_return_conditional_losses_1787842$
"dropout_72/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_72/StatefulPartitionedCall:output:0output_178850output_178852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1787322 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^dense_230/StatefulPartitionedCall#^dropout_72/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!dense_230/StatefulPartitionedCall!dense_230/StatefulPartitionedCall2H
"dropout_72/StatefulPartitionedCall"dropout_72/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
F__inference_dropout_72_layer_call_and_return_conditional_losses_178720

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�3
�
C__inference_model_7_layer_call_and_return_conditional_losses_179068

inputsB
(conv2d_24_conv2d_readvariableop_resource:27
)conv2d_24_biasadd_readvariableop_resource:2<
(dense_230_matmul_readvariableop_resource:
��8
)dense_230_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity�� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp� dense_230/BiasAdd/ReadVariableOp�dense_230/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_24/Conv2D/ReadVariableOp�
conv2d_24/Conv2DConv2Dinputs'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
conv2d_24/Conv2D�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������22
conv2d_24/Relu�
tf.reshape_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_14/Reshape/shape�
tf.reshape_14/ReshapeReshapeconv2d_24/Relu:activations:0$tf.reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_14/Reshape�
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim�
max_pooling1d_2/ExpandDims
ExpandDimstf.reshape_14/Reshape:output:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������22
max_pooling1d_2/ExpandDims�
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:���������
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_2/MaxPool�
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:���������
2*
squeeze_dims
2
max_pooling1d_2/Squeezes
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_9/Const�
flatten_9/ReshapeReshape max_pooling1d_2/Squeeze:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_9/Reshape�
dense_230/MatMul/ReadVariableOpReadVariableOp(dense_230_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
dense_230/MatMul/ReadVariableOp�
dense_230/MatMulMatMulflatten_9/Reshape:output:0'dense_230/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_230/MatMul�
 dense_230/BiasAdd/ReadVariableOpReadVariableOp)dense_230_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_230/BiasAdd/ReadVariableOp�
dense_230/BiasAddBiasAdddense_230/MatMul:product:0(dense_230/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_230/BiasAddw
dense_230/ReluReludense_230/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_230/Reluy
dropout_72/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_72/dropout/Const�
dropout_72/dropout/MulMuldense_230/Relu:activations:0!dropout_72/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_72/dropout/Mul�
dropout_72/dropout/ShapeShapedense_230/Relu:activations:0*
T0*
_output_shapes
:2
dropout_72/dropout/Shape�
/dropout_72/dropout/random_uniform/RandomUniformRandomUniform!dropout_72/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_72/dropout/random_uniform/RandomUniform�
!dropout_72/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2#
!dropout_72/dropout/GreaterEqual/y�
dropout_72/dropout/GreaterEqualGreaterEqual8dropout_72/dropout/random_uniform/RandomUniform:output:0*dropout_72/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_72/dropout/GreaterEqual�
dropout_72/dropout/CastCast#dropout_72/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_72/dropout/Cast�
dropout_72/dropout/Mul_1Muldropout_72/dropout/Mul:z:0dropout_72/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_72/dropout/Mul_1�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_72/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAdd�
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^dense_230/BiasAdd/ReadVariableOp ^dense_230/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 dense_230/BiasAdd/ReadVariableOp dense_230/BiasAdd/ReadVariableOp2B
dense_230/MatMul/ReadVariableOpdense_230/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_output_layer_call_and_return_conditional_losses_178732

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�m
�
"__inference__traced_restore_179348
file_prefix;
!assignvariableop_conv2d_24_kernel:2/
!assignvariableop_1_conv2d_24_bias:27
#assignvariableop_2_dense_230_kernel:
��0
!assignvariableop_3_dense_230_bias:	�3
 assignvariableop_4_output_kernel:	�,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: E
+assignvariableop_13_adam_conv2d_24_kernel_m:27
)assignvariableop_14_adam_conv2d_24_bias_m:2?
+assignvariableop_15_adam_dense_230_kernel_m:
��8
)assignvariableop_16_adam_dense_230_bias_m:	�;
(assignvariableop_17_adam_output_kernel_m:	�4
&assignvariableop_18_adam_output_bias_m:E
+assignvariableop_19_adam_conv2d_24_kernel_v:27
)assignvariableop_20_adam_conv2d_24_bias_v:2?
+assignvariableop_21_adam_dense_230_kernel_v:
��8
)assignvariableop_22_adam_dense_230_bias_v:	�;
(assignvariableop_23_adam_output_kernel_v:	�4
&assignvariableop_24_adam_output_bias_v:
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_24_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_230_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_230_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_24_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_24_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_230_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_230_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_24_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_24_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_230_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_230_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_output_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_output_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25�
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_179099

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
2:S O
+
_output_shapes
:���������
2
 
_user_specified_nameinputs
�
F
*__inference_flatten_9_layer_call_fn_179093

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_1786962
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
2:S O
+
_output_shapes
:���������
2
 
_user_specified_nameinputs
�
�
E__inference_conv2d_24_layer_call_and_return_conditional_losses_179088

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�0
�
!__inference__wrapped_model_178648
input_onehotJ
0model_7_conv2d_24_conv2d_readvariableop_resource:2?
1model_7_conv2d_24_biasadd_readvariableop_resource:2D
0model_7_dense_230_matmul_readvariableop_resource:
��@
1model_7_dense_230_biasadd_readvariableop_resource:	�@
-model_7_output_matmul_readvariableop_resource:	�<
.model_7_output_biasadd_readvariableop_resource:
identity��(model_7/conv2d_24/BiasAdd/ReadVariableOp�'model_7/conv2d_24/Conv2D/ReadVariableOp�(model_7/dense_230/BiasAdd/ReadVariableOp�'model_7/dense_230/MatMul/ReadVariableOp�%model_7/output/BiasAdd/ReadVariableOp�$model_7/output/MatMul/ReadVariableOp�
'model_7/conv2d_24/Conv2D/ReadVariableOpReadVariableOp0model_7_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02)
'model_7/conv2d_24/Conv2D/ReadVariableOp�
model_7/conv2d_24/Conv2DConv2Dinput_onehot/model_7/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
model_7/conv2d_24/Conv2D�
(model_7/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp1model_7_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(model_7/conv2d_24/BiasAdd/ReadVariableOp�
model_7/conv2d_24/BiasAddBiasAdd!model_7/conv2d_24/Conv2D:output:00model_7/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22
model_7/conv2d_24/BiasAdd�
model_7/conv2d_24/ReluRelu"model_7/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:���������22
model_7/conv2d_24/Relu�
#model_7/tf.reshape_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2%
#model_7/tf.reshape_14/Reshape/shape�
model_7/tf.reshape_14/ReshapeReshape$model_7/conv2d_24/Relu:activations:0,model_7/tf.reshape_14/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
model_7/tf.reshape_14/Reshape�
&model_7/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_7/max_pooling1d_2/ExpandDims/dim�
"model_7/max_pooling1d_2/ExpandDims
ExpandDims&model_7/tf.reshape_14/Reshape:output:0/model_7/max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������22$
"model_7/max_pooling1d_2/ExpandDims�
model_7/max_pooling1d_2/MaxPoolMaxPool+model_7/max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:���������
2*
ksize
*
paddingVALID*
strides
2!
model_7/max_pooling1d_2/MaxPool�
model_7/max_pooling1d_2/SqueezeSqueeze(model_7/max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:���������
2*
squeeze_dims
2!
model_7/max_pooling1d_2/Squeeze�
model_7/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
model_7/flatten_9/Const�
model_7/flatten_9/ReshapeReshape(model_7/max_pooling1d_2/Squeeze:output:0 model_7/flatten_9/Const:output:0*
T0*(
_output_shapes
:����������2
model_7/flatten_9/Reshape�
'model_7/dense_230/MatMul/ReadVariableOpReadVariableOp0model_7_dense_230_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02)
'model_7/dense_230/MatMul/ReadVariableOp�
model_7/dense_230/MatMulMatMul"model_7/flatten_9/Reshape:output:0/model_7/dense_230/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_7/dense_230/MatMul�
(model_7/dense_230/BiasAdd/ReadVariableOpReadVariableOp1model_7_dense_230_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_7/dense_230/BiasAdd/ReadVariableOp�
model_7/dense_230/BiasAddBiasAdd"model_7/dense_230/MatMul:product:00model_7/dense_230/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_7/dense_230/BiasAdd�
model_7/dense_230/ReluRelu"model_7/dense_230/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model_7/dense_230/Relu�
model_7/dropout_72/IdentityIdentity$model_7/dense_230/Relu:activations:0*
T0*(
_output_shapes
:����������2
model_7/dropout_72/Identity�
$model_7/output/MatMul/ReadVariableOpReadVariableOp-model_7_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$model_7/output/MatMul/ReadVariableOp�
model_7/output/MatMulMatMul$model_7/dropout_72/Identity:output:0,model_7/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_7/output/MatMul�
%model_7/output/BiasAdd/ReadVariableOpReadVariableOp.model_7_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_7/output/BiasAdd/ReadVariableOp�
model_7/output/BiasAddBiasAddmodel_7/output/MatMul:product:0-model_7/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_7/output/BiasAdd�
IdentityIdentitymodel_7/output/BiasAdd:output:0)^model_7/conv2d_24/BiasAdd/ReadVariableOp(^model_7/conv2d_24/Conv2D/ReadVariableOp)^model_7/dense_230/BiasAdd/ReadVariableOp(^model_7/dense_230/MatMul/ReadVariableOp&^model_7/output/BiasAdd/ReadVariableOp%^model_7/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2T
(model_7/conv2d_24/BiasAdd/ReadVariableOp(model_7/conv2d_24/BiasAdd/ReadVariableOp2R
'model_7/conv2d_24/Conv2D/ReadVariableOp'model_7/conv2d_24/Conv2D/ReadVariableOp2T
(model_7/dense_230/BiasAdd/ReadVariableOp(model_7/dense_230/BiasAdd/ReadVariableOp2R
'model_7/dense_230/MatMul/ReadVariableOp'model_7/dense_230/MatMul/ReadVariableOp2N
%model_7/output/BiasAdd/ReadVariableOp%model_7/output/BiasAdd/ReadVariableOp2L
$model_7/output/MatMul/ReadVariableOp$model_7/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
$__inference_signature_wrapper_178961
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_1786482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_178696

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
2:S O
+
_output_shapes
:���������
2
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
input_onehot=
serving_default_input_onehot:0���������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
g_default_save_signature
h__call__
*i&call_and_return_all_conditional_losses"�<
_tf_keras_network�;{"name": "model_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_14", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_14", "inbound_nodes": [["conv2d_24", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_2", "inbound_nodes": [[["tf.reshape_14", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["max_pooling1d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_230", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_230", "inbound_nodes": [[["flatten_9", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_72", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_72", "inbound_nodes": [[["dense_230", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_72", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_14", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_14", "inbound_nodes": [["conv2d_24", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_2", "inbound_nodes": [[["tf.reshape_14", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["max_pooling1d_2", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_230", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_230", "inbound_nodes": [[["flatten_9", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_72", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_72", "inbound_nodes": [[["dense_230", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_72", 0, 0, {}]]], "shared_object_id": 13}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}}
�
	keras_api"�
_tf_keras_layer�{"name": "tf.reshape_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.reshape_14", "trainable": true, "dtype": "float32", "function": "reshape"}, "inbound_nodes": [["conv2d_24", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}
�
regularization_losses
	variables
trainable_variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "max_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["tf.reshape_14", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 17}}
�
regularization_losses
	variables
trainable_variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["max_pooling1d_2", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
�	

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
p__call__
*q&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_230", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_230", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_9", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
�
$regularization_losses
%	variables
&trainable_variables
'	keras_api
r__call__
*s&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dropout_72", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_72", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_230", 0, 0, {}]]], "shared_object_id": 10}
�	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
t__call__
*u&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_72", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratem[m\m]m^(m_)m`vavbvcvd(ve)vf"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
(4
)5"
trackable_list_wrapper
J
0
1
2
3
(4
)5"
trackable_list_wrapper
�

regularization_losses
	variables
3layer_metrics
4non_trainable_variables

5layers
6layer_regularization_losses
7metrics
trainable_variables
h__call__
g_default_save_signature
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
*:(22conv2d_24/kernel
:22conv2d_24/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
	variables
8layer_metrics
9non_trainable_variables

:layers
;layer_regularization_losses
<metrics
trainable_variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
	variables
=layer_metrics
>non_trainable_variables

?layers
@layer_regularization_losses
Ametrics
trainable_variables
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
	variables
Blayer_metrics
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
Fmetrics
trainable_variables
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
$:"
��2dense_230/kernel
:�2dense_230/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 regularization_losses
!	variables
Glayer_metrics
Hnon_trainable_variables

Ilayers
Jlayer_regularization_losses
Kmetrics
"trainable_variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
$regularization_losses
%	variables
Llayer_metrics
Mnon_trainable_variables

Nlayers
Olayer_regularization_losses
Pmetrics
&trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 :	�2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
*regularization_losses
+	variables
Qlayer_metrics
Rnon_trainable_variables

Slayers
Tlayer_regularization_losses
Umetrics
,trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Wtotal
	Xcount
Y	variables
Z	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 21}
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
/:-22Adam/conv2d_24/kernel/m
!:22Adam/conv2d_24/bias/m
):'
��2Adam/dense_230/kernel/m
": �2Adam/dense_230/bias/m
%:#	�2Adam/output/kernel/m
:2Adam/output/bias/m
/:-22Adam/conv2d_24/kernel/v
!:22Adam/conv2d_24/bias/v
):'
��2Adam/dense_230/kernel/v
": �2Adam/dense_230/bias/v
%:#	�2Adam/output/kernel/v
:2Adam/output/bias/v
�2�
!__inference__wrapped_model_178648�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+
input_onehot���������
�2�
(__inference_model_7_layer_call_fn_178754
(__inference_model_7_layer_call_fn_178978
(__inference_model_7_layer_call_fn_178995
(__inference_model_7_layer_call_fn_178888�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_model_7_layer_call_and_return_conditional_losses_179028
C__inference_model_7_layer_call_and_return_conditional_losses_179068
C__inference_model_7_layer_call_and_return_conditional_losses_178912
C__inference_model_7_layer_call_and_return_conditional_losses_178936�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_conv2d_24_layer_call_fn_179077�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_24_layer_call_and_return_conditional_losses_179088�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_max_pooling1d_2_layer_call_fn_178663�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_178657�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
*__inference_flatten_9_layer_call_fn_179093�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_flatten_9_layer_call_and_return_conditional_losses_179099�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_230_layer_call_fn_179108�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_230_layer_call_and_return_conditional_losses_179119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dropout_72_layer_call_fn_179124
+__inference_dropout_72_layer_call_fn_179129�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_72_layer_call_and_return_conditional_losses_179134
F__inference_dropout_72_layer_call_and_return_conditional_losses_179146�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_output_layer_call_fn_179155�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_output_layer_call_and_return_conditional_losses_179165�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_178961input_onehot"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_178648x()=�:
3�0
.�+
input_onehot���������
� "/�,
*
output �
output����������
E__inference_conv2d_24_layer_call_and_return_conditional_losses_179088l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������2
� �
*__inference_conv2d_24_layer_call_fn_179077_7�4
-�*
(�%
inputs���������
� " ����������2�
E__inference_dense_230_layer_call_and_return_conditional_losses_179119^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_230_layer_call_fn_179108Q0�-
&�#
!�
inputs����������
� "������������
F__inference_dropout_72_layer_call_and_return_conditional_losses_179134^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_dropout_72_layer_call_and_return_conditional_losses_179146^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
+__inference_dropout_72_layer_call_fn_179124Q4�1
*�'
!�
inputs����������
p 
� "������������
+__inference_dropout_72_layer_call_fn_179129Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_flatten_9_layer_call_and_return_conditional_losses_179099]3�0
)�&
$�!
inputs���������
2
� "&�#
�
0����������
� ~
*__inference_flatten_9_layer_call_fn_179093P3�0
)�&
$�!
inputs���������
2
� "������������
K__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_178657�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
0__inference_max_pooling1d_2_layer_call_fn_178663wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
C__inference_model_7_layer_call_and_return_conditional_losses_178912v()E�B
;�8
.�+
input_onehot���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_7_layer_call_and_return_conditional_losses_178936v()E�B
;�8
.�+
input_onehot���������
p

 
� "%�"
�
0���������
� �
C__inference_model_7_layer_call_and_return_conditional_losses_179028p()?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_7_layer_call_and_return_conditional_losses_179068p()?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������
� �
(__inference_model_7_layer_call_fn_178754i()E�B
;�8
.�+
input_onehot���������
p 

 
� "�����������
(__inference_model_7_layer_call_fn_178888i()E�B
;�8
.�+
input_onehot���������
p

 
� "�����������
(__inference_model_7_layer_call_fn_178978c()?�<
5�2
(�%
inputs���������
p 

 
� "�����������
(__inference_model_7_layer_call_fn_178995c()?�<
5�2
(�%
inputs���������
p

 
� "�����������
B__inference_output_layer_call_and_return_conditional_losses_179165]()0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_output_layer_call_fn_179155P()0�-
&�#
!�
inputs����������
� "�����������
$__inference_signature_wrapper_178961�()M�J
� 
C�@
>
input_onehot.�+
input_onehot���������"/�,
*
output �
output���������