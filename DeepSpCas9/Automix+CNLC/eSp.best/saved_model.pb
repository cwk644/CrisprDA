��
��
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
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
�
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
:�*
dtype0
u
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_17/bias
n
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes	
:�*
dtype0

dense_2721/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*"
shared_namedense_2721/kernel
x
%dense_2721/kernel/Read/ReadVariableOpReadVariableOpdense_2721/kernel*
_output_shapes
:	�P*
dtype0
v
dense_2721/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P* 
shared_namedense_2721/bias
o
#dense_2721/bias/Read/ReadVariableOpReadVariableOpdense_2721/bias*
_output_shapes
:P*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:P*
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
Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/conv2d_17/kernel/m
�
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*'
_output_shapes
:�*
dtype0
�
Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_17/bias/m
|
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_2721/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*)
shared_nameAdam/dense_2721/kernel/m
�
,Adam/dense_2721/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2721/kernel/m*
_output_shapes
:	�P*
dtype0
�
Adam/dense_2721/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*'
shared_nameAdam/dense_2721/bias/m
}
*Adam/dense_2721/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2721/bias/m*
_output_shapes
:P*
dtype0
�
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:P*
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
Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/conv2d_17/kernel/v
�
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*'
_output_shapes
:�*
dtype0
�
Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_17/bias/v
|
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_2721/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*)
shared_nameAdam/dense_2721/kernel/v
�
,Adam/dense_2721/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2721/kernel/v*
_output_shapes
:	�P*
dtype0
�
Adam/dense_2721/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*'
shared_nameAdam/dense_2721/bias/v
}
*Adam/dense_2721/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2721/bias/v*
_output_shapes
:P*
dtype0
�
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:P*
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
�.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�.
value�.B�. B�.
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
layer-7
	layer_with_weights-2
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
�
6iter

7beta_1

8beta_2
	9decay
:learning_ratemmmn"mo#mp0mq1mrvsvt"vu#vv0vw1vx
*
0
1
"2
#3
04
15
 
*
0
1
"2
#3
04
15
�
trainable_variables
;layer_metrics
<non_trainable_variables
regularization_losses
=layer_regularization_losses
>metrics

?layers
	variables
 
\Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
@layer_metrics
Anon_trainable_variables
regularization_losses
Blayer_regularization_losses
Cmetrics

Dlayers
	variables
 
 
 
�
trainable_variables
Elayer_metrics
Fnon_trainable_variables
regularization_losses
Glayer_regularization_losses
Hmetrics

Ilayers
	variables
 
 
 
�
trainable_variables
Jlayer_metrics
Knon_trainable_variables
regularization_losses
Llayer_regularization_losses
Mmetrics

Nlayers
	variables
 
 
 
�
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
 	variables
][
VARIABLE_VALUEdense_2721/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_2721/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
�
$trainable_variables
Tlayer_metrics
Unon_trainable_variables
%regularization_losses
Vlayer_regularization_losses
Wmetrics

Xlayers
&	variables
 
 
 
�
(trainable_variables
Ylayer_metrics
Znon_trainable_variables
)regularization_losses
[layer_regularization_losses
\metrics

]layers
*	variables
 
 
 
�
,trainable_variables
^layer_metrics
_non_trainable_variables
-regularization_losses
`layer_regularization_losses
ametrics

blayers
.	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
�
2trainable_variables
clayer_metrics
dnon_trainable_variables
3regularization_losses
elayer_regularization_losses
fmetrics

glayers
4	variables
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
 

h0
?
0
1
2
3
4
5
6
7
	8
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
	itotal
	jcount
k	variables
l	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

k	variables
}
VARIABLE_VALUEAdam/conv2d_17/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_2721/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_2721/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_17/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/dense_2721/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_2721/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_onehotPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_17/kernelconv2d_17/biasdense_2721/kerneldense_2721/biasoutput/kerneloutput/bias*
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
GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_2890910
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp%dense_2721/kernel/Read/ReadVariableOp#dense_2721/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp,Adam/dense_2721/kernel/m/Read/ReadVariableOp*Adam/dense_2721/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp,Adam/dense_2721/kernel/v/Read/ReadVariableOp*Adam/dense_2721/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
GPU2 *0J 8� *)
f$R"
 __inference__traced_save_2891281
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_17/kernelconv2d_17/biasdense_2721/kerneldense_2721/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/mAdam/dense_2721/kernel/mAdam/dense_2721/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/vAdam/dense_2721/kernel/vAdam/dense_2721/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
GPU2 *0J 8� *,
f'R%
#__inference__traced_restore_2891366��
�m
�
#__inference__traced_restore_2891366
file_prefix<
!assignvariableop_conv2d_17_kernel:�0
!assignvariableop_1_conv2d_17_bias:	�7
$assignvariableop_2_dense_2721_kernel:	�P0
"assignvariableop_3_dense_2721_bias:P2
 assignvariableop_4_output_kernel:P,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: F
+assignvariableop_13_adam_conv2d_17_kernel_m:�8
)assignvariableop_14_adam_conv2d_17_bias_m:	�?
,assignvariableop_15_adam_dense_2721_kernel_m:	�P8
*assignvariableop_16_adam_dense_2721_bias_m:P:
(assignvariableop_17_adam_output_kernel_m:P4
&assignvariableop_18_adam_output_bias_m:F
+assignvariableop_19_adam_conv2d_17_kernel_v:�8
)assignvariableop_20_adam_conv2d_17_bias_v:	�?
,assignvariableop_21_adam_dense_2721_kernel_v:	�P8
*assignvariableop_22_adam_dense_2721_bias_v:P:
(assignvariableop_23_adam_output_kernel_v:P4
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_17_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_17_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_2721_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_2721_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_17_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_17_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_2721_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_2721_bias_mIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_17_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_17_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_2721_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_2721_bias_vIdentity_22:output:0"/device:CPU:0*
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
�
F
*__inference_re_lu_17_layer_call_fn_2891052

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_re_lu_17_layer_call_and_return_conditional_losses_28905712
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_911_layer_call_fn_2891164

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_911_layer_call_and_return_conditional_losses_28906812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�

�
G__inference_dense_2721_layer_call_and_return_conditional_losses_2890599

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_17_layer_call_fn_2891042

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_28905602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_model_56_layer_call_fn_2890837
input_onehot"
unknown:�
	unknown_0:	�
	unknown_1:	�P
	unknown_2:P
	unknown_3:P
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
GPU2 *0J 8� *N
fIRG
E__inference_model_56_layer_call_and_return_conditional_losses_28908052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�"
�
E__inference_model_56_layer_call_and_return_conditional_losses_2890861
input_onehot,
conv2d_17_2890840:� 
conv2d_17_2890842:	�%
dense_2721_2890848:	�P 
dense_2721_2890850:P 
output_2890855:P
output_2890857:
identity��!conv2d_17/StatefulPartitionedCall�"dense_2721/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_17_2890840conv2d_17_2890842*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_28905602#
!conv2d_17/StatefulPartitionedCall�
re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_re_lu_17_layer_call_and_return_conditional_losses_28905712
re_lu_17/PartitionedCall�
dropout_909/PartitionedCallPartitionedCall!re_lu_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_909_layer_call_and_return_conditional_losses_28905782
dropout_909/PartitionedCall�
flatten_56/PartitionedCallPartitionedCall$dropout_909/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_flatten_56_layer_call_and_return_conditional_losses_28905862
flatten_56/PartitionedCall�
"dense_2721/StatefulPartitionedCallStatefulPartitionedCall#flatten_56/PartitionedCall:output:0dense_2721_2890848dense_2721_2890850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_2721_layer_call_and_return_conditional_losses_28905992$
"dense_2721/StatefulPartitionedCall�
dropout_910/PartitionedCallPartitionedCall+dense_2721/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_910_layer_call_and_return_conditional_losses_28906102
dropout_910/PartitionedCall�
dropout_911/PartitionedCallPartitionedCall$dropout_910/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_911_layer_call_and_return_conditional_losses_28906172
dropout_911/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_911/PartitionedCall:output:0output_2890855output_2890857*
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
GPU2 *0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28906292 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall#^dense_2721/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2H
"dense_2721/StatefulPartitionedCall"dense_2721/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
*__inference_model_56_layer_call_fn_2890651
input_onehot"
unknown:�
	unknown_0:	�
	unknown_1:	�P
	unknown_2:P
	unknown_3:P
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
GPU2 *0J 8� *N
fIRG
E__inference_model_56_layer_call_and_return_conditional_losses_28906362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�"
�
E__inference_model_56_layer_call_and_return_conditional_losses_2890636

inputs,
conv2d_17_2890561:� 
conv2d_17_2890563:	�%
dense_2721_2890600:	�P 
dense_2721_2890602:P 
output_2890630:P
output_2890632:
identity��!conv2d_17/StatefulPartitionedCall�"dense_2721/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_17_2890561conv2d_17_2890563*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_28905602#
!conv2d_17/StatefulPartitionedCall�
re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_re_lu_17_layer_call_and_return_conditional_losses_28905712
re_lu_17/PartitionedCall�
dropout_909/PartitionedCallPartitionedCall!re_lu_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_909_layer_call_and_return_conditional_losses_28905782
dropout_909/PartitionedCall�
flatten_56/PartitionedCallPartitionedCall$dropout_909/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_flatten_56_layer_call_and_return_conditional_losses_28905862
flatten_56/PartitionedCall�
"dense_2721/StatefulPartitionedCallStatefulPartitionedCall#flatten_56/PartitionedCall:output:0dense_2721_2890600dense_2721_2890602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_2721_layer_call_and_return_conditional_losses_28905992$
"dense_2721/StatefulPartitionedCall�
dropout_910/PartitionedCallPartitionedCall+dense_2721/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_910_layer_call_and_return_conditional_losses_28906102
dropout_910/PartitionedCall�
dropout_911/PartitionedCallPartitionedCall$dropout_910/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_911_layer_call_and_return_conditional_losses_28906172
dropout_911/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_911/PartitionedCall:output:0output_2890630output_2890632*
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
GPU2 *0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28906292 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall#^dense_2721/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2H
"dense_2721/StatefulPartitionedCall"dense_2721/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_flatten_56_layer_call_and_return_conditional_losses_2890586

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_flatten_56_layer_call_and_return_conditional_losses_2891085

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
G__inference_dense_2721_layer_call_and_return_conditional_losses_2891101

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_dropout_911_layer_call_fn_2891159

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_911_layer_call_and_return_conditional_losses_28906172
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
I
-__inference_dropout_909_layer_call_fn_2891074

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_909_layer_call_and_return_conditional_losses_28905782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
H__inference_dropout_909_layer_call_and_return_conditional_losses_2891069

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
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
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
E__inference_model_56_layer_call_and_return_conditional_losses_2890939

inputsC
(conv2d_17_conv2d_readvariableop_resource:�8
)conv2d_17_biasadd_readvariableop_resource:	�<
)dense_2721_matmul_readvariableop_resource:	�P8
*dense_2721_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identity�� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp�!dense_2721/BiasAdd/ReadVariableOp� dense_2721/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02!
conv2d_17/Conv2D/ReadVariableOp�
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_17/Conv2D�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_17/BiasAdd}
re_lu_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
re_lu_17/Relu�
dropout_909/IdentityIdentityre_lu_17/Relu:activations:0*
T0*0
_output_shapes
:����������2
dropout_909/Identityu
flatten_56/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_56/Const�
flatten_56/ReshapeReshapedropout_909/Identity:output:0flatten_56/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_56/Reshape�
 dense_2721/MatMul/ReadVariableOpReadVariableOp)dense_2721_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02"
 dense_2721/MatMul/ReadVariableOp�
dense_2721/MatMulMatMulflatten_56/Reshape:output:0(dense_2721/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_2721/MatMul�
!dense_2721/BiasAdd/ReadVariableOpReadVariableOp*dense_2721_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02#
!dense_2721/BiasAdd/ReadVariableOp�
dense_2721/BiasAddBiasAdddense_2721/MatMul:product:0)dense_2721/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_2721/BiasAddy
dense_2721/ReluReludense_2721/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
dense_2721/Relu�
dropout_910/IdentityIdentitydense_2721/Relu:activations:0*
T0*'
_output_shapes
:���������P2
dropout_910/Identity�
dropout_911/IdentityIdentitydropout_910/Identity:output:0*
T0*'
_output_shapes
:���������P2
dropout_911/Identity�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_911/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp"^dense_2721/BiasAdd/ReadVariableOp!^dense_2721/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2F
!dense_2721/BiasAdd/ReadVariableOp!dense_2721/BiasAdd/ReadVariableOp2D
 dense_2721/MatMul/ReadVariableOp dense_2721/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_911_layer_call_and_return_conditional_losses_2890617

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
g
H__inference_dropout_911_layer_call_and_return_conditional_losses_2890681

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
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
T0*'
_output_shapes
:���������P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
f
H__inference_dropout_909_layer_call_and_return_conditional_losses_2890578

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�'
�
E__inference_model_56_layer_call_and_return_conditional_losses_2890805

inputs,
conv2d_17_2890784:� 
conv2d_17_2890786:	�%
dense_2721_2890792:	�P 
dense_2721_2890794:P 
output_2890799:P
output_2890801:
identity��!conv2d_17/StatefulPartitionedCall�"dense_2721/StatefulPartitionedCall�#dropout_909/StatefulPartitionedCall�#dropout_910/StatefulPartitionedCall�#dropout_911/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_17_2890784conv2d_17_2890786*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_28905602#
!conv2d_17/StatefulPartitionedCall�
re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_re_lu_17_layer_call_and_return_conditional_losses_28905712
re_lu_17/PartitionedCall�
#dropout_909/StatefulPartitionedCallStatefulPartitionedCall!re_lu_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_909_layer_call_and_return_conditional_losses_28907432%
#dropout_909/StatefulPartitionedCall�
flatten_56/PartitionedCallPartitionedCall,dropout_909/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_flatten_56_layer_call_and_return_conditional_losses_28905862
flatten_56/PartitionedCall�
"dense_2721/StatefulPartitionedCallStatefulPartitionedCall#flatten_56/PartitionedCall:output:0dense_2721_2890792dense_2721_2890794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_2721_layer_call_and_return_conditional_losses_28905992$
"dense_2721/StatefulPartitionedCall�
#dropout_910/StatefulPartitionedCallStatefulPartitionedCall+dense_2721/StatefulPartitionedCall:output:0$^dropout_909/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_910_layer_call_and_return_conditional_losses_28907042%
#dropout_910/StatefulPartitionedCall�
#dropout_911/StatefulPartitionedCallStatefulPartitionedCall,dropout_910/StatefulPartitionedCall:output:0$^dropout_910/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_911_layer_call_and_return_conditional_losses_28906812%
#dropout_911/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_911/StatefulPartitionedCall:output:0output_2890799output_2890801*
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
GPU2 *0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28906292 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall#^dense_2721/StatefulPartitionedCall$^dropout_909/StatefulPartitionedCall$^dropout_910/StatefulPartitionedCall$^dropout_911/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2H
"dense_2721/StatefulPartitionedCall"dense_2721/StatefulPartitionedCall2J
#dropout_909/StatefulPartitionedCall#dropout_909/StatefulPartitionedCall2J
#dropout_910/StatefulPartitionedCall#dropout_910/StatefulPartitionedCall2J
#dropout_911/StatefulPartitionedCall#dropout_911/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_model_56_layer_call_fn_2891006

inputs"
unknown:�
	unknown_0:	�
	unknown_1:	�P
	unknown_2:P
	unknown_3:P
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
GPU2 *0J 8� *N
fIRG
E__inference_model_56_layer_call_and_return_conditional_losses_28906362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�A
�
E__inference_model_56_layer_call_and_return_conditional_losses_2890989

inputsC
(conv2d_17_conv2d_readvariableop_resource:�8
)conv2d_17_biasadd_readvariableop_resource:	�<
)dense_2721_matmul_readvariableop_resource:	�P8
*dense_2721_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identity�� conv2d_17/BiasAdd/ReadVariableOp�conv2d_17/Conv2D/ReadVariableOp�!dense_2721/BiasAdd/ReadVariableOp� dense_2721/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02!
conv2d_17/Conv2D/ReadVariableOp�
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_17/Conv2D�
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp�
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_17/BiasAdd}
re_lu_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
re_lu_17/Relu{
dropout_909/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_909/dropout/Const�
dropout_909/dropout/MulMulre_lu_17/Relu:activations:0"dropout_909/dropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout_909/dropout/Mul�
dropout_909/dropout/ShapeShapere_lu_17/Relu:activations:0*
T0*
_output_shapes
:2
dropout_909/dropout/Shape�
0dropout_909/dropout/random_uniform/RandomUniformRandomUniform"dropout_909/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype022
0dropout_909/dropout/random_uniform/RandomUniform�
"dropout_909/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2$
"dropout_909/dropout/GreaterEqual/y�
 dropout_909/dropout/GreaterEqualGreaterEqual9dropout_909/dropout/random_uniform/RandomUniform:output:0+dropout_909/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2"
 dropout_909/dropout/GreaterEqual�
dropout_909/dropout/CastCast$dropout_909/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout_909/dropout/Cast�
dropout_909/dropout/Mul_1Muldropout_909/dropout/Mul:z:0dropout_909/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout_909/dropout/Mul_1u
flatten_56/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_56/Const�
flatten_56/ReshapeReshapedropout_909/dropout/Mul_1:z:0flatten_56/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_56/Reshape�
 dense_2721/MatMul/ReadVariableOpReadVariableOp)dense_2721_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02"
 dense_2721/MatMul/ReadVariableOp�
dense_2721/MatMulMatMulflatten_56/Reshape:output:0(dense_2721/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_2721/MatMul�
!dense_2721/BiasAdd/ReadVariableOpReadVariableOp*dense_2721_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02#
!dense_2721/BiasAdd/ReadVariableOp�
dense_2721/BiasAddBiasAdddense_2721/MatMul:product:0)dense_2721/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_2721/BiasAddy
dense_2721/ReluReludense_2721/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
dense_2721/Relu{
dropout_910/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_910/dropout/Const�
dropout_910/dropout/MulMuldense_2721/Relu:activations:0"dropout_910/dropout/Const:output:0*
T0*'
_output_shapes
:���������P2
dropout_910/dropout/Mul�
dropout_910/dropout/ShapeShapedense_2721/Relu:activations:0*
T0*
_output_shapes
:2
dropout_910/dropout/Shape�
0dropout_910/dropout/random_uniform/RandomUniformRandomUniform"dropout_910/dropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
dtype022
0dropout_910/dropout/random_uniform/RandomUniform�
"dropout_910/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2$
"dropout_910/dropout/GreaterEqual/y�
 dropout_910/dropout/GreaterEqualGreaterEqual9dropout_910/dropout/random_uniform/RandomUniform:output:0+dropout_910/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������P2"
 dropout_910/dropout/GreaterEqual�
dropout_910/dropout/CastCast$dropout_910/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
dropout_910/dropout/Cast�
dropout_910/dropout/Mul_1Muldropout_910/dropout/Mul:z:0dropout_910/dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
dropout_910/dropout/Mul_1{
dropout_911/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_911/dropout/Const�
dropout_911/dropout/MulMuldropout_910/dropout/Mul_1:z:0"dropout_911/dropout/Const:output:0*
T0*'
_output_shapes
:���������P2
dropout_911/dropout/Mul�
dropout_911/dropout/ShapeShapedropout_910/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dropout_911/dropout/Shape�
0dropout_911/dropout/random_uniform/RandomUniformRandomUniform"dropout_911/dropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
dtype022
0dropout_911/dropout/random_uniform/RandomUniform�
"dropout_911/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2$
"dropout_911/dropout/GreaterEqual/y�
 dropout_911/dropout/GreaterEqualGreaterEqual9dropout_911/dropout/random_uniform/RandomUniform:output:0+dropout_911/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������P2"
 dropout_911/dropout/GreaterEqual�
dropout_911/dropout/CastCast$dropout_911/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
dropout_911/dropout/Cast�
dropout_911/dropout/Mul_1Muldropout_911/dropout/Mul:z:0dropout_911/dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
dropout_911/dropout/Mul_1�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_911/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp"^dense_2721/BiasAdd/ReadVariableOp!^dense_2721/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2F
!dense_2721/BiasAdd/ReadVariableOp!dense_2721/BiasAdd/ReadVariableOp2D
 dense_2721/MatMul/ReadVariableOp dense_2721/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_909_layer_call_and_return_conditional_losses_2891057

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
H__inference_dropout_910_layer_call_and_return_conditional_losses_2890610

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
*__inference_model_56_layer_call_fn_2891023

inputs"
unknown:�
	unknown_0:	�
	unknown_1:	�P
	unknown_2:P
	unknown_3:P
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
GPU2 *0J 8� *N
fIRG
E__inference_model_56_layer_call_and_return_conditional_losses_28908052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�

 __inference__traced_save_2891281
file_prefix/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop0
,savev2_dense_2721_kernel_read_readvariableop.
*savev2_dense_2721_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableop7
3savev2_adam_dense_2721_kernel_m_read_readvariableop5
1savev2_adam_dense_2721_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableop7
3savev2_adam_dense_2721_kernel_v_read_readvariableop5
1savev2_adam_dense_2721_bias_v_read_readvariableop3
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

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop,savev2_dense_2721_kernel_read_readvariableop*savev2_dense_2721_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop3savev2_adam_dense_2721_kernel_m_read_readvariableop1savev2_adam_dense_2721_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop3savev2_adam_dense_2721_kernel_v_read_readvariableop1savev2_adam_dense_2721_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�: :�:�:	�P:P:P:: : : : : : : :�:�:	�P:P:P::�:�:	�P:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�P: 

_output_shapes
:P:$ 

_output_shapes

:P: 
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
: :-)
'
_output_shapes
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�P: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::-)
'
_output_shapes
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�P: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::

_output_shapes
: 
�*
�
"__inference__wrapped_model_2890543
input_onehotL
1model_56_conv2d_17_conv2d_readvariableop_resource:�A
2model_56_conv2d_17_biasadd_readvariableop_resource:	�E
2model_56_dense_2721_matmul_readvariableop_resource:	�PA
3model_56_dense_2721_biasadd_readvariableop_resource:P@
.model_56_output_matmul_readvariableop_resource:P=
/model_56_output_biasadd_readvariableop_resource:
identity��)model_56/conv2d_17/BiasAdd/ReadVariableOp�(model_56/conv2d_17/Conv2D/ReadVariableOp�*model_56/dense_2721/BiasAdd/ReadVariableOp�)model_56/dense_2721/MatMul/ReadVariableOp�&model_56/output/BiasAdd/ReadVariableOp�%model_56/output/MatMul/ReadVariableOp�
(model_56/conv2d_17/Conv2D/ReadVariableOpReadVariableOp1model_56_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02*
(model_56/conv2d_17/Conv2D/ReadVariableOp�
model_56/conv2d_17/Conv2DConv2Dinput_onehot0model_56/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
model_56/conv2d_17/Conv2D�
)model_56/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp2model_56_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)model_56/conv2d_17/BiasAdd/ReadVariableOp�
model_56/conv2d_17/BiasAddBiasAdd"model_56/conv2d_17/Conv2D:output:01model_56/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_56/conv2d_17/BiasAdd�
model_56/re_lu_17/ReluRelu#model_56/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_56/re_lu_17/Relu�
model_56/dropout_909/IdentityIdentity$model_56/re_lu_17/Relu:activations:0*
T0*0
_output_shapes
:����������2
model_56/dropout_909/Identity�
model_56/flatten_56/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
model_56/flatten_56/Const�
model_56/flatten_56/ReshapeReshape&model_56/dropout_909/Identity:output:0"model_56/flatten_56/Const:output:0*
T0*(
_output_shapes
:����������2
model_56/flatten_56/Reshape�
)model_56/dense_2721/MatMul/ReadVariableOpReadVariableOp2model_56_dense_2721_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02+
)model_56/dense_2721/MatMul/ReadVariableOp�
model_56/dense_2721/MatMulMatMul$model_56/flatten_56/Reshape:output:01model_56/dense_2721/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_56/dense_2721/MatMul�
*model_56/dense_2721/BiasAdd/ReadVariableOpReadVariableOp3model_56_dense_2721_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model_56/dense_2721/BiasAdd/ReadVariableOp�
model_56/dense_2721/BiasAddBiasAdd$model_56/dense_2721/MatMul:product:02model_56/dense_2721/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_56/dense_2721/BiasAdd�
model_56/dense_2721/ReluRelu$model_56/dense_2721/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
model_56/dense_2721/Relu�
model_56/dropout_910/IdentityIdentity&model_56/dense_2721/Relu:activations:0*
T0*'
_output_shapes
:���������P2
model_56/dropout_910/Identity�
model_56/dropout_911/IdentityIdentity&model_56/dropout_910/Identity:output:0*
T0*'
_output_shapes
:���������P2
model_56/dropout_911/Identity�
%model_56/output/MatMul/ReadVariableOpReadVariableOp.model_56_output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02'
%model_56/output/MatMul/ReadVariableOp�
model_56/output/MatMulMatMul&model_56/dropout_911/Identity:output:0-model_56/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_56/output/MatMul�
&model_56/output/BiasAdd/ReadVariableOpReadVariableOp/model_56_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_56/output/BiasAdd/ReadVariableOp�
model_56/output/BiasAddBiasAdd model_56/output/MatMul:product:0.model_56/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_56/output/BiasAdd�
IdentityIdentity model_56/output/BiasAdd:output:0*^model_56/conv2d_17/BiasAdd/ReadVariableOp)^model_56/conv2d_17/Conv2D/ReadVariableOp+^model_56/dense_2721/BiasAdd/ReadVariableOp*^model_56/dense_2721/MatMul/ReadVariableOp'^model_56/output/BiasAdd/ReadVariableOp&^model_56/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2V
)model_56/conv2d_17/BiasAdd/ReadVariableOp)model_56/conv2d_17/BiasAdd/ReadVariableOp2T
(model_56/conv2d_17/Conv2D/ReadVariableOp(model_56/conv2d_17/Conv2D/ReadVariableOp2X
*model_56/dense_2721/BiasAdd/ReadVariableOp*model_56/dense_2721/BiasAdd/ReadVariableOp2V
)model_56/dense_2721/MatMul/ReadVariableOp)model_56/dense_2721/MatMul/ReadVariableOp2P
&model_56/output/BiasAdd/ReadVariableOp&model_56/output/BiasAdd/ReadVariableOp2N
%model_56/output/MatMul/ReadVariableOp%model_56/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
,__inference_dense_2721_layer_call_fn_2891110

inputs
unknown:	�P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_2721_layer_call_and_return_conditional_losses_28905992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_909_layer_call_fn_2891079

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_909_layer_call_and_return_conditional_losses_28907432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
H__inference_dropout_911_layer_call_and_return_conditional_losses_2891142

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
f
H__inference_dropout_910_layer_call_and_return_conditional_losses_2891115

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
g
H__inference_dropout_911_layer_call_and_return_conditional_losses_2891154

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
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
T0*'
_output_shapes
:���������P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_2890910
input_onehot"
unknown:�
	unknown_0:	�
	unknown_1:	�P
	unknown_2:P
	unknown_3:P
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
GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_28905432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
a
E__inference_re_lu_17_layer_call_and_return_conditional_losses_2890571

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:����������2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�'
�
E__inference_model_56_layer_call_and_return_conditional_losses_2890885
input_onehot,
conv2d_17_2890864:� 
conv2d_17_2890866:	�%
dense_2721_2890872:	�P 
dense_2721_2890874:P 
output_2890879:P
output_2890881:
identity��!conv2d_17/StatefulPartitionedCall�"dense_2721/StatefulPartitionedCall�#dropout_909/StatefulPartitionedCall�#dropout_910/StatefulPartitionedCall�#dropout_911/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_17_2890864conv2d_17_2890866*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_28905602#
!conv2d_17/StatefulPartitionedCall�
re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_re_lu_17_layer_call_and_return_conditional_losses_28905712
re_lu_17/PartitionedCall�
#dropout_909/StatefulPartitionedCallStatefulPartitionedCall!re_lu_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_909_layer_call_and_return_conditional_losses_28907432%
#dropout_909/StatefulPartitionedCall�
flatten_56/PartitionedCallPartitionedCall,dropout_909/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_flatten_56_layer_call_and_return_conditional_losses_28905862
flatten_56/PartitionedCall�
"dense_2721/StatefulPartitionedCallStatefulPartitionedCall#flatten_56/PartitionedCall:output:0dense_2721_2890872dense_2721_2890874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_2721_layer_call_and_return_conditional_losses_28905992$
"dense_2721/StatefulPartitionedCall�
#dropout_910/StatefulPartitionedCallStatefulPartitionedCall+dense_2721/StatefulPartitionedCall:output:0$^dropout_909/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_910_layer_call_and_return_conditional_losses_28907042%
#dropout_910/StatefulPartitionedCall�
#dropout_911/StatefulPartitionedCallStatefulPartitionedCall,dropout_910/StatefulPartitionedCall:output:0$^dropout_910/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_911_layer_call_and_return_conditional_losses_28906812%
#dropout_911/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_911/StatefulPartitionedCall:output:0output_2890879output_2890881*
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
GPU2 *0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28906292 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall#^dense_2721/StatefulPartitionedCall$^dropout_909/StatefulPartitionedCall$^dropout_910/StatefulPartitionedCall$^dropout_911/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2H
"dense_2721/StatefulPartitionedCall"dense_2721/StatefulPartitionedCall2J
#dropout_909/StatefulPartitionedCall#dropout_909/StatefulPartitionedCall2J
#dropout_910/StatefulPartitionedCall#dropout_910/StatefulPartitionedCall2J
#dropout_911/StatefulPartitionedCall#dropout_911/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
g
H__inference_dropout_910_layer_call_and_return_conditional_losses_2891127

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
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
T0*'
_output_shapes
:���������P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
g
H__inference_dropout_910_layer_call_and_return_conditional_losses_2890704

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
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
T0*'
_output_shapes
:���������P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�

�
F__inference_conv2d_17_layer_call_and_return_conditional_losses_2890560

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
H__inference_dropout_909_layer_call_and_return_conditional_losses_2890743

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
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
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_re_lu_17_layer_call_and_return_conditional_losses_2891047

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:����������2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_output_layer_call_fn_2891183

inputs
unknown:P
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
GPU2 *0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28906292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
I
-__inference_dropout_910_layer_call_fn_2891132

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_910_layer_call_and_return_conditional_losses_28906102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
H
,__inference_flatten_56_layer_call_fn_2891090

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
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_flatten_56_layer_call_and_return_conditional_losses_28905862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_910_layer_call_fn_2891137

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_dropout_910_layer_call_and_return_conditional_losses_28907042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2891174

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�

�
F__inference_conv2d_17_layer_call_and_return_conditional_losses_2891033

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2890629

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
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
serving_default_input_onehot:0���������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�@
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*y&call_and_return_all_conditional_losses
z__call__
{_default_save_signature"�=
_tf_keras_network�<{"name": "model_56", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_56", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_17", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_17", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_909", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_909", "inbound_nodes": [[["re_lu_17", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_56", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_56", "inbound_nodes": [[["dropout_909", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2721", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2721", "inbound_nodes": [[["flatten_56", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_910", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_910", "inbound_nodes": [[["dense_2721", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_911", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_911", "inbound_nodes": [[["dropout_910", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_911", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_56", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "ReLU", "config": {"name": "re_lu_17", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_17", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_909", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_909", "inbound_nodes": [[["re_lu_17", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_56", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_56", "inbound_nodes": [[["dropout_909", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_2721", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2721", "inbound_nodes": [[["flatten_56", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_910", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_910", "inbound_nodes": [[["dense_2721", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_911", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_911", "inbound_nodes": [[["dropout_910", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_911", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*|&call_and_return_all_conditional_losses
}__call__"�	
_tf_keras_layer�	{"name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}}
�
trainable_variables
regularization_losses
	variables
	keras_api
*~&call_and_return_all_conditional_losses
__call__"�
_tf_keras_layer�{"name": "re_lu_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_17", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv2d_17", 0, 0, {}]]], "shared_object_id": 4}
�
trainable_variables
regularization_losses
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_909", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_909", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["re_lu_17", 0, 0, {}]]], "shared_object_id": 5}
�
trainable_variables
regularization_losses
 	variables
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_56", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["dropout_909", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
�	

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_2721", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2721", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_56", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3780}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3780]}}
�
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_910", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_910", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_2721", 0, 0, {}]]], "shared_object_id": 10}
�
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout_911", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_911", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dropout_910", 0, 0, {}]]], "shared_object_id": 11}
�	

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_911", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�
6iter

7beta_1

8beta_2
	9decay
:learning_ratemmmn"mo#mp0mq1mrvsvt"vu#vv0vw1vx"
	optimizer
J
0
1
"2
#3
04
15"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
"2
#3
04
15"
trackable_list_wrapper
�
trainable_variables
;layer_metrics
<non_trainable_variables
regularization_losses
=layer_regularization_losses
>metrics

?layers
	variables
z__call__
{_default_save_signature
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
+:)�2conv2d_17/kernel
:�2conv2d_17/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
@layer_metrics
Anon_trainable_variables
regularization_losses
Blayer_regularization_losses
Cmetrics

Dlayers
	variables
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
Elayer_metrics
Fnon_trainable_variables
regularization_losses
Glayer_regularization_losses
Hmetrics

Ilayers
	variables
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
Jlayer_metrics
Knon_trainable_variables
regularization_losses
Llayer_regularization_losses
Mmetrics

Nlayers
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
 	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
$:"	�P2dense_2721/kernel
:P2dense_2721/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
�
$trainable_variables
Tlayer_metrics
Unon_trainable_variables
%regularization_losses
Vlayer_regularization_losses
Wmetrics

Xlayers
&	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
(trainable_variables
Ylayer_metrics
Znon_trainable_variables
)regularization_losses
[layer_regularization_losses
\metrics

]layers
*	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
,trainable_variables
^layer_metrics
_non_trainable_variables
-regularization_losses
`layer_regularization_losses
ametrics

blayers
.	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:P2output/kernel
:2output/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
2trainable_variables
clayer_metrics
dnon_trainable_variables
3regularization_losses
elayer_regularization_losses
fmetrics

glayers
4	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
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
	itotal
	jcount
k	variables
l	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 21}
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
0:.�2Adam/conv2d_17/kernel/m
": �2Adam/conv2d_17/bias/m
):'	�P2Adam/dense_2721/kernel/m
": P2Adam/dense_2721/bias/m
$:"P2Adam/output/kernel/m
:2Adam/output/bias/m
0:.�2Adam/conv2d_17/kernel/v
": �2Adam/conv2d_17/bias/v
):'	�P2Adam/dense_2721/kernel/v
": P2Adam/dense_2721/bias/v
$:"P2Adam/output/kernel/v
:2Adam/output/bias/v
�2�
E__inference_model_56_layer_call_and_return_conditional_losses_2890939
E__inference_model_56_layer_call_and_return_conditional_losses_2890989
E__inference_model_56_layer_call_and_return_conditional_losses_2890861
E__inference_model_56_layer_call_and_return_conditional_losses_2890885�
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
�2�
*__inference_model_56_layer_call_fn_2890651
*__inference_model_56_layer_call_fn_2891006
*__inference_model_56_layer_call_fn_2891023
*__inference_model_56_layer_call_fn_2890837�
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
"__inference__wrapped_model_2890543�
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
input_onehot���������
�2�
F__inference_conv2d_17_layer_call_and_return_conditional_losses_2891033�
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
+__inference_conv2d_17_layer_call_fn_2891042�
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
E__inference_re_lu_17_layer_call_and_return_conditional_losses_2891047�
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
*__inference_re_lu_17_layer_call_fn_2891052�
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
H__inference_dropout_909_layer_call_and_return_conditional_losses_2891057
H__inference_dropout_909_layer_call_and_return_conditional_losses_2891069�
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
-__inference_dropout_909_layer_call_fn_2891074
-__inference_dropout_909_layer_call_fn_2891079�
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
G__inference_flatten_56_layer_call_and_return_conditional_losses_2891085�
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
,__inference_flatten_56_layer_call_fn_2891090�
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
G__inference_dense_2721_layer_call_and_return_conditional_losses_2891101�
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
,__inference_dense_2721_layer_call_fn_2891110�
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
H__inference_dropout_910_layer_call_and_return_conditional_losses_2891115
H__inference_dropout_910_layer_call_and_return_conditional_losses_2891127�
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
-__inference_dropout_910_layer_call_fn_2891132
-__inference_dropout_910_layer_call_fn_2891137�
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
H__inference_dropout_911_layer_call_and_return_conditional_losses_2891142
H__inference_dropout_911_layer_call_and_return_conditional_losses_2891154�
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
-__inference_dropout_911_layer_call_fn_2891159
-__inference_dropout_911_layer_call_fn_2891164�
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
C__inference_output_layer_call_and_return_conditional_losses_2891174�
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
(__inference_output_layer_call_fn_2891183�
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
%__inference_signature_wrapper_2890910input_onehot"�
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
"__inference__wrapped_model_2890543x"#01=�:
3�0
.�+
input_onehot���������
� "/�,
*
output �
output����������
F__inference_conv2d_17_layer_call_and_return_conditional_losses_2891033m7�4
-�*
(�%
inputs���������
� ".�+
$�!
0����������
� �
+__inference_conv2d_17_layer_call_fn_2891042`7�4
-�*
(�%
inputs���������
� "!������������
G__inference_dense_2721_layer_call_and_return_conditional_losses_2891101]"#0�-
&�#
!�
inputs����������
� "%�"
�
0���������P
� �
,__inference_dense_2721_layer_call_fn_2891110P"#0�-
&�#
!�
inputs����������
� "����������P�
H__inference_dropout_909_layer_call_and_return_conditional_losses_2891057n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
H__inference_dropout_909_layer_call_and_return_conditional_losses_2891069n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
-__inference_dropout_909_layer_call_fn_2891074a<�9
2�/
)�&
inputs����������
p 
� "!������������
-__inference_dropout_909_layer_call_fn_2891079a<�9
2�/
)�&
inputs����������
p
� "!������������
H__inference_dropout_910_layer_call_and_return_conditional_losses_2891115\3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
H__inference_dropout_910_layer_call_and_return_conditional_losses_2891127\3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
-__inference_dropout_910_layer_call_fn_2891132O3�0
)�&
 �
inputs���������P
p 
� "����������P�
-__inference_dropout_910_layer_call_fn_2891137O3�0
)�&
 �
inputs���������P
p
� "����������P�
H__inference_dropout_911_layer_call_and_return_conditional_losses_2891142\3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
H__inference_dropout_911_layer_call_and_return_conditional_losses_2891154\3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
-__inference_dropout_911_layer_call_fn_2891159O3�0
)�&
 �
inputs���������P
p 
� "����������P�
-__inference_dropout_911_layer_call_fn_2891164O3�0
)�&
 �
inputs���������P
p
� "����������P�
G__inference_flatten_56_layer_call_and_return_conditional_losses_2891085b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
,__inference_flatten_56_layer_call_fn_2891090U8�5
.�+
)�&
inputs����������
� "������������
E__inference_model_56_layer_call_and_return_conditional_losses_2890861v"#01E�B
;�8
.�+
input_onehot���������
p 

 
� "%�"
�
0���������
� �
E__inference_model_56_layer_call_and_return_conditional_losses_2890885v"#01E�B
;�8
.�+
input_onehot���������
p

 
� "%�"
�
0���������
� �
E__inference_model_56_layer_call_and_return_conditional_losses_2890939p"#01?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������
� �
E__inference_model_56_layer_call_and_return_conditional_losses_2890989p"#01?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������
� �
*__inference_model_56_layer_call_fn_2890651i"#01E�B
;�8
.�+
input_onehot���������
p 

 
� "�����������
*__inference_model_56_layer_call_fn_2890837i"#01E�B
;�8
.�+
input_onehot���������
p

 
� "�����������
*__inference_model_56_layer_call_fn_2891006c"#01?�<
5�2
(�%
inputs���������
p 

 
� "�����������
*__inference_model_56_layer_call_fn_2891023c"#01?�<
5�2
(�%
inputs���������
p

 
� "�����������
C__inference_output_layer_call_and_return_conditional_losses_2891174\01/�,
%�"
 �
inputs���������P
� "%�"
�
0���������
� {
(__inference_output_layer_call_fn_2891183O01/�,
%�"
 �
inputs���������P
� "�����������
E__inference_re_lu_17_layer_call_and_return_conditional_losses_2891047j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
*__inference_re_lu_17_layer_call_fn_2891052]8�5
.�+
)�&
inputs����������
� "!������������
%__inference_signature_wrapper_2890910�"#01M�J
� 
C�@
>
input_onehot.�+
input_onehot���������"/�,
*
output �
output���������