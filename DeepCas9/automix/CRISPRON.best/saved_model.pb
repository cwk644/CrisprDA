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
conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*!
shared_nameconv2d_63/kernel
}
$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*&
_output_shapes
:2*
dtype0
t
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameconv2d_63/bias
m
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes
:2*
dtype0
�
dense_13167/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*#
shared_namedense_13167/kernel
{
&dense_13167/kernel/Read/ReadVariableOpReadVariableOpdense_13167/kernel* 
_output_shapes
:
��*
dtype0
y
dense_13167/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_namedense_13167/bias
r
$dense_13167/bias/Read/ReadVariableOpReadVariableOpdense_13167/bias*
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
Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/conv2d_63/kernel/m
�
+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*&
_output_shapes
:2*
dtype0
�
Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/conv2d_63/bias/m
{
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_13167/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_nameAdam/dense_13167/kernel/m
�
-Adam/dense_13167/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13167/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_13167/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/dense_13167/bias/m
�
+Adam/dense_13167/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13167/bias/m*
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
Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/conv2d_63/kernel/v
�
+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*&
_output_shapes
:2*
dtype0
�
Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/conv2d_63/bias/v
{
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_13167/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_nameAdam/dense_13167/kernel/v
�
-Adam/dense_13167/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13167/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_13167/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameAdam/dense_13167/bias/v
�
+Adam/dense_13167/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13167/bias/v*
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
dtype0*�*
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

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
R
$trainable_variables
%	variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratem[m\m]m^(m_)m`vavbvcvd(ve)vf
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
 
�

3layers

	variables
4metrics
trainable_variables
5non_trainable_variables
6layer_regularization_losses
7layer_metrics
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_63/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_63/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�

8layers
trainable_variables
9metrics
	variables
:non_trainable_variables
;layer_regularization_losses
<layer_metrics
regularization_losses
 
 
 
 
�

=layers
trainable_variables
>metrics
	variables
?non_trainable_variables
@layer_regularization_losses
Alayer_metrics
regularization_losses
 
 
 
�

Blayers
trainable_variables
Cmetrics
	variables
Dnon_trainable_variables
Elayer_regularization_losses
Flayer_metrics
regularization_losses
^\
VARIABLE_VALUEdense_13167/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_13167/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�

Glayers
 trainable_variables
Hmetrics
!	variables
Inon_trainable_variables
Jlayer_regularization_losses
Klayer_metrics
"regularization_losses
 
 
 
�

Llayers
$trainable_variables
Mmetrics
%	variables
Nnon_trainable_variables
Olayer_regularization_losses
Player_metrics
&regularization_losses
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
�

Qlayers
*trainable_variables
Rmetrics
+	variables
Snon_trainable_variables
Tlayer_regularization_losses
Ulayer_metrics
,regularization_losses
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
8
0
1
2
3
4
5
6
7
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
VARIABLE_VALUEAdam/conv2d_63/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/dense_13167/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_13167/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_63/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/dense_13167/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_13167/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_63/kernelconv2d_63/biasdense_13167/kerneldense_13167/biasoutput/kerneloutput/bias*
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
GPU2 *0J 8� */
f*R(
&__inference_signature_wrapper_10647980
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp&dense_13167/kernel/Read/ReadVariableOp$dense_13167/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp-Adam/dense_13167/kernel/m/Read/ReadVariableOp+Adam/dense_13167/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp-Adam/dense_13167/kernel/v/Read/ReadVariableOp+Adam/dense_13167/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
GPU2 *0J 8� **
f%R#
!__inference__traced_save_10648282
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_63/kernelconv2d_63/biasdense_13167/kerneldense_13167/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/mAdam/dense_13167/kernel/mAdam/dense_13167/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/vAdam/dense_13167/kernel/vAdam/dense_13167/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
GPU2 *0J 8� *-
f(R&
$__inference__traced_restore_10648367�
�
�
G__inference_conv2d_63_layer_call_and_return_conditional_losses_10647700

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
�
�
G__inference_conv2d_63_layer_call_and_return_conditional_losses_10648098

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
�
�
,__inference_model_252_layer_call_fn_10647773
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
GPU2 *0J 8� *P
fKRI
G__inference_model_252_layer_call_and_return_conditional_losses_106477582
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
�
�
&__inference_signature_wrapper_10647980
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
GPU2 *0J 8� *,
f'R%
#__inference__wrapped_model_106476672
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
�m
�
$__inference__traced_restore_10648367
file_prefix;
!assignvariableop_conv2d_63_kernel:2/
!assignvariableop_1_conv2d_63_bias:29
%assignvariableop_2_dense_13167_kernel:
��2
#assignvariableop_3_dense_13167_bias:	�3
 assignvariableop_4_output_kernel:	�,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: E
+assignvariableop_13_adam_conv2d_63_kernel_m:27
)assignvariableop_14_adam_conv2d_63_bias_m:2A
-assignvariableop_15_adam_dense_13167_kernel_m:
��:
+assignvariableop_16_adam_dense_13167_bias_m:	�;
(assignvariableop_17_adam_output_kernel_m:	�4
&assignvariableop_18_adam_output_bias_m:E
+assignvariableop_19_adam_conv2d_63_kernel_v:27
)assignvariableop_20_adam_conv2d_63_bias_v:2A
-assignvariableop_21_adam_dense_13167_kernel_v:
��:
+assignvariableop_22_adam_dense_13167_bias_v:	�;
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_63_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_63_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_13167_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_13167_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_63_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_63_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_dense_13167_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_13167_bias_mIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_63_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_63_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_dense_13167_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_dense_13167_bias_vIdentity_22:output:0"/device:CPU:0*
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
e
I__inference_flatten_252_layer_call_and_return_conditional_losses_10647715

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
�	
�
D__inference_output_layer_call_and_return_conditional_losses_10647751

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
�

�
I__inference_dense_13167_layer_call_and_return_conditional_losses_10648129

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
�
,__inference_model_252_layer_call_fn_10648070

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
GPU2 *0J 8� *P
fKRI
G__inference_model_252_layer_call_and_return_conditional_losses_106477582
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
�
O
3__inference_max_pooling1d_63_layer_call_fn_10647682

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
GPU2 *0J 8� *W
fRRP
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_106476762
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
�
i
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10648155

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
�
h
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10648143

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
�2
�
#__inference__wrapped_model_10647667
input_onehotL
2model_252_conv2d_63_conv2d_readvariableop_resource:2A
3model_252_conv2d_63_biasadd_readvariableop_resource:2H
4model_252_dense_13167_matmul_readvariableop_resource:
��D
5model_252_dense_13167_biasadd_readvariableop_resource:	�B
/model_252_output_matmul_readvariableop_resource:	�>
0model_252_output_biasadd_readvariableop_resource:
identity��*model_252/conv2d_63/BiasAdd/ReadVariableOp�)model_252/conv2d_63/Conv2D/ReadVariableOp�,model_252/dense_13167/BiasAdd/ReadVariableOp�+model_252/dense_13167/MatMul/ReadVariableOp�'model_252/output/BiasAdd/ReadVariableOp�&model_252/output/MatMul/ReadVariableOp�
)model_252/conv2d_63/Conv2D/ReadVariableOpReadVariableOp2model_252_conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02+
)model_252/conv2d_63/Conv2D/ReadVariableOp�
model_252/conv2d_63/Conv2DConv2Dinput_onehot1model_252/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
model_252/conv2d_63/Conv2D�
*model_252/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp3model_252_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02,
*model_252/conv2d_63/BiasAdd/ReadVariableOp�
model_252/conv2d_63/BiasAddBiasAdd#model_252/conv2d_63/Conv2D:output:02model_252/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22
model_252/conv2d_63/BiasAdd�
model_252/conv2d_63/ReluRelu$model_252/conv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:���������22
model_252/conv2d_63/Relu�
&model_252/tf.reshape_189/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2(
&model_252/tf.reshape_189/Reshape/shape�
 model_252/tf.reshape_189/ReshapeReshape&model_252/conv2d_63/Relu:activations:0/model_252/tf.reshape_189/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22"
 model_252/tf.reshape_189/Reshape�
)model_252/max_pooling1d_63/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_252/max_pooling1d_63/ExpandDims/dim�
%model_252/max_pooling1d_63/ExpandDims
ExpandDims)model_252/tf.reshape_189/Reshape:output:02model_252/max_pooling1d_63/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������22'
%model_252/max_pooling1d_63/ExpandDims�
"model_252/max_pooling1d_63/MaxPoolMaxPool.model_252/max_pooling1d_63/ExpandDims:output:0*/
_output_shapes
:���������
2*
ksize
*
paddingVALID*
strides
2$
"model_252/max_pooling1d_63/MaxPool�
"model_252/max_pooling1d_63/SqueezeSqueeze+model_252/max_pooling1d_63/MaxPool:output:0*
T0*+
_output_shapes
:���������
2*
squeeze_dims
2$
"model_252/max_pooling1d_63/Squeeze�
model_252/flatten_252/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
model_252/flatten_252/Const�
model_252/flatten_252/ReshapeReshape+model_252/max_pooling1d_63/Squeeze:output:0$model_252/flatten_252/Const:output:0*
T0*(
_output_shapes
:����������2
model_252/flatten_252/Reshape�
+model_252/dense_13167/MatMul/ReadVariableOpReadVariableOp4model_252_dense_13167_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+model_252/dense_13167/MatMul/ReadVariableOp�
model_252/dense_13167/MatMulMatMul&model_252/flatten_252/Reshape:output:03model_252/dense_13167/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_252/dense_13167/MatMul�
,model_252/dense_13167/BiasAdd/ReadVariableOpReadVariableOp5model_252_dense_13167_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,model_252/dense_13167/BiasAdd/ReadVariableOp�
model_252/dense_13167/BiasAddBiasAdd&model_252/dense_13167/MatMul:product:04model_252/dense_13167/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_252/dense_13167/BiasAdd�
model_252/dense_13167/ReluRelu&model_252/dense_13167/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model_252/dense_13167/Relu�
model_252/dropout_4221/IdentityIdentity(model_252/dense_13167/Relu:activations:0*
T0*(
_output_shapes
:����������2!
model_252/dropout_4221/Identity�
&model_252/output/MatMul/ReadVariableOpReadVariableOp/model_252_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02(
&model_252/output/MatMul/ReadVariableOp�
model_252/output/MatMulMatMul(model_252/dropout_4221/Identity:output:0.model_252/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_252/output/MatMul�
'model_252/output/BiasAdd/ReadVariableOpReadVariableOp0model_252_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_252/output/BiasAdd/ReadVariableOp�
model_252/output/BiasAddBiasAdd!model_252/output/MatMul:product:0/model_252/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_252/output/BiasAdd�
IdentityIdentity!model_252/output/BiasAdd:output:0+^model_252/conv2d_63/BiasAdd/ReadVariableOp*^model_252/conv2d_63/Conv2D/ReadVariableOp-^model_252/dense_13167/BiasAdd/ReadVariableOp,^model_252/dense_13167/MatMul/ReadVariableOp(^model_252/output/BiasAdd/ReadVariableOp'^model_252/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2X
*model_252/conv2d_63/BiasAdd/ReadVariableOp*model_252/conv2d_63/BiasAdd/ReadVariableOp2V
)model_252/conv2d_63/Conv2D/ReadVariableOp)model_252/conv2d_63/Conv2D/ReadVariableOp2\
,model_252/dense_13167/BiasAdd/ReadVariableOp,model_252/dense_13167/BiasAdd/ReadVariableOp2Z
+model_252/dense_13167/MatMul/ReadVariableOp+model_252/dense_13167/MatMul/ReadVariableOp2R
'model_252/output/BiasAdd/ReadVariableOp'model_252/output/BiasAdd/ReadVariableOp2P
&model_252/output/MatMul/ReadVariableOp&model_252/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
i
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10647803

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
�
h
/__inference_dropout_4221_layer_call_fn_10648165

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
GPU2 *0J 8� *S
fNRL
J__inference_dropout_4221_layer_call_and_return_conditional_losses_106478032
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
�
�
,__inference_model_252_layer_call_fn_10647907
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
GPU2 *0J 8� *P
fKRI
G__inference_model_252_layer_call_and_return_conditional_losses_106478752
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
�
�
,__inference_model_252_layer_call_fn_10648087

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
GPU2 *0J 8� *P
fKRI
G__inference_model_252_layer_call_and_return_conditional_losses_106478752
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
�
�
.__inference_dense_13167_layer_call_fn_10648138

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
GPU2 *0J 8� *R
fMRK
I__inference_dense_13167_layer_call_and_return_conditional_losses_106477282
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
�;
�

!__inference__traced_save_10648282
file_prefix/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop1
-savev2_dense_13167_kernel_read_readvariableop/
+savev2_dense_13167_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableop8
4savev2_adam_dense_13167_kernel_m_read_readvariableop6
2savev2_adam_dense_13167_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableop8
4savev2_adam_dense_13167_kernel_v_read_readvariableop6
2savev2_adam_dense_13167_bias_v_read_readvariableop3
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

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop-savev2_dense_13167_kernel_read_readvariableop+savev2_dense_13167_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop4savev2_adam_dense_13167_kernel_m_read_readvariableop2savev2_adam_dense_13167_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop4savev2_adam_dense_13167_kernel_v_read_readvariableop2savev2_adam_dense_13167_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�	
�
D__inference_output_layer_call_and_return_conditional_losses_10648175

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
j
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_10647676

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
�
K
/__inference_dropout_4221_layer_call_fn_10648160

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
GPU2 *0J 8� *S
fNRL
J__inference_dropout_4221_layer_call_and_return_conditional_losses_106477392
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
�
�
,__inference_conv2d_63_layer_call_fn_10648107

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
GPU2 *0J 8� *P
fKRI
G__inference_conv2d_63_layer_call_and_return_conditional_losses_106477002
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
� 
�
G__inference_model_252_layer_call_and_return_conditional_losses_10647875

inputs,
conv2d_63_10647854:2 
conv2d_63_10647856:2(
dense_13167_10647863:
��#
dense_13167_10647865:	�"
output_10647869:	�
output_10647871:
identity��!conv2d_63/StatefulPartitionedCall�#dense_13167/StatefulPartitionedCall�$dropout_4221/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_63_10647854conv2d_63_10647856*
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
GPU2 *0J 8� *P
fKRI
G__inference_conv2d_63_layer_call_and_return_conditional_losses_106477002#
!conv2d_63/StatefulPartitionedCall�
tf.reshape_189/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_189/Reshape/shape�
tf.reshape_189/ReshapeReshape*conv2d_63/StatefulPartitionedCall:output:0%tf.reshape_189/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_189/Reshape�
 max_pooling1d_63/PartitionedCallPartitionedCalltf.reshape_189/Reshape:output:0*
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
GPU2 *0J 8� *W
fRRP
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_106476762"
 max_pooling1d_63/PartitionedCall�
flatten_252/PartitionedCallPartitionedCall)max_pooling1d_63/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_flatten_252_layer_call_and_return_conditional_losses_106477152
flatten_252/PartitionedCall�
#dense_13167/StatefulPartitionedCallStatefulPartitionedCall$flatten_252/PartitionedCall:output:0dense_13167_10647863dense_13167_10647865*
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
GPU2 *0J 8� *R
fMRK
I__inference_dense_13167_layer_call_and_return_conditional_losses_106477282%
#dense_13167/StatefulPartitionedCall�
$dropout_4221/StatefulPartitionedCallStatefulPartitionedCall,dense_13167/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *S
fNRL
J__inference_dropout_4221_layer_call_and_return_conditional_losses_106478032&
$dropout_4221/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_4221/StatefulPartitionedCall:output:0output_10647869output_10647871*
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
GPU2 *0J 8� *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_106477512 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_63/StatefulPartitionedCall$^dense_13167/StatefulPartitionedCall%^dropout_4221/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2J
#dense_13167/StatefulPartitionedCall#dense_13167/StatefulPartitionedCall2L
$dropout_4221/StatefulPartitionedCall$dropout_4221/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
G__inference_model_252_layer_call_and_return_conditional_losses_10648013

inputsB
(conv2d_63_conv2d_readvariableop_resource:27
)conv2d_63_biasadd_readvariableop_resource:2>
*dense_13167_matmul_readvariableop_resource:
��:
+dense_13167_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity�� conv2d_63/BiasAdd/ReadVariableOp�conv2d_63/Conv2D/ReadVariableOp�"dense_13167/BiasAdd/ReadVariableOp�!dense_13167/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_63/Conv2D/ReadVariableOp�
conv2d_63/Conv2DConv2Dinputs'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
conv2d_63/Conv2D�
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp�
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22
conv2d_63/BiasAdd~
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:���������22
conv2d_63/Relu�
tf.reshape_189/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_189/Reshape/shape�
tf.reshape_189/ReshapeReshapeconv2d_63/Relu:activations:0%tf.reshape_189/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_189/Reshape�
max_pooling1d_63/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_63/ExpandDims/dim�
max_pooling1d_63/ExpandDims
ExpandDimstf.reshape_189/Reshape:output:0(max_pooling1d_63/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������22
max_pooling1d_63/ExpandDims�
max_pooling1d_63/MaxPoolMaxPool$max_pooling1d_63/ExpandDims:output:0*/
_output_shapes
:���������
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_63/MaxPool�
max_pooling1d_63/SqueezeSqueeze!max_pooling1d_63/MaxPool:output:0*
T0*+
_output_shapes
:���������
2*
squeeze_dims
2
max_pooling1d_63/Squeezew
flatten_252/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_252/Const�
flatten_252/ReshapeReshape!max_pooling1d_63/Squeeze:output:0flatten_252/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_252/Reshape�
!dense_13167/MatMul/ReadVariableOpReadVariableOp*dense_13167_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!dense_13167/MatMul/ReadVariableOp�
dense_13167/MatMulMatMulflatten_252/Reshape:output:0)dense_13167/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13167/MatMul�
"dense_13167/BiasAdd/ReadVariableOpReadVariableOp+dense_13167_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"dense_13167/BiasAdd/ReadVariableOp�
dense_13167/BiasAddBiasAdddense_13167/MatMul:product:0*dense_13167/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13167/BiasAdd}
dense_13167/ReluReludense_13167/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_13167/Relu�
dropout_4221/IdentityIdentitydense_13167/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_4221/Identity�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_4221/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp#^dense_13167/BiasAdd/ReadVariableOp"^dense_13167/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2H
"dense_13167/BiasAdd/ReadVariableOp"dense_13167/BiasAdd/ReadVariableOp2F
!dense_13167/MatMul/ReadVariableOp!dense_13167/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
h
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10647739

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
G__inference_model_252_layer_call_and_return_conditional_losses_10647931
input_onehot,
conv2d_63_10647910:2 
conv2d_63_10647912:2(
dense_13167_10647919:
��#
dense_13167_10647921:	�"
output_10647925:	�
output_10647927:
identity��!conv2d_63/StatefulPartitionedCall�#dense_13167/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_63_10647910conv2d_63_10647912*
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
GPU2 *0J 8� *P
fKRI
G__inference_conv2d_63_layer_call_and_return_conditional_losses_106477002#
!conv2d_63/StatefulPartitionedCall�
tf.reshape_189/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_189/Reshape/shape�
tf.reshape_189/ReshapeReshape*conv2d_63/StatefulPartitionedCall:output:0%tf.reshape_189/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_189/Reshape�
 max_pooling1d_63/PartitionedCallPartitionedCalltf.reshape_189/Reshape:output:0*
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
GPU2 *0J 8� *W
fRRP
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_106476762"
 max_pooling1d_63/PartitionedCall�
flatten_252/PartitionedCallPartitionedCall)max_pooling1d_63/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_flatten_252_layer_call_and_return_conditional_losses_106477152
flatten_252/PartitionedCall�
#dense_13167/StatefulPartitionedCallStatefulPartitionedCall$flatten_252/PartitionedCall:output:0dense_13167_10647919dense_13167_10647921*
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
GPU2 *0J 8� *R
fMRK
I__inference_dense_13167_layer_call_and_return_conditional_losses_106477282%
#dense_13167/StatefulPartitionedCall�
dropout_4221/PartitionedCallPartitionedCall,dense_13167/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *S
fNRL
J__inference_dropout_4221_layer_call_and_return_conditional_losses_106477392
dropout_4221/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_4221/PartitionedCall:output:0output_10647925output_10647927*
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
GPU2 *0J 8� *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_106477512 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_63/StatefulPartitionedCall$^dense_13167/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2J
#dense_13167/StatefulPartitionedCall#dense_13167/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
G__inference_model_252_layer_call_and_return_conditional_losses_10647758

inputs,
conv2d_63_10647701:2 
conv2d_63_10647703:2(
dense_13167_10647729:
��#
dense_13167_10647731:	�"
output_10647752:	�
output_10647754:
identity��!conv2d_63/StatefulPartitionedCall�#dense_13167/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_63_10647701conv2d_63_10647703*
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
GPU2 *0J 8� *P
fKRI
G__inference_conv2d_63_layer_call_and_return_conditional_losses_106477002#
!conv2d_63/StatefulPartitionedCall�
tf.reshape_189/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_189/Reshape/shape�
tf.reshape_189/ReshapeReshape*conv2d_63/StatefulPartitionedCall:output:0%tf.reshape_189/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_189/Reshape�
 max_pooling1d_63/PartitionedCallPartitionedCalltf.reshape_189/Reshape:output:0*
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
GPU2 *0J 8� *W
fRRP
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_106476762"
 max_pooling1d_63/PartitionedCall�
flatten_252/PartitionedCallPartitionedCall)max_pooling1d_63/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_flatten_252_layer_call_and_return_conditional_losses_106477152
flatten_252/PartitionedCall�
#dense_13167/StatefulPartitionedCallStatefulPartitionedCall$flatten_252/PartitionedCall:output:0dense_13167_10647729dense_13167_10647731*
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
GPU2 *0J 8� *R
fMRK
I__inference_dense_13167_layer_call_and_return_conditional_losses_106477282%
#dense_13167/StatefulPartitionedCall�
dropout_4221/PartitionedCallPartitionedCall,dense_13167/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *S
fNRL
J__inference_dropout_4221_layer_call_and_return_conditional_losses_106477392
dropout_4221/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_4221/PartitionedCall:output:0output_10647752output_10647754*
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
GPU2 *0J 8� *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_106477512 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_63/StatefulPartitionedCall$^dense_13167/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2J
#dense_13167/StatefulPartitionedCall#dense_13167/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_flatten_252_layer_call_fn_10648118

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
GPU2 *0J 8� *R
fMRK
I__inference_flatten_252_layer_call_and_return_conditional_losses_106477152
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
�
e
I__inference_flatten_252_layer_call_and_return_conditional_losses_10648113

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
�
�
)__inference_output_layer_call_fn_10648184

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
GPU2 *0J 8� *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_106477512
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

�
I__inference_dense_13167_layer_call_and_return_conditional_losses_10647728

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
�4
�
G__inference_model_252_layer_call_and_return_conditional_losses_10648053

inputsB
(conv2d_63_conv2d_readvariableop_resource:27
)conv2d_63_biasadd_readvariableop_resource:2>
*dense_13167_matmul_readvariableop_resource:
��:
+dense_13167_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity�� conv2d_63/BiasAdd/ReadVariableOp�conv2d_63/Conv2D/ReadVariableOp�"dense_13167/BiasAdd/ReadVariableOp�!dense_13167/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_63/Conv2D/ReadVariableOp�
conv2d_63/Conv2DConv2Dinputs'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingVALID*
strides
2
conv2d_63/Conv2D�
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp�
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22
conv2d_63/BiasAdd~
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:���������22
conv2d_63/Relu�
tf.reshape_189/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_189/Reshape/shape�
tf.reshape_189/ReshapeReshapeconv2d_63/Relu:activations:0%tf.reshape_189/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_189/Reshape�
max_pooling1d_63/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_63/ExpandDims/dim�
max_pooling1d_63/ExpandDims
ExpandDimstf.reshape_189/Reshape:output:0(max_pooling1d_63/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������22
max_pooling1d_63/ExpandDims�
max_pooling1d_63/MaxPoolMaxPool$max_pooling1d_63/ExpandDims:output:0*/
_output_shapes
:���������
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_63/MaxPool�
max_pooling1d_63/SqueezeSqueeze!max_pooling1d_63/MaxPool:output:0*
T0*+
_output_shapes
:���������
2*
squeeze_dims
2
max_pooling1d_63/Squeezew
flatten_252/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_252/Const�
flatten_252/ReshapeReshape!max_pooling1d_63/Squeeze:output:0flatten_252/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_252/Reshape�
!dense_13167/MatMul/ReadVariableOpReadVariableOp*dense_13167_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!dense_13167/MatMul/ReadVariableOp�
dense_13167/MatMulMatMulflatten_252/Reshape:output:0)dense_13167/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13167/MatMul�
"dense_13167/BiasAdd/ReadVariableOpReadVariableOp+dense_13167_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"dense_13167/BiasAdd/ReadVariableOp�
dense_13167/BiasAddBiasAdddense_13167/MatMul:product:0*dense_13167/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13167/BiasAdd}
dense_13167/ReluReludense_13167/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_13167/Relu}
dropout_4221/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_4221/dropout/Const�
dropout_4221/dropout/MulMuldense_13167/Relu:activations:0#dropout_4221/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_4221/dropout/Mul�
dropout_4221/dropout/ShapeShapedense_13167/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4221/dropout/Shape�
1dropout_4221/dropout/random_uniform/RandomUniformRandomUniform#dropout_4221/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype023
1dropout_4221/dropout/random_uniform/RandomUniform�
#dropout_4221/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2%
#dropout_4221/dropout/GreaterEqual/y�
!dropout_4221/dropout/GreaterEqualGreaterEqual:dropout_4221/dropout/random_uniform/RandomUniform:output:0,dropout_4221/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2#
!dropout_4221/dropout/GreaterEqual�
dropout_4221/dropout/CastCast%dropout_4221/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_4221/dropout/Cast�
dropout_4221/dropout/Mul_1Muldropout_4221/dropout/Mul:z:0dropout_4221/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_4221/dropout/Mul_1�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_4221/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp#^dense_13167/BiasAdd/ReadVariableOp"^dense_13167/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2H
"dense_13167/BiasAdd/ReadVariableOp"dense_13167/BiasAdd/ReadVariableOp2F
!dense_13167/MatMul/ReadVariableOp!dense_13167/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
G__inference_model_252_layer_call_and_return_conditional_losses_10647955
input_onehot,
conv2d_63_10647934:2 
conv2d_63_10647936:2(
dense_13167_10647943:
��#
dense_13167_10647945:	�"
output_10647949:	�
output_10647951:
identity��!conv2d_63/StatefulPartitionedCall�#dense_13167/StatefulPartitionedCall�$dropout_4221/StatefulPartitionedCall�output/StatefulPartitionedCall�
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_63_10647934conv2d_63_10647936*
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
GPU2 *0J 8� *P
fKRI
G__inference_conv2d_63_layer_call_and_return_conditional_losses_106477002#
!conv2d_63/StatefulPartitionedCall�
tf.reshape_189/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   2   2
tf.reshape_189/Reshape/shape�
tf.reshape_189/ReshapeReshape*conv2d_63/StatefulPartitionedCall:output:0%tf.reshape_189/Reshape/shape:output:0*
T0*+
_output_shapes
:���������22
tf.reshape_189/Reshape�
 max_pooling1d_63/PartitionedCallPartitionedCalltf.reshape_189/Reshape:output:0*
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
GPU2 *0J 8� *W
fRRP
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_106476762"
 max_pooling1d_63/PartitionedCall�
flatten_252/PartitionedCallPartitionedCall)max_pooling1d_63/PartitionedCall:output:0*
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
GPU2 *0J 8� *R
fMRK
I__inference_flatten_252_layer_call_and_return_conditional_losses_106477152
flatten_252/PartitionedCall�
#dense_13167/StatefulPartitionedCallStatefulPartitionedCall$flatten_252/PartitionedCall:output:0dense_13167_10647943dense_13167_10647945*
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
GPU2 *0J 8� *R
fMRK
I__inference_dense_13167_layer_call_and_return_conditional_losses_106477282%
#dense_13167/StatefulPartitionedCall�
$dropout_4221/StatefulPartitionedCallStatefulPartitionedCall,dense_13167/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *S
fNRL
J__inference_dropout_4221_layer_call_and_return_conditional_losses_106478032&
$dropout_4221/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_4221/StatefulPartitionedCall:output:0output_10647949output_10647951*
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
GPU2 *0J 8� *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_106477512 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_63/StatefulPartitionedCall$^dense_13167/StatefulPartitionedCall%^dropout_4221/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2J
#dense_13167/StatefulPartitionedCall#dense_13167/StatefulPartitionedCall2L
$dropout_4221/StatefulPartitionedCall$dropout_4221/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameinput_onehot"�L
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

	variables
trainable_variables
regularization_losses
	keras_api

signatures
*g&call_and_return_all_conditional_losses
h_default_save_signature
i__call__"�<
_tf_keras_network�<{"name": "model_252", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_252", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_63", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_189", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_189", "inbound_nodes": [["conv2d_63", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_63", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_63", "inbound_nodes": [[["tf.reshape_189", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_252", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_252", "inbound_nodes": [[["max_pooling1d_63", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13167", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13167", "inbound_nodes": [[["flatten_252", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4221", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4221", "inbound_nodes": [[["dense_13167", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_4221", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_252", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_63", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_189", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_189", "inbound_nodes": [["conv2d_63", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_63", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_63", "inbound_nodes": [[["tf.reshape_189", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_252", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_252", "inbound_nodes": [[["max_pooling1d_63", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_13167", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13167", "inbound_nodes": [[["flatten_252", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_4221", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4221", "inbound_nodes": [[["dense_13167", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_4221", 0, 0, {}]]], "shared_object_id": 13}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"�	
_tf_keras_layer�	{"name": "conv2d_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}}
�
	keras_api"�
_tf_keras_layer�{"name": "tf.reshape_189", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.reshape_189", "trainable": true, "dtype": "float32", "function": "reshape"}, "inbound_nodes": [["conv2d_63", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}
�
trainable_variables
	variables
regularization_losses
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"�
_tf_keras_layer�{"name": "max_pooling1d_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_63", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["tf.reshape_189", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 17}}
�
trainable_variables
	variables
regularization_losses
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"�
_tf_keras_layer�{"name": "flatten_252", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_252", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["max_pooling1d_63", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
�	

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
*p&call_and_return_all_conditional_losses
q__call__"�
_tf_keras_layer�{"name": "dense_13167", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13167", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_252", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
�
$trainable_variables
%	variables
&regularization_losses
'	keras_api
*r&call_and_return_all_conditional_losses
s__call__"�
_tf_keras_layer�{"name": "dropout_4221", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4221", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_13167", 0, 0, {}]]], "shared_object_id": 10}
�	

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
*t&call_and_return_all_conditional_losses
u__call__"�
_tf_keras_layer�{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_4221", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratem[m\m]m^(m_)m`vavbvcvd(ve)vf"
	optimizer
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
 "
trackable_list_wrapper
�

3layers

	variables
4metrics
trainable_variables
5non_trainable_variables
6layer_regularization_losses
7layer_metrics
regularization_losses
i__call__
h_default_save_signature
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
*:(22conv2d_63/kernel
:22conv2d_63/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

8layers
trainable_variables
9metrics
	variables
:non_trainable_variables
;layer_regularization_losses
<layer_metrics
regularization_losses
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
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

=layers
trainable_variables
>metrics
	variables
?non_trainable_variables
@layer_regularization_losses
Alayer_metrics
regularization_losses
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Blayers
trainable_variables
Cmetrics
	variables
Dnon_trainable_variables
Elayer_regularization_losses
Flayer_metrics
regularization_losses
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
&:$
��2dense_13167/kernel
:�2dense_13167/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Glayers
 trainable_variables
Hmetrics
!	variables
Inon_trainable_variables
Jlayer_regularization_losses
Klayer_metrics
"regularization_losses
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Llayers
$trainable_variables
Mmetrics
%	variables
Nnon_trainable_variables
Olayer_regularization_losses
Player_metrics
&regularization_losses
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 :	�2output/kernel
:2output/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Qlayers
*trainable_variables
Rmetrics
+	variables
Snon_trainable_variables
Tlayer_regularization_losses
Ulayer_metrics
,regularization_losses
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
'
V0"
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
/:-22Adam/conv2d_63/kernel/m
!:22Adam/conv2d_63/bias/m
+:)
��2Adam/dense_13167/kernel/m
$:"�2Adam/dense_13167/bias/m
%:#	�2Adam/output/kernel/m
:2Adam/output/bias/m
/:-22Adam/conv2d_63/kernel/v
!:22Adam/conv2d_63/bias/v
+:)
��2Adam/dense_13167/kernel/v
$:"�2Adam/dense_13167/bias/v
%:#	�2Adam/output/kernel/v
:2Adam/output/bias/v
�2�
G__inference_model_252_layer_call_and_return_conditional_losses_10648013
G__inference_model_252_layer_call_and_return_conditional_losses_10648053
G__inference_model_252_layer_call_and_return_conditional_losses_10647931
G__inference_model_252_layer_call_and_return_conditional_losses_10647955�
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
#__inference__wrapped_model_10647667�
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
,__inference_model_252_layer_call_fn_10647773
,__inference_model_252_layer_call_fn_10648070
,__inference_model_252_layer_call_fn_10648087
,__inference_model_252_layer_call_fn_10647907�
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
G__inference_conv2d_63_layer_call_and_return_conditional_losses_10648098�
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
,__inference_conv2d_63_layer_call_fn_10648107�
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
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_10647676�
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
3__inference_max_pooling1d_63_layer_call_fn_10647682�
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
I__inference_flatten_252_layer_call_and_return_conditional_losses_10648113�
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
.__inference_flatten_252_layer_call_fn_10648118�
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
I__inference_dense_13167_layer_call_and_return_conditional_losses_10648129�
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
.__inference_dense_13167_layer_call_fn_10648138�
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
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10648143
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10648155�
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
/__inference_dropout_4221_layer_call_fn_10648160
/__inference_dropout_4221_layer_call_fn_10648165�
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
D__inference_output_layer_call_and_return_conditional_losses_10648175�
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
)__inference_output_layer_call_fn_10648184�
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
&__inference_signature_wrapper_10647980input_onehot"�
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
#__inference__wrapped_model_10647667x()=�:
3�0
.�+
input_onehot���������
� "/�,
*
output �
output����������
G__inference_conv2d_63_layer_call_and_return_conditional_losses_10648098l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������2
� �
,__inference_conv2d_63_layer_call_fn_10648107_7�4
-�*
(�%
inputs���������
� " ����������2�
I__inference_dense_13167_layer_call_and_return_conditional_losses_10648129^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
.__inference_dense_13167_layer_call_fn_10648138Q0�-
&�#
!�
inputs����������
� "������������
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10648143^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
J__inference_dropout_4221_layer_call_and_return_conditional_losses_10648155^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
/__inference_dropout_4221_layer_call_fn_10648160Q4�1
*�'
!�
inputs����������
p 
� "������������
/__inference_dropout_4221_layer_call_fn_10648165Q4�1
*�'
!�
inputs����������
p
� "������������
I__inference_flatten_252_layer_call_and_return_conditional_losses_10648113]3�0
)�&
$�!
inputs���������
2
� "&�#
�
0����������
� �
.__inference_flatten_252_layer_call_fn_10648118P3�0
)�&
$�!
inputs���������
2
� "������������
N__inference_max_pooling1d_63_layer_call_and_return_conditional_losses_10647676�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
3__inference_max_pooling1d_63_layer_call_fn_10647682wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
G__inference_model_252_layer_call_and_return_conditional_losses_10647931v()E�B
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
G__inference_model_252_layer_call_and_return_conditional_losses_10647955v()E�B
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
G__inference_model_252_layer_call_and_return_conditional_losses_10648013p()?�<
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
G__inference_model_252_layer_call_and_return_conditional_losses_10648053p()?�<
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
,__inference_model_252_layer_call_fn_10647773i()E�B
;�8
.�+
input_onehot���������
p 

 
� "�����������
,__inference_model_252_layer_call_fn_10647907i()E�B
;�8
.�+
input_onehot���������
p

 
� "�����������
,__inference_model_252_layer_call_fn_10648070c()?�<
5�2
(�%
inputs���������
p 

 
� "�����������
,__inference_model_252_layer_call_fn_10648087c()?�<
5�2
(�%
inputs���������
p

 
� "�����������
D__inference_output_layer_call_and_return_conditional_losses_10648175]()0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_output_layer_call_fn_10648184P()0�-
&�#
!�
inputs����������
� "�����������
&__inference_signature_wrapper_10647980�()M�J
� 
C�@
>
input_onehot.�+
input_onehot���������"/�,
*
output �
output���������