╙ъ
▄м
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
В
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
delete_old_dirsbool(И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718╤Є
Ж
conv2d_120/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameconv2d_120/kernel

%conv2d_120/kernel/Read/ReadVariableOpReadVariableOpconv2d_120/kernel*&
_output_shapes
:2*
dtype0
v
conv2d_120/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_nameconv2d_120/bias
o
#conv2d_120/bias/Read/ReadVariableOpReadVariableOpconv2d_120/bias*
_output_shapes
:2*
dtype0
А
dense_1776/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЇЇ*"
shared_namedense_1776/kernel
y
%dense_1776/kernel/Read/ReadVariableOpReadVariableOpdense_1776/kernel* 
_output_shapes
:
ЇЇ*
dtype0
w
dense_1776/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї* 
shared_namedense_1776/bias
p
#dense_1776/bias/Read/ReadVariableOpReadVariableOpdense_1776/bias*
_output_shapes	
:Ї*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ї*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	Ї*
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
Ф
Adam/conv2d_120/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/conv2d_120/kernel/m
Н
,Adam/conv2d_120/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_120/kernel/m*&
_output_shapes
:2*
dtype0
Д
Adam/conv2d_120/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/conv2d_120/bias/m
}
*Adam/conv2d_120/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_120/bias/m*
_output_shapes
:2*
dtype0
О
Adam/dense_1776/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЇЇ*)
shared_nameAdam/dense_1776/kernel/m
З
,Adam/dense_1776/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1776/kernel/m* 
_output_shapes
:
ЇЇ*
dtype0
Е
Adam/dense_1776/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*'
shared_nameAdam/dense_1776/bias/m
~
*Adam/dense_1776/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1776/bias/m*
_output_shapes	
:Ї*
dtype0
Е
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ї*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	Ї*
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
Ф
Adam/conv2d_120/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/conv2d_120/kernel/v
Н
,Adam/conv2d_120/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_120/kernel/v*&
_output_shapes
:2*
dtype0
Д
Adam/conv2d_120/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/conv2d_120/bias/v
}
*Adam/conv2d_120/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_120/bias/v*
_output_shapes
:2*
dtype0
О
Adam/dense_1776/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЇЇ*)
shared_nameAdam/dense_1776/kernel/v
З
,Adam/dense_1776/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1776/kernel/v* 
_output_shapes
:
ЇЇ*
dtype0
Е
Adam/dense_1776/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*'
shared_nameAdam/dense_1776/bias/v
~
*Adam/dense_1776/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1776/bias/v*
_output_shapes	
:Ї*
dtype0
Е
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ї*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	Ї*
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
╦*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ж*
value№)B∙) BЄ)
з
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
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
м
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
 
*
0
1
2
3
(4
)5
н
3layer_regularization_losses
4layer_metrics

	variables

5layers
regularization_losses
6metrics
trainable_variables
7non_trainable_variables
 
][
VARIABLE_VALUEconv2d_120/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_120/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
8layer_regularization_losses
9layer_metrics
	variables

:layers
regularization_losses
;metrics
trainable_variables
<non_trainable_variables
 
 
 
 
н
=layer_regularization_losses
>layer_metrics
	variables

?layers
regularization_losses
@metrics
trainable_variables
Anon_trainable_variables
 
 
 
н
Blayer_regularization_losses
Clayer_metrics
	variables

Dlayers
regularization_losses
Emetrics
trainable_variables
Fnon_trainable_variables
][
VARIABLE_VALUEdense_1776/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_1776/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
Glayer_regularization_losses
Hlayer_metrics
 	variables

Ilayers
!regularization_losses
Jmetrics
"trainable_variables
Knon_trainable_variables
 
 
 
н
Llayer_regularization_losses
Mlayer_metrics
$	variables

Nlayers
%regularization_losses
Ometrics
&trainable_variables
Pnon_trainable_variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
н
Qlayer_regularization_losses
Rlayer_metrics
*	variables

Slayers
+regularization_losses
Tmetrics
,trainable_variables
Unon_trainable_variables
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
А~
VARIABLE_VALUEAdam/conv2d_120/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_120/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/dense_1776/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1776/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv2d_120/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_120/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/dense_1776/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1776/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
П
serving_default_input_onehotPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
н
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_120/kernelconv2d_120/biasdense_1776/kerneldense_1776/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_1017258
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Б

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_120/kernel/Read/ReadVariableOp#conv2d_120/bias/Read/ReadVariableOp%dense_1776/kernel/Read/ReadVariableOp#dense_1776/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_120/kernel/m/Read/ReadVariableOp*Adam/conv2d_120/bias/m/Read/ReadVariableOp,Adam/dense_1776/kernel/m/Read/ReadVariableOp*Adam/dense_1776/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp,Adam/conv2d_120/kernel/v/Read/ReadVariableOp*Adam/conv2d_120/bias/v/Read/ReadVariableOp,Adam/dense_1776/kernel/v/Read/ReadVariableOp*Adam/dense_1776/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
GPU2 *0J 8В *)
f$R"
 __inference__traced_save_1017560
И
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_120/kernelconv2d_120/biasdense_1776/kerneldense_1776/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_120/kernel/mAdam/conv2d_120/bias/mAdam/dense_1776/kernel/mAdam/dense_1776/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_120/kernel/vAdam/conv2d_120/bias/vAdam/dense_1776/kernel/vAdam/dense_1776/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_1017645ўВ
б
Ц
(__inference_output_layer_call_fn_1017462

inputs
unknown:	Ї
	unknown_0:
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_10170292
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
│m
й
#__inference__traced_restore_1017645
file_prefix<
"assignvariableop_conv2d_120_kernel:20
"assignvariableop_1_conv2d_120_bias:28
$assignvariableop_2_dense_1776_kernel:
ЇЇ1
"assignvariableop_3_dense_1776_bias:	Ї3
 assignvariableop_4_output_kernel:	Ї,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: F
,assignvariableop_13_adam_conv2d_120_kernel_m:28
*assignvariableop_14_adam_conv2d_120_bias_m:2@
,assignvariableop_15_adam_dense_1776_kernel_m:
ЇЇ9
*assignvariableop_16_adam_dense_1776_bias_m:	Ї;
(assignvariableop_17_adam_output_kernel_m:	Ї4
&assignvariableop_18_adam_output_bias_m:F
,assignvariableop_19_adam_conv2d_120_kernel_v:28
*assignvariableop_20_adam_conv2d_120_bias_v:2@
,assignvariableop_21_adam_dense_1776_kernel_v:
ЇЇ9
*assignvariableop_22_adam_dense_1776_bias_v:	Ї;
(assignvariableop_23_adam_output_kernel_v:	Ї4
&assignvariableop_24_adam_output_bias_v:
identity_26ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▓
valueиBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names┬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesн
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

Identityб
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_120_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1з
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_120_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2й
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_1776_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3з
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_1776_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4е
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6б
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7г
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8г
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9в
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10о
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11б
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12б
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13┤
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_conv2d_120_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14▓
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_conv2d_120_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┤
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_1776_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16▓
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_1776_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17░
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18о
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┤
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv2d_120_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20▓
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_120_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21┤
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_1776_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22▓
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_1776_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23░
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_output_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24о
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_output_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpД
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25ў
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
й 
╤
E__inference_model_64_layer_call_and_return_conditional_losses_1017153

inputs,
conv2d_120_1017132:2 
conv2d_120_1017134:2&
dense_1776_1017141:
ЇЇ!
dense_1776_1017143:	Ї!
output_1017147:	Ї
output_1017149:
identityИв"conv2d_120/StatefulPartitionedCallв"dense_1776/StatefulPartitionedCallв#dropout_576/StatefulPartitionedCallвoutput/StatefulPartitionedCallо
"conv2d_120/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_120_1017132conv2d_120_1017134*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_120_layer_call_and_return_conditional_losses_10169782$
"conv2d_120/StatefulPartitionedCallП
tf.reshape_88/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       2   2
tf.reshape_88/Reshape/shape┬
tf.reshape_88/ReshapeReshape+conv2d_120/StatefulPartitionedCall:output:0$tf.reshape_88/Reshape/shape:output:0*
T0*+
_output_shapes
:         22
tf.reshape_88/ReshapeО
 max_pooling1d_32/PartitionedCallPartitionedCalltf.reshape_88/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_10169542"
 max_pooling1d_32/PartitionedCallД
flatten_72/PartitionedCallPartitionedCall)max_pooling1d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_72_layer_call_and_return_conditional_losses_10169932
flatten_72/PartitionedCall─
"dense_1776/StatefulPartitionedCallStatefulPartitionedCall#flatten_72/PartitionedCall:output:0dense_1776_1017141dense_1776_1017143*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dense_1776_layer_call_and_return_conditional_losses_10170062$
"dense_1776/StatefulPartitionedCallб
#dropout_576/StatefulPartitionedCallStatefulPartitionedCall+dense_1776/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dropout_576_layer_call_and_return_conditional_losses_10170812%
#dropout_576/StatefulPartitionedCall╕
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_576/StatefulPartitionedCall:output:0output_1017147output_1017149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_10170292 
output/StatefulPartitionedCallМ
IdentityIdentity'output/StatefulPartitionedCall:output:0#^conv2d_120/StatefulPartitionedCall#^dense_1776/StatefulPartitionedCall$^dropout_576/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 2H
"conv2d_120/StatefulPartitionedCall"conv2d_120/StatefulPartitionedCall2H
"dense_1776/StatefulPartitionedCall"dense_1776/StatefulPartitionedCall2J
#dropout_576/StatefulPartitionedCall#dropout_576/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╖
g
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017081

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
с
c
G__inference_flatten_72_layer_call_and_return_conditional_losses_1016993

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Ї  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         Ї2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
2:S O
+
_output_shapes
:         
2
 
_user_specified_nameinputs
с
c
G__inference_flatten_72_layer_call_and_return_conditional_losses_1017391

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    Ї  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         Ї2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
2:S O
+
_output_shapes
:         
2
 
_user_specified_nameinputs
╨
Р
%__inference_signature_wrapper_1017258
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
ЇЇ
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_10169452
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_onehot
╗

√
G__inference_dense_1776_layer_call_and_return_conditional_losses_1017006

inputs2
matmul_readvariableop_resource:
ЇЇ.
biasadd_readvariableop_resource:	Ї
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
ё
л
E__inference_model_64_layer_call_and_return_conditional_losses_1017036

inputs,
conv2d_120_1016979:2 
conv2d_120_1016981:2&
dense_1776_1017007:
ЇЇ!
dense_1776_1017009:	Ї!
output_1017030:	Ї
output_1017032:
identityИв"conv2d_120/StatefulPartitionedCallв"dense_1776/StatefulPartitionedCallвoutput/StatefulPartitionedCallо
"conv2d_120/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_120_1016979conv2d_120_1016981*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_120_layer_call_and_return_conditional_losses_10169782$
"conv2d_120/StatefulPartitionedCallП
tf.reshape_88/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       2   2
tf.reshape_88/Reshape/shape┬
tf.reshape_88/ReshapeReshape+conv2d_120/StatefulPartitionedCall:output:0$tf.reshape_88/Reshape/shape:output:0*
T0*+
_output_shapes
:         22
tf.reshape_88/ReshapeО
 max_pooling1d_32/PartitionedCallPartitionedCalltf.reshape_88/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_10169542"
 max_pooling1d_32/PartitionedCallД
flatten_72/PartitionedCallPartitionedCall)max_pooling1d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_72_layer_call_and_return_conditional_losses_10169932
flatten_72/PartitionedCall─
"dense_1776/StatefulPartitionedCallStatefulPartitionedCall#flatten_72/PartitionedCall:output:0dense_1776_1017007dense_1776_1017009*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dense_1776_layer_call_and_return_conditional_losses_10170062$
"dense_1776/StatefulPartitionedCallЙ
dropout_576/PartitionedCallPartitionedCall+dense_1776/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dropout_576_layer_call_and_return_conditional_losses_10170172
dropout_576/PartitionedCall░
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_576/PartitionedCall:output:0output_1017030output_1017032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_10170292 
output/StatefulPartitionedCallц
IdentityIdentity'output/StatefulPartitionedCall:output:0#^conv2d_120/StatefulPartitionedCall#^dense_1776/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 2H
"conv2d_120/StatefulPartitionedCall"conv2d_120/StatefulPartitionedCall2H
"dense_1776/StatefulPartitionedCall"dense_1776/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
°
Х
*__inference_model_64_layer_call_fn_1017051
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
ЇЇ
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_64_layer_call_and_return_conditional_losses_10170362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_onehot
╫*
Ъ
E__inference_model_64_layer_call_and_return_conditional_losses_1017291

inputsC
)conv2d_120_conv2d_readvariableop_resource:28
*conv2d_120_biasadd_readvariableop_resource:2=
)dense_1776_matmul_readvariableop_resource:
ЇЇ9
*dense_1776_biasadd_readvariableop_resource:	Ї8
%output_matmul_readvariableop_resource:	Ї4
&output_biasadd_readvariableop_resource:
identityИв!conv2d_120/BiasAdd/ReadVariableOpв conv2d_120/Conv2D/ReadVariableOpв!dense_1776/BiasAdd/ReadVariableOpв dense_1776/MatMul/ReadVariableOpвoutput/BiasAdd/ReadVariableOpвoutput/MatMul/ReadVariableOp╢
 conv2d_120/Conv2D/ReadVariableOpReadVariableOp)conv2d_120_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02"
 conv2d_120/Conv2D/ReadVariableOp┼
conv2d_120/Conv2DConv2Dinputs(conv2d_120/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2*
paddingVALID*
strides
2
conv2d_120/Conv2Dн
!conv2d_120/BiasAdd/ReadVariableOpReadVariableOp*conv2d_120_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02#
!conv2d_120/BiasAdd/ReadVariableOp┤
conv2d_120/BiasAddBiasAddconv2d_120/Conv2D:output:0)conv2d_120/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22
conv2d_120/BiasAddБ
conv2d_120/ReluReluconv2d_120/BiasAdd:output:0*
T0*/
_output_shapes
:         22
conv2d_120/ReluП
tf.reshape_88/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       2   2
tf.reshape_88/Reshape/shape┤
tf.reshape_88/ReshapeReshapeconv2d_120/Relu:activations:0$tf.reshape_88/Reshape/shape:output:0*
T0*+
_output_shapes
:         22
tf.reshape_88/ReshapeД
max_pooling1d_32/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_32/ExpandDims/dim╠
max_pooling1d_32/ExpandDims
ExpandDimstf.reshape_88/Reshape:output:0(max_pooling1d_32/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         22
max_pooling1d_32/ExpandDims╥
max_pooling1d_32/MaxPoolMaxPool$max_pooling1d_32/ExpandDims:output:0*/
_output_shapes
:         
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_32/MaxPoolп
max_pooling1d_32/SqueezeSqueeze!max_pooling1d_32/MaxPool:output:0*
T0*+
_output_shapes
:         
2*
squeeze_dims
2
max_pooling1d_32/Squeezeu
flatten_72/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Ї  2
flatten_72/Constд
flatten_72/ReshapeReshape!max_pooling1d_32/Squeeze:output:0flatten_72/Const:output:0*
T0*(
_output_shapes
:         Ї2
flatten_72/Reshape░
 dense_1776/MatMul/ReadVariableOpReadVariableOp)dense_1776_matmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02"
 dense_1776/MatMul/ReadVariableOpк
dense_1776/MatMulMatMulflatten_72/Reshape:output:0(dense_1776/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1776/MatMulо
!dense_1776/BiasAdd/ReadVariableOpReadVariableOp*dense_1776_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02#
!dense_1776/BiasAdd/ReadVariableOpо
dense_1776/BiasAddBiasAdddense_1776/MatMul:product:0)dense_1776/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1776/BiasAddz
dense_1776/ReluReludense_1776/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
dense_1776/ReluК
dropout_576/IdentityIdentitydense_1776/Relu:activations:0*
T0*(
_output_shapes
:         Ї2
dropout_576/Identityг
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
output/MatMul/ReadVariableOpЯ
output/MatMulMatMuldropout_576/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulб
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAdd╕
IdentityIdentityoutput/BiasAdd:output:0"^conv2d_120/BiasAdd/ReadVariableOp!^conv2d_120/Conv2D/ReadVariableOp"^dense_1776/BiasAdd/ReadVariableOp!^dense_1776/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 2F
!conv2d_120/BiasAdd/ReadVariableOp!conv2d_120/BiasAdd/ReadVariableOp2D
 conv2d_120/Conv2D/ReadVariableOp conv2d_120/Conv2D/ReadVariableOp2F
!dense_1776/BiasAdd/ReadVariableOp!dense_1776/BiasAdd/ReadVariableOp2D
 dense_1776/MatMul/ReadVariableOp dense_1776/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
°
Х
*__inference_model_64_layer_call_fn_1017185
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
ЇЇ
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_64_layer_call_and_return_conditional_losses_10171532
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_onehot
н
Ь
,__inference_dense_1776_layer_call_fn_1017416

inputs
unknown:
ЇЇ
	unknown_0:	Ї
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dense_1776_layer_call_and_return_conditional_losses_10170062
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ш
А
G__inference_conv2d_120_layer_call_and_return_conditional_losses_1017376

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         22
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ф
i
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_1016954

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Г
▒
E__inference_model_64_layer_call_and_return_conditional_losses_1017209
input_onehot,
conv2d_120_1017188:2 
conv2d_120_1017190:2&
dense_1776_1017197:
ЇЇ!
dense_1776_1017199:	Ї!
output_1017203:	Ї
output_1017205:
identityИв"conv2d_120/StatefulPartitionedCallв"dense_1776/StatefulPartitionedCallвoutput/StatefulPartitionedCall┤
"conv2d_120/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_120_1017188conv2d_120_1017190*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_120_layer_call_and_return_conditional_losses_10169782$
"conv2d_120/StatefulPartitionedCallП
tf.reshape_88/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       2   2
tf.reshape_88/Reshape/shape┬
tf.reshape_88/ReshapeReshape+conv2d_120/StatefulPartitionedCall:output:0$tf.reshape_88/Reshape/shape:output:0*
T0*+
_output_shapes
:         22
tf.reshape_88/ReshapeО
 max_pooling1d_32/PartitionedCallPartitionedCalltf.reshape_88/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_10169542"
 max_pooling1d_32/PartitionedCallД
flatten_72/PartitionedCallPartitionedCall)max_pooling1d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_72_layer_call_and_return_conditional_losses_10169932
flatten_72/PartitionedCall─
"dense_1776/StatefulPartitionedCallStatefulPartitionedCall#flatten_72/PartitionedCall:output:0dense_1776_1017197dense_1776_1017199*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dense_1776_layer_call_and_return_conditional_losses_10170062$
"dense_1776/StatefulPartitionedCallЙ
dropout_576/PartitionedCallPartitionedCall+dense_1776/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dropout_576_layer_call_and_return_conditional_losses_10170172
dropout_576/PartitionedCall░
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_576/PartitionedCall:output:0output_1017203output_1017205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_10170292 
output/StatefulPartitionedCallц
IdentityIdentity'output/StatefulPartitionedCall:output:0#^conv2d_120/StatefulPartitionedCall#^dense_1776/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 2H
"conv2d_120/StatefulPartitionedCall"conv2d_120/StatefulPartitionedCall2H
"dense_1776/StatefulPartitionedCall"dense_1776/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_onehot
╗ 
╫
E__inference_model_64_layer_call_and_return_conditional_losses_1017233
input_onehot,
conv2d_120_1017212:2 
conv2d_120_1017214:2&
dense_1776_1017221:
ЇЇ!
dense_1776_1017223:	Ї!
output_1017227:	Ї
output_1017229:
identityИв"conv2d_120/StatefulPartitionedCallв"dense_1776/StatefulPartitionedCallв#dropout_576/StatefulPartitionedCallвoutput/StatefulPartitionedCall┤
"conv2d_120/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_120_1017212conv2d_120_1017214*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_120_layer_call_and_return_conditional_losses_10169782$
"conv2d_120/StatefulPartitionedCallП
tf.reshape_88/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       2   2
tf.reshape_88/Reshape/shape┬
tf.reshape_88/ReshapeReshape+conv2d_120/StatefulPartitionedCall:output:0$tf.reshape_88/Reshape/shape:output:0*
T0*+
_output_shapes
:         22
tf.reshape_88/ReshapeО
 max_pooling1d_32/PartitionedCallPartitionedCalltf.reshape_88/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_10169542"
 max_pooling1d_32/PartitionedCallД
flatten_72/PartitionedCallPartitionedCall)max_pooling1d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_72_layer_call_and_return_conditional_losses_10169932
flatten_72/PartitionedCall─
"dense_1776/StatefulPartitionedCallStatefulPartitionedCall#flatten_72/PartitionedCall:output:0dense_1776_1017221dense_1776_1017223*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_dense_1776_layer_call_and_return_conditional_losses_10170062$
"dense_1776/StatefulPartitionedCallб
#dropout_576/StatefulPartitionedCallStatefulPartitionedCall+dense_1776/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dropout_576_layer_call_and_return_conditional_losses_10170812%
#dropout_576/StatefulPartitionedCall╕
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_576/StatefulPartitionedCall:output:0output_1017227output_1017229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_10170292 
output/StatefulPartitionedCallМ
IdentityIdentity'output/StatefulPartitionedCall:output:0#^conv2d_120/StatefulPartitionedCall#^dense_1776/StatefulPartitionedCall$^dropout_576/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 2H
"conv2d_120/StatefulPartitionedCall"conv2d_120/StatefulPartitionedCall2H
"dense_1776/StatefulPartitionedCall"dense_1776/StatefulPartitionedCall2J
#dropout_576/StatefulPartitionedCall#dropout_576/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_onehot
м
N
2__inference_max_pooling1d_32_layer_call_fn_1016960

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_10169542
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
┘
f
-__inference_dropout_576_layer_call_fn_1017443

inputs
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dropout_576_layer_call_and_return_conditional_losses_10170812
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         Ї22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
∙1
щ
"__inference__wrapped_model_1016945
input_onehotL
2model_64_conv2d_120_conv2d_readvariableop_resource:2A
3model_64_conv2d_120_biasadd_readvariableop_resource:2F
2model_64_dense_1776_matmul_readvariableop_resource:
ЇЇB
3model_64_dense_1776_biasadd_readvariableop_resource:	ЇA
.model_64_output_matmul_readvariableop_resource:	Ї=
/model_64_output_biasadd_readvariableop_resource:
identityИв*model_64/conv2d_120/BiasAdd/ReadVariableOpв)model_64/conv2d_120/Conv2D/ReadVariableOpв*model_64/dense_1776/BiasAdd/ReadVariableOpв)model_64/dense_1776/MatMul/ReadVariableOpв&model_64/output/BiasAdd/ReadVariableOpв%model_64/output/MatMul/ReadVariableOp╤
)model_64/conv2d_120/Conv2D/ReadVariableOpReadVariableOp2model_64_conv2d_120_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02+
)model_64/conv2d_120/Conv2D/ReadVariableOpц
model_64/conv2d_120/Conv2DConv2Dinput_onehot1model_64/conv2d_120/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2*
paddingVALID*
strides
2
model_64/conv2d_120/Conv2D╚
*model_64/conv2d_120/BiasAdd/ReadVariableOpReadVariableOp3model_64_conv2d_120_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02,
*model_64/conv2d_120/BiasAdd/ReadVariableOp╪
model_64/conv2d_120/BiasAddBiasAdd#model_64/conv2d_120/Conv2D:output:02model_64/conv2d_120/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22
model_64/conv2d_120/BiasAddЬ
model_64/conv2d_120/ReluRelu$model_64/conv2d_120/BiasAdd:output:0*
T0*/
_output_shapes
:         22
model_64/conv2d_120/Reluб
$model_64/tf.reshape_88/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       2   2&
$model_64/tf.reshape_88/Reshape/shape╪
model_64/tf.reshape_88/ReshapeReshape&model_64/conv2d_120/Relu:activations:0-model_64/tf.reshape_88/Reshape/shape:output:0*
T0*+
_output_shapes
:         22 
model_64/tf.reshape_88/ReshapeЦ
(model_64/max_pooling1d_32/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_64/max_pooling1d_32/ExpandDims/dimЁ
$model_64/max_pooling1d_32/ExpandDims
ExpandDims'model_64/tf.reshape_88/Reshape:output:01model_64/max_pooling1d_32/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         22&
$model_64/max_pooling1d_32/ExpandDimsэ
!model_64/max_pooling1d_32/MaxPoolMaxPool-model_64/max_pooling1d_32/ExpandDims:output:0*/
_output_shapes
:         
2*
ksize
*
paddingVALID*
strides
2#
!model_64/max_pooling1d_32/MaxPool╩
!model_64/max_pooling1d_32/SqueezeSqueeze*model_64/max_pooling1d_32/MaxPool:output:0*
T0*+
_output_shapes
:         
2*
squeeze_dims
2#
!model_64/max_pooling1d_32/SqueezeЗ
model_64/flatten_72/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Ї  2
model_64/flatten_72/Const╚
model_64/flatten_72/ReshapeReshape*model_64/max_pooling1d_32/Squeeze:output:0"model_64/flatten_72/Const:output:0*
T0*(
_output_shapes
:         Ї2
model_64/flatten_72/Reshape╦
)model_64/dense_1776/MatMul/ReadVariableOpReadVariableOp2model_64_dense_1776_matmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02+
)model_64/dense_1776/MatMul/ReadVariableOp╬
model_64/dense_1776/MatMulMatMul$model_64/flatten_72/Reshape:output:01model_64/dense_1776/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
model_64/dense_1776/MatMul╔
*model_64/dense_1776/BiasAdd/ReadVariableOpReadVariableOp3model_64_dense_1776_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02,
*model_64/dense_1776/BiasAdd/ReadVariableOp╥
model_64/dense_1776/BiasAddBiasAdd$model_64/dense_1776/MatMul:product:02model_64/dense_1776/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
model_64/dense_1776/BiasAddХ
model_64/dense_1776/ReluRelu$model_64/dense_1776/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
model_64/dense_1776/Reluе
model_64/dropout_576/IdentityIdentity&model_64/dense_1776/Relu:activations:0*
T0*(
_output_shapes
:         Ї2
model_64/dropout_576/Identity╛
%model_64/output/MatMul/ReadVariableOpReadVariableOp.model_64_output_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02'
%model_64/output/MatMul/ReadVariableOp├
model_64/output/MatMulMatMul&model_64/dropout_576/Identity:output:0-model_64/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_64/output/MatMul╝
&model_64/output/BiasAdd/ReadVariableOpReadVariableOp/model_64_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_64/output/BiasAdd/ReadVariableOp┴
model_64/output/BiasAddBiasAdd model_64/output/MatMul:product:0.model_64/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_64/output/BiasAddў
IdentityIdentity model_64/output/BiasAdd:output:0+^model_64/conv2d_120/BiasAdd/ReadVariableOp*^model_64/conv2d_120/Conv2D/ReadVariableOp+^model_64/dense_1776/BiasAdd/ReadVariableOp*^model_64/dense_1776/MatMul/ReadVariableOp'^model_64/output/BiasAdd/ReadVariableOp&^model_64/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 2X
*model_64/conv2d_120/BiasAdd/ReadVariableOp*model_64/conv2d_120/BiasAdd/ReadVariableOp2V
)model_64/conv2d_120/Conv2D/ReadVariableOp)model_64/conv2d_120/Conv2D/ReadVariableOp2X
*model_64/dense_1776/BiasAdd/ReadVariableOp*model_64/dense_1776/BiasAdd/ReadVariableOp2V
)model_64/dense_1776/MatMul/ReadVariableOp)model_64/dense_1776/MatMul/ReadVariableOp2P
&model_64/output/BiasAdd/ReadVariableOp&model_64/output/BiasAdd/ReadVariableOp2N
%model_64/output/MatMul/ReadVariableOp%model_64/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_onehot
╖;
╟

 __inference__traced_save_1017560
file_prefix0
,savev2_conv2d_120_kernel_read_readvariableop.
*savev2_conv2d_120_bias_read_readvariableop0
,savev2_dense_1776_kernel_read_readvariableop.
*savev2_dense_1776_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_120_kernel_m_read_readvariableop5
1savev2_adam_conv2d_120_bias_m_read_readvariableop7
3savev2_adam_dense_1776_kernel_m_read_readvariableop5
1savev2_adam_dense_1776_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop7
3savev2_adam_conv2d_120_kernel_v_read_readvariableop5
1savev2_adam_conv2d_120_bias_v_read_readvariableop7
3savev2_adam_dense_1776_kernel_v_read_readvariableop5
1savev2_adam_dense_1776_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameа
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▓
valueиBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╝
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╔

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_120_kernel_read_readvariableop*savev2_conv2d_120_bias_read_readvariableop,savev2_dense_1776_kernel_read_readvariableop*savev2_dense_1776_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_120_kernel_m_read_readvariableop1savev2_adam_conv2d_120_bias_m_read_readvariableop3savev2_adam_dense_1776_kernel_m_read_readvariableop1savev2_adam_dense_1776_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop3savev2_adam_conv2d_120_kernel_v_read_readvariableop1savev2_adam_conv2d_120_bias_v_read_readvariableop3savev2_adam_dense_1776_kernel_v_read_readvariableop1savev2_adam_dense_1776_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*█
_input_shapes╔
╞: :2:2:
ЇЇ:Ї:	Ї:: : : : : : : :2:2:
ЇЇ:Ї:	Ї::2:2:
ЇЇ:Ї:	Ї:: 2(
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
ЇЇ:!

_output_shapes	
:Ї:%!

_output_shapes
:	Ї: 
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
ЇЇ:!

_output_shapes	
:Ї:%!

_output_shapes
:	Ї: 
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
ЇЇ:!

_output_shapes	
:Ї:%!

_output_shapes
:	Ї: 

_output_shapes
::

_output_shapes
: 
╤
H
,__inference_flatten_72_layer_call_fn_1017396

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_72_layer_call_and_return_conditional_losses_10169932
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
2:S O
+
_output_shapes
:         
2
 
_user_specified_nameinputs
╖
g
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017433

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
∙
f
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017421

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         Ї2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
∙
f
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017017

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         Ї2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
╬
б
,__inference_conv2d_120_layer_call_fn_1017385

inputs!
unknown:2
	unknown_0:2
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_120_layer_call_and_return_conditional_losses_10169782
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▓4
Ъ
E__inference_model_64_layer_call_and_return_conditional_losses_1017331

inputsC
)conv2d_120_conv2d_readvariableop_resource:28
*conv2d_120_biasadd_readvariableop_resource:2=
)dense_1776_matmul_readvariableop_resource:
ЇЇ9
*dense_1776_biasadd_readvariableop_resource:	Ї8
%output_matmul_readvariableop_resource:	Ї4
&output_biasadd_readvariableop_resource:
identityИв!conv2d_120/BiasAdd/ReadVariableOpв conv2d_120/Conv2D/ReadVariableOpв!dense_1776/BiasAdd/ReadVariableOpв dense_1776/MatMul/ReadVariableOpвoutput/BiasAdd/ReadVariableOpвoutput/MatMul/ReadVariableOp╢
 conv2d_120/Conv2D/ReadVariableOpReadVariableOp)conv2d_120_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02"
 conv2d_120/Conv2D/ReadVariableOp┼
conv2d_120/Conv2DConv2Dinputs(conv2d_120/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2*
paddingVALID*
strides
2
conv2d_120/Conv2Dн
!conv2d_120/BiasAdd/ReadVariableOpReadVariableOp*conv2d_120_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02#
!conv2d_120/BiasAdd/ReadVariableOp┤
conv2d_120/BiasAddBiasAddconv2d_120/Conv2D:output:0)conv2d_120/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22
conv2d_120/BiasAddБ
conv2d_120/ReluReluconv2d_120/BiasAdd:output:0*
T0*/
_output_shapes
:         22
conv2d_120/ReluП
tf.reshape_88/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       2   2
tf.reshape_88/Reshape/shape┤
tf.reshape_88/ReshapeReshapeconv2d_120/Relu:activations:0$tf.reshape_88/Reshape/shape:output:0*
T0*+
_output_shapes
:         22
tf.reshape_88/ReshapeД
max_pooling1d_32/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_32/ExpandDims/dim╠
max_pooling1d_32/ExpandDims
ExpandDimstf.reshape_88/Reshape:output:0(max_pooling1d_32/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         22
max_pooling1d_32/ExpandDims╥
max_pooling1d_32/MaxPoolMaxPool$max_pooling1d_32/ExpandDims:output:0*/
_output_shapes
:         
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_32/MaxPoolп
max_pooling1d_32/SqueezeSqueeze!max_pooling1d_32/MaxPool:output:0*
T0*+
_output_shapes
:         
2*
squeeze_dims
2
max_pooling1d_32/Squeezeu
flatten_72/ConstConst*
_output_shapes
:*
dtype0*
valueB"    Ї  2
flatten_72/Constд
flatten_72/ReshapeReshape!max_pooling1d_32/Squeeze:output:0flatten_72/Const:output:0*
T0*(
_output_shapes
:         Ї2
flatten_72/Reshape░
 dense_1776/MatMul/ReadVariableOpReadVariableOp)dense_1776_matmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02"
 dense_1776/MatMul/ReadVariableOpк
dense_1776/MatMulMatMulflatten_72/Reshape:output:0(dense_1776/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1776/MatMulо
!dense_1776/BiasAdd/ReadVariableOpReadVariableOp*dense_1776_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02#
!dense_1776/BiasAdd/ReadVariableOpо
dense_1776/BiasAddBiasAdddense_1776/MatMul:product:0)dense_1776/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1776/BiasAddz
dense_1776/ReluReludense_1776/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
dense_1776/Relu{
dropout_576/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout_576/dropout/Constп
dropout_576/dropout/MulMuldense_1776/Relu:activations:0"dropout_576/dropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout_576/dropout/MulГ
dropout_576/dropout/ShapeShapedense_1776/Relu:activations:0*
T0*
_output_shapes
:2
dropout_576/dropout/Shape┘
0dropout_576/dropout/random_uniform/RandomUniformRandomUniform"dropout_576/dropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype022
0dropout_576/dropout/random_uniform/RandomUniformН
"dropout_576/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2$
"dropout_576/dropout/GreaterEqual/yя
 dropout_576/dropout/GreaterEqualGreaterEqual9dropout_576/dropout/random_uniform/RandomUniform:output:0+dropout_576/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2"
 dropout_576/dropout/GreaterEqualд
dropout_576/dropout/CastCast$dropout_576/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout_576/dropout/Castл
dropout_576/dropout/Mul_1Muldropout_576/dropout/Mul:z:0dropout_576/dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout_576/dropout/Mul_1г
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
output/MatMul/ReadVariableOpЯ
output/MatMulMatMuldropout_576/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulб
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAdd╕
IdentityIdentityoutput/BiasAdd:output:0"^conv2d_120/BiasAdd/ReadVariableOp!^conv2d_120/Conv2D/ReadVariableOp"^dense_1776/BiasAdd/ReadVariableOp!^dense_1776/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 2F
!conv2d_120/BiasAdd/ReadVariableOp!conv2d_120/BiasAdd/ReadVariableOp2D
 conv2d_120/Conv2D/ReadVariableOp conv2d_120/Conv2D/ReadVariableOp2F
!dense_1776/BiasAdd/ReadVariableOp!dense_1776/BiasAdd/ReadVariableOp2D
 dense_1776/MatMul/ReadVariableOp dense_1776/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ц
П
*__inference_model_64_layer_call_fn_1017348

inputs!
unknown:2
	unknown_0:2
	unknown_1:
ЇЇ
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_64_layer_call_and_return_conditional_losses_10170362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ц
П
*__inference_model_64_layer_call_fn_1017365

inputs!
unknown:2
	unknown_0:2
	unknown_1:
ЇЇ
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_64_layer_call_and_return_conditional_losses_10171532
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╗

√
G__inference_dense_1776_layer_call_and_return_conditional_losses_1017407

inputs2
matmul_readvariableop_resource:
ЇЇ.
biasadd_readvariableop_resource:	Ї
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ш
А
G__inference_conv2d_120_layer_call_and_return_conditional_losses_1016978

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         22
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╙	
ї
C__inference_output_layer_call_and_return_conditional_losses_1017029

inputs1
matmul_readvariableop_resource:	Ї-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
═
I
-__inference_dropout_576_layer_call_fn_1017438

inputs
identity╠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dropout_576_layer_call_and_return_conditional_losses_10170172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
╙	
ї
C__inference_output_layer_call_and_return_conditional_losses_1017453

inputs1
matmul_readvariableop_resource:	Ї-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultз
M
input_onehot=
serving_default_input_onehot:0         :
output0
StatefulPartitionedCall:0         tensorflow/serving/predict:ос
з?
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
regularization_losses
trainable_variables
	keras_api

signatures
*g&call_and_return_all_conditional_losses
h_default_save_signature
i__call__"ж<
_tf_keras_networkК<{"name": "model_64", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_64", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_120", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_120", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_88", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_88", "inbound_nodes": [["conv2d_120", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_32", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_32", "inbound_nodes": [[["tf.reshape_88", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_72", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_72", "inbound_nodes": [[["max_pooling1d_32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1776", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1776", "inbound_nodes": [[["flatten_72", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_576", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_576", "inbound_nodes": [[["dense_1776", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_576", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_64", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_120", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_120", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_88", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_88", "inbound_nodes": [["conv2d_120", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_32", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_32", "inbound_nodes": [[["tf.reshape_88", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_72", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_72", "inbound_nodes": [[["max_pooling1d_32", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_1776", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1776", "inbound_nodes": [[["flatten_72", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_576", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_576", "inbound_nodes": [[["dense_1776", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_576", 0, 0, {}]]], "shared_object_id": 13}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Б"■
_tf_keras_input_layer▐{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
В

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"▌	
_tf_keras_layer├	{"name": "conv2d_120", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_120", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}}
┘
	keras_api"╟
_tf_keras_layerн{"name": "tf.reshape_88", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.reshape_88", "trainable": true, "dtype": "float32", "function": "reshape"}, "inbound_nodes": [["conv2d_120", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}
▄
	variables
regularization_losses
trainable_variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"═
_tf_keras_layer│{"name": "max_pooling1d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_32", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["tf.reshape_88", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 17}}
╠
	variables
regularization_losses
trainable_variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"╜
_tf_keras_layerг{"name": "flatten_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_72", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["max_pooling1d_32", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
Е	

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
*p&call_and_return_all_conditional_losses
q__call__"р
_tf_keras_layer╞{"name": "dense_1776", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1776", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_72", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
░
$	variables
%regularization_losses
&trainable_variables
'	keras_api
*r&call_and_return_all_conditional_losses
s__call__"б
_tf_keras_layerЗ{"name": "dropout_576", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_576", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_1776", 0, 0, {}]]], "shared_object_id": 10}
Б	

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
*t&call_and_return_all_conditional_losses
u__call__"▄
_tf_keras_layer┬{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_576", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
┐
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
╩
3layer_regularization_losses
4layer_metrics

	variables

5layers
regularization_losses
6metrics
trainable_variables
7non_trainable_variables
i__call__
h_default_save_signature
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
+:)22conv2d_120/kernel
:22conv2d_120/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
8layer_regularization_losses
9layer_metrics
	variables

:layers
regularization_losses
;metrics
trainable_variables
<non_trainable_variables
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
н
=layer_regularization_losses
>layer_metrics
	variables

?layers
regularization_losses
@metrics
trainable_variables
Anon_trainable_variables
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
н
Blayer_regularization_losses
Clayer_metrics
	variables

Dlayers
regularization_losses
Emetrics
trainable_variables
Fnon_trainable_variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
%:#
ЇЇ2dense_1776/kernel
:Ї2dense_1776/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
Glayer_regularization_losses
Hlayer_metrics
 	variables

Ilayers
!regularization_losses
Jmetrics
"trainable_variables
Knon_trainable_variables
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
н
Llayer_regularization_losses
Mlayer_metrics
$	variables

Nlayers
%regularization_losses
Ometrics
&trainable_variables
Pnon_trainable_variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 :	Ї2output/kernel
:2output/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
н
Qlayer_regularization_losses
Rlayer_metrics
*	variables

Slayers
+regularization_losses
Tmetrics
,trainable_variables
Unon_trainable_variables
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
╘
	Wtotal
	Xcount
Y	variables
Z	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 21}
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
0:.22Adam/conv2d_120/kernel/m
": 22Adam/conv2d_120/bias/m
*:(
ЇЇ2Adam/dense_1776/kernel/m
#:!Ї2Adam/dense_1776/bias/m
%:#	Ї2Adam/output/kernel/m
:2Adam/output/bias/m
0:.22Adam/conv2d_120/kernel/v
": 22Adam/conv2d_120/bias/v
*:(
ЇЇ2Adam/dense_1776/kernel/v
#:!Ї2Adam/dense_1776/bias/v
%:#	Ї2Adam/output/kernel/v
:2Adam/output/bias/v
т2▀
E__inference_model_64_layer_call_and_return_conditional_losses_1017291
E__inference_model_64_layer_call_and_return_conditional_losses_1017331
E__inference_model_64_layer_call_and_return_conditional_losses_1017209
E__inference_model_64_layer_call_and_return_conditional_losses_1017233└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
"__inference__wrapped_model_1016945├
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+
input_onehot         
Ў2є
*__inference_model_64_layer_call_fn_1017051
*__inference_model_64_layer_call_fn_1017348
*__inference_model_64_layer_call_fn_1017365
*__inference_model_64_layer_call_fn_1017185└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ё2ю
G__inference_conv2d_120_layer_call_and_return_conditional_losses_1017376в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_conv2d_120_layer_call_fn_1017385в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_1016954╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
Н2К
2__inference_max_pooling1d_32_layer_call_fn_1016960╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
ё2ю
G__inference_flatten_72_layer_call_and_return_conditional_losses_1017391в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_flatten_72_layer_call_fn_1017396в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_dense_1776_layer_call_and_return_conditional_losses_1017407в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_dense_1776_layer_call_fn_1017416в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017421
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017433┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ш2Х
-__inference_dropout_576_layer_call_fn_1017438
-__inference_dropout_576_layer_call_fn_1017443┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_output_layer_call_and_return_conditional_losses_1017453в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_output_layer_call_fn_1017462в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤B╬
%__inference_signature_wrapper_1017258input_onehot"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Ю
"__inference__wrapped_model_1016945x()=в:
3в0
.К+
input_onehot         
к "/к,
*
output К
output         ╖
G__inference_conv2d_120_layer_call_and_return_conditional_losses_1017376l7в4
-в*
(К%
inputs         
к "-в*
#К 
0         2
Ъ П
,__inference_conv2d_120_layer_call_fn_1017385_7в4
-в*
(К%
inputs         
к " К         2й
G__inference_dense_1776_layer_call_and_return_conditional_losses_1017407^0в-
&в#
!К
inputs         Ї
к "&в#
К
0         Ї
Ъ Б
,__inference_dense_1776_layer_call_fn_1017416Q0в-
&в#
!К
inputs         Ї
к "К         Їк
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017421^4в1
*в'
!К
inputs         Ї
p 
к "&в#
К
0         Ї
Ъ к
H__inference_dropout_576_layer_call_and_return_conditional_losses_1017433^4в1
*в'
!К
inputs         Ї
p
к "&в#
К
0         Ї
Ъ В
-__inference_dropout_576_layer_call_fn_1017438Q4в1
*в'
!К
inputs         Ї
p 
к "К         ЇВ
-__inference_dropout_576_layer_call_fn_1017443Q4в1
*в'
!К
inputs         Ї
p
к "К         Їи
G__inference_flatten_72_layer_call_and_return_conditional_losses_1017391]3в0
)в&
$К!
inputs         
2
к "&в#
К
0         Ї
Ъ А
,__inference_flatten_72_layer_call_fn_1017396P3в0
)в&
$К!
inputs         
2
к "К         Ї╓
M__inference_max_pooling1d_32_layer_call_and_return_conditional_losses_1016954ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ н
2__inference_max_pooling1d_32_layer_call_fn_1016960wEвB
;в8
6К3
inputs'                           
к ".К+'                           ┐
E__inference_model_64_layer_call_and_return_conditional_losses_1017209v()EвB
;в8
.К+
input_onehot         
p 

 
к "%в"
К
0         
Ъ ┐
E__inference_model_64_layer_call_and_return_conditional_losses_1017233v()EвB
;в8
.К+
input_onehot         
p

 
к "%в"
К
0         
Ъ ╣
E__inference_model_64_layer_call_and_return_conditional_losses_1017291p()?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         
Ъ ╣
E__inference_model_64_layer_call_and_return_conditional_losses_1017331p()?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         
Ъ Ч
*__inference_model_64_layer_call_fn_1017051i()EвB
;в8
.К+
input_onehot         
p 

 
к "К         Ч
*__inference_model_64_layer_call_fn_1017185i()EвB
;в8
.К+
input_onehot         
p

 
к "К         С
*__inference_model_64_layer_call_fn_1017348c()?в<
5в2
(К%
inputs         
p 

 
к "К         С
*__inference_model_64_layer_call_fn_1017365c()?в<
5в2
(К%
inputs         
p

 
к "К         д
C__inference_output_layer_call_and_return_conditional_losses_1017453]()0в-
&в#
!К
inputs         Ї
к "%в"
К
0         
Ъ |
(__inference_output_layer_call_fn_1017462P()0в-
&в#
!К
inputs         Ї
к "К         ▓
%__inference_signature_wrapper_1017258И()MвJ
в 
Cк@
>
input_onehot.К+
input_onehot         "/к,
*
output К
output         