ы№
№ђ
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
Њ
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718дж
В
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:2*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:2*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
фф*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
фф*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ф*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:ф*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ф*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	ф*
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
Р
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/conv2d_4/kernel/m
Й
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:2*
dtype0
А
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:2*
dtype0
И
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
фф*&
shared_nameAdam/dense_4/kernel/m
Б
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
фф*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ф*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:ф*
dtype0
Е
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ф*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	ф*
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
Р
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/conv2d_4/kernel/v
Й
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:2*
dtype0
А
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:2*
dtype0
И
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
фф*&
shared_nameAdam/dense_4/kernel/v
Б
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
фф*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ф*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:ф*
dtype0
Е
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ф*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	ф*
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
©*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*д)
valueЏ)B„) B–)
І
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
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
ђ
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
≠
3metrics
4layer_regularization_losses

5layers
6layer_metrics
7non_trainable_variables

	variables
trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
8layer_regularization_losses
	variables
9layer_metrics

:layers
;non_trainable_variables
<metrics
trainable_variables
regularization_losses
 
 
 
 
≠
=layer_regularization_losses
	variables
>layer_metrics

?layers
@non_trainable_variables
Ametrics
trainable_variables
regularization_losses
 
 
 
≠
Blayer_regularization_losses
	variables
Clayer_metrics

Dlayers
Enon_trainable_variables
Fmetrics
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
Glayer_regularization_losses
 	variables
Hlayer_metrics

Ilayers
Jnon_trainable_variables
Kmetrics
!trainable_variables
"regularization_losses
 
 
 
≠
Llayer_regularization_losses
$	variables
Mlayer_metrics

Nlayers
Onon_trainable_variables
Pmetrics
%trainable_variables
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
≠
Qlayer_regularization_losses
*	variables
Rlayer_metrics

Slayers
Tnon_trainable_variables
Umetrics
+trainable_variables
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

V0
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
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
П
serving_default_input_onehotPlaceholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
Ґ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_4/kernelconv2d_4/biasdense_4/kerneldense_4/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *-
f(R&
$__inference_signature_wrapper_937578
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
в	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
GPU2 *0J 8В *(
f#R!
__inference__traced_save_937880
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasdense_4/kerneldense_4/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
GPU2 *0J 8В *+
f&R$
"__inference__traced_restore_937965рш
Ќ
F
*__inference_flatten_4_layer_call_fn_937716

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9373132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
2:S O
+
_output_shapes
:€€€€€€€€€
2
 
_user_specified_nameinputs
ф
У
(__inference_model_4_layer_call_fn_937371
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
фф
	unknown_2:	ф
	unknown_3:	ф
	unknown_4:
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9373562
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
ф
У
(__inference_model_4_layer_call_fn_937505
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
фф
	unknown_2:	ф
	unknown_3:	ф
	unknown_4:
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9374732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
ї
Є
C__inference_model_4_layer_call_and_return_conditional_losses_937473

inputs)
conv2d_4_937452:2
conv2d_4_937454:2"
dense_4_937461:
фф
dense_4_937463:	ф 
output_937467:	ф
output_937469:
identityИҐ conv2d_4/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCallҐoutput/StatefulPartitionedCall°
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_937452conv2d_4_937454*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_9372982"
 conv2d_4/StatefulPartitionedCallН
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   2   2
tf.reshape_4/Reshape/shapeљ
tf.reshape_4/ReshapeReshape)conv2d_4/StatefulPartitionedCall:output:0#tf.reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€22
tf.reshape_4/ReshapeЙ
max_pooling1d_4/PartitionedCallPartitionedCalltf.reshape_4/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_9372742!
max_pooling1d_4/PartitionedCall€
flatten_4/PartitionedCallPartitionedCall(max_pooling1d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9373132
flatten_4/PartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_937461dense_4_937463*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_9373262!
dense_4/StatefulPartitionedCallЧ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_9374012#
!dropout_4/StatefulPartitionedCall≥
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0output_937467output_937469*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9373492 
output/StatefulPartitionedCallЕ
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
я
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_937711

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ф  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
2:S O
+
_output_shapes
:€€€€€€€€€
2
 
_user_specified_nameinputs
цl
К
"__inference__traced_restore_937965
file_prefix:
 assignvariableop_conv2d_4_kernel:2.
 assignvariableop_1_conv2d_4_bias:25
!assignvariableop_2_dense_4_kernel:
фф.
assignvariableop_3_dense_4_bias:	ф3
 assignvariableop_4_output_kernel:	ф,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: D
*assignvariableop_13_adam_conv2d_4_kernel_m:26
(assignvariableop_14_adam_conv2d_4_bias_m:2=
)assignvariableop_15_adam_dense_4_kernel_m:
фф6
'assignvariableop_16_adam_dense_4_bias_m:	ф;
(assignvariableop_17_adam_output_kernel_m:	ф4
&assignvariableop_18_adam_output_bias_m:D
*assignvariableop_19_adam_conv2d_4_kernel_v:26
(assignvariableop_20_adam_conv2d_4_bias_v:2=
)assignvariableop_21_adam_dense_4_kernel_v:
фф6
'assignvariableop_22_adam_dense_4_bias_v:	ф;
(assignvariableop_23_adam_output_kernel_v:	ф4
&assignvariableop_24_adam_output_bias_v:
identity_26ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9¶
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*≤
value®B•B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices≠
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

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4•
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6°
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ґ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ѓ
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11°
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13≤
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_conv2d_4_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14∞
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_conv2d_4_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15±
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_4_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ѓ
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_4_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17∞
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ѓ
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19≤
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_4_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20∞
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_4_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21±
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_4_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ѓ
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_4_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23∞
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_output_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ѓ
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
Identity_25ч
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
•
Ш
(__inference_dense_4_layer_call_fn_937736

inputs
unknown:
фф
	unknown_0:	ф
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_9373262
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Э
Ъ
C__inference_model_4_layer_call_and_return_conditional_losses_937529
input_onehot)
conv2d_4_937508:2
conv2d_4_937510:2"
dense_4_937517:
фф
dense_4_937519:	ф 
output_937523:	ф
output_937525:
identityИҐ conv2d_4/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐoutput/StatefulPartitionedCallІ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_4_937508conv2d_4_937510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_9372982"
 conv2d_4/StatefulPartitionedCallН
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   2   2
tf.reshape_4/Reshape/shapeљ
tf.reshape_4/ReshapeReshape)conv2d_4/StatefulPartitionedCall:output:0#tf.reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€22
tf.reshape_4/ReshapeЙ
max_pooling1d_4/PartitionedCallPartitionedCalltf.reshape_4/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_9372742!
max_pooling1d_4/PartitionedCall€
flatten_4/PartitionedCallPartitionedCall(max_pooling1d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9373132
flatten_4/PartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_937517dense_4_937519*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_9373262!
dense_4/StatefulPartitionedCall€
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_9373372
dropout_4/PartitionedCallЂ
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0output_937523output_937525*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9373492 
output/StatefulPartitionedCallб
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
ѓ)
Д
C__inference_model_4_layer_call_and_return_conditional_losses_937611

inputsA
'conv2d_4_conv2d_readvariableop_resource:26
(conv2d_4_biasadd_readvariableop_resource:2:
&dense_4_matmul_readvariableop_resource:
фф6
'dense_4_biasadd_readvariableop_resource:	ф8
%output_matmul_readvariableop_resource:	ф4
&output_biasadd_readvariableop_resource:
identityИҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOp∞
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02 
conv2d_4/Conv2D/ReadVariableOpњ
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2*
paddingVALID*
strides
2
conv2d_4/Conv2DІ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpђ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22
conv2d_4/ReluН
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   2   2
tf.reshape_4/Reshape/shapeѓ
tf.reshape_4/ReshapeReshapeconv2d_4/Relu:activations:0#tf.reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€22
tf.reshape_4/ReshapeВ
max_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_4/ExpandDims/dim»
max_pooling1d_4/ExpandDims
ExpandDimstf.reshape_4/Reshape:output:0'max_pooling1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€22
max_pooling1d_4/ExpandDimsѕ
max_pooling1d_4/MaxPoolMaxPool#max_pooling1d_4/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_4/MaxPoolђ
max_pooling1d_4/SqueezeSqueeze max_pooling1d_4/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€
2*
squeeze_dims
2
max_pooling1d_4/Squeezes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ф  2
flatten_4/Const†
flatten_4/ReshapeReshape max_pooling1d_4/Squeeze:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
flatten_4/ReshapeІ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
фф*
dtype02
dense_4/MatMul/ReadVariableOp†
dense_4/MatMulMatMulflatten_4/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_4/MatMul•
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02 
dense_4/BiasAdd/ReadVariableOpҐ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_4/ReluГ
dropout_4/IdentityIdentitydense_4/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/Identity£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02
output/MatMul/ReadVariableOpЭ
output/MatMulMatMuldropout_4/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/BiasAddЃ
IdentityIdentityoutput/BiasAdd:output:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
к/
»
!__inference__wrapped_model_937265
input_onehotI
/model_4_conv2d_4_conv2d_readvariableop_resource:2>
0model_4_conv2d_4_biasadd_readvariableop_resource:2B
.model_4_dense_4_matmul_readvariableop_resource:
фф>
/model_4_dense_4_biasadd_readvariableop_resource:	ф@
-model_4_output_matmul_readvariableop_resource:	ф<
.model_4_output_biasadd_readvariableop_resource:
identityИҐ'model_4/conv2d_4/BiasAdd/ReadVariableOpҐ&model_4/conv2d_4/Conv2D/ReadVariableOpҐ&model_4/dense_4/BiasAdd/ReadVariableOpҐ%model_4/dense_4/MatMul/ReadVariableOpҐ%model_4/output/BiasAdd/ReadVariableOpҐ$model_4/output/MatMul/ReadVariableOp»
&model_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/model_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02(
&model_4/conv2d_4/Conv2D/ReadVariableOpЁ
model_4/conv2d_4/Conv2DConv2Dinput_onehot.model_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2*
paddingVALID*
strides
2
model_4/conv2d_4/Conv2Dњ
'model_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0model_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'model_4/conv2d_4/BiasAdd/ReadVariableOpћ
model_4/conv2d_4/BiasAddBiasAdd model_4/conv2d_4/Conv2D:output:0/model_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22
model_4/conv2d_4/BiasAddУ
model_4/conv2d_4/ReluRelu!model_4/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22
model_4/conv2d_4/ReluЭ
"model_4/tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   2   2$
"model_4/tf.reshape_4/Reshape/shapeѕ
model_4/tf.reshape_4/ReshapeReshape#model_4/conv2d_4/Relu:activations:0+model_4/tf.reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€22
model_4/tf.reshape_4/ReshapeТ
&model_4/max_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_4/max_pooling1d_4/ExpandDims/dimи
"model_4/max_pooling1d_4/ExpandDims
ExpandDims%model_4/tf.reshape_4/Reshape:output:0/model_4/max_pooling1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€22$
"model_4/max_pooling1d_4/ExpandDimsз
model_4/max_pooling1d_4/MaxPoolMaxPool+model_4/max_pooling1d_4/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€
2*
ksize
*
paddingVALID*
strides
2!
model_4/max_pooling1d_4/MaxPoolƒ
model_4/max_pooling1d_4/SqueezeSqueeze(model_4/max_pooling1d_4/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€
2*
squeeze_dims
2!
model_4/max_pooling1d_4/SqueezeГ
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ф  2
model_4/flatten_4/Constј
model_4/flatten_4/ReshapeReshape(model_4/max_pooling1d_4/Squeeze:output:0 model_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
model_4/flatten_4/Reshapeњ
%model_4/dense_4/MatMul/ReadVariableOpReadVariableOp.model_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
фф*
dtype02'
%model_4/dense_4/MatMul/ReadVariableOpј
model_4/dense_4/MatMulMatMul"model_4/flatten_4/Reshape:output:0-model_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
model_4/dense_4/MatMulљ
&model_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02(
&model_4/dense_4/BiasAdd/ReadVariableOp¬
model_4/dense_4/BiasAddBiasAdd model_4/dense_4/MatMul:product:0.model_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
model_4/dense_4/BiasAddЙ
model_4/dense_4/ReluRelu model_4/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
model_4/dense_4/ReluЫ
model_4/dropout_4/IdentityIdentity"model_4/dense_4/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
model_4/dropout_4/Identityї
$model_4/output/MatMul/ReadVariableOpReadVariableOp-model_4_output_matmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02&
$model_4/output/MatMul/ReadVariableOpљ
model_4/output/MatMulMatMul#model_4/dropout_4/Identity:output:0,model_4/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_4/output/MatMulє
%model_4/output/BiasAdd/ReadVariableOpReadVariableOp.model_4_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_4/output/BiasAdd/ReadVariableOpљ
model_4/output/BiasAddBiasAddmodel_4/output/MatMul:product:0-model_4/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_4/output/BiasAddж
IdentityIdentitymodel_4/output/BiasAdd:output:0(^model_4/conv2d_4/BiasAdd/ReadVariableOp'^model_4/conv2d_4/Conv2D/ReadVariableOp'^model_4/dense_4/BiasAdd/ReadVariableOp&^model_4/dense_4/MatMul/ReadVariableOp&^model_4/output/BiasAdd/ReadVariableOp%^model_4/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2R
'model_4/conv2d_4/BiasAdd/ReadVariableOp'model_4/conv2d_4/BiasAdd/ReadVariableOp2P
&model_4/conv2d_4/Conv2D/ReadVariableOp&model_4/conv2d_4/Conv2D/ReadVariableOp2P
&model_4/dense_4/BiasAdd/ReadVariableOp&model_4/dense_4/BiasAdd/ReadVariableOp2N
%model_4/dense_4/MatMul/ReadVariableOp%model_4/dense_4/MatMul/ReadVariableOp2N
%model_4/output/BiasAdd/ReadVariableOp%model_4/output/BiasAdd/ReadVariableOp2L
$model_4/output/MatMul/ReadVariableOp$model_4/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
Ќ
Њ
C__inference_model_4_layer_call_and_return_conditional_losses_937553
input_onehot)
conv2d_4_937532:2
conv2d_4_937534:2"
dense_4_937541:
фф
dense_4_937543:	ф 
output_937547:	ф
output_937549:
identityИҐ conv2d_4/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCallҐoutput/StatefulPartitionedCallІ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_4_937532conv2d_4_937534*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_9372982"
 conv2d_4/StatefulPartitionedCallН
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   2   2
tf.reshape_4/Reshape/shapeљ
tf.reshape_4/ReshapeReshape)conv2d_4/StatefulPartitionedCall:output:0#tf.reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€22
tf.reshape_4/ReshapeЙ
max_pooling1d_4/PartitionedCallPartitionedCalltf.reshape_4/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_9372742!
max_pooling1d_4/PartitionedCall€
flatten_4/PartitionedCallPartitionedCall(max_pooling1d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9373132
flatten_4/PartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_937541dense_4_937543*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_9373262!
dense_4/StatefulPartitionedCallЧ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_9374012#
!dropout_4/StatefulPartitionedCall≥
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0output_937547output_937549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9373492 
output/StatefulPartitionedCallЕ
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
“	
ф
B__inference_output_layer_call_and_return_conditional_losses_937773

inputs1
matmul_readvariableop_resource:	ф-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ц
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_937337

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
®
L
0__inference_max_pooling1d_4_layer_call_fn_937280

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_9372742
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Х
э
D__inference_conv2d_4_layer_call_and_return_conditional_losses_937298

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2*
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
:€€€€€€€€€22	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
“	
ф
B__inference_output_layer_call_and_return_conditional_losses_937349

inputs1
matmul_readvariableop_resource:	ф-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
і
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_937753

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ф2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
»
Ю
)__inference_conv2d_4_layer_call_fn_937705

inputs!
unknown:2
	unknown_0:2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_9372982
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
і
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_937401

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ф2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
я
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_937313

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ф  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
2:S O
+
_output_shapes
:€€€€€€€€€
2
 
_user_specified_nameinputs
Я
Х
'__inference_output_layer_call_fn_937782

inputs
unknown:	ф
	unknown_0:
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9373492
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ј

ч
C__inference_dense_4_layer_call_and_return_conditional_losses_937326

inputs2
matmul_readvariableop_resource:
фф.
biasadd_readvariableop_resource:	ф
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
фф*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ц
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_937741

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ј

ч
C__inference_dense_4_layer_call_and_return_conditional_losses_937727

inputs2
matmul_readvariableop_resource:
фф.
biasadd_readvariableop_resource:	ф
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
фф*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ќ
П
$__inference_signature_wrapper_937578
input_onehot!
unknown:2
	unknown_0:2
	unknown_1:
фф
	unknown_2:	ф
	unknown_3:	ф
	unknown_4:
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В **
f%R#
!__inference__wrapped_model_9372652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
Л
Ф
C__inference_model_4_layer_call_and_return_conditional_losses_937356

inputs)
conv2d_4_937299:2
conv2d_4_937301:2"
dense_4_937327:
фф
dense_4_937329:	ф 
output_937350:	ф
output_937352:
identityИҐ conv2d_4/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐoutput/StatefulPartitionedCall°
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_937299conv2d_4_937301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_9372982"
 conv2d_4/StatefulPartitionedCallН
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   2   2
tf.reshape_4/Reshape/shapeљ
tf.reshape_4/ReshapeReshape)conv2d_4/StatefulPartitionedCall:output:0#tf.reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€22
tf.reshape_4/ReshapeЙ
max_pooling1d_4/PartitionedCallPartitionedCalltf.reshape_4/Reshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_9372742!
max_pooling1d_4/PartitionedCall€
flatten_4/PartitionedCallPartitionedCall(max_pooling1d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_9373132
flatten_4/PartitionedCall±
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_4_937327dense_4_937329*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_9373262!
dense_4/StatefulPartitionedCall€
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_9373372
dropout_4/PartitionedCallЂ
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0output_937350output_937352*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9373492 
output/StatefulPartitionedCallб
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Х
э
D__inference_conv2d_4_layer_call_and_return_conditional_losses_937696

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2*
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
:€€€€€€€€€22	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
№2
Д
C__inference_model_4_layer_call_and_return_conditional_losses_937651

inputsA
'conv2d_4_conv2d_readvariableop_resource:26
(conv2d_4_biasadd_readvariableop_resource:2:
&dense_4_matmul_readvariableop_resource:
фф6
'dense_4_biasadd_readvariableop_resource:	ф8
%output_matmul_readvariableop_resource:	ф4
&output_biasadd_readvariableop_resource:
identityИҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOp∞
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02 
conv2d_4/Conv2D/ReadVariableOpњ
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2*
paddingVALID*
strides
2
conv2d_4/Conv2DІ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpђ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22
conv2d_4/ReluН
tf.reshape_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€   2   2
tf.reshape_4/Reshape/shapeѓ
tf.reshape_4/ReshapeReshapeconv2d_4/Relu:activations:0#tf.reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€22
tf.reshape_4/ReshapeВ
max_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_4/ExpandDims/dim»
max_pooling1d_4/ExpandDims
ExpandDimstf.reshape_4/Reshape:output:0'max_pooling1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€22
max_pooling1d_4/ExpandDimsѕ
max_pooling1d_4/MaxPoolMaxPool#max_pooling1d_4/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€
2*
ksize
*
paddingVALID*
strides
2
max_pooling1d_4/MaxPoolђ
max_pooling1d_4/SqueezeSqueeze max_pooling1d_4/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€
2*
squeeze_dims
2
max_pooling1d_4/Squeezes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ф  2
flatten_4/Const†
flatten_4/ReshapeReshape max_pooling1d_4/Squeeze:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
flatten_4/ReshapeІ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
фф*
dtype02
dense_4/MatMul/ReadVariableOp†
dense_4/MatMulMatMulflatten_4/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_4/MatMul•
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02 
dense_4/BiasAdd/ReadVariableOpҐ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_4/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_4/dropout/Const¶
dropout_4/dropout/MulMuldense_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape”
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф*
dtype020
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2"
 dropout_4/dropout/GreaterEqual/yз
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2 
dropout_4/dropout/GreaterEqualЮ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/dropout/Cast£
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/dropout/Mul_1£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02
output/MatMul/ReadVariableOpЭ
output/MatMulMatMuldropout_4/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/BiasAddЃ
IdentityIdentityoutput/BiasAdd:output:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Т
g
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_937274

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ъ:
®

__inference__traced_save_937880
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename†
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*≤
value®B•B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЉ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЂ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*џ
_input_shapes…
∆: :2:2:
фф:ф:	ф:: : : : : : : :2:2:
фф:ф:	ф::2:2:
фф:ф:	ф:: 2(
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
фф:!

_output_shapes	
:ф:%!

_output_shapes
:	ф: 
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
фф:!

_output_shapes	
:ф:%!

_output_shapes
:	ф: 
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
фф:!

_output_shapes	
:ф:%!

_output_shapes
:	ф: 

_output_shapes
::

_output_shapes
: 
в
Н
(__inference_model_4_layer_call_fn_937668

inputs!
unknown:2
	unknown_0:2
	unknown_1:
фф
	unknown_2:	ф
	unknown_3:	ф
	unknown_4:
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9373562
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
”
c
*__inference_dropout_4_layer_call_fn_937763

inputs
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_9374012
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ф22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
«
F
*__inference_dropout_4_layer_call_fn_937758

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_9373372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
в
Н
(__inference_model_4_layer_call_fn_937685

inputs!
unknown:2
	unknown_0:2
	unknown_1:
фф
	unknown_2:	ф
	unknown_3:	ф
	unknown_4:
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_9374732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ї
serving_defaultІ
M
input_onehot=
serving_default_input_onehot:0€€€€€€€€€:
output0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:√я
и>
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
g_default_save_signature
*h&call_and_return_all_conditional_losses
i__call__"з;
_tf_keras_networkЋ;{"name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_4", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_4", "inbound_nodes": [["conv2d_4", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_4", "inbound_nodes": [[["tf.reshape_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["max_pooling1d_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 23, 4, 1]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "TFOpLambda", "config": {"name": "tf.reshape_4", "trainable": true, "dtype": "float32", "function": "reshape"}, "name": "tf.reshape_4", "inbound_nodes": [["conv2d_4", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_4", "inbound_nodes": [[["tf.reshape_4", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["max_pooling1d_4", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_4", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 13}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Б"ю
_tf_keras_input_layerё{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
ю


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"ў	
_tf_keras_layerњ	{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 50, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4, 1]}}
’
	keras_api"√
_tf_keras_layer©{"name": "tf.reshape_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.reshape_4", "trainable": true, "dtype": "float32", "function": "reshape"}, "inbound_nodes": [["conv2d_4", 0, 0, {"shape": {"class_name": "__tuple__", "items": [-1, 20, 50]}}]], "shared_object_id": 4}
ў
	variables
trainable_variables
regularization_losses
	keras_api
*l&call_and_return_all_conditional_losses
m__call__" 
_tf_keras_layer∞{"name": "max_pooling1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["tf.reshape_4", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 17}}
…
	variables
trainable_variables
regularization_losses
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"Ї
_tf_keras_layer†{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["max_pooling1d_4", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
ю

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
*p&call_and_return_all_conditional_losses
q__call__"ў
_tf_keras_layerњ{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_4", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
©
$	variables
%trainable_variables
&regularization_losses
'	keras_api
*r&call_and_return_all_conditional_losses
s__call__"Ъ
_tf_keras_layerА{"name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_4", 0, 0, {}]]], "shared_object_id": 10}
€

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
*t&call_and_return_all_conditional_losses
u__call__"Џ
_tf_keras_layerј{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
њ
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
 
3metrics
4layer_regularization_losses

5layers
6layer_metrics
7non_trainable_variables

	variables
trainable_variables
regularization_losses
i__call__
g_default_save_signature
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
):'22conv2d_4/kernel
:22conv2d_4/bias
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
≠
8layer_regularization_losses
	variables
9layer_metrics

:layers
;non_trainable_variables
<metrics
trainable_variables
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
≠
=layer_regularization_losses
	variables
>layer_metrics

?layers
@non_trainable_variables
Ametrics
trainable_variables
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
≠
Blayer_regularization_losses
	variables
Clayer_metrics

Dlayers
Enon_trainable_variables
Fmetrics
trainable_variables
regularization_losses
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
": 
фф2dense_4/kernel
:ф2dense_4/bias
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
≠
Glayer_regularization_losses
 	variables
Hlayer_metrics

Ilayers
Jnon_trainable_variables
Kmetrics
!trainable_variables
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
≠
Llayer_regularization_losses
$	variables
Mlayer_metrics

Nlayers
Onon_trainable_variables
Pmetrics
%trainable_variables
&regularization_losses
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 :	ф2output/kernel
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
≠
Qlayer_regularization_losses
*	variables
Rlayer_metrics

Slayers
Tnon_trainable_variables
Umetrics
+trainable_variables
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
'
V0"
trackable_list_wrapper
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
trackable_dict_wrapper
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
‘
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
.:,22Adam/conv2d_4/kernel/m
 :22Adam/conv2d_4/bias/m
':%
фф2Adam/dense_4/kernel/m
 :ф2Adam/dense_4/bias/m
%:#	ф2Adam/output/kernel/m
:2Adam/output/bias/m
.:,22Adam/conv2d_4/kernel/v
 :22Adam/conv2d_4/bias/v
':%
фф2Adam/dense_4/kernel/v
 :ф2Adam/dense_4/bias/v
%:#	ф2Adam/output/kernel/v
:2Adam/output/bias/v
м2й
!__inference__wrapped_model_937265√
Л≤З
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
annotations™ *3Ґ0
.К+
input_onehot€€€€€€€€€
Џ2„
C__inference_model_4_layer_call_and_return_conditional_losses_937611
C__inference_model_4_layer_call_and_return_conditional_losses_937651
C__inference_model_4_layer_call_and_return_conditional_losses_937529
C__inference_model_4_layer_call_and_return_conditional_losses_937553ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
о2л
(__inference_model_4_layer_call_fn_937371
(__inference_model_4_layer_call_fn_937668
(__inference_model_4_layer_call_fn_937685
(__inference_model_4_layer_call_fn_937505ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
о2л
D__inference_conv2d_4_layer_call_and_return_conditional_losses_937696Ґ
Щ≤Х
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
annotations™ *
 
”2–
)__inference_conv2d_4_layer_call_fn_937705Ґ
Щ≤Х
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
annotations™ *
 
¶2£
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_937274”
Щ≤Х
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
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Л2И
0__inference_max_pooling1d_4_layer_call_fn_937280”
Щ≤Х
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
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
п2м
E__inference_flatten_4_layer_call_and_return_conditional_losses_937711Ґ
Щ≤Х
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
annotations™ *
 
‘2—
*__inference_flatten_4_layer_call_fn_937716Ґ
Щ≤Х
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
annotations™ *
 
н2к
C__inference_dense_4_layer_call_and_return_conditional_losses_937727Ґ
Щ≤Х
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
annotations™ *
 
“2ѕ
(__inference_dense_4_layer_call_fn_937736Ґ
Щ≤Х
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
annotations™ *
 
»2≈
E__inference_dropout_4_layer_call_and_return_conditional_losses_937741
E__inference_dropout_4_layer_call_and_return_conditional_losses_937753і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_4_layer_call_fn_937758
*__inference_dropout_4_layer_call_fn_937763і
Ђ≤І
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
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_output_layer_call_and_return_conditional_losses_937773Ґ
Щ≤Х
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
annotations™ *
 
—2ќ
'__inference_output_layer_call_fn_937782Ґ
Щ≤Х
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
annotations™ *
 
–BЌ
$__inference_signature_wrapper_937578input_onehot"Ф
Н≤Й
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
annotations™ *
 Э
!__inference__wrapped_model_937265x()=Ґ:
3Ґ0
.К+
input_onehot€€€€€€€€€
™ "/™,
*
output К
output€€€€€€€€€і
D__inference_conv2d_4_layer_call_and_return_conditional_losses_937696l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€2
Ъ М
)__inference_conv2d_4_layer_call_fn_937705_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€2•
C__inference_dense_4_layer_call_and_return_conditional_losses_937727^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ }
(__inference_dense_4_layer_call_fn_937736Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "К€€€€€€€€€фІ
E__inference_dropout_4_layer_call_and_return_conditional_losses_937741^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p 
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ І
E__inference_dropout_4_layer_call_and_return_conditional_losses_937753^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ 
*__inference_dropout_4_layer_call_fn_937758Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p 
™ "К€€€€€€€€€ф
*__inference_dropout_4_layer_call_fn_937763Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p
™ "К€€€€€€€€€ф¶
E__inference_flatten_4_layer_call_and_return_conditional_losses_937711]3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
2
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ ~
*__inference_flatten_4_layer_call_fn_937716P3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
2
™ "К€€€€€€€€€ф‘
K__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_937274ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ђ
0__inference_max_pooling1d_4_layer_call_fn_937280wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€љ
C__inference_model_4_layer_call_and_return_conditional_losses_937529v()EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ љ
C__inference_model_4_layer_call_and_return_conditional_losses_937553v()EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ј
C__inference_model_4_layer_call_and_return_conditional_losses_937611p()?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ј
C__inference_model_4_layer_call_and_return_conditional_losses_937651p()?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Х
(__inference_model_4_layer_call_fn_937371i()EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p 

 
™ "К€€€€€€€€€Х
(__inference_model_4_layer_call_fn_937505i()EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p

 
™ "К€€€€€€€€€П
(__inference_model_4_layer_call_fn_937668c()?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€П
(__inference_model_4_layer_call_fn_937685c()?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€£
B__inference_output_layer_call_and_return_conditional_losses_937773]()0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
'__inference_output_layer_call_fn_937782P()0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "К€€€€€€€€€±
$__inference_signature_wrapper_937578И()MҐJ
Ґ 
C™@
>
input_onehot.К+
input_onehot€€€€€€€€€"/™,
*
output К
output€€€€€€€€€