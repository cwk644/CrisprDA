Еі
Ўў
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
О
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718т

conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*!
shared_nameconv2d_41/kernel
~
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*'
_output_shapes
:Д*
dtype0
u
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*
shared_nameconv2d_41/bias
n
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes	
:Д*
dtype0

dense_6905/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*"
shared_namedense_6905/kernel
x
%dense_6905/kernel/Read/ReadVariableOpReadVariableOpdense_6905/kernel*
_output_shapes
:	ФP*
dtype0
v
dense_6905/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P* 
shared_namedense_6905/bias
o
#dense_6905/bias/Read/ReadVariableOpReadVariableOpdense_6905/bias*
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

Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*(
shared_nameAdam/conv2d_41/kernel/m

+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*'
_output_shapes
:Д*
dtype0

Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*&
shared_nameAdam/conv2d_41/bias/m
|
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes	
:Д*
dtype0

Adam/dense_6905/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*)
shared_nameAdam/dense_6905/kernel/m

,Adam/dense_6905/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6905/kernel/m*
_output_shapes
:	ФP*
dtype0

Adam/dense_6905/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*'
shared_nameAdam/dense_6905/bias/m
}
*Adam/dense_6905/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6905/bias/m*
_output_shapes
:P*
dtype0

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

Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*(
shared_nameAdam/conv2d_41/kernel/v

+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*'
_output_shapes
:Д*
dtype0

Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*&
shared_nameAdam/conv2d_41/bias/v
|
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes	
:Д*
dtype0

Adam/dense_6905/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*)
shared_nameAdam/dense_6905/kernel/v

,Adam/dense_6905/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6905/kernel/v*
_output_shapes
:	ФP*
dtype0

Adam/dense_6905/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*'
shared_nameAdam/dense_6905/bias/v
}
*Adam/dense_6905/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6905/bias/v*
_output_shapes
:P*
dtype0

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
ь.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ї.
value.B. B.
Д
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
Ќ
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
­
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
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
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
­
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
­
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
­
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
 	variables
][
VARIABLE_VALUEdense_6905/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_6905/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
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
­
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
­
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
­
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
VARIABLE_VALUEAdam/conv2d_41/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_6905/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6905/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_6905/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_6905/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_onehotPlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
Ћ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_41/kernelconv2d_41/biasdense_6905/kerneldense_6905/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference_signature_wrapper_7048549
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ћ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp%dense_6905/kernel/Read/ReadVariableOp#dense_6905/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp,Adam/dense_6905/kernel/m/Read/ReadVariableOp*Adam/dense_6905/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp,Adam/dense_6905/kernel/v/Read/ReadVariableOp*Adam/dense_6905/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
GPU2 *0J 8 *)
f$R"
 __inference__traced_save_7048920

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_41/kernelconv2d_41/biasdense_6905/kerneldense_6905/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/mAdam/dense_6905/kernel/mAdam/dense_6905/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/vAdam/dense_6905/kernel/vAdam/dense_6905/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_7049005аю
Џ
h
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048766

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
ч

+__inference_model_140_layer_call_fn_7048645

inputs"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_model_140_layer_call_and_return_conditional_losses_70482752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
g
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048781

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџP2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
 "
Ј
F__inference_model_140_layer_call_and_return_conditional_losses_7048275

inputs,
conv2d_41_7048200:Д 
conv2d_41_7048202:	Д%
dense_6905_7048239:	ФP 
dense_6905_7048241:P 
output_7048269:P
output_7048271:
identityЂ!conv2d_41/StatefulPartitionedCallЂ"dense_6905/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЊ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_41_7048200conv2d_41_7048202*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_70481992#
!conv2d_41/StatefulPartitionedCall
re_lu_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_41_layer_call_and_return_conditional_losses_70482102
re_lu_41/PartitionedCall
dropout_2301/PartitionedCallPartitionedCall!re_lu_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2301_layer_call_and_return_conditional_losses_70482172
dropout_2301/PartitionedCall
flatten_140/PartitionedCallPartitionedCall%dropout_2301/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџФ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_140_layer_call_and_return_conditional_losses_70482252
flatten_140/PartitionedCallФ
"dense_6905/StatefulPartitionedCallStatefulPartitionedCall$flatten_140/PartitionedCall:output:0dense_6905_7048239dense_6905_7048241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_6905_layer_call_and_return_conditional_losses_70482382$
"dense_6905/StatefulPartitionedCall
dropout_2302/PartitionedCallPartitionedCall+dense_6905/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2302_layer_call_and_return_conditional_losses_70482492
dropout_2302/PartitionedCall
dropout_2303/PartitionedCallPartitionedCall%dropout_2302/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2303_layer_call_and_return_conditional_losses_70482562
dropout_2303/PartitionedCallБ
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_2303/PartitionedCall:output:0output_7048269output_7048271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_70482682 
output/StatefulPartitionedCallх
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_41/StatefulPartitionedCall#^dense_6905/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2H
"dense_6905/StatefulPartitionedCall"dense_6905/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

a
E__inference_re_lu_41_layer_call_and_return_conditional_losses_7048686

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџД2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
і
g
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048249

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџP2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
а
Ђ
+__inference_conv2d_41_layer_call_fn_7048681

inputs"
unknown:Д
	unknown_0:	Д
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_70481992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ

+__inference_model_140_layer_call_fn_7048290
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_model_140_layer_call_and_return_conditional_losses_70482752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
Љ

,__inference_dense_6905_layer_call_fn_7048749

inputs
unknown:	ФP
	unknown_0:P
identityЂStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_6905_layer_call_and_return_conditional_losses_70482382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџФ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџФ
 
_user_specified_nameinputs


(__inference_output_layer_call_fn_7048822

inputs
unknown:P
	unknown_0:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_70482682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџP: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
і
g
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048754

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџP2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
з
g
.__inference_dropout_2303_layer_call_fn_7048803

inputs
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2303_layer_call_and_return_conditional_losses_70483202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Џ
h
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048793

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Я

%__inference_signature_wrapper_7048549
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__wrapped_model_70481822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
Ы
J
.__inference_dropout_2302_layer_call_fn_7048771

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2302_layer_call_and_return_conditional_losses_70482492
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Єm
 
#__inference__traced_restore_7049005
file_prefix<
!assignvariableop_conv2d_41_kernel:Д0
!assignvariableop_1_conv2d_41_bias:	Д7
$assignvariableop_2_dense_6905_kernel:	ФP0
"assignvariableop_3_dense_6905_bias:P2
 assignvariableop_4_output_kernel:P,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: F
+assignvariableop_13_adam_conv2d_41_kernel_m:Д8
)assignvariableop_14_adam_conv2d_41_bias_m:	Д?
,assignvariableop_15_adam_dense_6905_kernel_m:	ФP8
*assignvariableop_16_adam_dense_6905_bias_m:P:
(assignvariableop_17_adam_output_kernel_m:P4
&assignvariableop_18_adam_output_bias_m:F
+assignvariableop_19_adam_conv2d_41_kernel_v:Д8
)assignvariableop_20_adam_conv2d_41_bias_v:	Д?
,assignvariableop_21_adam_dense_6905_kernel_v:	ФP8
*assignvariableop_22_adam_dense_6905_bias_v:P:
(assignvariableop_23_adam_output_kernel_v:P4
&assignvariableop_24_adam_output_bias_v:
identity_26ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9І
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*В
valueЈBЅB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesТ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
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

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_41_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_41_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Љ
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_6905_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ї
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_6905_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѕ
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6Ё
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ё
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ё
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Г
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_41_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Б
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_41_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Д
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_6905_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16В
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_6905_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ў
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Г
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_41_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Б
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_41_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Д
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_6905_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22В
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_6905_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23А
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_output_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ў
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_output_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25ї
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

g
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048217

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџД2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
ч

+__inference_model_140_layer_call_fn_7048662

inputs"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_model_140_layer_call_and_return_conditional_losses_70484442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

a
E__inference_re_lu_41_layer_call_and_return_conditional_losses_7048210

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџД2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
з
g
.__inference_dropout_2302_layer_call_fn_7048776

inputs
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2302_layer_call_and_return_conditional_losses_70483432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
В"
Ў
F__inference_model_140_layer_call_and_return_conditional_losses_7048500
input_onehot,
conv2d_41_7048479:Д 
conv2d_41_7048481:	Д%
dense_6905_7048487:	ФP 
dense_6905_7048489:P 
output_7048494:P
output_7048496:
identityЂ!conv2d_41/StatefulPartitionedCallЂ"dense_6905/StatefulPartitionedCallЂoutput/StatefulPartitionedCallА
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_41_7048479conv2d_41_7048481*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_70481992#
!conv2d_41/StatefulPartitionedCall
re_lu_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_41_layer_call_and_return_conditional_losses_70482102
re_lu_41/PartitionedCall
dropout_2301/PartitionedCallPartitionedCall!re_lu_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2301_layer_call_and_return_conditional_losses_70482172
dropout_2301/PartitionedCall
flatten_140/PartitionedCallPartitionedCall%dropout_2301/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџФ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_140_layer_call_and_return_conditional_losses_70482252
flatten_140/PartitionedCallФ
"dense_6905/StatefulPartitionedCallStatefulPartitionedCall$flatten_140/PartitionedCall:output:0dense_6905_7048487dense_6905_7048489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_6905_layer_call_and_return_conditional_losses_70482382$
"dense_6905/StatefulPartitionedCall
dropout_2302/PartitionedCallPartitionedCall+dense_6905/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2302_layer_call_and_return_conditional_losses_70482492
dropout_2302/PartitionedCall
dropout_2303/PartitionedCallPartitionedCall%dropout_2302/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2303_layer_call_and_return_conditional_losses_70482562
dropout_2303/PartitionedCallБ
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_2303/PartitionedCall:output:0output_7048494output_7048496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_70482682 
output/StatefulPartitionedCallх
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_41/StatefulPartitionedCall#^dense_6905/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2H
"dense_6905/StatefulPartitionedCall"dense_6905/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
Я	
є
C__inference_output_layer_call_and_return_conditional_losses_7048813

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
ч
F
*__inference_re_lu_41_layer_call_fn_7048691

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_41_layer_call_and_return_conditional_losses_70482102
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
Г

љ
G__inference_dense_6905_layer_call_and_return_conditional_losses_7048740

inputs1
matmul_readvariableop_resource:	ФP-
biasadd_readvariableop_resource:P
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџФ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџФ
 
_user_specified_nameinputs
К


F__inference_conv2d_41_layer_call_and_return_conditional_losses_7048672

inputs9
conv2d_readvariableop_resource:Д.
biasadd_readvariableop_resource:	Д
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
g
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048256

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџP2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
я
J
.__inference_dropout_2301_layer_call_fn_7048713

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2301_layer_call_and_return_conditional_losses_70482172
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
К


F__inference_conv2d_41_layer_call_and_return_conditional_losses_7048199

inputs9
conv2d_readvariableop_resource:Д.
biasadd_readvariableop_resource:	Д
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д'
Ѓ
F__inference_model_140_layer_call_and_return_conditional_losses_7048524
input_onehot,
conv2d_41_7048503:Д 
conv2d_41_7048505:	Д%
dense_6905_7048511:	ФP 
dense_6905_7048513:P 
output_7048518:P
output_7048520:
identityЂ!conv2d_41/StatefulPartitionedCallЂ"dense_6905/StatefulPartitionedCallЂ$dropout_2301/StatefulPartitionedCallЂ$dropout_2302/StatefulPartitionedCallЂ$dropout_2303/StatefulPartitionedCallЂoutput/StatefulPartitionedCallА
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_41_7048503conv2d_41_7048505*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_70481992#
!conv2d_41/StatefulPartitionedCall
re_lu_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_41_layer_call_and_return_conditional_losses_70482102
re_lu_41/PartitionedCallЂ
$dropout_2301/StatefulPartitionedCallStatefulPartitionedCall!re_lu_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2301_layer_call_and_return_conditional_losses_70483822&
$dropout_2301/StatefulPartitionedCall
flatten_140/PartitionedCallPartitionedCall-dropout_2301/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџФ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_140_layer_call_and_return_conditional_losses_70482252
flatten_140/PartitionedCallФ
"dense_6905/StatefulPartitionedCallStatefulPartitionedCall$flatten_140/PartitionedCall:output:0dense_6905_7048511dense_6905_7048513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_6905_layer_call_and_return_conditional_losses_70482382$
"dense_6905/StatefulPartitionedCallЪ
$dropout_2302/StatefulPartitionedCallStatefulPartitionedCall+dense_6905/StatefulPartitionedCall:output:0%^dropout_2301/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2302_layer_call_and_return_conditional_losses_70483432&
$dropout_2302/StatefulPartitionedCallЬ
$dropout_2303/StatefulPartitionedCallStatefulPartitionedCall-dropout_2302/StatefulPartitionedCall:output:0%^dropout_2302/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2303_layer_call_and_return_conditional_losses_70483202&
$dropout_2303/StatefulPartitionedCallЙ
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_2303/StatefulPartitionedCall:output:0output_7048518output_7048520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_70482682 
output/StatefulPartitionedCallк
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_41/StatefulPartitionedCall#^dense_6905/StatefulPartitionedCall%^dropout_2301/StatefulPartitionedCall%^dropout_2302/StatefulPartitionedCall%^dropout_2303/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2H
"dense_6905/StatefulPartitionedCall"dense_6905/StatefulPartitionedCall2L
$dropout_2301/StatefulPartitionedCall$dropout_2301/StatefulPartitionedCall2L
$dropout_2302/StatefulPartitionedCall$dropout_2302/StatefulPartitionedCall2L
$dropout_2303/StatefulPartitionedCall$dropout_2303/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
Џ
h
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048320

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
љ
h
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048708

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeН
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџД*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yЧ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџД2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
ь
d
H__inference_flatten_140_layer_call_and_return_conditional_losses_7048724

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
Џ
h
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048343

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
Ы
J
.__inference_dropout_2303_layer_call_fn_7048798

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2303_layer_call_and_return_conditional_losses_70482562
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџP:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs
М$

F__inference_model_140_layer_call_and_return_conditional_losses_7048578

inputsC
(conv2d_41_conv2d_readvariableop_resource:Д8
)conv2d_41_biasadd_readvariableop_resource:	Д<
)dense_6905_matmul_readvariableop_resource:	ФP8
*dense_6905_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityЂ conv2d_41/BiasAdd/ReadVariableOpЂconv2d_41/Conv2D/ReadVariableOpЂ!dense_6905/BiasAdd/ReadVariableOpЂ dense_6905/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpД
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02!
conv2d_41/Conv2D/ReadVariableOpУ
conv2d_41/Conv2DConv2Dinputs'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
conv2d_41/Conv2DЋ
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOpБ
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
conv2d_41/BiasAdd}
re_lu_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
re_lu_41/Relu
dropout_2301/IdentityIdentityre_lu_41/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_2301/Identityw
flatten_140/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
flatten_140/ConstЄ
flatten_140/ReshapeReshapedropout_2301/Identity:output:0flatten_140/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
flatten_140/ReshapeЏ
 dense_6905/MatMul/ReadVariableOpReadVariableOp)dense_6905_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02"
 dense_6905/MatMul/ReadVariableOpЊ
dense_6905/MatMulMatMulflatten_140/Reshape:output:0(dense_6905/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_6905/MatMul­
!dense_6905/BiasAdd/ReadVariableOpReadVariableOp*dense_6905_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02#
!dense_6905/BiasAdd/ReadVariableOp­
dense_6905/BiasAddBiasAdddense_6905/MatMul:product:0)dense_6905/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_6905/BiasAddy
dense_6905/ReluReludense_6905/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_6905/Relu
dropout_2302/IdentityIdentitydense_6905/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_2302/Identity
dropout_2303/IdentityIdentitydropout_2302/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_2303/IdentityЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp 
output/MatMulMatMuldropout_2303/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddЖ
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp"^dense_6905/BiasAdd/ReadVariableOp!^dense_6905/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2F
!dense_6905/BiasAdd/ReadVariableOp!dense_6905/BiasAdd/ReadVariableOp2D
 dense_6905/MatMul/ReadVariableOp dense_6905/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѕ;
С

 __inference__traced_save_7048920
file_prefix/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop0
,savev2_dense_6905_kernel_read_readvariableop.
*savev2_dense_6905_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableop7
3savev2_adam_dense_6905_kernel_m_read_readvariableop5
1savev2_adam_dense_6905_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableop7
3savev2_adam_dense_6905_kernel_v_read_readvariableop5
1savev2_adam_dense_6905_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*В
valueЈBЅB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesМ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesУ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop,savev2_dense_6905_kernel_read_readvariableop*savev2_dense_6905_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop3savev2_adam_dense_6905_kernel_m_read_readvariableop1savev2_adam_dense_6905_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop3savev2_adam_dense_6905_kernel_v_read_readvariableop1savev2_adam_dense_6905_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*и
_input_shapesЦ
У: :Д:Д:	ФP:P:P:: : : : : : : :Д:Д:	ФP:P:P::Д:Д:	ФP:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:Д:!

_output_shapes	
:Д:%!

_output_shapes
:	ФP: 
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
:Д:!

_output_shapes	
:Д:%!

_output_shapes
:	ФP: 
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
:Д:!

_output_shapes	
:Д:%!

_output_shapes
:	ФP: 
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
ЃB

F__inference_model_140_layer_call_and_return_conditional_losses_7048628

inputsC
(conv2d_41_conv2d_readvariableop_resource:Д8
)conv2d_41_biasadd_readvariableop_resource:	Д<
)dense_6905_matmul_readvariableop_resource:	ФP8
*dense_6905_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityЂ conv2d_41/BiasAdd/ReadVariableOpЂconv2d_41/Conv2D/ReadVariableOpЂ!dense_6905/BiasAdd/ReadVariableOpЂ dense_6905/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpД
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02!
conv2d_41/Conv2D/ReadVariableOpУ
conv2d_41/Conv2DConv2Dinputs'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
conv2d_41/Conv2DЋ
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOpБ
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
conv2d_41/BiasAdd}
re_lu_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
re_lu_41/Relu}
dropout_2301/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_2301/dropout/ConstИ
dropout_2301/dropout/MulMulre_lu_41/Relu:activations:0#dropout_2301/dropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_2301/dropout/Mul
dropout_2301/dropout/ShapeShapere_lu_41/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2301/dropout/Shapeф
1dropout_2301/dropout/random_uniform/RandomUniformRandomUniform#dropout_2301/dropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџД*
dtype023
1dropout_2301/dropout/random_uniform/RandomUniform
#dropout_2301/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#dropout_2301/dropout/GreaterEqual/yћ
!dropout_2301/dropout/GreaterEqualGreaterEqual:dropout_2301/dropout/random_uniform/RandomUniform:output:0,dropout_2301/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2#
!dropout_2301/dropout/GreaterEqualЏ
dropout_2301/dropout/CastCast%dropout_2301/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџД2
dropout_2301/dropout/CastЗ
dropout_2301/dropout/Mul_1Muldropout_2301/dropout/Mul:z:0dropout_2301/dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_2301/dropout/Mul_1w
flatten_140/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
flatten_140/ConstЄ
flatten_140/ReshapeReshapedropout_2301/dropout/Mul_1:z:0flatten_140/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
flatten_140/ReshapeЏ
 dense_6905/MatMul/ReadVariableOpReadVariableOp)dense_6905_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02"
 dense_6905/MatMul/ReadVariableOpЊ
dense_6905/MatMulMatMulflatten_140/Reshape:output:0(dense_6905/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_6905/MatMul­
!dense_6905/BiasAdd/ReadVariableOpReadVariableOp*dense_6905_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02#
!dense_6905/BiasAdd/ReadVariableOp­
dense_6905/BiasAddBiasAdddense_6905/MatMul:product:0)dense_6905/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_6905/BiasAddy
dense_6905/ReluReludense_6905/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_6905/Relu}
dropout_2302/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_2302/dropout/ConstБ
dropout_2302/dropout/MulMuldense_6905/Relu:activations:0#dropout_2302/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_2302/dropout/Mul
dropout_2302/dropout/ShapeShapedense_6905/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2302/dropout/Shapeл
1dropout_2302/dropout/random_uniform/RandomUniformRandomUniform#dropout_2302/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype023
1dropout_2302/dropout/random_uniform/RandomUniform
#dropout_2302/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#dropout_2302/dropout/GreaterEqual/yђ
!dropout_2302/dropout/GreaterEqualGreaterEqual:dropout_2302/dropout/random_uniform/RandomUniform:output:0,dropout_2302/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2#
!dropout_2302/dropout/GreaterEqualІ
dropout_2302/dropout/CastCast%dropout_2302/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout_2302/dropout/CastЎ
dropout_2302/dropout/Mul_1Muldropout_2302/dropout/Mul:z:0dropout_2302/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_2302/dropout/Mul_1}
dropout_2303/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_2303/dropout/ConstВ
dropout_2303/dropout/MulMuldropout_2302/dropout/Mul_1:z:0#dropout_2303/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_2303/dropout/Mul
dropout_2303/dropout/ShapeShapedropout_2302/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dropout_2303/dropout/Shapeл
1dropout_2303/dropout/random_uniform/RandomUniformRandomUniform#dropout_2303/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype023
1dropout_2303/dropout/random_uniform/RandomUniform
#dropout_2303/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2%
#dropout_2303/dropout/GreaterEqual/yђ
!dropout_2303/dropout/GreaterEqualGreaterEqual:dropout_2303/dropout/random_uniform/RandomUniform:output:0,dropout_2303/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2#
!dropout_2303/dropout/GreaterEqualІ
dropout_2303/dropout/CastCast%dropout_2303/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout_2303/dropout/CastЎ
dropout_2303/dropout/Mul_1Muldropout_2303/dropout/Mul:z:0dropout_2303/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_2303/dropout/Mul_1Ђ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp 
output/MatMulMatMuldropout_2303/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddЖ
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp"^dense_6905/BiasAdd/ReadVariableOp!^dense_6905/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2F
!dense_6905/BiasAdd/ReadVariableOp!dense_6905/BiasAdd/ReadVariableOp2D
 dense_6905/MatMul/ReadVariableOp dense_6905/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ

+__inference_model_140_layer_call_fn_7048476
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_model_140_layer_call_and_return_conditional_losses_70484442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
Я+
№
"__inference__wrapped_model_7048182
input_onehotM
2model_140_conv2d_41_conv2d_readvariableop_resource:ДB
3model_140_conv2d_41_biasadd_readvariableop_resource:	ДF
3model_140_dense_6905_matmul_readvariableop_resource:	ФPB
4model_140_dense_6905_biasadd_readvariableop_resource:PA
/model_140_output_matmul_readvariableop_resource:P>
0model_140_output_biasadd_readvariableop_resource:
identityЂ*model_140/conv2d_41/BiasAdd/ReadVariableOpЂ)model_140/conv2d_41/Conv2D/ReadVariableOpЂ+model_140/dense_6905/BiasAdd/ReadVariableOpЂ*model_140/dense_6905/MatMul/ReadVariableOpЂ'model_140/output/BiasAdd/ReadVariableOpЂ&model_140/output/MatMul/ReadVariableOpв
)model_140/conv2d_41/Conv2D/ReadVariableOpReadVariableOp2model_140_conv2d_41_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02+
)model_140/conv2d_41/Conv2D/ReadVariableOpч
model_140/conv2d_41/Conv2DConv2Dinput_onehot1model_140/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
model_140/conv2d_41/Conv2DЩ
*model_140/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp3model_140_conv2d_41_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02,
*model_140/conv2d_41/BiasAdd/ReadVariableOpй
model_140/conv2d_41/BiasAddBiasAdd#model_140/conv2d_41/Conv2D:output:02model_140/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_140/conv2d_41/BiasAdd
model_140/re_lu_41/ReluRelu$model_140/conv2d_41/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_140/re_lu_41/ReluА
model_140/dropout_2301/IdentityIdentity%model_140/re_lu_41/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2!
model_140/dropout_2301/Identity
model_140/flatten_140/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
model_140/flatten_140/ConstЬ
model_140/flatten_140/ReshapeReshape(model_140/dropout_2301/Identity:output:0$model_140/flatten_140/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
model_140/flatten_140/ReshapeЭ
*model_140/dense_6905/MatMul/ReadVariableOpReadVariableOp3model_140_dense_6905_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02,
*model_140/dense_6905/MatMul/ReadVariableOpв
model_140/dense_6905/MatMulMatMul&model_140/flatten_140/Reshape:output:02model_140/dense_6905/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_140/dense_6905/MatMulЫ
+model_140/dense_6905/BiasAdd/ReadVariableOpReadVariableOp4model_140_dense_6905_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02-
+model_140/dense_6905/BiasAdd/ReadVariableOpе
model_140/dense_6905/BiasAddBiasAdd%model_140/dense_6905/MatMul:product:03model_140/dense_6905/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_140/dense_6905/BiasAdd
model_140/dense_6905/ReluRelu%model_140/dense_6905/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_140/dense_6905/ReluЉ
model_140/dropout_2302/IdentityIdentity'model_140/dense_6905/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџP2!
model_140/dropout_2302/IdentityЊ
model_140/dropout_2303/IdentityIdentity(model_140/dropout_2302/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2!
model_140/dropout_2303/IdentityР
&model_140/output/MatMul/ReadVariableOpReadVariableOp/model_140_output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02(
&model_140/output/MatMul/ReadVariableOpШ
model_140/output/MatMulMatMul(model_140/dropout_2303/Identity:output:0.model_140/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_140/output/MatMulП
'model_140/output/BiasAdd/ReadVariableOpReadVariableOp0model_140_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_140/output/BiasAdd/ReadVariableOpХ
model_140/output/BiasAddBiasAdd!model_140/output/MatMul:product:0/model_140/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_140/output/BiasAddќ
IdentityIdentity!model_140/output/BiasAdd:output:0+^model_140/conv2d_41/BiasAdd/ReadVariableOp*^model_140/conv2d_41/Conv2D/ReadVariableOp,^model_140/dense_6905/BiasAdd/ReadVariableOp+^model_140/dense_6905/MatMul/ReadVariableOp(^model_140/output/BiasAdd/ReadVariableOp'^model_140/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2X
*model_140/conv2d_41/BiasAdd/ReadVariableOp*model_140/conv2d_41/BiasAdd/ReadVariableOp2V
)model_140/conv2d_41/Conv2D/ReadVariableOp)model_140/conv2d_41/Conv2D/ReadVariableOp2Z
+model_140/dense_6905/BiasAdd/ReadVariableOp+model_140/dense_6905/BiasAdd/ReadVariableOp2X
*model_140/dense_6905/MatMul/ReadVariableOp*model_140/dense_6905/MatMul/ReadVariableOp2R
'model_140/output/BiasAdd/ReadVariableOp'model_140/output/BiasAdd/ReadVariableOp2P
&model_140/output/MatMul/ReadVariableOp&model_140/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
Ђ'

F__inference_model_140_layer_call_and_return_conditional_losses_7048444

inputs,
conv2d_41_7048423:Д 
conv2d_41_7048425:	Д%
dense_6905_7048431:	ФP 
dense_6905_7048433:P 
output_7048438:P
output_7048440:
identityЂ!conv2d_41/StatefulPartitionedCallЂ"dense_6905/StatefulPartitionedCallЂ$dropout_2301/StatefulPartitionedCallЂ$dropout_2302/StatefulPartitionedCallЂ$dropout_2303/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЊ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_41_7048423conv2d_41_7048425*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_70481992#
!conv2d_41/StatefulPartitionedCall
re_lu_41/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_41_layer_call_and_return_conditional_losses_70482102
re_lu_41/PartitionedCallЂ
$dropout_2301/StatefulPartitionedCallStatefulPartitionedCall!re_lu_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2301_layer_call_and_return_conditional_losses_70483822&
$dropout_2301/StatefulPartitionedCall
flatten_140/PartitionedCallPartitionedCall-dropout_2301/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџФ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_140_layer_call_and_return_conditional_losses_70482252
flatten_140/PartitionedCallФ
"dense_6905/StatefulPartitionedCallStatefulPartitionedCall$flatten_140/PartitionedCall:output:0dense_6905_7048431dense_6905_7048433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_6905_layer_call_and_return_conditional_losses_70482382$
"dense_6905/StatefulPartitionedCallЪ
$dropout_2302/StatefulPartitionedCallStatefulPartitionedCall+dense_6905/StatefulPartitionedCall:output:0%^dropout_2301/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2302_layer_call_and_return_conditional_losses_70483432&
$dropout_2302/StatefulPartitionedCallЬ
$dropout_2303/StatefulPartitionedCallStatefulPartitionedCall-dropout_2302/StatefulPartitionedCall:output:0%^dropout_2302/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџP* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2303_layer_call_and_return_conditional_losses_70483202&
$dropout_2303/StatefulPartitionedCallЙ
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_2303/StatefulPartitionedCall:output:0output_7048438output_7048440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_70482682 
output/StatefulPartitionedCallк
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_41/StatefulPartitionedCall#^dense_6905/StatefulPartitionedCall%^dropout_2301/StatefulPartitionedCall%^dropout_2302/StatefulPartitionedCall%^dropout_2303/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2H
"dense_6905/StatefulPartitionedCall"dense_6905/StatefulPartitionedCall2L
$dropout_2301/StatefulPartitionedCall$dropout_2301/StatefulPartitionedCall2L
$dropout_2302/StatefulPartitionedCall$dropout_2302/StatefulPartitionedCall2L
$dropout_2303/StatefulPartitionedCall$dropout_2303/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
d
H__inference_flatten_140_layer_call_and_return_conditional_losses_7048225

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
Г

љ
G__inference_dense_6905_layer_call_and_return_conditional_losses_7048238

inputs1
matmul_readvariableop_resource:	ФP-
biasadd_readvariableop_resource:P
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџP2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџФ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџФ
 
_user_specified_nameinputs
ћ
g
.__inference_dropout_2301_layer_call_fn_7048718

inputs
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџД* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_dropout_2301_layer_call_and_return_conditional_losses_70483822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
н
I
-__inference_flatten_140_layer_call_fn_7048729

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџФ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_flatten_140_layer_call_and_return_conditional_losses_70482252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
љ
h
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048382

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeН
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџД*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yЧ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџД2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs

g
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048696

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџД2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџД:X T
0
_output_shapes
:џџџџџџџџџД
 
_user_specified_nameinputs
Я	
є
C__inference_output_layer_call_and_return_conditional_losses_7048268

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџP
 
_user_specified_nameinputs"ЬL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultЇ
M
input_onehot=
serving_default_input_onehot:0џџџџџџџџџ:
output0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ё
П@
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
{_default_save_signature"Б=
_tf_keras_network={"name": "model_140", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_140", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_41", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_41", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_41", "inbound_nodes": [[["conv2d_41", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2301", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2301", "inbound_nodes": [[["re_lu_41", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_140", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_140", "inbound_nodes": [[["dropout_2301", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6905", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6905", "inbound_nodes": [[["flatten_140", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2302", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2302", "inbound_nodes": [[["dense_6905", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2303", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2303", "inbound_nodes": [[["dropout_2302", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_2303", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_140", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_41", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "ReLU", "config": {"name": "re_lu_41", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_41", "inbound_nodes": [[["conv2d_41", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_2301", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2301", "inbound_nodes": [[["re_lu_41", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_140", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_140", "inbound_nodes": [[["dropout_2301", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_6905", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6905", "inbound_nodes": [[["flatten_140", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_2302", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2302", "inbound_nodes": [[["dense_6905", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_2303", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_2303", "inbound_nodes": [[["dropout_2302", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_2303", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"ў
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*|&call_and_return_all_conditional_losses
}__call__"о	
_tf_keras_layerФ	{"name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}}
В
trainable_variables
regularization_losses
	variables
	keras_api
*~&call_and_return_all_conditional_losses
__call__"Ѓ
_tf_keras_layer{"name": "re_lu_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_41", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv2d_41", 0, 0, {}]]], "shared_object_id": 4}
Б
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__" 
_tf_keras_layer{"name": "dropout_2301", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2301", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["re_lu_41", 0, 0, {}]]], "shared_object_id": 5}
Ь
trainable_variables
regularization_losses
 	variables
!	keras_api
+&call_and_return_all_conditional_losses
__call__"Л
_tf_keras_layerЁ{"name": "flatten_140", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_140", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["dropout_2301", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
	

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"т
_tf_keras_layerШ{"name": "dense_6905", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6905", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_140", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3780}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3780]}}
Д
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+&call_and_return_all_conditional_losses
__call__"Ѓ
_tf_keras_layer{"name": "dropout_2302", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2302", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_6905", 0, 0, {}]]], "shared_object_id": 10}
Ж
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+&call_and_return_all_conditional_losses
__call__"Ѕ
_tf_keras_layer{"name": "dropout_2303", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2303", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dropout_2302", 0, 0, {}]]], "shared_object_id": 11}
	

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+&call_and_return_all_conditional_losses
__call__"л
_tf_keras_layerС{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_2303", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
П
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
Ъ
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
serving_default"
signature_map
+:)Д2conv2d_41/kernel
:Д2conv2d_41/bias
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
­
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
­
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
А
trainable_variables
Jlayer_metrics
Knon_trainable_variables
regularization_losses
Llayer_regularization_losses
Mmetrics

Nlayers
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
 	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
$:"	ФP2dense_6905/kernel
:P2dense_6905/bias
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
А
$trainable_variables
Tlayer_metrics
Unon_trainable_variables
%regularization_losses
Vlayer_regularization_losses
Wmetrics

Xlayers
&	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
(trainable_variables
Ylayer_metrics
Znon_trainable_variables
)regularization_losses
[layer_regularization_losses
\metrics

]layers
*	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
,trainable_variables
^layer_metrics
_non_trainable_variables
-regularization_losses
`layer_regularization_losses
ametrics

blayers
.	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
А
2trainable_variables
clayer_metrics
dnon_trainable_variables
3regularization_losses
elayer_regularization_losses
fmetrics

glayers
4	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
д
	itotal
	jcount
k	variables
l	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 21}
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
0:.Д2Adam/conv2d_41/kernel/m
": Д2Adam/conv2d_41/bias/m
):'	ФP2Adam/dense_6905/kernel/m
": P2Adam/dense_6905/bias/m
$:"P2Adam/output/kernel/m
:2Adam/output/bias/m
0:.Д2Adam/conv2d_41/kernel/v
": Д2Adam/conv2d_41/bias/v
):'	ФP2Adam/dense_6905/kernel/v
": P2Adam/dense_6905/bias/v
$:"P2Adam/output/kernel/v
:2Adam/output/bias/v
ц2у
F__inference_model_140_layer_call_and_return_conditional_losses_7048578
F__inference_model_140_layer_call_and_return_conditional_losses_7048628
F__inference_model_140_layer_call_and_return_conditional_losses_7048500
F__inference_model_140_layer_call_and_return_conditional_losses_7048524Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њ2ї
+__inference_model_140_layer_call_fn_7048290
+__inference_model_140_layer_call_fn_7048645
+__inference_model_140_layer_call_fn_7048662
+__inference_model_140_layer_call_fn_7048476Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
э2ъ
"__inference__wrapped_model_7048182У
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+
input_onehotџџџџџџџџџ
№2э
F__inference_conv2d_41_layer_call_and_return_conditional_losses_7048672Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv2d_41_layer_call_fn_7048681Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_re_lu_41_layer_call_and_return_conditional_losses_7048686Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_re_lu_41_layer_call_fn_7048691Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048696
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048708Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_dropout_2301_layer_call_fn_7048713
.__inference_dropout_2301_layer_call_fn_7048718Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
H__inference_flatten_140_layer_call_and_return_conditional_losses_7048724Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_flatten_140_layer_call_fn_7048729Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_6905_layer_call_and_return_conditional_losses_7048740Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_dense_6905_layer_call_fn_7048749Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048754
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048766Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_dropout_2302_layer_call_fn_7048771
.__inference_dropout_2302_layer_call_fn_7048776Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048781
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048793Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_dropout_2303_layer_call_fn_7048798
.__inference_dropout_2303_layer_call_fn_7048803Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
э2ъ
C__inference_output_layer_call_and_return_conditional_losses_7048813Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_output_layer_call_fn_7048822Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
бBЮ
%__inference_signature_wrapper_7048549input_onehot"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
"__inference__wrapped_model_7048182x"#01=Ђ:
3Ђ0
.+
input_onehotџџџџџџџџџ
Њ "/Њ,
*
output 
outputџџџџџџџџџЗ
F__inference_conv2d_41_layer_call_and_return_conditional_losses_7048672m7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџД
 
+__inference_conv2d_41_layer_call_fn_7048681`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "!џџџџџџџџџДЈ
G__inference_dense_6905_layer_call_and_return_conditional_losses_7048740]"#0Ђ-
&Ђ#
!
inputsџџџџџџџџџФ
Њ "%Ђ"

0џџџџџџџџџP
 
,__inference_dense_6905_layer_call_fn_7048749P"#0Ђ-
&Ђ#
!
inputsџџџџџџџџџФ
Њ "џџџџџџџџџPЛ
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048696n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p 
Њ ".Ђ+
$!
0џџџџџџџџџД
 Л
I__inference_dropout_2301_layer_call_and_return_conditional_losses_7048708n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p
Њ ".Ђ+
$!
0џџџџџџџџџД
 
.__inference_dropout_2301_layer_call_fn_7048713a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p 
Њ "!џџџџџџџџџД
.__inference_dropout_2301_layer_call_fn_7048718a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p
Њ "!џџџџџџџџџДЉ
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048754\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "%Ђ"

0џџџџџџџџџP
 Љ
I__inference_dropout_2302_layer_call_and_return_conditional_losses_7048766\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "%Ђ"

0џџџџџџџџџP
 
.__inference_dropout_2302_layer_call_fn_7048771O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "џџџџџџџџџP
.__inference_dropout_2302_layer_call_fn_7048776O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "џџџџџџџџџPЉ
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048781\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "%Ђ"

0џџџџџџџџџP
 Љ
I__inference_dropout_2303_layer_call_and_return_conditional_losses_7048793\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "%Ђ"

0џџџџџџџџџP
 
.__inference_dropout_2303_layer_call_fn_7048798O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "џџџџџџџџџP
.__inference_dropout_2303_layer_call_fn_7048803O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "џџџџџџџџџPЎ
H__inference_flatten_140_layer_call_and_return_conditional_losses_7048724b8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "&Ђ#

0џџџџџџџџџФ
 
-__inference_flatten_140_layer_call_fn_7048729U8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "џџџџџџџџџФР
F__inference_model_140_layer_call_and_return_conditional_losses_7048500v"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Р
F__inference_model_140_layer_call_and_return_conditional_losses_7048524v"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 К
F__inference_model_140_layer_call_and_return_conditional_losses_7048578p"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 К
F__inference_model_140_layer_call_and_return_conditional_losses_7048628p"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_model_140_layer_call_fn_7048290i"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
+__inference_model_140_layer_call_fn_7048476i"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
+__inference_model_140_layer_call_fn_7048645c"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
+__inference_model_140_layer_call_fn_7048662c"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЃ
C__inference_output_layer_call_and_return_conditional_losses_7048813\01/Ђ,
%Ђ"
 
inputsџџџџџџџџџP
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_output_layer_call_fn_7048822O01/Ђ,
%Ђ"
 
inputsџџџџџџџџџP
Њ "џџџџџџџџџГ
E__inference_re_lu_41_layer_call_and_return_conditional_losses_7048686j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ ".Ђ+
$!
0џџџџџџџџџД
 
*__inference_re_lu_41_layer_call_fn_7048691]8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "!џџџџџџџџџДВ
%__inference_signature_wrapper_7048549"#01MЂJ
Ђ 
CЊ@
>
input_onehot.+
input_onehotџџџџџџџџџ"/Њ,
*
output 
outputџџџџџџџџџ