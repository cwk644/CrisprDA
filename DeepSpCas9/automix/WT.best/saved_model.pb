Ня
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ом

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д* 
shared_nameconv2d_6/kernel
|
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*'
_output_shapes
:Д*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:Д*
dtype0

dense_1254/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*"
shared_namedense_1254/kernel
x
%dense_1254/kernel/Read/ReadVariableOpReadVariableOpdense_1254/kernel*
_output_shapes
:	ФP*
dtype0
v
dense_1254/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P* 
shared_namedense_1254/bias
o
#dense_1254/bias/Read/ReadVariableOpReadVariableOpdense_1254/bias*
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

Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*'
shared_nameAdam/conv2d_6/kernel/m

*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*'
_output_shapes
:Д*
dtype0

Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*%
shared_nameAdam/conv2d_6/bias/m
z
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes	
:Д*
dtype0

Adam/dense_1254/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*)
shared_nameAdam/dense_1254/kernel/m

,Adam/dense_1254/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1254/kernel/m*
_output_shapes
:	ФP*
dtype0

Adam/dense_1254/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*'
shared_nameAdam/dense_1254/bias/m
}
*Adam/dense_1254/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1254/bias/m*
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

Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*'
shared_nameAdam/conv2d_6/kernel/v

*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*'
_output_shapes
:Д*
dtype0

Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*%
shared_nameAdam/conv2d_6/bias/v
z
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes	
:Д*
dtype0

Adam/dense_1254/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*)
shared_nameAdam/dense_1254/kernel/v

,Adam/dense_1254/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1254/kernel/v*
_output_shapes
:	ФP*
dtype0

Adam/dense_1254/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*'
shared_nameAdam/dense_1254/bias/v
}
*Adam/dense_1254/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1254/bias/v*
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
ц.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ё.
value.B. B.
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
R
(regularization_losses
)trainable_variables
*	variables
+	keras_api
R
,regularization_losses
-trainable_variables
.	variables
/	keras_api
h

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
Ќ
6iter

7beta_1

8beta_2
	9decay
:learning_ratemmmn"mo#mp0mq1mrvsvt"vu#vv0vw1vx
 
*
0
1
"2
#3
04
15
*
0
1
"2
#3
04
15
­
regularization_losses
;non_trainable_variables
<layer_regularization_losses
=metrics
trainable_variables
>layer_metrics
	variables

?layers
 
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
@non_trainable_variables
Alayer_regularization_losses
Bmetrics
trainable_variables
Clayer_metrics
	variables

Dlayers
 
 
 
­
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
Gmetrics
trainable_variables
Hlayer_metrics
	variables

Ilayers
 
 
 
­
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
Lmetrics
trainable_variables
Mlayer_metrics
	variables

Nlayers
 
 
 
­
regularization_losses
Onon_trainable_variables
Player_regularization_losses
Qmetrics
trainable_variables
Rlayer_metrics
 	variables

Slayers
][
VARIABLE_VALUEdense_1254/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_1254/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
­
$regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
Vmetrics
%trainable_variables
Wlayer_metrics
&	variables

Xlayers
 
 
 
­
(regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
[metrics
)trainable_variables
\layer_metrics
*	variables

]layers
 
 
 
­
,regularization_losses
^non_trainable_variables
_layer_regularization_losses
`metrics
-trainable_variables
alayer_metrics
.	variables

blayers
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
­
2regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
emetrics
3trainable_variables
flayer_metrics
4	variables

glayers
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

h0
 
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
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_1254/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1254/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_1254/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_1254/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
Ј
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_6/kernelconv2d_6/biasdense_1254/kerneldense_1254/biasoutput/kerneloutput/bias*
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
GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_787557
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
є	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp%dense_1254/kernel/Read/ReadVariableOp#dense_1254/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp,Adam/dense_1254/kernel/m/Read/ReadVariableOp*Adam/dense_1254/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp,Adam/dense_1254/kernel/v/Read/ReadVariableOp*Adam/dense_1254/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
GPU2 *0J 8 *(
f#R!
__inference__traced_save_787928
ћ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasdense_1254/kerneldense_1254/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/dense_1254/kernel/mAdam/dense_1254/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/dense_1254/kernel/vAdam/dense_1254/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_788013Пщ
ї
f
G__inference_dropout_414_layer_call_and_return_conditional_losses_787716

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
$

D__inference_model_24_layer_call_and_return_conditional_losses_787586

inputsB
'conv2d_6_conv2d_readvariableop_resource:Д7
(conv2d_6_biasadd_readvariableop_resource:	Д<
)dense_1254_matmul_readvariableop_resource:	ФP8
*dense_1254_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityЂconv2d_6/BiasAdd/ReadVariableOpЂconv2d_6/Conv2D/ReadVariableOpЂ!dense_1254/BiasAdd/ReadVariableOpЂ dense_1254/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpБ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02 
conv2d_6/Conv2D/ReadVariableOpР
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
conv2d_6/Conv2DЈ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
conv2d_6/BiasAddz
re_lu_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
re_lu_6/Relu
dropout_414/IdentityIdentityre_lu_6/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_414/Identityu
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
flatten_24/Const 
flatten_24/ReshapeReshapedropout_414/Identity:output:0flatten_24/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
flatten_24/ReshapeЏ
 dense_1254/MatMul/ReadVariableOpReadVariableOp)dense_1254_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02"
 dense_1254/MatMul/ReadVariableOpЉ
dense_1254/MatMulMatMulflatten_24/Reshape:output:0(dense_1254/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_1254/MatMul­
!dense_1254/BiasAdd/ReadVariableOpReadVariableOp*dense_1254_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02#
!dense_1254/BiasAdd/ReadVariableOp­
dense_1254/BiasAddBiasAdddense_1254/MatMul:product:0)dense_1254/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_1254/BiasAddy
dense_1254/ReluReludense_1254/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_1254/Relu
dropout_415/IdentityIdentitydense_1254/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_415/Identity
dropout_416/IdentityIdentitydropout_415/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_416/IdentityЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_416/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAddД
IdentityIdentityoutput/BiasAdd:output:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp"^dense_1254/BiasAdd/ReadVariableOp!^dense_1254/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2F
!dense_1254/BiasAdd/ReadVariableOp!dense_1254/BiasAdd/ReadVariableOp2D
 dense_1254/MatMul/ReadVariableOp dense_1254/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о&

D__inference_model_24_layer_call_and_return_conditional_losses_787452

inputs*
conv2d_6_787431:Д
conv2d_6_787433:	Д$
dense_1254_787439:	ФP
dense_1254_787441:P
output_787446:P
output_787448:
identityЂ conv2d_6/StatefulPartitionedCallЂ"dense_1254/StatefulPartitionedCallЂ#dropout_414/StatefulPartitionedCallЂ#dropout_415/StatefulPartitionedCallЂ#dropout_416/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЂ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_787431conv2d_6_787433*
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
GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_7872072"
 conv2d_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *L
fGRE
C__inference_re_lu_6_layer_call_and_return_conditional_losses_7872182
re_lu_6/PartitionedCall
#dropout_414/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_414_layer_call_and_return_conditional_losses_7873902%
#dropout_414/StatefulPartitionedCall
flatten_24/PartitionedCallPartitionedCall,dropout_414/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *O
fJRH
F__inference_flatten_24_layer_call_and_return_conditional_losses_7872332
flatten_24/PartitionedCallР
"dense_1254/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_1254_787439dense_1254_787441*
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
GPU2 *0J 8 *O
fJRH
F__inference_dense_1254_layer_call_and_return_conditional_losses_7872462$
"dense_1254/StatefulPartitionedCallХ
#dropout_415/StatefulPartitionedCallStatefulPartitionedCall+dense_1254/StatefulPartitionedCall:output:0$^dropout_414/StatefulPartitionedCall*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_415_layer_call_and_return_conditional_losses_7873512%
#dropout_415/StatefulPartitionedCallЦ
#dropout_416/StatefulPartitionedCallStatefulPartitionedCall,dropout_415/StatefulPartitionedCall:output:0$^dropout_415/StatefulPartitionedCall*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_416_layer_call_and_return_conditional_losses_7873282%
#dropout_416/StatefulPartitionedCallЕ
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_416/StatefulPartitionedCall:output:0output_787446output_787448*
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
GPU2 *0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_7872762 
output/StatefulPartitionedCallж
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall#^dense_1254/StatefulPartitionedCall$^dropout_414/StatefulPartitionedCall$^dropout_415/StatefulPartitionedCall$^dropout_416/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2H
"dense_1254/StatefulPartitionedCall"dense_1254/StatefulPartitionedCall2J
#dropout_414/StatefulPartitionedCall#dropout_414/StatefulPartitionedCall2J
#dropout_415/StatefulPartitionedCall#dropout_415/StatefulPartitionedCall2J
#dropout_416/StatefulPartitionedCall#dropout_416/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

+__inference_dense_1254_layer_call_fn_787757

inputs
unknown:	ФP
	unknown_0:P
identityЂStatefulPartitionedCallћ
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
GPU2 *0J 8 *O
fJRH
F__inference_dense_1254_layer_call_and_return_conditional_losses_7872462
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
Ч*
п
!__inference__wrapped_model_787190
input_onehotK
0model_24_conv2d_6_conv2d_readvariableop_resource:Д@
1model_24_conv2d_6_biasadd_readvariableop_resource:	ДE
2model_24_dense_1254_matmul_readvariableop_resource:	ФPA
3model_24_dense_1254_biasadd_readvariableop_resource:P@
.model_24_output_matmul_readvariableop_resource:P=
/model_24_output_biasadd_readvariableop_resource:
identityЂ(model_24/conv2d_6/BiasAdd/ReadVariableOpЂ'model_24/conv2d_6/Conv2D/ReadVariableOpЂ*model_24/dense_1254/BiasAdd/ReadVariableOpЂ)model_24/dense_1254/MatMul/ReadVariableOpЂ&model_24/output/BiasAdd/ReadVariableOpЂ%model_24/output/MatMul/ReadVariableOpЬ
'model_24/conv2d_6/Conv2D/ReadVariableOpReadVariableOp0model_24_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02)
'model_24/conv2d_6/Conv2D/ReadVariableOpс
model_24/conv2d_6/Conv2DConv2Dinput_onehot/model_24/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
model_24/conv2d_6/Conv2DУ
(model_24/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp1model_24_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02*
(model_24/conv2d_6/BiasAdd/ReadVariableOpб
model_24/conv2d_6/BiasAddBiasAdd!model_24/conv2d_6/Conv2D:output:00model_24/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_24/conv2d_6/BiasAdd
model_24/re_lu_6/ReluRelu"model_24/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_24/re_lu_6/ReluЊ
model_24/dropout_414/IdentityIdentity#model_24/re_lu_6/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_24/dropout_414/Identity
model_24/flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
model_24/flatten_24/ConstФ
model_24/flatten_24/ReshapeReshape&model_24/dropout_414/Identity:output:0"model_24/flatten_24/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
model_24/flatten_24/ReshapeЪ
)model_24/dense_1254/MatMul/ReadVariableOpReadVariableOp2model_24_dense_1254_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02+
)model_24/dense_1254/MatMul/ReadVariableOpЭ
model_24/dense_1254/MatMulMatMul$model_24/flatten_24/Reshape:output:01model_24/dense_1254/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_24/dense_1254/MatMulШ
*model_24/dense_1254/BiasAdd/ReadVariableOpReadVariableOp3model_24_dense_1254_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02,
*model_24/dense_1254/BiasAdd/ReadVariableOpб
model_24/dense_1254/BiasAddBiasAdd$model_24/dense_1254/MatMul:product:02model_24/dense_1254/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_24/dense_1254/BiasAdd
model_24/dense_1254/ReluRelu$model_24/dense_1254/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_24/dense_1254/ReluЄ
model_24/dropout_415/IdentityIdentity&model_24/dense_1254/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_24/dropout_415/IdentityЄ
model_24/dropout_416/IdentityIdentity&model_24/dropout_415/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_24/dropout_416/IdentityН
%model_24/output/MatMul/ReadVariableOpReadVariableOp.model_24_output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02'
%model_24/output/MatMul/ReadVariableOpУ
model_24/output/MatMulMatMul&model_24/dropout_416/Identity:output:0-model_24/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_24/output/MatMulМ
&model_24/output/BiasAdd/ReadVariableOpReadVariableOp/model_24_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_24/output/BiasAdd/ReadVariableOpС
model_24/output/BiasAddBiasAdd model_24/output/MatMul:product:0.model_24/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_24/output/BiasAddѓ
IdentityIdentity model_24/output/BiasAdd:output:0)^model_24/conv2d_6/BiasAdd/ReadVariableOp(^model_24/conv2d_6/Conv2D/ReadVariableOp+^model_24/dense_1254/BiasAdd/ReadVariableOp*^model_24/dense_1254/MatMul/ReadVariableOp'^model_24/output/BiasAdd/ReadVariableOp&^model_24/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2T
(model_24/conv2d_6/BiasAdd/ReadVariableOp(model_24/conv2d_6/BiasAdd/ReadVariableOp2R
'model_24/conv2d_6/Conv2D/ReadVariableOp'model_24/conv2d_6/Conv2D/ReadVariableOp2X
*model_24/dense_1254/BiasAdd/ReadVariableOp*model_24/dense_1254/BiasAdd/ReadVariableOp2V
)model_24/dense_1254/MatMul/ReadVariableOp)model_24/dense_1254/MatMul/ReadVariableOp2P
&model_24/output/BiasAdd/ReadVariableOp&model_24/output/BiasAdd/ReadVariableOp2N
%model_24/output/MatMul/ReadVariableOp%model_24/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
г
e
,__inference_dropout_416_layer_call_fn_787811

inputs
identityЂStatefulPartitionedCallт
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_416_layer_call_and_return_conditional_losses_7873282
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
ъ
b
F__inference_flatten_24_layer_call_and_return_conditional_losses_787233

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
В

ј
F__inference_dense_1254_layer_call_and_return_conditional_losses_787246

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
є
e
G__inference_dropout_415_layer_call_and_return_conditional_losses_787257

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
­
f
G__inference_dropout_415_layer_call_and_return_conditional_losses_787774

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
ы
H
,__inference_dropout_414_layer_call_fn_787721

inputs
identityг
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_414_layer_call_and_return_conditional_losses_7872252
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
m

"__inference__traced_restore_788013
file_prefix;
 assignvariableop_conv2d_6_kernel:Д/
 assignvariableop_1_conv2d_6_bias:	Д7
$assignvariableop_2_dense_1254_kernel:	ФP0
"assignvariableop_3_dense_1254_bias:P2
 assignvariableop_4_output_kernel:P,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: E
*assignvariableop_13_adam_conv2d_6_kernel_m:Д7
(assignvariableop_14_adam_conv2d_6_bias_m:	Д?
,assignvariableop_15_adam_dense_1254_kernel_m:	ФP8
*assignvariableop_16_adam_dense_1254_bias_m:P:
(assignvariableop_17_adam_output_kernel_m:P4
&assignvariableop_18_adam_output_bias_m:E
*assignvariableop_19_adam_conv2d_6_kernel_v:Д7
(assignvariableop_20_adam_conv2d_6_bias_v:	Д?
,assignvariableop_21_adam_dense_1254_kernel_v:	ФP8
*assignvariableop_22_adam_dense_1254_bias_v:P:
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

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Љ
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_1254_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ї
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_1254_biasIdentity_3:output:0"/device:CPU:0*
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
Identity_13В
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_conv2d_6_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14А
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_conv2d_6_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Д
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_1254_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16В
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_1254_bias_mIdentity_16:output:0"/device:CPU:0*
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
Identity_19В
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_6_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20А
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_6_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Д
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_1254_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22В
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_1254_bias_vIdentity_22:output:0"/device:CPU:0*
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
ѕ

)__inference_model_24_layer_call_fn_787298
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallГ
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
GPU2 *0J 8 *M
fHRF
D__inference_model_24_layer_call_and_return_conditional_losses_7872832
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
Ю	
ѓ
B__inference_output_layer_call_and_return_conditional_losses_787276

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
;
К

__inference__traced_save_787928
file_prefix.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop0
,savev2_dense_1254_kernel_read_readvariableop.
*savev2_dense_1254_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop7
3savev2_adam_dense_1254_kernel_m_read_readvariableop5
1savev2_adam_dense_1254_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop7
3savev2_adam_dense_1254_kernel_v_read_readvariableop5
1savev2_adam_dense_1254_bias_v_read_readvariableop3
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
SaveV2/shape_and_slicesН

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop,savev2_dense_1254_kernel_read_readvariableop*savev2_dense_1254_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop3savev2_adam_dense_1254_kernel_m_read_readvariableop1savev2_adam_dense_1254_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop3savev2_adam_dense_1254_kernel_v_read_readvariableop1savev2_adam_dense_1254_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
у

)__inference_model_24_layer_call_fn_787670

inputs"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCall­
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
GPU2 *0J 8 *M
fHRF
D__inference_model_24_layer_call_and_return_conditional_losses_7874522
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
­
f
G__inference_dropout_416_layer_call_and_return_conditional_losses_787801

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


'__inference_output_layer_call_fn_787830

inputs
unknown:P
	unknown_0:
identityЂStatefulPartitionedCallї
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
GPU2 *0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_7872762
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
В

ј
F__inference_dense_1254_layer_call_and_return_conditional_losses_787748

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
№&

D__inference_model_24_layer_call_and_return_conditional_losses_787532
input_onehot*
conv2d_6_787511:Д
conv2d_6_787513:	Д$
dense_1254_787519:	ФP
dense_1254_787521:P
output_787526:P
output_787528:
identityЂ conv2d_6/StatefulPartitionedCallЂ"dense_1254/StatefulPartitionedCallЂ#dropout_414/StatefulPartitionedCallЂ#dropout_415/StatefulPartitionedCallЂ#dropout_416/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЈ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_6_787511conv2d_6_787513*
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
GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_7872072"
 conv2d_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *L
fGRE
C__inference_re_lu_6_layer_call_and_return_conditional_losses_7872182
re_lu_6/PartitionedCall
#dropout_414/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_414_layer_call_and_return_conditional_losses_7873902%
#dropout_414/StatefulPartitionedCall
flatten_24/PartitionedCallPartitionedCall,dropout_414/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *O
fJRH
F__inference_flatten_24_layer_call_and_return_conditional_losses_7872332
flatten_24/PartitionedCallР
"dense_1254/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_1254_787519dense_1254_787521*
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
GPU2 *0J 8 *O
fJRH
F__inference_dense_1254_layer_call_and_return_conditional_losses_7872462$
"dense_1254/StatefulPartitionedCallХ
#dropout_415/StatefulPartitionedCallStatefulPartitionedCall+dense_1254/StatefulPartitionedCall:output:0$^dropout_414/StatefulPartitionedCall*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_415_layer_call_and_return_conditional_losses_7873512%
#dropout_415/StatefulPartitionedCallЦ
#dropout_416/StatefulPartitionedCallStatefulPartitionedCall,dropout_415/StatefulPartitionedCall:output:0$^dropout_415/StatefulPartitionedCall*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_416_layer_call_and_return_conditional_losses_7873282%
#dropout_416/StatefulPartitionedCallЕ
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_416/StatefulPartitionedCall:output:0output_787526output_787528*
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
GPU2 *0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_7872762 
output/StatefulPartitionedCallж
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall#^dense_1254/StatefulPartitionedCall$^dropout_414/StatefulPartitionedCall$^dropout_415/StatefulPartitionedCall$^dropout_416/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2H
"dense_1254/StatefulPartitionedCall"dense_1254/StatefulPartitionedCall2J
#dropout_414/StatefulPartitionedCall#dropout_414/StatefulPartitionedCall2J
#dropout_415/StatefulPartitionedCall#dropout_415/StatefulPartitionedCall2J
#dropout_416/StatefulPartitionedCall#dropout_416/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
Ю	
ѓ
B__inference_output_layer_call_and_return_conditional_losses_787821

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
џ
_
C__inference_re_lu_6_layer_call_and_return_conditional_losses_787218

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
ї
e
,__inference_dropout_414_layer_call_fn_787726

inputs
identityЂStatefulPartitionedCallы
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_414_layer_call_and_return_conditional_losses_7873902
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
й
G
+__inference_flatten_24_layer_call_fn_787737

inputs
identityЪ
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
GPU2 *0J 8 *O
fJRH
F__inference_flatten_24_layer_call_and_return_conditional_losses_7872332
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
Ч
H
,__inference_dropout_415_layer_call_fn_787779

inputs
identityЪ
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_415_layer_call_and_return_conditional_losses_7872572
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
И

џ
D__inference_conv2d_6_layer_call_and_return_conditional_losses_787207

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
ќ!
Ѓ
D__inference_model_24_layer_call_and_return_conditional_losses_787508
input_onehot*
conv2d_6_787487:Д
conv2d_6_787489:	Д$
dense_1254_787495:	ФP
dense_1254_787497:P
output_787502:P
output_787504:
identityЂ conv2d_6/StatefulPartitionedCallЂ"dense_1254/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЈ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_6_787487conv2d_6_787489*
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
GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_7872072"
 conv2d_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *L
fGRE
C__inference_re_lu_6_layer_call_and_return_conditional_losses_7872182
re_lu_6/PartitionedCall
dropout_414/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_414_layer_call_and_return_conditional_losses_7872252
dropout_414/PartitionedCallў
flatten_24/PartitionedCallPartitionedCall$dropout_414/PartitionedCall:output:0*
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
GPU2 *0J 8 *O
fJRH
F__inference_flatten_24_layer_call_and_return_conditional_losses_7872332
flatten_24/PartitionedCallР
"dense_1254/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_1254_787495dense_1254_787497*
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
GPU2 *0J 8 *O
fJRH
F__inference_dense_1254_layer_call_and_return_conditional_losses_7872462$
"dense_1254/StatefulPartitionedCall
dropout_415/PartitionedCallPartitionedCall+dense_1254/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_415_layer_call_and_return_conditional_losses_7872572
dropout_415/PartitionedCall
dropout_416/PartitionedCallPartitionedCall$dropout_415/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_416_layer_call_and_return_conditional_losses_7872642
dropout_416/PartitionedCall­
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_416/PartitionedCall:output:0output_787502output_787504*
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
GPU2 *0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_7872762 
output/StatefulPartitionedCallф
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall#^dense_1254/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2H
"dense_1254/StatefulPartitionedCall"dense_1254/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot

e
G__inference_dropout_414_layer_call_and_return_conditional_losses_787225

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
­
f
G__inference_dropout_416_layer_call_and_return_conditional_losses_787328

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
є
e
G__inference_dropout_416_layer_call_and_return_conditional_losses_787264

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

e
G__inference_dropout_414_layer_call_and_return_conditional_losses_787704

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
г
e
,__inference_dropout_415_layer_call_fn_787784

inputs
identityЂStatefulPartitionedCallт
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_415_layer_call_and_return_conditional_losses_7873512
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
­
f
G__inference_dropout_415_layer_call_and_return_conditional_losses_787351

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
ЖA

D__inference_model_24_layer_call_and_return_conditional_losses_787636

inputsB
'conv2d_6_conv2d_readvariableop_resource:Д7
(conv2d_6_biasadd_readvariableop_resource:	Д<
)dense_1254_matmul_readvariableop_resource:	ФP8
*dense_1254_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityЂconv2d_6/BiasAdd/ReadVariableOpЂconv2d_6/Conv2D/ReadVariableOpЂ!dense_1254/BiasAdd/ReadVariableOpЂ dense_1254/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpБ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02 
conv2d_6/Conv2D/ReadVariableOpР
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
conv2d_6/Conv2DЈ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
conv2d_6/BiasAddz
re_lu_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
re_lu_6/Relu{
dropout_414/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_414/dropout/ConstД
dropout_414/dropout/MulMulre_lu_6/Relu:activations:0"dropout_414/dropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_414/dropout/Mul
dropout_414/dropout/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
dropout_414/dropout/Shapeс
0dropout_414/dropout/random_uniform/RandomUniformRandomUniform"dropout_414/dropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџД*
dtype022
0dropout_414/dropout/random_uniform/RandomUniform
"dropout_414/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_414/dropout/GreaterEqual/yї
 dropout_414/dropout/GreaterEqualGreaterEqual9dropout_414/dropout/random_uniform/RandomUniform:output:0+dropout_414/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2"
 dropout_414/dropout/GreaterEqualЌ
dropout_414/dropout/CastCast$dropout_414/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџД2
dropout_414/dropout/CastГ
dropout_414/dropout/Mul_1Muldropout_414/dropout/Mul:z:0dropout_414/dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_414/dropout/Mul_1u
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
flatten_24/Const 
flatten_24/ReshapeReshapedropout_414/dropout/Mul_1:z:0flatten_24/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
flatten_24/ReshapeЏ
 dense_1254/MatMul/ReadVariableOpReadVariableOp)dense_1254_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02"
 dense_1254/MatMul/ReadVariableOpЉ
dense_1254/MatMulMatMulflatten_24/Reshape:output:0(dense_1254/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_1254/MatMul­
!dense_1254/BiasAdd/ReadVariableOpReadVariableOp*dense_1254_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02#
!dense_1254/BiasAdd/ReadVariableOp­
dense_1254/BiasAddBiasAdddense_1254/MatMul:product:0)dense_1254/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_1254/BiasAddy
dense_1254/ReluReludense_1254/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_1254/Relu{
dropout_415/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_415/dropout/ConstЎ
dropout_415/dropout/MulMuldense_1254/Relu:activations:0"dropout_415/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_415/dropout/Mul
dropout_415/dropout/ShapeShapedense_1254/Relu:activations:0*
T0*
_output_shapes
:2
dropout_415/dropout/Shapeи
0dropout_415/dropout/random_uniform/RandomUniformRandomUniform"dropout_415/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype022
0dropout_415/dropout/random_uniform/RandomUniform
"dropout_415/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_415/dropout/GreaterEqual/yю
 dropout_415/dropout/GreaterEqualGreaterEqual9dropout_415/dropout/random_uniform/RandomUniform:output:0+dropout_415/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2"
 dropout_415/dropout/GreaterEqualЃ
dropout_415/dropout/CastCast$dropout_415/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout_415/dropout/CastЊ
dropout_415/dropout/Mul_1Muldropout_415/dropout/Mul:z:0dropout_415/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_415/dropout/Mul_1{
dropout_416/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_416/dropout/ConstЎ
dropout_416/dropout/MulMuldropout_415/dropout/Mul_1:z:0"dropout_416/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_416/dropout/Mul
dropout_416/dropout/ShapeShapedropout_415/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dropout_416/dropout/Shapeи
0dropout_416/dropout/random_uniform/RandomUniformRandomUniform"dropout_416/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype022
0dropout_416/dropout/random_uniform/RandomUniform
"dropout_416/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_416/dropout/GreaterEqual/yю
 dropout_416/dropout/GreaterEqualGreaterEqual9dropout_416/dropout/random_uniform/RandomUniform:output:0+dropout_416/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2"
 dropout_416/dropout/GreaterEqualЃ
dropout_416/dropout/CastCast$dropout_416/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout_416/dropout/CastЊ
dropout_416/dropout/Mul_1Muldropout_416/dropout/Mul:z:0dropout_416/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_416/dropout/Mul_1Ђ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_416/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAddД
IdentityIdentityoutput/BiasAdd:output:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp"^dense_1254/BiasAdd/ReadVariableOp!^dense_1254/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2F
!dense_1254/BiasAdd/ReadVariableOp!dense_1254/BiasAdd/ReadVariableOp2D
 dense_1254/MatMul/ReadVariableOp dense_1254/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
 
)__inference_conv2d_6_layer_call_fn_787689

inputs"
unknown:Д
	unknown_0:	Д
identityЂStatefulPartitionedCall
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
GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_7872072
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
є
e
G__inference_dropout_415_layer_call_and_return_conditional_losses_787762

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
у

)__inference_model_24_layer_call_fn_787653

inputs"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCall­
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
GPU2 *0J 8 *M
fHRF
D__inference_model_24_layer_call_and_return_conditional_losses_7872832
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
ъ!

D__inference_model_24_layer_call_and_return_conditional_losses_787283

inputs*
conv2d_6_787208:Д
conv2d_6_787210:	Д$
dense_1254_787247:	ФP
dense_1254_787249:P
output_787277:P
output_787279:
identityЂ conv2d_6/StatefulPartitionedCallЂ"dense_1254/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЂ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_787208conv2d_6_787210*
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
GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_7872072"
 conv2d_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *L
fGRE
C__inference_re_lu_6_layer_call_and_return_conditional_losses_7872182
re_lu_6/PartitionedCall
dropout_414/PartitionedCallPartitionedCall re_lu_6/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_414_layer_call_and_return_conditional_losses_7872252
dropout_414/PartitionedCallў
flatten_24/PartitionedCallPartitionedCall$dropout_414/PartitionedCall:output:0*
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
GPU2 *0J 8 *O
fJRH
F__inference_flatten_24_layer_call_and_return_conditional_losses_7872332
flatten_24/PartitionedCallР
"dense_1254/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_1254_787247dense_1254_787249*
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
GPU2 *0J 8 *O
fJRH
F__inference_dense_1254_layer_call_and_return_conditional_losses_7872462$
"dense_1254/StatefulPartitionedCall
dropout_415/PartitionedCallPartitionedCall+dense_1254/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_415_layer_call_and_return_conditional_losses_7872572
dropout_415/PartitionedCall
dropout_416/PartitionedCallPartitionedCall$dropout_415/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_416_layer_call_and_return_conditional_losses_7872642
dropout_416/PartitionedCall­
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_416/PartitionedCall:output:0output_787277output_787279*
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
GPU2 *0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_7872762 
output/StatefulPartitionedCallф
IdentityIdentity'output/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall#^dense_1254/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2H
"dense_1254/StatefulPartitionedCall"dense_1254/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
b
F__inference_flatten_24_layer_call_and_return_conditional_losses_787732

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
Э

$__inference_signature_wrapper_787557
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCall
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
GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_7871902
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
ѕ

)__inference_model_24_layer_call_fn_787484
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallГ
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
GPU2 *0J 8 *M
fHRF
D__inference_model_24_layer_call_and_return_conditional_losses_7874522
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
џ
_
C__inference_re_lu_6_layer_call_and_return_conditional_losses_787694

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
у
D
(__inference_re_lu_6_layer_call_fn_787699

inputs
identityЯ
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
GPU2 *0J 8 *L
fGRE
C__inference_re_lu_6_layer_call_and_return_conditional_losses_7872182
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
є
e
G__inference_dropout_416_layer_call_and_return_conditional_losses_787789

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
Ч
H
,__inference_dropout_416_layer_call_fn_787806

inputs
identityЪ
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
GPU2 *0J 8 *P
fKRI
G__inference_dropout_416_layer_call_and_return_conditional_losses_7872642
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
ї
f
G__inference_dropout_414_layer_call_and_return_conditional_losses_787390

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
И

џ
D__inference_conv2d_6_layer_call_and_return_conditional_losses_787680

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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Йџ
@
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*y&call_and_return_all_conditional_losses
z__call__
{_default_save_signature"=
_tf_keras_networkю<{"name": "model_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_414", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_414", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_24", "inbound_nodes": [[["dropout_414", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1254", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1254", "inbound_nodes": [[["flatten_24", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_415", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_415", "inbound_nodes": [[["dense_1254", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_416", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_416", "inbound_nodes": [[["dropout_415", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_416", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_414", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_414", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_24", "inbound_nodes": [[["dropout_414", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_1254", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1254", "inbound_nodes": [[["flatten_24", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_415", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_415", "inbound_nodes": [[["dense_1254", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_416", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_416", "inbound_nodes": [[["dropout_415", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_416", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"ў
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*|&call_and_return_all_conditional_losses
}__call__"м	
_tf_keras_layerТ	{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}}
Џ
regularization_losses
trainable_variables
	variables
	keras_api
*~&call_and_return_all_conditional_losses
__call__" 
_tf_keras_layer{"name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv2d_6", 0, 0, {}]]], "shared_object_id": 4}
Ў
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"name": "dropout_414", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_414", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["re_lu_6", 0, 0, {}]]], "shared_object_id": 5}
Щ
regularization_losses
trainable_variables
 	variables
!	keras_api
+&call_and_return_all_conditional_losses
__call__"И
_tf_keras_layer{"name": "flatten_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["dropout_414", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
	

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"с
_tf_keras_layerЧ{"name": "dense_1254", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1254", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_24", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3780}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3780]}}
В
(regularization_losses
)trainable_variables
*	variables
+	keras_api
+&call_and_return_all_conditional_losses
__call__"Ё
_tf_keras_layer{"name": "dropout_415", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_415", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_1254", 0, 0, {}]]], "shared_object_id": 10}
Г
,regularization_losses
-trainable_variables
.	variables
/	keras_api
+&call_and_return_all_conditional_losses
__call__"Ђ
_tf_keras_layer{"name": "dropout_416", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_416", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dropout_415", 0, 0, {}]]], "shared_object_id": 11}
	

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+&call_and_return_all_conditional_losses
__call__"к
_tf_keras_layerР{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_416", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
П
6iter

7beta_1

8beta_2
	9decay
:learning_ratemmmn"mo#mp0mq1mrvsvt"vu#vv0vw1vx"
	optimizer
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
J
0
1
"2
#3
04
15"
trackable_list_wrapper
Ъ
regularization_losses
;non_trainable_variables
<layer_regularization_losses
=metrics
trainable_variables
>layer_metrics
	variables

?layers
z__call__
{_default_save_signature
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
*:(Д2conv2d_6/kernel
:Д2conv2d_6/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
@non_trainable_variables
Alayer_regularization_losses
Bmetrics
trainable_variables
Clayer_metrics
	variables

Dlayers
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
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
Gmetrics
trainable_variables
Hlayer_metrics
	variables

Ilayers
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
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
Lmetrics
trainable_variables
Mlayer_metrics
	variables

Nlayers
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
regularization_losses
Onon_trainable_variables
Player_regularization_losses
Qmetrics
trainable_variables
Rlayer_metrics
 	variables

Slayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
$:"	ФP2dense_1254/kernel
:P2dense_1254/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
А
$regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
Vmetrics
%trainable_variables
Wlayer_metrics
&	variables

Xlayers
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
(regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
[metrics
)trainable_variables
\layer_metrics
*	variables

]layers
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
,regularization_losses
^non_trainable_variables
_layer_regularization_losses
`metrics
-trainable_variables
alayer_metrics
.	variables

blayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:P2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
А
2regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
emetrics
3trainable_variables
flayer_metrics
4	variables

glayers
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
trackable_list_wrapper
 "
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_dict_wrapper
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
/:-Д2Adam/conv2d_6/kernel/m
!:Д2Adam/conv2d_6/bias/m
):'	ФP2Adam/dense_1254/kernel/m
": P2Adam/dense_1254/bias/m
$:"P2Adam/output/kernel/m
:2Adam/output/bias/m
/:-Д2Adam/conv2d_6/kernel/v
!:Д2Adam/conv2d_6/bias/v
):'	ФP2Adam/dense_1254/kernel/v
": P2Adam/dense_1254/bias/v
$:"P2Adam/output/kernel/v
:2Adam/output/bias/v
о2л
D__inference_model_24_layer_call_and_return_conditional_losses_787586
D__inference_model_24_layer_call_and_return_conditional_losses_787636
D__inference_model_24_layer_call_and_return_conditional_losses_787508
D__inference_model_24_layer_call_and_return_conditional_losses_787532Р
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
ђ2я
)__inference_model_24_layer_call_fn_787298
)__inference_model_24_layer_call_fn_787653
)__inference_model_24_layer_call_fn_787670
)__inference_model_24_layer_call_fn_787484Р
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
ь2щ
!__inference__wrapped_model_787190У
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
ю2ы
D__inference_conv2d_6_layer_call_and_return_conditional_losses_787680Ђ
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
г2а
)__inference_conv2d_6_layer_call_fn_787689Ђ
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
э2ъ
C__inference_re_lu_6_layer_call_and_return_conditional_losses_787694Ђ
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
(__inference_re_lu_6_layer_call_fn_787699Ђ
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
Ь2Щ
G__inference_dropout_414_layer_call_and_return_conditional_losses_787704
G__inference_dropout_414_layer_call_and_return_conditional_losses_787716Д
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
2
,__inference_dropout_414_layer_call_fn_787721
,__inference_dropout_414_layer_call_fn_787726Д
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
№2э
F__inference_flatten_24_layer_call_and_return_conditional_losses_787732Ђ
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
+__inference_flatten_24_layer_call_fn_787737Ђ
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
№2э
F__inference_dense_1254_layer_call_and_return_conditional_losses_787748Ђ
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
+__inference_dense_1254_layer_call_fn_787757Ђ
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
Ь2Щ
G__inference_dropout_415_layer_call_and_return_conditional_losses_787762
G__inference_dropout_415_layer_call_and_return_conditional_losses_787774Д
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
2
,__inference_dropout_415_layer_call_fn_787779
,__inference_dropout_415_layer_call_fn_787784Д
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
Ь2Щ
G__inference_dropout_416_layer_call_and_return_conditional_losses_787789
G__inference_dropout_416_layer_call_and_return_conditional_losses_787801Д
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
2
,__inference_dropout_416_layer_call_fn_787806
,__inference_dropout_416_layer_call_fn_787811Д
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
ь2щ
B__inference_output_layer_call_and_return_conditional_losses_787821Ђ
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
б2Ю
'__inference_output_layer_call_fn_787830Ђ
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
аBЭ
$__inference_signature_wrapper_787557input_onehot"
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
 
!__inference__wrapped_model_787190x"#01=Ђ:
3Ђ0
.+
input_onehotџџџџџџџџџ
Њ "/Њ,
*
output 
outputџџџџџџџџџЕ
D__inference_conv2d_6_layer_call_and_return_conditional_losses_787680m7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџД
 
)__inference_conv2d_6_layer_call_fn_787689`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "!џџџџџџџџџДЇ
F__inference_dense_1254_layer_call_and_return_conditional_losses_787748]"#0Ђ-
&Ђ#
!
inputsџџџџџџџџџФ
Њ "%Ђ"

0џџџџџџџџџP
 
+__inference_dense_1254_layer_call_fn_787757P"#0Ђ-
&Ђ#
!
inputsџџџџџџџџџФ
Њ "џџџџџџџџџPЙ
G__inference_dropout_414_layer_call_and_return_conditional_losses_787704n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p 
Њ ".Ђ+
$!
0џџџџџџџџџД
 Й
G__inference_dropout_414_layer_call_and_return_conditional_losses_787716n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p
Њ ".Ђ+
$!
0џџџџџџџџџД
 
,__inference_dropout_414_layer_call_fn_787721a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p 
Њ "!џџџџџџџџџД
,__inference_dropout_414_layer_call_fn_787726a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p
Њ "!џџџџџџџџџДЇ
G__inference_dropout_415_layer_call_and_return_conditional_losses_787762\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "%Ђ"

0џџџџџџџџџP
 Ї
G__inference_dropout_415_layer_call_and_return_conditional_losses_787774\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "%Ђ"

0џџџџџџџџџP
 
,__inference_dropout_415_layer_call_fn_787779O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "џџџџџџџџџP
,__inference_dropout_415_layer_call_fn_787784O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "џџџџџџџџџPЇ
G__inference_dropout_416_layer_call_and_return_conditional_losses_787789\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "%Ђ"

0џџџџџџџџџP
 Ї
G__inference_dropout_416_layer_call_and_return_conditional_losses_787801\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "%Ђ"

0џџџџџџџџџP
 
,__inference_dropout_416_layer_call_fn_787806O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "џџџџџџџџџP
,__inference_dropout_416_layer_call_fn_787811O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "џџџџџџџџџPЌ
F__inference_flatten_24_layer_call_and_return_conditional_losses_787732b8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "&Ђ#

0џџџџџџџџџФ
 
+__inference_flatten_24_layer_call_fn_787737U8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "џџџџџџџџџФО
D__inference_model_24_layer_call_and_return_conditional_losses_787508v"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 О
D__inference_model_24_layer_call_and_return_conditional_losses_787532v"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 И
D__inference_model_24_layer_call_and_return_conditional_losses_787586p"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 И
D__inference_model_24_layer_call_and_return_conditional_losses_787636p"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
)__inference_model_24_layer_call_fn_787298i"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
)__inference_model_24_layer_call_fn_787484i"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
)__inference_model_24_layer_call_fn_787653c"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
)__inference_model_24_layer_call_fn_787670c"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЂ
B__inference_output_layer_call_and_return_conditional_losses_787821\01/Ђ,
%Ђ"
 
inputsџџџџџџџџџP
Њ "%Ђ"

0џџџџџџџџџ
 z
'__inference_output_layer_call_fn_787830O01/Ђ,
%Ђ"
 
inputsџџџџџџџџџP
Њ "џџџџџџџџџБ
C__inference_re_lu_6_layer_call_and_return_conditional_losses_787694j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ ".Ђ+
$!
0џџџџџџџџџД
 
(__inference_re_lu_6_layer_call_fn_787699]8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "!џџџџџџџџџДБ
$__inference_signature_wrapper_787557"#01MЂJ
Ђ 
CЊ@
>
input_onehot.+
input_onehotџџџџџџџџџ"/Њ,
*
output 
outputџџџџџџџџџ