Уш
Ѓю
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718йг
Е
conv2d_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*!
shared_nameconv2d_70/kernel
~
$conv2d_70/kernel/Read/ReadVariableOpReadVariableOpconv2d_70/kernel*'
_output_shapes
:і*
dtype0
u
conv2d_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*
shared_nameconv2d_70/bias
n
"conv2d_70/bias/Read/ReadVariableOpReadVariableOpconv2d_70/bias*
_output_shapes	
:і*
dtype0
Б
dense_14630/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ƒP*#
shared_namedense_14630/kernel
z
&dense_14630/kernel/Read/ReadVariableOpReadVariableOpdense_14630/kernel*
_output_shapes
:	ƒP*
dtype0
x
dense_14630/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*!
shared_namedense_14630/bias
q
$dense_14630/bias/Read/ReadVariableOpReadVariableOpdense_14630/bias*
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
У
Adam/conv2d_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*(
shared_nameAdam/conv2d_70/kernel/m
М
+Adam/conv2d_70/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/kernel/m*'
_output_shapes
:і*
dtype0
Г
Adam/conv2d_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*&
shared_nameAdam/conv2d_70/bias/m
|
)Adam/conv2d_70/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/bias/m*
_output_shapes	
:і*
dtype0
П
Adam/dense_14630/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ƒP**
shared_nameAdam/dense_14630/kernel/m
И
-Adam/dense_14630/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14630/kernel/m*
_output_shapes
:	ƒP*
dtype0
Ж
Adam/dense_14630/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*(
shared_nameAdam/dense_14630/bias/m

+Adam/dense_14630/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14630/bias/m*
_output_shapes
:P*
dtype0
Д
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
У
Adam/conv2d_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*(
shared_nameAdam/conv2d_70/kernel/v
М
+Adam/conv2d_70/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/kernel/v*'
_output_shapes
:і*
dtype0
Г
Adam/conv2d_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*&
shared_nameAdam/conv2d_70/bias/v
|
)Adam/conv2d_70/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/bias/v*
_output_shapes	
:і*
dtype0
П
Adam/dense_14630/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ƒP**
shared_nameAdam/dense_14630/kernel/v
И
-Adam/dense_14630/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14630/kernel/v*
_output_shapes
:	ƒP*
dtype0
Ж
Adam/dense_14630/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*(
shared_nameAdam/dense_14630/bias/v

+Adam/dense_14630/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14630/bias/v*
_output_shapes
:P*
dtype0
Д
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
т.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≠.
value£.B†. BЩ.
і
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
ђ
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
≠
regularization_losses

;layers
<metrics
trainable_variables
=layer_metrics
>non_trainable_variables
?layer_regularization_losses
	variables
 
\Z
VARIABLE_VALUEconv2d_70/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_70/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠

@layers
Ametrics
regularization_losses
trainable_variables
Blayer_metrics
Cnon_trainable_variables
Dlayer_regularization_losses
	variables
 
 
 
≠

Elayers
Fmetrics
regularization_losses
trainable_variables
Glayer_metrics
Hnon_trainable_variables
Ilayer_regularization_losses
	variables
 
 
 
≠

Jlayers
Kmetrics
regularization_losses
trainable_variables
Llayer_metrics
Mnon_trainable_variables
Nlayer_regularization_losses
	variables
 
 
 
≠

Olayers
Pmetrics
regularization_losses
trainable_variables
Qlayer_metrics
Rnon_trainable_variables
Slayer_regularization_losses
 	variables
^\
VARIABLE_VALUEdense_14630/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_14630/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
≠

Tlayers
Umetrics
$regularization_losses
%trainable_variables
Vlayer_metrics
Wnon_trainable_variables
Xlayer_regularization_losses
&	variables
 
 
 
≠

Ylayers
Zmetrics
(regularization_losses
)trainable_variables
[layer_metrics
\non_trainable_variables
]layer_regularization_losses
*	variables
 
 
 
≠

^layers
_metrics
,regularization_losses
-trainable_variables
`layer_metrics
anon_trainable_variables
blayer_regularization_losses
.	variables
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
≠

clayers
dmetrics
2regularization_losses
3trainable_variables
elayer_metrics
fnon_trainable_variables
glayer_regularization_losses
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

h0
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
VARIABLE_VALUEAdam/conv2d_70/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_70/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/dense_14630/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_14630/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_70/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_70/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/dense_14630/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_14630/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
П
serving_default_input_onehotPlaceholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
≠
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_70/kernelconv2d_70/biasdense_14630/kerneldense_14630/biasoutput/kerneloutput/bias*
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
GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_4709088
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Б

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_70/kernel/Read/ReadVariableOp"conv2d_70/bias/Read/ReadVariableOp&dense_14630/kernel/Read/ReadVariableOp$dense_14630/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_70/kernel/m/Read/ReadVariableOp)Adam/conv2d_70/bias/m/Read/ReadVariableOp-Adam/dense_14630/kernel/m/Read/ReadVariableOp+Adam/dense_14630/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv2d_70/kernel/v/Read/ReadVariableOp)Adam/conv2d_70/bias/v/Read/ReadVariableOp-Adam/dense_14630/kernel/v/Read/ReadVariableOp+Adam/dense_14630/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
 __inference__traced_save_4709459
И
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_70/kernelconv2d_70/biasdense_14630/kerneldense_14630/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_70/kernel/mAdam/conv2d_70/bias/mAdam/dense_14630/kernel/mAdam/dense_14630/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_70/kernel/vAdam/conv2d_70/bias/vAdam/dense_14630/kernel/vAdam/dense_14630/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
#__inference__traced_restore_4709544оп
ѕ
П
%__inference_signature_wrapper_4709088
input_onehot"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallС
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
GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_47087212
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
щ
h
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4708921

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€і*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€і2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
„$
Ъ
F__inference_model_280_layer_call_and_return_conditional_losses_4709151

inputsC
(conv2d_70_conv2d_readvariableop_resource:і8
)conv2d_70_biasadd_readvariableop_resource:	і=
*dense_14630_matmul_readvariableop_resource:	ƒP9
+dense_14630_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityИҐ conv2d_70/BiasAdd/ReadVariableOpҐconv2d_70/Conv2D/ReadVariableOpҐ"dense_14630/BiasAdd/ReadVariableOpҐ!dense_14630/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOpі
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02!
conv2d_70/Conv2D/ReadVariableOp√
conv2d_70/Conv2DConv2Dinputs'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
conv2d_70/Conv2DЂ
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02"
 conv2d_70/BiasAdd/ReadVariableOp±
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2
conv2d_70/BiasAdd}
re_lu_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
re_lu_70/ReluТ
dropout_4830/IdentityIdentityre_lu_70/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout_4830/Identityw
flatten_280/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
flatten_280/Const§
flatten_280/ReshapeReshapedropout_4830/Identity:output:0flatten_280/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2
flatten_280/Reshape≤
!dense_14630/MatMul/ReadVariableOpReadVariableOp*dense_14630_matmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02#
!dense_14630/MatMul/ReadVariableOp≠
dense_14630/MatMulMatMulflatten_280/Reshape:output:0)dense_14630/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_14630/MatMul∞
"dense_14630/BiasAdd/ReadVariableOpReadVariableOp+dense_14630_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02$
"dense_14630/BiasAdd/ReadVariableOp±
dense_14630/BiasAddBiasAdddense_14630/MatMul:product:0*dense_14630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_14630/BiasAdd|
dense_14630/ReluReludense_14630/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_14630/ReluМ
dropout_4831/IdentityIdentitydense_14630/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_4831/IdentityМ
dropout_4832/IdentityIdentitydropout_4831/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_4832/IdentityҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp†
output/MatMulMatMuldropout_4832/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAddЄ
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp#^dense_14630/BiasAdd/ReadVariableOp"^dense_14630/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2H
"dense_14630/BiasAdd/ReadVariableOp"dense_14630/BiasAdd/ReadVariableOp2F
!dense_14630/MatMul/ReadVariableOp!dense_14630/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ї

Б
F__inference_conv2d_70_layer_call_and_return_conditional_losses_4709220

inputs9
conv2d_readvariableop_resource:і.
biasadd_readvariableop_resource:	і
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
±;
«

 __inference__traced_save_4709459
file_prefix/
+savev2_conv2d_70_kernel_read_readvariableop-
)savev2_conv2d_70_bias_read_readvariableop1
-savev2_dense_14630_kernel_read_readvariableop/
+savev2_dense_14630_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_70_kernel_m_read_readvariableop4
0savev2_adam_conv2d_70_bias_m_read_readvariableop8
4savev2_adam_dense_14630_kernel_m_read_readvariableop6
2savev2_adam_dense_14630_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv2d_70_kernel_v_read_readvariableop4
0savev2_adam_conv2d_70_bias_v_read_readvariableop8
4savev2_adam_dense_14630_kernel_v_read_readvariableop6
2savev2_adam_dense_14630_bias_v_read_readvariableop3
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
SaveV2/shape_and_slices…

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_70_kernel_read_readvariableop)savev2_conv2d_70_bias_read_readvariableop-savev2_dense_14630_kernel_read_readvariableop+savev2_dense_14630_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_70_kernel_m_read_readvariableop0savev2_adam_conv2d_70_bias_m_read_readvariableop4savev2_adam_dense_14630_kernel_m_read_readvariableop2savev2_adam_dense_14630_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv2d_70_kernel_v_read_readvariableop0savev2_adam_conv2d_70_bias_v_read_readvariableop4savev2_adam_dense_14630_kernel_v_read_readvariableop2savev2_adam_dense_14630_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*Ў
_input_shapes∆
√: :і:і:	ƒP:P:P:: : : : : : : :і:і:	ƒP:P:P::і:і:	ƒP:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:і:!

_output_shapes	
:і:%!

_output_shapes
:	ƒP: 
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
:і:!

_output_shapes	
:і:%!

_output_shapes
:	ƒP: 
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
:і:!

_output_shapes	
:і:%!

_output_shapes
:	ƒP: 
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
ѓ
h
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4709315

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
ѓ
h
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4708859

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
„
g
.__inference_dropout_4832_layer_call_fn_4709325

inputs
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4832_layer_call_and_return_conditional_losses_47088592
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
–
Ґ
+__inference_conv2d_70_layer_call_fn_4709210

inputs"
unknown:і
	unknown_0:	і
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_70_layer_call_and_return_conditional_losses_47087382
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ю
Х
(__inference_output_layer_call_fn_4709351

inputs
unknown:P
	unknown_0:
identityИҐStatefulPartitionedCallш
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
GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_47088072
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
ѓ
h
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4709342

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
з
П
+__inference_model_280_layer_call_fn_4709122

inputs"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallѓ
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
GPU2 *0J 8В *O
fJRH
F__inference_model_280_layer_call_and_return_conditional_losses_47089832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ё
I
-__inference_flatten_280_layer_call_fn_4709262

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ƒ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_flatten_280_layer_call_and_return_conditional_losses_47087642
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
ђ"
Ђ
F__inference_model_280_layer_call_and_return_conditional_losses_4708814

inputs,
conv2d_70_4708739:і 
conv2d_70_4708741:	і&
dense_14630_4708778:	ƒP!
dense_14630_4708780:P 
output_4708808:P
output_4708810:
identityИҐ!conv2d_70/StatefulPartitionedCallҐ#dense_14630/StatefulPartitionedCallҐoutput/StatefulPartitionedCall™
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_70_4708739conv2d_70_4708741*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_70_layer_call_and_return_conditional_losses_47087382#
!conv2d_70/StatefulPartitionedCallЗ
re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_re_lu_70_layer_call_and_return_conditional_losses_47087492
re_lu_70/PartitionedCallК
dropout_4830/PartitionedCallPartitionedCall!re_lu_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4830_layer_call_and_return_conditional_losses_47087562
dropout_4830/PartitionedCallГ
flatten_280/PartitionedCallPartitionedCall%dropout_4830/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ƒ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_flatten_280_layer_call_and_return_conditional_losses_47087642
flatten_280/PartitionedCall…
#dense_14630/StatefulPartitionedCallStatefulPartitionedCall$flatten_280/PartitionedCall:output:0dense_14630_4708778dense_14630_4708780*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dense_14630_layer_call_and_return_conditional_losses_47087772%
#dense_14630/StatefulPartitionedCallМ
dropout_4831/PartitionedCallPartitionedCall,dense_14630/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4831_layer_call_and_return_conditional_losses_47087882
dropout_4831/PartitionedCallЕ
dropout_4832/PartitionedCallPartitionedCall%dropout_4831/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4832_layer_call_and_return_conditional_losses_47087952
dropout_4832/PartitionedCall±
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_4832/PartitionedCall:output:0output_4708808output_4708810*
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
GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_47088072 
output/StatefulPartitionedCallж
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_70/StatefulPartitionedCall$^dense_14630/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2J
#dense_14630/StatefulPartitionedCall#dense_14630/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Њ"
±
F__inference_model_280_layer_call_and_return_conditional_losses_4709039
input_onehot,
conv2d_70_4709018:і 
conv2d_70_4709020:	і&
dense_14630_4709026:	ƒP!
dense_14630_4709028:P 
output_4709033:P
output_4709035:
identityИҐ!conv2d_70/StatefulPartitionedCallҐ#dense_14630/StatefulPartitionedCallҐoutput/StatefulPartitionedCall∞
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_70_4709018conv2d_70_4709020*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_70_layer_call_and_return_conditional_losses_47087382#
!conv2d_70/StatefulPartitionedCallЗ
re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_re_lu_70_layer_call_and_return_conditional_losses_47087492
re_lu_70/PartitionedCallК
dropout_4830/PartitionedCallPartitionedCall!re_lu_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4830_layer_call_and_return_conditional_losses_47087562
dropout_4830/PartitionedCallГ
flatten_280/PartitionedCallPartitionedCall%dropout_4830/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ƒ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_flatten_280_layer_call_and_return_conditional_losses_47087642
flatten_280/PartitionedCall…
#dense_14630/StatefulPartitionedCallStatefulPartitionedCall$flatten_280/PartitionedCall:output:0dense_14630_4709026dense_14630_4709028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dense_14630_layer_call_and_return_conditional_losses_47087772%
#dense_14630/StatefulPartitionedCallМ
dropout_4831/PartitionedCallPartitionedCall,dense_14630/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4831_layer_call_and_return_conditional_losses_47087882
dropout_4831/PartitionedCallЕ
dropout_4832/PartitionedCallPartitionedCall%dropout_4831/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4832_layer_call_and_return_conditional_losses_47087952
dropout_4832/PartitionedCall±
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_4832/PartitionedCall:output:0output_4709033output_4709035*
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
GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_47088072 
output/StatefulPartitionedCallж
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_70/StatefulPartitionedCall$^dense_14630/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2J
#dense_14630/StatefulPartitionedCall#dense_14630/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
ц
g
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4708795

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
ѓ
h
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4708882

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
п
J
.__inference_dropout_4830_layer_call_fn_4709235

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4830_layer_call_and_return_conditional_losses_47087562
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
м
d
H__inference_flatten_280_layer_call_and_return_conditional_losses_4709268

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
з
П
+__inference_model_280_layer_call_fn_4709105

inputs"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallѓ
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
GPU2 *0J 8В *O
fJRH
F__inference_model_280_layer_call_and_return_conditional_losses_47088142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞m
¶
#__inference__traced_restore_4709544
file_prefix<
!assignvariableop_conv2d_70_kernel:і0
!assignvariableop_1_conv2d_70_bias:	і8
%assignvariableop_2_dense_14630_kernel:	ƒP1
#assignvariableop_3_dense_14630_bias:P2
 assignvariableop_4_output_kernel:P,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: F
+assignvariableop_13_adam_conv2d_70_kernel_m:і8
)assignvariableop_14_adam_conv2d_70_bias_m:	і@
-assignvariableop_15_adam_dense_14630_kernel_m:	ƒP9
+assignvariableop_16_adam_dense_14630_bias_m:P:
(assignvariableop_17_adam_output_kernel_m:P4
&assignvariableop_18_adam_output_bias_m:F
+assignvariableop_19_adam_conv2d_70_kernel_v:і8
)assignvariableop_20_adam_conv2d_70_bias_v:	і@
-assignvariableop_21_adam_dense_14630_kernel_v:	ƒP9
+assignvariableop_22_adam_dense_14630_bias_v:P:
(assignvariableop_23_adam_output_kernel_v:P4
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

Identity†
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_70_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_70_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2™
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_14630_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3®
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_14630_biasIdentity_3:output:0"/device:CPU:0*
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
Identity_13≥
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_70_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14±
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_70_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15µ
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_dense_14630_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16≥
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_14630_bias_mIdentity_16:output:0"/device:CPU:0*
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
Identity_19≥
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_70_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_70_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21µ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_dense_14630_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22≥
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_dense_14630_bias_vIdentity_22:output:0"/device:CPU:0*
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
ѕ	
ф
C__inference_output_layer_call_and_return_conditional_losses_4708807

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
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
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
Б
a
E__inference_re_lu_70_layer_call_and_return_conditional_losses_4708749

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€і2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
щ
h
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4709257

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€і*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€і2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
Ћ
J
.__inference_dropout_4832_layer_call_fn_4709320

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4832_layer_call_and_return_conditional_losses_47087952
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
з
F
*__inference_re_lu_70_layer_call_fn_4709225

inputs
identity—
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_re_lu_70_layer_call_and_return_conditional_losses_47087492
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
щ
Х
+__inference_model_280_layer_call_fn_4708829
input_onehot"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallµ
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
GPU2 *0J 8В *O
fJRH
F__inference_model_280_layer_call_and_return_conditional_losses_47088142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
м
d
H__inference_flatten_280_layer_call_and_return_conditional_losses_4708764

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
„
g
.__inference_dropout_4831_layer_call_fn_4709298

inputs
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4831_layer_call_and_return_conditional_losses_47088822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
і

ъ
H__inference_dense_14630_layer_call_and_return_conditional_losses_4708777

inputs1
matmul_readvariableop_resource:	ƒP-
biasadd_readvariableop_resource:P
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ƒ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ƒ
 
_user_specified_nameinputs
Б
a
E__inference_re_lu_70_layer_call_and_return_conditional_losses_4709230

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€і2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
Ъ
g
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4709245

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€і2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
Ђ
Ы
-__inference_dense_14630_layer_call_fn_4709277

inputs
unknown:	ƒP
	unknown_0:P
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dense_14630_layer_call_and_return_conditional_losses_47087772
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ƒ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ƒ
 
_user_specified_nameinputs
ѕ	
ф
C__inference_output_layer_call_and_return_conditional_losses_4709361

inputs0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
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
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
Ѓ'
†
F__inference_model_280_layer_call_and_return_conditional_losses_4708983

inputs,
conv2d_70_4708962:і 
conv2d_70_4708964:	і&
dense_14630_4708970:	ƒP!
dense_14630_4708972:P 
output_4708977:P
output_4708979:
identityИҐ!conv2d_70/StatefulPartitionedCallҐ#dense_14630/StatefulPartitionedCallҐ$dropout_4830/StatefulPartitionedCallҐ$dropout_4831/StatefulPartitionedCallҐ$dropout_4832/StatefulPartitionedCallҐoutput/StatefulPartitionedCall™
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_70_4708962conv2d_70_4708964*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_70_layer_call_and_return_conditional_losses_47087382#
!conv2d_70/StatefulPartitionedCallЗ
re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_re_lu_70_layer_call_and_return_conditional_losses_47087492
re_lu_70/PartitionedCallҐ
$dropout_4830/StatefulPartitionedCallStatefulPartitionedCall!re_lu_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4830_layer_call_and_return_conditional_losses_47089212&
$dropout_4830/StatefulPartitionedCallЛ
flatten_280/PartitionedCallPartitionedCall-dropout_4830/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ƒ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_flatten_280_layer_call_and_return_conditional_losses_47087642
flatten_280/PartitionedCall…
#dense_14630/StatefulPartitionedCallStatefulPartitionedCall$flatten_280/PartitionedCall:output:0dense_14630_4708970dense_14630_4708972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dense_14630_layer_call_and_return_conditional_losses_47087772%
#dense_14630/StatefulPartitionedCallЋ
$dropout_4831/StatefulPartitionedCallStatefulPartitionedCall,dense_14630/StatefulPartitionedCall:output:0%^dropout_4830/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4831_layer_call_and_return_conditional_losses_47088822&
$dropout_4831/StatefulPartitionedCallћ
$dropout_4832/StatefulPartitionedCallStatefulPartitionedCall-dropout_4831/StatefulPartitionedCall:output:0%^dropout_4831/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4832_layer_call_and_return_conditional_losses_47088592&
$dropout_4832/StatefulPartitionedCallє
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_4832/StatefulPartitionedCall:output:0output_4708977output_4708979*
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
GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_47088072 
output/StatefulPartitionedCallџ
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_70/StatefulPartitionedCall$^dense_14630/StatefulPartitionedCall%^dropout_4830/StatefulPartitionedCall%^dropout_4831/StatefulPartitionedCall%^dropout_4832/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2J
#dense_14630/StatefulPartitionedCall#dense_14630/StatefulPartitionedCall2L
$dropout_4830/StatefulPartitionedCall$dropout_4830/StatefulPartitionedCall2L
$dropout_4831/StatefulPartitionedCall$dropout_4831/StatefulPartitionedCall2L
$dropout_4832/StatefulPartitionedCall$dropout_4832/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ц
g
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4709303

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
ц
g
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4709330

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
Ћ
J
.__inference_dropout_4831_layer_call_fn_4709293

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4831_layer_call_and_return_conditional_losses_47087882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
щ
Х
+__inference_model_280_layer_call_fn_4709015
input_onehot"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallµ
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
GPU2 *0J 8В *O
fJRH
F__inference_model_280_layer_call_and_return_conditional_losses_47089832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
Ъ
g
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4708756

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€і2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
ј'
¶
F__inference_model_280_layer_call_and_return_conditional_losses_4709063
input_onehot,
conv2d_70_4709042:і 
conv2d_70_4709044:	і&
dense_14630_4709050:	ƒP!
dense_14630_4709052:P 
output_4709057:P
output_4709059:
identityИҐ!conv2d_70/StatefulPartitionedCallҐ#dense_14630/StatefulPartitionedCallҐ$dropout_4830/StatefulPartitionedCallҐ$dropout_4831/StatefulPartitionedCallҐ$dropout_4832/StatefulPartitionedCallҐoutput/StatefulPartitionedCall∞
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_70_4709042conv2d_70_4709044*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_70_layer_call_and_return_conditional_losses_47087382#
!conv2d_70/StatefulPartitionedCallЗ
re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_re_lu_70_layer_call_and_return_conditional_losses_47087492
re_lu_70/PartitionedCallҐ
$dropout_4830/StatefulPartitionedCallStatefulPartitionedCall!re_lu_70/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4830_layer_call_and_return_conditional_losses_47089212&
$dropout_4830/StatefulPartitionedCallЛ
flatten_280/PartitionedCallPartitionedCall-dropout_4830/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ƒ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_flatten_280_layer_call_and_return_conditional_losses_47087642
flatten_280/PartitionedCall…
#dense_14630/StatefulPartitionedCallStatefulPartitionedCall$flatten_280/PartitionedCall:output:0dense_14630_4709050dense_14630_4709052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_dense_14630_layer_call_and_return_conditional_losses_47087772%
#dense_14630/StatefulPartitionedCallЋ
$dropout_4831/StatefulPartitionedCallStatefulPartitionedCall,dense_14630/StatefulPartitionedCall:output:0%^dropout_4830/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4831_layer_call_and_return_conditional_losses_47088822&
$dropout_4831/StatefulPartitionedCallћ
$dropout_4832/StatefulPartitionedCallStatefulPartitionedCall-dropout_4831/StatefulPartitionedCall:output:0%^dropout_4831/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€P* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4832_layer_call_and_return_conditional_losses_47088592&
$dropout_4832/StatefulPartitionedCallє
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_4832/StatefulPartitionedCall:output:0output_4709057output_4709059*
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
GPU2 *0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_47088072 
output/StatefulPartitionedCallџ
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_70/StatefulPartitionedCall$^dense_14630/StatefulPartitionedCall%^dropout_4830/StatefulPartitionedCall%^dropout_4831/StatefulPartitionedCall%^dropout_4832/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2J
#dense_14630/StatefulPartitionedCall#dense_14630/StatefulPartitionedCall2L
$dropout_4830/StatefulPartitionedCall$dropout_4830/StatefulPartitionedCall2L
$dropout_4831/StatefulPartitionedCall$dropout_4831/StatefulPartitionedCall2L
$dropout_4832/StatefulPartitionedCall$dropout_4832/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
ы
g
.__inference_dropout_4830_layer_call_fn_4709240

inputs
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_dropout_4830_layer_call_and_return_conditional_losses_47089212
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€і22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
Ї

Б
F__inference_conv2d_70_layer_call_and_return_conditional_losses_4708738

inputs9
conv2d_readvariableop_resource:і.
biasadd_readvariableop_resource:	і
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€і2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
к+
ф
"__inference__wrapped_model_4708721
input_onehotM
2model_280_conv2d_70_conv2d_readvariableop_resource:іB
3model_280_conv2d_70_biasadd_readvariableop_resource:	іG
4model_280_dense_14630_matmul_readvariableop_resource:	ƒPC
5model_280_dense_14630_biasadd_readvariableop_resource:PA
/model_280_output_matmul_readvariableop_resource:P>
0model_280_output_biasadd_readvariableop_resource:
identityИҐ*model_280/conv2d_70/BiasAdd/ReadVariableOpҐ)model_280/conv2d_70/Conv2D/ReadVariableOpҐ,model_280/dense_14630/BiasAdd/ReadVariableOpҐ+model_280/dense_14630/MatMul/ReadVariableOpҐ'model_280/output/BiasAdd/ReadVariableOpҐ&model_280/output/MatMul/ReadVariableOp“
)model_280/conv2d_70/Conv2D/ReadVariableOpReadVariableOp2model_280_conv2d_70_conv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02+
)model_280/conv2d_70/Conv2D/ReadVariableOpз
model_280/conv2d_70/Conv2DConv2Dinput_onehot1model_280/conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
model_280/conv2d_70/Conv2D…
*model_280/conv2d_70/BiasAdd/ReadVariableOpReadVariableOp3model_280_conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02,
*model_280/conv2d_70/BiasAdd/ReadVariableOpў
model_280/conv2d_70/BiasAddBiasAdd#model_280/conv2d_70/Conv2D:output:02model_280/conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2
model_280/conv2d_70/BiasAddЫ
model_280/re_lu_70/ReluRelu$model_280/conv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
model_280/re_lu_70/Relu∞
model_280/dropout_4830/IdentityIdentity%model_280/re_lu_70/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€і2!
model_280/dropout_4830/IdentityЛ
model_280/flatten_280/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
model_280/flatten_280/Constћ
model_280/flatten_280/ReshapeReshape(model_280/dropout_4830/Identity:output:0$model_280/flatten_280/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2
model_280/flatten_280/Reshape–
+model_280/dense_14630/MatMul/ReadVariableOpReadVariableOp4model_280_dense_14630_matmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02-
+model_280/dense_14630/MatMul/ReadVariableOp’
model_280/dense_14630/MatMulMatMul&model_280/flatten_280/Reshape:output:03model_280/dense_14630/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
model_280/dense_14630/MatMulќ
,model_280/dense_14630/BiasAdd/ReadVariableOpReadVariableOp5model_280_dense_14630_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02.
,model_280/dense_14630/BiasAdd/ReadVariableOpў
model_280/dense_14630/BiasAddBiasAdd&model_280/dense_14630/MatMul:product:04model_280/dense_14630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
model_280/dense_14630/BiasAddЪ
model_280/dense_14630/ReluRelu&model_280/dense_14630/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
model_280/dense_14630/Relu™
model_280/dropout_4831/IdentityIdentity(model_280/dense_14630/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€P2!
model_280/dropout_4831/Identity™
model_280/dropout_4832/IdentityIdentity(model_280/dropout_4831/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2!
model_280/dropout_4832/Identityј
&model_280/output/MatMul/ReadVariableOpReadVariableOp/model_280_output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02(
&model_280/output/MatMul/ReadVariableOp»
model_280/output/MatMulMatMul(model_280/dropout_4832/Identity:output:0.model_280/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_280/output/MatMulњ
'model_280/output/BiasAdd/ReadVariableOpReadVariableOp0model_280_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_280/output/BiasAdd/ReadVariableOp≈
model_280/output/BiasAddBiasAdd!model_280/output/MatMul:product:0/model_280/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_280/output/BiasAddю
IdentityIdentity!model_280/output/BiasAdd:output:0+^model_280/conv2d_70/BiasAdd/ReadVariableOp*^model_280/conv2d_70/Conv2D/ReadVariableOp-^model_280/dense_14630/BiasAdd/ReadVariableOp,^model_280/dense_14630/MatMul/ReadVariableOp(^model_280/output/BiasAdd/ReadVariableOp'^model_280/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2X
*model_280/conv2d_70/BiasAdd/ReadVariableOp*model_280/conv2d_70/BiasAdd/ReadVariableOp2V
)model_280/conv2d_70/Conv2D/ReadVariableOp)model_280/conv2d_70/Conv2D/ReadVariableOp2\
,model_280/dense_14630/BiasAdd/ReadVariableOp,model_280/dense_14630/BiasAdd/ReadVariableOp2Z
+model_280/dense_14630/MatMul/ReadVariableOp+model_280/dense_14630/MatMul/ReadVariableOp2R
'model_280/output/BiasAdd/ReadVariableOp'model_280/output/BiasAdd/ReadVariableOp2P
&model_280/output/MatMul/ReadVariableOp&model_280/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
ц
g
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4708788

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€P:O K
'
_output_shapes
:€€€€€€€€€P
 
_user_specified_nameinputs
і

ъ
H__inference_dense_14630_layer_call_and_return_conditional_losses_4709288

inputs1
matmul_readvariableop_resource:	ƒP-
biasadd_readvariableop_resource:P
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ƒ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ƒ
 
_user_specified_nameinputs
њB
Ъ
F__inference_model_280_layer_call_and_return_conditional_losses_4709201

inputsC
(conv2d_70_conv2d_readvariableop_resource:і8
)conv2d_70_biasadd_readvariableop_resource:	і=
*dense_14630_matmul_readvariableop_resource:	ƒP9
+dense_14630_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityИҐ conv2d_70/BiasAdd/ReadVariableOpҐconv2d_70/Conv2D/ReadVariableOpҐ"dense_14630/BiasAdd/ReadVariableOpҐ!dense_14630/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOpі
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02!
conv2d_70/Conv2D/ReadVariableOp√
conv2d_70/Conv2DConv2Dinputs'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
conv2d_70/Conv2DЂ
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02"
 conv2d_70/BiasAdd/ReadVariableOp±
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2
conv2d_70/BiasAdd}
re_lu_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
re_lu_70/Relu}
dropout_4830/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_4830/dropout/ConstЄ
dropout_4830/dropout/MulMulre_lu_70/Relu:activations:0#dropout_4830/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout_4830/dropout/MulГ
dropout_4830/dropout/ShapeShapere_lu_70/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4830/dropout/Shapeд
1dropout_4830/dropout/random_uniform/RandomUniformRandomUniform#dropout_4830/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€і*
dtype023
1dropout_4830/dropout/random_uniform/RandomUniformП
#dropout_4830/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#dropout_4830/dropout/GreaterEqual/yы
!dropout_4830/dropout/GreaterEqualGreaterEqual:dropout_4830/dropout/random_uniform/RandomUniform:output:0,dropout_4830/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2#
!dropout_4830/dropout/GreaterEqualѓ
dropout_4830/dropout/CastCast%dropout_4830/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€і2
dropout_4830/dropout/CastЈ
dropout_4830/dropout/Mul_1Muldropout_4830/dropout/Mul:z:0dropout_4830/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout_4830/dropout/Mul_1w
flatten_280/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
flatten_280/Const§
flatten_280/ReshapeReshapedropout_4830/dropout/Mul_1:z:0flatten_280/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2
flatten_280/Reshape≤
!dense_14630/MatMul/ReadVariableOpReadVariableOp*dense_14630_matmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02#
!dense_14630/MatMul/ReadVariableOp≠
dense_14630/MatMulMatMulflatten_280/Reshape:output:0)dense_14630/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_14630/MatMul∞
"dense_14630/BiasAdd/ReadVariableOpReadVariableOp+dense_14630_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02$
"dense_14630/BiasAdd/ReadVariableOp±
dense_14630/BiasAddBiasAdddense_14630/MatMul:product:0*dense_14630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_14630/BiasAdd|
dense_14630/ReluReludense_14630/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_14630/Relu}
dropout_4831/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_4831/dropout/Const≤
dropout_4831/dropout/MulMuldense_14630/Relu:activations:0#dropout_4831/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_4831/dropout/MulЖ
dropout_4831/dropout/ShapeShapedense_14630/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4831/dropout/Shapeџ
1dropout_4831/dropout/random_uniform/RandomUniformRandomUniform#dropout_4831/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype023
1dropout_4831/dropout/random_uniform/RandomUniformП
#dropout_4831/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#dropout_4831/dropout/GreaterEqual/yт
!dropout_4831/dropout/GreaterEqualGreaterEqual:dropout_4831/dropout/random_uniform/RandomUniform:output:0,dropout_4831/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2#
!dropout_4831/dropout/GreaterEqual¶
dropout_4831/dropout/CastCast%dropout_4831/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout_4831/dropout/CastЃ
dropout_4831/dropout/Mul_1Muldropout_4831/dropout/Mul:z:0dropout_4831/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_4831/dropout/Mul_1}
dropout_4832/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_4832/dropout/Const≤
dropout_4832/dropout/MulMuldropout_4831/dropout/Mul_1:z:0#dropout_4832/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_4832/dropout/MulЖ
dropout_4832/dropout/ShapeShapedropout_4831/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dropout_4832/dropout/Shapeџ
1dropout_4832/dropout/random_uniform/RandomUniformRandomUniform#dropout_4832/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype023
1dropout_4832/dropout/random_uniform/RandomUniformП
#dropout_4832/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#dropout_4832/dropout/GreaterEqual/yт
!dropout_4832/dropout/GreaterEqualGreaterEqual:dropout_4832/dropout/random_uniform/RandomUniform:output:0,dropout_4832/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2#
!dropout_4832/dropout/GreaterEqual¶
dropout_4832/dropout/CastCast%dropout_4832/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout_4832/dropout/CastЃ
dropout_4832/dropout/Mul_1Muldropout_4832/dropout/Mul:z:0dropout_4832/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_4832/dropout/Mul_1Ґ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp†
output/MatMulMatMuldropout_4832/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAddЄ
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp#^dense_14630/BiasAdd/ReadVariableOp"^dense_14630/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2H
"dense_14630/BiasAdd/ReadVariableOp"dense_14630/BiasAdd/ReadVariableOp2F
!dense_14630/MatMul/ReadVariableOp!dense_14630/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
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
serving_default_input_onehot:0€€€€€€€€€:
output0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ДБ
≈@
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
y__call__
z_default_save_signature
*{&call_and_return_all_conditional_losses"Ј=
_tf_keras_networkЫ={"name": "model_280", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_280", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_70", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_70", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_70", "inbound_nodes": [[["conv2d_70", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4830", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4830", "inbound_nodes": [[["re_lu_70", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_280", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_280", "inbound_nodes": [[["dropout_4830", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14630", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14630", "inbound_nodes": [[["flatten_280", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4831", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4831", "inbound_nodes": [[["dense_14630", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4832", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4832", "inbound_nodes": [[["dropout_4831", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_4832", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_280", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_70", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "ReLU", "config": {"name": "re_lu_70", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_70", "inbound_nodes": [[["conv2d_70", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_4830", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4830", "inbound_nodes": [[["re_lu_70", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_280", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_280", "inbound_nodes": [[["dropout_4830", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_14630", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14630", "inbound_nodes": [[["flatten_280", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_4831", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4831", "inbound_nodes": [[["dense_14630", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_4832", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_4832", "inbound_nodes": [[["dropout_4831", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_4832", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Б"ю
_tf_keras_input_layerё{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
Г

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
|__call__
*}&call_and_return_all_conditional_losses"ё	
_tf_keras_layerƒ	{"name": "conv2d_70", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}}
≤
regularization_losses
trainable_variables
	variables
	keras_api
~__call__
*&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"name": "re_lu_70", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_70", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv2d_70", 0, 0, {}]]], "shared_object_id": 4}
±
regularization_losses
trainable_variables
	variables
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"†
_tf_keras_layerЖ{"name": "dropout_4830", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4830", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["re_lu_70", 0, 0, {}]]], "shared_object_id": 5}
ћ
regularization_losses
trainable_variables
 	variables
!	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"ї
_tf_keras_layer°{"name": "flatten_280", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_280", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["dropout_4830", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
Л	

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"д
_tf_keras_layer {"name": "dense_14630", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14630", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_280", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3780}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3780]}}
µ
(regularization_losses
)trainable_variables
*	variables
+	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"§
_tf_keras_layerК{"name": "dropout_4831", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4831", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_14630", 0, 0, {}]]], "shared_object_id": 10}
ґ
,regularization_losses
-trainable_variables
.	variables
/	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"•
_tf_keras_layerЛ{"name": "dropout_4832", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4832", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dropout_4831", 0, 0, {}]]], "shared_object_id": 11}
В	

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"џ
_tf_keras_layerЅ{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_4832", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
њ
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
 
regularization_losses

;layers
<metrics
trainable_variables
=layer_metrics
>non_trainable_variables
?layer_regularization_losses
	variables
y__call__
z_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
-
Мserving_default"
signature_map
+:)і2conv2d_70/kernel
:і2conv2d_70/bias
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
≠

@layers
Ametrics
regularization_losses
trainable_variables
Blayer_metrics
Cnon_trainable_variables
Dlayer_regularization_losses
	variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠

Elayers
Fmetrics
regularization_losses
trainable_variables
Glayer_metrics
Hnon_trainable_variables
Ilayer_regularization_losses
	variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞

Jlayers
Kmetrics
regularization_losses
trainable_variables
Llayer_metrics
Mnon_trainable_variables
Nlayer_regularization_losses
	variables
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞

Olayers
Pmetrics
regularization_losses
trainable_variables
Qlayer_metrics
Rnon_trainable_variables
Slayer_regularization_losses
 	variables
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
%:#	ƒP2dense_14630/kernel
:P2dense_14630/bias
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
∞

Tlayers
Umetrics
$regularization_losses
%trainable_variables
Vlayer_metrics
Wnon_trainable_variables
Xlayer_regularization_losses
&	variables
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞

Ylayers
Zmetrics
(regularization_losses
)trainable_variables
[layer_metrics
\non_trainable_variables
]layer_regularization_losses
*	variables
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞

^layers
_metrics
,regularization_losses
-trainable_variables
`layer_metrics
anon_trainable_variables
blayer_regularization_losses
.	variables
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
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
∞

clayers
dmetrics
2regularization_losses
3trainable_variables
elayer_metrics
fnon_trainable_variables
glayer_regularization_losses
4	variables
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
'
h0"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
	itotal
	jcount
k	variables
l	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 21}
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
0:.і2Adam/conv2d_70/kernel/m
": і2Adam/conv2d_70/bias/m
*:(	ƒP2Adam/dense_14630/kernel/m
#:!P2Adam/dense_14630/bias/m
$:"P2Adam/output/kernel/m
:2Adam/output/bias/m
0:.і2Adam/conv2d_70/kernel/v
": і2Adam/conv2d_70/bias/v
*:(	ƒP2Adam/dense_14630/kernel/v
#:!P2Adam/dense_14630/bias/v
$:"P2Adam/output/kernel/v
:2Adam/output/bias/v
ъ2ч
+__inference_model_280_layer_call_fn_4708829
+__inference_model_280_layer_call_fn_4709105
+__inference_model_280_layer_call_fn_4709122
+__inference_model_280_layer_call_fn_4709015ј
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
н2к
"__inference__wrapped_model_4708721√
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
input_onehot€€€€€€€€€
ж2г
F__inference_model_280_layer_call_and_return_conditional_losses_4709151
F__inference_model_280_layer_call_and_return_conditional_losses_4709201
F__inference_model_280_layer_call_and_return_conditional_losses_4709039
F__inference_model_280_layer_call_and_return_conditional_losses_4709063ј
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
’2“
+__inference_conv2d_70_layer_call_fn_4709210Ґ
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
р2н
F__inference_conv2d_70_layer_call_and_return_conditional_losses_4709220Ґ
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
*__inference_re_lu_70_layer_call_fn_4709225Ґ
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
п2м
E__inference_re_lu_70_layer_call_and_return_conditional_losses_4709230Ґ
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
Ъ2Ч
.__inference_dropout_4830_layer_call_fn_4709235
.__inference_dropout_4830_layer_call_fn_4709240і
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
–2Ќ
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4709245
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4709257і
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
„2‘
-__inference_flatten_280_layer_call_fn_4709262Ґ
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
т2п
H__inference_flatten_280_layer_call_and_return_conditional_losses_4709268Ґ
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
„2‘
-__inference_dense_14630_layer_call_fn_4709277Ґ
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
т2п
H__inference_dense_14630_layer_call_and_return_conditional_losses_4709288Ґ
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
Ъ2Ч
.__inference_dropout_4831_layer_call_fn_4709293
.__inference_dropout_4831_layer_call_fn_4709298і
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
–2Ќ
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4709303
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4709315і
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
Ъ2Ч
.__inference_dropout_4832_layer_call_fn_4709320
.__inference_dropout_4832_layer_call_fn_4709325і
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
–2Ќ
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4709330
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4709342і
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
“2ѕ
(__inference_output_layer_call_fn_4709351Ґ
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
C__inference_output_layer_call_and_return_conditional_losses_4709361Ґ
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
—Bќ
%__inference_signature_wrapper_4709088input_onehot"Ф
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
 Ю
"__inference__wrapped_model_4708721x"#01=Ґ:
3Ґ0
.К+
input_onehot€€€€€€€€€
™ "/™,
*
output К
output€€€€€€€€€Ј
F__inference_conv2d_70_layer_call_and_return_conditional_losses_4709220m7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ П
+__inference_conv2d_70_layer_call_fn_4709210`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "!К€€€€€€€€€і©
H__inference_dense_14630_layer_call_and_return_conditional_losses_4709288]"#0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ƒ
™ "%Ґ"
К
0€€€€€€€€€P
Ъ Б
-__inference_dense_14630_layer_call_fn_4709277P"#0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ƒ
™ "К€€€€€€€€€Pї
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4709245n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p 
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ ї
I__inference_dropout_4830_layer_call_and_return_conditional_losses_4709257n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ У
.__inference_dropout_4830_layer_call_fn_4709235a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p 
™ "!К€€€€€€€€€іУ
.__inference_dropout_4830_layer_call_fn_4709240a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p
™ "!К€€€€€€€€€і©
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4709303\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "%Ґ"
К
0€€€€€€€€€P
Ъ ©
I__inference_dropout_4831_layer_call_and_return_conditional_losses_4709315\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "%Ґ"
К
0€€€€€€€€€P
Ъ Б
.__inference_dropout_4831_layer_call_fn_4709293O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "К€€€€€€€€€PБ
.__inference_dropout_4831_layer_call_fn_4709298O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "К€€€€€€€€€P©
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4709330\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "%Ґ"
К
0€€€€€€€€€P
Ъ ©
I__inference_dropout_4832_layer_call_and_return_conditional_losses_4709342\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "%Ґ"
К
0€€€€€€€€€P
Ъ Б
.__inference_dropout_4832_layer_call_fn_4709320O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "К€€€€€€€€€PБ
.__inference_dropout_4832_layer_call_fn_4709325O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "К€€€€€€€€€PЃ
H__inference_flatten_280_layer_call_and_return_conditional_losses_4709268b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ "&Ґ#
К
0€€€€€€€€€ƒ
Ъ Ж
-__inference_flatten_280_layer_call_fn_4709262U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ "К€€€€€€€€€ƒј
F__inference_model_280_layer_call_and_return_conditional_losses_4709039v"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ј
F__inference_model_280_layer_call_and_return_conditional_losses_4709063v"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
F__inference_model_280_layer_call_and_return_conditional_losses_4709151p"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
F__inference_model_280_layer_call_and_return_conditional_losses_4709201p"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ш
+__inference_model_280_layer_call_fn_4708829i"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ш
+__inference_model_280_layer_call_fn_4709015i"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p

 
™ "К€€€€€€€€€Т
+__inference_model_280_layer_call_fn_4709105c"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Т
+__inference_model_280_layer_call_fn_4709122c"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€£
C__inference_output_layer_call_and_return_conditional_losses_4709361\01/Ґ,
%Ґ"
 К
inputs€€€€€€€€€P
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_output_layer_call_fn_4709351O01/Ґ,
%Ґ"
 К
inputs€€€€€€€€€P
™ "К€€€€€€€€€≥
E__inference_re_lu_70_layer_call_and_return_conditional_losses_4709230j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ Л
*__inference_re_lu_70_layer_call_fn_4709225]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ "!К€€€€€€€€€і≤
%__inference_signature_wrapper_4709088И"#01MҐJ
Ґ 
C™@
>
input_onehot.К+
input_onehot€€€€€€€€€"/™,
*
output К
output€€€€€€€€€