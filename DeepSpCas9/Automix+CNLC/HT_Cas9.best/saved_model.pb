ещ
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
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ыд
Е
conv2d_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*!
shared_nameconv2d_98/kernel
~
$conv2d_98/kernel/Read/ReadVariableOpReadVariableOpconv2d_98/kernel*'
_output_shapes
:і*
dtype0
u
conv2d_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*
shared_nameconv2d_98/bias
n
"conv2d_98/bias/Read/ReadVariableOpReadVariableOpconv2d_98/bias*
_output_shapes	
:і*
dtype0
Б
dense_16738/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ƒP*#
shared_namedense_16738/kernel
z
&dense_16738/kernel/Read/ReadVariableOpReadVariableOpdense_16738/kernel*
_output_shapes
:	ƒP*
dtype0
x
dense_16738/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*!
shared_namedense_16738/bias
q
$dense_16738/bias/Read/ReadVariableOpReadVariableOpdense_16738/bias*
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
Adam/conv2d_98/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*(
shared_nameAdam/conv2d_98/kernel/m
М
+Adam/conv2d_98/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_98/kernel/m*'
_output_shapes
:і*
dtype0
Г
Adam/conv2d_98/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*&
shared_nameAdam/conv2d_98/bias/m
|
)Adam/conv2d_98/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_98/bias/m*
_output_shapes	
:і*
dtype0
П
Adam/dense_16738/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ƒP**
shared_nameAdam/dense_16738/kernel/m
И
-Adam/dense_16738/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16738/kernel/m*
_output_shapes
:	ƒP*
dtype0
Ж
Adam/dense_16738/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*(
shared_nameAdam/dense_16738/bias/m

+Adam/dense_16738/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16738/bias/m*
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
Adam/conv2d_98/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*(
shared_nameAdam/conv2d_98/kernel/v
М
+Adam/conv2d_98/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_98/kernel/v*'
_output_shapes
:і*
dtype0
Г
Adam/conv2d_98/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:і*&
shared_nameAdam/conv2d_98/bias/v
|
)Adam/conv2d_98/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_98/bias/v*
_output_shapes	
:і*
dtype0
П
Adam/dense_16738/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ƒP**
shared_nameAdam/dense_16738/kernel/v
И
-Adam/dense_16738/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16738/kernel/v*
_output_shapes
:	ƒP*
dtype0
Ж
Adam/dense_16738/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*(
shared_nameAdam/dense_16738/bias/v

+Adam/dense_16738/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16738/bias/v*
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
ђ
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
≠
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
VARIABLE_VALUEconv2d_98/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_98/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
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
≠
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
≠
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
≠
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
 	variables
^\
VARIABLE_VALUEdense_16738/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_16738/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
≠
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
≠
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
≠
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
≠
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
VARIABLE_VALUEAdam/conv2d_98/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_98/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/dense_16738/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_16738/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_98/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_98/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/dense_16738/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_16738/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
Ѓ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_98/kernelconv2d_98/biasdense_16738/kerneldense_16738/biasoutput/kerneloutput/bias*
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
GPU2 *0J 8В */
f*R(
&__inference_signature_wrapper_19967821
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
В

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_98/kernel/Read/ReadVariableOp"conv2d_98/bias/Read/ReadVariableOp&dense_16738/kernel/Read/ReadVariableOp$dense_16738/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_98/kernel/m/Read/ReadVariableOp)Adam/conv2d_98/bias/m/Read/ReadVariableOp-Adam/dense_16738/kernel/m/Read/ReadVariableOp+Adam/dense_16738/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv2d_98/kernel/v/Read/ReadVariableOp)Adam/conv2d_98/bias/v/Read/ReadVariableOp-Adam/dense_16738/kernel/v/Read/ReadVariableOp+Adam/dense_16738/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
GPU2 *0J 8В **
f%R#
!__inference__traced_save_19968192
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_98/kernelconv2d_98/biasdense_16738/kerneldense_16738/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_98/kernel/mAdam/conv2d_98/bias/mAdam/dense_16738/kernel/mAdam/dense_16738/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_98/kernel/vAdam/conv2d_98/bias/vAdam/dense_16738/kernel/vAdam/dense_16738/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
GPU2 *0J 8В *-
f(R&
$__inference__traced_restore_19968277эр
–	
х
D__inference_output_layer_call_and_return_conditional_losses_19968085

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
ї

В
G__inference_conv2d_98_layer_call_and_return_conditional_losses_19967944

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
ў
h
/__inference_dropout_5576_layer_call_fn_19968075

inputs
identityИҐStatefulPartitionedCallе
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5576_layer_call_and_return_conditional_losses_199675922
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
ы
Ц
,__inference_model_338_layer_call_fn_19967562
input_onehot"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallґ
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
GPU2 *0J 8В *P
fKRI
G__inference_model_338_layer_call_and_return_conditional_losses_199675472
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
∞
i
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19967592

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
Ы
h
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967968

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
Ѕ"
≤
G__inference_model_338_layer_call_and_return_conditional_losses_19967547

inputs-
conv2d_98_19967472:і!
conv2d_98_19967474:	і'
dense_16738_19967511:	ƒP"
dense_16738_19967513:P!
output_19967541:P
output_19967543:
identityИҐ!conv2d_98/StatefulPartitionedCallҐ#dense_16738/StatefulPartitionedCallҐoutput/StatefulPartitionedCall≠
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_98_19967472conv2d_98_19967474*
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
GPU2 *0J 8В *P
fKRI
G__inference_conv2d_98_layer_call_and_return_conditional_losses_199674712#
!conv2d_98/StatefulPartitionedCallИ
re_lu_98/PartitionedCallPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *O
fJRH
F__inference_re_lu_98_layer_call_and_return_conditional_losses_199674822
re_lu_98/PartitionedCallЛ
dropout_5574/PartitionedCallPartitionedCall!re_lu_98/PartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5574_layer_call_and_return_conditional_losses_199674892
dropout_5574/PartitionedCallД
flatten_338/PartitionedCallPartitionedCall%dropout_5574/PartitionedCall:output:0*
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
GPU2 *0J 8В *R
fMRK
I__inference_flatten_338_layer_call_and_return_conditional_losses_199674972
flatten_338/PartitionedCallћ
#dense_16738/StatefulPartitionedCallStatefulPartitionedCall$flatten_338/PartitionedCall:output:0dense_16738_19967511dense_16738_19967513*
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
GPU2 *0J 8В *R
fMRK
I__inference_dense_16738_layer_call_and_return_conditional_losses_199675102%
#dense_16738/StatefulPartitionedCallН
dropout_5575/PartitionedCallPartitionedCall,dense_16738/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5575_layer_call_and_return_conditional_losses_199675212
dropout_5575/PartitionedCallЖ
dropout_5576/PartitionedCallPartitionedCall%dropout_5575/PartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5576_layer_call_and_return_conditional_losses_199675282
dropout_5576/PartitionedCallі
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_5576/PartitionedCall:output:0output_19967541output_19967543*
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
GPU2 *0J 8В *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_199675402 
output/StatefulPartitionedCallж
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_98/StatefulPartitionedCall$^dense_16738/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2J
#dense_16738/StatefulPartitionedCall#dense_16738/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞
i
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19968038

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
ы
Ц
,__inference_model_338_layer_call_fn_19967748
input_onehot"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallґ
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
GPU2 *0J 8В *P
fKRI
G__inference_model_338_layer_call_and_return_conditional_losses_199677162
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
й
G
+__inference_re_lu_98_layer_call_fn_19967963

inputs
identity“
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
GPU2 *0J 8В *O
fJRH
F__inference_re_lu_98_layer_call_and_return_conditional_losses_199674822
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
≤;
»

!__inference__traced_save_19968192
file_prefix/
+savev2_conv2d_98_kernel_read_readvariableop-
)savev2_conv2d_98_bias_read_readvariableop1
-savev2_dense_16738_kernel_read_readvariableop/
+savev2_dense_16738_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_98_kernel_m_read_readvariableop4
0savev2_adam_conv2d_98_bias_m_read_readvariableop8
4savev2_adam_dense_16738_kernel_m_read_readvariableop6
2savev2_adam_dense_16738_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv2d_98_kernel_v_read_readvariableop4
0savev2_adam_conv2d_98_bias_v_read_readvariableop8
4savev2_adam_dense_16738_kernel_v_read_readvariableop6
2savev2_adam_dense_16738_bias_v_read_readvariableop3
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

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_98_kernel_read_readvariableop)savev2_conv2d_98_bias_read_readvariableop-savev2_dense_16738_kernel_read_readvariableop+savev2_dense_16738_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_98_kernel_m_read_readvariableop0savev2_adam_conv2d_98_bias_m_read_readvariableop4savev2_adam_dense_16738_kernel_m_read_readvariableop2savev2_adam_dense_16738_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv2d_98_kernel_v_read_readvariableop0savev2_adam_conv2d_98_bias_v_read_readvariableop4savev2_adam_dense_16738_kernel_v_read_readvariableop2savev2_adam_dense_16738_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
”"
Є
G__inference_model_338_layer_call_and_return_conditional_losses_19967772
input_onehot-
conv2d_98_19967751:і!
conv2d_98_19967753:	і'
dense_16738_19967759:	ƒP"
dense_16738_19967761:P!
output_19967766:P
output_19967768:
identityИҐ!conv2d_98/StatefulPartitionedCallҐ#dense_16738/StatefulPartitionedCallҐoutput/StatefulPartitionedCall≥
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_98_19967751conv2d_98_19967753*
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
GPU2 *0J 8В *P
fKRI
G__inference_conv2d_98_layer_call_and_return_conditional_losses_199674712#
!conv2d_98/StatefulPartitionedCallИ
re_lu_98/PartitionedCallPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *O
fJRH
F__inference_re_lu_98_layer_call_and_return_conditional_losses_199674822
re_lu_98/PartitionedCallЛ
dropout_5574/PartitionedCallPartitionedCall!re_lu_98/PartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5574_layer_call_and_return_conditional_losses_199674892
dropout_5574/PartitionedCallД
flatten_338/PartitionedCallPartitionedCall%dropout_5574/PartitionedCall:output:0*
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
GPU2 *0J 8В *R
fMRK
I__inference_flatten_338_layer_call_and_return_conditional_losses_199674972
flatten_338/PartitionedCallћ
#dense_16738/StatefulPartitionedCallStatefulPartitionedCall$flatten_338/PartitionedCall:output:0dense_16738_19967759dense_16738_19967761*
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
GPU2 *0J 8В *R
fMRK
I__inference_dense_16738_layer_call_and_return_conditional_losses_199675102%
#dense_16738/StatefulPartitionedCallН
dropout_5575/PartitionedCallPartitionedCall,dense_16738/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5575_layer_call_and_return_conditional_losses_199675212
dropout_5575/PartitionedCallЖ
dropout_5576/PartitionedCallPartitionedCall%dropout_5575/PartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5576_layer_call_and_return_conditional_losses_199675282
dropout_5576/PartitionedCallі
output/StatefulPartitionedCallStatefulPartitionedCall%dropout_5576/PartitionedCall:output:0output_19967766output_19967768*
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
GPU2 *0J 8В *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_199675402 
output/StatefulPartitionedCallж
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_98/StatefulPartitionedCall$^dense_16738/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2J
#dense_16738/StatefulPartitionedCall#dense_16738/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
В
b
F__inference_re_lu_98_layer_call_and_return_conditional_losses_19967958

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
’'
≠
G__inference_model_338_layer_call_and_return_conditional_losses_19967796
input_onehot-
conv2d_98_19967775:і!
conv2d_98_19967777:	і'
dense_16738_19967783:	ƒP"
dense_16738_19967785:P!
output_19967790:P
output_19967792:
identityИҐ!conv2d_98/StatefulPartitionedCallҐ#dense_16738/StatefulPartitionedCallҐ$dropout_5574/StatefulPartitionedCallҐ$dropout_5575/StatefulPartitionedCallҐ$dropout_5576/StatefulPartitionedCallҐoutput/StatefulPartitionedCall≥
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_98_19967775conv2d_98_19967777*
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
GPU2 *0J 8В *P
fKRI
G__inference_conv2d_98_layer_call_and_return_conditional_losses_199674712#
!conv2d_98/StatefulPartitionedCallИ
re_lu_98/PartitionedCallPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *O
fJRH
F__inference_re_lu_98_layer_call_and_return_conditional_losses_199674822
re_lu_98/PartitionedCall£
$dropout_5574/StatefulPartitionedCallStatefulPartitionedCall!re_lu_98/PartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5574_layer_call_and_return_conditional_losses_199676542&
$dropout_5574/StatefulPartitionedCallМ
flatten_338/PartitionedCallPartitionedCall-dropout_5574/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *R
fMRK
I__inference_flatten_338_layer_call_and_return_conditional_losses_199674972
flatten_338/PartitionedCallћ
#dense_16738/StatefulPartitionedCallStatefulPartitionedCall$flatten_338/PartitionedCall:output:0dense_16738_19967783dense_16738_19967785*
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
GPU2 *0J 8В *R
fMRK
I__inference_dense_16738_layer_call_and_return_conditional_losses_199675102%
#dense_16738/StatefulPartitionedCallћ
$dropout_5575/StatefulPartitionedCallStatefulPartitionedCall,dense_16738/StatefulPartitionedCall:output:0%^dropout_5574/StatefulPartitionedCall*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5575_layer_call_and_return_conditional_losses_199676152&
$dropout_5575/StatefulPartitionedCallЌ
$dropout_5576/StatefulPartitionedCallStatefulPartitionedCall-dropout_5575/StatefulPartitionedCall:output:0%^dropout_5575/StatefulPartitionedCall*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5576_layer_call_and_return_conditional_losses_199675922&
$dropout_5576/StatefulPartitionedCallЉ
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_5576/StatefulPartitionedCall:output:0output_19967790output_19967792*
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
GPU2 *0J 8В *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_199675402 
output/StatefulPartitionedCallџ
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_98/StatefulPartitionedCall$^dense_16738/StatefulPartitionedCall%^dropout_5574/StatefulPartitionedCall%^dropout_5575/StatefulPartitionedCall%^dropout_5576/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2J
#dense_16738/StatefulPartitionedCall#dense_16738/StatefulPartitionedCall2L
$dropout_5574/StatefulPartitionedCall$dropout_5574/StatefulPartitionedCall2L
$dropout_5575/StatefulPartitionedCall$dropout_5575/StatefulPartitionedCall2L
$dropout_5576/StatefulPartitionedCall$dropout_5576/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
—
Р
&__inference_signature_wrapper_19967821
input_onehot"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCallТ
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
GPU2 *0J 8В *,
f'R%
#__inference__wrapped_model_199674542
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
н
e
I__inference_flatten_338_layer_call_and_return_conditional_losses_19967497

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
н
e
I__inference_flatten_338_layer_call_and_return_conditional_losses_19967996

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
јB
Ы
G__inference_model_338_layer_call_and_return_conditional_losses_19967900

inputsC
(conv2d_98_conv2d_readvariableop_resource:і8
)conv2d_98_biasadd_readvariableop_resource:	і=
*dense_16738_matmul_readvariableop_resource:	ƒP9
+dense_16738_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityИҐ conv2d_98/BiasAdd/ReadVariableOpҐconv2d_98/Conv2D/ReadVariableOpҐ"dense_16738/BiasAdd/ReadVariableOpҐ!dense_16738/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOpі
conv2d_98/Conv2D/ReadVariableOpReadVariableOp(conv2d_98_conv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02!
conv2d_98/Conv2D/ReadVariableOp√
conv2d_98/Conv2DConv2Dinputs'conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
conv2d_98/Conv2DЂ
 conv2d_98/BiasAdd/ReadVariableOpReadVariableOp)conv2d_98_biasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02"
 conv2d_98/BiasAdd/ReadVariableOp±
conv2d_98/BiasAddBiasAddconv2d_98/Conv2D:output:0(conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2
conv2d_98/BiasAdd}
re_lu_98/ReluReluconv2d_98/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
re_lu_98/Relu}
dropout_5574/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_5574/dropout/ConstЄ
dropout_5574/dropout/MulMulre_lu_98/Relu:activations:0#dropout_5574/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout_5574/dropout/MulГ
dropout_5574/dropout/ShapeShapere_lu_98/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5574/dropout/Shapeд
1dropout_5574/dropout/random_uniform/RandomUniformRandomUniform#dropout_5574/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€і*
dtype023
1dropout_5574/dropout/random_uniform/RandomUniformП
#dropout_5574/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#dropout_5574/dropout/GreaterEqual/yы
!dropout_5574/dropout/GreaterEqualGreaterEqual:dropout_5574/dropout/random_uniform/RandomUniform:output:0,dropout_5574/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2#
!dropout_5574/dropout/GreaterEqualѓ
dropout_5574/dropout/CastCast%dropout_5574/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€і2
dropout_5574/dropout/CastЈ
dropout_5574/dropout/Mul_1Muldropout_5574/dropout/Mul:z:0dropout_5574/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout_5574/dropout/Mul_1w
flatten_338/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
flatten_338/Const§
flatten_338/ReshapeReshapedropout_5574/dropout/Mul_1:z:0flatten_338/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2
flatten_338/Reshape≤
!dense_16738/MatMul/ReadVariableOpReadVariableOp*dense_16738_matmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02#
!dense_16738/MatMul/ReadVariableOp≠
dense_16738/MatMulMatMulflatten_338/Reshape:output:0)dense_16738/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_16738/MatMul∞
"dense_16738/BiasAdd/ReadVariableOpReadVariableOp+dense_16738_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02$
"dense_16738/BiasAdd/ReadVariableOp±
dense_16738/BiasAddBiasAdddense_16738/MatMul:product:0*dense_16738/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_16738/BiasAdd|
dense_16738/ReluReludense_16738/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_16738/Relu}
dropout_5575/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_5575/dropout/Const≤
dropout_5575/dropout/MulMuldense_16738/Relu:activations:0#dropout_5575/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_5575/dropout/MulЖ
dropout_5575/dropout/ShapeShapedense_16738/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5575/dropout/Shapeџ
1dropout_5575/dropout/random_uniform/RandomUniformRandomUniform#dropout_5575/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype023
1dropout_5575/dropout/random_uniform/RandomUniformП
#dropout_5575/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#dropout_5575/dropout/GreaterEqual/yт
!dropout_5575/dropout/GreaterEqualGreaterEqual:dropout_5575/dropout/random_uniform/RandomUniform:output:0,dropout_5575/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2#
!dropout_5575/dropout/GreaterEqual¶
dropout_5575/dropout/CastCast%dropout_5575/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout_5575/dropout/CastЃ
dropout_5575/dropout/Mul_1Muldropout_5575/dropout/Mul:z:0dropout_5575/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_5575/dropout/Mul_1}
dropout_5576/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_5576/dropout/Const≤
dropout_5576/dropout/MulMuldropout_5575/dropout/Mul_1:z:0#dropout_5576/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_5576/dropout/MulЖ
dropout_5576/dropout/ShapeShapedropout_5575/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dropout_5576/dropout/Shapeџ
1dropout_5576/dropout/random_uniform/RandomUniformRandomUniform#dropout_5576/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€P*
dtype023
1dropout_5576/dropout/random_uniform/RandomUniformП
#dropout_5576/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2%
#dropout_5576/dropout/GreaterEqual/yт
!dropout_5576/dropout/GreaterEqualGreaterEqual:dropout_5576/dropout/random_uniform/RandomUniform:output:0,dropout_5576/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2#
!dropout_5576/dropout/GreaterEqual¶
dropout_5576/dropout/CastCast%dropout_5576/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€P2
dropout_5576/dropout/CastЃ
dropout_5576/dropout/Mul_1Muldropout_5576/dropout/Mul:z:0dropout_5576/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_5576/dropout/Mul_1Ґ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp†
output/MatMulMatMuldropout_5576/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_98/BiasAdd/ReadVariableOp ^conv2d_98/Conv2D/ReadVariableOp#^dense_16738/BiasAdd/ReadVariableOp"^dense_16738/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_98/BiasAdd/ReadVariableOp conv2d_98/BiasAdd/ReadVariableOp2B
conv2d_98/Conv2D/ReadVariableOpconv2d_98/Conv2D/ReadVariableOp2H
"dense_16738/BiasAdd/ReadVariableOp"dense_16738/BiasAdd/ReadVariableOp2F
!dense_16738/MatMul/ReadVariableOp!dense_16738/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ї

В
G__inference_conv2d_98_layer_call_and_return_conditional_losses_19967471

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
й
Р
,__inference_model_338_layer_call_fn_19967917

inputs"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCall∞
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
GPU2 *0J 8В *P
fKRI
G__inference_model_338_layer_call_and_return_conditional_losses_199675472
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
В
b
F__inference_re_lu_98_layer_call_and_return_conditional_losses_19967482

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
ў
h
/__inference_dropout_5575_layer_call_fn_19968048

inputs
identityИҐStatefulPartitionedCallе
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5575_layer_call_and_return_conditional_losses_199676152
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
Ы
h
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967489

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
Ў$
Ы
G__inference_model_338_layer_call_and_return_conditional_losses_19967850

inputsC
(conv2d_98_conv2d_readvariableop_resource:і8
)conv2d_98_biasadd_readvariableop_resource:	і=
*dense_16738_matmul_readvariableop_resource:	ƒP9
+dense_16738_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityИҐ conv2d_98/BiasAdd/ReadVariableOpҐconv2d_98/Conv2D/ReadVariableOpҐ"dense_16738/BiasAdd/ReadVariableOpҐ!dense_16738/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOpі
conv2d_98/Conv2D/ReadVariableOpReadVariableOp(conv2d_98_conv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02!
conv2d_98/Conv2D/ReadVariableOp√
conv2d_98/Conv2DConv2Dinputs'conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
conv2d_98/Conv2DЂ
 conv2d_98/BiasAdd/ReadVariableOpReadVariableOp)conv2d_98_biasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02"
 conv2d_98/BiasAdd/ReadVariableOp±
conv2d_98/BiasAddBiasAddconv2d_98/Conv2D:output:0(conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2
conv2d_98/BiasAdd}
re_lu_98/ReluReluconv2d_98/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
re_lu_98/ReluТ
dropout_5574/IdentityIdentityre_lu_98/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€і2
dropout_5574/Identityw
flatten_338/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
flatten_338/Const§
flatten_338/ReshapeReshapedropout_5574/Identity:output:0flatten_338/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2
flatten_338/Reshape≤
!dense_16738/MatMul/ReadVariableOpReadVariableOp*dense_16738_matmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02#
!dense_16738/MatMul/ReadVariableOp≠
dense_16738/MatMulMatMulflatten_338/Reshape:output:0)dense_16738/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_16738/MatMul∞
"dense_16738/BiasAdd/ReadVariableOpReadVariableOp+dense_16738_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02$
"dense_16738/BiasAdd/ReadVariableOp±
dense_16738/BiasAddBiasAdddense_16738/MatMul:product:0*dense_16738/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_16738/BiasAdd|
dense_16738/ReluReludense_16738/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dense_16738/ReluМ
dropout_5575/IdentityIdentitydense_16738/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_5575/IdentityМ
dropout_5576/IdentityIdentitydropout_5575/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
dropout_5576/IdentityҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp†
output/MatMulMatMuldropout_5576/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_98/BiasAdd/ReadVariableOp ^conv2d_98/Conv2D/ReadVariableOp#^dense_16738/BiasAdd/ReadVariableOp"^dense_16738/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2D
 conv2d_98/BiasAdd/ReadVariableOp conv2d_98/BiasAdd/ReadVariableOp2B
conv2d_98/Conv2D/ReadVariableOpconv2d_98/Conv2D/ReadVariableOp2H
"dense_16738/BiasAdd/ReadVariableOp"dense_16738/BiasAdd/ReadVariableOp2F
!dense_16738/MatMul/ReadVariableOp!dense_16738/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
±m
І
$__inference__traced_restore_19968277
file_prefix<
!assignvariableop_conv2d_98_kernel:і0
!assignvariableop_1_conv2d_98_bias:	і8
%assignvariableop_2_dense_16738_kernel:	ƒP1
#assignvariableop_3_dense_16738_bias:P2
 assignvariableop_4_output_kernel:P,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: F
+assignvariableop_13_adam_conv2d_98_kernel_m:і8
)assignvariableop_14_adam_conv2d_98_bias_m:	і@
-assignvariableop_15_adam_dense_16738_kernel_m:	ƒP9
+assignvariableop_16_adam_dense_16738_bias_m:P:
(assignvariableop_17_adam_output_kernel_m:P4
&assignvariableop_18_adam_output_bias_m:F
+assignvariableop_19_adam_conv2d_98_kernel_v:і8
)assignvariableop_20_adam_conv2d_98_bias_v:	і@
-assignvariableop_21_adam_dense_16738_kernel_v:	ƒP9
+assignvariableop_22_adam_dense_16738_bias_v:P:
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_98_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_98_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2™
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_16738_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3®
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_16738_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_98_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14±
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_98_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15µ
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_dense_16738_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16≥
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_16738_bias_mIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_98_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_98_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21µ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_dense_16738_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22≥
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_dense_16738_bias_vIdentity_22:output:0"/device:CPU:0*
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
с
K
/__inference_dropout_5574_layer_call_fn_19967985

inputs
identity÷
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5574_layer_call_and_return_conditional_losses_199674892
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
ч
h
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19967521

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
–	
х
D__inference_output_layer_call_and_return_conditional_losses_19967540

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
я
J
.__inference_flatten_338_layer_call_fn_19968001

inputs
identityЌ
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
GPU2 *0J 8В *R
fMRK
I__inference_flatten_338_layer_call_and_return_conditional_losses_199674972
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
∞
i
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19967615

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
й
Р
,__inference_model_338_layer_call_fn_19967934

inputs"
unknown:і
	unknown_0:	і
	unknown_1:	ƒP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityИҐStatefulPartitionedCall∞
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
GPU2 *0J 8В *P
fKRI
G__inference_model_338_layer_call_and_return_conditional_losses_199677162
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
≠
Ь
.__inference_dense_16738_layer_call_fn_19968021

inputs
unknown:	ƒP
	unknown_0:P
identityИҐStatefulPartitionedCallю
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
GPU2 *0J 8В *R
fMRK
I__inference_dense_16738_layer_call_and_return_conditional_losses_199675102
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
µ

ы
I__inference_dense_16738_layer_call_and_return_conditional_losses_19967510

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
ч
h
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19968026

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
э
h
/__inference_dropout_5574_layer_call_fn_19967990

inputs
identityИҐStatefulPartitionedCallо
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5574_layer_call_and_return_conditional_losses_199676542
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
“
£
,__inference_conv2d_98_layer_call_fn_19967953

inputs"
unknown:і
	unknown_0:	і
identityИҐStatefulPartitionedCallЕ
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
GPU2 *0J 8В *P
fKRI
G__inference_conv2d_98_layer_call_and_return_conditional_losses_199674712
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
Ќ
K
/__inference_dropout_5575_layer_call_fn_19968043

inputs
identityЌ
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5575_layer_call_and_return_conditional_losses_199675212
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
ч
h
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19968053

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
ч
h
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19967528

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
ъ
i
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967654

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
ъ
i
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967980

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
∞
i
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19968065

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
µ

ы
I__inference_dense_16738_layer_call_and_return_conditional_losses_19968012

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
†
Ц
)__inference_output_layer_call_fn_19968094

inputs
unknown:P
	unknown_0:
identityИҐStatefulPartitionedCallщ
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
GPU2 *0J 8В *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_199675402
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
л+
х
#__inference__wrapped_model_19967454
input_onehotM
2model_338_conv2d_98_conv2d_readvariableop_resource:іB
3model_338_conv2d_98_biasadd_readvariableop_resource:	іG
4model_338_dense_16738_matmul_readvariableop_resource:	ƒPC
5model_338_dense_16738_biasadd_readvariableop_resource:PA
/model_338_output_matmul_readvariableop_resource:P>
0model_338_output_biasadd_readvariableop_resource:
identityИҐ*model_338/conv2d_98/BiasAdd/ReadVariableOpҐ)model_338/conv2d_98/Conv2D/ReadVariableOpҐ,model_338/dense_16738/BiasAdd/ReadVariableOpҐ+model_338/dense_16738/MatMul/ReadVariableOpҐ'model_338/output/BiasAdd/ReadVariableOpҐ&model_338/output/MatMul/ReadVariableOp“
)model_338/conv2d_98/Conv2D/ReadVariableOpReadVariableOp2model_338_conv2d_98_conv2d_readvariableop_resource*'
_output_shapes
:і*
dtype02+
)model_338/conv2d_98/Conv2D/ReadVariableOpз
model_338/conv2d_98/Conv2DConv2Dinput_onehot1model_338/conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і*
paddingVALID*
strides
2
model_338/conv2d_98/Conv2D…
*model_338/conv2d_98/BiasAdd/ReadVariableOpReadVariableOp3model_338_conv2d_98_biasadd_readvariableop_resource*
_output_shapes	
:і*
dtype02,
*model_338/conv2d_98/BiasAdd/ReadVariableOpў
model_338/conv2d_98/BiasAddBiasAdd#model_338/conv2d_98/Conv2D:output:02model_338/conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€і2
model_338/conv2d_98/BiasAddЫ
model_338/re_lu_98/ReluRelu$model_338/conv2d_98/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€і2
model_338/re_lu_98/Relu∞
model_338/dropout_5574/IdentityIdentity%model_338/re_lu_98/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€і2!
model_338/dropout_5574/IdentityЛ
model_338/flatten_338/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ƒ  2
model_338/flatten_338/Constћ
model_338/flatten_338/ReshapeReshape(model_338/dropout_5574/Identity:output:0$model_338/flatten_338/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ƒ2
model_338/flatten_338/Reshape–
+model_338/dense_16738/MatMul/ReadVariableOpReadVariableOp4model_338_dense_16738_matmul_readvariableop_resource*
_output_shapes
:	ƒP*
dtype02-
+model_338/dense_16738/MatMul/ReadVariableOp’
model_338/dense_16738/MatMulMatMul&model_338/flatten_338/Reshape:output:03model_338/dense_16738/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
model_338/dense_16738/MatMulќ
,model_338/dense_16738/BiasAdd/ReadVariableOpReadVariableOp5model_338_dense_16738_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02.
,model_338/dense_16738/BiasAdd/ReadVariableOpў
model_338/dense_16738/BiasAddBiasAdd&model_338/dense_16738/MatMul:product:04model_338/dense_16738/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P2
model_338/dense_16738/BiasAddЪ
model_338/dense_16738/ReluRelu&model_338/dense_16738/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2
model_338/dense_16738/Relu™
model_338/dropout_5575/IdentityIdentity(model_338/dense_16738/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€P2!
model_338/dropout_5575/Identity™
model_338/dropout_5576/IdentityIdentity(model_338/dropout_5575/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€P2!
model_338/dropout_5576/Identityј
&model_338/output/MatMul/ReadVariableOpReadVariableOp/model_338_output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02(
&model_338/output/MatMul/ReadVariableOp»
model_338/output/MatMulMatMul(model_338/dropout_5576/Identity:output:0.model_338/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_338/output/MatMulњ
'model_338/output/BiasAdd/ReadVariableOpReadVariableOp0model_338_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_338/output/BiasAdd/ReadVariableOp≈
model_338/output/BiasAddBiasAdd!model_338/output/MatMul:product:0/model_338/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_338/output/BiasAddю
IdentityIdentity!model_338/output/BiasAdd:output:0+^model_338/conv2d_98/BiasAdd/ReadVariableOp*^model_338/conv2d_98/Conv2D/ReadVariableOp-^model_338/dense_16738/BiasAdd/ReadVariableOp,^model_338/dense_16738/MatMul/ReadVariableOp(^model_338/output/BiasAdd/ReadVariableOp'^model_338/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2X
*model_338/conv2d_98/BiasAdd/ReadVariableOp*model_338/conv2d_98/BiasAdd/ReadVariableOp2V
)model_338/conv2d_98/Conv2D/ReadVariableOp)model_338/conv2d_98/Conv2D/ReadVariableOp2\
,model_338/dense_16738/BiasAdd/ReadVariableOp,model_338/dense_16738/BiasAdd/ReadVariableOp2Z
+model_338/dense_16738/MatMul/ReadVariableOp+model_338/dense_16738/MatMul/ReadVariableOp2R
'model_338/output/BiasAdd/ReadVariableOp'model_338/output/BiasAdd/ReadVariableOp2P
&model_338/output/MatMul/ReadVariableOp&model_338/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_onehot
√'
І
G__inference_model_338_layer_call_and_return_conditional_losses_19967716

inputs-
conv2d_98_19967695:і!
conv2d_98_19967697:	і'
dense_16738_19967703:	ƒP"
dense_16738_19967705:P!
output_19967710:P
output_19967712:
identityИҐ!conv2d_98/StatefulPartitionedCallҐ#dense_16738/StatefulPartitionedCallҐ$dropout_5574/StatefulPartitionedCallҐ$dropout_5575/StatefulPartitionedCallҐ$dropout_5576/StatefulPartitionedCallҐoutput/StatefulPartitionedCall≠
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_98_19967695conv2d_98_19967697*
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
GPU2 *0J 8В *P
fKRI
G__inference_conv2d_98_layer_call_and_return_conditional_losses_199674712#
!conv2d_98/StatefulPartitionedCallИ
re_lu_98/PartitionedCallPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *O
fJRH
F__inference_re_lu_98_layer_call_and_return_conditional_losses_199674822
re_lu_98/PartitionedCall£
$dropout_5574/StatefulPartitionedCallStatefulPartitionedCall!re_lu_98/PartitionedCall:output:0*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5574_layer_call_and_return_conditional_losses_199676542&
$dropout_5574/StatefulPartitionedCallМ
flatten_338/PartitionedCallPartitionedCall-dropout_5574/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8В *R
fMRK
I__inference_flatten_338_layer_call_and_return_conditional_losses_199674972
flatten_338/PartitionedCallћ
#dense_16738/StatefulPartitionedCallStatefulPartitionedCall$flatten_338/PartitionedCall:output:0dense_16738_19967703dense_16738_19967705*
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
GPU2 *0J 8В *R
fMRK
I__inference_dense_16738_layer_call_and_return_conditional_losses_199675102%
#dense_16738/StatefulPartitionedCallћ
$dropout_5575/StatefulPartitionedCallStatefulPartitionedCall,dense_16738/StatefulPartitionedCall:output:0%^dropout_5574/StatefulPartitionedCall*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5575_layer_call_and_return_conditional_losses_199676152&
$dropout_5575/StatefulPartitionedCallЌ
$dropout_5576/StatefulPartitionedCallStatefulPartitionedCall-dropout_5575/StatefulPartitionedCall:output:0%^dropout_5575/StatefulPartitionedCall*
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5576_layer_call_and_return_conditional_losses_199675922&
$dropout_5576/StatefulPartitionedCallЉ
output/StatefulPartitionedCallStatefulPartitionedCall-dropout_5576/StatefulPartitionedCall:output:0output_19967710output_19967712*
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
GPU2 *0J 8В *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_199675402 
output/StatefulPartitionedCallџ
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_98/StatefulPartitionedCall$^dense_16738/StatefulPartitionedCall%^dropout_5574/StatefulPartitionedCall%^dropout_5575/StatefulPartitionedCall%^dropout_5576/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2J
#dense_16738/StatefulPartitionedCall#dense_16738/StatefulPartitionedCall2L
$dropout_5574/StatefulPartitionedCall$dropout_5574/StatefulPartitionedCall2L
$dropout_5575/StatefulPartitionedCall$dropout_5575/StatefulPartitionedCall2L
$dropout_5576/StatefulPartitionedCall$dropout_5576/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ
K
/__inference_dropout_5576_layer_call_fn_19968070

inputs
identityЌ
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
GPU2 *0J 8В *S
fNRL
J__inference_dropout_5576_layer_call_and_return_conditional_losses_199675282
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
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ƒБ
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*y&call_and_return_all_conditional_losses
z__call__
{_default_save_signature"Ј=
_tf_keras_networkЫ={"name": "model_338", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_338", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_98", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_98", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_98", "inbound_nodes": [[["conv2d_98", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5574", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5574", "inbound_nodes": [[["re_lu_98", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_338", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_338", "inbound_nodes": [[["dropout_5574", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16738", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16738", "inbound_nodes": [[["flatten_338", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5575", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5575", "inbound_nodes": [[["dense_16738", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5576", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5576", "inbound_nodes": [[["dropout_5575", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_5576", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_338", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_98", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "ReLU", "config": {"name": "re_lu_98", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_98", "inbound_nodes": [[["conv2d_98", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_5574", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5574", "inbound_nodes": [[["re_lu_98", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_338", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_338", "inbound_nodes": [[["dropout_5574", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_16738", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16738", "inbound_nodes": [[["flatten_338", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_5575", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5575", "inbound_nodes": [[["dense_16738", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_5576", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_5576", "inbound_nodes": [[["dropout_5575", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_5576", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Б"ю
_tf_keras_input_layerё{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
Г

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*|&call_and_return_all_conditional_losses
}__call__"ё	
_tf_keras_layerƒ	{"name": "conv2d_98", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}}
≤
trainable_variables
regularization_losses
	variables
	keras_api
*~&call_and_return_all_conditional_losses
__call__"£
_tf_keras_layerЙ{"name": "re_lu_98", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_98", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv2d_98", 0, 0, {}]]], "shared_object_id": 4}
±
trainable_variables
regularization_losses
	variables
	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"†
_tf_keras_layerЖ{"name": "dropout_5574", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_5574", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["re_lu_98", 0, 0, {}]]], "shared_object_id": 5}
ћ
trainable_variables
regularization_losses
 	variables
!	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"ї
_tf_keras_layer°{"name": "flatten_338", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_338", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["dropout_5574", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
Л	

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"д
_tf_keras_layer {"name": "dense_16738", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_16738", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_338", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3780}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3780]}}
µ
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"§
_tf_keras_layerК{"name": "dropout_5575", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_5575", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_16738", 0, 0, {}]]], "shared_object_id": 10}
ґ
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"•
_tf_keras_layerЛ{"name": "dropout_5576", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_5576", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dropout_5575", 0, 0, {}]]], "shared_object_id": 11}
В	

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"џ
_tf_keras_layerЅ{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_5576", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
њ
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
 
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
Мserving_default"
signature_map
+:)і2conv2d_98/kernel
:і2conv2d_98/bias
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
≠
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
≠
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
∞
trainable_variables
Jlayer_metrics
Knon_trainable_variables
regularization_losses
Llayer_regularization_losses
Mmetrics

Nlayers
	variables
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
 	variables
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
%:#	ƒP2dense_16738/kernel
:P2dense_16738/bias
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
∞
$trainable_variables
Tlayer_metrics
Unon_trainable_variables
%regularization_losses
Vlayer_regularization_losses
Wmetrics

Xlayers
&	variables
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
(trainable_variables
Ylayer_metrics
Znon_trainable_variables
)regularization_losses
[layer_regularization_losses
\metrics

]layers
*	variables
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
,trainable_variables
^layer_metrics
_non_trainable_variables
-regularization_losses
`layer_regularization_losses
ametrics

blayers
.	variables
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
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
∞
2trainable_variables
clayer_metrics
dnon_trainable_variables
3regularization_losses
elayer_regularization_losses
fmetrics

glayers
4	variables
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
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
0:.і2Adam/conv2d_98/kernel/m
": і2Adam/conv2d_98/bias/m
*:(	ƒP2Adam/dense_16738/kernel/m
#:!P2Adam/dense_16738/bias/m
$:"P2Adam/output/kernel/m
:2Adam/output/bias/m
0:.і2Adam/conv2d_98/kernel/v
": і2Adam/conv2d_98/bias/v
*:(	ƒP2Adam/dense_16738/kernel/v
#:!P2Adam/dense_16738/bias/v
$:"P2Adam/output/kernel/v
:2Adam/output/bias/v
к2з
G__inference_model_338_layer_call_and_return_conditional_losses_19967850
G__inference_model_338_layer_call_and_return_conditional_losses_19967900
G__inference_model_338_layer_call_and_return_conditional_losses_19967772
G__inference_model_338_layer_call_and_return_conditional_losses_19967796ј
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
ю2ы
,__inference_model_338_layer_call_fn_19967562
,__inference_model_338_layer_call_fn_19967917
,__inference_model_338_layer_call_fn_19967934
,__inference_model_338_layer_call_fn_19967748ј
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
#__inference__wrapped_model_19967454√
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
с2о
G__inference_conv2d_98_layer_call_and_return_conditional_losses_19967944Ґ
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
÷2”
,__inference_conv2d_98_layer_call_fn_19967953Ґ
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
F__inference_re_lu_98_layer_call_and_return_conditional_losses_19967958Ґ
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
’2“
+__inference_re_lu_98_layer_call_fn_19967963Ґ
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
“2ѕ
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967968
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967980і
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
Ь2Щ
/__inference_dropout_5574_layer_call_fn_19967985
/__inference_dropout_5574_layer_call_fn_19967990і
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
у2р
I__inference_flatten_338_layer_call_and_return_conditional_losses_19967996Ґ
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
Ў2’
.__inference_flatten_338_layer_call_fn_19968001Ґ
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
у2р
I__inference_dense_16738_layer_call_and_return_conditional_losses_19968012Ґ
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
Ў2’
.__inference_dense_16738_layer_call_fn_19968021Ґ
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
“2ѕ
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19968026
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19968038і
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
Ь2Щ
/__inference_dropout_5575_layer_call_fn_19968043
/__inference_dropout_5575_layer_call_fn_19968048і
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
“2ѕ
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19968053
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19968065і
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
Ь2Щ
/__inference_dropout_5576_layer_call_fn_19968070
/__inference_dropout_5576_layer_call_fn_19968075і
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
о2л
D__inference_output_layer_call_and_return_conditional_losses_19968085Ґ
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
)__inference_output_layer_call_fn_19968094Ґ
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
“Bѕ
&__inference_signature_wrapper_19967821input_onehot"Ф
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
 Я
#__inference__wrapped_model_19967454x"#01=Ґ:
3Ґ0
.К+
input_onehot€€€€€€€€€
™ "/™,
*
output К
output€€€€€€€€€Є
G__inference_conv2d_98_layer_call_and_return_conditional_losses_19967944m7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ Р
,__inference_conv2d_98_layer_call_fn_19967953`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "!К€€€€€€€€€і™
I__inference_dense_16738_layer_call_and_return_conditional_losses_19968012]"#0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ƒ
™ "%Ґ"
К
0€€€€€€€€€P
Ъ В
.__inference_dense_16738_layer_call_fn_19968021P"#0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ƒ
™ "К€€€€€€€€€PЉ
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967968n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p 
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ Љ
J__inference_dropout_5574_layer_call_and_return_conditional_losses_19967980n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ Ф
/__inference_dropout_5574_layer_call_fn_19967985a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p 
™ "!К€€€€€€€€€іФ
/__inference_dropout_5574_layer_call_fn_19967990a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€і
p
™ "!К€€€€€€€€€і™
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19968026\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "%Ґ"
К
0€€€€€€€€€P
Ъ ™
J__inference_dropout_5575_layer_call_and_return_conditional_losses_19968038\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "%Ґ"
К
0€€€€€€€€€P
Ъ В
/__inference_dropout_5575_layer_call_fn_19968043O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "К€€€€€€€€€PВ
/__inference_dropout_5575_layer_call_fn_19968048O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "К€€€€€€€€€P™
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19968053\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "%Ґ"
К
0€€€€€€€€€P
Ъ ™
J__inference_dropout_5576_layer_call_and_return_conditional_losses_19968065\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "%Ґ"
К
0€€€€€€€€€P
Ъ В
/__inference_dropout_5576_layer_call_fn_19968070O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p 
™ "К€€€€€€€€€PВ
/__inference_dropout_5576_layer_call_fn_19968075O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€P
p
™ "К€€€€€€€€€Pѓ
I__inference_flatten_338_layer_call_and_return_conditional_losses_19967996b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ "&Ґ#
К
0€€€€€€€€€ƒ
Ъ З
.__inference_flatten_338_layer_call_fn_19968001U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ "К€€€€€€€€€ƒЅ
G__inference_model_338_layer_call_and_return_conditional_losses_19967772v"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ѕ
G__inference_model_338_layer_call_and_return_conditional_losses_19967796v"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ї
G__inference_model_338_layer_call_and_return_conditional_losses_19967850p"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ї
G__inference_model_338_layer_call_and_return_conditional_losses_19967900p"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Щ
,__inference_model_338_layer_call_fn_19967562i"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p 

 
™ "К€€€€€€€€€Щ
,__inference_model_338_layer_call_fn_19967748i"#01EҐB
;Ґ8
.К+
input_onehot€€€€€€€€€
p

 
™ "К€€€€€€€€€У
,__inference_model_338_layer_call_fn_19967917c"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€У
,__inference_model_338_layer_call_fn_19967934c"#01?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€§
D__inference_output_layer_call_and_return_conditional_losses_19968085\01/Ґ,
%Ґ"
 К
inputs€€€€€€€€€P
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_output_layer_call_fn_19968094O01/Ґ,
%Ґ"
 К
inputs€€€€€€€€€P
™ "К€€€€€€€€€і
F__inference_re_lu_98_layer_call_and_return_conditional_losses_19967958j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ ".Ґ+
$К!
0€€€€€€€€€і
Ъ М
+__inference_re_lu_98_layer_call_fn_19967963]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€і
™ "!К€€€€€€€€€і≥
&__inference_signature_wrapper_19967821И"#01MҐJ
Ґ 
C™@
>
input_onehot.К+
input_onehot€€€€€€€€€"/™,
*
output К
output€€€€€€€€€