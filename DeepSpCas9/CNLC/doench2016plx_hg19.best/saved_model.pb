ея
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Им

conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*!
shared_nameconv2d_35/kernel
~
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*'
_output_shapes
:Д*
dtype0
u
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*
shared_nameconv2d_35/bias
n
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes	
:Д*
dtype0
{
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP* 
shared_namedense_35/kernel
t
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes
:	ФP*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
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
Adam/conv2d_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*(
shared_nameAdam/conv2d_35/kernel/m

+Adam/conv2d_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/m*'
_output_shapes
:Д*
dtype0

Adam/conv2d_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*&
shared_nameAdam/conv2d_35/bias/m
|
)Adam/conv2d_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/m*
_output_shapes	
:Д*
dtype0

Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*'
shared_nameAdam/dense_35/kernel/m

*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes
:	ФP*
dtype0

Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
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
Adam/conv2d_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*(
shared_nameAdam/conv2d_35/kernel/v

+Adam/conv2d_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/v*'
_output_shapes
:Д*
dtype0

Adam/conv2d_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Д*&
shared_nameAdam/conv2d_35/bias/v
|
)Adam/conv2d_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/v*
_output_shapes	
:Д*
dtype0

Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ФP*'
shared_nameAdam/dense_35/kernel/v

*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes
:	ФP*
dtype0

Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
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
о.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*.
value.B. B.
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
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
R
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
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
	variables
regularization_losses
trainable_variables
;layer_regularization_losses
<non_trainable_variables

=layers
>layer_metrics
?metrics
 
\Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
regularization_losses
trainable_variables
@layer_regularization_losses
Anon_trainable_variables

Blayers
Clayer_metrics
Dmetrics
 
 
 
­
	variables
regularization_losses
trainable_variables
Elayer_regularization_losses
Fnon_trainable_variables

Glayers
Hlayer_metrics
Imetrics
 
 
 
­
	variables
regularization_losses
trainable_variables
Jlayer_regularization_losses
Knon_trainable_variables

Llayers
Mlayer_metrics
Nmetrics
 
 
 
­
	variables
regularization_losses
 trainable_variables
Olayer_regularization_losses
Pnon_trainable_variables

Qlayers
Rlayer_metrics
Smetrics
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_35/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
$	variables
%regularization_losses
&trainable_variables
Tlayer_regularization_losses
Unon_trainable_variables

Vlayers
Wlayer_metrics
Xmetrics
 
 
 
­
(	variables
)regularization_losses
*trainable_variables
Ylayer_regularization_losses
Znon_trainable_variables

[layers
\layer_metrics
]metrics
 
 
 
­
,	variables
-regularization_losses
.trainable_variables
^layer_regularization_losses
_non_trainable_variables

`layers
alayer_metrics
bmetrics
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
2	variables
3regularization_losses
4trainable_variables
clayer_regularization_losses
dnon_trainable_variables

elayers
flayer_metrics
gmetrics
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
VARIABLE_VALUEAdam/conv2d_35/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_35/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_35/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_35/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
Ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv2d_35/kernelconv2d_35/biasdense_35/kerneldense_35/biasoutput/kerneloutput/bias*
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
%__inference_signature_wrapper_6282827
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
я	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_35/kernel/m/Read/ReadVariableOp)Adam/conv2d_35/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv2d_35/kernel/v/Read/ReadVariableOp)Adam/conv2d_35/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
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
 __inference__traced_save_6283198
і
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_35/kernelconv2d_35/biasdense_35/kerneldense_35/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_35/kernel/mAdam/conv2d_35/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_35/kernel/vAdam/conv2d_35/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/vAdam/output/kernel/vAdam/output/bias/v*%
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
#__inference__traced_restore_6283283Фщ
ѕ
f
H__inference_dropout_106_layer_call_and_return_conditional_losses_6283032

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
ы
c
G__inference_flatten_35_layer_call_and_return_conditional_losses_6283002

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
"
Ї
E__inference_model_35_layer_call_and_return_conditional_losses_6282778
input_onehot,
conv2d_35_6282757:Д 
conv2d_35_6282759:	Д#
dense_35_6282765:	ФP
dense_35_6282767:P 
output_6282772:P
output_6282774:
identityЂ!conv2d_35/StatefulPartitionedCallЂ dense_35/StatefulPartitionedCallЂoutput/StatefulPartitionedCallА
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_35_6282757conv2d_35_6282759*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_62824772#
!conv2d_35/StatefulPartitionedCall
re_lu_35/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_62824882
re_lu_35/PartitionedCall
dropout_105/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_105_layer_call_and_return_conditional_losses_62824952
dropout_105/PartitionedCallџ
flatten_35/PartitionedCallPartitionedCall$dropout_105/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_flatten_35_layer_call_and_return_conditional_losses_62825032
flatten_35/PartitionedCallЙ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_6282765dense_35_6282767*
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
GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_62825162"
 dense_35/StatefulPartitionedCall
dropout_106/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_106_layer_call_and_return_conditional_losses_62825272
dropout_106/PartitionedCall
dropout_107/PartitionedCallPartitionedCall$dropout_106/PartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_107_layer_call_and_return_conditional_losses_62825342
dropout_107/PartitionedCallА
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_107/PartitionedCall:output:0output_6282772output_6282774*
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
C__inference_output_layer_call_and_return_conditional_losses_62825462 
output/StatefulPartitionedCallу
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot

a
E__inference_re_lu_35_layer_call_and_return_conditional_losses_6282488

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
Ў
g
H__inference_dropout_106_layer_call_and_return_conditional_losses_6282621

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
Ѕ

*__inference_dense_35_layer_call_fn_6283027

inputs
unknown:	ФP
	unknown_0:P
identityЂStatefulPartitionedCallњ
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
GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_62825162
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
К


F__inference_conv2d_35_layer_call_and_return_conditional_losses_6282477

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
э
I
-__inference_dropout_105_layer_call_fn_6282991

inputs
identityд
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_105_layer_call_and_return_conditional_losses_62824952
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
­*
м
"__inference__wrapped_model_6282460
input_onehotL
1model_35_conv2d_35_conv2d_readvariableop_resource:ДA
2model_35_conv2d_35_biasadd_readvariableop_resource:	ДC
0model_35_dense_35_matmul_readvariableop_resource:	ФP?
1model_35_dense_35_biasadd_readvariableop_resource:P@
.model_35_output_matmul_readvariableop_resource:P=
/model_35_output_biasadd_readvariableop_resource:
identityЂ)model_35/conv2d_35/BiasAdd/ReadVariableOpЂ(model_35/conv2d_35/Conv2D/ReadVariableOpЂ(model_35/dense_35/BiasAdd/ReadVariableOpЂ'model_35/dense_35/MatMul/ReadVariableOpЂ&model_35/output/BiasAdd/ReadVariableOpЂ%model_35/output/MatMul/ReadVariableOpЯ
(model_35/conv2d_35/Conv2D/ReadVariableOpReadVariableOp1model_35_conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02*
(model_35/conv2d_35/Conv2D/ReadVariableOpф
model_35/conv2d_35/Conv2DConv2Dinput_onehot0model_35/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
model_35/conv2d_35/Conv2DЦ
)model_35/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp2model_35_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02+
)model_35/conv2d_35/BiasAdd/ReadVariableOpе
model_35/conv2d_35/BiasAddBiasAdd"model_35/conv2d_35/Conv2D:output:01model_35/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_35/conv2d_35/BiasAdd
model_35/re_lu_35/ReluRelu#model_35/conv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_35/re_lu_35/ReluЋ
model_35/dropout_105/IdentityIdentity$model_35/re_lu_35/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2
model_35/dropout_105/Identity
model_35/flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
model_35/flatten_35/ConstФ
model_35/flatten_35/ReshapeReshape&model_35/dropout_105/Identity:output:0"model_35/flatten_35/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
model_35/flatten_35/ReshapeФ
'model_35/dense_35/MatMul/ReadVariableOpReadVariableOp0model_35_dense_35_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02)
'model_35/dense_35/MatMul/ReadVariableOpЧ
model_35/dense_35/MatMulMatMul$model_35/flatten_35/Reshape:output:0/model_35/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_35/dense_35/MatMulТ
(model_35/dense_35/BiasAdd/ReadVariableOpReadVariableOp1model_35_dense_35_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02*
(model_35/dense_35/BiasAdd/ReadVariableOpЩ
model_35/dense_35/BiasAddBiasAdd"model_35/dense_35/MatMul:product:00model_35/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_35/dense_35/BiasAdd
model_35/dense_35/ReluRelu"model_35/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_35/dense_35/ReluЂ
model_35/dropout_106/IdentityIdentity$model_35/dense_35/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_35/dropout_106/IdentityЄ
model_35/dropout_107/IdentityIdentity&model_35/dropout_106/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
model_35/dropout_107/IdentityН
%model_35/output/MatMul/ReadVariableOpReadVariableOp.model_35_output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02'
%model_35/output/MatMul/ReadVariableOpУ
model_35/output/MatMulMatMul&model_35/dropout_107/Identity:output:0-model_35/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_35/output/MatMulМ
&model_35/output/BiasAdd/ReadVariableOpReadVariableOp/model_35_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_35/output/BiasAdd/ReadVariableOpС
model_35/output/BiasAddBiasAdd model_35/output/MatMul:product:0.model_35/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_35/output/BiasAddё
IdentityIdentity model_35/output/BiasAdd:output:0*^model_35/conv2d_35/BiasAdd/ReadVariableOp)^model_35/conv2d_35/Conv2D/ReadVariableOp)^model_35/dense_35/BiasAdd/ReadVariableOp(^model_35/dense_35/MatMul/ReadVariableOp'^model_35/output/BiasAdd/ReadVariableOp&^model_35/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2V
)model_35/conv2d_35/BiasAdd/ReadVariableOp)model_35/conv2d_35/BiasAdd/ReadVariableOp2T
(model_35/conv2d_35/Conv2D/ReadVariableOp(model_35/conv2d_35/Conv2D/ReadVariableOp2T
(model_35/dense_35/BiasAdd/ReadVariableOp(model_35/dense_35/BiasAdd/ReadVariableOp2R
'model_35/dense_35/MatMul/ReadVariableOp'model_35/dense_35/MatMul/ReadVariableOp2P
&model_35/output/BiasAdd/ReadVariableOp&model_35/output/BiasAdd/ReadVariableOp2N
%model_35/output/MatMul/ReadVariableOp%model_35/output/MatMul/ReadVariableOp:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
;
Е

 __inference__traced_save_6283198
file_prefix/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_35_kernel_m_read_readvariableop4
0savev2_adam_conv2d_35_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv2d_35_kernel_v_read_readvariableop4
0savev2_adam_conv2d_35_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop3
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
SaveV2/shape_and_slicesЗ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_35_kernel_m_read_readvariableop0savev2_adam_conv2d_35_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv2d_35_kernel_v_read_readvariableop0savev2_adam_conv2d_35_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ї

*__inference_model_35_layer_call_fn_6282754
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallД
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
GPU2 *0J 8 *N
fIRG
E__inference_model_35_layer_call_and_return_conditional_losses_62827222
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
ј
g
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282986

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
ї

*__inference_model_35_layer_call_fn_6282568
input_onehot"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallД
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
GPU2 *0J 8 *N
fIRG
E__inference_model_35_layer_call_and_return_conditional_losses_62825532
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
Щ
I
-__inference_dropout_107_layer_call_fn_6283076

inputs
identityЫ
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_107_layer_call_and_return_conditional_losses_62825342
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
е
f
-__inference_dropout_107_layer_call_fn_6283081

inputs
identityЂStatefulPartitionedCallу
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_107_layer_call_and_return_conditional_losses_62825982
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

a
E__inference_re_lu_35_layer_call_and_return_conditional_losses_6282964

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
е
f
-__inference_dropout_106_layer_call_fn_6283054

inputs
identityЂStatefulPartitionedCallу
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_106_layer_call_and_return_conditional_losses_62826212
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
ы&

E__inference_model_35_layer_call_and_return_conditional_losses_6282722

inputs,
conv2d_35_6282701:Д 
conv2d_35_6282703:	Д#
dense_35_6282709:	ФP
dense_35_6282711:P 
output_6282716:P
output_6282718:
identityЂ!conv2d_35/StatefulPartitionedCallЂ dense_35/StatefulPartitionedCallЂ#dropout_105/StatefulPartitionedCallЂ#dropout_106/StatefulPartitionedCallЂ#dropout_107/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЊ
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_6282701conv2d_35_6282703*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_62824772#
!conv2d_35/StatefulPartitionedCall
re_lu_35/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_62824882
re_lu_35/PartitionedCall
#dropout_105/StatefulPartitionedCallStatefulPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_105_layer_call_and_return_conditional_losses_62826602%
#dropout_105/StatefulPartitionedCall
flatten_35/PartitionedCallPartitionedCall,dropout_105/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_flatten_35_layer_call_and_return_conditional_losses_62825032
flatten_35/PartitionedCallЙ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_6282709dense_35_6282711*
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
GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_62825162"
 dense_35/StatefulPartitionedCallФ
#dropout_106/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0$^dropout_105/StatefulPartitionedCall*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_106_layer_call_and_return_conditional_losses_62826212%
#dropout_106/StatefulPartitionedCallЧ
#dropout_107/StatefulPartitionedCallStatefulPartitionedCall,dropout_106/StatefulPartitionedCall:output:0$^dropout_106/StatefulPartitionedCall*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_107_layer_call_and_return_conditional_losses_62825982%
#dropout_107/StatefulPartitionedCallИ
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_107/StatefulPartitionedCall:output:0output_6282716output_6282718*
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
C__inference_output_layer_call_and_return_conditional_losses_62825462 
output/StatefulPartitionedCallе
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall$^dropout_105/StatefulPartitionedCall$^dropout_106/StatefulPartitionedCall$^dropout_107/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2J
#dropout_105/StatefulPartitionedCall#dropout_105/StatefulPartitionedCall2J
#dropout_106/StatefulPartitionedCall#dropout_106/StatefulPartitionedCall2J
#dropout_107/StatefulPartitionedCall#dropout_107/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
g
H__inference_dropout_106_layer_call_and_return_conditional_losses_6283044

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
а
Ђ
+__inference_conv2d_35_layer_call_fn_6282959

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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_62824772
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
ј
g
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282660

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
л
H
,__inference_flatten_35_layer_call_fn_6283007

inputs
identityЫ
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
GPU2 *0J 8 *P
fKRI
G__inference_flatten_35_layer_call_and_return_conditional_losses_62825032
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


(__inference_output_layer_call_fn_6283100

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
C__inference_output_layer_call_and_return_conditional_losses_62825462
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
К


F__inference_conv2d_35_layer_call_and_return_conditional_losses_6282950

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
§&

E__inference_model_35_layer_call_and_return_conditional_losses_6282802
input_onehot,
conv2d_35_6282781:Д 
conv2d_35_6282783:	Д#
dense_35_6282789:	ФP
dense_35_6282791:P 
output_6282796:P
output_6282798:
identityЂ!conv2d_35/StatefulPartitionedCallЂ dense_35/StatefulPartitionedCallЂ#dropout_105/StatefulPartitionedCallЂ#dropout_106/StatefulPartitionedCallЂ#dropout_107/StatefulPartitionedCallЂoutput/StatefulPartitionedCallА
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv2d_35_6282781conv2d_35_6282783*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_62824772#
!conv2d_35/StatefulPartitionedCall
re_lu_35/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_62824882
re_lu_35/PartitionedCall
#dropout_105/StatefulPartitionedCallStatefulPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_105_layer_call_and_return_conditional_losses_62826602%
#dropout_105/StatefulPartitionedCall
flatten_35/PartitionedCallPartitionedCall,dropout_105/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_flatten_35_layer_call_and_return_conditional_losses_62825032
flatten_35/PartitionedCallЙ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_6282789dense_35_6282791*
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
GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_62825162"
 dense_35/StatefulPartitionedCallФ
#dropout_106/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0$^dropout_105/StatefulPartitionedCall*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_106_layer_call_and_return_conditional_losses_62826212%
#dropout_106/StatefulPartitionedCallЧ
#dropout_107/StatefulPartitionedCallStatefulPartitionedCall,dropout_106/StatefulPartitionedCall:output:0$^dropout_106/StatefulPartitionedCall*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_107_layer_call_and_return_conditional_losses_62825982%
#dropout_107/StatefulPartitionedCallИ
output/StatefulPartitionedCallStatefulPartitionedCall,dropout_107/StatefulPartitionedCall:output:0output_6282796output_6282798*
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
C__inference_output_layer_call_and_return_conditional_losses_62825462 
output/StatefulPartitionedCallе
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall$^dropout_105/StatefulPartitionedCall$^dropout_106/StatefulPartitionedCall$^dropout_107/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2J
#dropout_105/StatefulPartitionedCall#dropout_105/StatefulPartitionedCall2J
#dropout_106/StatefulPartitionedCall#dropout_106/StatefulPartitionedCall2J
#dropout_107/StatefulPartitionedCall#dropout_107/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_onehot
љ
f
-__inference_dropout_105_layer_call_fn_6282996

inputs
identityЂStatefulPartitionedCallь
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_105_layer_call_and_return_conditional_losses_62826602
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
ы
c
G__inference_flatten_35_layer_call_and_return_conditional_losses_6282503

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
Ў
g
H__inference_dropout_107_layer_call_and_return_conditional_losses_6283071

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
ѕ
f
H__inference_dropout_106_layer_call_and_return_conditional_losses_6282527

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
ѕ
f
H__inference_dropout_107_layer_call_and_return_conditional_losses_6282534

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

f
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282495

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
Б

ї
E__inference_dense_35_layer_call_and_return_conditional_losses_6282516

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
Ў
g
H__inference_dropout_107_layer_call_and_return_conditional_losses_6282598

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
ѕ
f
H__inference_dropout_107_layer_call_and_return_conditional_losses_6283059

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
х

*__inference_model_35_layer_call_fn_6282923

inputs"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallЎ
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
GPU2 *0J 8 *N
fIRG
E__inference_model_35_layer_call_and_return_conditional_losses_62825532
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
Я

%__inference_signature_wrapper_6282827
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
"__inference__wrapped_model_62824602
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
ї!
Ё
E__inference_model_35_layer_call_and_return_conditional_losses_6282553

inputs,
conv2d_35_6282478:Д 
conv2d_35_6282480:	Д#
dense_35_6282517:	ФP
dense_35_6282519:P 
output_6282547:P
output_6282549:
identityЂ!conv2d_35/StatefulPartitionedCallЂ dense_35/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЊ
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_6282478conv2d_35_6282480*
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_62824772#
!conv2d_35/StatefulPartitionedCall
re_lu_35/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_62824882
re_lu_35/PartitionedCall
dropout_105/PartitionedCallPartitionedCall!re_lu_35/PartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_105_layer_call_and_return_conditional_losses_62824952
dropout_105/PartitionedCallџ
flatten_35/PartitionedCallPartitionedCall$dropout_105/PartitionedCall:output:0*
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
GPU2 *0J 8 *P
fKRI
G__inference_flatten_35_layer_call_and_return_conditional_losses_62825032
flatten_35/PartitionedCallЙ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_6282517dense_35_6282519*
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
GPU2 *0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_62825162"
 dense_35/StatefulPartitionedCall
dropout_106/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_106_layer_call_and_return_conditional_losses_62825272
dropout_106/PartitionedCall
dropout_107/PartitionedCallPartitionedCall$dropout_106/PartitionedCall:output:0*
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_107_layer_call_and_return_conditional_losses_62825342
dropout_107/PartitionedCallА
output/StatefulPartitionedCallStatefulPartitionedCall$dropout_107/PartitionedCall:output:0output_6282547output_6282549*
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
C__inference_output_layer_call_and_return_conditional_losses_62825462 
output/StatefulPartitionedCallу
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
m

#__inference__traced_restore_6283283
file_prefix<
!assignvariableop_conv2d_35_kernel:Д0
!assignvariableop_1_conv2d_35_bias:	Д5
"assignvariableop_2_dense_35_kernel:	ФP.
 assignvariableop_3_dense_35_bias:P2
 assignvariableop_4_output_kernel:P,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: F
+assignvariableop_13_adam_conv2d_35_kernel_m:Д8
)assignvariableop_14_adam_conv2d_35_bias_m:	Д=
*assignvariableop_15_adam_dense_35_kernel_m:	ФP6
(assignvariableop_16_adam_dense_35_bias_m:P:
(assignvariableop_17_adam_output_kernel_m:P4
&assignvariableop_18_adam_output_bias_m:F
+assignvariableop_19_adam_conv2d_35_kernel_v:Д8
)assignvariableop_20_adam_conv2d_35_bias_v:	Д=
*assignvariableop_21_adam_dense_35_kernel_v:	ФP6
(assignvariableop_22_adam_dense_35_bias_v:P:
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_35_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_35_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_35_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_35_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_35_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Б
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_35_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15В
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_35_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16А
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_35_bias_mIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_35_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Б
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_35_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_35_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_35_bias_vIdentity_22:output:0"/device:CPU:0*
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
і#

E__inference_model_35_layer_call_and_return_conditional_losses_6282856

inputsC
(conv2d_35_conv2d_readvariableop_resource:Д8
)conv2d_35_biasadd_readvariableop_resource:	Д:
'dense_35_matmul_readvariableop_resource:	ФP6
(dense_35_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityЂ conv2d_35/BiasAdd/ReadVariableOpЂconv2d_35/Conv2D/ReadVariableOpЂdense_35/BiasAdd/ReadVariableOpЂdense_35/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpД
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02!
conv2d_35/Conv2D/ReadVariableOpУ
conv2d_35/Conv2DConv2Dinputs'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
conv2d_35/Conv2DЋ
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOpБ
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
conv2d_35/BiasAdd}
re_lu_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
re_lu_35/Relu
dropout_105/IdentityIdentityre_lu_35/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_105/Identityu
flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
flatten_35/Const 
flatten_35/ReshapeReshapedropout_105/Identity:output:0flatten_35/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
flatten_35/ReshapeЉ
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02 
dense_35/MatMul/ReadVariableOpЃ
dense_35/MatMulMatMulflatten_35/Reshape:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_35/MatMulЇ
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
dense_35/BiasAdd/ReadVariableOpЅ
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_35/BiasAdds
dense_35/ReluReludense_35/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_35/Relu
dropout_106/IdentityIdentitydense_35/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_106/Identity
dropout_107/IdentityIdentitydropout_106/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_107/IdentityЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_107/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAddВ
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
A

E__inference_model_35_layer_call_and_return_conditional_losses_6282906

inputsC
(conv2d_35_conv2d_readvariableop_resource:Д8
)conv2d_35_biasadd_readvariableop_resource:	Д:
'dense_35_matmul_readvariableop_resource:	ФP6
(dense_35_biasadd_readvariableop_resource:P7
%output_matmul_readvariableop_resource:P4
&output_biasadd_readvariableop_resource:
identityЂ conv2d_35/BiasAdd/ReadVariableOpЂconv2d_35/Conv2D/ReadVariableOpЂdense_35/BiasAdd/ReadVariableOpЂdense_35/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpД
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:Д*
dtype02!
conv2d_35/Conv2D/ReadVariableOpУ
conv2d_35/Conv2DConv2Dinputs'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД*
paddingVALID*
strides
2
conv2d_35/Conv2DЋ
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOpБ
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџД2
conv2d_35/BiasAdd}
re_lu_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
re_lu_35/Relu{
dropout_105/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_105/dropout/ConstЕ
dropout_105/dropout/MulMulre_lu_35/Relu:activations:0"dropout_105/dropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_105/dropout/Mul
dropout_105/dropout/ShapeShapere_lu_35/Relu:activations:0*
T0*
_output_shapes
:2
dropout_105/dropout/Shapeс
0dropout_105/dropout/random_uniform/RandomUniformRandomUniform"dropout_105/dropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџД*
dtype022
0dropout_105/dropout/random_uniform/RandomUniform
"dropout_105/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_105/dropout/GreaterEqual/yї
 dropout_105/dropout/GreaterEqualGreaterEqual9dropout_105/dropout/random_uniform/RandomUniform:output:0+dropout_105/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџД2"
 dropout_105/dropout/GreaterEqualЌ
dropout_105/dropout/CastCast$dropout_105/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџД2
dropout_105/dropout/CastГ
dropout_105/dropout/Mul_1Muldropout_105/dropout/Mul:z:0dropout_105/dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџД2
dropout_105/dropout/Mul_1u
flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџФ  2
flatten_35/Const 
flatten_35/ReshapeReshapedropout_105/dropout/Mul_1:z:0flatten_35/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџФ2
flatten_35/ReshapeЉ
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	ФP*
dtype02 
dense_35/MatMul/ReadVariableOpЃ
dense_35/MatMulMatMulflatten_35/Reshape:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_35/MatMulЇ
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
dense_35/BiasAdd/ReadVariableOpЅ
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_35/BiasAdds
dense_35/ReluReludense_35/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dense_35/Relu{
dropout_106/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_106/dropout/ConstЌ
dropout_106/dropout/MulMuldense_35/Relu:activations:0"dropout_106/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_106/dropout/Mul
dropout_106/dropout/ShapeShapedense_35/Relu:activations:0*
T0*
_output_shapes
:2
dropout_106/dropout/Shapeи
0dropout_106/dropout/random_uniform/RandomUniformRandomUniform"dropout_106/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype022
0dropout_106/dropout/random_uniform/RandomUniform
"dropout_106/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_106/dropout/GreaterEqual/yю
 dropout_106/dropout/GreaterEqualGreaterEqual9dropout_106/dropout/random_uniform/RandomUniform:output:0+dropout_106/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2"
 dropout_106/dropout/GreaterEqualЃ
dropout_106/dropout/CastCast$dropout_106/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout_106/dropout/CastЊ
dropout_106/dropout/Mul_1Muldropout_106/dropout/Mul:z:0dropout_106/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_106/dropout/Mul_1{
dropout_107/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_107/dropout/ConstЎ
dropout_107/dropout/MulMuldropout_106/dropout/Mul_1:z:0"dropout_107/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_107/dropout/Mul
dropout_107/dropout/ShapeShapedropout_106/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dropout_107/dropout/Shapeи
0dropout_107/dropout/random_uniform/RandomUniformRandomUniform"dropout_107/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџP*
dtype022
0dropout_107/dropout/random_uniform/RandomUniform
"dropout_107/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_107/dropout/GreaterEqual/yю
 dropout_107/dropout/GreaterEqualGreaterEqual9dropout_107/dropout/random_uniform/RandomUniform:output:0+dropout_107/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџP2"
 dropout_107/dropout/GreaterEqualЃ
dropout_107/dropout/CastCast$dropout_107/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџP2
dropout_107/dropout/CastЊ
dropout_107/dropout/Mul_1Muldropout_107/dropout/Mul:z:0dropout_107/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP2
dropout_107/dropout/Mul_1Ђ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_107/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAddВ
IdentityIdentityoutput/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х

*__inference_model_35_layer_call_fn_6282940

inputs"
unknown:Д
	unknown_0:	Д
	unknown_1:	ФP
	unknown_2:P
	unknown_3:P
	unknown_4:
identityЂStatefulPartitionedCallЎ
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
GPU2 *0J 8 *N
fIRG
E__inference_model_35_layer_call_and_return_conditional_losses_62827222
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
ч
F
*__inference_re_lu_35_layer_call_fn_6282969

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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_62824882
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
Б

ї
E__inference_dense_35_layer_call_and_return_conditional_losses_6283018

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
Я	
є
C__inference_output_layer_call_and_return_conditional_losses_6283091

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

f
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282974

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
Щ
I
-__inference_dropout_106_layer_call_fn_6283049

inputs
identityЫ
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
GPU2 *0J 8 *Q
fLRJ
H__inference_dropout_106_layer_call_and_return_conditional_losses_62825272
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
Я	
є
C__inference_output_layer_call_and_return_conditional_losses_6282546

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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:їџ
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*y&call_and_return_all_conditional_losses
z__call__
{_default_save_signature"=
_tf_keras_networkю<{"name": "model_35", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_35", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_35", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_105", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_105", "inbound_nodes": [[["re_lu_35", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_35", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_35", "inbound_nodes": [[["dropout_105", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["flatten_35", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_106", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_106", "inbound_nodes": [[["dense_35", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_107", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_107", "inbound_nodes": [[["dropout_106", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_107", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_35", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "ReLU", "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_35", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_105", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_105", "inbound_nodes": [[["re_lu_35", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten_35", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_35", "inbound_nodes": [[["dropout_105", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["flatten_35", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_106", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_106", "inbound_nodes": [[["dense_35", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_107", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_107", "inbound_nodes": [[["dropout_106", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_107", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"ў
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*|&call_and_return_all_conditional_losses
}__call__"о	
_tf_keras_layerФ	{"name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 23, 4]}}
В
	variables
regularization_losses
trainable_variables
	keras_api
*~&call_and_return_all_conditional_losses
__call__"Ѓ
_tf_keras_layer{"name": "re_lu_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_35", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv2d_35", 0, 0, {}]]], "shared_object_id": 4}
Џ
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"name": "dropout_105", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_105", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["re_lu_35", 0, 0, {}]]], "shared_object_id": 5}
Щ
	variables
regularization_losses
 trainable_variables
!	keras_api
+&call_and_return_all_conditional_losses
__call__"И
_tf_keras_layer{"name": "flatten_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_35", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["dropout_105", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
	

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"н
_tf_keras_layerУ{"name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_35", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3780}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3780]}}
А
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"name": "dropout_106", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_106", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_35", 0, 0, {}]]], "shared_object_id": 10}
Г
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+&call_and_return_all_conditional_losses
__call__"Ђ
_tf_keras_layer{"name": "dropout_107", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_107", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dropout_106", 0, 0, {}]]], "shared_object_id": 11}
	

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+&call_and_return_all_conditional_losses
__call__"к
_tf_keras_layerР{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_107", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
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
	variables
regularization_losses
trainable_variables
;layer_regularization_losses
<non_trainable_variables

=layers
>layer_metrics
?metrics
z__call__
{_default_save_signature
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
+:)Д2conv2d_35/kernel
:Д2conv2d_35/bias
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
	variables
regularization_losses
trainable_variables
@layer_regularization_losses
Anon_trainable_variables

Blayers
Clayer_metrics
Dmetrics
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
	variables
regularization_losses
trainable_variables
Elayer_regularization_losses
Fnon_trainable_variables

Glayers
Hlayer_metrics
Imetrics
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
	variables
regularization_losses
trainable_variables
Jlayer_regularization_losses
Knon_trainable_variables

Llayers
Mlayer_metrics
Nmetrics
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
	variables
regularization_losses
 trainable_variables
Olayer_regularization_losses
Pnon_trainable_variables

Qlayers
Rlayer_metrics
Smetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 	ФP2dense_35/kernel
:P2dense_35/bias
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
$	variables
%regularization_losses
&trainable_variables
Tlayer_regularization_losses
Unon_trainable_variables

Vlayers
Wlayer_metrics
Xmetrics
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
(	variables
)regularization_losses
*trainable_variables
Ylayer_regularization_losses
Znon_trainable_variables

[layers
\layer_metrics
]metrics
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
,	variables
-regularization_losses
.trainable_variables
^layer_regularization_losses
_non_trainable_variables

`layers
alayer_metrics
bmetrics
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
2	variables
3regularization_losses
4trainable_variables
clayer_regularization_losses
dnon_trainable_variables

elayers
flayer_metrics
gmetrics
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
'
h0"
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
0:.Д2Adam/conv2d_35/kernel/m
": Д2Adam/conv2d_35/bias/m
':%	ФP2Adam/dense_35/kernel/m
 :P2Adam/dense_35/bias/m
$:"P2Adam/output/kernel/m
:2Adam/output/bias/m
0:.Д2Adam/conv2d_35/kernel/v
": Д2Adam/conv2d_35/bias/v
':%	ФP2Adam/dense_35/kernel/v
 :P2Adam/dense_35/bias/v
$:"P2Adam/output/kernel/v
:2Adam/output/bias/v
т2п
E__inference_model_35_layer_call_and_return_conditional_losses_6282856
E__inference_model_35_layer_call_and_return_conditional_losses_6282906
E__inference_model_35_layer_call_and_return_conditional_losses_6282778
E__inference_model_35_layer_call_and_return_conditional_losses_6282802Р
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
і2ѓ
*__inference_model_35_layer_call_fn_6282568
*__inference_model_35_layer_call_fn_6282923
*__inference_model_35_layer_call_fn_6282940
*__inference_model_35_layer_call_fn_6282754Р
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
"__inference__wrapped_model_6282460У
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
F__inference_conv2d_35_layer_call_and_return_conditional_losses_6282950Ђ
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
+__inference_conv2d_35_layer_call_fn_6282959Ђ
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
E__inference_re_lu_35_layer_call_and_return_conditional_losses_6282964Ђ
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
*__inference_re_lu_35_layer_call_fn_6282969Ђ
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
Ю2Ы
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282974
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282986Д
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
2
-__inference_dropout_105_layer_call_fn_6282991
-__inference_dropout_105_layer_call_fn_6282996Д
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
ё2ю
G__inference_flatten_35_layer_call_and_return_conditional_losses_6283002Ђ
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
,__inference_flatten_35_layer_call_fn_6283007Ђ
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
E__inference_dense_35_layer_call_and_return_conditional_losses_6283018Ђ
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
*__inference_dense_35_layer_call_fn_6283027Ђ
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
Ю2Ы
H__inference_dropout_106_layer_call_and_return_conditional_losses_6283032
H__inference_dropout_106_layer_call_and_return_conditional_losses_6283044Д
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
2
-__inference_dropout_106_layer_call_fn_6283049
-__inference_dropout_106_layer_call_fn_6283054Д
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
Ю2Ы
H__inference_dropout_107_layer_call_and_return_conditional_losses_6283059
H__inference_dropout_107_layer_call_and_return_conditional_losses_6283071Д
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
2
-__inference_dropout_107_layer_call_fn_6283076
-__inference_dropout_107_layer_call_fn_6283081Д
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
C__inference_output_layer_call_and_return_conditional_losses_6283091Ђ
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
(__inference_output_layer_call_fn_6283100Ђ
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
%__inference_signature_wrapper_6282827input_onehot"
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
"__inference__wrapped_model_6282460x"#01=Ђ:
3Ђ0
.+
input_onehotџџџџџџџџџ
Њ "/Њ,
*
output 
outputџџџџџџџџџЗ
F__inference_conv2d_35_layer_call_and_return_conditional_losses_6282950m7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџД
 
+__inference_conv2d_35_layer_call_fn_6282959`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "!џџџџџџџџџДІ
E__inference_dense_35_layer_call_and_return_conditional_losses_6283018]"#0Ђ-
&Ђ#
!
inputsџџџџџџџџџФ
Њ "%Ђ"

0џџџџџџџџџP
 ~
*__inference_dense_35_layer_call_fn_6283027P"#0Ђ-
&Ђ#
!
inputsџџџџџџџџџФ
Њ "џџџџџџџџџPК
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282974n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p 
Њ ".Ђ+
$!
0џџџџџџџџџД
 К
H__inference_dropout_105_layer_call_and_return_conditional_losses_6282986n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p
Њ ".Ђ+
$!
0џџџџџџџџџД
 
-__inference_dropout_105_layer_call_fn_6282991a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p 
Њ "!џџџџџџџџџД
-__inference_dropout_105_layer_call_fn_6282996a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџД
p
Њ "!џџџџџџџџџДЈ
H__inference_dropout_106_layer_call_and_return_conditional_losses_6283032\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "%Ђ"

0џџџџџџџџџP
 Ј
H__inference_dropout_106_layer_call_and_return_conditional_losses_6283044\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "%Ђ"

0џџџџџџџџџP
 
-__inference_dropout_106_layer_call_fn_6283049O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "џџџџџџџџџP
-__inference_dropout_106_layer_call_fn_6283054O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "џџџџџџџџџPЈ
H__inference_dropout_107_layer_call_and_return_conditional_losses_6283059\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "%Ђ"

0џџџџџџџџџP
 Ј
H__inference_dropout_107_layer_call_and_return_conditional_losses_6283071\3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "%Ђ"

0џџџџџџџџџP
 
-__inference_dropout_107_layer_call_fn_6283076O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p 
Њ "џџџџџџџџџP
-__inference_dropout_107_layer_call_fn_6283081O3Ђ0
)Ђ&
 
inputsџџџџџџџџџP
p
Њ "џџџџџџџџџP­
G__inference_flatten_35_layer_call_and_return_conditional_losses_6283002b8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "&Ђ#

0џџџџџџџџџФ
 
,__inference_flatten_35_layer_call_fn_6283007U8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "џџџџџџџџџФП
E__inference_model_35_layer_call_and_return_conditional_losses_6282778v"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 П
E__inference_model_35_layer_call_and_return_conditional_losses_6282802v"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Й
E__inference_model_35_layer_call_and_return_conditional_losses_6282856p"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Й
E__inference_model_35_layer_call_and_return_conditional_losses_6282906p"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
*__inference_model_35_layer_call_fn_6282568i"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
*__inference_model_35_layer_call_fn_6282754i"#01EЂB
;Ђ8
.+
input_onehotџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
*__inference_model_35_layer_call_fn_6282923c"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
*__inference_model_35_layer_call_fn_6282940c"#01?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЃ
C__inference_output_layer_call_and_return_conditional_losses_6283091\01/Ђ,
%Ђ"
 
inputsџџџџџџџџџP
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_output_layer_call_fn_6283100O01/Ђ,
%Ђ"
 
inputsџџџџџџџџџP
Њ "џџџџџџџџџГ
E__inference_re_lu_35_layer_call_and_return_conditional_losses_6282964j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ ".Ђ+
$!
0џџџџџџџџџД
 
*__inference_re_lu_35_layer_call_fn_6282969]8Ђ5
.Ђ+
)&
inputsџџџџџџџџџД
Њ "!џџџџџџџџџДВ
%__inference_signature_wrapper_6282827"#01MЂJ
Ђ 
CЊ@
>
input_onehot.+
input_onehotџџџџџџџџџ"/Њ,
*
output 
outputџџџџџџџџџ