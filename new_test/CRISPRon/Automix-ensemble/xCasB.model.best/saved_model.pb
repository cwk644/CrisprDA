��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
�
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
z
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv_3/kernel
s
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*"
_output_shapes
:d*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:d*
dtype0
z
conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_nameconv_5/kernel
s
!conv_5/kernel/Read/ReadVariableOpReadVariableOpconv_5/kernel*"
_output_shapes
:F*
dtype0
n
conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_nameconv_5/bias
g
conv_5/bias/Read/ReadVariableOpReadVariableOpconv_5/bias*
_output_shapes
:F*
dtype0
z
conv_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv_7/kernel
s
!conv_7/kernel/Read/ReadVariableOpReadVariableOpconv_7/kernel*"
_output_shapes
:(*
dtype0
n
conv_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv_7/bias
g
conv_7/bias/Read/ReadVariableOpReadVariableOpconv_7/bias*
_output_shapes
:(*
dtype0
y
dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*
shared_namedense_0/kernel
r
"dense_0/kernel/Read/ReadVariableOpReadVariableOpdense_0/kernel*
_output_shapes
:	�P*
dtype0
p
dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_0/bias
i
 dense_0/bias/Read/ReadVariableOpReadVariableOpdense_0/bias*
_output_shapes
:P*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:PP*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:P*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P<*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:P<*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:<*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:<*
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
Adam/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv_3/kernel/m
�
(Adam/conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/m*"
_output_shapes
:d*
dtype0
|
Adam/conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/conv_3/bias/m
u
&Adam/conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/m*
_output_shapes
:d*
dtype0
�
Adam/conv_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*%
shared_nameAdam/conv_5/kernel/m
�
(Adam/conv_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_5/kernel/m*"
_output_shapes
:F*
dtype0
|
Adam/conv_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*#
shared_nameAdam/conv_5/bias/m
u
&Adam/conv_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_5/bias/m*
_output_shapes
:F*
dtype0
�
Adam/conv_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/conv_7/kernel/m
�
(Adam/conv_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_7/kernel/m*"
_output_shapes
:(*
dtype0
|
Adam/conv_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/conv_7/bias/m
u
&Adam/conv_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_7/bias/m*
_output_shapes
:(*
dtype0
�
Adam/dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*&
shared_nameAdam/dense_0/kernel/m
�
)Adam/dense_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_0/kernel/m*
_output_shapes
:	�P*
dtype0
~
Adam/dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameAdam/dense_0/bias/m
w
'Adam/dense_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_0/bias/m*
_output_shapes
:P*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:PP*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:P*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P<*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:P<*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:<*
dtype0
�
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:<*
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
Adam/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/conv_3/kernel/v
�
(Adam/conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/v*"
_output_shapes
:d*
dtype0
|
Adam/conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/conv_3/bias/v
u
&Adam/conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/v*
_output_shapes
:d*
dtype0
�
Adam/conv_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*%
shared_nameAdam/conv_5/kernel/v
�
(Adam/conv_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_5/kernel/v*"
_output_shapes
:F*
dtype0
|
Adam/conv_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*#
shared_nameAdam/conv_5/bias/v
u
&Adam/conv_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_5/bias/v*
_output_shapes
:F*
dtype0
�
Adam/conv_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/conv_7/kernel/v
�
(Adam/conv_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_7/kernel/v*"
_output_shapes
:(*
dtype0
|
Adam/conv_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/conv_7/bias/v
u
&Adam/conv_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_7/bias/v*
_output_shapes
:(*
dtype0
�
Adam/dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*&
shared_nameAdam/dense_0/kernel/v
�
)Adam/dense_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_0/kernel/v*
_output_shapes
:	�P*
dtype0
~
Adam/dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameAdam/dense_0/bias/v
w
'Adam/dense_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_0/bias/v*
_output_shapes
:P*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:PP*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:P*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P<*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:P<*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:<*
dtype0
�
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:<*
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
�e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�d
value�dB�d B�d
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-3
layer-14
layer-15
layer_with_weights-4
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer_with_weights-6
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
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
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
R
6	variables
7trainable_variables
8regularization_losses
9	keras_api
R
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
R
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
R
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
R
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
h

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
R
\	variables
]trainable_variables
^regularization_losses
_	keras_api
h

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
R
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
h

tkernel
ubias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
�
ziter

{beta_1

|beta_2
	}decay
~learning_ratem�m�"m�#m�(m�)m�Vm�Wm�`m�am�jm�km�tm�um�v�v�"v�#v�(v�)v�Vv�Wv�`v�av�jv�kv�tv�uv�
f
0
1
"2
#3
(4
)5
V6
W7
`8
a9
j10
k11
t12
u13
 
f
0
1
"2
#3
(4
)5
V6
W7
`8
a9
j10
k11
t12
u13
�
layer_metrics
	variables
�metrics
 �layer_regularization_losses
regularization_losses
�non_trainable_variables
�layers
trainable_variables
 
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
�metrics
trainable_variables
 �layer_regularization_losses
 regularization_losses
�non_trainable_variables
�layers
�layer_metrics
YW
VARIABLE_VALUEconv_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
$	variables
�metrics
%trainable_variables
 �layer_regularization_losses
&regularization_losses
�non_trainable_variables
�layers
�layer_metrics
YW
VARIABLE_VALUEconv_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
�
*	variables
�metrics
+trainable_variables
 �layer_regularization_losses
,regularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
.	variables
�metrics
/trainable_variables
 �layer_regularization_losses
0regularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
2	variables
�metrics
3trainable_variables
 �layer_regularization_losses
4regularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
6	variables
�metrics
7trainable_variables
 �layer_regularization_losses
8regularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
:	variables
�metrics
;trainable_variables
 �layer_regularization_losses
<regularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
>	variables
�metrics
?trainable_variables
 �layer_regularization_losses
@regularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
B	variables
�metrics
Ctrainable_variables
 �layer_regularization_losses
Dregularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
F	variables
�metrics
Gtrainable_variables
 �layer_regularization_losses
Hregularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
J	variables
�metrics
Ktrainable_variables
 �layer_regularization_losses
Lregularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
N	variables
�metrics
Otrainable_variables
 �layer_regularization_losses
Pregularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
R	variables
�metrics
Strainable_variables
 �layer_regularization_losses
Tregularization_losses
�non_trainable_variables
�layers
�layer_metrics
ZX
VARIABLE_VALUEdense_0/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_0/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

V0
W1
 
�
X	variables
�metrics
Ytrainable_variables
 �layer_regularization_losses
Zregularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
\	variables
�metrics
]trainable_variables
 �layer_regularization_losses
^regularization_losses
�non_trainable_variables
�layers
�layer_metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

`0
a1
 
�
b	variables
�metrics
ctrainable_variables
 �layer_regularization_losses
dregularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
f	variables
�metrics
gtrainable_variables
 �layer_regularization_losses
hregularization_losses
�non_trainable_variables
�layers
�layer_metrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
�
l	variables
�metrics
mtrainable_variables
 �layer_regularization_losses
nregularization_losses
�non_trainable_variables
�layers
�layer_metrics
 
 
 
�
p	variables
�metrics
qtrainable_variables
 �layer_regularization_losses
rregularization_losses
�non_trainable_variables
�layers
�layer_metrics
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

t0
u1
 
�
v	variables
�metrics
wtrainable_variables
 �layer_regularization_losses
xregularization_losses
�non_trainable_variables
�layers
�layer_metrics
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

�0
 
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
|z
VARIABLE_VALUEAdam/conv_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_0/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_0/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_0/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_0/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_onehotPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_onehotconv_7/kernelconv_7/biasconv_5/kernelconv_5/biasconv_3/kernelconv_3/biasdense_0/kerneldense_0/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_2020537
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp!conv_5/kernel/Read/ReadVariableOpconv_5/bias/Read/ReadVariableOp!conv_7/kernel/Read/ReadVariableOpconv_7/bias/Read/ReadVariableOp"dense_0/kernel/Read/ReadVariableOp dense_0/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv_3/kernel/m/Read/ReadVariableOp&Adam/conv_3/bias/m/Read/ReadVariableOp(Adam/conv_5/kernel/m/Read/ReadVariableOp&Adam/conv_5/bias/m/Read/ReadVariableOp(Adam/conv_7/kernel/m/Read/ReadVariableOp&Adam/conv_7/bias/m/Read/ReadVariableOp)Adam/dense_0/kernel/m/Read/ReadVariableOp'Adam/dense_0/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp(Adam/conv_3/kernel/v/Read/ReadVariableOp&Adam/conv_3/bias/v/Read/ReadVariableOp(Adam/conv_5/kernel/v/Read/ReadVariableOp&Adam/conv_5/bias/v/Read/ReadVariableOp(Adam/conv_7/kernel/v/Read/ReadVariableOp&Adam/conv_7/bias/v/Read/ReadVariableOp)Adam/dense_0/kernel/v/Read/ReadVariableOp'Adam/dense_0/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
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
 __inference__traced_save_2021365
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_3/kernelconv_3/biasconv_5/kernelconv_5/biasconv_7/kernelconv_7/biasdense_0/kerneldense_0/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv_3/kernel/mAdam/conv_3/bias/mAdam/conv_5/kernel/mAdam/conv_5/bias/mAdam/conv_7/kernel/mAdam/conv_7/bias/mAdam/dense_0/kernel/mAdam/dense_0/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv_3/kernel/vAdam/conv_3/bias/vAdam/conv_5/kernel/vAdam/conv_5/bias/vAdam/conv_7/kernel/vAdam/conv_7/bias/vAdam/dense_0/kernel/vAdam/dense_0/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/output/kernel/vAdam/output/bias/v*=
Tin6
422*
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
#__inference__traced_restore_2021522��
�
�
C__inference_conv_3_layer_call_and_return_conditional_losses_2019823

inputsA
+conv1d_expanddims_1_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
C__inference_drop_5_layer_call_and_return_conditional_losses_2020938

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������F2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������F2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������F:S O
+
_output_shapes
:���������F
 
_user_specified_nameinputs
�
j
0__inference_concatenate_14_layer_call_fn_2021035
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_20198852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/2
�
�
*__inference_model_26_layer_call_fn_2020007
input_onehot
unknown:(
	unknown_0:(
	unknown_1:F
	unknown_2:F
	unknown_3:d
	unknown_4:d
	unknown_5:	�P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P<

unknown_10:<

unknown_11:<

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_model_26_layer_call_and_return_conditional_losses_20199762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_onehot
��
�
"__inference__wrapped_model_2019711
input_onehotQ
;model_26_conv_7_conv1d_expanddims_1_readvariableop_resource:(=
/model_26_conv_7_biasadd_readvariableop_resource:(Q
;model_26_conv_5_conv1d_expanddims_1_readvariableop_resource:F=
/model_26_conv_5_biasadd_readvariableop_resource:FQ
;model_26_conv_3_conv1d_expanddims_1_readvariableop_resource:d=
/model_26_conv_3_biasadd_readvariableop_resource:dB
/model_26_dense_0_matmul_readvariableop_resource:	�P>
0model_26_dense_0_biasadd_readvariableop_resource:PA
/model_26_dense_1_matmul_readvariableop_resource:PP>
0model_26_dense_1_biasadd_readvariableop_resource:PA
/model_26_dense_2_matmul_readvariableop_resource:P<>
0model_26_dense_2_biasadd_readvariableop_resource:<@
.model_26_output_matmul_readvariableop_resource:<=
/model_26_output_biasadd_readvariableop_resource:
identity��&model_26/conv_3/BiasAdd/ReadVariableOp�2model_26/conv_3/conv1d/ExpandDims_1/ReadVariableOp�&model_26/conv_5/BiasAdd/ReadVariableOp�2model_26/conv_5/conv1d/ExpandDims_1/ReadVariableOp�&model_26/conv_7/BiasAdd/ReadVariableOp�2model_26/conv_7/conv1d/ExpandDims_1/ReadVariableOp�'model_26/dense_0/BiasAdd/ReadVariableOp�&model_26/dense_0/MatMul/ReadVariableOp�'model_26/dense_1/BiasAdd/ReadVariableOp�&model_26/dense_1/MatMul/ReadVariableOp�'model_26/dense_2/BiasAdd/ReadVariableOp�&model_26/dense_2/MatMul/ReadVariableOp�&model_26/output/BiasAdd/ReadVariableOp�%model_26/output/MatMul/ReadVariableOp�
%model_26/conv_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%model_26/conv_7/conv1d/ExpandDims/dim�
!model_26/conv_7/conv1d/ExpandDims
ExpandDimsinput_onehot.model_26/conv_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2#
!model_26/conv_7/conv1d/ExpandDims�
2model_26/conv_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;model_26_conv_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:(*
dtype024
2model_26/conv_7/conv1d/ExpandDims_1/ReadVariableOp�
'model_26/conv_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_26/conv_7/conv1d/ExpandDims_1/dim�
#model_26/conv_7/conv1d/ExpandDims_1
ExpandDims:model_26/conv_7/conv1d/ExpandDims_1/ReadVariableOp:value:00model_26/conv_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:(2%
#model_26/conv_7/conv1d/ExpandDims_1�
model_26/conv_7/conv1dConv2D*model_26/conv_7/conv1d/ExpandDims:output:0,model_26/conv_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������(*
paddingVALID*
strides
2
model_26/conv_7/conv1d�
model_26/conv_7/conv1d/SqueezeSqueezemodel_26/conv_7/conv1d:output:0*
T0*+
_output_shapes
:���������(*
squeeze_dims

���������2 
model_26/conv_7/conv1d/Squeeze�
&model_26/conv_7/BiasAdd/ReadVariableOpReadVariableOp/model_26_conv_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02(
&model_26/conv_7/BiasAdd/ReadVariableOp�
model_26/conv_7/BiasAddBiasAdd'model_26/conv_7/conv1d/Squeeze:output:0.model_26/conv_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
model_26/conv_7/BiasAdd�
model_26/conv_7/ReluRelu model_26/conv_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
model_26/conv_7/Relu�
%model_26/conv_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%model_26/conv_5/conv1d/ExpandDims/dim�
!model_26/conv_5/conv1d/ExpandDims
ExpandDimsinput_onehot.model_26/conv_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2#
!model_26/conv_5/conv1d/ExpandDims�
2model_26/conv_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;model_26_conv_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:F*
dtype024
2model_26/conv_5/conv1d/ExpandDims_1/ReadVariableOp�
'model_26/conv_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_26/conv_5/conv1d/ExpandDims_1/dim�
#model_26/conv_5/conv1d/ExpandDims_1
ExpandDims:model_26/conv_5/conv1d/ExpandDims_1/ReadVariableOp:value:00model_26/conv_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:F2%
#model_26/conv_5/conv1d/ExpandDims_1�
model_26/conv_5/conv1dConv2D*model_26/conv_5/conv1d/ExpandDims:output:0,model_26/conv_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������F*
paddingVALID*
strides
2
model_26/conv_5/conv1d�
model_26/conv_5/conv1d/SqueezeSqueezemodel_26/conv_5/conv1d:output:0*
T0*+
_output_shapes
:���������F*
squeeze_dims

���������2 
model_26/conv_5/conv1d/Squeeze�
&model_26/conv_5/BiasAdd/ReadVariableOpReadVariableOp/model_26_conv_5_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02(
&model_26/conv_5/BiasAdd/ReadVariableOp�
model_26/conv_5/BiasAddBiasAdd'model_26/conv_5/conv1d/Squeeze:output:0.model_26/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������F2
model_26/conv_5/BiasAdd�
model_26/conv_5/ReluRelu model_26/conv_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������F2
model_26/conv_5/Relu�
%model_26/conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%model_26/conv_3/conv1d/ExpandDims/dim�
!model_26/conv_3/conv1d/ExpandDims
ExpandDimsinput_onehot.model_26/conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2#
!model_26/conv_3/conv1d/ExpandDims�
2model_26/conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;model_26_conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype024
2model_26/conv_3/conv1d/ExpandDims_1/ReadVariableOp�
'model_26/conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_26/conv_3/conv1d/ExpandDims_1/dim�
#model_26/conv_3/conv1d/ExpandDims_1
ExpandDims:model_26/conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:00model_26/conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d2%
#model_26/conv_3/conv1d/ExpandDims_1�
model_26/conv_3/conv1dConv2D*model_26/conv_3/conv1d/ExpandDims:output:0,model_26/conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
model_26/conv_3/conv1d�
model_26/conv_3/conv1d/SqueezeSqueezemodel_26/conv_3/conv1d:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims

���������2 
model_26/conv_3/conv1d/Squeeze�
&model_26/conv_3/BiasAdd/ReadVariableOpReadVariableOp/model_26_conv_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02(
&model_26/conv_3/BiasAdd/ReadVariableOp�
model_26/conv_3/BiasAddBiasAdd'model_26/conv_3/conv1d/Squeeze:output:0.model_26/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d2
model_26/conv_3/BiasAdd�
model_26/conv_3/ReluRelu model_26/conv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������d2
model_26/conv_3/Relu�
model_26/drop_7/IdentityIdentity"model_26/conv_7/Relu:activations:0*
T0*+
_output_shapes
:���������(2
model_26/drop_7/Identity�
model_26/drop_5/IdentityIdentity"model_26/conv_5/Relu:activations:0*
T0*+
_output_shapes
:���������F2
model_26/drop_5/Identity�
model_26/drop_3/IdentityIdentity"model_26/conv_3/Relu:activations:0*
T0*+
_output_shapes
:���������d2
model_26/drop_3/Identity�
model_26/pool_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
model_26/pool_7/ExpandDims/dim�
model_26/pool_7/ExpandDims
ExpandDims!model_26/drop_7/Identity:output:0'model_26/pool_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������(2
model_26/pool_7/ExpandDims�
model_26/pool_7/AvgPoolAvgPool#model_26/pool_7/ExpandDims:output:0*
T0*/
_output_shapes
:���������	(*
ksize
*
paddingSAME*
strides
2
model_26/pool_7/AvgPool�
model_26/pool_7/SqueezeSqueeze model_26/pool_7/AvgPool:output:0*
T0*+
_output_shapes
:���������	(*
squeeze_dims
2
model_26/pool_7/Squeeze�
model_26/pool_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
model_26/pool_5/ExpandDims/dim�
model_26/pool_5/ExpandDims
ExpandDims!model_26/drop_5/Identity:output:0'model_26/pool_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������F2
model_26/pool_5/ExpandDims�
model_26/pool_5/AvgPoolAvgPool#model_26/pool_5/ExpandDims:output:0*
T0*/
_output_shapes
:���������
F*
ksize
*
paddingSAME*
strides
2
model_26/pool_5/AvgPool�
model_26/pool_5/SqueezeSqueeze model_26/pool_5/AvgPool:output:0*
T0*+
_output_shapes
:���������
F*
squeeze_dims
2
model_26/pool_5/Squeeze�
model_26/pool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
model_26/pool_3/ExpandDims/dim�
model_26/pool_3/ExpandDims
ExpandDims!model_26/drop_3/Identity:output:0'model_26/pool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d2
model_26/pool_3/ExpandDims�
model_26/pool_3/AvgPoolAvgPool#model_26/pool_3/ExpandDims:output:0*
T0*/
_output_shapes
:���������d*
ksize
*
paddingSAME*
strides
2
model_26/pool_3/AvgPool�
model_26/pool_3/SqueezeSqueeze model_26/pool_3/AvgPool:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims
2
model_26/pool_3/Squeeze�
model_26/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����L  2
model_26/flatten_3/Const�
model_26/flatten_3/ReshapeReshape model_26/pool_3/Squeeze:output:0!model_26/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
model_26/flatten_3/Reshape�
model_26/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
model_26/flatten_5/Const�
model_26/flatten_5/ReshapeReshape model_26/pool_5/Squeeze:output:0!model_26/flatten_5/Const:output:0*
T0*(
_output_shapes
:����������2
model_26/flatten_5/Reshape�
model_26/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����h  2
model_26/flatten_7/Const�
model_26/flatten_7/ReshapeReshape model_26/pool_7/Squeeze:output:0!model_26/flatten_7/Const:output:0*
T0*(
_output_shapes
:����������2
model_26/flatten_7/Reshape�
#model_26/concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_26/concatenate_14/concat/axis�
model_26/concatenate_14/concatConcatV2#model_26/flatten_3/Reshape:output:0#model_26/flatten_5/Reshape:output:0#model_26/flatten_7/Reshape:output:0,model_26/concatenate_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2 
model_26/concatenate_14/concat�
&model_26/dense_0/MatMul/ReadVariableOpReadVariableOp/model_26_dense_0_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02(
&model_26/dense_0/MatMul/ReadVariableOp�
model_26/dense_0/MatMulMatMul'model_26/concatenate_14/concat:output:0.model_26/dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_26/dense_0/MatMul�
'model_26/dense_0/BiasAdd/ReadVariableOpReadVariableOp0model_26_dense_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02)
'model_26/dense_0/BiasAdd/ReadVariableOp�
model_26/dense_0/BiasAddBiasAdd!model_26/dense_0/MatMul:product:0/model_26/dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_26/dense_0/BiasAdd�
model_26/dense_0/ReluRelu!model_26/dense_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
model_26/dense_0/Relu�
model_26/drop_d0/IdentityIdentity#model_26/dense_0/Relu:activations:0*
T0*'
_output_shapes
:���������P2
model_26/drop_d0/Identity�
&model_26/dense_1/MatMul/ReadVariableOpReadVariableOp/model_26_dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02(
&model_26/dense_1/MatMul/ReadVariableOp�
model_26/dense_1/MatMulMatMul"model_26/drop_d0/Identity:output:0.model_26/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_26/dense_1/MatMul�
'model_26/dense_1/BiasAdd/ReadVariableOpReadVariableOp0model_26_dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02)
'model_26/dense_1/BiasAdd/ReadVariableOp�
model_26/dense_1/BiasAddBiasAdd!model_26/dense_1/MatMul:product:0/model_26/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
model_26/dense_1/BiasAdd�
model_26/dense_1/ReluRelu!model_26/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
model_26/dense_1/Relu�
model_26/drop_d1/IdentityIdentity#model_26/dense_1/Relu:activations:0*
T0*'
_output_shapes
:���������P2
model_26/drop_d1/Identity�
&model_26/dense_2/MatMul/ReadVariableOpReadVariableOp/model_26_dense_2_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype02(
&model_26/dense_2/MatMul/ReadVariableOp�
model_26/dense_2/MatMulMatMul"model_26/drop_d1/Identity:output:0.model_26/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
model_26/dense_2/MatMul�
'model_26/dense_2/BiasAdd/ReadVariableOpReadVariableOp0model_26_dense_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02)
'model_26/dense_2/BiasAdd/ReadVariableOp�
model_26/dense_2/BiasAddBiasAdd!model_26/dense_2/MatMul:product:0/model_26/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
model_26/dense_2/BiasAdd�
model_26/dense_2/ReluRelu!model_26/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
model_26/dense_2/Relu�
model_26/drop_d2/IdentityIdentity#model_26/dense_2/Relu:activations:0*
T0*'
_output_shapes
:���������<2
model_26/drop_d2/Identity�
%model_26/output/MatMul/ReadVariableOpReadVariableOp.model_26_output_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02'
%model_26/output/MatMul/ReadVariableOp�
model_26/output/MatMulMatMul"model_26/drop_d2/Identity:output:0-model_26/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_26/output/MatMul�
&model_26/output/BiasAdd/ReadVariableOpReadVariableOp/model_26_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_26/output/BiasAdd/ReadVariableOp�
model_26/output/BiasAddBiasAdd model_26/output/MatMul:product:0.model_26/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_26/output/BiasAdd�
IdentityIdentity model_26/output/BiasAdd:output:0'^model_26/conv_3/BiasAdd/ReadVariableOp3^model_26/conv_3/conv1d/ExpandDims_1/ReadVariableOp'^model_26/conv_5/BiasAdd/ReadVariableOp3^model_26/conv_5/conv1d/ExpandDims_1/ReadVariableOp'^model_26/conv_7/BiasAdd/ReadVariableOp3^model_26/conv_7/conv1d/ExpandDims_1/ReadVariableOp(^model_26/dense_0/BiasAdd/ReadVariableOp'^model_26/dense_0/MatMul/ReadVariableOp(^model_26/dense_1/BiasAdd/ReadVariableOp'^model_26/dense_1/MatMul/ReadVariableOp(^model_26/dense_2/BiasAdd/ReadVariableOp'^model_26/dense_2/MatMul/ReadVariableOp'^model_26/output/BiasAdd/ReadVariableOp&^model_26/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2P
&model_26/conv_3/BiasAdd/ReadVariableOp&model_26/conv_3/BiasAdd/ReadVariableOp2h
2model_26/conv_3/conv1d/ExpandDims_1/ReadVariableOp2model_26/conv_3/conv1d/ExpandDims_1/ReadVariableOp2P
&model_26/conv_5/BiasAdd/ReadVariableOp&model_26/conv_5/BiasAdd/ReadVariableOp2h
2model_26/conv_5/conv1d/ExpandDims_1/ReadVariableOp2model_26/conv_5/conv1d/ExpandDims_1/ReadVariableOp2P
&model_26/conv_7/BiasAdd/ReadVariableOp&model_26/conv_7/BiasAdd/ReadVariableOp2h
2model_26/conv_7/conv1d/ExpandDims_1/ReadVariableOp2model_26/conv_7/conv1d/ExpandDims_1/ReadVariableOp2R
'model_26/dense_0/BiasAdd/ReadVariableOp'model_26/dense_0/BiasAdd/ReadVariableOp2P
&model_26/dense_0/MatMul/ReadVariableOp&model_26/dense_0/MatMul/ReadVariableOp2R
'model_26/dense_1/BiasAdd/ReadVariableOp'model_26/dense_1/BiasAdd/ReadVariableOp2P
&model_26/dense_1/MatMul/ReadVariableOp&model_26/dense_1/MatMul/ReadVariableOp2R
'model_26/dense_2/BiasAdd/ReadVariableOp'model_26/dense_2/BiasAdd/ReadVariableOp2P
&model_26/dense_2/MatMul/ReadVariableOp&model_26/dense_2/MatMul/ReadVariableOp2P
&model_26/output/BiasAdd/ReadVariableOp&model_26/output/BiasAdd/ReadVariableOp2N
%model_26/output/MatMul/ReadVariableOp%model_26/output/MatMul/ReadVariableOp:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
�
%__inference_signature_wrapper_2020537
input_onehot
unknown:(
	unknown_0:(
	unknown_1:F
	unknown_2:F
	unknown_3:d
	unknown_4:d
	unknown_5:	�P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P<

unknown_10:<

unknown_11:<

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_20197112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
a
C__inference_drop_5_layer_call_and_return_conditional_losses_2019841

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������F2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������F2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������F:S O
+
_output_shapes
:���������F
 
_user_specified_nameinputs
�
_
C__inference_pool_5_layer_call_and_return_conditional_losses_2019735

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
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingSAME*
strides
2	
AvgPool�
SqueezeSqueezeAvgPool:output:0*
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
�
a
(__inference_drop_5_layer_call_fn_2020960

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_5_layer_call_and_return_conditional_losses_20201852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������F22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������F
 
_user_specified_nameinputs
�
c
D__inference_drop_d0_layer_call_and_return_conditional_losses_2020103

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
�
b
)__inference_drop_d1_layer_call_fn_2021129

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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d1_layer_call_and_return_conditional_losses_20200702
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
�U
�
E__inference_model_26_layer_call_and_return_conditional_losses_2020328

inputs$
conv_7_2020279:(
conv_7_2020281:($
conv_5_2020284:F
conv_5_2020286:F$
conv_3_2020289:d
conv_3_2020291:d"
dense_0_2020304:	�P
dense_0_2020306:P!
dense_1_2020310:PP
dense_1_2020312:P!
dense_2_2020316:P<
dense_2_2020318:< 
output_2020322:<
output_2020324:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�drop_3/StatefulPartitionedCall�drop_5/StatefulPartitionedCall�drop_7/StatefulPartitionedCall�drop_d0/StatefulPartitionedCall�drop_d1/StatefulPartitionedCall�drop_d2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv_7_2020279conv_7_2020281*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_7_layer_call_and_return_conditional_losses_20197792 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_5_2020284conv_5_2020286*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_5_layer_call_and_return_conditional_losses_20198012 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv_3_2020289conv_3_2020291*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_20198232 
conv_3/StatefulPartitionedCall�
drop_7/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_7_layer_call_and_return_conditional_losses_20202082 
drop_7/StatefulPartitionedCall�
drop_5/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0^drop_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_5_layer_call_and_return_conditional_losses_20201852 
drop_5/StatefulPartitionedCall�
drop_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0^drop_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_3_layer_call_and_return_conditional_losses_20201622 
drop_3/StatefulPartitionedCall�
pool_7/PartitionedCallPartitionedCall'drop_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_7_layer_call_and_return_conditional_losses_20197502
pool_7/PartitionedCall�
pool_5/PartitionedCallPartitionedCall'drop_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_5_layer_call_and_return_conditional_losses_20197352
pool_5/PartitionedCall�
pool_3/PartitionedCallPartitionedCall'drop_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_3_layer_call_and_return_conditional_losses_20197202
pool_3/PartitionedCall�
flatten_3/PartitionedCallPartitionedCallpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_20198592
flatten_3/PartitionedCall�
flatten_5/PartitionedCallPartitionedCallpool_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20198672
flatten_5/PartitionedCall�
flatten_7/PartitionedCallPartitionedCallpool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_20198752
flatten_7/PartitionedCall�
concatenate_14/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_20198852 
concatenate_14/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense_0_2020304dense_0_2020306*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_0_layer_call_and_return_conditional_losses_20198982!
dense_0/StatefulPartitionedCall�
drop_d0/StatefulPartitionedCallStatefulPartitionedCall(dense_0/StatefulPartitionedCall:output:0^drop_3/StatefulPartitionedCall*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d0_layer_call_and_return_conditional_losses_20201032!
drop_d0/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(drop_d0/StatefulPartitionedCall:output:0dense_1_2020310dense_1_2020312*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20199222!
dense_1/StatefulPartitionedCall�
drop_d1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^drop_d0/StatefulPartitionedCall*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d1_layer_call_and_return_conditional_losses_20200702!
drop_d1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(drop_d1/StatefulPartitionedCall:output:0dense_2_2020316dense_2_2020318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20199462!
dense_2/StatefulPartitionedCall�
drop_d2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 ^drop_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_drop_d2_layer_call_and_return_conditional_losses_20200372!
drop_d2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(drop_d2/StatefulPartitionedCall:output:0output_2020322output_2020324*
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
C__inference_output_layer_call_and_return_conditional_losses_20199692 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_3/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_7/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^drop_3/StatefulPartitionedCall^drop_5/StatefulPartitionedCall^drop_7/StatefulPartitionedCall ^drop_d0/StatefulPartitionedCall ^drop_d1/StatefulPartitionedCall ^drop_d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
drop_3/StatefulPartitionedCalldrop_3/StatefulPartitionedCall2@
drop_5/StatefulPartitionedCalldrop_5/StatefulPartitionedCall2@
drop_7/StatefulPartitionedCalldrop_7/StatefulPartitionedCall2B
drop_d0/StatefulPartitionedCalldrop_d0/StatefulPartitionedCall2B
drop_d1/StatefulPartitionedCalldrop_d1/StatefulPartitionedCall2B
drop_d2/StatefulPartitionedCalldrop_d2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�M
�
E__inference_model_26_layer_call_and_return_conditional_losses_2020444
input_onehot$
conv_7_2020395:(
conv_7_2020397:($
conv_5_2020400:F
conv_5_2020402:F$
conv_3_2020405:d
conv_3_2020407:d"
dense_0_2020420:	�P
dense_0_2020422:P!
dense_1_2020426:PP
dense_1_2020428:P!
dense_2_2020432:P<
dense_2_2020434:< 
output_2020438:<
output_2020440:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_7_2020395conv_7_2020397*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_7_layer_call_and_return_conditional_losses_20197792 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_5_2020400conv_5_2020402*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_5_layer_call_and_return_conditional_losses_20198012 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_3_2020405conv_3_2020407*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_20198232 
conv_3/StatefulPartitionedCall�
drop_7/PartitionedCallPartitionedCall'conv_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_7_layer_call_and_return_conditional_losses_20198342
drop_7/PartitionedCall�
drop_5/PartitionedCallPartitionedCall'conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_5_layer_call_and_return_conditional_losses_20198412
drop_5/PartitionedCall�
drop_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_3_layer_call_and_return_conditional_losses_20198482
drop_3/PartitionedCall�
pool_7/PartitionedCallPartitionedCalldrop_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_7_layer_call_and_return_conditional_losses_20197502
pool_7/PartitionedCall�
pool_5/PartitionedCallPartitionedCalldrop_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_5_layer_call_and_return_conditional_losses_20197352
pool_5/PartitionedCall�
pool_3/PartitionedCallPartitionedCalldrop_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_3_layer_call_and_return_conditional_losses_20197202
pool_3/PartitionedCall�
flatten_3/PartitionedCallPartitionedCallpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_20198592
flatten_3/PartitionedCall�
flatten_5/PartitionedCallPartitionedCallpool_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20198672
flatten_5/PartitionedCall�
flatten_7/PartitionedCallPartitionedCallpool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_20198752
flatten_7/PartitionedCall�
concatenate_14/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_20198852 
concatenate_14/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense_0_2020420dense_0_2020422*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_0_layer_call_and_return_conditional_losses_20198982!
dense_0/StatefulPartitionedCall�
drop_d0/PartitionedCallPartitionedCall(dense_0/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d0_layer_call_and_return_conditional_losses_20199092
drop_d0/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall drop_d0/PartitionedCall:output:0dense_1_2020426dense_1_2020428*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20199222!
dense_1/StatefulPartitionedCall�
drop_d1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d1_layer_call_and_return_conditional_losses_20199332
drop_d1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall drop_d1/PartitionedCall:output:0dense_2_2020432dense_2_2020434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20199462!
dense_2/StatefulPartitionedCall�
drop_d2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_drop_d2_layer_call_and_return_conditional_losses_20199572
drop_d2/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall drop_d2/PartitionedCall:output:0output_2020438output_2020440*
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
C__inference_output_layer_call_and_return_conditional_losses_20199692 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_3/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_7/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�

�
D__inference_dense_0_layer_call_and_return_conditional_losses_2019898

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_2_layer_call_fn_2021149

inputs
unknown:P<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20199462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������<2

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
G
+__inference_flatten_5_layer_call_fn_2021009

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
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20198672
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
F:S O
+
_output_shapes
:���������
F
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2021186

inputs0
matmul_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
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
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
_
C__inference_pool_3_layer_call_and_return_conditional_losses_2019720

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
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingSAME*
strides
2	
AvgPool�
SqueezeSqueezeAvgPool:output:0*
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
E
)__inference_drop_d2_layer_call_fn_2021171

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
:���������<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_drop_d2_layer_call_and_return_conditional_losses_20199572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
G
+__inference_flatten_3_layer_call_fn_2020998

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
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_20198592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_2021004

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
F:S O
+
_output_shapes
:���������
F
 
_user_specified_nameinputs
�
�
(__inference_conv_3_layer_call_fn_2020856

inputs
unknown:d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_20198232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
D__inference_drop_d2_layer_call_and_return_conditional_losses_2019957

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������<2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_2019867

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
F:S O
+
_output_shapes
:���������
F
 
_user_specified_nameinputs
�
b
C__inference_drop_5_layer_call_and_return_conditional_losses_2020185

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������F2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������F*
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
T0*+
_output_shapes
:���������F2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������F2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������F2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������F:S O
+
_output_shapes
:���������F
 
_user_specified_nameinputs
�
b
C__inference_drop_3_layer_call_and_return_conditional_losses_2020923

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������d*
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
T0*+
_output_shapes
:���������d2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������d2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������d2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
D__inference_dense_1_layer_call_and_return_conditional_losses_2019922

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
C__inference_conv_3_layer_call_and_return_conditional_losses_2020847

inputsA
+conv1d_expanddims_1_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_pool_5_layer_call_fn_2019741

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
GPU2 *0J 8� *L
fGRE
C__inference_pool_5_layer_call_and_return_conditional_losses_20197352
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
�
C__inference_output_layer_call_and_return_conditional_losses_2019969

inputs0
matmul_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
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
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
c
D__inference_drop_d1_layer_call_and_return_conditional_losses_2020070

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
�
�
C__inference_conv_5_layer_call_and_return_conditional_losses_2020872

inputsA
+conv1d_expanddims_1_readvariableop_resource:F-
biasadd_readvariableop_resource:F
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:F*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:F2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������F*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������F*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������F2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������F2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
C__inference_drop_7_layer_call_and_return_conditional_losses_2020977

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������(*
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
T0*+
_output_shapes
:���������(2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������(2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������(2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
(__inference_conv_5_layer_call_fn_2020881

inputs
unknown:F
	unknown_0:F
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_5_layer_call_and_return_conditional_losses_20198012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_drop_3_layer_call_fn_2020928

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_3_layer_call_and_return_conditional_losses_20198482
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
K__inference_concatenate_14_layer_call_and_return_conditional_losses_2019885

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_2021015

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	(:S O
+
_output_shapes
:���������	(
 
_user_specified_nameinputs
�

�
D__inference_dense_0_layer_call_and_return_conditional_losses_2021046

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_2021522
file_prefix4
assignvariableop_conv_3_kernel:d,
assignvariableop_1_conv_3_bias:d6
 assignvariableop_2_conv_5_kernel:F,
assignvariableop_3_conv_5_bias:F6
 assignvariableop_4_conv_7_kernel:(,
assignvariableop_5_conv_7_bias:(4
!assignvariableop_6_dense_0_kernel:	�P-
assignvariableop_7_dense_0_bias:P3
!assignvariableop_8_dense_1_kernel:PP-
assignvariableop_9_dense_1_bias:P4
"assignvariableop_10_dense_2_kernel:P<.
 assignvariableop_11_dense_2_bias:<3
!assignvariableop_12_output_kernel:<-
assignvariableop_13_output_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: >
(assignvariableop_21_adam_conv_3_kernel_m:d4
&assignvariableop_22_adam_conv_3_bias_m:d>
(assignvariableop_23_adam_conv_5_kernel_m:F4
&assignvariableop_24_adam_conv_5_bias_m:F>
(assignvariableop_25_adam_conv_7_kernel_m:(4
&assignvariableop_26_adam_conv_7_bias_m:(<
)assignvariableop_27_adam_dense_0_kernel_m:	�P5
'assignvariableop_28_adam_dense_0_bias_m:P;
)assignvariableop_29_adam_dense_1_kernel_m:PP5
'assignvariableop_30_adam_dense_1_bias_m:P;
)assignvariableop_31_adam_dense_2_kernel_m:P<5
'assignvariableop_32_adam_dense_2_bias_m:<:
(assignvariableop_33_adam_output_kernel_m:<4
&assignvariableop_34_adam_output_bias_m:>
(assignvariableop_35_adam_conv_3_kernel_v:d4
&assignvariableop_36_adam_conv_3_bias_v:d>
(assignvariableop_37_adam_conv_5_kernel_v:F4
&assignvariableop_38_adam_conv_5_bias_v:F>
(assignvariableop_39_adam_conv_7_kernel_v:(4
&assignvariableop_40_adam_conv_7_bias_v:(<
)assignvariableop_41_adam_dense_0_kernel_v:	�P5
'assignvariableop_42_adam_dense_0_bias_v:P;
)assignvariableop_43_adam_dense_1_kernel_v:PP5
'assignvariableop_44_adam_dense_1_bias_v:P;
)assignvariableop_45_adam_dense_2_kernel_v:P<5
'assignvariableop_46_adam_dense_2_bias_v:<:
(assignvariableop_47_adam_output_kernel_v:<4
&assignvariableop_48_adam_output_bias_v:
identity_50��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_0_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_0_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_conv_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_conv_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv_5_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv_5_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv_7_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv_7_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_0_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_0_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_output_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_output_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv_3_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv_3_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv_5_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv_5_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv_7_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv_7_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_0_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_0_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_output_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_output_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49�	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
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
�
b
C__inference_drop_3_layer_call_and_return_conditional_losses_2020162

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������d*
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
T0*+
_output_shapes
:���������d2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������d2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������d2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
a
(__inference_drop_7_layer_call_fn_2020987

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_7_layer_call_and_return_conditional_losses_20202082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
b
D__inference_drop_d0_layer_call_and_return_conditional_losses_2021060

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

�
D__inference_dense_2_layer_call_and_return_conditional_losses_2019946

inputs0
matmul_readvariableop_resource:P<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������<2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������<2

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
�
�
)__inference_dense_1_layer_call_fn_2021102

inputs
unknown:PP
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20199222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

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
�
�
(__inference_conv_7_layer_call_fn_2020906

inputs
unknown:(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_7_layer_call_and_return_conditional_losses_20197792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
C__inference_drop_7_layer_call_and_return_conditional_losses_2020208

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������(*
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
T0*+
_output_shapes
:���������(2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������(2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������(2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
��
�
E__inference_model_26_layer_call_and_return_conditional_losses_2020765

inputsH
2conv_7_conv1d_expanddims_1_readvariableop_resource:(4
&conv_7_biasadd_readvariableop_resource:(H
2conv_5_conv1d_expanddims_1_readvariableop_resource:F4
&conv_5_biasadd_readvariableop_resource:FH
2conv_3_conv1d_expanddims_1_readvariableop_resource:d4
&conv_3_biasadd_readvariableop_resource:d9
&dense_0_matmul_readvariableop_resource:	�P5
'dense_0_biasadd_readvariableop_resource:P8
&dense_1_matmul_readvariableop_resource:PP5
'dense_1_biasadd_readvariableop_resource:P8
&dense_2_matmul_readvariableop_resource:P<5
'dense_2_biasadd_readvariableop_resource:<7
%output_matmul_readvariableop_resource:<4
&output_biasadd_readvariableop_resource:
identity��conv_3/BiasAdd/ReadVariableOp�)conv_3/conv1d/ExpandDims_1/ReadVariableOp�conv_5/BiasAdd/ReadVariableOp�)conv_5/conv1d/ExpandDims_1/ReadVariableOp�conv_7/BiasAdd/ReadVariableOp�)conv_7/conv1d/ExpandDims_1/ReadVariableOp�dense_0/BiasAdd/ReadVariableOp�dense_0/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_7/conv1d/ExpandDims/dim�
conv_7/conv1d/ExpandDims
ExpandDimsinputs%conv_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv_7/conv1d/ExpandDims�
)conv_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:(*
dtype02+
)conv_7/conv1d/ExpandDims_1/ReadVariableOp�
conv_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_7/conv1d/ExpandDims_1/dim�
conv_7/conv1d/ExpandDims_1
ExpandDims1conv_7/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:(2
conv_7/conv1d/ExpandDims_1�
conv_7/conv1dConv2D!conv_7/conv1d/ExpandDims:output:0#conv_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������(*
paddingVALID*
strides
2
conv_7/conv1d�
conv_7/conv1d/SqueezeSqueezeconv_7/conv1d:output:0*
T0*+
_output_shapes
:���������(*
squeeze_dims

���������2
conv_7/conv1d/Squeeze�
conv_7/BiasAdd/ReadVariableOpReadVariableOp&conv_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
conv_7/BiasAdd/ReadVariableOp�
conv_7/BiasAddBiasAddconv_7/conv1d/Squeeze:output:0%conv_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
conv_7/BiasAddq
conv_7/ReluReluconv_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
conv_7/Relu�
conv_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_5/conv1d/ExpandDims/dim�
conv_5/conv1d/ExpandDims
ExpandDimsinputs%conv_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv_5/conv1d/ExpandDims�
)conv_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:F*
dtype02+
)conv_5/conv1d/ExpandDims_1/ReadVariableOp�
conv_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_5/conv1d/ExpandDims_1/dim�
conv_5/conv1d/ExpandDims_1
ExpandDims1conv_5/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:F2
conv_5/conv1d/ExpandDims_1�
conv_5/conv1dConv2D!conv_5/conv1d/ExpandDims:output:0#conv_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������F*
paddingVALID*
strides
2
conv_5/conv1d�
conv_5/conv1d/SqueezeSqueezeconv_5/conv1d:output:0*
T0*+
_output_shapes
:���������F*
squeeze_dims

���������2
conv_5/conv1d/Squeeze�
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
conv_5/BiasAdd/ReadVariableOp�
conv_5/BiasAddBiasAddconv_5/conv1d/Squeeze:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������F2
conv_5/BiasAddq
conv_5/ReluReluconv_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������F2
conv_5/Relu�
conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_3/conv1d/ExpandDims/dim�
conv_3/conv1d/ExpandDims
ExpandDimsinputs%conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv_3/conv1d/ExpandDims�
)conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype02+
)conv_3/conv1d/ExpandDims_1/ReadVariableOp�
conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_3/conv1d/ExpandDims_1/dim�
conv_3/conv1d/ExpandDims_1
ExpandDims1conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d2
conv_3/conv1d/ExpandDims_1�
conv_3/conv1dConv2D!conv_3/conv1d/ExpandDims:output:0#conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
conv_3/conv1d�
conv_3/conv1d/SqueezeSqueezeconv_3/conv1d:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims

���������2
conv_3/conv1d/Squeeze�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/conv1d/Squeeze:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d2
conv_3/BiasAddq
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������d2
conv_3/Reluq
drop_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
drop_7/dropout/Const�
drop_7/dropout/MulMulconv_7/Relu:activations:0drop_7/dropout/Const:output:0*
T0*+
_output_shapes
:���������(2
drop_7/dropout/Mulu
drop_7/dropout/ShapeShapeconv_7/Relu:activations:0*
T0*
_output_shapes
:2
drop_7/dropout/Shape�
+drop_7/dropout/random_uniform/RandomUniformRandomUniformdrop_7/dropout/Shape:output:0*
T0*+
_output_shapes
:���������(*
dtype02-
+drop_7/dropout/random_uniform/RandomUniform�
drop_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
drop_7/dropout/GreaterEqual/y�
drop_7/dropout/GreaterEqualGreaterEqual4drop_7/dropout/random_uniform/RandomUniform:output:0&drop_7/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������(2
drop_7/dropout/GreaterEqual�
drop_7/dropout/CastCastdrop_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������(2
drop_7/dropout/Cast�
drop_7/dropout/Mul_1Muldrop_7/dropout/Mul:z:0drop_7/dropout/Cast:y:0*
T0*+
_output_shapes
:���������(2
drop_7/dropout/Mul_1q
drop_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
drop_5/dropout/Const�
drop_5/dropout/MulMulconv_5/Relu:activations:0drop_5/dropout/Const:output:0*
T0*+
_output_shapes
:���������F2
drop_5/dropout/Mulu
drop_5/dropout/ShapeShapeconv_5/Relu:activations:0*
T0*
_output_shapes
:2
drop_5/dropout/Shape�
+drop_5/dropout/random_uniform/RandomUniformRandomUniformdrop_5/dropout/Shape:output:0*
T0*+
_output_shapes
:���������F*
dtype02-
+drop_5/dropout/random_uniform/RandomUniform�
drop_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
drop_5/dropout/GreaterEqual/y�
drop_5/dropout/GreaterEqualGreaterEqual4drop_5/dropout/random_uniform/RandomUniform:output:0&drop_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������F2
drop_5/dropout/GreaterEqual�
drop_5/dropout/CastCastdrop_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������F2
drop_5/dropout/Cast�
drop_5/dropout/Mul_1Muldrop_5/dropout/Mul:z:0drop_5/dropout/Cast:y:0*
T0*+
_output_shapes
:���������F2
drop_5/dropout/Mul_1q
drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
drop_3/dropout/Const�
drop_3/dropout/MulMulconv_3/Relu:activations:0drop_3/dropout/Const:output:0*
T0*+
_output_shapes
:���������d2
drop_3/dropout/Mulu
drop_3/dropout/ShapeShapeconv_3/Relu:activations:0*
T0*
_output_shapes
:2
drop_3/dropout/Shape�
+drop_3/dropout/random_uniform/RandomUniformRandomUniformdrop_3/dropout/Shape:output:0*
T0*+
_output_shapes
:���������d*
dtype02-
+drop_3/dropout/random_uniform/RandomUniform�
drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
drop_3/dropout/GreaterEqual/y�
drop_3/dropout/GreaterEqualGreaterEqual4drop_3/dropout/random_uniform/RandomUniform:output:0&drop_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������d2
drop_3/dropout/GreaterEqual�
drop_3/dropout/CastCastdrop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������d2
drop_3/dropout/Cast�
drop_3/dropout/Mul_1Muldrop_3/dropout/Mul:z:0drop_3/dropout/Cast:y:0*
T0*+
_output_shapes
:���������d2
drop_3/dropout/Mul_1p
pool_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_7/ExpandDims/dim�
pool_7/ExpandDims
ExpandDimsdrop_7/dropout/Mul_1:z:0pool_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������(2
pool_7/ExpandDims�
pool_7/AvgPoolAvgPoolpool_7/ExpandDims:output:0*
T0*/
_output_shapes
:���������	(*
ksize
*
paddingSAME*
strides
2
pool_7/AvgPool�
pool_7/SqueezeSqueezepool_7/AvgPool:output:0*
T0*+
_output_shapes
:���������	(*
squeeze_dims
2
pool_7/Squeezep
pool_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_5/ExpandDims/dim�
pool_5/ExpandDims
ExpandDimsdrop_5/dropout/Mul_1:z:0pool_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������F2
pool_5/ExpandDims�
pool_5/AvgPoolAvgPoolpool_5/ExpandDims:output:0*
T0*/
_output_shapes
:���������
F*
ksize
*
paddingSAME*
strides
2
pool_5/AvgPool�
pool_5/SqueezeSqueezepool_5/AvgPool:output:0*
T0*+
_output_shapes
:���������
F*
squeeze_dims
2
pool_5/Squeezep
pool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_3/ExpandDims/dim�
pool_3/ExpandDims
ExpandDimsdrop_3/dropout/Mul_1:z:0pool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d2
pool_3/ExpandDims�
pool_3/AvgPoolAvgPoolpool_3/ExpandDims:output:0*
T0*/
_output_shapes
:���������d*
ksize
*
paddingSAME*
strides
2
pool_3/AvgPool�
pool_3/SqueezeSqueezepool_3/AvgPool:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims
2
pool_3/Squeezes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����L  2
flatten_3/Const�
flatten_3/ReshapeReshapepool_3/Squeeze:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_5/Const�
flatten_5/ReshapeReshapepool_5/Squeeze:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_5/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����h  2
flatten_7/Const�
flatten_7/ReshapeReshapepool_7/Squeeze:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_7/Reshapez
concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_14/concat/axis�
concatenate_14/concatConcatV2flatten_3/Reshape:output:0flatten_5/Reshape:output:0flatten_7/Reshape:output:0#concatenate_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate_14/concat�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02
dense_0/MatMul/ReadVariableOp�
dense_0/MatMulMatMulconcatenate_14/concat:output:0%dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_0/MatMul�
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_0/BiasAdd/ReadVariableOp�
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_0/BiasAddp
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
dense_0/Relus
drop_d0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
drop_d0/dropout/Const�
drop_d0/dropout/MulMuldense_0/Relu:activations:0drop_d0/dropout/Const:output:0*
T0*'
_output_shapes
:���������P2
drop_d0/dropout/Mulx
drop_d0/dropout/ShapeShapedense_0/Relu:activations:0*
T0*
_output_shapes
:2
drop_d0/dropout/Shape�
,drop_d0/dropout/random_uniform/RandomUniformRandomUniformdrop_d0/dropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
dtype02.
,drop_d0/dropout/random_uniform/RandomUniform�
drop_d0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2 
drop_d0/dropout/GreaterEqual/y�
drop_d0/dropout/GreaterEqualGreaterEqual5drop_d0/dropout/random_uniform/RandomUniform:output:0'drop_d0/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������P2
drop_d0/dropout/GreaterEqual�
drop_d0/dropout/CastCast drop_d0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
drop_d0/dropout/Cast�
drop_d0/dropout/Mul_1Muldrop_d0/dropout/Mul:z:0drop_d0/dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
drop_d0/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldrop_d0/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
dense_1/Relus
drop_d1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
drop_d1/dropout/Const�
drop_d1/dropout/MulMuldense_1/Relu:activations:0drop_d1/dropout/Const:output:0*
T0*'
_output_shapes
:���������P2
drop_d1/dropout/Mulx
drop_d1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
drop_d1/dropout/Shape�
,drop_d1/dropout/random_uniform/RandomUniformRandomUniformdrop_d1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������P*
dtype02.
,drop_d1/dropout/random_uniform/RandomUniform�
drop_d1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2 
drop_d1/dropout/GreaterEqual/y�
drop_d1/dropout/GreaterEqualGreaterEqual5drop_d1/dropout/random_uniform/RandomUniform:output:0'drop_d1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������P2
drop_d1/dropout/GreaterEqual�
drop_d1/dropout/CastCast drop_d1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������P2
drop_d1/dropout/Cast�
drop_d1/dropout/Mul_1Muldrop_d1/dropout/Mul:z:0drop_d1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������P2
drop_d1/dropout/Mul_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldrop_d1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
dense_2/Relus
drop_d2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
drop_d2/dropout/Const�
drop_d2/dropout/MulMuldense_2/Relu:activations:0drop_d2/dropout/Const:output:0*
T0*'
_output_shapes
:���������<2
drop_d2/dropout/Mulx
drop_d2/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
drop_d2/dropout/Shape�
,drop_d2/dropout/random_uniform/RandomUniformRandomUniformdrop_d2/dropout/Shape:output:0*
T0*'
_output_shapes
:���������<*
dtype02.
,drop_d2/dropout/random_uniform/RandomUniform�
drop_d2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2 
drop_d2/dropout/GreaterEqual/y�
drop_d2/dropout/GreaterEqualGreaterEqual5drop_d2/dropout/random_uniform/RandomUniform:output:0'drop_d2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������<2
drop_d2/dropout/GreaterEqual�
drop_d2/dropout/CastCast drop_d2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������<2
drop_d2/dropout/Cast�
drop_d2/dropout/Mul_1Muldrop_d2/dropout/Mul:z:0drop_d2/dropout/Cast:y:0*
T0*'
_output_shapes
:���������<2
drop_d2/dropout/Mul_1�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldrop_d2/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAdd�
IdentityIdentityoutput/BiasAdd:output:0^conv_3/BiasAdd/ReadVariableOp*^conv_3/conv1d/ExpandDims_1/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp*^conv_5/conv1d/ExpandDims_1/ReadVariableOp^conv_7/BiasAdd/ReadVariableOp*^conv_7/conv1d/ExpandDims_1/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2V
)conv_3/conv1d/ExpandDims_1/ReadVariableOp)conv_3/conv1d/ExpandDims_1/ReadVariableOp2>
conv_5/BiasAdd/ReadVariableOpconv_5/BiasAdd/ReadVariableOp2V
)conv_5/conv1d/ExpandDims_1/ReadVariableOp)conv_5/conv1d/ExpandDims_1/ReadVariableOp2>
conv_7/BiasAdd/ReadVariableOpconv_7/BiasAdd/ReadVariableOp2V
)conv_7/conv1d/ExpandDims_1/ReadVariableOp)conv_7/conv1d/ExpandDims_1/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
D__inference_drop_d2_layer_call_and_return_conditional_losses_2021154

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������<2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������<2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
c
D__inference_drop_d2_layer_call_and_return_conditional_losses_2021166

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
:���������<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������<*
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
:���������<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
b
D__inference_drop_d1_layer_call_and_return_conditional_losses_2021107

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
�
a
C__inference_drop_3_layer_call_and_return_conditional_losses_2019848

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������d2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_2019859

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����L  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
D
(__inference_pool_7_layer_call_fn_2019756

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
GPU2 *0J 8� *L
fGRE
C__inference_pool_7_layer_call_and_return_conditional_losses_20197502
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
�
b
)__inference_drop_d0_layer_call_fn_2021082

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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d0_layer_call_and_return_conditional_losses_20201032
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
�c
�
 __inference__traced_save_2021365
file_prefix,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop,
(savev2_conv_5_kernel_read_readvariableop*
&savev2_conv_5_bias_read_readvariableop,
(savev2_conv_7_kernel_read_readvariableop*
&savev2_conv_7_bias_read_readvariableop-
)savev2_dense_0_kernel_read_readvariableop+
'savev2_dense_0_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv_3_kernel_m_read_readvariableop1
-savev2_adam_conv_3_bias_m_read_readvariableop3
/savev2_adam_conv_5_kernel_m_read_readvariableop1
-savev2_adam_conv_5_bias_m_read_readvariableop3
/savev2_adam_conv_7_kernel_m_read_readvariableop1
-savev2_adam_conv_7_bias_m_read_readvariableop4
0savev2_adam_dense_0_kernel_m_read_readvariableop2
.savev2_adam_dense_0_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop3
/savev2_adam_conv_3_kernel_v_read_readvariableop1
-savev2_adam_conv_3_bias_v_read_readvariableop3
/savev2_adam_conv_5_kernel_v_read_readvariableop1
-savev2_adam_conv_5_bias_v_read_readvariableop3
/savev2_adam_conv_7_kernel_v_read_readvariableop1
-savev2_adam_conv_7_bias_v_read_readvariableop4
0savev2_adam_dense_0_kernel_v_read_readvariableop2
.savev2_adam_dense_0_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop3
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop(savev2_conv_7_kernel_read_readvariableop&savev2_conv_7_bias_read_readvariableop)savev2_dense_0_kernel_read_readvariableop'savev2_dense_0_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv_3_kernel_m_read_readvariableop-savev2_adam_conv_3_bias_m_read_readvariableop/savev2_adam_conv_5_kernel_m_read_readvariableop-savev2_adam_conv_5_bias_m_read_readvariableop/savev2_adam_conv_7_kernel_m_read_readvariableop-savev2_adam_conv_7_bias_m_read_readvariableop0savev2_adam_dense_0_kernel_m_read_readvariableop.savev2_adam_dense_0_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop/savev2_adam_conv_3_kernel_v_read_readvariableop-savev2_adam_conv_3_bias_v_read_readvariableop/savev2_adam_conv_5_kernel_v_read_readvariableop-savev2_adam_conv_5_bias_v_read_readvariableop/savev2_adam_conv_7_kernel_v_read_readvariableop-savev2_adam_conv_7_bias_v_read_readvariableop0savev2_adam_dense_0_kernel_v_read_readvariableop.savev2_adam_dense_0_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :d:d:F:F:(:(:	�P:P:PP:P:P<:<:<:: : : : : : : :d:d:F:F:(:(:	�P:P:PP:P:P<:<:<::d:d:F:F:(:(:	�P:P:PP:P:P<:<:<:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:d: 

_output_shapes
:d:($
"
_output_shapes
:F: 

_output_shapes
:F:($
"
_output_shapes
:(: 

_output_shapes
:(:%!

_output_shapes
:	�P: 

_output_shapes
:P:$	 

_output_shapes

:PP: 


_output_shapes
:P:$ 

_output_shapes

:P<: 

_output_shapes
:<:$ 

_output_shapes

:<: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:d: 

_output_shapes
:d:($
"
_output_shapes
:F: 

_output_shapes
:F:($
"
_output_shapes
:(: 

_output_shapes
:(:%!

_output_shapes
:	�P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$  

_output_shapes

:P<: !

_output_shapes
:<:$" 

_output_shapes

:<: #

_output_shapes
::($$
"
_output_shapes
:d: %

_output_shapes
:d:(&$
"
_output_shapes
:F: '

_output_shapes
:F:(($
"
_output_shapes
:(: )

_output_shapes
:(:%*!

_output_shapes
:	�P: +

_output_shapes
:P:$, 

_output_shapes

:PP: -

_output_shapes
:P:$. 

_output_shapes

:P<: /

_output_shapes
:<:$0 

_output_shapes

:<: 1

_output_shapes
::2

_output_shapes
: 
�
�
)__inference_dense_0_layer_call_fn_2021055

inputs
unknown:	�P
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_0_layer_call_and_return_conditional_losses_20198982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�V
�
E__inference_model_26_layer_call_and_return_conditional_losses_2020496
input_onehot$
conv_7_2020447:(
conv_7_2020449:($
conv_5_2020452:F
conv_5_2020454:F$
conv_3_2020457:d
conv_3_2020459:d"
dense_0_2020472:	�P
dense_0_2020474:P!
dense_1_2020478:PP
dense_1_2020480:P!
dense_2_2020484:P<
dense_2_2020486:< 
output_2020490:<
output_2020492:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�drop_3/StatefulPartitionedCall�drop_5/StatefulPartitionedCall�drop_7/StatefulPartitionedCall�drop_d0/StatefulPartitionedCall�drop_d1/StatefulPartitionedCall�drop_d2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_7_2020447conv_7_2020449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_7_layer_call_and_return_conditional_losses_20197792 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_5_2020452conv_5_2020454*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_5_layer_call_and_return_conditional_losses_20198012 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinput_onehotconv_3_2020457conv_3_2020459*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_20198232 
conv_3/StatefulPartitionedCall�
drop_7/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_7_layer_call_and_return_conditional_losses_20202082 
drop_7/StatefulPartitionedCall�
drop_5/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0^drop_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_5_layer_call_and_return_conditional_losses_20201852 
drop_5/StatefulPartitionedCall�
drop_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0^drop_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_3_layer_call_and_return_conditional_losses_20201622 
drop_3/StatefulPartitionedCall�
pool_7/PartitionedCallPartitionedCall'drop_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_7_layer_call_and_return_conditional_losses_20197502
pool_7/PartitionedCall�
pool_5/PartitionedCallPartitionedCall'drop_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_5_layer_call_and_return_conditional_losses_20197352
pool_5/PartitionedCall�
pool_3/PartitionedCallPartitionedCall'drop_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_3_layer_call_and_return_conditional_losses_20197202
pool_3/PartitionedCall�
flatten_3/PartitionedCallPartitionedCallpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_20198592
flatten_3/PartitionedCall�
flatten_5/PartitionedCallPartitionedCallpool_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20198672
flatten_5/PartitionedCall�
flatten_7/PartitionedCallPartitionedCallpool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_20198752
flatten_7/PartitionedCall�
concatenate_14/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_20198852 
concatenate_14/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense_0_2020472dense_0_2020474*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_0_layer_call_and_return_conditional_losses_20198982!
dense_0/StatefulPartitionedCall�
drop_d0/StatefulPartitionedCallStatefulPartitionedCall(dense_0/StatefulPartitionedCall:output:0^drop_3/StatefulPartitionedCall*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d0_layer_call_and_return_conditional_losses_20201032!
drop_d0/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(drop_d0/StatefulPartitionedCall:output:0dense_1_2020478dense_1_2020480*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20199222!
dense_1/StatefulPartitionedCall�
drop_d1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^drop_d0/StatefulPartitionedCall*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d1_layer_call_and_return_conditional_losses_20200702!
drop_d1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(drop_d1/StatefulPartitionedCall:output:0dense_2_2020484dense_2_2020486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20199462!
dense_2/StatefulPartitionedCall�
drop_d2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 ^drop_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_drop_d2_layer_call_and_return_conditional_losses_20200372!
drop_d2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(drop_d2/StatefulPartitionedCall:output:0output_2020490output_2020492*
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
C__inference_output_layer_call_and_return_conditional_losses_20199692 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_3/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_7/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^drop_3/StatefulPartitionedCall^drop_5/StatefulPartitionedCall^drop_7/StatefulPartitionedCall ^drop_d0/StatefulPartitionedCall ^drop_d1/StatefulPartitionedCall ^drop_d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
drop_3/StatefulPartitionedCalldrop_3/StatefulPartitionedCall2@
drop_5/StatefulPartitionedCalldrop_5/StatefulPartitionedCall2@
drop_7/StatefulPartitionedCalldrop_7/StatefulPartitionedCall2B
drop_d0/StatefulPartitionedCalldrop_d0/StatefulPartitionedCall2B
drop_d1/StatefulPartitionedCalldrop_d1/StatefulPartitionedCall2B
drop_d2/StatefulPartitionedCalldrop_d2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�
G
+__inference_flatten_7_layer_call_fn_2021020

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
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_20198752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	(:S O
+
_output_shapes
:���������	(
 
_user_specified_nameinputs
�
E
)__inference_drop_d0_layer_call_fn_2021077

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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d0_layer_call_and_return_conditional_losses_20199092
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
�
�
C__inference_conv_5_layer_call_and_return_conditional_losses_2019801

inputsA
+conv1d_expanddims_1_readvariableop_resource:F-
biasadd_readvariableop_resource:F
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:F*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:F2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������F*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������F*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������F2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������F2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
D__inference_drop_d1_layer_call_and_return_conditional_losses_2019933

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
�
D
(__inference_drop_7_layer_call_fn_2020982

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_7_layer_call_and_return_conditional_losses_20198342
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
C__inference_conv_7_layer_call_and_return_conditional_losses_2019779

inputsA
+conv1d_expanddims_1_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:(*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:(2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������(*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������(*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������(2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
C__inference_drop_5_layer_call_and_return_conditional_losses_2020950

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������F2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������F*
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
T0*+
_output_shapes
:���������F2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������F2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������F2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������F:S O
+
_output_shapes
:���������F
 
_user_specified_nameinputs
�
c
D__inference_drop_d2_layer_call_and_return_conditional_losses_2020037

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
:���������<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������<*
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
:���������<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������<2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������<2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������<:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
*__inference_model_26_layer_call_fn_2020831

inputs
unknown:(
	unknown_0:(
	unknown_1:F
	unknown_2:F
	unknown_3:d
	unknown_4:d
	unknown_5:	�P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P<

unknown_10:<

unknown_11:<

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_model_26_layer_call_and_return_conditional_losses_20203282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
D__inference_drop_d1_layer_call_and_return_conditional_losses_2021119

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
a
C__inference_drop_3_layer_call_and_return_conditional_losses_2020911

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������d2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
D__inference_dense_1_layer_call_and_return_conditional_losses_2021093

inputs0
matmul_readvariableop_resource:PP-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
D
(__inference_drop_5_layer_call_fn_2020955

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_5_layer_call_and_return_conditional_losses_20198412
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������F2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������F:S O
+
_output_shapes
:���������F
 
_user_specified_nameinputs
�
b
D__inference_drop_d0_layer_call_and_return_conditional_losses_2019909

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
�
*__inference_model_26_layer_call_fn_2020798

inputs
unknown:(
	unknown_0:(
	unknown_1:F
	unknown_2:F
	unknown_3:d
	unknown_4:d
	unknown_5:	�P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P<

unknown_10:<

unknown_11:<

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_model_26_layer_call_and_return_conditional_losses_20199762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
(__inference_drop_3_layer_call_fn_2020933

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_3_layer_call_and_return_conditional_losses_20201622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
_
C__inference_pool_7_layer_call_and_return_conditional_losses_2019750

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
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingSAME*
strides
2	
AvgPool�
SqueezeSqueezeAvgPool:output:0*
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
�
b
)__inference_drop_d2_layer_call_fn_2021176

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
:���������<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_drop_d2_layer_call_and_return_conditional_losses_20200372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������<22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_2019875

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	(:S O
+
_output_shapes
:���������	(
 
_user_specified_nameinputs
�
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_2020993

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����L  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�L
�
E__inference_model_26_layer_call_and_return_conditional_losses_2019976

inputs$
conv_7_2019780:(
conv_7_2019782:($
conv_5_2019802:F
conv_5_2019804:F$
conv_3_2019824:d
conv_3_2019826:d"
dense_0_2019899:	�P
dense_0_2019901:P!
dense_1_2019923:PP
dense_1_2019925:P!
dense_2_2019947:P<
dense_2_2019949:< 
output_2019970:<
output_2019972:
identity��conv_3/StatefulPartitionedCall�conv_5/StatefulPartitionedCall�conv_7/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�output/StatefulPartitionedCall�
conv_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv_7_2019780conv_7_2019782*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_7_layer_call_and_return_conditional_losses_20197792 
conv_7/StatefulPartitionedCall�
conv_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_5_2019802conv_5_2019804*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_5_layer_call_and_return_conditional_losses_20198012 
conv_5/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv_3_2019824conv_3_2019826*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_20198232 
conv_3/StatefulPartitionedCall�
drop_7/PartitionedCallPartitionedCall'conv_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_7_layer_call_and_return_conditional_losses_20198342
drop_7/PartitionedCall�
drop_5/PartitionedCallPartitionedCall'conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_5_layer_call_and_return_conditional_losses_20198412
drop_5/PartitionedCall�
drop_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_drop_3_layer_call_and_return_conditional_losses_20198482
drop_3/PartitionedCall�
pool_7/PartitionedCallPartitionedCalldrop_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_7_layer_call_and_return_conditional_losses_20197502
pool_7/PartitionedCall�
pool_5/PartitionedCallPartitionedCalldrop_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
F* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_5_layer_call_and_return_conditional_losses_20197352
pool_5/PartitionedCall�
pool_3/PartitionedCallPartitionedCalldrop_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_pool_3_layer_call_and_return_conditional_losses_20197202
pool_3/PartitionedCall�
flatten_3/PartitionedCallPartitionedCallpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_20198592
flatten_3/PartitionedCall�
flatten_5/PartitionedCallPartitionedCallpool_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20198672
flatten_5/PartitionedCall�
flatten_7/PartitionedCallPartitionedCallpool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_20198752
flatten_7/PartitionedCall�
concatenate_14/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_concatenate_14_layer_call_and_return_conditional_losses_20198852 
concatenate_14/PartitionedCall�
dense_0/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0dense_0_2019899dense_0_2019901*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_0_layer_call_and_return_conditional_losses_20198982!
dense_0/StatefulPartitionedCall�
drop_d0/PartitionedCallPartitionedCall(dense_0/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d0_layer_call_and_return_conditional_losses_20199092
drop_d0/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall drop_d0/PartitionedCall:output:0dense_1_2019923dense_1_2019925*
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
GPU2 *0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20199222!
dense_1/StatefulPartitionedCall�
drop_d1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d1_layer_call_and_return_conditional_losses_20199332
drop_d1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall drop_d1/PartitionedCall:output:0dense_2_2019947dense_2_2019949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20199462!
dense_2/StatefulPartitionedCall�
drop_d2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_drop_d2_layer_call_and_return_conditional_losses_20199572
drop_d2/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall drop_d2/PartitionedCall:output:0output_2019970output_2019972*
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
C__inference_output_layer_call_and_return_conditional_losses_20199692 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_3/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_7/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_output_layer_call_fn_2021195

inputs
unknown:<
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
C__inference_output_layer_call_and_return_conditional_losses_20199692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
a
C__inference_drop_7_layer_call_and_return_conditional_losses_2019834

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������(2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
D
(__inference_pool_3_layer_call_fn_2019726

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
GPU2 *0J 8� *L
fGRE
C__inference_pool_3_layer_call_and_return_conditional_losses_20197202
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
�
a
C__inference_drop_7_layer_call_and_return_conditional_losses_2020965

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������(2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
K__inference_concatenate_14_layer_call_and_return_conditional_losses_2021028
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/2
�
�
*__inference_model_26_layer_call_fn_2020392
input_onehot
unknown:(
	unknown_0:(
	unknown_1:F
	unknown_2:F
	unknown_3:d
	unknown_4:d
	unknown_5:	�P
	unknown_6:P
	unknown_7:PP
	unknown_8:P
	unknown_9:P<

unknown_10:<

unknown_11:<

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_onehotunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_model_26_layer_call_and_return_conditional_losses_20203282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:���������
&
_user_specified_nameinput_onehot
�

�
D__inference_dense_2_layer_call_and_return_conditional_losses_2021140

inputs0
matmul_readvariableop_resource:P<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������<2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������<2

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
�
E
)__inference_drop_d1_layer_call_fn_2021124

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
GPU2 *0J 8� *M
fHRF
D__inference_drop_d1_layer_call_and_return_conditional_losses_20199332
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
�v
�
E__inference_model_26_layer_call_and_return_conditional_losses_2020630

inputsH
2conv_7_conv1d_expanddims_1_readvariableop_resource:(4
&conv_7_biasadd_readvariableop_resource:(H
2conv_5_conv1d_expanddims_1_readvariableop_resource:F4
&conv_5_biasadd_readvariableop_resource:FH
2conv_3_conv1d_expanddims_1_readvariableop_resource:d4
&conv_3_biasadd_readvariableop_resource:d9
&dense_0_matmul_readvariableop_resource:	�P5
'dense_0_biasadd_readvariableop_resource:P8
&dense_1_matmul_readvariableop_resource:PP5
'dense_1_biasadd_readvariableop_resource:P8
&dense_2_matmul_readvariableop_resource:P<5
'dense_2_biasadd_readvariableop_resource:<7
%output_matmul_readvariableop_resource:<4
&output_biasadd_readvariableop_resource:
identity��conv_3/BiasAdd/ReadVariableOp�)conv_3/conv1d/ExpandDims_1/ReadVariableOp�conv_5/BiasAdd/ReadVariableOp�)conv_5/conv1d/ExpandDims_1/ReadVariableOp�conv_7/BiasAdd/ReadVariableOp�)conv_7/conv1d/ExpandDims_1/ReadVariableOp�dense_0/BiasAdd/ReadVariableOp�dense_0/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
conv_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_7/conv1d/ExpandDims/dim�
conv_7/conv1d/ExpandDims
ExpandDimsinputs%conv_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv_7/conv1d/ExpandDims�
)conv_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:(*
dtype02+
)conv_7/conv1d/ExpandDims_1/ReadVariableOp�
conv_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_7/conv1d/ExpandDims_1/dim�
conv_7/conv1d/ExpandDims_1
ExpandDims1conv_7/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:(2
conv_7/conv1d/ExpandDims_1�
conv_7/conv1dConv2D!conv_7/conv1d/ExpandDims:output:0#conv_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������(*
paddingVALID*
strides
2
conv_7/conv1d�
conv_7/conv1d/SqueezeSqueezeconv_7/conv1d:output:0*
T0*+
_output_shapes
:���������(*
squeeze_dims

���������2
conv_7/conv1d/Squeeze�
conv_7/BiasAdd/ReadVariableOpReadVariableOp&conv_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
conv_7/BiasAdd/ReadVariableOp�
conv_7/BiasAddBiasAddconv_7/conv1d/Squeeze:output:0%conv_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
conv_7/BiasAddq
conv_7/ReluReluconv_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
conv_7/Relu�
conv_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_5/conv1d/ExpandDims/dim�
conv_5/conv1d/ExpandDims
ExpandDimsinputs%conv_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv_5/conv1d/ExpandDims�
)conv_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:F*
dtype02+
)conv_5/conv1d/ExpandDims_1/ReadVariableOp�
conv_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_5/conv1d/ExpandDims_1/dim�
conv_5/conv1d/ExpandDims_1
ExpandDims1conv_5/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:F2
conv_5/conv1d/ExpandDims_1�
conv_5/conv1dConv2D!conv_5/conv1d/ExpandDims:output:0#conv_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������F*
paddingVALID*
strides
2
conv_5/conv1d�
conv_5/conv1d/SqueezeSqueezeconv_5/conv1d:output:0*
T0*+
_output_shapes
:���������F*
squeeze_dims

���������2
conv_5/conv1d/Squeeze�
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
conv_5/BiasAdd/ReadVariableOp�
conv_5/BiasAddBiasAddconv_5/conv1d/Squeeze:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������F2
conv_5/BiasAddq
conv_5/ReluReluconv_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������F2
conv_5/Relu�
conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_3/conv1d/ExpandDims/dim�
conv_3/conv1d/ExpandDims
ExpandDimsinputs%conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv_3/conv1d/ExpandDims�
)conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:d*
dtype02+
)conv_3/conv1d/ExpandDims_1/ReadVariableOp�
conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_3/conv1d/ExpandDims_1/dim�
conv_3/conv1d/ExpandDims_1
ExpandDims1conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:d2
conv_3/conv1d/ExpandDims_1�
conv_3/conv1dConv2D!conv_3/conv1d/ExpandDims:output:0#conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
conv_3/conv1d�
conv_3/conv1d/SqueezeSqueezeconv_3/conv1d:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims

���������2
conv_3/conv1d/Squeeze�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/conv1d/Squeeze:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d2
conv_3/BiasAddq
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������d2
conv_3/Relu
drop_7/IdentityIdentityconv_7/Relu:activations:0*
T0*+
_output_shapes
:���������(2
drop_7/Identity
drop_5/IdentityIdentityconv_5/Relu:activations:0*
T0*+
_output_shapes
:���������F2
drop_5/Identity
drop_3/IdentityIdentityconv_3/Relu:activations:0*
T0*+
_output_shapes
:���������d2
drop_3/Identityp
pool_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_7/ExpandDims/dim�
pool_7/ExpandDims
ExpandDimsdrop_7/Identity:output:0pool_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������(2
pool_7/ExpandDims�
pool_7/AvgPoolAvgPoolpool_7/ExpandDims:output:0*
T0*/
_output_shapes
:���������	(*
ksize
*
paddingSAME*
strides
2
pool_7/AvgPool�
pool_7/SqueezeSqueezepool_7/AvgPool:output:0*
T0*+
_output_shapes
:���������	(*
squeeze_dims
2
pool_7/Squeezep
pool_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_5/ExpandDims/dim�
pool_5/ExpandDims
ExpandDimsdrop_5/Identity:output:0pool_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������F2
pool_5/ExpandDims�
pool_5/AvgPoolAvgPoolpool_5/ExpandDims:output:0*
T0*/
_output_shapes
:���������
F*
ksize
*
paddingSAME*
strides
2
pool_5/AvgPool�
pool_5/SqueezeSqueezepool_5/AvgPool:output:0*
T0*+
_output_shapes
:���������
F*
squeeze_dims
2
pool_5/Squeezep
pool_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_3/ExpandDims/dim�
pool_3/ExpandDims
ExpandDimsdrop_3/Identity:output:0pool_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d2
pool_3/ExpandDims�
pool_3/AvgPoolAvgPoolpool_3/ExpandDims:output:0*
T0*/
_output_shapes
:���������d*
ksize
*
paddingSAME*
strides
2
pool_3/AvgPool�
pool_3/SqueezeSqueezepool_3/AvgPool:output:0*
T0*+
_output_shapes
:���������d*
squeeze_dims
2
pool_3/Squeezes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����L  2
flatten_3/Const�
flatten_3/ReshapeReshapepool_3/Squeeze:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_5/Const�
flatten_5/ReshapeReshapepool_5/Squeeze:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_5/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����h  2
flatten_7/Const�
flatten_7/ReshapeReshapepool_7/Squeeze:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_7/Reshapez
concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_14/concat/axis�
concatenate_14/concatConcatV2flatten_3/Reshape:output:0flatten_5/Reshape:output:0flatten_7/Reshape:output:0#concatenate_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate_14/concat�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype02
dense_0/MatMul/ReadVariableOp�
dense_0/MatMulMatMulconcatenate_14/concat:output:0%dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_0/MatMul�
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_0/BiasAdd/ReadVariableOp�
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_0/BiasAddp
dense_0/ReluReludense_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
dense_0/Relu~
drop_d0/IdentityIdentitydense_0/Relu:activations:0*
T0*'
_output_shapes
:���������P2
drop_d0/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldrop_d0/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
dense_1/Relu~
drop_d1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:���������P2
drop_d1/Identity�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldrop_d1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
dense_2/Relu~
drop_d2/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:���������<2
drop_d2/Identity�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldrop_d2/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
output/BiasAdd�
IdentityIdentityoutput/BiasAdd:output:0^conv_3/BiasAdd/ReadVariableOp*^conv_3/conv1d/ExpandDims_1/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp*^conv_5/conv1d/ExpandDims_1/ReadVariableOp^conv_7/BiasAdd/ReadVariableOp*^conv_7/conv1d/ExpandDims_1/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : 2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2V
)conv_3/conv1d/ExpandDims_1/ReadVariableOp)conv_3/conv1d/ExpandDims_1/ReadVariableOp2>
conv_5/BiasAdd/ReadVariableOpconv_5/BiasAdd/ReadVariableOp2V
)conv_5/conv1d/ExpandDims_1/ReadVariableOp)conv_5/conv1d/ExpandDims_1/ReadVariableOp2>
conv_7/BiasAdd/ReadVariableOpconv_7/BiasAdd/ReadVariableOp2V
)conv_7/conv1d/ExpandDims_1/ReadVariableOp)conv_7/conv1d/ExpandDims_1/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv_7_layer_call_and_return_conditional_losses_2020897

inputsA
+conv1d_expanddims_1_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:(*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:(2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������(*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������(*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������(2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
D__inference_drop_d0_layer_call_and_return_conditional_losses_2021072

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
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
input_onehot9
serving_default_input_onehot:0���������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
��
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-3
layer-14
layer-15
layer_with_weights-4
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer_with_weights-6
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"ڈ
_tf_keras_network��{"name": "model_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_26", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 70, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["input_onehot", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_5", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_7", "inbound_nodes": [[["conv_7", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "pool_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_3", "inbound_nodes": [[["drop_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "pool_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_5", "inbound_nodes": [[["drop_5", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "pool_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_7", "inbound_nodes": [[["drop_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["pool_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["pool_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["pool_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_14", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_14", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_0", "inbound_nodes": [[["concatenate_14", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_d0", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d0", "inbound_nodes": [[["dense_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["drop_d0", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_d1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["drop_d1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_d2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d2", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["drop_d2", 0, 0, {}]]]}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 35, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 23, 4]}, "float32", "input_onehot"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_26", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}, "name": "input_onehot", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv1D", "config": {"name": "conv_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 70, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv1D", "config": {"name": "conv_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "drop_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_5", "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "drop_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_7", "inbound_nodes": [[["conv_7", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "AveragePooling1D", "config": {"name": "pool_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_3", "inbound_nodes": [[["drop_3", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "AveragePooling1D", "config": {"name": "pool_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_5", "inbound_nodes": [[["drop_5", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "AveragePooling1D", "config": {"name": "pool_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "name": "pool_7", "inbound_nodes": [[["drop_7", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["pool_3", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["pool_5", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["pool_7", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Concatenate", "config": {"name": "concatenate_14", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_14", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_7", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_0", "inbound_nodes": [[["concatenate_14", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dropout", "config": {"name": "drop_d0", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d0", "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["drop_d0", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Dropout", "config": {"name": "drop_d1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["drop_d1", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Dropout", "config": {"name": "drop_d2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "drop_d2", "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["drop_d2", 0, 0, {}]]], "shared_object_id": 34}], "input_layers": [["input_onehot", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_onehot", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_onehot"}}
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�

_tf_keras_layer�
{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}}
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�

_tf_keras_layer�
{"name": "conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 70, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}}
�

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�

_tf_keras_layer�
{"name": "conv_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 23, 4]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_onehot", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 4]}}
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "drop_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 10}
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "drop_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 11}
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "drop_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_7", 0, 0, {}]]], "shared_object_id": 12}
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "pool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "pool_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["drop_3", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 40}}
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "pool_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "pool_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["drop_5", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "pool_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling1D", "config": {"name": "pool_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "inbound_nodes": [[["drop_7", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}}
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool_3", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 43}}
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool_5", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 44}}
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool_7", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 45}}
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "concatenate_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_14", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["flatten_3", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_7", 0, 0, {}]]], "shared_object_id": 19, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1100]}, {"class_name": "TensorShape", "items": [null, 700]}, {"class_name": "TensorShape", "items": [null, 360]}]}
�	

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_14", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2160}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2160]}}
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "drop_d0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_d0", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_0", 0, 0, {}]]], "shared_object_id": 23}
�

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_d0", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "drop_d1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_d1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 27}
�

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_d1", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "drop_d2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_d2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 31}
�

tkernel
ubias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_d2", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
�
ziter

{beta_1

|beta_2
	}decay
~learning_ratem�m�"m�#m�(m�)m�Vm�Wm�`m�am�jm�km�tm�um�v�v�"v�#v�(v�)v�Vv�Wv�`v�av�jv�kv�tv�uv�"
	optimizer
�
0
1
"2
#3
(4
)5
V6
W7
`8
a9
j10
k11
t12
u13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
"2
#3
(4
)5
V6
W7
`8
a9
j10
k11
t12
u13"
trackable_list_wrapper
�
layer_metrics
	variables
�metrics
 �layer_regularization_losses
regularization_losses
�non_trainable_variables
�layers
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
#:!d2conv_3/kernel
:d2conv_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
�metrics
trainable_variables
 �layer_regularization_losses
 regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!F2conv_5/kernel
:F2conv_5/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
$	variables
�metrics
%trainable_variables
 �layer_regularization_losses
&regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!(2conv_7/kernel
:(2conv_7/bias
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
*	variables
�metrics
+trainable_variables
 �layer_regularization_losses
,regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
.	variables
�metrics
/trainable_variables
 �layer_regularization_losses
0regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
2	variables
�metrics
3trainable_variables
 �layer_regularization_losses
4regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
6	variables
�metrics
7trainable_variables
 �layer_regularization_losses
8regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
:	variables
�metrics
;trainable_variables
 �layer_regularization_losses
<regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
>	variables
�metrics
?trainable_variables
 �layer_regularization_losses
@regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
B	variables
�metrics
Ctrainable_variables
 �layer_regularization_losses
Dregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
F	variables
�metrics
Gtrainable_variables
 �layer_regularization_losses
Hregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
J	variables
�metrics
Ktrainable_variables
 �layer_regularization_losses
Lregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
N	variables
�metrics
Otrainable_variables
 �layer_regularization_losses
Pregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
R	variables
�metrics
Strainable_variables
 �layer_regularization_losses
Tregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�P2dense_0/kernel
:P2dense_0/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
X	variables
�metrics
Ytrainable_variables
 �layer_regularization_losses
Zregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\	variables
�metrics
]trainable_variables
 �layer_regularization_losses
^regularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :PP2dense_1/kernel
:P2dense_1/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
b	variables
�metrics
ctrainable_variables
 �layer_regularization_losses
dregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
f	variables
�metrics
gtrainable_variables
 �layer_regularization_losses
hregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :P<2dense_2/kernel
:<2dense_2/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
l	variables
�metrics
mtrainable_variables
 �layer_regularization_losses
nregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
p	variables
�metrics
qtrainable_variables
 �layer_regularization_losses
rregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:<2output/kernel
:2output/bias
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
v	variables
�metrics
wtrainable_variables
 �layer_regularization_losses
xregularization_losses
�non_trainable_variables
�layers
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
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

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 50}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
(:&d2Adam/conv_3/kernel/m
:d2Adam/conv_3/bias/m
(:&F2Adam/conv_5/kernel/m
:F2Adam/conv_5/bias/m
(:&(2Adam/conv_7/kernel/m
:(2Adam/conv_7/bias/m
&:$	�P2Adam/dense_0/kernel/m
:P2Adam/dense_0/bias/m
%:#PP2Adam/dense_1/kernel/m
:P2Adam/dense_1/bias/m
%:#P<2Adam/dense_2/kernel/m
:<2Adam/dense_2/bias/m
$:"<2Adam/output/kernel/m
:2Adam/output/bias/m
(:&d2Adam/conv_3/kernel/v
:d2Adam/conv_3/bias/v
(:&F2Adam/conv_5/kernel/v
:F2Adam/conv_5/bias/v
(:&(2Adam/conv_7/kernel/v
:(2Adam/conv_7/bias/v
&:$	�P2Adam/dense_0/kernel/v
:P2Adam/dense_0/bias/v
%:#PP2Adam/dense_1/kernel/v
:P2Adam/dense_1/bias/v
%:#P<2Adam/dense_2/kernel/v
:<2Adam/dense_2/bias/v
$:"<2Adam/output/kernel/v
:2Adam/output/bias/v
�2�
E__inference_model_26_layer_call_and_return_conditional_losses_2020630
E__inference_model_26_layer_call_and_return_conditional_losses_2020765
E__inference_model_26_layer_call_and_return_conditional_losses_2020444
E__inference_model_26_layer_call_and_return_conditional_losses_2020496�
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
*__inference_model_26_layer_call_fn_2020007
*__inference_model_26_layer_call_fn_2020798
*__inference_model_26_layer_call_fn_2020831
*__inference_model_26_layer_call_fn_2020392�
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
"__inference__wrapped_model_2019711�
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
annotations� */�,
*�'
input_onehot���������
�2�
C__inference_conv_3_layer_call_and_return_conditional_losses_2020847�
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
(__inference_conv_3_layer_call_fn_2020856�
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
C__inference_conv_5_layer_call_and_return_conditional_losses_2020872�
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
(__inference_conv_5_layer_call_fn_2020881�
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
C__inference_conv_7_layer_call_and_return_conditional_losses_2020897�
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
(__inference_conv_7_layer_call_fn_2020906�
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
C__inference_drop_3_layer_call_and_return_conditional_losses_2020911
C__inference_drop_3_layer_call_and_return_conditional_losses_2020923�
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
(__inference_drop_3_layer_call_fn_2020928
(__inference_drop_3_layer_call_fn_2020933�
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
C__inference_drop_5_layer_call_and_return_conditional_losses_2020938
C__inference_drop_5_layer_call_and_return_conditional_losses_2020950�
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
(__inference_drop_5_layer_call_fn_2020955
(__inference_drop_5_layer_call_fn_2020960�
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
C__inference_drop_7_layer_call_and_return_conditional_losses_2020965
C__inference_drop_7_layer_call_and_return_conditional_losses_2020977�
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
(__inference_drop_7_layer_call_fn_2020982
(__inference_drop_7_layer_call_fn_2020987�
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
C__inference_pool_3_layer_call_and_return_conditional_losses_2019720�
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
(__inference_pool_3_layer_call_fn_2019726�
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
C__inference_pool_5_layer_call_and_return_conditional_losses_2019735�
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
(__inference_pool_5_layer_call_fn_2019741�
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
C__inference_pool_7_layer_call_and_return_conditional_losses_2019750�
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
(__inference_pool_7_layer_call_fn_2019756�
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_2020993�
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
+__inference_flatten_3_layer_call_fn_2020998�
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
F__inference_flatten_5_layer_call_and_return_conditional_losses_2021004�
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
+__inference_flatten_5_layer_call_fn_2021009�
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
F__inference_flatten_7_layer_call_and_return_conditional_losses_2021015�
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
+__inference_flatten_7_layer_call_fn_2021020�
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
K__inference_concatenate_14_layer_call_and_return_conditional_losses_2021028�
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
0__inference_concatenate_14_layer_call_fn_2021035�
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
D__inference_dense_0_layer_call_and_return_conditional_losses_2021046�
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
)__inference_dense_0_layer_call_fn_2021055�
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
D__inference_drop_d0_layer_call_and_return_conditional_losses_2021060
D__inference_drop_d0_layer_call_and_return_conditional_losses_2021072�
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
)__inference_drop_d0_layer_call_fn_2021077
)__inference_drop_d0_layer_call_fn_2021082�
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
D__inference_dense_1_layer_call_and_return_conditional_losses_2021093�
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
)__inference_dense_1_layer_call_fn_2021102�
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
D__inference_drop_d1_layer_call_and_return_conditional_losses_2021107
D__inference_drop_d1_layer_call_and_return_conditional_losses_2021119�
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
)__inference_drop_d1_layer_call_fn_2021124
)__inference_drop_d1_layer_call_fn_2021129�
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
D__inference_dense_2_layer_call_and_return_conditional_losses_2021140�
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
)__inference_dense_2_layer_call_fn_2021149�
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
D__inference_drop_d2_layer_call_and_return_conditional_losses_2021154
D__inference_drop_d2_layer_call_and_return_conditional_losses_2021166�
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
)__inference_drop_d2_layer_call_fn_2021171
)__inference_drop_d2_layer_call_fn_2021176�
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
C__inference_output_layer_call_and_return_conditional_losses_2021186�
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
(__inference_output_layer_call_fn_2021195�
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
%__inference_signature_wrapper_2020537input_onehot"�
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
"__inference__wrapped_model_2019711|()"#VW`ajktu9�6
/�,
*�'
input_onehot���������
� "/�,
*
output �
output����������
K__inference_concatenate_14_layer_call_and_return_conditional_losses_2021028���~
w�t
r�o
#� 
inputs/0����������
#� 
inputs/1����������
#� 
inputs/2����������
� "&�#
�
0����������
� �
0__inference_concatenate_14_layer_call_fn_2021035���~
w�t
r�o
#� 
inputs/0����������
#� 
inputs/1����������
#� 
inputs/2����������
� "������������
C__inference_conv_3_layer_call_and_return_conditional_losses_2020847d3�0
)�&
$�!
inputs���������
� ")�&
�
0���������d
� �
(__inference_conv_3_layer_call_fn_2020856W3�0
)�&
$�!
inputs���������
� "����������d�
C__inference_conv_5_layer_call_and_return_conditional_losses_2020872d"#3�0
)�&
$�!
inputs���������
� ")�&
�
0���������F
� �
(__inference_conv_5_layer_call_fn_2020881W"#3�0
)�&
$�!
inputs���������
� "����������F�
C__inference_conv_7_layer_call_and_return_conditional_losses_2020897d()3�0
)�&
$�!
inputs���������
� ")�&
�
0���������(
� �
(__inference_conv_7_layer_call_fn_2020906W()3�0
)�&
$�!
inputs���������
� "����������(�
D__inference_dense_0_layer_call_and_return_conditional_losses_2021046]VW0�-
&�#
!�
inputs����������
� "%�"
�
0���������P
� }
)__inference_dense_0_layer_call_fn_2021055PVW0�-
&�#
!�
inputs����������
� "����������P�
D__inference_dense_1_layer_call_and_return_conditional_losses_2021093\`a/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� |
)__inference_dense_1_layer_call_fn_2021102O`a/�,
%�"
 �
inputs���������P
� "����������P�
D__inference_dense_2_layer_call_and_return_conditional_losses_2021140\jk/�,
%�"
 �
inputs���������P
� "%�"
�
0���������<
� |
)__inference_dense_2_layer_call_fn_2021149Ojk/�,
%�"
 �
inputs���������P
� "����������<�
C__inference_drop_3_layer_call_and_return_conditional_losses_2020911d7�4
-�*
$�!
inputs���������d
p 
� ")�&
�
0���������d
� �
C__inference_drop_3_layer_call_and_return_conditional_losses_2020923d7�4
-�*
$�!
inputs���������d
p
� ")�&
�
0���������d
� �
(__inference_drop_3_layer_call_fn_2020928W7�4
-�*
$�!
inputs���������d
p 
� "����������d�
(__inference_drop_3_layer_call_fn_2020933W7�4
-�*
$�!
inputs���������d
p
� "����������d�
C__inference_drop_5_layer_call_and_return_conditional_losses_2020938d7�4
-�*
$�!
inputs���������F
p 
� ")�&
�
0���������F
� �
C__inference_drop_5_layer_call_and_return_conditional_losses_2020950d7�4
-�*
$�!
inputs���������F
p
� ")�&
�
0���������F
� �
(__inference_drop_5_layer_call_fn_2020955W7�4
-�*
$�!
inputs���������F
p 
� "����������F�
(__inference_drop_5_layer_call_fn_2020960W7�4
-�*
$�!
inputs���������F
p
� "����������F�
C__inference_drop_7_layer_call_and_return_conditional_losses_2020965d7�4
-�*
$�!
inputs���������(
p 
� ")�&
�
0���������(
� �
C__inference_drop_7_layer_call_and_return_conditional_losses_2020977d7�4
-�*
$�!
inputs���������(
p
� ")�&
�
0���������(
� �
(__inference_drop_7_layer_call_fn_2020982W7�4
-�*
$�!
inputs���������(
p 
� "����������(�
(__inference_drop_7_layer_call_fn_2020987W7�4
-�*
$�!
inputs���������(
p
� "����������(�
D__inference_drop_d0_layer_call_and_return_conditional_losses_2021060\3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
D__inference_drop_d0_layer_call_and_return_conditional_losses_2021072\3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� |
)__inference_drop_d0_layer_call_fn_2021077O3�0
)�&
 �
inputs���������P
p 
� "����������P|
)__inference_drop_d0_layer_call_fn_2021082O3�0
)�&
 �
inputs���������P
p
� "����������P�
D__inference_drop_d1_layer_call_and_return_conditional_losses_2021107\3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
D__inference_drop_d1_layer_call_and_return_conditional_losses_2021119\3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� |
)__inference_drop_d1_layer_call_fn_2021124O3�0
)�&
 �
inputs���������P
p 
� "����������P|
)__inference_drop_d1_layer_call_fn_2021129O3�0
)�&
 �
inputs���������P
p
� "����������P�
D__inference_drop_d2_layer_call_and_return_conditional_losses_2021154\3�0
)�&
 �
inputs���������<
p 
� "%�"
�
0���������<
� �
D__inference_drop_d2_layer_call_and_return_conditional_losses_2021166\3�0
)�&
 �
inputs���������<
p
� "%�"
�
0���������<
� |
)__inference_drop_d2_layer_call_fn_2021171O3�0
)�&
 �
inputs���������<
p 
� "����������<|
)__inference_drop_d2_layer_call_fn_2021176O3�0
)�&
 �
inputs���������<
p
� "����������<�
F__inference_flatten_3_layer_call_and_return_conditional_losses_2020993]3�0
)�&
$�!
inputs���������d
� "&�#
�
0����������
� 
+__inference_flatten_3_layer_call_fn_2020998P3�0
)�&
$�!
inputs���������d
� "������������
F__inference_flatten_5_layer_call_and_return_conditional_losses_2021004]3�0
)�&
$�!
inputs���������
F
� "&�#
�
0����������
� 
+__inference_flatten_5_layer_call_fn_2021009P3�0
)�&
$�!
inputs���������
F
� "������������
F__inference_flatten_7_layer_call_and_return_conditional_losses_2021015]3�0
)�&
$�!
inputs���������	(
� "&�#
�
0����������
� 
+__inference_flatten_7_layer_call_fn_2021020P3�0
)�&
$�!
inputs���������	(
� "������������
E__inference_model_26_layer_call_and_return_conditional_losses_2020444z()"#VW`ajktuA�>
7�4
*�'
input_onehot���������
p 

 
� "%�"
�
0���������
� �
E__inference_model_26_layer_call_and_return_conditional_losses_2020496z()"#VW`ajktuA�>
7�4
*�'
input_onehot���������
p

 
� "%�"
�
0���������
� �
E__inference_model_26_layer_call_and_return_conditional_losses_2020630t()"#VW`ajktu;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
E__inference_model_26_layer_call_and_return_conditional_losses_2020765t()"#VW`ajktu;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
*__inference_model_26_layer_call_fn_2020007m()"#VW`ajktuA�>
7�4
*�'
input_onehot���������
p 

 
� "�����������
*__inference_model_26_layer_call_fn_2020392m()"#VW`ajktuA�>
7�4
*�'
input_onehot���������
p

 
� "�����������
*__inference_model_26_layer_call_fn_2020798g()"#VW`ajktu;�8
1�.
$�!
inputs���������
p 

 
� "�����������
*__inference_model_26_layer_call_fn_2020831g()"#VW`ajktu;�8
1�.
$�!
inputs���������
p

 
� "�����������
C__inference_output_layer_call_and_return_conditional_losses_2021186\tu/�,
%�"
 �
inputs���������<
� "%�"
�
0���������
� {
(__inference_output_layer_call_fn_2021195Otu/�,
%�"
 �
inputs���������<
� "�����������
C__inference_pool_3_layer_call_and_return_conditional_losses_2019720�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
(__inference_pool_3_layer_call_fn_2019726wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
C__inference_pool_5_layer_call_and_return_conditional_losses_2019735�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
(__inference_pool_5_layer_call_fn_2019741wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
C__inference_pool_7_layer_call_and_return_conditional_losses_2019750�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
(__inference_pool_7_layer_call_fn_2019756wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
%__inference_signature_wrapper_2020537�()"#VW`ajktuI�F
� 
?�<
:
input_onehot*�'
input_onehot���������"/�,
*
output �
output���������